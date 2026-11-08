#include "ember/layers.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <span>
#include <vector>

namespace {

using ember::Arena;
using ember::DType;
using ember::Shape;
using ember::StatusCode;
using ember::Tensor;

Tensor Random(Arena& arena, std::initializer_list<std::size_t> dims, std::uint32_t seed) {
  const Tensor tensor = *Tensor::Uninitialized(arena, DType::kF32, *Shape::Make(dims));
  std::mt19937 random(seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  for (std::size_t i = 0; i < tensor.element_count(); ++i) tensor.data<float>()[i] = normal(random);
  return tensor;
}

Tensor Zeros(Arena& arena, std::initializer_list<std::size_t> dims) {
  return *Tensor::Zeros(arena, DType::kF32, *Shape::Make(dims));
}

// --- Linear ---

TEST(Linear, MatchesTheDefinition) {
  Arena arena;
  // 19 inputs, so the eight-wide loop and the leftover loop both run.
  const Tensor x = Random(arena, {3, 19}, 1);
  const Tensor w = Random(arena, {5, 19}, 2);
  const Tensor out = Zeros(arena, {3, 5});
  ASSERT_TRUE(ember::Linear(x, w, out).ok());
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t f = 0; f < 5; ++f) {
      double expected = 0;
      for (std::size_t k = 0; k < 19; ++k) {
        expected += static_cast<double>(x.at<float>(r, k)) * w.at<float>(f, k);
      }
      EXPECT_NEAR(out.at<float>(r, f), expected, 1e-5);
    }
  }
}

TEST(Linear, RefusesBadShapesStridesAndOverlap) {
  Arena arena;
  const Tensor x = Random(arena, {2, 4}, 1);
  const Tensor w = Random(arena, {3, 4}, 2);
  EXPECT_FALSE(ember::Linear(x, w, Zeros(arena, {2, 4})).ok());
  EXPECT_FALSE(ember::Linear(x, Random(arena, {3, 5}, 3), Zeros(arena, {2, 3})).ok());
  EXPECT_FALSE(ember::Linear(*x.Transpose(0, 1), w, Zeros(arena, {4, 3})).ok());
  const Tensor square = Random(arena, {4, 4}, 4);
  EXPECT_FALSE(ember::Linear(square, square, square).ok());
}

TEST(Linear, AnEmptyInputGivesZeros) {
  Arena arena;
  const Tensor out = Random(arena, {2, 3}, 1);
  ASSERT_TRUE(ember::Linear(Zeros(arena, {2, 0}), Zeros(arena, {3, 0}), out).ok());
  for (std::size_t i = 0; i < out.element_count(); ++i) EXPECT_EQ(out.data<float>()[i], 0.0F);
}

// --- RmsNorm ---

TEST(RmsNorm, MatchesTheDefinitionAndWorksInPlace) {
  Arena arena;
  const Tensor x = Random(arena, {3, 7}, 1);
  const Tensor w = Random(arena, {7}, 2);
  const Tensor out = Zeros(arena, {3, 7});
  ASSERT_TRUE(ember::RmsNorm(x, w, 1e-5F, out).ok());
  for (std::size_t r = 0; r < 3; ++r) {
    double squares = 0;
    for (std::size_t i = 0; i < 7; ++i) squares += std::pow(x.at<float>(r, i), 2);
    const double inverse = 1.0 / std::sqrt(squares / 7 + 1e-5);
    for (std::size_t i = 0; i < 7; ++i) {
      EXPECT_NEAR(out.at<float>(r, i), w.at<float>(i) * x.at<float>(r, i) * inverse, 1e-5);
    }
  }
  ASSERT_TRUE(ember::RmsNorm(x, w, 1e-5F, x).ok());
  for (std::size_t i = 0; i < x.element_count(); ++i) {
    EXPECT_EQ(x.data<float>()[i], out.data<float>()[i]);
  }
}

TEST(RmsNorm, AZeroRowStaysZeroThanksToEps) {
  Arena arena;
  const Tensor x = Zeros(arena, {1, 4});
  const Tensor w = Random(arena, {4}, 1);
  ASSERT_TRUE(ember::RmsNorm(x, w, 1e-5F, x).ok());
  for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(x.at<float>(0, i), 0.0F);
}

TEST(RmsNorm, RefusesBadArguments) {
  Arena arena;
  const Tensor x = Random(arena, {2, 4}, 1);
  EXPECT_FALSE(ember::RmsNorm(x, Random(arena, {3}, 2), 1e-5F, Zeros(arena, {2, 4})).ok());
  EXPECT_FALSE(ember::RmsNorm(x, Random(arena, {4}, 2), 0.0F, Zeros(arena, {2, 4})).ok());
  const Tensor big = Random(arena, {3, 4}, 3);
  const Tensor shifted = *big.Slice(0, 1, 2);
  EXPECT_FALSE(ember::RmsNorm(*big.Slice(0, 0, 2), Random(arena, {4}, 2), 1e-5F, shifted).ok());
}

// --- Embed ---

TEST(Embed, CopiesOneRowPerToken) {
  Arena arena;
  const Tensor table = Random(arena, {5, 3}, 1);
  const std::vector<std::int32_t> tokens{4, 0, 4};
  const Tensor out = Zeros(arena, {3, 3});
  ASSERT_TRUE(ember::Embed(table, tokens, out).ok());
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(out.at<float>(i, j), table.at<float>(static_cast<std::size_t>(tokens[i]), j));
    }
  }
}

TEST(Embed, RefusesTokensOutsideTheVocabulary) {
  Arena arena;
  const Tensor table = Random(arena, {5, 3}, 1);
  const Tensor out = Zeros(arena, {1, 3});
  for (const std::int32_t bad : {-1, 5, 1 << 30}) {
    const std::vector<std::int32_t> tokens{bad};
    const auto status = ember::Embed(table, tokens, out);
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << bad;
  }
}

// --- ApplyRope ---

TEST(ApplyRope, PositionZeroChangesNothing) {
  Arena arena;
  const Tensor x = Random(arena, {1, 2, 8}, 1);
  const Tensor before = *Tensor::CopyOf(arena, x);
  ASSERT_TRUE(ember::ApplyRope(x, 0, 10000.0F).ok());
  for (std::size_t i = 0; i < x.element_count(); ++i) {
    EXPECT_EQ(x.data<float>()[i], before.data<float>()[i]);
  }
}

TEST(ApplyRope, RotatesEachPairByPositionTimesItsFrequency) {
  Arena arena;
  const Tensor x = Random(arena, {3, 2, 8}, 1);
  const Tensor before = *Tensor::CopyOf(arena, x);
  ASSERT_TRUE(ember::ApplyRope(x, 5, 10000.0F).ok());
  for (std::size_t t = 0; t < 3; ++t) {
    for (std::size_t h = 0; h < 2; ++h) {
      for (std::size_t i = 0; i < 4; ++i) {
        const double angle =
            static_cast<double>(5 + t) * std::pow(10000.0, -2.0 * static_cast<double>(i) / 8);
        const double a = before.at<float>(t, h, i);
        const double b = before.at<float>(t, h, i + 4);
        EXPECT_NEAR(x.at<float>(t, h, i), a * std::cos(angle) - b * std::sin(angle), 1e-5);
        EXPECT_NEAR(x.at<float>(t, h, i + 4), b * std::cos(angle) + a * std::sin(angle), 1e-5);
      }
    }
  }
}

TEST(ApplyRope, KeepsLengthsAndMakesScoresDependOnDistanceOnly) {
  // The reason RoPE exists: rotating q at position m and k at position n makes q.k depend on m - n.
  Arena arena;
  const Tensor q = Random(arena, {1, 1, 16}, 1);
  const Tensor k = Random(arena, {1, 1, 16}, 2);
  const auto score_at = [&](std::size_t m, std::size_t n) {
    const Tensor qm = *Tensor::CopyOf(arena, q);
    const Tensor kn = *Tensor::CopyOf(arena, k);
    EXPECT_TRUE(ember::ApplyRope(qm, m, 10000.0F).ok());
    EXPECT_TRUE(ember::ApplyRope(kn, n, 10000.0F).ok());
    double dot = 0;
    for (std::size_t i = 0; i < 16; ++i)
      dot += static_cast<double>(qm.data<float>()[i]) * kn.data<float>()[i];
    return dot;
  };
  EXPECT_NEAR(score_at(7, 3), score_at(104, 100), 1e-3);
  EXPECT_NEAR(score_at(0, 0), score_at(50, 50), 1e-3);
}

TEST(ApplyRope, RefusesAnOddHeadDim) {
  Arena arena;
  EXPECT_FALSE(ember::ApplyRope(Random(arena, {1, 1, 3}, 1), 0, 10000.0F).ok());
}

// --- CausalAttention ---

// Attention straight from its definition, in double, one query at a time.
std::vector<double> ReferenceAttention(const Tensor& q, const Tensor& k, const Tensor& v) {
  const std::size_t queries = q.dim(0), heads = q.dim(1), d = q.dim(2);
  const std::size_t keys = k.dim(0), kv_heads = k.dim(1);
  std::vector<double> out(queries * heads * d, 0.0);
  for (std::size_t i = 0; i < queries; ++i) {
    for (std::size_t h = 0; h < heads; ++h) {
      const std::size_t g = h / (heads / kv_heads);
      const std::size_t visible = keys - queries + i + 1;
      std::vector<double> weights(visible);
      double largest = -1e300;
      for (std::size_t j = 0; j < visible; ++j) {
        double dot = 0;
        for (std::size_t e = 0; e < d; ++e)
          dot += static_cast<double>(q.at<float>(i, h, e)) * k.at<float>(j, g, e);
        weights[j] = dot / std::sqrt(static_cast<double>(d));
        largest = std::max(largest, weights[j]);
      }
      double total = 0;
      for (double& w : weights) total += (w = std::exp(w - largest));
      for (std::size_t j = 0; j < visible; ++j) {
        for (std::size_t e = 0; e < d; ++e) {
          out[(i * heads + h) * d + e] += weights[j] / total * v.at<float>(j, g, e);
        }
      }
    }
  }
  return out;
}

void ExpectMatchesReference(std::size_t queries, std::size_t keys, std::size_t heads,
                            std::size_t kv_heads) {
  Arena arena;
  const Tensor q = Random(arena, {queries, heads, 4}, 1);
  const Tensor k = Random(arena, {keys, kv_heads, 4}, 2);
  const Tensor v = Random(arena, {keys, kv_heads, 4}, 3);
  const Tensor out = Zeros(arena, {queries, heads, 4});
  const std::size_t used_before = arena.used_bytes();
  ASSERT_TRUE(ember::CausalAttention(q, k, v, out, arena).ok());
  // Scratch is given back, so only out remains.
  EXPECT_EQ(arena.used_bytes(), used_before);
  const std::vector<double> expected = ReferenceAttention(q, k, v);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(out.data<float>()[i], expected[i], 1e-5) << "element " << i;
  }
}

TEST(CausalAttention, MatchesTheDefinition) { ExpectMatchesReference(5, 5, 2, 2); }

TEST(CausalAttention, SharesKeyHeadsAcrossQueryGroups) { ExpectMatchesReference(4, 4, 6, 2); }

TEST(CausalAttention, LetsNewQueriesSeeEveryEarlierKey) {
  // Two new tokens after three cached ones, as the KV cache will do.
  ExpectMatchesReference(2, 5, 4, 1);
}

TEST(CausalAttention, TheFirstTokenOnlySeesItself) {
  Arena arena;
  const Tensor q = Random(arena, {3, 1, 4}, 1);
  const Tensor k = Random(arena, {3, 1, 4}, 2);
  const Tensor v = Random(arena, {3, 1, 4}, 3);
  const Tensor out = Zeros(arena, {3, 1, 4});
  ASSERT_TRUE(ember::CausalAttention(q, k, v, out, arena).ok());
  for (std::size_t e = 0; e < 4; ++e) EXPECT_FLOAT_EQ(out.at<float>(0, 0, e), v.at<float>(0, 0, e));
}

TEST(CausalAttention, RefusesBadShapesAndOverlap) {
  Arena arena;
  const Tensor q = Random(arena, {3, 2, 4}, 1);
  const Tensor k = Random(arena, {3, 2, 4}, 2);
  const Tensor v = Random(arena, {3, 2, 4}, 3);
  EXPECT_FALSE(ember::CausalAttention(q, Random(arena, {2, 2, 4}, 4), Random(arena, {2, 2, 4}, 5),
                                      Zeros(arena, {3, 2, 4}), arena)
                   .ok());  // fewer keys than queries
  EXPECT_FALSE(ember::CausalAttention(Random(arena, {3, 3, 4}, 6), k, v, Zeros(arena, {3, 3, 4}),
                                      arena)
                   .ok());  // 3 heads do not split over 2 key heads
  EXPECT_FALSE(ember::CausalAttention(q, k, v, q, arena).ok());
}

}  // namespace
