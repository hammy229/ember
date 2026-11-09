// Checks the forward pass against a second implementation written straight from the Llama
// definition, in double precision, with plain loops and none of the engine's ops. If both agree on
// random models of several shapes, a bug would have to be in both, written two different ways.

#include "ember/forward.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "tiny_llama.h"

namespace {

using ember::Arena;
using ember::LlamaConfig;
using ember::LlamaModel;
using ember::TensorView;
using ember::testing::TinyLlamaConfig;
using ember::testing::TinyLlamaOptions;
using ember::testing::TinyLlamaWeights;

using Matrix = std::vector<std::vector<double>>;

double W(const TensorView& t, std::size_t i) { return t.data<float>()[i]; }

// x [rows, in] times weight [out, in] turned sideways.
Matrix Project(const Matrix& x, const TensorView& weight) {
  const std::size_t out = weight.dim(0), in = weight.dim(1);
  Matrix y(x.size(), std::vector<double>(out, 0.0));
  for (std::size_t r = 0; r < x.size(); ++r) {
    for (std::size_t o = 0; o < out; ++o) {
      for (std::size_t k = 0; k < in; ++k) y[r][o] += x[r][k] * W(weight, o * in + k);
    }
  }
  return y;
}

Matrix Norm(const Matrix& x, const TensorView& weight, double eps) {
  Matrix y = x;
  for (auto& row : y) {
    double squares = 0;
    for (const double v : row) squares += v * v;
    const double inverse = 1.0 / std::sqrt(squares / static_cast<double>(row.size()) + eps);
    for (std::size_t i = 0; i < row.size(); ++i) row[i] = row[i] * inverse * W(weight, i);
  }
  return y;
}

// Rotates each head's pairs (j, j + d/2) by position * theta^(-2j/d).
void Rope(Matrix& x, std::size_t heads, std::size_t d, double theta) {
  for (std::size_t t = 0; t < x.size(); ++t) {
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t j = 0; j < d / 2; ++j) {
        const double angle =
            static_cast<double>(t) *
            std::pow(theta, -2.0 * static_cast<double>(j) / static_cast<double>(d));
        double& a = x[t][h * d + j];
        double& b = x[t][h * d + j + d / 2];
        const double first = a, second = b;
        a = first * std::cos(angle) - second * std::sin(angle);
        b = second * std::cos(angle) + first * std::sin(angle);
      }
    }
  }
}

// Qwen3's RMSNorm over each head's d values on its own, with one weight shared by every head.
void HeadNorm(Matrix& x, std::size_t heads, std::size_t d, const TensorView& weight, double eps) {
  for (auto& row : x) {
    for (std::size_t h = 0; h < heads; ++h) {
      double squares = 0;
      for (std::size_t e = 0; e < d; ++e) squares += row[h * d + e] * row[h * d + e];
      const double inverse = 1.0 / std::sqrt(squares / static_cast<double>(d) + eps);
      for (std::size_t e = 0; e < d; ++e) row[h * d + e] *= inverse * W(weight, e);
    }
  }
}

Matrix ReferenceForward(const LlamaModel& model, const std::vector<std::int32_t>& tokens) {
  const LlamaConfig& c = model.config();
  const auto& w = model.weights();
  const std::size_t n = tokens.size(), d = c.head_dim;
  Matrix x(n, std::vector<double>(c.hidden_size));
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t i = 0; i < c.hidden_size; ++i) {
      x[t][i] = W(w.embedding, static_cast<std::size_t>(tokens[t]) * c.hidden_size + i);
    }
  }
  for (const auto& layer : w.layers) {
    const Matrix h = Norm(x, layer.attention_norm, c.rms_norm_eps);
    Matrix q = Project(h, layer.query), k = Project(h, layer.key);
    const Matrix v = Project(h, layer.value);
    if (c.qk_norm) {
      HeadNorm(q, c.head_count, d, layer.query_norm, c.rms_norm_eps);
      HeadNorm(k, c.kv_head_count, d, layer.key_norm, c.rms_norm_eps);
    }
    Rope(q, c.head_count, d, c.rope_theta);
    Rope(k, c.kv_head_count, d, c.rope_theta);
    Matrix attended(n, std::vector<double>(c.head_count * d, 0.0));
    for (std::size_t head = 0; head < c.head_count; ++head) {
      const std::size_t g = head / (c.head_count / c.kv_head_count);
      for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> p(i + 1);
        double largest = -1e300, total = 0;
        for (std::size_t j = 0; j <= i; ++j) {
          double dot = 0;
          for (std::size_t e = 0; e < d; ++e) dot += q[i][head * d + e] * k[j][g * d + e];
          p[j] = dot / std::sqrt(static_cast<double>(d));
          largest = std::max(largest, p[j]);
        }
        for (double& value : p) total += (value = std::exp(value - largest));
        for (std::size_t j = 0; j <= i; ++j) {
          for (std::size_t e = 0; e < d; ++e) {
            attended[i][head * d + e] += p[j] / total * v[j][g * d + e];
          }
        }
      }
    }
    const Matrix o = Project(attended, layer.output);
    for (std::size_t t = 0; t < n; ++t) {
      for (std::size_t i = 0; i < c.hidden_size; ++i) x[t][i] += o[t][i];
    }
    const Matrix m = Norm(x, layer.mlp_norm, c.rms_norm_eps);
    Matrix gate = Project(m, layer.gate);
    const Matrix up = Project(m, layer.up);
    for (std::size_t t = 0; t < n; ++t) {
      for (std::size_t i = 0; i < gate[t].size(); ++i) {
        gate[t][i] = gate[t][i] / (1.0 + std::exp(-gate[t][i])) * up[t][i];
      }
    }
    const Matrix down = Project(gate, layer.down);
    for (std::size_t t = 0; t < n; ++t) {
      for (std::size_t i = 0; i < c.hidden_size; ++i) x[t][i] += down[t][i];
    }
  }
  return Project(Norm(x, w.final_norm, c.rms_norm_eps), w.lm_head);
}

struct Loaded {
  std::vector<std::byte> bytes;
  std::unique_ptr<LlamaModel> model;
};

Loaded Load(const TinyLlamaOptions& options) {
  Loaded loaded{TinyLlamaWeights(options), nullptr};
  auto model = LlamaModel::FromBytes(TinyLlamaConfig(options), loaded.bytes);
  EXPECT_TRUE(model.ok()) << model.status().ToString();
  if (model.ok()) loaded.model = std::move(model).value();
  return loaded;
}

void ExpectMatchesReference(const TinyLlamaOptions& options,
                            const std::vector<std::int32_t>& tokens) {
  const Loaded loaded = Load(options);
  ASSERT_NE(loaded.model, nullptr);
  Arena arena;
  const auto logits = ember::Forward(*loaded.model, tokens, arena);
  ASSERT_TRUE(logits.ok()) << logits.status().ToString();
  ASSERT_EQ(logits->dim(0), tokens.size());
  ASSERT_EQ(logits->dim(1), options.vocab);
  const Matrix expected = ReferenceForward(*loaded.model, tokens);
  double worst = 0;
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    for (std::size_t i = 0; i < options.vocab; ++i) {
      worst = std::max(worst, std::abs(logits->at<float>(t, i) - expected[t][i]));
    }
  }
  // f32 against double through several layers. Logits here are around 1 to 10 in size.
  EXPECT_LT(worst, 1e-4) << "largest difference from the reference";
}

TEST(Forward, MatchesTheReferenceOnATinyModel) { ExpectMatchesReference({}, {1, 5, 3, 10, 0, 7}); }

TEST(Forward, MatchesTheReferenceWithGroupedQueryAttention) {
  TinyLlamaOptions options;
  options.hidden = 16;
  options.heads = 4;
  options.kv_heads = 1;
  options.layers = 3;
  options.seed = 5;
  ExpectMatchesReference(options, {2, 2, 9, 4});
}

TEST(Forward, MatchesTheReferenceWithTiedEmbeddingsAndF16Weights) {
  TinyLlamaOptions options;
  options.tie = true;
  options.f16 = true;
  options.seed = 9;
  ExpectMatchesReference(options, {3, 1, 4, 1, 5, 9, 2, 6});
}

TEST(Forward, MatchesTheReferenceWithTwoKeyHeadsForFour) {
  // With one key head, h / group and h % kv_heads both give 0. Two tells them apart.
  TinyLlamaOptions options;
  options.hidden = 16;
  options.heads = 4;
  options.kv_heads = 2;
  options.seed = 3;
  ExpectMatchesReference(options, {5, 8, 1, 0, 2});
}

TEST(Forward, MatchesTheReferenceWhenHeadsDoNotSpanTheHiddenSize) {
  // 3 heads of 6 is 18 wide, against a hidden size of 8, as some newer models do.
  TinyLlamaOptions options;
  options.heads = 3;
  options.kv_heads = 1;
  options.head_dim = 6;
  options.seed = 4;
  ExpectMatchesReference(options, {1, 2, 3, 4});
}

TEST(Forward, MatchesTheReferenceWithOtherRopeAndNormSettingsOverALongSequence) {
  TinyLlamaOptions options;
  options.max_positions = 64;
  options.rope_theta = "500000.0";
  options.rms_norm_eps = "1e-06";
  options.seed = 6;
  std::vector<std::int32_t> tokens;
  for (std::int32_t i = 0; i < 40; ++i) tokens.push_back((i * 7) % 11);
  ExpectMatchesReference(options, tokens);
}

TEST(Forward, MatchesTheReferenceForAQwen3StyleModel) {
  TinyLlamaOptions options;
  options.qk_norm = true;
  options.hidden = 16;
  options.heads = 4;
  options.kv_heads = 2;
  options.head_dim = 8;  // wider heads than hidden / heads, as Qwen3 0.6B has
  options.tie = true;
  options.bf16 = true;
  options.rope_theta = "1000000.0";
  options.rms_norm_eps = "1e-06";
  options.seed = 12;
  ExpectMatchesReference(options, {3, 1, 4, 1, 5, 9, 2, 6});
}

TEST(Forward, MatchesTheReferenceForOneToken) { ExpectMatchesReference({}, {4}); }

TEST(Forward, EarlierLogitsIgnoreLaterTokens) {
  // Causal: appending tokens must not change any earlier row.
  const Loaded loaded = Load({});
  Arena arena;
  const std::vector<std::int32_t> longer{6, 2, 8, 1, 3};
  const std::vector<std::int32_t> shorter(longer.begin(), longer.begin() + 3);
  const auto full = ember::Forward(*loaded.model, longer, arena);
  const auto prefix = ember::Forward(*loaded.model, shorter, arena);
  ASSERT_TRUE(full.ok() && prefix.ok());
  for (std::size_t t = 0; t < shorter.size(); ++t) {
    for (std::size_t i = 0; i < 11; ++i) {
      EXPECT_NEAR(full->at<float>(t, i), prefix->at<float>(t, i), 1e-6) << t << ", " << i;
    }
  }
}

TEST(Forward, LeavesOnlyTheLogitsInTheArena) {
  const Loaded loaded = Load({});
  Arena arena;
  const std::vector<std::int32_t> tokens{1, 2, 3};
  const auto logits = ember::Forward(*loaded.model, tokens, arena);
  ASSERT_TRUE(logits.ok());
  EXPECT_LE(arena.used_bytes(),
            logits->byte_size() + Arena::kDefaultAlignment + Arena::kRedzoneBytes);
}

TEST(Forward, RefusesSequencesItCannotRun) {
  const Loaded loaded = Load({});
  Arena arena;
  EXPECT_FALSE(ember::Forward(*loaded.model, std::vector<std::int32_t>{}, arena).ok());
  EXPECT_FALSE(ember::Forward(*loaded.model, std::vector<std::int32_t>(17, 1), arena).ok());
  EXPECT_TRUE(ember::Forward(*loaded.model, std::vector<std::int32_t>(16, 1), arena).ok());
  EXPECT_FALSE(ember::Forward(*loaded.model, std::vector<std::int32_t>{1, 11}, arena).ok());
  EXPECT_FALSE(ember::Forward(*loaded.model, std::vector<std::int32_t>{-1}, arena).ok());
}

// Feeds `tokens` through ForwardCached in pieces of the given sizes and checks every row against
// Forward on the whole sequence.
void ExpectCachedMatchesForward(const TinyLlamaOptions& options,
                                const std::vector<std::int32_t>& tokens,
                                const std::vector<std::size_t>& pieces) {
  const Loaded loaded = Load(options);
  ASSERT_NE(loaded.model, nullptr);
  Arena arena;
  const auto full = ember::Forward(*loaded.model, tokens, arena);
  ASSERT_TRUE(full.ok()) << full.status().ToString();
  auto cache = ember::KvCache::Make(loaded.model->config(), tokens.size());
  ASSERT_TRUE(cache.ok()) << cache.status().ToString();
  std::size_t start = 0;
  double worst = 0;
  for (const std::size_t piece : pieces) {
    Arena scratch;
    const std::span<const std::int32_t> part(tokens.data() + start, piece);
    const auto logits = ember::ForwardCached(*loaded.model, part, **cache, scratch, false);
    ASSERT_TRUE(logits.ok()) << logits.status().ToString();
    ASSERT_EQ(logits->dim(0), piece);
    for (std::size_t t = 0; t < piece; ++t) {
      for (std::size_t i = 0; i < options.vocab; ++i) {
        worst = std::max(worst, std::abs(static_cast<double>(logits->at<float>(t, i)) -
                                         full->at<float>(start + t, i)));
      }
    }
    start += piece;
    EXPECT_EQ((*cache)->size(), start);
  }
  ASSERT_EQ(start, tokens.size());
  EXPECT_LT(worst, 1e-5) << "largest difference between cached and uncached logits";
}

TEST(ForwardCached, OneTokenAtATimeMatchesForward) {
  ExpectCachedMatchesForward({}, {1, 5, 3, 10, 0, 7, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1});
}

TEST(ForwardCached, APromptThenSingleTokensMatchesForward) {
  TinyLlamaOptions options;
  options.hidden = 16;
  options.heads = 4;
  options.kv_heads = 2;
  options.layers = 3;
  options.seed = 9;
  ExpectCachedMatchesForward(options, {2, 2, 9, 4, 1, 0, 6, 3, 8, 5}, {5, 1, 1, 3});
}

TEST(ForwardCached, FillsTheWholeContext) {
  std::vector<std::int32_t> tokens(16);
  for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = static_cast<std::int32_t>(i % 11);
  ExpectCachedMatchesForward({}, tokens, {7, 1, 1, 1, 6});
}

TEST(ForwardCached, LastOnlyGivesTheLastRow) {
  const Loaded loaded = Load({});
  Arena arena;
  const std::vector<std::int32_t> tokens{4, 1, 7};
  const auto full = ember::Forward(*loaded.model, tokens, arena);
  ASSERT_TRUE(full.ok());
  auto cache = ember::KvCache::Make(loaded.model->config(), 8);
  ASSERT_TRUE(cache.ok());
  const auto last = ember::ForwardCached(*loaded.model, tokens, **cache, arena, true);
  ASSERT_TRUE(last.ok()) << last.status().ToString();
  ASSERT_EQ(last->dim(0), 1U);
  for (std::size_t i = 0; i < 11; ++i)
    EXPECT_NEAR(last->at<float>(0, i), full->at<float>(2, i), 1e-5);
}

TEST(ForwardCached, TruncateRewindsToAPrefix) {
  const Loaded loaded = Load({});
  auto cache = ember::KvCache::Make(loaded.model->config(), 8);
  ASSERT_TRUE(cache.ok());
  Arena arena;
  const std::vector<std::int32_t> prompt{3, 3, 8};
  ASSERT_TRUE(ember::ForwardCached(*loaded.model, prompt, **cache, arena, true).ok());
  const auto first =
      ember::ForwardCached(*loaded.model, std::vector<std::int32_t>{5}, **cache, arena, true);
  ASSERT_TRUE(first.ok());
  (*cache)->Truncate(3);
  const auto again =
      ember::ForwardCached(*loaded.model, std::vector<std::int32_t>{5}, **cache, arena, true);
  ASSERT_TRUE(again.ok());
  for (std::size_t i = 0; i < 11; ++i) EXPECT_EQ(first->at<float>(0, i), again->at<float>(0, i));
}

TEST(ForwardCached, RefusesWhatItCannotRunAndKeepsTheCache) {
  const Loaded loaded = Load({});
  Arena arena;
  EXPECT_FALSE(ember::KvCache::Make(loaded.model->config(), 0).ok());
  EXPECT_FALSE(ember::KvCache::Make(loaded.model->config(), 17).ok());
  auto cache = ember::KvCache::Make(loaded.model->config(), 4);
  ASSERT_TRUE(cache.ok());
  EXPECT_EQ((*cache)->bytes(), 2U * 2U * 4U * 8U * sizeof(float));
  ASSERT_TRUE(
      ember::ForwardCached(*loaded.model, std::vector<std::int32_t>{1, 2, 3}, **cache, arena, false)
          .ok());
  EXPECT_FALSE(
      ember::ForwardCached(*loaded.model, std::vector<std::int32_t>{1, 2}, **cache, arena, false)
          .ok());
  EXPECT_FALSE(
      ember::ForwardCached(*loaded.model, std::vector<std::int32_t>{}, **cache, arena, false).ok());
  EXPECT_FALSE(
      ember::ForwardCached(*loaded.model, std::vector<std::int32_t>{11}, **cache, arena, false)
          .ok());
  EXPECT_EQ((*cache)->size(), 3U);

  TinyLlamaOptions other;
  other.layers = 3;
  const Loaded different = Load(other);
  EXPECT_FALSE(
      ember::ForwardCached(*different.model, std::vector<std::int32_t>{1}, **cache, arena, false)
          .ok());
}

TEST(ForwardCached, MatchesForwardForAQwen3StyleModel) {
  TinyLlamaOptions options;
  options.qk_norm = true;
  options.hidden = 16;
  options.heads = 4;
  options.kv_heads = 2;
  options.head_dim = 8;
  options.seed = 13;
  ExpectCachedMatchesForward(options, {3, 1, 4, 1, 5, 9, 2, 6}, {4, 1, 1, 2});
}

TEST(ForwardTrace, TracingChangesNothingAndRecordsEveryLayer) {
  TinyLlamaOptions options;
  options.hidden = 16;
  options.heads = 4;
  options.kv_heads = 2;
  options.layers = 3;
  options.seed = 31;
  const Loaded loaded = Load(options);
  const std::vector<std::int32_t> tokens{1, 5, 3, 10, 0, 7};
  Arena a;
  Arena b;
  const auto plain = ember::Forward(*loaded.model, tokens, a);
  ember::ForwardTrace trace;
  const auto traced = ember::Forward(*loaded.model, tokens, b, &trace);
  ASSERT_TRUE(plain.ok() && traced.ok());
  for (std::size_t i = 0; i < plain->element_count(); ++i) {
    ASSERT_EQ(plain->data<float>()[i], traced->data<float>()[i]);
  }
  ASSERT_EQ(trace.attention.size(), 3U);
  ASSERT_EQ(trace.residual.size(), 3U);
  const std::size_t n = tokens.size();
  for (const auto& layer : trace.attention) {
    ASSERT_EQ(layer.size(), 4U * n * n);
    for (std::size_t h = 0; h < 4; ++h) {
      for (std::size_t i = 0; i < n; ++i) {
        double total = 0;
        for (std::size_t j = 0; j < n; ++j) {
          const float p = layer[(h * n + i) * n + j];
          if (j > i) EXPECT_EQ(p, 0.0F) << "token " << i << " looked ahead at " << j;
          total += p;
        }
        EXPECT_NEAR(total, 1.0, 1e-5);
      }
    }
  }
  for (const auto& layer : trace.residual) EXPECT_EQ(layer.size(), n * 16U);
}

TEST(ForwardTrace, TheLastLayerOfTheLensIsTheModelsOwnPrediction) {
  const Loaded loaded = Load({});
  const std::vector<std::int32_t> tokens{4, 1, 7, 2, 9};
  Arena arena;
  ember::ForwardTrace trace;
  const auto logits = ember::Forward(*loaded.model, tokens, arena, &trace);
  ASSERT_TRUE(logits.ok());
  Arena lens_arena;
  const auto lens = ember::LogitLens(*loaded.model, trace, lens_arena);
  ASSERT_TRUE(lens.ok()) << lens.status().ToString();
  ASSERT_EQ(lens->size(), 2U);
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < 11; ++i) {
      if (logits->at<float>(t, i) > logits->at<float>(t, best)) best = i;
    }
    EXPECT_EQ((*lens)[1][t].id, static_cast<std::int32_t>(best));
    EXPECT_GT((*lens)[1][t].probability, 0.0F);
  }
  EXPECT_FALSE(ember::LogitLens(*loaded.model, ember::ForwardTrace{}, lens_arena).ok());
}

}  // namespace
