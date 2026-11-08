// Checks the forward pass against a second implementation written straight from the Llama
// definition, in double precision, with plain loops and none of the engine's ops. If both agree on
// random models of several shapes, a bug would have to be in both, written two different ways.

#include "ember/forward.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
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

}  // namespace
