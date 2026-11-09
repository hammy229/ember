#include "ember/quant.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "ember/arena.h"
#include "ember/forward.h"
#include "ember/generate.h"
#include "ember/layers.h"
#include "ember/llama.h"
#include "ember/tensor.h"
#include "ember/threads.h"
#include "ember/tokenizer.h"
#include "tiny_llama.h"

namespace {

using ember::Arena;
using ember::Q8Matrix;
using ember::Tensor;

Tensor Random(Arena& arena, std::size_t rows, std::size_t cols, std::uint32_t seed,
              float spread = 1.0F) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, spread);
  std::vector<float> values(rows * cols);
  for (float& v : values) v = normal(rng);
  return Tensor::FromValues<float>(arena, *ember::Shape::Make({rows, cols}), values).value();
}

float Dequantized(const Q8Matrix& q, std::size_t r, std::size_t c) {
  return static_cast<float>(q.values[r * q.cols + c]) *
         q.scales[r * (q.cols / Q8Matrix::kBlock) + c / Q8Matrix::kBlock];
}

TEST(Quant, EveryWeightMovesByAtMostHalfAStep) {
  Arena arena;
  const Tensor w = Random(arena, 20, 96, 1);
  const auto q = ember::QuantizeQ8(w, arena);
  ASSERT_TRUE(q.ok()) << q.status().ToString();
  EXPECT_EQ(q->bytes(), 20U * 96U + 20U * 3U * sizeof(float));
  for (std::size_t r = 0; r < 20; ++r) {
    for (std::size_t c = 0; c < 96; ++c) {
      const float scale = q->scales[r * 3 + c / 32];
      EXPECT_LE(std::abs(Dequantized(*q, r, c) - w.at<float>(r, c)), scale / 2 + 1e-7F);
    }
  }
}

TEST(Quant, TheLargestWeightInEachBlockIsExact) {
  Arena arena;
  std::vector<float> values(32, 0.25F);
  values[7] = -3.0F;
  const Tensor w = Tensor::FromValues<float>(arena, *ember::Shape::Make({1, 32}), values).value();
  const auto q = ember::QuantizeQ8(w, arena);
  ASSERT_TRUE(q.ok());
  EXPECT_EQ(q->values[7], -127);
  EXPECT_FLOAT_EQ(Dequantized(*q, 0, 7), -3.0F);
}

TEST(Quant, AnAllZeroBlockStaysZero) {
  Arena arena;
  const std::vector<float> zeros(64, 0.0F);
  const Tensor w = Tensor::FromValues<float>(arena, *ember::Shape::Make({2, 32}), zeros).value();
  const auto q = ember::QuantizeQ8(w, arena);
  ASSERT_TRUE(q.ok());
  for (std::size_t i = 0; i < 64; ++i) EXPECT_EQ(q->values[i], 0);
  EXPECT_EQ(q->scales[0], 0.0F);
}

TEST(Quant, RefusesWhatItCannotQuantize) {
  Arena arena;
  EXPECT_FALSE(ember::QuantizeQ8(Random(arena, 4, 33, 2), arena).ok());
  std::vector<float> bad(32, 1.0F);
  bad[3] = std::numeric_limits<float>::quiet_NaN();
  const Tensor w = Tensor::FromValues<float>(arena, *ember::Shape::Make({1, 32}), bad).value();
  EXPECT_FALSE(ember::QuantizeQ8(w, arena).ok());
}

TEST(Quant, LinearQ8MatchesLinearOnTheDequantizedWeights) {
  Arena arena;
  const Tensor x = Random(arena, 3, 128, 3);
  const Tensor w = Random(arena, 70, 128, 4);
  const auto q = ember::QuantizeQ8(w, arena);
  ASSERT_TRUE(q.ok());
  std::vector<float> back(70 * 128);
  for (std::size_t r = 0; r < 70; ++r) {
    for (std::size_t c = 0; c < 128; ++c) back[r * 128 + c] = Dequantized(*q, r, c);
  }
  const Tensor w2 = Tensor::FromValues<float>(arena, *ember::Shape::Make({70, 128}), back).value();
  const auto shape = *ember::Shape::Make({3, 70});
  const Tensor a = Tensor::Uninitialized(arena, ember::DType::kF32, shape).value();
  const Tensor b = Tensor::Uninitialized(arena, ember::DType::kF32, shape).value();
  ASSERT_TRUE(ember::LinearQ8(x, *q, a).ok());
  ASSERT_TRUE(ember::Linear(x, w2, b).ok());
  for (std::size_t i = 0; i < 3 * 70; ++i) {
    EXPECT_NEAR(a.data<float>()[i], b.data<float>()[i], 1e-4) << i;
  }
}

TEST(Quant, LinearQ8IsBitIdenticalForAnyThreadCount) {
  Arena arena;
  const Tensor x = Random(arena, 2, 256, 5);
  const Tensor w = Random(arena, 500, 256, 6);
  const auto q = ember::QuantizeQ8(w, arena);
  ASSERT_TRUE(q.ok());
  std::vector<float> first;
  for (const std::size_t threads : {1U, 3U, 8U}) {
    ember::SetThreadCount(threads);
    const Tensor out =
        Tensor::Uninitialized(arena, ember::DType::kF32, *ember::Shape::Make({2, 500})).value();
    ASSERT_TRUE(ember::LinearQ8(x, *q, out).ok());
    std::vector<float> now(out.data<float>(), out.data<float>() + 1000);
    if (first.empty()) {
      first = now;
    } else {
      EXPECT_EQ(std::memcmp(first.data(), now.data(), 1000 * sizeof(float)), 0);
    }
  }
  ember::SetThreadCount(1);
}

TEST(Quant, LinearQ8RefusesMismatchedShapes) {
  Arena arena;
  const Tensor w = Random(arena, 8, 64, 7);
  const auto q = ember::QuantizeQ8(w, arena);
  ASSERT_TRUE(q.ok());
  const Tensor x = Random(arena, 2, 32, 8);
  const Tensor out =
      Tensor::Uninitialized(arena, ember::DType::kF32, *ember::Shape::Make({2, 8})).value();
  EXPECT_FALSE(ember::LinearQ8(x, *q, out).ok());
  EXPECT_FALSE(ember::LinearQ8(Random(arena, 2, 64, 9), Q8Matrix{}, out).ok());
}

ember::testing::TinyLlamaOptions QuantizableModel() {
  ember::testing::TinyLlamaOptions o;
  o.hidden = 64;
  o.intermediate = 96;
  o.heads = 4;
  o.kv_heads = 2;
  o.vocab = 200;
  o.max_positions = 24;
  o.seed = 21;
  return o;
}

TEST(Quant, AnEightBitModelStaysCloseToTheFullOne) {
  const auto options = QuantizableModel();
  const std::vector<std::byte> bytes = ember::testing::TinyLlamaWeights(options);
  const std::string config = ember::testing::TinyLlamaConfig(options);
  const auto full = ember::LlamaModel::FromBytes(config, bytes);
  ember::LoadOptions load;
  load.q8 = true;
  const auto small = ember::LlamaModel::FromBytes(config, bytes, load);
  ASSERT_TRUE(full.ok() && small.ok()) << small.status().ToString();
  EXPECT_TRUE((*small)->q8());
  EXPECT_EQ((*small)->parameter_count(), (*full)->parameter_count());
  EXPECT_TRUE((*small)->weights().layers[0].query.element_count() == 0 ||
              (*small)->weights().layers[0].query.rank() == 0);

  const std::vector<std::int32_t> tokens{3, 50, 199, 7, 7, 120, 64, 0, 18};
  Arena a;
  Arena b;
  const auto la = ember::Forward(**full, tokens, a);
  const auto lb = ember::Forward(**small, tokens, b);
  ASSERT_TRUE(la.ok() && lb.ok()) << lb.status().ToString();
  double diff = 0;
  double size = 0;
  for (std::size_t i = 0; i < la->element_count(); ++i) {
    const double d = la->data<float>()[i] - lb->data<float>()[i];
    diff += d * d;
    size += static_cast<double>(la->data<float>()[i]) * la->data<float>()[i];
  }
  // 8 bits cost a little accuracy on purpose. Random weights amplify it more than trained ones do,
  // so this only checks the error stays small on average; the real model's check is below.
  EXPECT_LT(std::sqrt(diff / size), 0.08) << "relative RMS error of the logits";
}

// The real 15M model: the 8-bit copy must pick the same next token almost everywhere.
TEST(Quant, TheRealTinyModelKeepsItsPicksInEightBits) {
  const char* directory = std::getenv("EMBER_MODEL_DIR");
  if (directory == nullptr || *directory == '\0') GTEST_SKIP() << "set EMBER_MODEL_DIR";
  const auto full = ember::LlamaModel::Load(directory);
  ember::LoadOptions load;
  load.q8 = true;
  const auto small = ember::LlamaModel::Load(directory, load);
  ASSERT_TRUE(full.ok() && small.ok()) << small.status().ToString();
  EXPECT_LT((*small)->widened_bytes(), (*full)->widened_bytes());
  // Real text: random ids make every prediction a near tie, which says nothing about quality.
  const auto tokenizer = ember::Tokenizer::Load(std::string(directory) + "/tokenizer.json");
  ASSERT_TRUE(tokenizer.ok());
  const std::vector<std::int32_t> tokens = tokenizer->Encode(
      "Once upon a time, there was a little girl named Lily. She loved to play outside in the "
      "park with her dog, Max. One day, Lily and Max found a big red ball under a tree. Lily "
      "wanted to keep the ball, but her mom said they should look for its owner. They walked "
      "around the park and asked everyone. At last, a little boy named Tim said it was his ball. "
      "Tim was very happy and said thank you. Lily felt proud because she did the right thing.",
      true);
  Arena a;
  Arena b;
  const auto la = ember::Forward(**full, tokens, a);
  const auto lb = ember::Forward(**small, tokens, b);
  ASSERT_TRUE(la.ok() && lb.ok());
  const std::size_t vocab = la->dim(1);
  std::size_t same = 0;
  double diff = 0;
  double size = 0;
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    const float* ra = la->data<float>() + t * vocab;
    const float* rb = lb->data<float>() + t * vocab;
    std::size_t pa = 0;
    std::size_t pb = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
      if (ra[i] > ra[pa]) pa = i;
      if (rb[i] > rb[pb]) pb = i;
      diff += static_cast<double>(ra[i] - rb[i]) * (ra[i] - rb[i]);
      size += static_cast<double>(ra[i]) * ra[i];
    }
    same += pa == pb ? 1 : 0;
  }
  std::cout << "tiny model in 8 bits: same pick at " << same << " of " << tokens.size()
            << " positions, relative RMS error " << std::sqrt(diff / size) << "\n";
  EXPECT_GE(same, tokens.size() * 95 / 100);
  EXPECT_LT(std::sqrt(diff / size), 0.02);
}

TEST(Quant, AnEightBitModelGeneratesWithTheKvCache) {
  const auto options = QuantizableModel();
  const std::vector<std::byte> bytes = ember::testing::TinyLlamaWeights(options);
  ember::LoadOptions load;
  load.q8 = true;
  const auto model =
      ember::LlamaModel::FromBytes(ember::testing::TinyLlamaConfig(options), bytes, load);
  ASSERT_TRUE(model.ok());
  ember::GenerateOptions generate;
  generate.max_new_tokens = 10;
  const auto result = ember::Generate(**model, std::vector<std::int32_t>{1, 2, 3}, generate);
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  EXPECT_EQ(result->tokens.size(), 10U);
}

TEST(Quant, RefusesToQuantizeAModelWithOddWidths) {
  const ember::testing::TinyLlamaOptions options;  // hidden 8, not a multiple of 32
  const std::vector<std::byte> bytes = ember::testing::TinyLlamaWeights(options);
  ember::LoadOptions load;
  load.q8 = true;
  EXPECT_FALSE(
      ember::LlamaModel::FromBytes(ember::testing::TinyLlamaConfig(options), bytes, load).ok());
}

}  // namespace
