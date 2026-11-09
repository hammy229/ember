#include "ember/generate.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ember/forward.h"
#include "ember/llama.h"
#include "tiny_llama.h"

namespace {

using ember::GenerateOptions;
using ember::GenerateResult;
using ember::LlamaModel;
using ember::testing::TinyLlamaConfig;
using ember::testing::TinyLlamaOptions;
using ember::testing::TinyLlamaWeights;

struct Loaded {
  std::vector<std::byte> bytes;
  std::unique_ptr<LlamaModel> model;
};

Loaded Load(const TinyLlamaOptions& options = {}) {
  Loaded loaded{TinyLlamaWeights(options), nullptr};
  auto model = LlamaModel::FromBytes(TinyLlamaConfig(options), loaded.bytes);
  EXPECT_TRUE(model.ok()) << model.status().ToString();
  if (model.ok()) loaded.model = std::move(model).value();
  return loaded;
}

// Greedy decoding the slow way: rerun Forward on the whole sequence for every token.
std::vector<std::int32_t> SlowGreedy(const LlamaModel& model, std::vector<std::int32_t> tokens,
                                     std::size_t count) {
  std::vector<std::int32_t> out;
  for (std::size_t n = 0; n < count; ++n) {
    ember::Arena arena;
    const auto logits = ember::Forward(model, tokens, arena);
    EXPECT_TRUE(logits.ok());
    const std::size_t vocab = logits->dim(1);
    const float* row = logits->data<float>() + (tokens.size() - 1) * vocab;
    std::size_t best = 0;
    for (std::size_t i = 1; i < vocab; ++i) {
      if (row[i] > row[best]) best = i;
    }
    out.push_back(static_cast<std::int32_t>(best));
    tokens.push_back(static_cast<std::int32_t>(best));
  }
  return out;
}

std::vector<std::int32_t> Ids(const GenerateResult& result) {
  std::vector<std::int32_t> ids;
  for (const auto& t : result.tokens) ids.push_back(t.id);
  return ids;
}

TEST(Generate, GreedyMatchesRerunningTheWholeSequence) {
  const Loaded loaded = Load();
  const std::vector<std::int32_t> prompt{1, 4, 9};
  GenerateOptions options;
  options.max_new_tokens = 10;
  const auto result = ember::Generate(*loaded.model, prompt, options);
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  EXPECT_EQ(result->stop, GenerateResult::Stop::kMaxTokens);
  EXPECT_EQ(Ids(*result), SlowGreedy(*loaded.model, prompt, 10));
}

TEST(Generate, ReportsTheUncertaintyOfEachPick) {
  const Loaded loaded = Load();
  GenerateOptions options;
  options.max_new_tokens = 5;
  const auto result = ember::Generate(*loaded.model, std::vector<std::int32_t>{2}, options);
  ASSERT_TRUE(result.ok());
  for (const auto& token : result->tokens) {
    // Greedy picks the most likely token, so it is the one the stats name.
    EXPECT_EQ(token.id, token.stats.top1_id);
    EXPECT_GE(token.stats.entropy, 0.0F);
    EXPECT_LE(token.stats.entropy, std::log(11.0F) + 1e-5F);
    EXPECT_GE(token.stats.margin, 0.0F);
  }
}

TEST(Generate, StopsAtTheStopTokenAndReportsIt) {
  const Loaded loaded = Load();
  GenerateOptions options;
  options.max_new_tokens = 10;
  const std::vector<std::int32_t> prompt{1, 4, 9};
  const std::vector<std::int32_t> greedy = SlowGreedy(*loaded.model, prompt, 10);
  options.stop_tokens = {100, greedy[3]};  // 100 is outside the vocab, so never comes out
  const auto result = ember::Generate(*loaded.model, prompt, options);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->stop, GenerateResult::Stop::kStopToken);
  const std::vector<std::int32_t> ids = Ids(*result);
  ASSERT_FALSE(ids.empty());
  EXPECT_EQ(ids.back(), greedy[3]);
  EXPECT_LE(ids.size(), 4U);
}

TEST(Generate, TheCallbackCanStopEarly) {
  const Loaded loaded = Load();
  GenerateOptions options;
  options.max_new_tokens = 10;
  std::size_t calls = 0;
  const auto result = ember::Generate(*loaded.model, std::vector<std::int32_t>{3}, options,
                                      [&](const ember::GeneratedToken&, std::size_t index) {
                                        ++calls;
                                        return index < 2;
                                      });
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->stop, GenerateResult::Stop::kCallback);
  EXPECT_EQ(result->tokens.size(), 3U);
  EXPECT_EQ(calls, 3U);
}

TEST(Generate, StopsWhenTheContextIsFull) {
  const Loaded loaded = Load();  // context of 16
  GenerateOptions options;
  options.max_new_tokens = 100;
  const auto result =
      ember::Generate(*loaded.model, std::vector<std::int32_t>{1, 2, 3, 4}, options);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->stop, GenerateResult::Stop::kContextFull);
  // 4 prompt tokens and 12 fed back fill 16 positions; the 13th new token is picked from the last.
  EXPECT_EQ(result->tokens.size(), 13U);
}

TEST(Generate, SampledGenerationIsRepeatableWithASeed) {
  const Loaded loaded = Load();
  GenerateOptions options;
  options.max_new_tokens = 8;
  options.sampling.temperature = 1.0F;
  options.sampling.seed = 99;
  const auto a = ember::Generate(*loaded.model, std::vector<std::int32_t>{5}, options);
  const auto b = ember::Generate(*loaded.model, std::vector<std::int32_t>{5}, options);
  ASSERT_TRUE(a.ok() && b.ok());
  EXPECT_EQ(Ids(*a), Ids(*b));
}

TEST(Generate, RefusesPromptsItCannotRun) {
  const Loaded loaded = Load();
  GenerateOptions options;
  EXPECT_FALSE(ember::Generate(*loaded.model, std::vector<std::int32_t>{}, options).ok());
  EXPECT_FALSE(ember::Generate(*loaded.model, std::vector<std::int32_t>(16, 1), options).ok());
  EXPECT_FALSE(ember::Generate(*loaded.model, std::vector<std::int32_t>{11}, options).ok());
  options.sampling.top_p = 2.0F;
  EXPECT_FALSE(ember::Generate(*loaded.model, std::vector<std::int32_t>{1}, options).ok());
}

TEST(Generate, ZeroNewTokensGivesNothing) {
  const Loaded loaded = Load();
  GenerateOptions options;
  options.max_new_tokens = 0;
  const auto result = ember::Generate(*loaded.model, std::vector<std::int32_t>{1}, options);
  ASSERT_TRUE(result.ok());
  EXPECT_TRUE(result->tokens.empty());
}

}  // namespace
