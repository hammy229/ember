#include "ember/threads.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "ember/arena.h"
#include "ember/forward.h"
#include "ember/layers.h"
#include "ember/llama.h"
#include "ember/tensor.h"
#include "tiny_llama.h"

namespace {

// Puts the thread count back to 1 when a test ends, so other tests run as before.
struct ThreadsGuard {
  ~ThreadsGuard() { ember::SetThreadCount(1); }
};

TEST(Threads, CoversEveryIndexExactlyOnce) {
  const ThreadsGuard guard;
  for (const std::size_t threads : {1U, 2U, 3U, 8U}) {
    ember::SetThreadCount(threads);
    EXPECT_EQ(ember::ThreadCount(), threads);
    for (const std::size_t count : {1U, 7U, 64U, 1000U, 4097U}) {
      std::vector<std::atomic<int>> hits(count);
      ember::ParallelFor(count, 1, [&](std::size_t begin, std::size_t end) {
        ASSERT_LT(begin, end);
        for (std::size_t i = begin; i < end; ++i) hits[i].fetch_add(1);
      });
      for (std::size_t i = 0; i < count; ++i) ASSERT_EQ(hits[i].load(), 1) << i;
    }
  }
}

TEST(Threads, RespectsTheGrainAndSkipsEmptyLoops) {
  const ThreadsGuard guard;
  ember::SetThreadCount(4);
  std::atomic<int> calls{0};
  ember::ParallelFor(10, 100, [&](std::size_t begin, std::size_t end) {
    EXPECT_EQ(begin, 0U);
    EXPECT_EQ(end, 10U);
    calls.fetch_add(1);
  });
  EXPECT_EQ(calls.load(), 1);
  ember::ParallelFor(0, 1, [&](std::size_t, std::size_t) { calls.fetch_add(1); });
  EXPECT_EQ(calls.load(), 1);
}

TEST(Threads, NestedCallsRunInline) {
  const ThreadsGuard guard;
  ember::SetThreadCount(4);
  std::atomic<int> total{0};
  ember::ParallelFor(8, 1, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      ember::ParallelFor(
          5, 1, [&](std::size_t b, std::size_t e) { total.fetch_add(static_cast<int>(e - b)); });
    }
  });
  EXPECT_EQ(total.load(), 40);
}

TEST(Threads, ManyQuickJobsInARow) {
  const ThreadsGuard guard;
  ember::SetThreadCount(6);
  std::atomic<std::size_t> sum{0};
  for (int job = 0; job < 2000; ++job) {
    ember::ParallelFor(97, 1, [&](std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; ++i) sum.fetch_add(i);
    });
  }
  EXPECT_EQ(sum.load(), 2000U * (96U * 97U / 2U));
}

TEST(Threads, LinearIsBitIdenticalForAnyThreadCount) {
  const ThreadsGuard guard;
  ember::Arena arena;
  std::mt19937 rng(3);
  std::normal_distribution<float> normal;
  const auto make = [&](std::size_t r, std::size_t c) {
    std::vector<float> values(r * c);
    for (float& v : values) v = normal(rng);
    return ember::Tensor::FromValues<float>(arena, *ember::Shape::Make({r, c}), values).value();
  };
  const ember::Tensor x = make(3, 333);
  const ember::Tensor w = make(1000, 333);
  std::vector<float> first;
  for (const std::size_t threads : {1U, 2U, 5U, 8U}) {
    ember::SetThreadCount(threads);
    const ember::Tensor out =
        ember::Tensor::Uninitialized(arena, ember::DType::kF32, *ember::Shape::Make({3, 1000}))
            .value();
    ASSERT_TRUE(ember::Linear(x, w, out).ok());
    std::vector<float> now(out.data<float>(), out.data<float>() + out.element_count());
    if (first.empty()) {
      first = now;
    } else {
      ASSERT_EQ(std::memcmp(first.data(), now.data(), first.size() * sizeof(float)), 0)
          << threads << " threads";
    }
  }
}

TEST(Threads, TheWholeModelIsBitIdenticalForAnyThreadCount) {
  const ThreadsGuard guard;
  ember::testing::TinyLlamaOptions options;
  options.hidden = 64;
  options.intermediate = 160;
  options.heads = 4;
  options.kv_heads = 2;
  options.vocab = 300;
  options.max_positions = 32;
  const std::vector<std::byte> bytes = ember::testing::TinyLlamaWeights(options);
  const auto model = ember::LlamaModel::FromBytes(ember::testing::TinyLlamaConfig(options), bytes);
  ASSERT_TRUE(model.ok()) << model.status().ToString();
  const std::vector<std::int32_t> tokens{5, 17, 200, 3, 3, 99, 250, 1};
  std::vector<float> first;
  for (const std::size_t threads : {1U, 3U, 8U}) {
    ember::SetThreadCount(threads);
    ember::Arena arena;
    const auto logits = ember::Forward(**model, tokens, arena);
    ASSERT_TRUE(logits.ok());
    std::vector<float> now(logits->data<float>(), logits->data<float>() + logits->element_count());
    if (first.empty()) {
      first = now;
    } else {
      ASSERT_EQ(std::memcmp(first.data(), now.data(), first.size() * sizeof(float)), 0)
          << threads << " threads";
    }
  }
}

}  // namespace
