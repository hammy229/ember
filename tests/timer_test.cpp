#include "ember/timer.h"

#include <gtest/gtest.h>

#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

TEST(Stopwatch, ElapsedIsNonNegativeAndGrows) {
  ember::Stopwatch sw;
  double first = sw.ElapsedMs();
  EXPECT_GE(first, 0.0);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_GE(sw.ElapsedMs(), first + 4.0);
}

TEST(Stopwatch, ResetStartsOver) {
  ember::Stopwatch sw;
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ember::Stopwatch started_before_reset;
  sw.Reset();
  const double since_reset = sw.ElapsedMs();
  EXPECT_LE(since_reset, started_before_reset.ElapsedMs());
}

TEST(Summarize, OddCount) {
  auto s = ember::Summarize({3.0, 1.0, 2.0});
  EXPECT_EQ(s.count, 3U);
  EXPECT_DOUBLE_EQ(s.min_ms, 1.0);
  EXPECT_DOUBLE_EQ(s.max_ms, 3.0);
  EXPECT_DOUBLE_EQ(s.mean_ms, 2.0);
  EXPECT_DOUBLE_EQ(s.median_ms, 2.0);
}

TEST(Summarize, EvenCountMedianIsMiddlePairAverage) {
  auto s = ember::Summarize({4.0, 1.0, 2.0, 10.0});
  EXPECT_DOUBLE_EQ(s.median_ms, 3.0);
  EXPECT_DOUBLE_EQ(s.mean_ms, 4.25);
}

TEST(Summarize, SingleSample) {
  auto s = ember::Summarize({7.5});
  EXPECT_EQ(s.count, 1U);
  EXPECT_DOUBLE_EQ(s.min_ms, 7.5);
  EXPECT_DOUBLE_EQ(s.max_ms, 7.5);
  EXPECT_DOUBLE_EQ(s.median_ms, 7.5);
}

TEST(Summarize, EmptyGivesZeros) {
  auto s = ember::Summarize({});
  EXPECT_EQ(s.count, 0U);
  EXPECT_DOUBLE_EQ(s.mean_ms, 0.0);
  EXPECT_DOUBLE_EQ(s.median_ms, 0.0);
}

TEST(Summarize, DoesNotNeedSortedInput) {
  std::vector<double> samples = {9.0, 1.0, 5.0, 3.0, 7.0};
  EXPECT_DOUBLE_EQ(ember::Summarize(samples).median_ms, 5.0);
}

TEST(Repeat, RunsTheFunctionWarmupPlusRepeatsTimes) {
  int calls = 0;
  auto s = ember::Repeat([&] { ++calls; }, 5, 2);
  EXPECT_EQ(calls, 7);
  EXPECT_EQ(s.count, 5U);
}

TEST(Repeat, ZeroRepeatsGivesEmptySummary) {
  int calls = 0;
  auto s = ember::Repeat([&] { ++calls; }, 0, 0);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(s.count, 0U);
}

TEST(Repeat, MeasuresTheWorkInsideTheFunction) {
  auto s = ember::Repeat([] { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }, 3, 0);
  EXPECT_GE(s.min_ms, 4.0);
  EXPECT_LE(s.min_ms, s.median_ms);
  EXPECT_LE(s.median_ms, s.max_ms);
}

TEST(Repeat, PropagatesExceptions) {
  EXPECT_THROW(ember::Repeat([] { throw std::runtime_error("boom"); }, 1, 0), std::runtime_error);
}
