#include "ember/route.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

namespace {

using ember::Router;
using ember::TokenStats;

TokenStats Stats(float entropy, float margin) {
  TokenStats s;
  s.entropy = entropy;
  s.margin = margin;
  return s;
}

TEST(Router, EscalatesAtTheFirstTokenOverTheThreshold) {
  Router router({Router::Signal::kMaxEntropy, 1.0F, 16});
  EXPECT_FALSE(router.Update(Stats(0.2F, 0.9F)));
  EXPECT_FALSE(router.Update(Stats(0.9F, 0.5F)));
  EXPECT_TRUE(router.Update(Stats(1.4F, 0.1F)));
  EXPECT_EQ(router.escalated_at(), 2U);
  // Once escalated it stays escalated, whatever comes next.
  EXPECT_TRUE(router.Update(Stats(0.0F, 1.0F)));
  EXPECT_EQ(router.escalated_at(), 2U);
  EXPECT_FLOAT_EQ(router.score(), 1.4F);
}

TEST(Router, LowestMarginCountsAsMinusTheMargin) {
  Router router({Router::Signal::kLowestMargin, -0.2F, 16});
  EXPECT_FALSE(router.Update(Stats(0.0F, 0.9F)));
  EXPECT_FLOAT_EQ(router.score(), -0.9F);
  EXPECT_FALSE(router.Update(Stats(0.0F, 0.3F)));
  EXPECT_TRUE(router.Update(Stats(0.0F, 0.15F)));
  EXPECT_EQ(router.escalated_at(), 2U);
}

TEST(Router, AveragesWaitForEnoughTokens) {
  Router router({Router::Signal::kMeanEntropy, 1.0F, 4});
  EXPECT_FALSE(router.Update(Stats(3.0F, 0.0F)));  // mean 3, but only one token
  EXPECT_FALSE(router.Update(Stats(3.0F, 0.0F)));
  EXPECT_FALSE(router.Update(Stats(3.0F, 0.0F)));
  EXPECT_TRUE(router.Update(Stats(3.0F, 0.0F)));
  EXPECT_EQ(router.escalated_at(), 3U);

  Router share({Router::Signal::kLowMarginShare, 0.5F, 2});
  EXPECT_FALSE(share.Update(Stats(0.0F, 0.1F)));
  EXPECT_TRUE(share.Update(Stats(0.0F, 0.9F)));  // one of two under 0.5
  EXPECT_FLOAT_EQ(share.score(), 0.5F);
}

TEST(Router, NeverEscalatesOnASureAnswer) {
  Router router({Router::Signal::kMaxEntropy, 2.0F, 16});
  for (int i = 0; i < 500; ++i) EXPECT_FALSE(router.Update(Stats(0.3F, 0.8F)));
  EXPECT_FALSE(router.escalated());
  EXPECT_EQ(router.tokens(), 500U);
}

// For the growing signals, stopping early decides exactly what the whole answer would have.
TEST(Router, GrowingSignalsMatchTheDecisionOnTheFullAnswer) {
  std::mt19937 rng(9);
  std::exponential_distribution<float> entropy(1.5F);
  std::uniform_real_distribution<float> margin(0.0F, 1.0F);
  for (int trial = 0; trial < 2000; ++trial) {
    std::vector<TokenStats> answer(static_cast<std::size_t>(1 + trial % 60));
    for (TokenStats& s : answer) s = Stats(entropy(rng), margin(rng));
    float max_entropy = 0;
    float lowest_margin = 1;
    for (const TokenStats& s : answer) {
      max_entropy = std::max(max_entropy, s.entropy);
      lowest_margin = std::min(lowest_margin, s.margin);
    }
    Router by_entropy({Router::Signal::kMaxEntropy, 2.0F, 16});
    Router by_margin({Router::Signal::kLowestMargin, -0.05F, 16});
    for (const TokenStats& s : answer) {
      by_entropy.Update(s);
      by_margin.Update(s);
    }
    ASSERT_EQ(by_entropy.escalated(), max_entropy >= 2.0F) << trial;
    ASSERT_EQ(by_margin.escalated(), -lowest_margin >= -0.05F) << trial;
  }
}

TEST(Router, ParsesTheEvaluationsSignalNames) {
  EXPECT_EQ(*Router::ParseSignal("max entropy"), Router::Signal::kMaxEntropy);
  EXPECT_EQ(*Router::ParseSignal("lowest margin"), Router::Signal::kLowestMargin);
  EXPECT_EQ(*Router::ParseSignal("mean entropy"), Router::Signal::kMeanEntropy);
  EXPECT_EQ(*Router::ParseSignal("share of tokens with margin under 0.5"),
            Router::Signal::kLowMarginShare);
  EXPECT_FALSE(Router::ParseSignal("vibes").ok());
}

}  // namespace
