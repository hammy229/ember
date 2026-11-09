#include "ember/sampling.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using ember::Sampler;
using ember::SamplingOptions;
using ember::Uncertainty;

TEST(Uncertainty, UniformLogitsHaveMaximumEntropy) {
  const std::vector<float> logits(1000, 3.5F);
  const auto stats = Uncertainty(logits);
  ASSERT_TRUE(stats.ok());
  EXPECT_NEAR(stats->entropy, std::log(1000.0), 1e-5);
  EXPECT_NEAR(stats->top1_prob, 0.001, 1e-7);
  EXPECT_NEAR(stats->margin, 0.0, 1e-7);
  EXPECT_EQ(stats->top1_id, 0);  // lowest id on a tie
}

TEST(Uncertainty, ADominantLogitHasNearZeroEntropy) {
  std::vector<float> logits(50, 0.0F);
  logits[17] = 100.0F;
  const auto stats = Uncertainty(logits);
  ASSERT_TRUE(stats.ok());
  EXPECT_LT(stats->entropy, 1e-30);
  EXPECT_FLOAT_EQ(stats->top1_prob, 1.0F);
  EXPECT_EQ(stats->top1_id, 17);
  EXPECT_FLOAT_EQ(stats->margin, 1.0F);
}

TEST(Uncertainty, MatchesAHandComputedCase) {
  // Probabilities 0.5, 0.25, 0.25: entropy is 1.5 * ln 2.
  const std::vector<float> logits{std::log(2.0F), 0.0F, 0.0F};
  const auto stats = Uncertainty(logits);
  ASSERT_TRUE(stats.ok());
  EXPECT_NEAR(stats->entropy, 1.5 * std::log(2.0), 1e-6);
  EXPECT_NEAR(stats->top1_prob, 0.5, 1e-6);
  EXPECT_NEAR(stats->top2_prob, 0.25, 1e-6);
  EXPECT_NEAR(stats->margin, 0.25, 1e-6);
}

TEST(Uncertainty, IgnoresAShiftOfEveryLogit) {
  const std::vector<float> a{1.0F, 2.0F, -3.0F, 0.5F};
  const std::vector<float> b{1001.0F, 1002.0F, 997.0F, 1000.5F};
  const auto sa = Uncertainty(a);
  const auto sb = Uncertainty(b);
  ASSERT_TRUE(sa.ok() && sb.ok());
  EXPECT_NEAR(sa->entropy, sb->entropy, 1e-5);
  EXPECT_NEAR(sa->margin, sb->margin, 1e-6);
}

TEST(Uncertainty, HandlesAVocabularyOfOne) {
  const auto stats = Uncertainty(std::vector<float>{2.0F});
  ASSERT_TRUE(stats.ok());
  EXPECT_FLOAT_EQ(stats->top1_prob, 1.0F);
  EXPECT_FLOAT_EQ(stats->top2_prob, 0.0F);
}

TEST(Uncertainty, RefusesRowsItCannotScore) {
  EXPECT_FALSE(Uncertainty(std::vector<float>{}).ok());
  EXPECT_FALSE(Uncertainty(std::vector<float>{1.0F, std::nanf("")}).ok());
  EXPECT_FALSE(Uncertainty(std::vector<float>{std::numeric_limits<float>::infinity()}).ok());
}

TEST(Sampler, GreedyPicksTheLargestLogitAndTheLowestIdOnATie) {
  auto sampler = Sampler::Make({});
  ASSERT_TRUE(sampler.ok());
  EXPECT_EQ(*sampler->Sample(std::vector<float>{0.1F, 5.0F, 2.0F}), 1);
  EXPECT_EQ(*sampler->Sample(std::vector<float>{3.0F, 7.0F, 7.0F}), 1);
}

TEST(Sampler, TopKOfOneAndATinyTopPAreGreedy) {
  const std::vector<float> logits{0.3F, -1.0F, 2.5F, 2.4F, 0.0F};
  SamplingOptions k1;
  k1.temperature = 1.5F;
  k1.top_k = 1;
  SamplingOptions p0;
  p0.temperature = 1.5F;
  p0.top_p = 0.0F;
  for (const SamplingOptions& options : {k1, p0}) {
    auto sampler = Sampler::Make(options);
    ASSERT_TRUE(sampler.ok());
    for (int i = 0; i < 50; ++i) EXPECT_EQ(*sampler->Sample(logits), 2);
  }
}

// Draws many samples and gives how often each token came up.
std::vector<double> Frequencies(const SamplingOptions& options, const std::vector<float>& logits,
                                int draws) {
  auto sampler = Sampler::Make(options);
  EXPECT_TRUE(sampler.ok());
  std::vector<double> counts(logits.size(), 0);
  for (int i = 0; i < draws; ++i) counts[static_cast<std::size_t>(*sampler->Sample(logits))] += 1;
  for (double& c : counts) c /= draws;
  return counts;
}

TEST(Sampler, SamplesInProportionToTheProbabilities) {
  // Probabilities 0.5, 0.25, 0.125, 0.125.
  const std::vector<float> logits{std::log(4.0F), std::log(2.0F), 0.0F, 0.0F};
  SamplingOptions options;
  options.temperature = 1.0F;
  options.seed = 7;
  const std::vector<double> f = Frequencies(options, logits, 200000);
  EXPECT_NEAR(f[0], 0.5, 0.005);
  EXPECT_NEAR(f[1], 0.25, 0.005);
  EXPECT_NEAR(f[2], 0.125, 0.005);
  EXPECT_NEAR(f[3], 0.125, 0.005);
}

TEST(Sampler, TemperatureSharpensAndFlattens) {
  const std::vector<float> logits{1.0F, 0.0F};
  SamplingOptions cold;
  cold.temperature = 0.25F;
  SamplingOptions hot;
  hot.temperature = 4.0F;
  // p0 = 1 / (1 + e^(-1/T)).
  EXPECT_NEAR(Frequencies(cold, logits, 100000)[0], 1.0 / (1.0 + std::exp(-4.0)), 0.005);
  EXPECT_NEAR(Frequencies(hot, logits, 100000)[0], 1.0 / (1.0 + std::exp(-0.25)), 0.005);
}

TEST(Sampler, TopKAndTopPOnlyKeepTheMostLikely) {
  // Probabilities 0.4, 0.3, 0.2, 0.1.
  const std::vector<float> logits{std::log(4.0F), std::log(3.0F), std::log(2.0F), 0.0F};
  SamplingOptions k2;
  k2.temperature = 1.0F;
  k2.top_k = 2;
  const std::vector<double> fk = Frequencies(k2, logits, 100000);
  EXPECT_NEAR(fk[0], 4.0 / 7.0, 0.006);
  EXPECT_EQ(fk[2] + fk[3], 0.0);

  // 0.4 + 0.3 = 0.7 reaches top_p 0.65, so only the first two remain.
  SamplingOptions p;
  p.temperature = 1.0F;
  p.top_p = 0.65F;
  const std::vector<double> fp = Frequencies(p, logits, 100000);
  EXPECT_NEAR(fp[1], 3.0 / 7.0, 0.006);
  EXPECT_EQ(fp[2] + fp[3], 0.0);
}

TEST(Sampler, TheSameSeedGivesTheSameTokens) {
  std::vector<float> logits(300);
  for (std::size_t i = 0; i < logits.size(); ++i) logits[i] = std::sin(static_cast<float>(i));
  SamplingOptions options;
  options.temperature = 0.9F;
  options.top_p = 0.95F;
  options.seed = 42;
  auto a = Sampler::Make(options);
  auto b = Sampler::Make(options);
  options.seed = 43;
  auto c = Sampler::Make(options);
  ASSERT_TRUE(a.ok() && b.ok() && c.ok());
  int differences = 0;
  for (int i = 0; i < 200; ++i) {
    const std::int32_t x = *a->Sample(logits);
    EXPECT_EQ(x, *b->Sample(logits));
    differences += x != *c->Sample(logits) ? 1 : 0;
  }
  EXPECT_GT(differences, 0);
}

TEST(Sampler, RefusesBadOptionsAndRows) {
  SamplingOptions bad_p;
  bad_p.top_p = 1.5F;
  EXPECT_FALSE(Sampler::Make(bad_p).ok());
  SamplingOptions nan_t;
  nan_t.temperature = std::nanf("");
  EXPECT_FALSE(Sampler::Make(nan_t).ok());
  auto sampler = Sampler::Make({});
  ASSERT_TRUE(sampler.ok());
  EXPECT_FALSE(sampler->Sample(std::vector<float>{}).ok());
  EXPECT_FALSE(sampler->Sample(std::vector<float>{std::nanf("")}).ok());
}

}  // namespace
