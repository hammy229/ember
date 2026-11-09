#include "ember/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>

namespace ember {

namespace {

Status CheckRow(std::span<const float> logits) {
  if (logits.empty()) return Status::InvalidArgument("an empty row of logits");
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i])) {
      return Status::InvalidArgument("logit " + std::to_string(i) + " is not a finite number");
    }
  }
  return Status::Ok();
}

std::int32_t ArgMax(std::span<const float> logits) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<std::int32_t>(best);
}

}  // namespace

StatusOr<TokenStats> Uncertainty(std::span<const float> logits) {
  EMBER_RETURN_IF_ERROR(CheckRow(logits));
  // Softmax in double, shifted by the largest logit so exp cannot overflow.
  const double top = *std::max_element(logits.begin(), logits.end());
  double sum = 0;
  for (const float logit : logits) sum += std::exp(static_cast<double>(logit) - top);
  const double log_sum = std::log(sum);
  double entropy = 0;
  double first = -1;
  double second = -1;
  std::size_t first_id = 0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const double log_p = static_cast<double>(logits[i]) - top - log_sum;
    const double p = std::exp(log_p);
    if (p > 0) entropy -= p * log_p;
    if (p > first) {
      second = first;
      first = p;
      first_id = i;
    } else if (p > second) {
      second = p;
    }
  }
  if (second < 0) second = 0;  // a vocabulary of one
  TokenStats stats;
  stats.entropy = static_cast<float>(std::max(entropy, 0.0));
  stats.top1_prob = static_cast<float>(first);
  stats.top2_prob = static_cast<float>(second);
  stats.margin = static_cast<float>(first - second);
  stats.top1_id = static_cast<std::int32_t>(first_id);
  return stats;
}

StatusOr<Sampler> Sampler::Make(const SamplingOptions& options) {
  if (std::isnan(options.temperature)) return Status::InvalidArgument("temperature is NaN");
  if (!(options.top_p >= 0 && options.top_p <= 1)) {
    return Status::InvalidArgument("top_p must be between 0 and 1");
  }
  return Sampler(options);
}

Sampler::Sampler(const SamplingOptions& options) : options_(options), rng_(options.seed) {}

double Sampler::Uniform() {
  // The top 53 bits of one 64-bit draw, as a double in [0, 1).
  return static_cast<double>(rng_() >> 11) * 0x1.0p-53;
}

StatusOr<std::int32_t> Sampler::Sample(std::span<const float> logits) {
  EMBER_RETURN_IF_ERROR(CheckRow(logits));
  if (options_.temperature <= 0) return ArgMax(logits);

  // Most likely first, lowest id first among equals, so the order never depends on the sort.
  order_.resize(logits.size());
  std::iota(order_.begin(), order_.end(), std::size_t{0});
  const auto more_likely = [&](std::size_t a, std::size_t b) {
    return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
  };
  std::size_t keep = logits.size();
  if (options_.top_k > 0 && options_.top_k < keep) {
    keep = options_.top_k;
    std::partial_sort(order_.begin(), order_.begin() + static_cast<std::ptrdiff_t>(keep),
                      order_.end(), more_likely);
  } else {
    std::sort(order_.begin(), order_.end(), more_likely);
  }

  const double inverse_t = 1.0 / static_cast<double>(options_.temperature);
  const double top = static_cast<double>(logits[order_[0]]);
  probs_.resize(keep);
  double sum = 0;
  for (std::size_t i = 0; i < keep; ++i) {
    probs_[i] = std::exp((static_cast<double>(logits[order_[i]]) - top) * inverse_t);
    sum += probs_[i];
  }
  if (options_.top_p < 1) {
    // The fewest tokens whose probabilities reach top_p, and always at least one.
    double running = 0;
    std::size_t cut = 0;
    while (cut < keep) {
      running += probs_[cut] / sum;
      ++cut;
      if (running >= static_cast<double>(options_.top_p)) break;
    }
    keep = cut;
    sum = std::accumulate(probs_.begin(), probs_.begin() + static_cast<std::ptrdiff_t>(keep), 0.0);
  }

  const double target = Uniform() * sum;
  double running = 0;
  for (std::size_t i = 0; i < keep; ++i) {
    running += probs_[i];
    if (target < running) return static_cast<std::int32_t>(order_[i]);
  }
  return static_cast<std::int32_t>(order_[keep - 1]);  // rounding left target at the very end
}

}  // namespace ember
