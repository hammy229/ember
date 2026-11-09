#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "ember/status_or.h"

namespace ember {

// How sure the model is about the next token, read from one row of logits at temperature 1,
// whatever the sampler then does. This is the signal the router (docs/PLAN.md) uses to decide when
// a local model is out of its depth.
struct TokenStats {
  // Entropy of the next-token distribution in nats: 0 when one token has all the probability,
  // log(vocab) when every token is equally likely.
  float entropy = 0;
  // The two highest probabilities, and their difference. A small margin means the model is torn
  // between two tokens even if the entropy is low.
  float top1_prob = 0;
  float top2_prob = 0;
  float margin = 0;
  std::int32_t top1_id = -1;
};

// Fails on an empty row or one with NaN or infinite logits.
StatusOr<TokenStats> Uncertainty(std::span<const float> logits);

struct SamplingOptions {
  // Zero or less means greedy: always the most likely token, lowest id on a tie.
  float temperature = 0;
  // Keep only the k most likely tokens. Zero keeps all.
  std::size_t top_k = 0;
  // Keep the fewest most likely tokens whose probabilities add up to at least top_p. One keeps all.
  float top_p = 1;
  std::uint64_t seed = 0;
};

// Picks next tokens from logits. The same seed and logits give the same tokens on every platform:
// the random numbers come from std::mt19937_64, which the standard pins down exactly, turned into
// doubles by hand rather than by std::uniform_real_distribution, which it does not.
class Sampler {
 public:
  // Fails on a negative top_p or one above 1, or a temperature that is NaN.
  static StatusOr<Sampler> Make(const SamplingOptions& options);

  // Fails on an empty row or one with NaN or infinite logits.
  StatusOr<std::int32_t> Sample(std::span<const float> logits);

 private:
  explicit Sampler(const SamplingOptions& options);
  double Uniform();

  SamplingOptions options_;
  std::mt19937_64 rng_;
  std::vector<std::size_t> order_;  // reused between calls
  std::vector<double> probs_;
};

}  // namespace ember
