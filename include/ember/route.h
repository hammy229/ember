#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

#include "ember/sampling.h"
#include "ember/status_or.h"

namespace ember {

// Confidence-gated routing: answer on the local model, and hand the request to a bigger model the
// moment the local one shows it is unsure. Each new token's TokenStats updates a score; once the
// score reaches the threshold, the router says to escalate.
//
// The signals are the ones scripts/route_eval.py measures on GSM8K. Two of them only ever grow as
// an answer goes on (the highest entropy so far, and the lowest margin so far, counted as minus the
// margin), so stopping as soon as they cross the threshold makes exactly the decision the full
// answer would have made, only earlier. The averages can rise and fall, so they wait for
// min_tokens before deciding.
class Router {
 public:
  enum class Signal { kMaxEntropy, kLowestMargin, kMeanEntropy, kLowMarginShare };

  struct Options {
    Signal signal = Signal::kMaxEntropy;
    float threshold = 0;
    // Averages are not trusted before this many tokens. The two growing signals ignore it.
    std::size_t min_tokens = 16;
  };

  // The names used in scripts/route_eval.py's results: "max entropy", "lowest margin",
  // "mean entropy", "share of tokens with margin under 0.5".
  static StatusOr<Signal> ParseSignal(std::string_view name);

  explicit Router(const Options& options) : options_(options) {}

  // Takes the next token's stats. True once the router decides to escalate, and from then on.
  bool Update(const TokenStats& stats);

  // The score so far, in the signal's own units. Zero before any token.
  float score() const { return score_; }
  std::size_t tokens() const { return tokens_; }
  bool escalated() const { return escalated_at_.has_value(); }
  // Which token (counting from 0) tipped it over, if any.
  std::optional<std::size_t> escalated_at() const { return escalated_at_; }

 private:
  Options options_;
  std::size_t tokens_ = 0;
  double entropy_sum_ = 0;
  std::size_t low_margin_ = 0;
  float score_ = 0;
  std::optional<std::size_t> escalated_at_;
};

}  // namespace ember
