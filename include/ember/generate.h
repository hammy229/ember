#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "ember/llama.h"
#include "ember/sampling.h"
#include "ember/status_or.h"

namespace ember {

struct GenerateOptions {
  std::size_t max_new_tokens = 64;
  SamplingOptions sampling;
  // Stop when this token comes out (it is still reported). Negative never stops.
  std::int32_t stop_token = -1;
};

struct GeneratedToken {
  std::int32_t id = -1;
  // How sure the model was when it picked this token, from the logits it was picked from.
  TokenStats stats;
};

// Called after each new token with everything generated so far. Returning false stops generation
// there, which is how the router hands a request off as soon as the model gets unsure.
using TokenCallback = std::function<bool(const GeneratedToken& token, std::size_t index)>;

struct GenerateResult {
  std::vector<GeneratedToken> tokens;
  enum class Stop : std::uint8_t { kMaxTokens, kStopToken, kContextFull, kCallback };
  Stop stop = Stop::kMaxTokens;
};

// Runs the prompt through the model once, then picks one token at a time from the logits of the
// last, reusing a KV cache so each step costs one row. Stops at max_new_tokens, the stop token, the
// end of the model's context, or when the callback says so.
//
// Fails on an empty prompt, one that fills the whole context, or a bad sampling option.
StatusOr<GenerateResult> Generate(const LlamaModel& model, std::span<const std::int32_t> prompt,
                                  const GenerateOptions& options,
                                  const TokenCallback& on_token = nullptr);

}  // namespace ember
