#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "ember/arena.h"
#include "ember/forward.h"
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

// Generation one token at a time, for callers that need control between tokens, like a web page
// that draws each token as it comes. Generate below is a loop over this.
class Generator {
 public:
  // Fails on an empty prompt, one that fills the whole context, or a bad sampling option.
  static StatusOr<std::unique_ptr<Generator>> Make(const LlamaModel& model,
                                                   std::span<const std::int32_t> prompt,
                                                   const GenerateOptions& options);

  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;

  // The next token, or nothing once generation has stopped; stop() then says why. The first call
  // runs the whole prompt, later ones one token each.
  StatusOr<std::optional<GeneratedToken>> Next();

  // Stops generation, as if the callback had said so.
  void Cancel();
  bool done() const { return done_; }
  GenerateResult::Stop stop() const { return stop_; }
  std::size_t generated() const { return generated_; }

 private:
  Generator(const LlamaModel& model, const GenerateOptions& options, Sampler sampler,
            std::unique_ptr<KvCache> cache, std::vector<std::int32_t> prompt);

  const LlamaModel& model_;
  GenerateOptions options_;
  Sampler sampler_;
  std::unique_ptr<KvCache> cache_;
  Arena arena_;
  std::vector<std::int32_t> pending_;  // tokens to feed on the next step: the prompt, then one
  std::size_t generated_ = 0;
  bool done_ = false;
  GenerateResult::Stop stop_ = GenerateResult::Stop::kMaxTokens;
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
