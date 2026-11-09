#include "ember/generate.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "ember/arena.h"
#include "ember/forward.h"

namespace ember {

StatusOr<GenerateResult> Generate(const LlamaModel& model, std::span<const std::int32_t> prompt,
                                  const GenerateOptions& options, const TokenCallback& on_token) {
  const std::size_t context = model.config().max_positions;
  if (prompt.empty()) {
    return Status::InvalidArgument("Generate needs a prompt of at least one token");
  }
  if (prompt.size() >= context) {
    return Status::InvalidArgument("a prompt of " + std::to_string(prompt.size()) +
                                   " tokens leaves no room in the model's context of " +
                                   std::to_string(context));
  }
  EMBER_ASSIGN_OR_RETURN(Sampler sampler, Sampler::Make(options.sampling));
  const std::size_t capacity = std::min(context, prompt.size() + options.max_new_tokens);
  EMBER_ASSIGN_OR_RETURN(auto cache, KvCache::Make(model.config(), capacity));

  GenerateResult result;
  Arena arena;
  std::span<const std::int32_t> input = prompt;
  std::int32_t last = -1;
  while (true) {
    if (result.tokens.size() == options.max_new_tokens) {
      result.stop = GenerateResult::Stop::kMaxTokens;
      break;
    }
    if (cache->size() + input.size() > context) {
      result.stop = GenerateResult::Stop::kContextFull;
      break;
    }
    arena.Reset();
    EMBER_ASSIGN_OR_RETURN(const Tensor logits, ForwardCached(model, input, *cache, arena, true));
    const std::span<const float> row(logits.data<float>(), logits.dim(1));
    GeneratedToken token;
    EMBER_ASSIGN_OR_RETURN(token.stats, Uncertainty(row));
    EMBER_ASSIGN_OR_RETURN(token.id, sampler.Sample(row));
    result.tokens.push_back(token);
    if (on_token && !on_token(token, result.tokens.size() - 1)) {
      result.stop = GenerateResult::Stop::kCallback;
      break;
    }
    if (options.stop_token >= 0 && token.id == options.stop_token) {
      result.stop = GenerateResult::Stop::kStopToken;
      break;
    }
    last = token.id;
    input = std::span<const std::int32_t>(&last, 1);
  }
  return result;
}

}  // namespace ember
