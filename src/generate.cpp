#include "ember/generate.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ember/arena.h"
#include "ember/forward.h"

namespace ember {

StatusOr<std::unique_ptr<Generator>> Generator::Make(const LlamaModel& model,
                                                     std::span<const std::int32_t> prompt,
                                                     const GenerateOptions& options) {
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
  // Written so a huge max_new_tokens cannot wrap the sum below the prompt size.
  const std::size_t capacity =
      prompt.size() + std::min(options.max_new_tokens, context - prompt.size());
  EMBER_ASSIGN_OR_RETURN(auto cache, KvCache::Make(model.config(), capacity));
  return std::unique_ptr<Generator>(
      new Generator(model, options, std::move(sampler), std::move(cache),
                    std::vector<std::int32_t>(prompt.begin(), prompt.end())));
}

Generator::Generator(const LlamaModel& model, const GenerateOptions& options, Sampler sampler,
                     std::unique_ptr<KvCache> cache, std::vector<std::int32_t> prompt)
    : model_(model),
      options_(options),
      sampler_(std::move(sampler)),
      cache_(std::move(cache)),
      pending_(std::move(prompt)) {}

void Generator::Cancel() {
  if (done_) return;
  done_ = true;
  stop_ = GenerateResult::Stop::kCallback;
}

StatusOr<std::optional<GeneratedToken>> Generator::Next() {
  if (done_) return std::optional<GeneratedToken>();
  if (generated_ == options_.max_new_tokens) {
    done_ = true;
    stop_ = GenerateResult::Stop::kMaxTokens;
    return std::optional<GeneratedToken>();
  }
  if (cache_->size() + pending_.size() > model_.config().max_positions) {
    done_ = true;
    stop_ = GenerateResult::Stop::kContextFull;
    return std::optional<GeneratedToken>();
  }
  arena_.Reset();
  EMBER_ASSIGN_OR_RETURN(const Tensor logits,
                         ForwardCached(model_, pending_, *cache_, arena_, true));
  const std::span<const float> row(logits.data<float>(), logits.dim(1));
  GeneratedToken token;
  EMBER_ASSIGN_OR_RETURN(token.stats, Uncertainty(row));
  EMBER_ASSIGN_OR_RETURN(token.id, sampler_.Sample(row));
  ++generated_;
  pending_.assign(1, token.id);
  if (std::find(options_.stop_tokens.begin(), options_.stop_tokens.end(), token.id) !=
      options_.stop_tokens.end()) {
    done_ = true;
    stop_ = GenerateResult::Stop::kStopToken;
  }
  return std::optional<GeneratedToken>(token);
}

StatusOr<GenerateResult> Generate(const LlamaModel& model, std::span<const std::int32_t> prompt,
                                  const GenerateOptions& options, const TokenCallback& on_token) {
  EMBER_ASSIGN_OR_RETURN(auto generator, Generator::Make(model, prompt, options));
  GenerateResult result;
  while (true) {
    EMBER_ASSIGN_OR_RETURN(const std::optional<GeneratedToken> token, generator->Next());
    if (!token) break;
    result.tokens.push_back(*token);
    if (on_token && !on_token(*token, result.tokens.size() - 1)) generator->Cancel();
  }
  result.stop = generator->stop();
  return result;
}

}  // namespace ember
