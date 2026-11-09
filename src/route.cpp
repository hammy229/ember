#include "ember/route.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>

namespace ember {

StatusOr<Router::Signal> Router::ParseSignal(std::string_view name) {
  if (name == "max entropy") return Signal::kMaxEntropy;
  if (name == "lowest margin") return Signal::kLowestMargin;
  if (name == "mean entropy") return Signal::kMeanEntropy;
  if (name == "share of tokens with margin under 0.5") return Signal::kLowMarginShare;
  return Status::InvalidArgument("unknown routing signal \"" + std::string(name) + "\"");
}

bool Router::Update(const TokenStats& stats) {
  if (escalated()) return true;
  const std::size_t index = tokens_;
  ++tokens_;
  entropy_sum_ += stats.entropy;
  if (stats.margin < 0.5F) ++low_margin_;
  bool growing = false;
  switch (options_.signal) {
    case Signal::kMaxEntropy:
      score_ = index == 0 ? stats.entropy : std::max(score_, stats.entropy);
      growing = true;
      break;
    case Signal::kLowestMargin:
      score_ = index == 0 ? -stats.margin : std::max(score_, -stats.margin);
      growing = true;
      break;
    case Signal::kMeanEntropy:
      score_ = static_cast<float>(entropy_sum_ / static_cast<double>(tokens_));
      break;
    case Signal::kLowMarginShare:
      score_ = static_cast<float>(low_margin_) / static_cast<float>(tokens_);
      break;
  }
  if ((growing || tokens_ >= options_.min_tokens) && score_ >= options_.threshold) {
    escalated_at_ = index;
  }
  return escalated();
}

}  // namespace ember
