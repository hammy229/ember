#pragma once

#include <charconv>
#include <cstddef>
#include <optional>
#include <string_view>
#include <system_error>

namespace ember_tools {

// Parses a whole argument as a count from 1 to max. Anything else gives nullopt.
inline std::optional<std::size_t> ParseCount(std::string_view text, std::size_t max) {
  std::size_t value = 0;
  const char* end = text.data() + text.size();
  const auto [parsed_to, error] = std::from_chars(text.data(), end, value);
  if (error != std::errc() || parsed_to != end || value == 0 || value > max) return std::nullopt;
  return value;
}

}  // namespace ember_tools
