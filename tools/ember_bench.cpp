// Tiny benchmark harness. It times a placeholder workload (summing a float
// array) to exercise the repeat-and-report loop. Real kernels get added here
// as they exist. It prints only what it measures on this machine.
#include <charconv>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <system_error>
#include <vector>

#include "ember/timer.h"

namespace {

// 1 GiB of floats. Anything bigger is a typo, not a benchmark.
constexpr std::size_t kMaxElements = std::size_t{1} << 28;

volatile float g_sink = 0.0F;

float SumAll(const std::vector<float>& v) {
  float total = 0.0F;
  for (float x : v) total += x;
  return total;
}

// Parses a whole argument as a count from 1 to max. Anything else gives nullopt.
std::optional<std::size_t> ParseCount(std::string_view text, std::size_t max) {
  std::size_t value = 0;
  const char* end = text.data() + text.size();
  const auto [parsed_to, error] = std::from_chars(text.data(), end, value);
  if (error != std::errc() || parsed_to != end || value == 0 || value > max) return std::nullopt;
  return value;
}

int Usage() {
  std::cerr << "usage: ember_bench [repeats, 1 or more] [elements, 1 to " << kMaxElements << "]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t repeats = 20;
  std::size_t size = 1U << 22;
  if (argc > 3) return Usage();
  if (argc > 1) {
    const auto parsed = ParseCount(argv[1], std::numeric_limits<std::size_t>::max());
    if (!parsed) return Usage();
    repeats = *parsed;
  }
  if (argc > 2) {
    const auto parsed = ParseCount(argv[2], kMaxElements);
    if (!parsed) return Usage();
    size = *parsed;
  }

  std::vector<float> data(size, 1.0F);
  const ember::TimingSummary s = ember::Repeat([&] { g_sink = SumAll(data); }, repeats, 2);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "bench: sum of " << size << " floats, " << s.count << " runs\n";
  std::cout << "  min    " << s.min_ms << " ms\n";
  std::cout << "  median " << s.median_ms << " ms\n";
  std::cout << "  mean   " << s.mean_ms << " ms\n";
  std::cout << "  max    " << s.max_ms << " ms\n";
  return 0;
}
