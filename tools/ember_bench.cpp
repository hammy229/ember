// Tiny benchmark harness. It times a placeholder workload (summing a float
// array) and the arena against the system allocator. Real kernels get added
// here as they exist. It prints only what it measures on this machine.
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "ember/aligned_buffer.h"
#include "ember/arena.h"
#include "ember/timer.h"

namespace {

// 1 GiB of floats. Anything bigger is a typo, not a benchmark.
constexpr std::size_t kMaxElements = std::size_t{1} << 28;

volatile float g_sink = 0.0F;
volatile std::byte g_byte_sink{};

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

void Print(const std::string& name, const ember::TimingSummary& s) {
  std::cout << "bench: " << name << ", " << s.count << " runs\n";
  std::cout << "  min    " << s.min_ms << " ms\n";
  std::cout << "  median " << s.median_ms << " ms\n";
  std::cout << "  mean   " << s.mean_ms << " ms\n";
  std::cout << "  max    " << s.max_ms << " ms\n";
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
  // Change the input each run. If it never changed, the compiler could sum once and reuse the
  // answer, and the benchmark would time nothing.
  std::size_t run = 0;
  const ember::TimingSummary s = ember::Repeat(
      [&] {
        data[0] = static_cast<float>(run++ % 2);
        g_sink = SumAll(data);
      },
      repeats, 2);

  std::cout << std::fixed << std::setprecision(3);
  Print("sum of " + std::to_string(size) + " floats", s);

  // Allocate the same batch of scratch blocks both ways. The arena has already been through one
  // round, as it has after the first token of a generation, so this times reuse and not setup.
  constexpr std::size_t kBlocks = 1000;
  constexpr std::size_t kBlockBytes = 4096;
  const std::string batch =
      std::to_string(kBlocks) + " blocks of " + std::to_string(kBlockBytes) + " bytes";

  ember::Arena arena;
  const ember::Arena::Mark start = arena.mark();
  const ember::TimingSummary arena_timing = ember::Repeat(
      [&] {
        for (std::size_t i = 0; i < kBlocks; ++i) {
          const auto block = arena.Allocate(kBlockBytes);
          if (!block.ok()) std::abort();
          // Touch each block so the work cannot be optimized away.
          **block = std::byte{1};
          g_byte_sink = **block;
        }
        arena.Rewind(start);
      },
      repeats, 2);
  Print("arena, " + batch, arena_timing);

  std::vector<ember::AlignedBuffer> held;
  held.reserve(kBlocks);
  const ember::TimingSummary system_timing = ember::Repeat(
      [&] {
        for (std::size_t i = 0; i < kBlocks; ++i) {
          auto block = ember::AlignedBuffer::Allocate(kBlockBytes, ember::Arena::kDefaultAlignment);
          if (!block.ok()) std::abort();
          *block->data() = std::byte{1};
          g_byte_sink = *block->data();
          held.push_back(std::move(*block));
        }
        held.clear();
      },
      repeats, 2);
  Print("system allocator, " + batch, system_timing);
  return 0;
}
