// Tiny benchmark harness. It times a placeholder workload (summing a float
// array) and the arena against the system allocator. Real kernels get added
// here as they exist. It prints only what it measures on this machine.
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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
// Each run's time is stored, so a huge count would only run out of memory.
constexpr std::size_t kMaxRepeats = 1000000;

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
  std::cerr << "usage: ember_bench [repeats, 1 to " << kMaxRepeats << "] [elements, 1 to "
            << kMaxElements << "]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t repeats = 20;
  std::size_t size = 1U << 22;
  if (argc > 3) return Usage();
  if (argc > 1) {
    const auto parsed = ParseCount(argv[1], kMaxRepeats);
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

  // Allocate the same scratch blocks both ways, then free them all, like one forward pass does. The
  // sizes are mixed because identical blocks are the system allocator's best case, and a real model
  // does not ask for those. Rounds keep each timing far above the clock's resolution. The first
  // runs are warmup, so the arena is timing reuse and not setup, as it is after the first generated
  // token.
  constexpr std::array<std::size_t, 6> kBlockBytes{64, 256, 1024, 4096, 16384, 65536};
  constexpr std::size_t kBlocksPerRound = 1000;
  constexpr std::size_t kRounds = 100;
  const auto block_bytes = [&](std::size_t i) { return kBlockBytes[i % kBlockBytes.size()]; };

  ember::Arena arena;
  const ember::Arena::Mark start = arena.mark();
  const ember::TimingSummary arena_timing = ember::Repeat(
      [&] {
        for (std::size_t round = 0; round < kRounds; ++round) {
          for (std::size_t i = 0; i < kBlocksPerRound; ++i) {
            const auto block = arena.Allocate(block_bytes(i));
            if (!block.ok()) std::abort();
            // Touch each block so the work cannot be optimized away.
            **block = std::byte{1};
            g_byte_sink = **block;
          }
          arena.Rewind(start);
        }
      },
      repeats, 2);

  std::vector<ember::AlignedBuffer> held;
  held.reserve(kBlocksPerRound);
  const ember::TimingSummary system_timing = ember::Repeat(
      [&] {
        for (std::size_t round = 0; round < kRounds; ++round) {
          for (std::size_t i = 0; i < kBlocksPerRound; ++i) {
            auto block =
                ember::AlignedBuffer::Allocate(block_bytes(i), ember::Arena::kDefaultAlignment);
            if (!block.ok()) std::abort();
            *block->data() = std::byte{1};
            g_byte_sink = *block->data();
            held.push_back(std::move(*block));
          }
          held.clear();
        }
      },
      repeats, 2);

  Print("arena, mixed block sizes", arena_timing);
  Print("system allocator, mixed block sizes", system_timing);

  // Other programs can only slow a run down, so the fastest run is the steadiest number on a busy
  // machine, and the median shows what a typical run got.
  const double allocations = static_cast<double>(kRounds * kBlocksPerRound);
  const auto print_per_allocation = [&](const char* name, double arena_ms, double system_ms) {
    const double arena_ns = arena_ms * 1e6 / allocations;
    const double system_ns = system_ms * 1e6 / allocations;
    std::cout << std::setprecision(1) << "per allocation and free, " << name << ": arena "
              << arena_ns << " ns, system " << system_ns << " ns, ratio " << system_ns / arena_ns
              << "\n";
  };
  print_per_allocation("fastest run", arena_timing.min_ms, system_timing.min_ms);
  print_per_allocation("median run", arena_timing.median_ms, system_timing.median_ms);
  return 0;
}
