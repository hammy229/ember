// Runs the arena through long random sequences of allocations and rewinds, and checks it against a
// simple model of what should be alive. Hand-written tests only cover the orders someone thought
// of. The seeds are fixed, so a failure here happens the same way every time.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "ember/arena.h"

namespace {

using ember::Arena;

struct Block {
  std::byte* data;
  std::size_t bytes;
  std::byte fill;
};

struct Frame {
  Arena::Mark mark;
  std::size_t live_blocks;
  std::size_t used_bytes;
};

// std::uniform_int_distribution gives different numbers on different standard libraries. Taking a
// remainder is slightly biased, which does not matter here, and is the same everywhere.
std::size_t Below(std::mt19937& rng, std::size_t limit) {
  return static_cast<std::size_t>(rng()) % limit;
}

void ExpectBlockIntact(const Block& block) {
  for (std::size_t i = 0; i < block.bytes; ++i) {
    ASSERT_EQ(block.data[i], block.fill) << "byte " << i << " of a live block changed";
  }
}

class ArenaRandomTest : public ::testing::TestWithParam<std::uint32_t> {};

TEST_P(ArenaRandomTest, BehavesLikeTheModel) {
  std::mt19937 rng(GetParam());
  Arena::Options options;
  options.chunk_bytes = std::size_t{1} << (8 + Below(rng, 8));  // 256 bytes to 32 KiB.
  options.max_bytes = std::size_t{1} << 24;
  Arena arena(options);

  std::vector<Block> live;
  std::vector<Frame> frames;
  constexpr int kSteps = 3000;

  for (int step = 0; step < kSteps; ++step) {
    const std::size_t action = Below(rng, 10);
    if (action < 7) {
      // Mostly small blocks, with the occasional one bigger than a chunk.
      const std::size_t bytes = Below(rng, 20) == 0 ? Below(rng, 100000) : Below(rng, 512);
      const std::size_t alignment = std::size_t{1} << Below(rng, 13);
      const auto memory = arena.Allocate(bytes, alignment);
      if (!memory.ok()) {
        // The only allowed failure is the limit, and it must not disturb anything.
        ASSERT_EQ(memory.status().code(), ember::StatusCode::kOutOfMemory);
        continue;
      }
      ASSERT_EQ(reinterpret_cast<std::uintptr_t>(*memory) % alignment, 0U);
      const auto fill = static_cast<std::byte>(step % 251 + 1);
      std::memset(*memory, static_cast<int>(fill), bytes);
      live.push_back(Block{*memory, bytes, fill});
    } else if (action == 7) {
      frames.push_back(Frame{arena.mark(), live.size(), arena.used_bytes()});
    } else if (action == 8 && !frames.empty()) {
      const Frame frame = frames.back();
      frames.pop_back();
      arena.Rewind(frame.mark);
      live.resize(frame.live_blocks);
      ASSERT_EQ(arena.used_bytes(), frame.used_bytes);
    } else if (action == 9) {
      // Each live block still holds exactly what was written, so nothing overlapped.
      for (const Block& block : live) ExpectBlockIntact(block);
      if (HasFatalFailure()) return;
    }

    ASSERT_LE(arena.capacity_bytes(), options.max_bytes);
    ASSERT_LE(arena.used_bytes(), arena.peak_bytes());
  }

  for (const Block& block : live) ExpectBlockIntact(block);
}

INSTANTIATE_TEST_SUITE_P(Seeds, ArenaRandomTest,
                         ::testing::Values(1U, 2U, 3U, 5U, 8U, 13U, 21U, 34U, 55U, 89U));

}  // namespace
