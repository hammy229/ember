#include "ember/arena.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using ember::Arena;

bool IsAligned(const void* pointer, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

Arena::Options SmallChunks(std::size_t chunk_bytes) {
  Arena::Options options;
  options.chunk_bytes = chunk_bytes;
  return options;
}

}  // namespace

TEST(Arena, StartsEmptyAndHoldsNothing) {
  const Arena arena;
  EXPECT_EQ(arena.used_bytes(), 0U);
  EXPECT_EQ(arena.peak_bytes(), 0U);
  EXPECT_EQ(arena.capacity_bytes(), 0U);
  EXPECT_EQ(arena.chunk_count(), 0U);
}

TEST(Arena, DefaultAlignmentIsACacheLine) {
  Arena arena;
  for (int i = 0; i < 10; ++i) {
    const auto memory = arena.Allocate(7);
    ASSERT_TRUE(memory.ok());
    EXPECT_TRUE(IsAligned(*memory, Arena::kDefaultAlignment));
  }
}

TEST(Arena, RespectsEveryAllowedAlignment) {
  Arena arena;
  for (std::size_t alignment = 1; alignment <= Arena::kMaxAlignment; alignment *= 2) {
    // A one byte allocation first, so the next address is not already aligned by luck.
    ASSERT_TRUE(arena.Allocate(1, 1).ok());
    const auto memory = arena.Allocate(16, alignment);
    ASSERT_TRUE(memory.ok()) << alignment;
    EXPECT_TRUE(IsAligned(*memory, alignment)) << alignment;
  }
}

TEST(Arena, RejectsBadAlignments) {
  Arena arena;
  for (const std::size_t alignment : {std::size_t{0}, std::size_t{3}, Arena::kMaxAlignment * 2}) {
    const auto memory = arena.Allocate(16, alignment);
    ASSERT_FALSE(memory.ok()) << alignment;
    EXPECT_EQ(memory.status().code(), ember::StatusCode::kInvalidArgument);
  }
}

TEST(Arena, AllocationsDoNotOverlap) {
  Arena arena(SmallChunks(256));
  std::vector<std::byte*> blocks;
  constexpr std::size_t kBlockBytes = 100;
  constexpr int kBlocks = 50;
  for (int i = 0; i < kBlocks; ++i) {
    const auto memory = arena.Allocate(kBlockBytes);
    ASSERT_TRUE(memory.ok());
    std::memset(*memory, i, kBlockBytes);
    blocks.push_back(*memory);
  }
  for (int i = 0; i < kBlocks; ++i) {
    for (std::size_t b = 0; b < kBlockBytes; ++b) {
      ASSERT_EQ(blocks[static_cast<std::size_t>(i)][b], static_cast<std::byte>(i));
    }
  }
}

TEST(Arena, GrowsByAddingChunksAndKeepsOldPointersValid) {
  Arena arena(SmallChunks(128));
  const auto first = arena.Allocate(64);
  ASSERT_TRUE(first.ok());
  std::memset(*first, 0x5A, 64);

  for (int i = 0; i < 100; ++i) ASSERT_TRUE(arena.Allocate(64).ok());

  EXPECT_GT(arena.chunk_count(), 1U);
  EXPECT_EQ((*first)[63], std::byte{0x5A});
}

TEST(Arena, ARequestLargerThanAChunkGetsItsOwnChunk) {
  Arena arena(SmallChunks(128));
  const auto memory = arena.Allocate(10000);
  ASSERT_TRUE(memory.ok());
  std::memset(*memory, 1, 10000);
  EXPECT_EQ(arena.chunk_count(), 1U);
  EXPECT_GE(arena.capacity_bytes(), 10000U);
}

TEST(Arena, ZeroBytesGivesAnAlignedPointerAndUsesNoSpace) {
  Arena arena;
  const auto memory = arena.Allocate(0);
  ASSERT_TRUE(memory.ok());
  EXPECT_NE(*memory, nullptr);
  EXPECT_TRUE(IsAligned(*memory, Arena::kDefaultAlignment));
  EXPECT_EQ(arena.used_bytes(), 0U);
}

TEST(Arena, TracksUsedAndPeakBytes) {
  Arena arena;
  ASSERT_TRUE(arena.Allocate(100, 1).ok());
  EXPECT_EQ(arena.used_bytes(), 100U);
  ASSERT_TRUE(arena.Allocate(28, 1).ok());
  EXPECT_EQ(arena.used_bytes(), 128U);
  EXPECT_EQ(arena.peak_bytes(), 128U);
}

TEST(Arena, UsedBytesCountsAlignmentPadding) {
  Arena arena;
  ASSERT_TRUE(arena.Allocate(1).ok());
  ASSERT_TRUE(arena.Allocate(1).ok());
  // The second block starts on the next 64 byte boundary, so 63 bytes of padding were spent.
  EXPECT_EQ(arena.used_bytes(), 65U);
}

TEST(Arena, StopsAtTheMemoryLimit) {
  Arena::Options options;
  options.chunk_bytes = 1024;
  options.max_bytes = 4096;
  Arena arena(options);

  int successes = 0;
  ember::Status failure;
  while (failure.ok()) {
    const auto memory = arena.Allocate(512);
    if (memory.ok()) {
      ++successes;
    } else {
      failure = memory.status();
    }
  }
  EXPECT_GT(successes, 0);
  EXPECT_EQ(failure.code(), ember::StatusCode::kOutOfMemory);
  EXPECT_LE(arena.capacity_bytes(), options.max_bytes);
}

TEST(Arena, ARequestOverTheLimitFailsWithoutTouchingTheSystem) {
  Arena::Options options;
  options.max_bytes = 1024;
  Arena arena(options);
  const auto memory = arena.Allocate(1 << 20);
  ASSERT_FALSE(memory.ok());
  EXPECT_EQ(memory.status().code(), ember::StatusCode::kOutOfMemory);
  EXPECT_EQ(arena.capacity_bytes(), 0U);
}

TEST(Arena, UsesTheLastBytesBeforeTheLimit) {
  // The limit is smaller than one chunk, so the arena must settle for a smaller chunk.
  Arena::Options options;
  options.chunk_bytes = 1 << 20;
  options.max_bytes = 4096;
  Arena arena(options);
  EXPECT_TRUE(arena.Allocate(1000).ok());
  EXPECT_LE(arena.capacity_bytes(), 4096U);
}

TEST(Arena, HugeRequestsFailInsteadOfOverflowing) {
  Arena arena;
  const std::size_t max = std::numeric_limits<std::size_t>::max();
  for (const std::size_t bytes : {max, max - 1, max - 63}) {
    const auto memory = arena.Allocate(bytes);
    ASSERT_FALSE(memory.ok()) << bytes;
    EXPECT_EQ(memory.status().code(), ember::StatusCode::kOutOfMemory);
  }
  EXPECT_EQ(arena.used_bytes(), 0U);
}

TEST(Arena, AFailedAllocationLeavesTheArenaUsable) {
  Arena::Options options;
  options.chunk_bytes = 1024;
  options.max_bytes = 2048;
  Arena arena(options);
  ASSERT_TRUE(arena.Allocate(100).ok());
  const std::size_t used = arena.used_bytes();
  ASSERT_FALSE(arena.Allocate(1 << 20).ok());
  EXPECT_EQ(arena.used_bytes(), used);
  EXPECT_TRUE(arena.Allocate(100).ok());
}

TEST(Arena, RewindGivesTheSameMemoryBack) {
  Arena arena;
  ASSERT_TRUE(arena.Allocate(100).ok());
  const Arena::Mark mark = arena.mark();
  const std::size_t used = arena.used_bytes();
  const auto first = arena.Allocate(100);
  ASSERT_TRUE(first.ok());

  arena.Rewind(mark);
  EXPECT_EQ(arena.used_bytes(), used);

  const auto again = arena.Allocate(100);
  ASSERT_TRUE(again.ok());
  EXPECT_EQ(*again, *first);
}

TEST(Arena, MemoryBeforeTheMarkSurvivesARewind) {
  Arena arena;
  const auto kept = arena.Allocate(64);
  ASSERT_TRUE(kept.ok());
  std::memset(*kept, 0x11, 64);

  const Arena::Mark mark = arena.mark();
  const auto scratch = arena.Allocate(64);
  ASSERT_TRUE(scratch.ok());
  std::memset(*scratch, 0x22, 64);
  arena.Rewind(mark);

  const auto reused = arena.Allocate(64);
  ASSERT_TRUE(reused.ok());
  std::memset(*reused, 0x33, 64);
  EXPECT_EQ((*kept)[63], std::byte{0x11});
}

TEST(Arena, RewindAcrossChunksReusesThemWithoutAskingForMore) {
  Arena arena(SmallChunks(256));
  const Arena::Mark start = arena.mark();
  for (int i = 0; i < 20; ++i) ASSERT_TRUE(arena.Allocate(128).ok());
  const std::size_t chunks = arena.chunk_count();
  const std::size_t capacity = arena.capacity_bytes();
  ASSERT_GT(chunks, 1U);

  // Doing the same work again must not hold any more memory. This is what makes generating many
  // tokens in a row cost a fixed amount.
  for (int round = 0; round < 5; ++round) {
    arena.Rewind(start);
    for (int i = 0; i < 20; ++i) ASSERT_TRUE(arena.Allocate(128).ok());
  }
  EXPECT_EQ(arena.chunk_count(), chunks);
  EXPECT_EQ(arena.capacity_bytes(), capacity);
}

TEST(Arena, ResetKeepsChunksAndPeak) {
  Arena arena;
  ASSERT_TRUE(arena.Allocate(1000, 1).ok());
  const std::size_t capacity = arena.capacity_bytes();
  arena.Reset();
  EXPECT_EQ(arena.used_bytes(), 0U);
  EXPECT_EQ(arena.peak_bytes(), 1000U);
  EXPECT_EQ(arena.capacity_bytes(), capacity);
}

TEST(Arena, ResettingAnEmptyArenaIsFine) {
  Arena arena;
  arena.Reset();
  EXPECT_TRUE(arena.Allocate(8).ok());
}

TEST(Arena, AFailedRequestDoesNotSkipReusableChunks) {
  Arena::Options options;
  options.chunk_bytes = 1024;
  options.max_bytes = 4096;
  Arena arena(options);
  const Arena::Mark start = arena.mark();
  for (int i = 0; i < 3; ++i) ASSERT_TRUE(arena.Allocate(900).ok());
  const std::size_t chunks = arena.chunk_count();
  arena.Rewind(start);

  // Too big for every chunk and over the limit. It must fail, and the chunks stay available.
  ASSERT_FALSE(arena.Allocate(1 << 20).ok());
  for (int i = 0; i < 3; ++i) ASSERT_TRUE(arena.Allocate(900).ok());
  EXPECT_EQ(arena.chunk_count(), chunks);
}

TEST(ArenaScope, RewindsWhenTheScopeEnds) {
  Arena arena;
  ASSERT_TRUE(arena.Allocate(100).ok());
  const std::size_t used = arena.used_bytes();
  {
    const ember::ArenaScope scope(arena);
    ASSERT_TRUE(arena.Allocate(1000).ok());
    EXPECT_GT(arena.used_bytes(), used);
  }
  EXPECT_EQ(arena.used_bytes(), used);
}

TEST(ArenaScope, NestsNewestFirst) {
  Arena arena;
  const ember::ArenaScope outer(arena);
  ASSERT_TRUE(arena.Allocate(100).ok());
  const std::size_t after_outer = arena.used_bytes();
  {
    const ember::ArenaScope inner(arena);
    ASSERT_TRUE(arena.Allocate(100).ok());
  }
  EXPECT_EQ(arena.used_bytes(), after_outer);
}

TEST(ArenaDeathTest, RewindingForwardIsABug) {
  Arena arena;
  ASSERT_TRUE(arena.Allocate(100).ok());
  const Arena::Mark later = arena.mark();
  arena.Reset();
  EXPECT_DEATH(arena.Rewind(later), "ahead of it");
}

TEST(ArenaDeathTest, ZeroSizedChunksAreABug) {
  EXPECT_DEATH(Arena{SmallChunks(0)}, "at least one byte");
}
