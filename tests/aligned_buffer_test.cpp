#include "ember/aligned_buffer.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "ember/internal/asan.h"

// AddressSanitizer treats an impossible allocation as fatal by default. Tests here need to see the
// same failure a real out of memory gives, so ask it to return null instead.
#if EMBER_ASAN
extern "C" const char* __asan_default_options() { return "allocator_may_return_null=1"; }
#endif

namespace {

bool IsAligned(const void* pointer, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

}  // namespace

TEST(AlignedBuffer, DefaultIsEmpty) {
  const ember::AlignedBuffer buffer;
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size(), 0U);
}

TEST(AlignedBuffer, RespectsEveryPowerOfTwoAlignment) {
  for (std::size_t alignment = 1; alignment <= 65536; alignment *= 2) {
    auto buffer = ember::AlignedBuffer::Allocate(100, alignment);
    ASSERT_TRUE(buffer.ok()) << alignment;
    EXPECT_TRUE(IsAligned(buffer->data(), alignment)) << alignment;
    EXPECT_EQ(buffer->size(), 100U);
  }
}

TEST(AlignedBuffer, WholeRangeCanBeWritten) {
  auto buffer = ember::AlignedBuffer::Allocate(4096, 64);
  ASSERT_TRUE(buffer.ok());
  std::memset(buffer->data(), 0xAB, buffer->size());
  EXPECT_EQ(buffer->data()[4095], std::byte{0xAB});
}

TEST(AlignedBuffer, ZeroBytesStillGivesAPointer) {
  auto buffer = ember::AlignedBuffer::Allocate(0, 64);
  ASSERT_TRUE(buffer.ok());
  EXPECT_NE(buffer->data(), nullptr);
  EXPECT_EQ(buffer->size(), 0U);
}

TEST(AlignedBuffer, RejectsAlignmentsThatAreNotPowersOfTwo) {
  for (const std::size_t alignment : {std::size_t{0}, std::size_t{3}, std::size_t{48}}) {
    const auto buffer = ember::AlignedBuffer::Allocate(16, alignment);
    ASSERT_FALSE(buffer.ok()) << alignment;
    EXPECT_EQ(buffer.status().code(), ember::StatusCode::kInvalidArgument);
  }
}

TEST(AlignedBuffer, ReportsSizesNoObjectCouldHave) {
  const auto buffer = ember::AlignedBuffer::Allocate(std::numeric_limits<std::size_t>::max(), 64);
  ASSERT_FALSE(buffer.ok());
  EXPECT_EQ(buffer.status().code(), ember::StatusCode::kOutOfMemory);
}

TEST(AlignedBuffer, ReportsWhenTheSystemRefuses) {
  // 2^62 bytes is a legal object size, but far beyond any machine's address space.
  const auto buffer = ember::AlignedBuffer::Allocate(std::size_t{1} << 62, 64);
  ASSERT_FALSE(buffer.ok());
  EXPECT_EQ(buffer.status().code(), ember::StatusCode::kOutOfMemory);
}

TEST(AlignedBuffer, MoveHandsOverOwnership) {
  auto source = ember::AlignedBuffer::Allocate(32, 64);
  ASSERT_TRUE(source.ok());
  const std::byte* const memory = source->data();

  ember::AlignedBuffer moved = std::move(*source);
  EXPECT_EQ(moved.data(), memory);
  EXPECT_EQ(moved.size(), 32U);
  EXPECT_EQ(source->data(), nullptr);
  EXPECT_EQ(source->size(), 0U);
}

TEST(AlignedBuffer, MoveAssignFreesTheOldMemory) {
  auto first = ember::AlignedBuffer::Allocate(32, 64);
  auto second = ember::AlignedBuffer::Allocate(64, 128);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  const std::byte* const memory = second->data();

  *first = std::move(*second);
  EXPECT_EQ(first->data(), memory);
  EXPECT_EQ(first->size(), 64U);
  EXPECT_EQ(second->data(), nullptr);
}
