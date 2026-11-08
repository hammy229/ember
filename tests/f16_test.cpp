#include "ember/f16.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

// The value an f16 stands for, worked out from its definition rather than from bit tricks.
double Reference(std::uint16_t bits) {
  const int sign = (bits & 0x8000) != 0 ? -1 : 1;
  const int exponent = (bits >> 10) & 0x1F;
  const int mantissa = bits & 0x3FF;
  if (exponent == 0x1F) {
    return mantissa == 0 ? sign * std::numeric_limits<double>::infinity()
                         : std::numeric_limits<double>::quiet_NaN();
  }
  if (exponent == 0) return sign * std::ldexp(mantissa, -24);
  return sign * std::ldexp(1024 + mantissa, exponent - 25);
}

TEST(F16, EveryBitPatternMatchesItsDefinition) {
  for (std::uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
    const auto half = static_cast<std::uint16_t>(bits);
    const float widened = ember::F16ToF32(half);
    const double expected = Reference(half);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(widened)) << "bits " << bits;
    } else {
      // Exact, because every f16 value is also an f32 value.
      EXPECT_EQ(static_cast<double>(widened), expected) << "bits " << bits;
      EXPECT_EQ(std::signbit(widened), (half & 0x8000) != 0) << "bits " << bits;
    }
  }
}

TEST(F16, KnownValues) {
  EXPECT_EQ(ember::F16ToF32(0x3C00), 1.0F);
  EXPECT_EQ(ember::F16ToF32(0xC000), -2.0F);
  EXPECT_EQ(ember::F16ToF32(0x7BFF), 65504.0F);
  EXPECT_EQ(ember::F16ToF32(0x0001), std::ldexp(1.0F, -24));
  EXPECT_EQ(ember::F16ToF32(0x7C00), std::numeric_limits<float>::infinity());
}

TEST(F16, ReadsLittleEndianBytesAtAnyAlignment) {
  // One byte of padding first, so the values start at an odd address.
  const std::array<std::byte, 5> bytes{std::byte{0xAA}, std::byte{0x00}, std::byte{0x3C},
                                       std::byte{0x00}, std::byte{0xC0}};
  std::array<float, 2> out{};
  ember::F16ToF32(bytes.data() + 1, out.size(), out.data());
  EXPECT_EQ(out[0], 1.0F);
  EXPECT_EQ(out[1], -2.0F);
}

}  // namespace
