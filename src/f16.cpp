#include "ember/f16.h"

#include <bit>
#include <cstddef>
#include <cstdint>

namespace ember {

float F16ToF32(std::uint16_t bits) {
  const std::uint32_t sign = (std::uint32_t{bits} & 0x8000U) << 16;
  const std::uint32_t exponent = (std::uint32_t{bits} >> 10) & 0x1FU;
  std::uint32_t mantissa = std::uint32_t{bits} & 0x3FFU;

  std::uint32_t result = 0;
  if (exponent == 0x1F) {
    // Infinity or NaN. Keeping the mantissa keeps a NaN a NaN.
    result = sign | 0x7F800000U | (mantissa << 13);
  } else if (exponent != 0) {
    // f16 offsets its exponent by 15 and f32 by 127.
    result = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  } else if (mantissa == 0) {
    result = sign;
  } else {
    // A subnormal f16 is mantissa * 2^-24, which is a normal f32. Shift until the leading one is
    // where an implied bit would be, and lower the exponent once per shift.
    std::uint32_t shifts = 0;
    while ((mantissa & 0x400U) == 0) {
      mantissa <<= 1;
      ++shifts;
    }
    mantissa &= 0x3FFU;
    result = sign | ((113U - shifts) << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(result);
}

void F16ToF32(const std::byte* from, std::size_t count, float* out) {
  for (std::size_t i = 0; i < count; ++i) {
    const auto low = static_cast<std::uint16_t>(from[2 * i]);
    const auto high = static_cast<std::uint16_t>(from[2 * i + 1]);
    out[i] = F16ToF32(static_cast<std::uint16_t>(low | (high << 8)));
  }
}

}  // namespace ember
