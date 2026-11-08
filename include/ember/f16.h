#pragma once

#include <cstddef>
#include <cstdint>

namespace ember {

// Half precision floats, which is how most small models ship their weights. The engine does its
// math in f32, so the loader widens them once at load time. Every f16 value has an exact f32
// value, so nothing is lost doing that.

// The f32 with the same value as these 16 bits. Subnormals, infinities, and NaN all carry over.
float F16ToF32(std::uint16_t bits);

// Widens `count` f16 values, read as little-endian byte pairs, into `out`. Taking bytes rather than
// uint16_t means a file's bytes can be read without caring how they are aligned.
void F16ToF32(const std::byte* from, std::size_t count, float* out);

}  // namespace ember
