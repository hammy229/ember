#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ember {

// What one element of a tensor is. kI32 holds token ids and kI8 holds quantized weights.
enum class DType : std::uint8_t {
  kF32,
  kI32,
  kI8,
};

constexpr std::size_t SizeOf(DType dtype) {
  switch (dtype) {
    case DType::kF32:
    case DType::kI32:
      return 4;
    case DType::kI8:
      return 1;
  }
  return 0;
}

constexpr std::string_view Name(DType dtype) {
  switch (dtype) {
    case DType::kF32:
      return "f32";
    case DType::kI32:
      return "i32";
    case DType::kI8:
      return "i8";
  }
  return "unknown";
}

// The C++ types a tensor can be read as. Anything else does not compile.
template <typename T>
concept TensorElement =
    std::same_as<T, float> || std::same_as<T, std::int32_t> || std::same_as<T, std::int8_t>;

template <TensorElement T>
inline constexpr DType kDTypeOf = std::same_as<T, float>          ? DType::kF32
                                  : std::same_as<T, std::int32_t> ? DType::kI32
                                                                  : DType::kI8;

static_assert(sizeof(float) == SizeOf(DType::kF32), "the engine assumes 32 bit floats");
static_assert(sizeof(std::int32_t) == SizeOf(DType::kI32));
static_assert(sizeof(std::int8_t) == SizeOf(DType::kI8));

}  // namespace ember
