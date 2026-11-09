#pragma once

#include <cstddef>
#include <cstdint>

#include "ember/arena.h"
#include "ember/status_or.h"
#include "ember/tensor.h"

namespace ember {

// A weight matrix in 8 bits: each row is cut into blocks of 32 numbers, and each block is stored as
// 32 int8 values and one f32 scale, value * scale being the weight. That is 34 bytes for 32 weights
// instead of 128, the same layout as llama.cpp's Q8_0. Generation reads every weight once per
// token, so on a model too big for the CPU's caches the speed follows the bytes.
struct Q8Matrix {
  static constexpr std::size_t kBlock = 32;

  std::size_t rows = 0;
  std::size_t cols = 0;
  const std::int8_t* values = nullptr;  // [rows, cols]
  const float* scales = nullptr;        // [rows, cols / kBlock]

  bool empty() const { return values == nullptr; }
  std::size_t bytes() const { return rows * cols + rows * (cols / kBlock) * sizeof(float); }
};

// Quantizes weight [rows, cols] (f32, contiguous, cols a multiple of 32) into memory from the
// arena. Each block's scale is its largest absolute value / 127 and each value is rounded to
// nearest, so no weight moves by more than half a step. An all-zero block gets scale 0.
StatusOr<Q8Matrix> QuantizeQ8(const TensorView& weight, Arena& arena);

// The same as Linear with the quantized weight: x [rows, in] times weight [features, in] turned
// sideways, into out [rows, features]. Split across threads like Linear, with the same answer for
// any number of threads.
Status LinearQ8(const TensorView& x, const Q8Matrix& weight, const Tensor& out);

}  // namespace ember
