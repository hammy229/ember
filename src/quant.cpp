#include "ember/quant.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "ember/f16.h"
#include "ember/threads.h"

namespace ember {

namespace {

using Float4 = float __attribute__((vector_size(16)));
using Char4 = std::int8_t __attribute__((vector_size(4)));

Float4 Load4(const float* from) {
  Float4 value;
  std::memcpy(&value, from, sizeof(value));
  return value;
}

Float4 Widen4(const std::int8_t* from) {
  Char4 value;
  std::memcpy(&value, from, sizeof(value));
  return __builtin_convertvector(value, Float4);
}

// One block of 32: x times the int8 values, before the block's scale.
float Block(const float* x, const std::int8_t* w) {
  Float4 a{};
  Float4 b{};
  for (std::size_t k = 0; k < Q8Matrix::kBlock; k += 8) {
    a += Load4(x + k) * Widen4(w + k);
    b += Load4(x + k + 4) * Widen4(w + k + 4);
  }
  const Float4 s = a + b;
  return (s[0] + s[1]) + (s[2] + s[3]);
}

}  // namespace

namespace {

// Quantizes one row of `cols` floats into values and scales.
Status QuantizeRow(const float* from, std::size_t cols, std::int8_t* values, float* scales) {
  for (std::size_t b = 0; b < cols / Q8Matrix::kBlock; ++b) {
    const float* const block = from + b * Q8Matrix::kBlock;
    float largest = 0;
    for (std::size_t k = 0; k < Q8Matrix::kBlock; ++k) {
      if (!std::isfinite(block[k])) {
        return Status::InvalidArgument("QuantizeQ8 found a weight that is not a finite number");
      }
      largest = std::max(largest, std::abs(block[k]));
    }
    const float scale = largest / 127.0F;
    scales[b] = scale;
    for (std::size_t k = 0; k < Q8Matrix::kBlock; ++k) {
      const float step = scale == 0 ? 0 : std::nearbyint(block[k] / scale);
      values[b * Q8Matrix::kBlock + k] =
          static_cast<std::int8_t>(std::clamp(step, -127.0F, 127.0F));
    }
  }
  return Status::Ok();
}

StatusOr<Q8Matrix> Allocate(std::size_t rows, std::size_t cols, Arena& arena) {
  if (cols == 0 || cols % Q8Matrix::kBlock != 0) {
    return Status::InvalidArgument("QuantizeQ8 needs rows a multiple of 32 long, got " +
                                   std::to_string(cols));
  }
  Q8Matrix q;
  q.rows = rows;
  q.cols = cols;
  EMBER_ASSIGN_OR_RETURN(std::byte* const values, arena.Allocate(rows * cols));
  EMBER_ASSIGN_OR_RETURN(std::byte* const scales,
                         arena.Allocate(rows * (cols / Q8Matrix::kBlock) * sizeof(float)));
  q.values = reinterpret_cast<std::int8_t*>(values);
  q.scales = reinterpret_cast<float*>(scales);
  return q;
}

}  // namespace

StatusOr<Q8Matrix> QuantizeQ8(const TensorView& weight, Arena& arena) {
  if (weight.dtype() != DType::kF32 || weight.rank() != 2 || !weight.is_contiguous()) {
    return Status::InvalidArgument("QuantizeQ8 needs a contiguous f32 matrix, got " +
                                   weight.ToString());
  }
  EMBER_ASSIGN_OR_RETURN(Q8Matrix q, Allocate(weight.dim(0), weight.dim(1), arena));
  const std::size_t blocks = q.cols / Q8Matrix::kBlock;
  for (std::size_t r = 0; r < q.rows; ++r) {
    EMBER_RETURN_IF_ERROR(QuantizeRow(weight.data<float>() + r * q.cols, q.cols,
                                      const_cast<std::int8_t*>(q.values) + r * q.cols,
                                      const_cast<float*>(q.scales) + r * blocks));
  }
  return q;
}

StatusOr<Q8Matrix> QuantizeQ8(DType dtype, std::span<const std::byte> bytes, std::size_t rows,
                              std::size_t cols, Arena& arena) {
  const std::size_t width = dtype == DType::kF32 ? 4 : 2;
  if ((dtype != DType::kF32 && dtype != DType::kF16 && dtype != DType::kBF16) ||
      bytes.size() != rows * cols * width) {
    return Status::InvalidArgument("QuantizeQ8 needs f32, f16, or bf16 bytes of the right size");
  }
  EMBER_ASSIGN_OR_RETURN(Q8Matrix q, Allocate(rows, cols, arena));
  const std::size_t blocks = cols / Q8Matrix::kBlock;
  std::vector<float> row(cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const std::byte* const from = bytes.data() + r * cols * width;
    if (dtype == DType::kF32) {
      std::memcpy(row.data(), from, cols * sizeof(float));
    } else if (dtype == DType::kF16) {
      F16ToF32(from, cols, row.data());
    } else {
      BF16ToF32(from, cols, row.data());
    }
    EMBER_RETURN_IF_ERROR(QuantizeRow(row.data(), cols,
                                      const_cast<std::int8_t*>(q.values) + r * cols,
                                      const_cast<float*>(q.scales) + r * blocks));
  }
  return q;
}

Status LinearQ8(const TensorView& x, const Q8Matrix& weight, const Tensor& out) {
  if (x.dtype() != DType::kF32 || x.rank() != 2 || !x.is_contiguous() ||
      out.dtype() != DType::kF32 || out.rank() != 2 || !out.is_contiguous() || weight.empty() ||
      x.dim(1) != weight.cols || out.dim(0) != x.dim(0) || out.dim(1) != weight.rows) {
    return Status::InvalidArgument("LinearQ8 cannot apply a " + std::to_string(weight.rows) +
                                   " x " + std::to_string(weight.cols) + " weight to " +
                                   x.ToString() + " into " + out.ToString());
  }
  if (MemoryRangesIntersect(x, out)) {
    return Status::InvalidArgument("LinearQ8 cannot write to an output that overlaps x");
  }
  const std::size_t rows = x.dim(0);
  const std::size_t in = weight.cols;
  const std::size_t features = weight.rows;
  const std::size_t blocks = in / Q8Matrix::kBlock;
  const float* const xs = x.data<float>();
  float* const to = out.data<float>();
  ParallelFor(features, 64, [&](std::size_t begin, std::size_t end) {
    for (std::size_t r = 0; r < rows; ++r) {
      const float* const row = xs + r * in;
      for (std::size_t f = begin; f < end; ++f) {
        const std::int8_t* const w = weight.values + f * in;
        const float* const scale = weight.scales + f * blocks;
        float total = 0;
        for (std::size_t b = 0; b < blocks; ++b) {
          total += scale[b] * Block(row + b * Q8Matrix::kBlock, w + b * Q8Matrix::kBlock);
        }
        to[r * features + f] = total;
      }
    }
  });
  return Status::Ok();
}

}  // namespace ember
