#include "ember/quant.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

#include "ember/f16.h"
#include "ember/threads.h"

namespace ember {

namespace {

#if !defined(__ARM_FEATURE_DOTPROD)
// One block of 32: the exact integer dot product of two rows of int8 values, as a plain loop the
// compiler can vectorize. ARM chips with the dot product extension use vdotq_s32 in Row instead.
std::int32_t Dot32(const std::int8_t* a, const std::int8_t* b) {
  std::int32_t total = 0;
  for (std::size_t k = 0; k < Q8Matrix::kBlock; ++k) {
    total += static_cast<std::int32_t>(a[k]) * static_cast<std::int32_t>(b[k]);
  }
  return total;
}
#endif

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
  if (rows > std::numeric_limits<std::size_t>::max() / cols / 4) {
    return Status::InvalidArgument("QuantizeQ8 got a matrix too large to address");
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

// One output: the sum over blocks of each block's integer dot product times its two scales. With
// the dot product extension, the per-block sums stay in vector registers and only the final total
// is added across lanes, rather than once per block.
float Row(const std::int8_t* x, const float* sx, const std::int8_t* w, const float* sw,
          std::size_t blocks) {
#if defined(__ARM_FEATURE_DOTPROD)
  float32x4_t total = vdupq_n_f32(0);
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::int8_t* const xb = x + b * Q8Matrix::kBlock;
    const std::int8_t* const wb = w + b * Q8Matrix::kBlock;
    int32x4_t dot = vdotq_s32(vdupq_n_s32(0), vld1q_s8(xb), vld1q_s8(wb));
    dot = vdotq_s32(dot, vld1q_s8(xb + 16), vld1q_s8(wb + 16));
    total = vmlaq_n_f32(total, vcvtq_f32_s32(dot), sw[b] * sx[b]);
  }
  return vaddvq_f32(total);
#else
  float total = 0;
  for (std::size_t b = 0; b < blocks; ++b) {
    total += static_cast<float>(Dot32(x + b * Q8Matrix::kBlock, w + b * Q8Matrix::kBlock)) *
             (sw[b] * sx[b]);
  }
  return total;
#endif
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
  if (cols == 0 || rows > std::numeric_limits<std::size_t>::max() / cols / width) {
    return Status::InvalidArgument("QuantizeQ8 got a matrix too large to address");
  }
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
  // The inputs are quantized the same way as the weights, once per call, so every block is an
  // integer dot product times two scales. llama.cpp does the same for its 8-bit format. The integer
  // sums are exact, so the answer is still the same for any number of threads.
  std::vector<std::int8_t> xq(rows * in);
  std::vector<float> xscale(rows * blocks);
  for (std::size_t r = 0; r < rows; ++r) {
    EMBER_RETURN_IF_ERROR(
        QuantizeRow(x.data<float>() + r * in, in, xq.data() + r * in, xscale.data() + r * blocks));
  }
  float* const to = out.data<float>();
  ParallelFor(features, 64, [&](std::size_t begin, std::size_t end) {
    for (std::size_t r = 0; r < rows; ++r) {
      const std::int8_t* const xr = xq.data() + r * in;
      const float* const sx = xscale.data() + r * blocks;
      for (std::size_t f = begin; f < end; ++f) {
        const std::int8_t* const w = weight.values + f * in;
        const float* const sw = weight.scales + f * blocks;
        to[r * features + f] = Row(xr, sx, w, sw, blocks);
      }
    }
  });
  return Status::Ok();
}

}  // namespace ember
