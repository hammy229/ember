#include "ember/layers.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>

#include "ember/ops.h"

namespace ember {

namespace {

Status RequireContiguousF32(const TensorView& tensor, std::size_t rank, std::string_view op) {
  if (tensor.dtype() != DType::kF32 || tensor.rank() != rank || !tensor.is_contiguous()) {
    return Status::InvalidArgument(std::string(op) + " needs a contiguous f32 tensor of rank " +
                                   std::to_string(rank) + ", got " + tensor.ToString());
  }
  return Status::Ok();
}

Status RefuseOverlap(const TensorView& input, const TensorView& out, std::string_view op) {
  if (!MemoryRangesIntersect(input, out)) return Status::Ok();
  return Status::InvalidArgument(std::string(op) +
                                 " cannot write to an output that overlaps an input");
}

}  // namespace

Status Linear(const TensorView& x, const TensorView& weight, const Tensor& out) {
  EMBER_RETURN_IF_ERROR(RequireContiguousF32(x, 2, "Linear"));
  EMBER_RETURN_IF_ERROR(RequireContiguousF32(weight, 2, "Linear"));
  EMBER_RETURN_IF_ERROR(RequireContiguousF32(out, 2, "Linear"));
  const std::size_t rows = x.dim(0);
  const std::size_t in = x.dim(1);
  const std::size_t features = weight.dim(0);
  if (weight.dim(1) != in || out.dim(0) != rows || out.dim(1) != features) {
    return Status::InvalidArgument("Linear cannot apply " + weight.ToString() + " to " +
                                   x.ToString() + " into " + out.ToString());
  }
  EMBER_RETURN_IF_ERROR(RefuseOverlap(x, out, "Linear"));
  EMBER_RETURN_IF_ERROR(RefuseOverlap(weight, out, "Linear"));
  if (out.element_count() == 0) return Status::Ok();
  if (in == 0) {
    out.Fill(0.0F);
    return Status::Ok();
  }

  const float* const xs = x.data<float>();
  const float* const ws = weight.data<float>();
  float* const to = out.data<float>();
  for (std::size_t r = 0; r < rows; ++r) {
    const float* const row = xs + r * in;
    for (std::size_t f = 0; f < features; ++f) {
      const float* const w = ws + f * in;
      // Eight running totals instead of one, so the additions do not all wait on each other and
      // the compiler can keep them in one SIMD register. It changes the rounding slightly, the way
      // any fast matrix library does.
      float sums[8] = {};
      std::size_t k = 0;
      for (; k + 8 <= in; k += 8) {
        for (std::size_t lane = 0; lane < 8; ++lane) sums[lane] += row[k + lane] * w[k + lane];
      }
      float total =
          ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
      for (; k < in; ++k) total += row[k] * w[k];
      to[r * features + f] = total;
    }
  }
  return Status::Ok();
}

Status RmsNorm(const TensorView& x, const TensorView& weight, float eps, const Tensor& out) {
  EMBER_RETURN_IF_ERROR(RequireContiguousF32(x, 2, "RmsNorm"));
  EMBER_RETURN_IF_ERROR(RequireContiguousF32(weight, 1, "RmsNorm"));
  EMBER_RETURN_IF_ERROR(RequireContiguousF32(out, 2, "RmsNorm"));
  if (!(x.shape() == out.shape()) || weight.dim(0) != x.dim(1)) {
    return Status::InvalidArgument("RmsNorm cannot normalize " + x.ToString() + " with " +
                                   weight.ToString() + " into " + out.ToString());
  }
  if (!(eps > 0.0F) || !std::isfinite(eps)) {
    return Status::InvalidArgument("RmsNorm needs a positive eps");
  }
  // Both are contiguous with the same shape, so exactly on top of x is the only safe overlap.
  if (MemoryRangesIntersect(x, out) && x.raw_data() != out.raw_data()) {
    return Status::InvalidArgument("RmsNorm cannot write to an output that partly overlaps x");
  }
  EMBER_RETURN_IF_ERROR(RefuseOverlap(weight, out, "RmsNorm"));
  const std::size_t size = x.dim(1);
  if (x.element_count() == 0) return Status::Ok();

  const float* const ws = weight.data<float>();
  for (std::size_t r = 0; r < x.dim(0); ++r) {
    const float* const from = x.data<float>() + r * size;
    float* const to = out.data<float>() + r * size;
    double squares = 0.0;
    for (std::size_t i = 0; i < size; ++i) squares += static_cast<double>(from[i]) * from[i];
    const auto mean = static_cast<float>(squares / static_cast<double>(size));
    const float inverse = 1.0F / std::sqrt(mean + eps);
    // Hugging Face scales x first and multiplies by the weight after, so this does too.
    for (std::size_t i = 0; i < size; ++i) to[i] = ws[i] * (from[i] * inverse);
  }
  return Status::Ok();
}

Status Embed(const TensorView& table, std::span<const std::int32_t> tokens, const Tensor& out) {
  EMBER_RETURN_IF_ERROR(RequireContiguousF32(table, 2, "Embed"));
  EMBER_RETURN_IF_ERROR(RequireContiguousF32(out, 2, "Embed"));
  const std::size_t size = table.dim(1);
  if (out.dim(0) != tokens.size() || out.dim(1) != size) {
    return Status::InvalidArgument("Embed cannot look up " + std::to_string(tokens.size()) +
                                   " tokens in " + table.ToString() + " into " + out.ToString());
  }
  EMBER_RETURN_IF_ERROR(RefuseOverlap(table, out, "Embed"));
  for (const std::int32_t token : tokens) {
    if (token < 0 || static_cast<std::size_t>(token) >= table.dim(0)) {
      return Status::InvalidArgument("token id " + std::to_string(token) +
                                     " is outside a vocabulary of " + std::to_string(table.dim(0)));
    }
  }
  if (out.element_count() == 0) return Status::Ok();
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    std::memcpy(out.data<float>() + i * size,
                table.data<float>() + static_cast<std::size_t>(tokens[i]) * size,
                size * sizeof(float));
  }
  return Status::Ok();
}

Status ApplyRope(const Tensor& x, std::size_t first_position, float theta) {
  EMBER_RETURN_IF_ERROR(RequireContiguousF32(x, 3, "ApplyRope"));
  const std::size_t tokens = x.dim(0);
  const std::size_t heads = x.dim(1);
  const std::size_t head_dim = x.dim(2);
  if (head_dim % 2 != 0) return Status::InvalidArgument("ApplyRope needs an even head_dim");
  if (!(theta > 0.0F) || !std::isfinite(theta)) {
    return Status::InvalidArgument("ApplyRope needs a positive theta");
  }
  if (x.element_count() == 0) return Status::Ok();
  const std::size_t half = head_dim / 2;

  for (std::size_t t = 0; t < tokens; ++t) {
    const auto position = static_cast<float>(first_position + t);
    for (std::size_t i = 0; i < half; ++i) {
      // Hugging Face computes these in f32: theta^(-2i/d), times the position, then cos and sin.
      const float frequency =
          1.0F / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
      const float angle = position * frequency;
      const float cos = std::cos(angle);
      const float sin = std::sin(angle);
      for (std::size_t h = 0; h < heads; ++h) {
        float* const v = x.data<float>() + (t * heads + h) * head_dim;
        const float first = v[i];
        const float second = v[i + half];
        v[i] = first * cos - second * sin;
        v[i + half] = second * cos + first * sin;
      }
    }
  }
  return Status::Ok();
}

Status CausalAttention(const TensorView& q, const TensorView& k, const TensorView& v,
                       const Tensor& out, Arena& arena) {
  for (const TensorView* tensor : {&q, &k, &v}) {
    EMBER_RETURN_IF_ERROR(RequireContiguousF32(*tensor, 3, "CausalAttention"));
  }
  EMBER_RETURN_IF_ERROR(RequireContiguousF32(out, 3, "CausalAttention"));
  const std::size_t queries = q.dim(0);
  const std::size_t heads = q.dim(1);
  const std::size_t head_dim = q.dim(2);
  const std::size_t keys = k.dim(0);
  const std::size_t kv_heads = k.dim(1);
  if (!(k.shape() == v.shape()) || !(out.shape() == q.shape()) || k.dim(2) != head_dim ||
      kv_heads == 0 || heads % kv_heads != 0 || keys < queries) {
    return Status::InvalidArgument("CausalAttention cannot attend " + q.ToString() + " to keys " +
                                   k.ToString() + " and values " + v.ToString() + " into " +
                                   out.ToString());
  }
  for (const TensorView* input : {&q, &k, &v}) {
    EMBER_RETURN_IF_ERROR(RefuseOverlap(*input, out, "CausalAttention"));
  }
  if (out.element_count() == 0) return Status::Ok();

  const ArenaScope scope(arena);
  const std::size_t group = heads / kv_heads;
  const std::size_t past = keys - queries;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  // [heads, tokens, head_dim] views, so one head is a slice. Nothing is copied.
  EMBER_ASSIGN_OR_RETURN(const TensorView q_heads, q.Transpose(0, 1));
  EMBER_ASSIGN_OR_RETURN(const TensorView k_heads, k.Transpose(0, 1));
  EMBER_ASSIGN_OR_RETURN(const TensorView v_heads, v.Transpose(0, 1));
  EMBER_ASSIGN_OR_RETURN(const Tensor out_heads, out.Transpose(0, 1));
  EMBER_ASSIGN_OR_RETURN(const Shape scores_shape, Shape::Make({1, queries, keys}));
  EMBER_ASSIGN_OR_RETURN(const Tensor scores,
                         Tensor::Uninitialized(arena, DType::kF32, scores_shape));

  for (std::size_t g = 0; g < kv_heads; ++g) {
    // Keys turned sideways to [head_dim, keys]. A transposed matrix makes MatMul about ten times
    // slower (docs/ops.md), so copy it once per key head and reuse it for the whole group.
    EMBER_ASSIGN_OR_RETURN(const TensorView k_head, k_heads.Slice(0, g, 1));
    EMBER_ASSIGN_OR_RETURN(const TensorView k_sideways, k_head.Transpose(1, 2));
    EMBER_ASSIGN_OR_RETURN(const Tensor k_copy, Tensor::CopyOf(arena, k_sideways));
    EMBER_ASSIGN_OR_RETURN(const TensorView v_head, v_heads.Slice(0, g, 1));

    for (std::size_t h = g * group; h < (g + 1) * group; ++h) {
      EMBER_ASSIGN_OR_RETURN(const TensorView q_head, q_heads.Slice(0, h, 1));
      EMBER_RETURN_IF_ERROR(MatMul(q_head, k_copy, scores));
      EMBER_RETURN_IF_ERROR(Scale(scores, scale, scores));
      // Hide the future. Each row keeps at least its own key, so Softmax always has a real score.
      float* const rows = scores.data<float>();
      for (std::size_t i = 0; i < queries; ++i) {
        for (std::size_t j = past + i + 1; j < keys; ++j) {
          rows[i * keys + j] = -std::numeric_limits<float>::infinity();
        }
      }
      EMBER_RETURN_IF_ERROR(Softmax(scores, 2, scores));
      EMBER_ASSIGN_OR_RETURN(const Tensor out_head, out_heads.Slice(0, h, 1));
      EMBER_RETURN_IF_ERROR(MatMul(scores, v_head, out_head));
    }
  }
  return Status::Ok();
}

}  // namespace ember
