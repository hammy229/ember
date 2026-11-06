#include "ember/ops.h"

#include <cstddef>
#include <string>
#include <string_view>

#include "ember/dtype.h"

namespace ember {

namespace {

Status RequireF32(const TensorView& tensor, std::string_view op) {
  if (tensor.dtype() == DType::kF32) return Status::Ok();
  return Status::InvalidArgument(std::string(op) + " only works on f32 tensors, got " +
                                 tensor.ToString());
}

Status RequireSameShape(const TensorView& a, const TensorView& b, std::string_view op) {
  if (a.shape() == b.shape()) return Status::Ok();
  return Status::InvalidArgument(std::string(op) + " needs matching shapes, got " + a.ToString() +
                                 " and " + b.ToString());
}

// True if both read and write the same numbers in the same order.
bool SameLayout(const TensorView& a, const TensorView& b) {
  if (a.raw_data() != b.raw_data() || !(a.shape() == b.shape())) return false;
  for (std::size_t axis = 0; axis < a.rank(); ++axis) {
    // A dimension of size 1 is never stepped along, so its stride does not matter.
    if (a.dim(axis) > 1 && a.stride(axis) != b.stride(axis)) return false;
  }
  return true;
}

// For ops that read each input element once, right before writing the matching output element.
// Writing exactly over an input is fine for those. Any other overlap would read numbers the op has
// already overwritten and silently give the wrong answer.
Status RequireExactlyOverOrApart(const TensorView& input, const TensorView& out,
                                 std::string_view op) {
  if (!MemoryRangesIntersect(input, out) || SameLayout(input, out)) return Status::Ok();
  return Status::InvalidArgument(std::string(op) +
                                 " cannot write to an output that partly overlaps its input");
}

// Runs op over matching elements of a and b and stores the results in out. The innermost dimension
// gets its own loop, and when nothing has a stride there the compiler can use SIMD on it.
template <typename Op>
void ZipAxis(const TensorView& a, const TensorView& b, const Tensor& out, std::size_t axis,
             const float* from_a, const float* from_b, float* to, Op op) {
  if (axis == out.rank()) {
    *to = op(*from_a, *from_b);
    return;
  }
  const std::size_t size = out.dim(axis);
  const std::size_t step_a = a.stride(axis);
  const std::size_t step_b = b.stride(axis);
  const std::size_t step_out = out.stride(axis);
  if (axis + 1 < out.rank()) {
    for (std::size_t i = 0; i < size; ++i) {
      ZipAxis(a, b, out, axis + 1, from_a + i * step_a, from_b + i * step_b, to + i * step_out, op);
    }
  } else if (step_a == 1 && step_b == 1 && step_out == 1) {
    for (std::size_t i = 0; i < size; ++i) to[i] = op(from_a[i], from_b[i]);
  } else {
    for (std::size_t i = 0; i < size; ++i) {
      to[i * step_out] = op(from_a[i * step_a], from_b[i * step_b]);
    }
  }
}

template <typename Op>
Status Zip(const TensorView& a, const TensorView& b, const Tensor& out, std::string_view name,
           Op op) {
  EMBER_RETURN_IF_ERROR(RequireF32(a, name));
  EMBER_RETURN_IF_ERROR(RequireF32(b, name));
  EMBER_RETURN_IF_ERROR(RequireF32(out, name));
  EMBER_RETURN_IF_ERROR(RequireSameShape(a, out, name));
  EMBER_RETURN_IF_ERROR(RequireSameShape(b, out, name));
  EMBER_RETURN_IF_ERROR(RequireExactlyOverOrApart(a, out, name));
  EMBER_RETURN_IF_ERROR(RequireExactlyOverOrApart(b, out, name));
  // An empty tensor can hold a null pointer, and moving a null pointer is undefined behavior.
  if (out.element_count() == 0) return Status::Ok();
  ZipAxis(a, b, out, 0, a.data<float>(), b.data<float>(), out.data<float>(), op);
  return Status::Ok();
}

// row[j] += factor * values[j], along the strides. Nearly all the time in a matrix product goes
// here. The strides are almost always 1, and then the compiler can use SIMD on the first loop.
inline void AddScaled(float factor, const float* values, std::size_t values_step, float* row,
                      std::size_t row_step, std::size_t size) {
  if (values_step == 1 && row_step == 1) {
    for (std::size_t j = 0; j < size; ++j) row[j] += factor * values[j];
  } else {
    for (std::size_t j = 0; j < size; ++j) row[j * row_step] += factor * values[j * values_step];
  }
}

// Goes row, then middle, then column rather than the textbook row, column, middle. Both do the same
// additions, but this order walks b and out through memory in a straight line instead of jumping a
// whole row of b on every step, which is much kinder to the cache.
void MatMul2D(const TensorView& a, const TensorView& b, const Tensor& out, std::size_t axis,
              const float* from_a, const float* from_b, float* to) {
  const std::size_t rows = a.dim(axis);
  const std::size_t middle = a.dim(axis + 1);
  const std::size_t columns = b.dim(axis + 1);
  const std::size_t a_row_step = a.stride(axis);
  const std::size_t a_middle_step = a.stride(axis + 1);
  const std::size_t b_middle_step = b.stride(axis);
  const std::size_t b_column_step = b.stride(axis + 1);
  const std::size_t out_row_step = out.stride(axis);
  const std::size_t out_column_step = out.stride(axis + 1);
  for (std::size_t i = 0; i < rows; ++i) {
    float* const row = to + i * out_row_step;
    for (std::size_t j = 0; j < columns; ++j) row[j * out_column_step] = 0.0F;
    const float* const a_row = from_a + i * a_row_step;
    for (std::size_t k = 0; k < middle; ++k) {
      AddScaled(a_row[k * a_middle_step], from_b + k * b_middle_step, b_column_step, row,
                out_column_step, columns);
    }
  }
}

Status RequireMatMulShapes(const TensorView& a, const TensorView& b, const Tensor& out) {
  const auto refuse = [&] {
    return Status::InvalidArgument("MatMul cannot multiply " + a.ToString() + " by " +
                                   b.ToString() + " into " + out.ToString());
  };
  if (a.rank() != 2 || b.rank() != 2 || out.rank() != 2) return refuse();
  if (a.dim(1) != b.dim(0) || out.dim(0) != a.dim(0) || out.dim(1) != b.dim(1)) return refuse();
  return Status::Ok();
}

}  // namespace

Status MatMul(const TensorView& a, const TensorView& b, const Tensor& out) {
  EMBER_RETURN_IF_ERROR(RequireF32(a, "MatMul"));
  EMBER_RETURN_IF_ERROR(RequireF32(b, "MatMul"));
  EMBER_RETURN_IF_ERROR(RequireF32(out, "MatMul"));
  EMBER_RETURN_IF_ERROR(RequireMatMulShapes(a, b, out));
  // An empty tensor can hold a null pointer, and moving a null pointer is undefined behavior.
  if (out.element_count() == 0) return Status::Ok();
  // With an empty middle dimension a and b can be null too, but the loops never read them then.
  MatMul2D(a, b, out, 0, a.data<float>(), b.data<float>(), out.data<float>());
  return Status::Ok();
}

Status Add(const TensorView& a, const TensorView& b, const Tensor& out) {
  return Zip(a, b, out, "Add", [](float x, float y) { return x + y; });
}

Status Multiply(const TensorView& a, const TensorView& b, const Tensor& out) {
  return Zip(a, b, out, "Multiply", [](float x, float y) { return x * y; });
}

Status Scale(const TensorView& a, float factor, const Tensor& out) {
  return Zip(a, a, out, "Scale", [factor](float x, float) { return x * factor; });
}

}  // namespace ember
