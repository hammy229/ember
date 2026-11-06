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

}  // namespace

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
