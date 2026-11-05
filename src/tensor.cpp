#include "ember/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace ember {

namespace {

// Row by row: the last dimension sits next to itself in memory. A zero dimension counts as one so
// the other strides stay useful, and Shape::Make already proved these products cannot overflow.
std::array<std::size_t, Shape::kMaxRank> RowByRowStrides(const Shape& shape) {
  std::array<std::size_t, Shape::kMaxRank> strides{};
  std::size_t stride = 1;
  for (std::size_t i = shape.rank(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= std::max<std::size_t>(shape.dim(i - 1), 1);
  }
  return strides;
}

Shape EmptyShape() {
  const StatusOr<Shape> shape = Shape::Make({0});
  return shape.value();
}

}  // namespace

TensorView::TensorView()
    : data_(nullptr),
      dtype_(DType::kF32),
      shape_(EmptyShape()),
      strides_(RowByRowStrides(shape_)) {}

StatusOr<TensorView> TensorView::Wrap(const void* data, DType dtype, Shape shape) {
  if (data == nullptr && shape.element_count() > 0) {
    return Status::InvalidArgument("cannot wrap a null pointer as " + shape.ToString());
  }
  if (reinterpret_cast<std::uintptr_t>(data) % SizeOf(dtype) != 0) {
    return Status::InvalidArgument(std::string("pointer is not aligned for ") +
                                   std::string(Name(dtype)));
  }
  return TensorView(static_cast<const std::byte*>(data), dtype, shape, RowByRowStrides(shape));
}

std::size_t TensorView::stride(std::size_t index) const {
  EMBER_CHECK(index < shape_.rank(), "asked for a stride past the end of a tensor");
  return strides_[index];
}

std::string TensorView::ToString() const { return std::string(Name(dtype_)) + shape_.ToString(); }

StatusOr<Tensor> Tensor::Uninitialized(Arena& arena, DType dtype, Shape shape) {
  // Shape::kMaxElements is what guarantees this multiplication cannot overflow.
  EMBER_ASSIGN_OR_RETURN(std::byte* const memory,
                         arena.Allocate(shape.element_count() * SizeOf(dtype)));
  return Wrap(memory, dtype, shape);
}

StatusOr<Tensor> Tensor::Zeros(Arena& arena, DType dtype, Shape shape) {
  EMBER_ASSIGN_OR_RETURN(const Tensor tensor, Uninitialized(arena, dtype, shape));
  std::memset(tensor.raw_data(), 0, tensor.byte_size());
  return tensor;
}

StatusOr<Tensor> Tensor::FromBytes(Arena& arena, DType dtype, Shape shape, const void* values,
                                   std::size_t value_count) {
  if (value_count != shape.element_count()) {
    return Status::InvalidArgument("got " + std::to_string(value_count) + " values for shape " +
                                   shape.ToString());
  }
  EMBER_ASSIGN_OR_RETURN(const Tensor tensor, Uninitialized(arena, dtype, shape));
  if (value_count > 0) std::memcpy(tensor.raw_data(), values, tensor.byte_size());
  return tensor;
}

StatusOr<Tensor> Tensor::Wrap(void* data, DType dtype, Shape shape) {
  EMBER_ASSIGN_OR_RETURN(const TensorView view, TensorView::Wrap(data, dtype, shape));
  return Tensor(view);
}

}  // namespace ember
