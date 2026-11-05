#include "ember/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <utility>

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

using Dims = std::array<std::size_t, Shape::kMaxRank>;

Dims CopyDims(const Shape& shape) {
  Dims dims{};
  std::copy(shape.dims().begin(), shape.dims().end(), dims.begin());
  return dims;
}

struct ByteRange {
  const std::byte* first;
  const std::byte* end;
};

// From the lowest to one past the highest byte the tensor touches. Strides are never negative, so
// the lowest byte is always the first element. Only call this on a tensor with elements.
ByteRange RangeOf(const TensorView& tensor) {
  std::size_t last_element = 0;
  for (std::size_t i = 0; i < tensor.rank(); ++i) {
    last_element += (tensor.dim(i) - 1) * tensor.stride(i);
  }
  const std::size_t element_bytes = SizeOf(tensor.dtype());
  return {tensor.raw_data(), tensor.raw_data() + (last_element + 1) * element_bytes};
}

bool Overlaps(const ByteRange& a, const ByteRange& b) {
  // std::less gives a total order even for pointers into different blocks of memory.
  const std::less<const std::byte*> before;
  return before(a.first, b.end) && before(b.first, a.end);
}

void CopyAxis(const TensorView& source, const Tensor& target, std::size_t axis,
              const std::byte* from, std::byte* to) {
  const std::size_t element_bytes = SizeOf(source.dtype());
  if (axis == source.rank()) {
    std::memcpy(to, from, element_bytes);
    return;
  }
  const std::size_t from_step = source.stride(axis) * element_bytes;
  const std::size_t to_step = target.stride(axis) * element_bytes;
  for (std::size_t i = 0; i < source.dim(axis); ++i) {
    CopyAxis(source, target, axis + 1, from + i * from_step, to + i * to_step);
  }
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
  // The element type can come straight from a model file, so it may not be one we know.
  if (SizeOf(dtype) == 0) return Status::InvalidFormat("unknown element type");
  if (data == nullptr && shape.element_count() > 0) {
    return Status::InvalidArgument("cannot wrap a null pointer as " + shape.ToString());
  }
  if (reinterpret_cast<std::uintptr_t>(data) % SizeOf(dtype) != 0) {
    return Status::InvalidArgument(std::string("pointer is not aligned for ") +
                                   std::string(Name(dtype)));
  }
  return TensorView(static_cast<const std::byte*>(data), dtype, shape, RowByRowStrides(shape));
}

StatusOr<TensorView> TensorView::Wrap(std::span<const std::byte> bytes, DType dtype, Shape shape) {
  // Shape::kMaxElements is what guarantees this multiplication cannot overflow.
  const std::size_t needed = shape.element_count() * SizeOf(dtype);
  if (needed > bytes.size()) {
    return Status::InvalidFormat(std::string(Name(dtype)) + shape.ToString() + " needs " +
                                 std::to_string(needed) + " bytes, but there are " +
                                 std::to_string(bytes.size()));
  }
  return Wrap(static_cast<const void*>(bytes.data()), dtype, shape);
}

std::size_t TensorView::stride(std::size_t index) const {
  EMBER_CHECK(index < shape_.rank(), "asked for a stride past the end of a tensor");
  return strides_[index];
}

bool TensorView::is_contiguous() const {
  if (element_count() == 0) return true;
  std::size_t expected = 1;
  for (std::size_t i = rank(); i > 0; --i) {
    const std::size_t size = dim(i - 1);
    // A dimension of size 1 is never stepped along, so its stride does not matter.
    if (size == 1) continue;
    if (strides_[i - 1] != expected) return false;
    expected *= size;
  }
  return true;
}

StatusOr<TensorView> TensorView::Slice(std::size_t axis, std::size_t start,
                                       std::size_t length) const {
  if (axis >= rank()) {
    return Status::InvalidArgument("cannot slice dimension " + std::to_string(axis) + " of " +
                                   ToString());
  }
  // Written this way so a huge start plus a huge length cannot wrap around and pass.
  if (start > dim(axis) || length > dim(axis) - start) {
    return Status::InvalidArgument("slice of " + std::to_string(length) + " entries at " +
                                   std::to_string(start) + " does not fit dimension " +
                                   std::to_string(axis) + " of " + ToString());
  }
  Dims dims = CopyDims(shape_);
  dims[axis] = length;
  EMBER_ASSIGN_OR_RETURN(const Shape sliced, Shape::Make(std::span(dims.data(), rank())));
  // An empty tensor can hold a null pointer, and moving a null pointer is undefined behavior.
  const std::byte* const start_of_slice =
      element_count() == 0 ? data_ : data_ + start * strides_[axis] * SizeOf(dtype_);
  return TensorView(start_of_slice, dtype_, sliced, strides_);
}

StatusOr<TensorView> TensorView::Transpose(std::size_t axis_a, std::size_t axis_b) const {
  if (axis_a >= rank() || axis_b >= rank()) {
    return Status::InvalidArgument("cannot swap dimensions " + std::to_string(axis_a) + " and " +
                                   std::to_string(axis_b) + " of " + ToString());
  }
  Dims dims = CopyDims(shape_);
  Strides strides = strides_;
  std::swap(dims[axis_a], dims[axis_b]);
  std::swap(strides[axis_a], strides[axis_b]);
  EMBER_ASSIGN_OR_RETURN(const Shape swapped, Shape::Make(std::span(dims.data(), rank())));
  return TensorView(data_, dtype_, swapped, strides);
}

StatusOr<TensorView> TensorView::Reshape(Shape shape) const {
  if (shape.element_count() != element_count()) {
    return Status::InvalidArgument("cannot reshape " + ToString() + " to " + shape.ToString() +
                                   ", the element counts differ");
  }
  if (!is_contiguous()) {
    return Status::InvalidArgument("cannot reshape " + ToString() +
                                   " in place because it is not contiguous, copy it first");
  }
  return TensorView(data_, dtype_, shape, RowByRowStrides(shape));
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

StatusOr<Tensor> Tensor::Slice(std::size_t axis, std::size_t start, std::size_t length) const {
  EMBER_ASSIGN_OR_RETURN(const TensorView view, TensorView::Slice(axis, start, length));
  return Tensor(view);
}

StatusOr<Tensor> Tensor::Transpose(std::size_t axis_a, std::size_t axis_b) const {
  EMBER_ASSIGN_OR_RETURN(const TensorView view, TensorView::Transpose(axis_a, axis_b));
  return Tensor(view);
}

StatusOr<Tensor> Tensor::Reshape(Shape shape) const {
  EMBER_ASSIGN_OR_RETURN(const TensorView view, TensorView::Reshape(shape));
  return Tensor(view);
}

Status Tensor::CopyFrom(const TensorView& source) const {
  if (source.dtype() != dtype() || !(source.shape() == shape())) {
    return Status::InvalidArgument("cannot copy " + source.ToString() + " into " + ToString());
  }
  if (element_count() == 0) return Status::Ok();
  if (Overlaps(RangeOf(source), RangeOf(*this))) {
    return Status::InvalidArgument("cannot copy between tensors whose memory ranges intersect");
  }
  if (source.is_contiguous() && is_contiguous()) {
    std::memcpy(raw_data(), source.raw_data(), byte_size());
  } else {
    CopyAxis(source, *this, 0, source.raw_data(), raw_data());
  }
  return Status::Ok();
}

StatusOr<Tensor> Tensor::CopyOf(Arena& arena, const TensorView& source) {
  EMBER_ASSIGN_OR_RETURN(const Tensor copy, Uninitialized(arena, source.dtype(), source.shape()));
  EMBER_RETURN_IF_ERROR(copy.CopyFrom(source));
  return copy;
}

}  // namespace ember
