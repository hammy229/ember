#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <string>

#include "ember/arena.h"
#include "ember/check.h"
#include "ember/dtype.h"
#include "ember/shape.h"
#include "ember/status_or.h"

namespace ember {

// A read-only look at a block of numbers: a pointer, an element type, a shape, and strides. It does
// not own the numbers, which live in an Arena or a memory-mapped model file, and it is cheap to
// copy. It is only valid as long as whatever owns the memory.
//
// Weights mapped from a file sit on read-only pages, and writing to one kills the process. Handing
// them out as TensorView makes that mistake a compile error.
class TensorView {
 public:
  // An empty f32 tensor of shape [0]. It exists so a struct can hold a tensor it fills in later.
  TensorView();

  // Looks at memory someone else owns, laid out row by row. Fails on a null pointer (unless there
  // are no elements) and on a pointer not aligned for the element type, since reading through one
  // is undefined behavior and the pointer may come from a file.
  static StatusOr<TensorView> Wrap(const void* data, DType dtype, Shape shape);

  // The same, for bytes whose size is known, and it also fails if the shape needs more bytes than
  // there are. Use this one for anything read from a file, because a corrupt file can claim a shape
  // far larger than the file itself.
  static StatusOr<TensorView> Wrap(std::span<const std::byte> bytes, DType dtype, Shape shape);

  DType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  std::size_t rank() const { return shape_.rank(); }
  std::size_t dim(std::size_t index) const { return shape_.dim(index); }
  std::size_t element_count() const { return shape_.element_count(); }
  std::size_t byte_size() const { return element_count() * SizeOf(dtype_); }

  // How many elements apart two neighbors along this dimension are in memory.
  std::size_t stride(std::size_t index) const;

  // The first element. Reading as the wrong type is a bug and aborts. Only walk it as a flat array
  // if is_contiguous(), otherwise use the strides.
  template <TensorElement T>
  const T* data() const {
    EMBER_CHECK(dtype_ == kDTypeOf<T>, "read a tensor as the wrong element type");
    return reinterpret_cast<const T*>(data_);
  }
  const std::byte* raw_data() const { return data_; }

  // One element by index. Bounds are checked in debug builds only, because this runs per element.
  template <TensorElement T, std::integral... Indices>
  const T& at(Indices... indices) const {
    EMBER_DCHECK(dtype_ == kDTypeOf<T>, "read a tensor as the wrong element type");
    return reinterpret_cast<const T*>(data_)[OffsetOf(indices...)];
  }

  // True if the elements sit next to each other in row by row order, so the tensor can be walked as
  // a flat array. Slicing a column or transposing can make this false.
  bool is_contiguous() const;

  // These give a new look at the same numbers. Nothing is copied, so they cost next to nothing.

  // Keeps `length` entries of one dimension, starting at `start`. Getting the last token out of
  // [tokens, features] is Slice(0, tokens - 1, 1).
  StatusOr<TensorView> Slice(std::size_t axis, std::size_t start, std::size_t length) const;

  // Swaps two dimensions. Attention turns [tokens, heads, features] into [heads, tokens, features]
  // this way, without moving a single number.
  StatusOr<TensorView> Transpose(std::size_t axis_a, std::size_t axis_b) const;

  // Reads the same numbers with another shape of the same element count. Only works on contiguous
  // tensors, because anything else would need the numbers moved. Copy it first in that case.
  StatusOr<TensorView> Reshape(Shape shape) const;

  // "f32[2, 3]".
  std::string ToString() const;

 protected:
  using Strides = std::array<std::size_t, Shape::kMaxRank>;

  TensorView(const std::byte* data, DType dtype, Shape shape, Strides strides)
      : data_(data), dtype_(dtype), shape_(shape), strides_(strides) {}

  // Elements from the first one to the one at these indices, along the strides.
  template <std::integral... Indices>
  std::size_t OffsetOf(Indices... indices) const {
    static_assert(sizeof...(Indices) <= Shape::kMaxRank, "more indices than a tensor can have");
    EMBER_DCHECK(sizeof...(Indices) == shape_.rank(), "wrong number of indices for this tensor");
    std::size_t offset = 0;
    std::size_t axis = 0;
    // A negative index turns into a huge size_t here and fails the bounds check.
    [[maybe_unused]] const auto add = [&](std::size_t index) {
      EMBER_DCHECK(index < shape_.dim(axis), "tensor index out of bounds");
      offset += index * strides_[axis];
      ++axis;
    };
    (add(static_cast<std::size_t>(indices)), ...);
    return offset;
  }

 private:
  const std::byte* data_;
  DType dtype_;
  Shape shape_;
  Strides strides_;
};

// A tensor whose numbers can be written. Like a std::span, it is a handle: copying one does not
// copy the numbers, and a const Tensor still points at writable numbers. It converts to a
// TensorView wherever only reading is needed.
class Tensor : public TensorView {
 public:
  Tensor() = default;

  // Numbers are garbage until written. Use this for an output that is about to be overwritten
  // anyway, since zeroing it first would be wasted work.
  static StatusOr<Tensor> Uninitialized(Arena& arena, DType dtype, Shape shape);
  static StatusOr<Tensor> Zeros(Arena& arena, DType dtype, Shape shape);

  // Copies the values in, which must match the shape's element count.
  template <TensorElement T>
  static StatusOr<Tensor> FromValues(Arena& arena, Shape shape, std::span<const T> values) {
    return FromBytes(arena, kDTypeOf<T>, shape, values.data(), values.size());
  }

  // Writes into memory someone else owns. Fails like TensorView::Wrap does.
  static StatusOr<Tensor> Wrap(void* data, DType dtype, Shape shape);

  template <TensorElement T>
  T* data() const {
    return const_cast<T*>(TensorView::data<T>());
  }
  std::byte* raw_data() const { return const_cast<std::byte*>(TensorView::raw_data()); }

  template <TensorElement T, std::integral... Indices>
  T& at(Indices... indices) const {
    return const_cast<T&>(TensorView::at<T>(indices...));
  }

  // Writable versions of the views above. They still share numbers with this tensor.
  StatusOr<Tensor> Slice(std::size_t axis, std::size_t start, std::size_t length) const;
  StatusOr<Tensor> Transpose(std::size_t axis_a, std::size_t axis_b) const;
  StatusOr<Tensor> Reshape(Shape shape) const;

  // Copies the numbers of a tensor with the same type and shape into this one, following both sets
  // of strides. Fails if the two overlap in memory, because transposing a matrix onto itself this
  // way would overwrite numbers before reading them and silently give the wrong answer.
  Status CopyFrom(const TensorView& source) const;

  // A new contiguous tensor holding a copy of the numbers. This is how to reshape a tensor that is
  // not contiguous.
  static StatusOr<Tensor> CopyOf(Arena& arena, const TensorView& source);

  // Sets every element. Using the wrong element type is a bug and aborts.
  template <TensorElement T>
  void Fill(T value) const {
    EMBER_CHECK(dtype() == kDTypeOf<T>, "filled a tensor with the wrong element type");
    FillAxis(0, data<T>(), value);
  }

 private:
  template <TensorElement T>
  void FillAxis(std::size_t axis, T* first, T value) const {
    if (element_count() == 0) return;
    if (axis == rank()) {
      *first = value;
      return;
    }
    for (std::size_t i = 0; i < dim(axis); ++i) FillAxis(axis + 1, first + i * stride(axis), value);
  }

  // Every Tensor comes from writable memory, which is what makes the const_casts above safe.
  explicit Tensor(TensorView view) : TensorView(view) {}

  static StatusOr<Tensor> FromBytes(Arena& arena, DType dtype, Shape shape, const void* values,
                                    std::size_t value_count);
};

}  // namespace ember
