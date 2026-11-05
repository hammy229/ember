#include "ember/shape.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>

namespace ember {

StatusOr<Shape> Shape::Make(std::span<const std::size_t> dims) {
  if (dims.size() > kMaxRank) {
    return Status::InvalidArgument("a shape can have at most " + std::to_string(kMaxRank) +
                                   " dimensions, got " + std::to_string(dims.size()));
  }

  // Multiply only the nonzero sizes, so [0, 2^40, 2^40] is rejected too. It holds no elements, but
  // its strides would still overflow.
  std::size_t nonzero_product = 1;
  bool has_zero = false;
  for (const std::size_t dim : dims) {
    if (dim == 0) {
      has_zero = true;
      continue;
    }
    if (dim > kMaxElements / nonzero_product) {
      return Status::InvalidArgument("a shape has too many elements to fit in memory");
    }
    nonzero_product *= dim;
  }

  Shape shape;
  std::copy(dims.begin(), dims.end(), shape.dims_.begin());
  shape.rank_ = dims.size();
  shape.element_count_ = has_zero ? 0 : nonzero_product;
  return shape;
}

StatusOr<Shape> Shape::Make(std::initializer_list<std::size_t> dims) {
  return Make(std::span<const std::size_t>(dims.begin(), dims.size()));
}

std::size_t Shape::dim(std::size_t index) const {
  EMBER_CHECK(index < rank_, "asked for a dimension past the end of a shape");
  return dims_[index];
}

std::string Shape::ToString() const {
  std::string out = "[";
  for (std::size_t i = 0; i < rank_; ++i) {
    if (i > 0) out += ", ";
    out += std::to_string(dims_[i]);
  }
  out += "]";
  return out;
}

}  // namespace ember
