#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <string>

#include "ember/status_or.h"

namespace ember {

// The size of each dimension of a tensor. Valid by construction: a Shape that exists is guaranteed
// to fit in memory arithmetic, so code that uses one never has to worry about overflow.
//
// Sizes in a model file are untrusted. A corrupt file saying [2^32, 2^32] would multiply to zero on
// a 64-bit machine and to garbage on 32-bit WebAssembly, and then the engine would read far past a
// small buffer. Make() is where that gets stopped.
class Shape {
 public:
  // [batch, heads, tokens, features] is the most a transformer needs.
  static constexpr std::size_t kMaxRank = 4;
  // Any element count up to this, times the largest element size, still fits in a size_t.
  static constexpr std::size_t kMaxElements = std::numeric_limits<std::size_t>::max() / 8;

  // A scalar: no dimensions and one element.
  Shape() = default;

  // Dimensions may be zero, which gives an empty tensor. An empty cache of past tokens is one.
  static StatusOr<Shape> Make(std::span<const std::size_t> dims);
  static StatusOr<Shape> Make(std::initializer_list<std::size_t> dims);

  std::size_t rank() const {
    // Make() never builds a bigger shape. Saying so lets the optimizer see that a loop over the
    // dimensions stays inside the arrays. Without it GCC 15 warns about index 4 of a 4 element
    // array on code that can never get there.
    if (rank_ > kMaxRank) __builtin_unreachable();
    return rank_;
  }
  std::span<const std::size_t> dims() const { return {dims_.data(), rank_}; }
  std::size_t element_count() const { return element_count_; }

  // Size of one dimension. An index past the rank is a bug and aborts.
  std::size_t dim(std::size_t index) const;

  // "[2, 3]", or "[]" for a scalar.
  std::string ToString() const;

  friend bool operator==(const Shape& a, const Shape& b) {
    return a.rank_ == b.rank_ && a.dims_ == b.dims_;
  }

 private:
  std::array<std::size_t, kMaxRank> dims_{};
  std::size_t rank_ = 0;
  std::size_t element_count_ = 1;
};

}  // namespace ember
