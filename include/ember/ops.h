#pragma once

#include <cstddef>

#include "ember/status.h"
#include "ember/tensor.h"

namespace ember {

// The math the forward pass is built from. Every op works on f32 tensors and follows their strides,
// so it takes a slice or a transpose as readily as a plain tensor. Outputs are passed in rather
// than allocated, so a caller can reuse scratch memory from an Arena.
//
// Add, Multiply, Scale, and Softmax can write exactly over an input. An output that only partly
// overlaps an input is refused, because the op would read numbers it had already overwritten and
// give a wrong answer without any sign of it.

// out[i] = a[i] + b[i]. All three must have the same shape.
Status Add(const TensorView& a, const TensorView& b, const Tensor& out);

// out[i] = a[i] * b[i]. All three must have the same shape.
Status Multiply(const TensorView& a, const TensorView& b, const Tensor& out);

// out[i] = a[i] * factor. Both must have the same shape.
Status Scale(const TensorView& a, float factor, const Tensor& out);

// Matrix product: [m, k] times [k, n] gives [m, n]. Whatever out held is replaced. Any dimensions
// before the last two are a batch, must match exactly, and each matrix in it is multiplied on its
// own, so [heads, tokens, features] times [heads, features, tokens] works. The output must not
// overlap either input at all, since each input element is read many times.
Status MatMul(const TensorView& a, const TensorView& b, const Tensor& out);

// Turns scores into chances that add up to 1 along one dimension. It stays accurate for very large
// or very small scores, and a score of negative infinity gets a chance of exactly zero, which is
// how attention hides a token.
Status Softmax(const TensorView& logits, std::size_t axis, const Tensor& out);

}  // namespace ember
