#pragma once

#include "ember/status.h"
#include "ember/tensor.h"

namespace ember {

// The math the forward pass is built from. Every op works on f32 tensors and follows their strides,
// so it takes a slice or a transpose as readily as a plain tensor. Outputs are passed in rather
// than allocated, so a caller can reuse scratch memory from an Arena.
//
// Add, Multiply, and Scale can write exactly over an input. An output that only partly overlaps an
// input is refused, because the op would read numbers it had already overwritten and give a wrong
// answer without any sign of it.

// out[i] = a[i] + b[i]. All three must have the same shape.
Status Add(const TensorView& a, const TensorView& b, const Tensor& out);

// out[i] = a[i] * b[i]. All three must have the same shape.
Status Multiply(const TensorView& a, const TensorView& b, const Tensor& out);

// out[i] = a[i] * factor. Both must have the same shape.
Status Scale(const TensorView& a, float factor, const Tensor& out);

// Matrix product: [m, k] times [k, n] gives [m, n]. Whatever out held is replaced. Any dimensions
// before the last two are a batch, must match exactly, and each matrix in it is multiplied on its
// own, so [heads, tokens, features] times [heads, features, tokens] works.
Status MatMul(const TensorView& a, const TensorView& b, const Tensor& out);

}  // namespace ember
