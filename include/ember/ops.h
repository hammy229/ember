#pragma once

#include "ember/status.h"
#include "ember/tensor.h"

namespace ember {

// The math the forward pass is built from. Every op works on f32 tensors and follows their strides,
// so it takes a slice or a transpose as readily as a plain tensor. Outputs are passed in rather
// than allocated, so a caller can reuse scratch memory from an Arena.

// out[i] = a[i] + b[i]. All three must have the same shape.
Status Add(const TensorView& a, const TensorView& b, const Tensor& out);

// out[i] = a[i] * b[i]. All three must have the same shape.
Status Multiply(const TensorView& a, const TensorView& b, const Tensor& out);

// out[i] = a[i] * factor. Both must have the same shape.
Status Scale(const TensorView& a, float factor, const Tensor& out);

}  // namespace ember
