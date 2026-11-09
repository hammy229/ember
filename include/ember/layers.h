#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "ember/arena.h"
#include "ember/status.h"
#include "ember/tensor.h"

namespace ember {

// The pieces a Llama layer is built from, on top of the general math in ops.h. Each one follows
// Hugging Face's Llama code step by step, including where it rounds to f32, because the goal is
// logits that match it, not just logits that look reasonable.
//
// Unlike the ops in ops.h these want contiguous tensors, which is what the forward pass has, and
// refuse anything else rather than run a slow path nobody tests.

// A Linear layer with no bias: x [rows, in] times weight [out_features, in] turned sideways, giving
// out [rows, out_features]. That is how PyTorch stores the weight, and reading each weight row in a
// straight line is why this is not MatMul with a transposed view. out must not overlap the inputs.
Status Linear(const TensorView& x, const TensorView& weight, const Tensor& out);

// Root mean square norm over the last dimension: each row of x [rows, size] is divided by the
// square root of (mean of its squares + eps), then multiplied by weight [size] element by element.
// out may be x itself.
Status RmsNorm(const TensorView& x, const TensorView& weight, float eps, const Tensor& out);

// Looks up one row of table [vocab, size] per token into out [tokens.size(), size]. A token id
// outside the table is an error, not a read past the end.
Status Embed(const TensorView& table, std::span<const std::int32_t> tokens, const Tensor& out);

// Rotary position embedding, in place, on x [tokens, heads, head_dim]. Token i is at position
// first_position + i. Uses Hugging Face's layout, which rotates element j with element
// j + head_dim / 2 rather than with its neighbor.
Status ApplyRope(const Tensor& x, std::size_t first_position, float theta);

// Causal self-attention for one layer. q is [queries, heads, head_dim]; k and v are [keys,
// kv_heads, head_dim] with keys >= queries, and the queries are the last `queries` of the keys'
// positions, so query i sees keys 0 through i + keys - queries. With fewer kv_heads than heads,
// each key and value head serves heads / kv_heads query heads in a row. out is [queries, heads,
// head_dim] and must not overlap any input. Scratch memory comes from the arena and is given back
// before returning.
//
// If `probabilities` is given, each head's attention weights after softmax are copied into it as
// [heads, queries, keys], for looking inside the model.
Status CausalAttention(const TensorView& q, const TensorView& k, const TensorView& v,
                       const Tensor& out, Arena& arena, float* probabilities = nullptr);

}  // namespace ember
