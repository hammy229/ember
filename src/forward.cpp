#include "ember/forward.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>

#include "ember/layers.h"
#include "ember/ops.h"

namespace ember {

namespace {

StatusOr<Tensor> Scratch(Arena& arena, std::initializer_list<std::size_t> dims) {
  EMBER_ASSIGN_OR_RETURN(const Shape shape, Shape::Make(dims));
  return Tensor::Uninitialized(arena, DType::kF32, shape);
}

// One transformer layer, updating the residual stream x [tokens, hidden] in place:
//
//   x = x + Attention(RmsNorm(x))
//   x = x + Down(Silu(Gate(RmsNorm(x))) * Up(RmsNorm(x)))
Status RunLayer(const LlamaConfig& c, const LlamaLayer& layer, const Tensor& x, Arena& arena) {
  const ArenaScope scope(arena);
  const std::size_t tokens = x.dim(0);
  const std::size_t q_width = c.head_count * c.head_dim;
  const std::size_t kv_width = c.kv_head_count * c.head_dim;

  EMBER_ASSIGN_OR_RETURN(const Tensor normed, Scratch(arena, {tokens, c.hidden_size}));
  EMBER_RETURN_IF_ERROR(RmsNorm(x, layer.attention_norm, c.rms_norm_eps, normed));

  EMBER_ASSIGN_OR_RETURN(const Tensor q, Scratch(arena, {tokens, q_width}));
  EMBER_ASSIGN_OR_RETURN(const Tensor k, Scratch(arena, {tokens, kv_width}));
  EMBER_ASSIGN_OR_RETURN(const Tensor v, Scratch(arena, {tokens, kv_width}));
  EMBER_RETURN_IF_ERROR(Linear(normed, layer.query, q));
  EMBER_RETURN_IF_ERROR(Linear(normed, layer.key, k));
  EMBER_RETURN_IF_ERROR(Linear(normed, layer.value, v));

  // The same numbers seen as [tokens, heads, head_dim], so each head is its own slice.
  EMBER_ASSIGN_OR_RETURN(const Shape q_heads, Shape::Make({tokens, c.head_count, c.head_dim}));
  EMBER_ASSIGN_OR_RETURN(const Shape kv_heads, Shape::Make({tokens, c.kv_head_count, c.head_dim}));
  EMBER_ASSIGN_OR_RETURN(const Tensor q3, q.Reshape(q_heads));
  EMBER_ASSIGN_OR_RETURN(const Tensor k3, k.Reshape(kv_heads));
  EMBER_ASSIGN_OR_RETURN(const Tensor v3, v.Reshape(kv_heads));
  EMBER_RETURN_IF_ERROR(ApplyRope(q3, 0, c.rope_theta));
  EMBER_RETURN_IF_ERROR(ApplyRope(k3, 0, c.rope_theta));

  EMBER_ASSIGN_OR_RETURN(const Tensor attended, Tensor::Uninitialized(arena, DType::kF32, q_heads));
  EMBER_RETURN_IF_ERROR(CausalAttention(q3, k3, v3, attended, arena));
  EMBER_ASSIGN_OR_RETURN(const Shape flat, Shape::Make({tokens, q_width}));
  EMBER_ASSIGN_OR_RETURN(const Tensor attended_flat, attended.Reshape(flat));

  // The attention output reuses `normed`, which is no longer needed and has the right shape.
  EMBER_RETURN_IF_ERROR(Linear(attended_flat, layer.output, normed));
  EMBER_RETURN_IF_ERROR(Add(x, normed, x));

  EMBER_RETURN_IF_ERROR(RmsNorm(x, layer.mlp_norm, c.rms_norm_eps, normed));
  EMBER_ASSIGN_OR_RETURN(const Tensor gate, Scratch(arena, {tokens, c.intermediate_size}));
  EMBER_ASSIGN_OR_RETURN(const Tensor up, Scratch(arena, {tokens, c.intermediate_size}));
  EMBER_RETURN_IF_ERROR(Linear(normed, layer.gate, gate));
  EMBER_RETURN_IF_ERROR(Linear(normed, layer.up, up));
  EMBER_RETURN_IF_ERROR(Silu(gate, gate));
  EMBER_RETURN_IF_ERROR(Multiply(gate, up, gate));
  EMBER_RETURN_IF_ERROR(Linear(gate, layer.down, normed));
  return Add(x, normed, x);
}

}  // namespace

StatusOr<Tensor> Forward(const LlamaModel& model, std::span<const std::int32_t> tokens,
                         Arena& arena) {
  const LlamaConfig& c = model.config();
  const LlamaWeights& w = model.weights();
  if (tokens.empty()) return Status::InvalidArgument("Forward needs at least one token");
  if (tokens.size() > c.max_positions) {
    return Status::InvalidArgument(std::to_string(tokens.size()) +
                                   " tokens is more than the model's context of " +
                                   std::to_string(c.max_positions));
  }

  // The logits come first, so rewinding past the scratch leaves them in place.
  EMBER_ASSIGN_OR_RETURN(const Tensor logits, Scratch(arena, {tokens.size(), c.vocab_size}));
  {
    const ArenaScope scope(arena);
    EMBER_ASSIGN_OR_RETURN(const Tensor x, Scratch(arena, {tokens.size(), c.hidden_size}));
    EMBER_RETURN_IF_ERROR(Embed(w.embedding, tokens, x));
    for (const LlamaLayer& layer : w.layers) EMBER_RETURN_IF_ERROR(RunLayer(c, layer, x, arena));
    EMBER_RETURN_IF_ERROR(RmsNorm(x, w.final_norm, c.rms_norm_eps, x));
    EMBER_RETURN_IF_ERROR(Linear(x, w.lm_head, logits));
  }
  return logits;
}

}  // namespace ember
