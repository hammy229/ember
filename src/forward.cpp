#include "ember/forward.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
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

// Where a layer keeps its keys and values across calls. Without one, a layer attends only to the
// tokens it was given.
struct LayerCache {
  const Tensor* keys;    // [capacity, kv_heads, head_dim]
  const Tensor* values;  // [capacity, kv_heads, head_dim]
};

// One transformer layer on tokens at positions first_position onward, updating the residual stream
// x [tokens, hidden] in place:
//
//   x = x + Attention(RmsNorm(x))
//   x = x + Down(Silu(Gate(RmsNorm(x))) * Up(RmsNorm(x)))
Status RunLayer(const LlamaConfig& c, const LlamaLayer& layer, const Tensor& x,
                std::size_t first_position, const LayerCache* cache, Arena& arena) {
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
  EMBER_RETURN_IF_ERROR(ApplyRope(q3, first_position, c.rope_theta));
  EMBER_RETURN_IF_ERROR(ApplyRope(k3, first_position, c.rope_theta));

  EMBER_ASSIGN_OR_RETURN(const Tensor attended, Tensor::Uninitialized(arena, DType::kF32, q_heads));
  if (cache == nullptr) {
    EMBER_RETURN_IF_ERROR(CausalAttention(q3, k3, v3, attended, arena));
  } else {
    // Store this call's keys and values after the ones already cached, then attend to all of them.
    const std::size_t end = first_position + tokens;
    EMBER_ASSIGN_OR_RETURN(const Tensor k_slot, cache->keys->Slice(0, first_position, tokens));
    EMBER_ASSIGN_OR_RETURN(const Tensor v_slot, cache->values->Slice(0, first_position, tokens));
    std::copy_n(k3.data<float>(), k3.element_count(), k_slot.data<float>());
    std::copy_n(v3.data<float>(), v3.element_count(), v_slot.data<float>());
    EMBER_ASSIGN_OR_RETURN(const Tensor all_keys, cache->keys->Slice(0, 0, end));
    EMBER_ASSIGN_OR_RETURN(const Tensor all_values, cache->values->Slice(0, 0, end));
    EMBER_RETURN_IF_ERROR(CausalAttention(q3, all_keys, all_values, attended, arena));
  }
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
    for (const LlamaLayer& layer : w.layers) {
      EMBER_RETURN_IF_ERROR(RunLayer(c, layer, x, 0, nullptr, arena));
    }
    EMBER_RETURN_IF_ERROR(RmsNorm(x, w.final_norm, c.rms_norm_eps, x));
    EMBER_RETURN_IF_ERROR(Linear(x, w.lm_head, logits));
  }
  return logits;
}

StatusOr<std::unique_ptr<KvCache>> KvCache::Make(const LlamaConfig& config, std::size_t capacity) {
  if (capacity == 0 || capacity > config.max_positions) {
    return Status::InvalidArgument("a KV cache of " + std::to_string(capacity) +
                                   " positions does not fit the model's context of " +
                                   std::to_string(config.max_positions));
  }
  std::unique_ptr<KvCache> cache(new KvCache());
  cache->layers_ = config.layer_count;
  cache->kv_heads_ = config.kv_head_count;
  cache->head_dim_ = config.head_dim;
  cache->capacity_ = capacity;
  EMBER_ASSIGN_OR_RETURN(const Shape shape,
                         Shape::Make({capacity, config.kv_head_count, config.head_dim}));
  for (std::size_t i = 0; i < config.layer_count; ++i) {
    EMBER_ASSIGN_OR_RETURN(Tensor keys, Tensor::Uninitialized(cache->arena_, DType::kF32, shape));
    EMBER_ASSIGN_OR_RETURN(Tensor values, Tensor::Uninitialized(cache->arena_, DType::kF32, shape));
    cache->bytes_ += 2 * keys.element_count() * sizeof(float);
    cache->keys_.push_back(std::move(keys));
    cache->values_.push_back(std::move(values));
  }
  return cache;
}

StatusOr<Tensor> ForwardCached(const LlamaModel& model, std::span<const std::int32_t> tokens,
                               KvCache& cache, Arena& arena, bool last_only) {
  const LlamaConfig& c = model.config();
  const LlamaWeights& w = model.weights();
  if (tokens.empty()) return Status::InvalidArgument("ForwardCached needs at least one token");
  if (cache.layers_ != c.layer_count || cache.kv_heads_ != c.kv_head_count ||
      cache.head_dim_ != c.head_dim) {
    return Status::InvalidArgument("the KV cache was made for a different model");
  }
  if (tokens.size() > cache.capacity_ - cache.size_) {
    return Status::InvalidArgument(
        std::to_string(tokens.size()) + " more tokens overflow a cache of " +
        std::to_string(cache.capacity_) + " holding " + std::to_string(cache.size_));
  }
  for (const std::int32_t token : tokens) {
    if (token < 0 || static_cast<std::size_t>(token) >= c.vocab_size) {
      return Status::InvalidArgument("token id " + std::to_string(token) +
                                     " is outside the vocabulary");
    }
  }

  const std::size_t rows = last_only ? 1 : tokens.size();
  EMBER_ASSIGN_OR_RETURN(const Tensor logits, Scratch(arena, {rows, c.vocab_size}));
  {
    const ArenaScope scope(arena);
    EMBER_ASSIGN_OR_RETURN(const Tensor x, Scratch(arena, {tokens.size(), c.hidden_size}));
    EMBER_RETURN_IF_ERROR(Embed(w.embedding, tokens, x));
    for (std::size_t i = 0; i < c.layer_count; ++i) {
      const LayerCache layer_cache{&cache.keys_[i], &cache.values_[i]};
      EMBER_RETURN_IF_ERROR(RunLayer(c, w.layers[i], x, cache.size_, &layer_cache, arena));
    }
    EMBER_ASSIGN_OR_RETURN(const Tensor tail, x.Slice(0, tokens.size() - rows, rows));
    EMBER_RETURN_IF_ERROR(RmsNorm(tail, w.final_norm, c.rms_norm_eps, tail));
    EMBER_RETURN_IF_ERROR(Linear(tail, w.lm_head, logits));
  }
  cache.size_ += tokens.size();
  return logits;
}

}  // namespace ember
