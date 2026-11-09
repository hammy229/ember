#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "ember/arena.h"
#include "ember/llama.h"
#include "ember/status_or.h"
#include "ember/tensor.h"

namespace ember {

// Runs the whole model on a sequence of token ids and gives back logits [tokens, vocab]: row i
// scores every possible next token after tokens[0..i]. The logits live in the arena. Every
// scratch tensor along the way is given back before returning, so the arena ends up holding only
// the logits.
//
// This version reprocesses the whole sequence on every call. ForwardCached below keeps each layer's
// keys and values, so a new token costs one row instead of the whole sequence.
//
// Fails on an empty sequence, one longer than the model's context, or a token id outside its
// vocabulary.
StatusOr<Tensor> Forward(const LlamaModel& model, std::span<const std::int32_t> tokens,
                         Arena& arena);

// Every layer's keys and values for the tokens seen so far, so generation does not recompute them.
// Memory for the full capacity is taken once, up front: 2 * layers * capacity * kv_heads * head_dim
// floats.
class KvCache {
 public:
  // Fails if capacity is zero or more than the model's context, or if the memory is not there.
  static StatusOr<std::unique_ptr<KvCache>> Make(const LlamaConfig& config, std::size_t capacity);

  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;

  // How many positions hold keys and values.
  std::size_t size() const { return size_; }
  std::size_t capacity() const { return capacity_; }
  std::size_t bytes() const { return bytes_; }
  // Forgets every position, keeping the memory.
  void Clear() { size_ = 0; }
  // Forgets every position from `size` on, so a prompt prefix can be reused.
  void Truncate(std::size_t size) {
    if (size < size_) size_ = size;
  }

 private:
  friend StatusOr<Tensor> ForwardCached(const LlamaModel&, std::span<const std::int32_t>, KvCache&,
                                        Arena&, bool);
  KvCache() = default;

  Arena arena_;
  std::vector<Tensor> keys_;    // per layer, [capacity, kv_heads, head_dim]
  std::vector<Tensor> values_;  // per layer, [capacity, kv_heads, head_dim]
  std::size_t layers_ = 0;
  std::size_t kv_heads_ = 0;
  std::size_t head_dim_ = 0;
  std::size_t capacity_ = 0;
  std::size_t size_ = 0;
  std::size_t bytes_ = 0;
};

// Runs `tokens` as positions cache.size() onward, adds their keys and values to the cache, and
// gives back logits for them: [tokens, vocab], or just [1, vocab] for the last token when
// `last_only` is set, which is all generation needs and skips the biggest matrix for every other
// row.
//
// The math is Forward's, so a prompt run in one call or one token at a time gives the same logits
// up to float rounding (checked in tests/forward_test.cpp). Fails if the cache would overflow, and
// then leaves it as it was.
StatusOr<Tensor> ForwardCached(const LlamaModel& model, std::span<const std::int32_t> tokens,
                               KvCache& cache, Arena& arena, bool last_only);

}  // namespace ember
