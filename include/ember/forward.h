#pragma once

#include <cstdint>
#include <span>

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
// This is the plain version that reprocesses the whole sequence on every call. The KV cache, which
// keeps each layer's keys and values so a new token only costs one row, comes next.
//
// Fails on an empty sequence, one longer than the model's context, or a token id outside its
// vocabulary.
StatusOr<Tensor> Forward(const LlamaModel& model, std::span<const std::int32_t> tokens,
                         Arena& arena);

}  // namespace ember
