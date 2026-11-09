#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ember/arena.h"
#include "ember/mapped_file.h"
#include "ember/quant.h"
#include "ember/status_or.h"
#include "ember/tensor.h"

namespace ember {

// The shape of a Llama-style model, read from the config.json Hugging Face saves next to it.
struct LlamaConfig {
  std::size_t vocab_size = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t layer_count = 0;
  std::size_t head_count = 0;
  // Fewer key and value heads than query heads is grouped-query attention. Each key and value head
  // is shared by head_count / kv_head_count query heads.
  std::size_t kv_head_count = 0;
  std::size_t head_dim = 0;
  std::size_t max_positions = 0;
  float rms_norm_eps = 0;
  float rope_theta = 0;
  // The output layer reuses the token embedding instead of having weights of its own.
  bool tie_word_embeddings = false;
  // Qwen3: each head's queries and keys go through their own RMSNorm before RoPE.
  bool qk_norm = false;

  // Reads "llama" and "qwen3" configs. Refuses anything the engine cannot run exactly as Hugging
  // Face would, rather than running it slightly wrong: another architecture, another activation,
  // RoPE scaling, sliding-window attention, or sizes that do not divide evenly.
  static StatusOr<LlamaConfig> FromJson(std::string_view json);
};

// Each matrix is [out_features, in_features], as PyTorch stores a Linear layer. All f32 and
// contiguous, whatever the file held.
struct LlamaLayer {
  TensorView attention_norm;  // [hidden]
  TensorView query;           // [heads * head_dim, hidden]
  TensorView key;             // [kv_heads * head_dim, hidden]
  TensorView value;           // [kv_heads * head_dim, hidden]
  TensorView output;          // [hidden, heads * head_dim]
  TensorView mlp_norm;        // [hidden]
  TensorView gate;            // [intermediate, hidden]
  TensorView up;              // [intermediate, hidden]
  TensorView down;            // [hidden, intermediate]
  TensorView query_norm;      // [head_dim], only with qk_norm
  TensorView key_norm;        // [head_dim], only with qk_norm

  // With LoadOptions::q8, the seven matrices above live here in 8 bits instead, and their
  // TensorViews are empty.
  Q8Matrix query_q8, key_q8, value_q8, output_q8, gate_q8, up_q8, down_q8;
};

struct LlamaWeights {
  TensorView embedding;  // [vocab, hidden]
  std::vector<LlamaLayer> layers;
  TensorView final_norm;  // [hidden]
  TensorView lm_head;     // [vocab, hidden], the embedding itself when tied, empty with q8
  Q8Matrix lm_head_q8;    // with LoadOptions::q8
};

struct LoadOptions {
  // Store every matrix multiplied in generation (attention, MLP, and output layer) as 8 bits with a
  // scale per 32 weights. The embedding table and the norms stay f32.
  bool q8 = false;
};

// A loaded model: its config and every weight, checked against the config's shapes.
//
// f32 weights are read straight from the mapped file with no copy. f16 and bf16 weights are widened
// to f32 once, into memory the model owns, because every op works in f32. Either way the weights
// are read-only TensorViews.
class LlamaModel {
 public:
  // Reads config.json and model.safetensors from a directory, the layout Hugging Face downloads.
  static StatusOr<std::unique_ptr<LlamaModel>> Load(const std::string& directory,
                                                    const LoadOptions& options = {});

  // The same from bytes already in memory, which is how the browser will do it. `safetensors` is
  // not copied and must outlive the model.
  static StatusOr<std::unique_ptr<LlamaModel>> FromBytes(std::string_view config_json,
                                                         std::span<const std::byte> safetensors,
                                                         const LoadOptions& options = {});

  const LlamaConfig& config() const { return config_; }
  const LlamaWeights& weights() const { return weights_; }

  // Every number in the model, counting a tied embedding once.
  std::size_t parameter_count() const { return parameter_count_; }
  bool q8() const { return q8_; }
  // Bytes the model allocated for weights: f16 or bf16 widened to f32, and 8-bit copies. Zero for
  // an f32 file loaded as is.
  std::size_t widened_bytes() const { return arena_.used_bytes(); }

  LlamaModel(const LlamaModel&) = delete;
  LlamaModel& operator=(const LlamaModel&) = delete;

 private:
  LlamaModel() = default;
  Status Bind(std::span<const std::byte> safetensors, bool quantize);
  Status Finish(std::span<const std::byte> safetensors, const LoadOptions& options);
  std::size_t CountParameters() const;

  MappedFile file_;
  Arena arena_;
  LlamaConfig config_;
  LlamaWeights weights_;
  std::size_t parameter_count_ = 0;
  bool q8_ = false;
};

}  // namespace ember
