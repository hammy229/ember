#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ember/arena.h"
#include "ember/mapped_file.h"
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

  // Refuses anything the engine cannot run exactly as Hugging Face would, rather than running it
  // slightly wrong: another architecture, another activation, RoPE scaling, or sizes that do not
  // divide evenly.
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
};

struct LlamaWeights {
  TensorView embedding;  // [vocab, hidden]
  std::vector<LlamaLayer> layers;
  TensorView final_norm;  // [hidden]
  TensorView lm_head;     // [vocab, hidden], the embedding itself when tied
};

// A loaded model: its config and every weight, checked against the config's shapes.
//
// f32 weights are read straight from the mapped file with no copy. f16 weights are widened to f32
// once, into memory the model owns, because every op works in f32. Either way the weights are
// read-only TensorViews.
class LlamaModel {
 public:
  // Reads config.json and model.safetensors from a directory, the layout Hugging Face downloads.
  static StatusOr<std::unique_ptr<LlamaModel>> Load(const std::string& directory);

  // The same from bytes already in memory, which is how the browser will do it. `safetensors` is
  // not copied and must outlive the model.
  static StatusOr<std::unique_ptr<LlamaModel>> FromBytes(std::string_view config_json,
                                                         std::span<const std::byte> safetensors);

  const LlamaConfig& config() const { return config_; }
  const LlamaWeights& weights() const { return weights_; }

  // Every number in the model, counting a tied embedding once.
  std::size_t parameter_count() const;
  // Bytes the model allocated to widen f16 weights. Zero for an f32 file.
  std::size_t widened_bytes() const { return arena_.used_bytes(); }

  LlamaModel(const LlamaModel&) = delete;
  LlamaModel& operator=(const LlamaModel&) = delete;

 private:
  LlamaModel() = default;
  Status Bind(std::span<const std::byte> safetensors);

  MappedFile file_;
  Arena arena_;
  LlamaConfig config_;
  LlamaWeights weights_;
};

}  // namespace ember
