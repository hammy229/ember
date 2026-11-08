// libFuzzer entry point for the whole loader: the safetensors parser and then binding the weights
// to a fixed tiny Llama config. Any input must load or fail, never crash or read out of bounds.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "ember/llama.h"
#include "ember/safetensors.h"

namespace {

// The config the seeds in make_seeds.cpp are built for, so mutated seeds reach the weight checks.
constexpr const char* kConfig =
    R"({"model_type": "llama", "hidden_act": "silu", "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
        "vocab_size": 11, "hidden_size": 8, "intermediate_size": 12, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 2, "max_position_embeddings": 16,
        "tie_word_embeddings": true})";

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  // Copied to a buffer of exactly this size, so AddressSanitizer catches a read one byte past it.
  const std::vector<std::byte> file(reinterpret_cast<const std::byte*>(data),
                                    reinterpret_cast<const std::byte*>(data) + size);
  const auto parsed = ember::Safetensors::Parse(file);
  if (parsed.ok()) {
    for (const auto& entry : parsed->tensors()) {
      const auto view = ember::Safetensors::View(entry);
      // Touch the last byte, which must be inside the file.
      if (view.ok() && !entry.bytes.empty()) static_cast<void>(entry.bytes.back());
    }
  }
  static_cast<void>(ember::LlamaModel::FromBytes(kConfig, file));
  return 0;
}
