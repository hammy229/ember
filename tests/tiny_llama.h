#pragma once

// Makes small Llama models with random weights, so loader and forward pass tests need no download.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "safetensors_writer.h"

namespace ember::testing {

struct TinyLlamaOptions {
  std::size_t vocab = 11;
  std::size_t hidden = 8;
  std::size_t intermediate = 12;
  std::size_t layers = 2;
  std::size_t heads = 2;
  std::size_t kv_heads = 2;
  std::size_t max_positions = 16;
  bool tie = false;
  bool f16 = false;
  bool bf16 = false;
  std::uint32_t seed = 1;
};

// f32 to f16 for normal numbers, rounding toward zero. Tiny values become zero. Only for tests.
inline std::uint16_t ToF16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
  const auto exponent = static_cast<int>((bits >> 23) & 0xFF) - 112;
  if (exponent <= 0) return sign;
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7BFF);
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) |
                                    ((bits >> 13) & 0x3FFU));
}

inline std::string TinyLlamaConfig(const TinyLlamaOptions& o) {
  return R"({"model_type": "llama", "hidden_act": "silu", "rms_norm_eps": 1e-05,)"
         R"( "rope_theta": 10000.0, "rope_scaling": null, "vocab_size": )" +
         std::to_string(o.vocab) + R"(, "hidden_size": )" + std::to_string(o.hidden) +
         R"(, "intermediate_size": )" + std::to_string(o.intermediate) +
         R"(, "num_hidden_layers": )" + std::to_string(o.layers) + R"(, "num_attention_heads": )" +
         std::to_string(o.heads) + R"(, "num_key_value_heads": )" + std::to_string(o.kv_heads) +
         R"(, "max_position_embeddings": )" + std::to_string(o.max_positions) +
         R"(, "tie_word_embeddings": )" + (o.tie ? "true" : "false") + "}";
}

inline std::vector<std::byte> TinyLlamaWeights(const TinyLlamaOptions& o) {
  std::mt19937 random(o.seed);
  std::normal_distribution<float> normal(0.0F, 0.5F);
  SafetensorsWriter writer;
  const auto add = [&](const std::string& name, std::vector<std::size_t> shape, float center) {
    std::size_t count = 1;
    for (const std::size_t dim : shape) count *= dim;
    std::vector<float> values(count);
    for (float& value : values) value = center + normal(random);
    if (o.bf16) {
      std::vector<std::uint16_t> bits(count);
      for (std::size_t i = 0; i < count; ++i) {
        bits[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(values[i]) >> 16);
      }
      writer.AddBF16(name, shape, bits);
    } else if (o.f16) {
      std::vector<std::uint16_t> bits(count);
      for (std::size_t i = 0; i < count; ++i) bits[i] = ToF16(values[i]);
      writer.AddF16(name, shape, bits);
    } else {
      writer.AddF32(name, shape, values);
    }
  };
  const std::size_t head_dim = o.hidden / o.heads;
  add("model.embed_tokens.weight", {o.vocab, o.hidden}, 0);
  for (std::size_t i = 0; i < o.layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    // Norm weights sit near 1, as in a trained model.
    add(p + "input_layernorm.weight", {o.hidden}, 1);
    add(p + "self_attn.q_proj.weight", {o.heads * head_dim, o.hidden}, 0);
    add(p + "self_attn.k_proj.weight", {o.kv_heads * head_dim, o.hidden}, 0);
    add(p + "self_attn.v_proj.weight", {o.kv_heads * head_dim, o.hidden}, 0);
    add(p + "self_attn.o_proj.weight", {o.hidden, o.heads * head_dim}, 0);
    add(p + "post_attention_layernorm.weight", {o.hidden}, 1);
    add(p + "mlp.gate_proj.weight", {o.intermediate, o.hidden}, 0);
    add(p + "mlp.up_proj.weight", {o.intermediate, o.hidden}, 0);
    add(p + "mlp.down_proj.weight", {o.hidden, o.intermediate}, 0);
  }
  add("model.norm.weight", {o.hidden}, 1);
  if (!o.tie) add("lm_head.weight", {o.vocab, o.hidden}, 0);
  return writer.Build();
}

}  // namespace ember::testing
