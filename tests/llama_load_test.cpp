#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "ember/f16.h"
#include "ember/llama.h"
#include "safetensors_writer.h"
#include "tiny_llama.h"

namespace {

using ember::LlamaConfig;
using ember::LlamaModel;
using ember::StatusCode;
using ember::testing::SafetensorsWriter;
using ember::testing::TinyLlamaConfig;
using ember::testing::TinyLlamaOptions;
using ember::testing::TinyLlamaWeights;

// The tiny model's real config.json, as Hugging Face saved it.
constexpr const char* kStories15MConfig = R"({
  "architectures": ["LlamaForCausalLM"], "bos_token_id": 1, "eos_token_id": 2,
  "hidden_act": "silu", "hidden_size": 288, "initializer_range": 0.02,
  "intermediate_size": 768, "max_position_embeddings": 256, "model_type": "llama",
  "num_attention_heads": 6, "num_hidden_layers": 6, "num_key_value_heads": 6,
  "pretraining_tp": 1, "rms_norm_eps": 1e-05, "rope_scaling": null, "rope_theta": 10000.0,
  "tie_word_embeddings": true, "transformers_version": "4.33.1", "use_cache": true,
  "vocab_size": 32000})";

TEST(LlamaConfig, ReadsARealConfig) {
  const auto config = LlamaConfig::FromJson(kStories15MConfig);
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  EXPECT_EQ(config->vocab_size, 32000U);
  EXPECT_EQ(config->hidden_size, 288U);
  EXPECT_EQ(config->intermediate_size, 768U);
  EXPECT_EQ(config->layer_count, 6U);
  EXPECT_EQ(config->head_count, 6U);
  EXPECT_EQ(config->kv_head_count, 6U);
  EXPECT_EQ(config->head_dim, 48U);
  EXPECT_EQ(config->max_positions, 256U);
  EXPECT_FLOAT_EQ(config->rms_norm_eps, 1e-5F);
  EXPECT_FLOAT_EQ(config->rope_theta, 10000.0F);
  EXPECT_TRUE(config->tie_word_embeddings);
}

// The config with one field replaced, given as `"key": value`.
std::string With(const std::string& replacement) {
  std::string config = TinyLlamaConfig({});
  const std::string key = replacement.substr(0, replacement.find(':'));
  const std::size_t start = config.find(key);
  if (start == std::string::npos)
    return config.substr(0, config.size() - 1) + ", " + replacement + "}";
  std::size_t end = config.find_first_of(",}", start);
  return config.replace(start, end - start, replacement);
}

bool Refused(const std::string& json) {
  const auto config = LlamaConfig::FromJson(json);
  return !config.ok() && config.status().code() == StatusCode::kInvalidFormat;
}

TEST(LlamaConfig, RefusesWhatItCannotRunExactly) {
  EXPECT_FALSE(Refused(TinyLlamaConfig({})));
  EXPECT_TRUE(Refused("[]"));
  EXPECT_TRUE(Refused(With(R"("model_type": "gpt2")")));
  EXPECT_TRUE(Refused(With(R"("hidden_act": "gelu")")));
  EXPECT_TRUE(Refused(With(R"("rope_scaling": {"type": "linear"})")));
  EXPECT_TRUE(Refused(With(R"("attention_bias": true)")));
  EXPECT_TRUE(Refused(With(R"("vocab_size": 0)")));
  EXPECT_TRUE(Refused(With(R"("vocab_size": -3)")));
  EXPECT_TRUE(Refused(With(R"("vocab_size": 2.5)")));
  EXPECT_TRUE(Refused(With(R"("hidden_size": 18446744073709551615)")));
  EXPECT_TRUE(Refused(With(R"("num_attention_heads": 3)")));  // 8 does not divide by 3
  EXPECT_TRUE(Refused(With(R"("num_key_value_heads": 3)")));
  EXPECT_TRUE(Refused(With(R"("hidden_size": 6)")));  // head_dim 3 is odd
  EXPECT_TRUE(Refused(With(R"("rms_norm_eps": 0)")));
  EXPECT_TRUE(Refused(With(R"("rope_theta": -1)")));
  EXPECT_TRUE(Refused(With(R"("tie_word_embeddings": 1)")));
}

TEST(LlamaConfig, AllowsGroupedQueryAttentionAndAnExplicitHeadDim) {
  const auto grouped = LlamaConfig::FromJson(With(R"("num_key_value_heads": 1)"));
  ASSERT_TRUE(grouped.ok()) << grouped.status().ToString();
  EXPECT_EQ(grouped->kv_head_count, 1U);
  const auto explicit_dim = LlamaConfig::FromJson(With(R"("head_dim": 6)"));
  ASSERT_TRUE(explicit_dim.ok()) << explicit_dim.status().ToString();
  EXPECT_EQ(explicit_dim->head_dim, 6U);
}

TEST(LlamaModel, LoadsEveryWeightWithItsShape) {
  const TinyLlamaOptions options;
  const std::vector<std::byte> weights = TinyLlamaWeights(options);
  const auto model = LlamaModel::FromBytes(TinyLlamaConfig(options), weights);
  ASSERT_TRUE(model.ok()) << model.status().ToString();
  const auto& w = (*model)->weights();
  ASSERT_EQ(w.layers.size(), 2U);
  EXPECT_EQ(w.embedding.ToString(), "f32[11, 8]");
  EXPECT_EQ(w.layers[1].query.ToString(), "f32[8, 8]");
  EXPECT_EQ(w.layers[1].down.ToString(), "f32[8, 12]");
  EXPECT_EQ(w.lm_head.ToString(), "f32[11, 8]");
  EXPECT_NE(w.lm_head.raw_data(), w.embedding.raw_data());
  // f32 weights point straight into the file.
  EXPECT_GE(w.embedding.raw_data(), weights.data());
  EXPECT_LT(w.embedding.raw_data(), weights.data() + weights.size());
  EXPECT_EQ((*model)->widened_bytes(), 0U);
  // 11*8*2 embeddings and head, 2 layers of 2*8 norms + 4*64 attention + 3*96 MLP, 8 final norm.
  EXPECT_EQ((*model)->parameter_count(), 176U + 2 * (16 + 256 + 288) + 8);
}

TEST(LlamaModel, TiedEmbeddingsShareOneMatrix) {
  TinyLlamaOptions options;
  options.tie = true;
  const std::vector<std::byte> weights = TinyLlamaWeights(options);
  const auto model = LlamaModel::FromBytes(TinyLlamaConfig(options), weights);
  ASSERT_TRUE(model.ok()) << model.status().ToString();
  EXPECT_EQ((*model)->weights().lm_head.raw_data(), (*model)->weights().embedding.raw_data());
}

TEST(LlamaModel, WidensF16WeightsExactly) {
  TinyLlamaOptions f16;
  f16.f16 = true;
  const std::vector<std::byte> weights = TinyLlamaWeights(f16);
  const auto model = LlamaModel::FromBytes(TinyLlamaConfig(f16), weights);
  ASSERT_TRUE(model.ok()) << model.status().ToString();
  EXPECT_GT((*model)->widened_bytes(), 0U);
  // The same seed in f32 gives the values before rounding to f16.
  const std::vector<std::byte> exact = TinyLlamaWeights({});
  const auto reference = LlamaModel::FromBytes(TinyLlamaConfig({}), exact);
  ASSERT_TRUE(reference.ok());
  const auto& widened = (*model)->weights().layers[0].gate;
  const auto& original = (*reference)->weights().layers[0].gate;
  for (std::size_t i = 0; i < widened.element_count(); ++i) {
    const float before = original.data<float>()[i];
    EXPECT_EQ(widened.data<float>()[i], ember::F16ToF32(ember::testing::ToF16(before)));
    EXPECT_NEAR(widened.data<float>()[i], before, std::abs(before) / 512 + 1e-4F);
  }
}

TEST(LlamaModel, WidensBF16WeightsExactly) {
  TinyLlamaOptions bf16;
  bf16.bf16 = true;
  const std::vector<std::byte> weights = TinyLlamaWeights(bf16);
  const auto model = LlamaModel::FromBytes(TinyLlamaConfig(bf16), weights);
  ASSERT_TRUE(model.ok()) << model.status().ToString();
  const std::vector<std::byte> exact = TinyLlamaWeights({});
  const auto reference = LlamaModel::FromBytes(TinyLlamaConfig({}), exact);
  ASSERT_TRUE(reference.ok());
  const auto& widened = (*model)->weights().layers[1].up;
  const auto& original = (*reference)->weights().layers[1].up;
  for (std::size_t i = 0; i < widened.element_count(); ++i) {
    // Cutting off the low 16 bits is what the file did, so that is what must come back.
    const auto cut = std::bit_cast<std::uint32_t>(original.data<float>()[i]) & 0xFFFF0000U;
    EXPECT_EQ(widened.data<float>()[i], std::bit_cast<float>(cut));
  }
}

TEST(LlamaModel, RefusesAMissingWeight) {
  SafetensorsWriter writer;
  writer.AddF32("model.embed_tokens.weight", {11, 8}, std::vector<float>(88));
  const std::vector<std::byte> weights = writer.Build();
  const auto model = LlamaModel::FromBytes(TinyLlamaConfig({}), weights);
  ASSERT_FALSE(model.ok());
  EXPECT_NE(model.status().message().find("missing weight model.layers.0.input_layernorm.weight"),
            std::string::npos)
      << model.status().message();
}

TEST(LlamaModel, RefusesAWeightOfTheWrongShape) {
  TinyLlamaOptions bigger;
  bigger.vocab = 12;
  const std::vector<std::byte> weights = TinyLlamaWeights(bigger);
  const auto model = LlamaModel::FromBytes(TinyLlamaConfig({}), weights);
  ASSERT_FALSE(model.ok());
  EXPECT_NE(model.status().message().find("[12, 8], expected [11, 8]"), std::string::npos)
      << model.status().message();
}

TEST(LlamaModel, RefusesWeightsFromADifferentArchitecture) {
  // Three layers in the file, two in the config.
  TinyLlamaOptions deeper;
  deeper.layers = 3;
  const std::vector<std::byte> weights = TinyLlamaWeights(deeper);
  const auto model = LlamaModel::FromBytes(TinyLlamaConfig({}), weights);
  ASSERT_FALSE(model.ok());
  EXPECT_NE(model.status().message().find("unexpected weight model.layers.2"), std::string::npos)
      << model.status().message();
}

TEST(LlamaModel, LoadsADirectoryFromDisk) {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() / "ember_llama_load_test";
  std::filesystem::create_directories(directory);
  const TinyLlamaOptions options;
  {
    std::ofstream(directory / "config.json") << TinyLlamaConfig(options);
    const std::vector<std::byte> weights = TinyLlamaWeights(options);
    std::ofstream(directory / "model.safetensors", std::ios::binary)
        .write(reinterpret_cast<const char*>(weights.data()),
               static_cast<std::streamsize>(weights.size()));
  }
  const auto model = LlamaModel::Load(directory.string());
  EXPECT_TRUE(model.ok()) << model.status().ToString();

  std::filesystem::remove(directory / "model.safetensors");
  const auto missing = LlamaModel::Load(directory.string());
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().code(), StatusCode::kNotFound);
  std::filesystem::remove_all(directory);
}

}  // namespace
