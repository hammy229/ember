#include "ember/llama.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ember/f16.h"
#include "ember/json.h"
#include "ember/safetensors.h"

namespace ember {

namespace {

Status BadConfig(std::string_view what) {
  return Status::InvalidFormat("config.json: " + std::string(what));
}

// A field, or null if it is missing or written as null. Hugging Face treats the two the same, as
// "use the default".
const Json* Present(const Json& config, std::string_view key) {
  const Json* value = config.Find(key);
  return value == nullptr || value->is_null() ? nullptr : value;
}

// A size field, which must be a whole number from 1 to `limit`. The limits are far above any real
// model. They do not keep every product of sizes inside a size_t (vocab times hidden can reach
// 2^36), but Shape::Make refuses any tensor that big when the weights are bound, so a huge config
// fails there with a clear error instead of overflowing.
StatusOr<std::size_t> ReadSize(const Json& config, std::string_view key, std::size_t limit) {
  const Json* value = Present(config, key);
  if (value == nullptr) return BadConfig("missing " + std::string(key));
  const std::optional<std::uint64_t> exact = value->as_uint64();
  if (!exact || *exact == 0 || *exact > limit) {
    return BadConfig(std::string(key) + " must be a whole number from 1 to " +
                     std::to_string(limit));
  }
  return static_cast<std::size_t>(*exact);
}

StatusOr<float> ReadPositive(const Json& config, std::string_view key, float fallback) {
  const Json* value = Present(config, key);
  if (value == nullptr) return fallback;
  if (!value->is_number()) return BadConfig(std::string(key) + " must be a number");
  const auto number = static_cast<float>(value->as_number());
  if (!std::isfinite(number) || number <= 0) return BadConfig(std::string(key) + " must be > 0");
  return number;
}

Status RequireString(const Json& config, std::string_view key, std::string_view expected) {
  const Json* value = config.Find(key);
  if (value == nullptr || !value->is_string() || value->as_string() != expected) {
    return BadConfig(std::string(key) + " must be \"" + std::string(expected) + "\"");
  }
  return Status::Ok();
}

// RoPE settings. Older configs say rope_theta and rope_scaling at the top level. Newer ones put
// both in rope_parameters, and a config that set the base there but was read the old way would run
// with the default base and give fluent-looking garbage. So both places are read, and anything but
// plain RoPE is refused: scaled variants change the math and Ember does not do them yet.
StatusOr<float> ReadRope(const Json& config) {
  const Json* scaling = Present(config, "rope_scaling");
  const Json* parameters = Present(config, "rope_parameters");
  if (scaling != nullptr && parameters != nullptr) {
    return BadConfig("give rope_scaling or rope_parameters, not both");
  }
  const Json* rope = scaling != nullptr ? scaling : parameters;
  float theta = 10000.0F;
  if (Present(config, "rope_theta") != nullptr) {
    EMBER_ASSIGN_OR_RETURN(theta, ReadPositive(config, "rope_theta", theta));
  }
  if (rope == nullptr) return theta;
  if (!rope->is_object()) return BadConfig("rope settings must be an object");
  for (std::size_t i = 0; i < rope->size(); ++i) {
    const std::string& key = rope->keys()[i];
    const Json& value = rope->items()[i];
    if (key == "rope_type" || key == "type") {
      if (!value.is_string() || value.as_string() != "default") {
        return BadConfig("only plain RoPE is supported, not rope_type " +
                         (value.is_string() ? value.as_string() : std::string("?")));
      }
    } else if (key == "rope_theta") {
      if (Present(config, "rope_theta") != nullptr) {
        return BadConfig("rope_theta is given twice");
      }
      EMBER_ASSIGN_OR_RETURN(theta, ReadPositive(*rope, "rope_theta", theta));
    } else {
      return BadConfig("rope setting " + key + " is not supported");
    }
  }
  return theta;
}

constexpr std::size_t kMaxVocab = std::size_t{1} << 20;
constexpr std::size_t kMaxWidth = std::size_t{1} << 16;
constexpr std::size_t kMaxLayers = 1024;
constexpr std::size_t kMaxHeads = 1024;
constexpr std::size_t kMaxPositions = std::size_t{1} << 20;

}  // namespace

StatusOr<LlamaConfig> LlamaConfig::FromJson(std::string_view json_text) {
  StatusOr<Json> parsed = Json::Parse(json_text);
  if (!parsed.ok()) return BadConfig(parsed.status().message());
  const Json& json = *parsed;
  if (!json.is_object()) return BadConfig("not a JSON object");

  EMBER_RETURN_IF_ERROR(RequireString(json, "model_type", "llama"));
  if (Present(json, "hidden_act") != nullptr) {
    EMBER_RETURN_IF_ERROR(RequireString(json, "hidden_act", "silu"));
  }
  for (const std::string_view key : {"attention_bias", "mlp_bias"}) {
    const Json* bias = Present(json, key);
    if (bias != nullptr && !(bias->is_bool() && !bias->as_bool())) {
      return BadConfig(std::string(key) + " is not supported");
    }
  }

  LlamaConfig config;
  EMBER_ASSIGN_OR_RETURN(config.vocab_size, ReadSize(json, "vocab_size", kMaxVocab));
  EMBER_ASSIGN_OR_RETURN(config.hidden_size, ReadSize(json, "hidden_size", kMaxWidth));
  EMBER_ASSIGN_OR_RETURN(config.intermediate_size,
                         ReadSize(json, "intermediate_size", 4 * kMaxWidth));
  EMBER_ASSIGN_OR_RETURN(config.layer_count, ReadSize(json, "num_hidden_layers", kMaxLayers));
  EMBER_ASSIGN_OR_RETURN(config.head_count, ReadSize(json, "num_attention_heads", kMaxHeads));
  config.kv_head_count = config.head_count;
  if (Present(json, "num_key_value_heads") != nullptr) {
    EMBER_ASSIGN_OR_RETURN(config.kv_head_count, ReadSize(json, "num_key_value_heads", kMaxHeads));
  }
  if (Present(json, "head_dim") != nullptr) {
    EMBER_ASSIGN_OR_RETURN(config.head_dim, ReadSize(json, "head_dim", kMaxWidth));
  } else {
    if (config.hidden_size % config.head_count != 0) {
      return BadConfig("hidden_size must divide evenly by num_attention_heads");
    }
    config.head_dim = config.hidden_size / config.head_count;
  }
  // Hugging Face's LlamaConfig defaults, for fields an old config leaves out.
  config.max_positions = 2048;
  if (Present(json, "max_position_embeddings") != nullptr) {
    EMBER_ASSIGN_OR_RETURN(config.max_positions,
                           ReadSize(json, "max_position_embeddings", kMaxPositions));
  }
  EMBER_ASSIGN_OR_RETURN(config.rms_norm_eps, ReadPositive(json, "rms_norm_eps", 1e-6F));
  EMBER_ASSIGN_OR_RETURN(config.rope_theta, ReadRope(json));

  if (const Json* tie = Present(json, "tie_word_embeddings"); tie != nullptr) {
    if (!tie->is_bool()) return BadConfig("tie_word_embeddings must be true or false");
    config.tie_word_embeddings = tie->as_bool();
  }

  if (config.head_count % config.kv_head_count != 0) {
    return BadConfig("num_attention_heads must divide evenly by num_key_value_heads");
  }
  if (config.head_dim % 2 != 0) return BadConfig("head_dim must be even for rotary embeddings");
  if (config.head_count * config.head_dim > 4 * kMaxWidth) {
    return BadConfig("num_attention_heads * head_dim is too large");
  }
  return config;
}

std::size_t LlamaModel::parameter_count() const {
  std::size_t count = weights_.embedding.element_count() + weights_.final_norm.element_count();
  if (!config_.tie_word_embeddings) count += weights_.lm_head.element_count();
  for (const LlamaLayer& layer : weights_.layers) {
    for (const TensorView* weight :
         {&layer.attention_norm, &layer.query, &layer.key, &layer.value, &layer.output,
          &layer.mlp_norm, &layer.gate, &layer.up, &layer.down}) {
      count += weight->element_count();
    }
  }
  return count;
}

StatusOr<std::unique_ptr<LlamaModel>> LlamaModel::Load(const std::string& directory) {
  EMBER_ASSIGN_OR_RETURN(const MappedFile config_file,
                         MappedFile::Open(directory + "/config.json"));
  StatusOr<MappedFile> weights_file = MappedFile::Open(directory + "/model.safetensors");
  if (!weights_file.ok()) {
    if (MappedFile::Open(directory + "/model.safetensors.index.json").ok()) {
      return Status::InvalidArgument(directory + ": weights split across files are not supported");
    }
    return weights_file.status();
  }

  const std::string_view config_text(reinterpret_cast<const char*>(config_file.bytes().data()),
                                     config_file.size());
  std::unique_ptr<LlamaModel> model(new LlamaModel());
  EMBER_ASSIGN_OR_RETURN(model->config_, LlamaConfig::FromJson(config_text));
  model->file_ = std::move(weights_file).value();
  EMBER_RETURN_IF_ERROR(model->Bind(model->file_.bytes()));
  return model;
}

StatusOr<std::unique_ptr<LlamaModel>> LlamaModel::FromBytes(
    std::string_view config_json, std::span<const std::byte> safetensors) {
  std::unique_ptr<LlamaModel> model(new LlamaModel());
  EMBER_ASSIGN_OR_RETURN(model->config_, LlamaConfig::FromJson(config_json));
  EMBER_RETURN_IF_ERROR(model->Bind(safetensors));
  return model;
}

Status LlamaModel::Bind(std::span<const std::byte> safetensors) {
  EMBER_ASSIGN_OR_RETURN(const Safetensors file, Safetensors::Parse(safetensors));
  const LlamaConfig& c = config_;
  std::vector<bool> used(file.tensors().size(), false);

  // Finds one weight, checks its shape, and gives it back as contiguous f32.
  const auto bind = [&](const std::string& name,
                        std::initializer_list<std::size_t> dims) -> StatusOr<TensorView> {
    const Safetensors::Entry* entry = file.Find(name);
    if (entry == nullptr) return Status::InvalidFormat("missing weight " + name);
    used[static_cast<std::size_t>(entry - file.tensors().data())] = true;
    EMBER_ASSIGN_OR_RETURN(const Shape expected, Shape::Make(dims));
    if (!(entry->shape == expected)) {
      return Status::InvalidFormat(name + " is " + entry->shape.ToString() + ", expected " +
                                   expected.ToString());
    }
    if (entry->dtype == DType::kF32) {
      StatusOr<TensorView> view = Safetensors::View(*entry);
      if (view.ok()) return view;
      // Misaligned in the file, so copy it somewhere aligned.
      EMBER_ASSIGN_OR_RETURN(const Tensor copy,
                             Tensor::Uninitialized(arena_, DType::kF32, expected));
      std::memcpy(copy.raw_data(), entry->bytes.data(), entry->bytes.size());
      return TensorView(copy);
    }
    if (entry->dtype == DType::kF16) {
      EMBER_ASSIGN_OR_RETURN(const Tensor widened,
                             Tensor::Uninitialized(arena_, DType::kF32, expected));
      F16ToF32(entry->bytes.data(), expected.element_count(), widened.data<float>());
      return TensorView(widened);
    }
    if (entry->dtype == DType::kBF16) {
      EMBER_ASSIGN_OR_RETURN(const Tensor widened,
                             Tensor::Uninitialized(arena_, DType::kF32, expected));
      BF16ToF32(entry->bytes.data(), expected.element_count(), widened.data<float>());
      return TensorView(widened);
    }
    return Status::InvalidFormat(name + " is " + std::string(Name(entry->dtype)) +
                                 ", but weights must be f32, f16, or bf16");
  };

  const std::size_t q_width = c.head_count * c.head_dim;
  const std::size_t kv_width = c.kv_head_count * c.head_dim;
  EMBER_ASSIGN_OR_RETURN(weights_.embedding,
                         bind("model.embed_tokens.weight", {c.vocab_size, c.hidden_size}));
  weights_.layers.resize(c.layer_count);
  for (std::size_t i = 0; i < c.layer_count; ++i) {
    const std::string prefix = "model.layers." + std::to_string(i) + ".";
    LlamaLayer& layer = weights_.layers[i];
    EMBER_ASSIGN_OR_RETURN(layer.attention_norm,
                           bind(prefix + "input_layernorm.weight", {c.hidden_size}));
    EMBER_ASSIGN_OR_RETURN(layer.query,
                           bind(prefix + "self_attn.q_proj.weight", {q_width, c.hidden_size}));
    EMBER_ASSIGN_OR_RETURN(layer.key,
                           bind(prefix + "self_attn.k_proj.weight", {kv_width, c.hidden_size}));
    EMBER_ASSIGN_OR_RETURN(layer.value,
                           bind(prefix + "self_attn.v_proj.weight", {kv_width, c.hidden_size}));
    EMBER_ASSIGN_OR_RETURN(layer.output,
                           bind(prefix + "self_attn.o_proj.weight", {c.hidden_size, q_width}));
    EMBER_ASSIGN_OR_RETURN(layer.mlp_norm,
                           bind(prefix + "post_attention_layernorm.weight", {c.hidden_size}));
    EMBER_ASSIGN_OR_RETURN(
        layer.gate, bind(prefix + "mlp.gate_proj.weight", {c.intermediate_size, c.hidden_size}));
    EMBER_ASSIGN_OR_RETURN(
        layer.up, bind(prefix + "mlp.up_proj.weight", {c.intermediate_size, c.hidden_size}));
    EMBER_ASSIGN_OR_RETURN(
        layer.down, bind(prefix + "mlp.down_proj.weight", {c.hidden_size, c.intermediate_size}));
  }
  EMBER_ASSIGN_OR_RETURN(weights_.final_norm, bind("model.norm.weight", {c.hidden_size}));
  if (c.tie_word_embeddings) {
    weights_.lm_head = weights_.embedding;
    // Some files save the tied copy anyway. Hugging Face ignores it, and so does Ember.
    if (const Safetensors::Entry* head = file.Find("lm_head.weight"); head != nullptr) {
      used[static_cast<std::size_t>(head - file.tensors().data())] = true;
    }
  } else {
    EMBER_ASSIGN_OR_RETURN(weights_.lm_head, bind("lm_head.weight", {c.vocab_size, c.hidden_size}));
  }

  // A tensor nobody asked for means the file is some other architecture, and running it as this
  // one would quietly give wrong answers. Older exports save RoPE frequencies, which Ember
  // computes.
  for (std::size_t i = 0; i < used.size(); ++i) {
    const std::string& name = file.tensors()[i].name;
    if (!used[i] && !name.ends_with("rotary_emb.inv_freq")) {
      return Status::InvalidFormat("unexpected weight " + name + " for this config");
    }
  }
  return Status::Ok();
}

}  // namespace ember
