#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ember/autograd.h"

namespace ember {

// A tiny transformer that learns text one character at a time, trained from scratch with
// ember/autograd.h. It is the same shape as the Llama models Ember runs (embeddings, causal
// self-attention with several heads, an MLP, RMSNorm before each, residual connections), small
// enough to train in seconds natively and live in a browser tab.
struct CharModelConfig {
  std::size_t vocab = 0;
  std::size_t context = 64;
  std::size_t width = 64;
  std::size_t layers = 2;
  std::size_t heads = 4;
  std::size_t mlp = 256;
};

// Characters to ids: every distinct byte in the training text, in byte order.
class CharVocab {
 public:
  explicit CharVocab(std::string_view text) {
    std::array<bool, 256> seen{};
    for (const char c : text) seen[static_cast<unsigned char>(c)] = true;
    ids_.fill(-1);
    for (std::size_t b = 0; b < 256; ++b) {
      if (seen[b]) {
        ids_[b] = static_cast<std::int32_t>(chars_.size());
        chars_.push_back(static_cast<char>(b));
      }
    }
  }

  std::size_t size() const { return chars_.size(); }
  // Characters the vocabulary has never seen are skipped.
  std::vector<std::int32_t> Encode(std::string_view text) const {
    std::vector<std::int32_t> out;
    out.reserve(text.size());
    for (const char c : text) {
      const std::int32_t id = ids_[static_cast<unsigned char>(c)];
      if (id >= 0) out.push_back(id);
    }
    return out;
  }
  char Char(std::int32_t id) const { return chars_[static_cast<std::size_t>(id)]; }

 private:
  std::array<std::int32_t, 256> ids_{};
  std::vector<char> chars_;
};

template <typename T>
class CharModel {
 public:
  using V = autograd::Var<T>;

  CharModel(const CharModelConfig& config, std::uint32_t seed) : config_(config) {
    std::mt19937 rng(seed);
    const auto random = [&](std::size_t rows, std::size_t cols, T spread) {
      std::normal_distribution<double> normal(0.0, static_cast<double>(spread));
      std::vector<T> values(rows * cols);
      for (T& v : values) v = static_cast<T>(normal(rng));
      V leaf = autograd::Leaf<T>(rows, cols, std::move(values));
      parameters_.push_back(leaf);
      return leaf;
    };
    const auto ones = [&](std::size_t cols) {
      V leaf = autograd::Leaf<T>(1, cols, std::vector<T>(cols, T(1)));
      parameters_.push_back(leaf);
      return leaf;
    };
    const std::size_t w = config.width;
    const T in = T(1) / std::sqrt(T(w));
    // Each layer's output projections start smaller, so the residual stream stays calm at first.
    const T out = in / std::sqrt(T(2 * config.layers));
    token_ = random(config.vocab, w, T(0.1));
    position_ = random(config.context, w, T(0.1));
    for (std::size_t l = 0; l < config.layers; ++l) {
      Layer layer;
      layer.norm1 = ones(w);
      layer.query = random(w, w, in);
      layer.key = random(w, w, in);
      layer.value = random(w, w, in);
      layer.output = random(w, w, out);
      layer.norm2 = ones(w);
      layer.up = random(w, config.mlp, in);
      layer.down =
          random(config.mlp, w, T(1) / std::sqrt(T(config.mlp)) / std::sqrt(T(2 * config.layers)));
      layers_.push_back(layer);
    }
    final_norm_ = ones(w);
    head_ = random(w, config.vocab, in);
  }

  const CharModelConfig& config() const { return config_; }
  const std::vector<V>& parameters() const { return parameters_; }
  std::size_t parameter_count() const {
    std::size_t n = 0;
    for (const V& p : parameters_) n += p->value.size();
    return n;
  }

  // Logits [tokens, vocab] for every position. At most `context` tokens.
  V Forward(std::span<const std::int32_t> tokens) const {
    const std::size_t n = tokens.size();
    const std::size_t head_width = config_.width / config_.heads;
    const T scale = T(1) / std::sqrt(T(head_width));
    V x = autograd::Add(autograd::Embedding(token_, tokens), autograd::TopRows(position_, n));
    for (const Layer& layer : layers_) {
      const V h = autograd::RmsNorm(x, layer.norm1);
      const V q = autograd::MatMul(h, layer.query);
      const V k = autograd::MatMul(h, layer.key);
      const V v = autograd::MatMul(h, layer.value);
      std::vector<V> heads;
      for (std::size_t i = 0; i < config_.heads; ++i) {
        const V qh = autograd::Columns(q, i * head_width, head_width);
        const V kh = autograd::Columns(k, i * head_width, head_width);
        const V vh = autograd::Columns(v, i * head_width, head_width);
        const V att = autograd::CausalSoftmax(
            autograd::Scale(autograd::MatMul(qh, autograd::Transpose(kh)), scale));
        heads.push_back(autograd::MatMul(att, vh));
      }
      x = autograd::Add(x, autograd::MatMul(autograd::ConcatColumns(heads), layer.output));
      const V m = autograd::RmsNorm(x, layer.norm2);
      x = autograd::Add(
          x, autograd::MatMul(autograd::Relu(autograd::MatMul(m, layer.up)), layer.down));
    }
    return autograd::MatMul(autograd::RmsNorm(x, final_norm_), head_);
  }

  // Mean cross-entropy of predicting tokens[1..] from tokens[..n-1], in nats per character.
  V Loss(std::span<const std::int32_t> tokens) const {
    return autograd::CrossEntropy(Forward(tokens.first(tokens.size() - 1)), tokens.subspan(1));
  }

 private:
  struct Layer {
    V norm1, query, key, value, output, norm2, up, down;
  };

  CharModelConfig config_;
  std::vector<V> parameters_;
  V token_, position_, final_norm_, head_;
  std::vector<Layer> layers_;
};

// Trains a CharModel<float> on text, holding the last 10% back to measure loss on text it never
// trains on.
class CharTrainer {
 public:
  struct Options {
    CharModelConfig model;
    std::size_t batch = 8;
    float rate = 3e-3F;
    std::uint32_t seed = 0;
  };

  CharTrainer(std::string_view text, Options options)
      : vocab_(text),
        options_(Fix(options, vocab_.size())),
        model_(options_.model, options_.seed),
        optimizer_(model_.parameters(), options_.rate, 0.01F),
        rng_(options_.seed) {
    const std::vector<std::int32_t> ids = vocab_.Encode(text);
    const std::size_t split = ids.size() * 9 / 10;
    train_.assign(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(split));
    held_out_.assign(ids.begin() + static_cast<std::ptrdiff_t>(split), ids.end());
  }

  // Enough text for at least one window in each part.
  bool ready() const {
    return train_.size() > options_.model.context + 1 &&
           held_out_.size() > options_.model.context + 1;
  }

  // One optimizer step on a batch of random windows. Gives the batch's mean loss.
  float Step() {
    autograd::ZeroGrad(model_.parameters());
    float total = 0;
    for (std::size_t b = 0; b < options_.batch; ++b) {
      const std::span<const std::int32_t> window = Window(train_);
      auto loss = autograd::Scale(model_.Loss(window), 1.0F / static_cast<float>(options_.batch));
      autograd::Backward(loss);
      total += loss->value[0];
    }
    optimizer_.Step();
    ++steps_;
    return total;
  }

  // Mean loss on `windows` evenly spaced windows of the held-out text, without training on them.
  float HeldOutLoss(std::size_t windows = 16) const {
    const std::size_t span = options_.model.context + 1;
    const std::size_t last = held_out_.size() - span;
    float total = 0;
    for (std::size_t i = 0; i < windows; ++i) {
      const std::size_t start = last * i / std::max<std::size_t>(windows - 1, 1);
      total += model_.Loss(std::span(held_out_).subspan(start, span))->value[0];
    }
    return total / static_cast<float>(windows);
  }

  // Continues `prefix` by `count` characters, sampling at the given temperature.
  std::string Sample(std::string_view prefix, std::size_t count, float temperature,
                     std::uint32_t seed) const {
    std::mt19937 rng(seed);
    std::vector<std::int32_t> ids = vocab_.Encode(prefix);
    if (ids.empty()) ids.push_back(0);
    std::string out(prefix);
    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t n = std::min(ids.size(), options_.model.context);
      const auto logits = model_.Forward(std::span(ids).last(n));
      const float* row = &logits->value[(n - 1) * vocab_.size()];
      std::vector<double> weights(vocab_.size());
      // Written so a NaN temperature from the browser falls back to the floor instead of
      // giving NaN weights, which std::discrete_distribution does not allow.
      const float t = temperature > 1e-3F ? temperature : 1e-3F;
      const float largest = *std::max_element(row, row + vocab_.size());
      for (std::size_t j = 0; j < vocab_.size(); ++j) {
        weights[j] = std::exp(static_cast<double>(row[j] - largest) / t);
      }
      std::discrete_distribution<std::int32_t> pick(weights.begin(), weights.end());
      ids.push_back(pick(rng));
      out += vocab_.Char(ids.back());
    }
    return out;
  }

  std::size_t steps() const { return steps_; }
  const CharVocab& vocab() const { return vocab_; }
  const CharModel<float>& model() const { return model_; }

 private:
  static Options Fix(Options options, std::size_t vocab) {
    options.model.vocab = vocab;
    return options;
  }

  std::span<const std::int32_t> Window(const std::vector<std::int32_t>& ids) {
    std::uniform_int_distribution<std::size_t> start(0, ids.size() - options_.model.context - 1);
    return std::span(ids).subspan(start(rng_), options_.model.context + 1);
  }

  CharVocab vocab_;
  Options options_;
  CharModel<float> model_;
  autograd::AdamW<float> optimizer_;
  std::mt19937 rng_;
  std::vector<std::int32_t> train_;
  std::vector<std::int32_t> held_out_;
  std::size_t steps_ = 0;
};

}  // namespace ember
