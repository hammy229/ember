// Milestone 1: Ember's logits against Hugging Face's own, on the real model.
//
// The reference in tests/reference/ came out of `transformers` (scripts/hf_reference.py), so unlike
// forward_test.cpp and scripts/reference_logits.py it was not written by this project. The file is
// read with Ember's own JSON parser, and the first test checks it is whole even on machines without
// the model. The second needs the model: set EMBER_MODEL_DIR to the directory fetch-model.sh fills.
// It skips without one, unless EMBER_REQUIRE_MODEL is set, which CI does so a missing download
// fails instead of quietly passing.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include "ember/arena.h"
#include "ember/forward.h"
#include "ember/json.h"
#include "ember/llama.h"

namespace {

using ember::Json;

// float32 sums of a few hundred terms drift by about 1e-6 of their size, and these logits reach
// about 25, so two correct float32 implementations land within a few times 1e-5. A wrong RoPE,
// scale, or mask moves logits by 0.1 or more.
constexpr double kTolerance = 1e-4;

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

const Json& Reference() {
  static const Json* reference = [] {
    const std::string text = ReadFile(EMBER_REFERENCE_DIR "/hf_logits_tinyllama-15M.json");
    auto parsed = Json::Parse(text);
    if (!parsed.ok()) {
      std::cerr << "cannot parse the reference: " << parsed.status().ToString() << '\n';
      std::abort();
    }
    return new Json(*std::move(parsed));
  }();
  return *reference;
}

const Json& Member(const Json& object, const char* key) {
  const Json* value = object.Find(key);
  if (value == nullptr) {
    std::cerr << "the reference has no \"" << key << "\"\n";
    std::abort();
  }
  return *value;
}

std::size_t Count(const Json& object, const char* key) {
  return static_cast<std::size_t>(Member(object, key).as_uint64().value());
}

std::vector<double> Numbers(const Json& array) {
  std::vector<double> out;
  for (const Json& item : array.items()) out.push_back(item.as_number());
  return out;
}

std::vector<std::int32_t> Ids(const Json& array) {
  std::vector<std::int32_t> out;
  for (const Json& item : array.items()) {
    out.push_back(static_cast<std::int32_t>(item.as_uint64().value()));
  }
  return out;
}

TEST(HfReferenceTest, FileIsWholeAndConsistent) {
  const Json& reference = Reference();
  const std::size_t vocab = Count(reference, "vocab_size");
  const std::size_t top_k = Count(reference, "top_k");
  const std::size_t stride = Count(reference, "stride");
  ASSERT_EQ(vocab, 32000U);
  ASSERT_GT(top_k, 1U);
  const std::size_t samples = (vocab + stride - 1) / stride;

  const Json& inputs = Member(reference, "inputs");
  ASSERT_EQ(inputs.size(), 3U);
  std::size_t positions_seen = 0;
  for (const Json& input : inputs.items()) {
    const std::vector<std::int32_t> tokens = Ids(Member(input, "tokens"));
    const Json& positions = Member(input, "positions");
    ASSERT_EQ(positions.size(), tokens.size());
    for (const std::int32_t token : tokens) ASSERT_LT(static_cast<std::size_t>(token), vocab);
    for (const Json& position : positions.items()) {
      const std::vector<std::int32_t> top_ids = Ids(Member(position, "top_ids"));
      const std::vector<double> top_logits = Numbers(Member(position, "top_logits"));
      ASSERT_EQ(top_ids.size(), top_k);
      ASSERT_EQ(top_logits.size(), top_k);
      EXPECT_EQ(std::set<std::int32_t>(top_ids.begin(), top_ids.end()).size(), top_k);
      EXPECT_TRUE(std::is_sorted(top_logits.rbegin(), top_logits.rend()));
      EXPECT_EQ(Member(position, "sampled").size(), samples);
      // The log of a sum of exps is at least its largest term and at most that plus log(vocab).
      const double logsumexp = Member(position, "logsumexp").as_number();
      EXPECT_GE(logsumexp, top_logits[0]);
      EXPECT_LE(logsumexp, top_logits[0] + std::log(static_cast<double>(vocab)));
      ++positions_seen;
    }
  }
  EXPECT_EQ(positions_seen, 156U);
}

TEST(HfReferenceTest, EmberMatchesHuggingFace) {
  const char* directory = std::getenv("EMBER_MODEL_DIR");
  if (directory == nullptr || *directory == '\0') {
    if (std::getenv("EMBER_REQUIRE_MODEL") != nullptr) {
      FAIL() << "EMBER_REQUIRE_MODEL is set but EMBER_MODEL_DIR is not";
    }
    GTEST_SKIP() << "set EMBER_MODEL_DIR to models/tinyllama-15M (scripts/fetch-model.sh)";
  }
  const auto model = ember::LlamaModel::Load(directory);
  ASSERT_TRUE(model.ok()) << model.status().ToString();

  const Json& reference = Reference();
  const std::size_t stride = Count(reference, "stride");
  double worst = 0;
  std::size_t positions_checked = 0;
  for (const Json& input : Member(reference, "inputs").items()) {
    const std::string& name = Member(input, "name").as_string();
    const std::vector<std::int32_t> tokens = Ids(Member(input, "tokens"));
    ember::Arena arena;
    const auto logits = ember::Forward(**model, tokens, arena);
    ASSERT_TRUE(logits.ok()) << logits.status().ToString();
    const std::size_t vocab = logits->dim(1);
    ASSERT_EQ(vocab, Count(reference, "vocab_size"));

    const auto positions = Member(input, "positions").items();
    for (std::size_t t = 0; t < tokens.size(); ++t) {
      SCOPED_TRACE(name + ", position " + std::to_string(t));
      const float* row = logits->data<float>() + t * vocab;
      const auto note = [&](double ours, double theirs) {
        worst = std::max(worst, std::abs(ours - theirs));
        return std::abs(ours - theirs);
      };

      const std::vector<std::int32_t> top_ids = Ids(Member(positions[t], "top_ids"));
      const std::vector<double> top_logits = Numbers(Member(positions[t], "top_logits"));
      for (std::size_t i = 0; i < top_ids.size(); ++i) {
        ASSERT_LE(note(row[top_ids[i]], top_logits[i]), kTolerance) << "token " << top_ids[i];
      }
      const std::vector<double> sampled = Numbers(Member(positions[t], "sampled"));
      for (std::size_t i = 0; i < sampled.size(); ++i) {
        ASSERT_LE(note(row[i * stride], sampled[i]), kTolerance) << "token " << i * stride;
      }

      // Over the whole row: the likeliest token, and the softmax normalizer, which every one of
      // the 32,000 logits feeds into.
      const float* best = std::max_element(row, row + vocab);
      if (top_logits[0] - top_logits[1] > 2 * kTolerance) {
        EXPECT_EQ(best - row, top_ids[0]);
      }
      double sum = 0;
      for (std::size_t i = 0; i < vocab; ++i) sum += std::exp(double{row[i]} - double{*best});
      ASSERT_LE(note(double{*best} + std::log(sum), Member(positions[t], "logsumexp").as_number()),
                kTolerance);
      ++positions_checked;
    }
  }
  std::cout << "largest difference from Hugging Face " << worst << " over " << positions_checked
            << " positions\n";
  EXPECT_EQ(positions_checked, 156U);
}

}  // namespace
