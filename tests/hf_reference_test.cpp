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
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ember/arena.h"
#include "ember/forward.h"
#include "ember/json.h"
#include "ember/llama.h"

namespace {

using ember::Json;
using ember::StatusOr;

// Measured: Ember lands within 3.0e-5 to 3.5e-5 of Hugging Face (release and debug on Apple M4;
// CI prints its own figure). Logits reach about 25, and float32 rounding differs with summation
// order and fused multiply-adds, so 1e-4 leaves about 3x room. A wrong RoPE angle, norm eps, or key
// position moves logits by far more and fails here.
constexpr double kTolerance = 1e-4;

// The parsed reference, or null with the reason in *error. A missing or broken file fails the
// tests that read it, not the whole test binary.
const Json* Reference(std::string* error) {
  static const StatusOr<Json> parsed = [] {
    std::ifstream in(EMBER_REFERENCE_DIR "/hf_logits_tinyllama-15M.json", std::ios::binary);
    if (!in) return StatusOr<Json>(ember::Status::NotFound("no reference file in tests/reference"));
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    return Json::Parse(text);
  }();
  if (!parsed.ok()) {
    *error = parsed.status().ToString();
    return nullptr;
  }
  return &*parsed;
}

// Missing or mistyped fields are a broken reference file. They fail the test through a null
// value or an empty array, which the shape checks below catch, instead of aborting.
const Json kNull;

const Json& Member(const Json& object, const char* key) {
  if (!object.is_object()) {
    ADD_FAILURE() << "the reference has a non-object where \"" << key << "\" should be";
    return kNull;
  }
  const Json* value = object.Find(key);
  if (value == nullptr) ADD_FAILURE() << "the reference has no \"" << key << "\"";
  return value == nullptr ? kNull : *value;
}

double Number(const Json& value) {
  if (!value.is_number()) {
    ADD_FAILURE() << "the reference has a non-number where a number should be";
    return std::nan("");
  }
  return value.as_number();
}

std::size_t Count(const Json& object, const char* key) {
  const Json& value = Member(object, key);
  const auto count = value.is_number() ? value.as_uint64() : std::nullopt;
  if (!count.has_value()) ADD_FAILURE() << "\"" << key << "\" is not a whole number";
  return static_cast<std::size_t>(count.value_or(0));
}

std::span<const Json> Items(const Json& value) {
  if (value.is_array()) return value.items();
  ADD_FAILURE() << "the reference has a non-array where an array should be";
  return {};
}

std::vector<double> Numbers(const Json& array) {
  std::vector<double> out;
  for (const Json& item : Items(array)) out.push_back(Number(item));
  return out;
}

// Token ids, each checked against the vocabulary, since the test indexes logits with them. An id
// out of range fails and comes back as 0 so nothing reads past the row.
std::vector<std::size_t> Ids(const Json& array, std::size_t vocab) {
  std::vector<std::size_t> out;
  for (const Json& item : Items(array)) {
    const auto id = item.is_number() ? item.as_uint64() : std::nullopt;
    if (!id.has_value() || *id >= vocab) {
      ADD_FAILURE() << "the reference has a token id outside the vocabulary of " << vocab;
      out.push_back(0);
    } else {
      out.push_back(static_cast<std::size_t>(*id));
    }
  }
  return out;
}

TEST(HfReferenceTest, FileIsWholeAndConsistent) {
  std::string error;
  const Json* reference = Reference(&error);
  ASSERT_NE(reference, nullptr) << error;
  const std::size_t vocab = Count(*reference, "vocab_size");
  const std::size_t top_k = Count(*reference, "top_k");
  const std::size_t stride = Count(*reference, "stride");
  ASSERT_EQ(vocab, 32000U);
  ASSERT_GT(top_k, 1U);
  ASSERT_GT(stride, 0U);
  const std::size_t samples = (vocab + stride - 1) / stride;

  const auto inputs = Items(Member(*reference, "inputs"));
  ASSERT_EQ(inputs.size(), 3U);
  std::size_t positions_seen = 0;
  for (const Json& input : inputs) {
    const std::vector<std::size_t> tokens = Ids(Member(input, "tokens"), vocab);
    const auto positions = Items(Member(input, "positions"));
    ASSERT_EQ(positions.size(), tokens.size());
    for (const Json& position : positions) {
      const std::vector<std::size_t> top_ids = Ids(Member(position, "top_ids"), vocab);
      const std::vector<double> top_logits = Numbers(Member(position, "top_logits"));
      ASSERT_EQ(top_ids.size(), top_k);
      ASSERT_EQ(top_logits.size(), top_k);
      EXPECT_EQ(std::set<std::size_t>(top_ids.begin(), top_ids.end()).size(), top_k);
      EXPECT_TRUE(std::is_sorted(top_logits.rbegin(), top_logits.rend()));
      EXPECT_EQ(Numbers(Member(position, "sampled")).size(), samples);
      // The log of a sum of exps is at least its largest term and at most that plus log(vocab).
      const double logsumexp = Number(Member(position, "logsumexp"));
      EXPECT_GE(logsumexp, top_logits[0]);
      EXPECT_LE(logsumexp, top_logits[0] + std::log(static_cast<double>(vocab)));
      // The root mean square is never below the size of the mean.
      EXPECT_GE(Number(Member(position, "rms")), std::abs(Number(Member(position, "mean"))));
      ++positions_seen;
    }
  }
  EXPECT_EQ(positions_seen, 156U);
}

TEST(HfReferenceTest, EmberMatchesHuggingFace) {
  const char* directory = std::getenv("EMBER_MODEL_DIR");
  if (directory == nullptr || *directory == '\0') {
    const char* required = std::getenv("EMBER_REQUIRE_MODEL");
    if (required != nullptr && std::string_view(required) == "1") {
      FAIL() << "EMBER_REQUIRE_MODEL is 1 but EMBER_MODEL_DIR is not set";
    }
    GTEST_SKIP() << "set EMBER_MODEL_DIR to models/tinyllama-15M (scripts/fetch-model.sh)";
  }
  const auto model = ember::LlamaModel::Load(directory);
  ASSERT_TRUE(model.ok()) << model.status().ToString();

  std::string error;
  const Json* reference = Reference(&error);
  ASSERT_NE(reference, nullptr) << error;
  const std::size_t vocab = Count(*reference, "vocab_size");
  const std::size_t stride = Count(*reference, "stride");
  ASSERT_EQ(vocab, (*model)->config().vocab_size);
  ASSERT_GT(stride, 0U);
  double worst = 0;
  std::size_t positions_checked = 0;
  for (const Json& input : Items(Member(*reference, "inputs"))) {
    const Json& name = Member(input, "name");
    ASSERT_TRUE(name.is_string());
    const std::vector<std::size_t> ids = Ids(Member(input, "tokens"), vocab);
    const std::vector<std::int32_t> tokens(ids.begin(), ids.end());
    ember::Arena arena;
    const auto logits = ember::Forward(**model, tokens, arena);
    ASSERT_TRUE(logits.ok()) << logits.status().ToString();
    ASSERT_EQ(logits->dim(1), vocab);

    const auto positions = Items(Member(input, "positions"));
    ASSERT_EQ(positions.size(), tokens.size());
    for (std::size_t t = 0; t < tokens.size(); ++t) {
      SCOPED_TRACE(name.as_string() + ", position " + std::to_string(t));
      const float* row = logits->data<float>() + t * vocab;
      const auto note = [&](double ours, double theirs) {
        worst = std::max(worst, std::abs(ours - theirs));
        return std::abs(ours - theirs);
      };

      const std::vector<std::size_t> top_ids = Ids(Member(positions[t], "top_ids"), vocab);
      const std::vector<double> top_logits = Numbers(Member(positions[t], "top_logits"));
      ASSERT_EQ(top_ids.size(), top_logits.size());
      ASSERT_GE(top_ids.size(), 2U);
      for (std::size_t i = 0; i < top_ids.size(); ++i) {
        ASSERT_LE(note(row[top_ids[i]], top_logits[i]), kTolerance) << "token " << top_ids[i];
      }
      const std::vector<double> sampled = Numbers(Member(positions[t], "sampled"));
      ASSERT_EQ(sampled.size(), (vocab + stride - 1) / stride);
      for (std::size_t i = 0; i < sampled.size(); ++i) {
        ASSERT_LE(note(row[i * stride], sampled[i]), kTolerance) << "token " << i * stride;
      }

      // Over the whole row: the likeliest token, the softmax normalizer, which moves if a likely
      // token is off, and the mean and root mean square, which move if many tokens are off.
      const float* best = std::max_element(row, row + vocab);
      if (top_logits[0] - top_logits[1] > 2 * kTolerance) {
        EXPECT_EQ(static_cast<std::size_t>(best - row), top_ids[0]);
      }
      double exps = 0, sum = 0, squares = 0;
      for (std::size_t i = 0; i < vocab; ++i) {
        const double v = row[i];
        exps += std::exp(v - double{*best});
        sum += v;
        squares += v * v;
      }
      const double n = static_cast<double>(vocab);
      ASSERT_LE(note(double{*best} + std::log(exps), Number(Member(positions[t], "logsumexp"))),
                kTolerance);
      ASSERT_LE(note(sum / n, Number(Member(positions[t], "mean"))), kTolerance);
      ASSERT_LE(note(std::sqrt(squares / n), Number(Member(positions[t], "rms"))), kTolerance);
      ++positions_checked;
    }
  }
  std::cout << "largest difference from Hugging Face " << worst << " over " << positions_checked
            << " positions\n";
  EXPECT_EQ(positions_checked, 156U);
}

// The KV cache on the real model across its whole context, one token at a time after a prompt, so
// positions past the 128 the reference covers are checked too.
TEST(HfReferenceTest, CachedGenerationMatchesForwardOverTheWholeContext) {
  const char* directory = std::getenv("EMBER_MODEL_DIR");
  if (directory == nullptr || *directory == '\0') {
    const char* required = std::getenv("EMBER_REQUIRE_MODEL");
    if (required != nullptr && std::string_view(required) == "1") {
      FAIL() << "EMBER_REQUIRE_MODEL is 1 but EMBER_MODEL_DIR is not set";
    }
    GTEST_SKIP() << "set EMBER_MODEL_DIR to models/tinyllama-15M (scripts/fetch-model.sh)";
  }
  const auto model = ember::LlamaModel::Load(directory);
  ASSERT_TRUE(model.ok()) << model.status().ToString();
  const std::size_t context = (*model)->config().max_positions;
  std::vector<std::int32_t> tokens(context);
  for (std::size_t i = 0; i < context; ++i) {
    tokens[i] = static_cast<std::int32_t>((i * 7919 + 13) % (*model)->config().vocab_size);
  }
  ember::Arena arena;
  const auto full = ember::Forward(**model, tokens, arena);
  ASSERT_TRUE(full.ok()) << full.status().ToString();
  auto cache = ember::KvCache::Make((*model)->config(), context);
  ASSERT_TRUE(cache.ok()) << cache.status().ToString();
  const std::size_t vocab = (*model)->config().vocab_size;
  const std::size_t prompt = 20;
  double worst = 0;
  for (std::size_t start = 0; start < context;) {
    const std::size_t count = start == 0 ? prompt : 1;
    ember::Arena scratch;
    const auto logits =
        ember::ForwardCached(**model, std::span<const std::int32_t>(tokens.data() + start, count),
                             **cache, scratch, true);
    ASSERT_TRUE(logits.ok()) << logits.status().ToString();
    const std::size_t row = start + count - 1;
    for (std::size_t i = 0; i < vocab; ++i) {
      worst = std::max(
          worst, std::abs(static_cast<double>(logits->at<float>(0, i)) - full->at<float>(row, i)));
    }
    start += count;
  }
  EXPECT_LT(worst, 1e-4) << "largest difference between cached and uncached logits";
}

}  // namespace
