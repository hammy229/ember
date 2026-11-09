#include "ember/tokenizer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "ember/json.h"

namespace {

using ember::Json;
using ember::Tokenizer;
using Ids = std::vector<std::int32_t>;

constexpr std::int32_t kByteBase = 3;  // <0x00> in the tiny vocab below
std::int32_t ByteId(int byte) { return kByteBase + byte; }

// A small tokenizer.json in the Llama 2 format. Ids: <unk> 0, <s> 1, </s> 2, bytes 3 to 258, then
// ▁ 259, a 260, b 261, c 262, ab 263, ▁a 264, ▁ab 265, bc 266.
struct TinyOptions {
  std::string merges = R"(["a b", "▁ a", "b c", "▁ ab"])";
  std::string pre_tokenizer = "null";
  bool byte_fallback = true;
  bool drop_byte_ff = false;
  std::string extra_added;
};

std::string TinyJson(const TinyOptions& o = {}) {
  std::ostringstream vocab;
  vocab << R"("<unk>": 0, "<s>": 1, "</s>": 2)";
  for (int b = 0; b < 256; ++b) {
    if (o.drop_byte_ff && b == 0xFF) continue;
    char piece[8];
    std::snprintf(piece, sizeof piece, "<0x%02X>", b);
    vocab << ", \"" << piece << "\": " << ByteId(b);
  }
  const int next = o.drop_byte_ff ? 258 : 259;
  const char* extra[] = {"▁", "a", "b", "c", "ab", "▁a", "▁ab", "bc"};
  for (int i = 0; i < 8; ++i) vocab << ", \"" << extra[i] << "\": " << next + i;

  const std::string special = R"(, "single_word": false, "lstrip": false, "rstrip": false,
      "normalized": true, "special": true})";
  std::ostringstream json;
  json << R"({"version": "1.0", "added_tokens": [)"
       << R"({"id": 0, "content": "<unk>")" << special << ", "
       << R"({"id": 1, "content": "<s>")" << special << ", "
       << R"({"id": 2, "content": "</s>")" << special << o.extra_added << "],"
       << R"( "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Prepend", "prepend": "▁"},
            {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},)"
       << " \"pre_tokenizer\": " << o.pre_tokenizer << ","
       << R"( "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},)"
       << R"( "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>",)"
       << " \"byte_fallback\": " << (o.byte_fallback ? "true" : "false") << ","
       << " \"vocab\": {" << vocab.str() << "}, \"merges\": " << o.merges << "}}";
  return json.str();
}

Tokenizer Tiny() {
  auto t = Tokenizer::FromJson(TinyJson());
  EXPECT_TRUE(t.ok()) << t.status().ToString();
  return std::move(*t);
}

TEST(TokenizerTest, LoadsTheTinyVocab) {
  const Tokenizer t = Tiny();
  EXPECT_EQ(t.vocab_size(), 267U);
  EXPECT_EQ(t.bos_id(), 1);
  EXPECT_EQ(t.eos_id(), 2);
  EXPECT_EQ(t.piece(263), "ab");
}

TEST(TokenizerTest, MergesByRankNotByPosition) {
  const Tokenizer t = Tiny();
  // ▁ a b c: "a b" has the best rank, then "▁ ab". "b c" never gets a chance.
  EXPECT_EQ(t.Encode("abc", false), (Ids{265, 262}));
  // ▁ b c: "b c" is the only merge that applies.
  EXPECT_EQ(t.Encode("bc", false), (Ids{259, 266}));
  EXPECT_EQ(t.Encode("a", false), (Ids{264}));
}

TEST(TokenizerTest, SpacesBecomeSpacePieces) {
  const Tokenizer t = Tiny();
  EXPECT_EQ(t.Encode("ab ab", false), (Ids{265, 265}));
  EXPECT_EQ(t.Encode(" a", false), (Ids{259, 264}));
}

TEST(TokenizerTest, UnknownCharactersFallBackToBytes) {
  const Tokenizer t = Tiny();
  EXPECT_EQ(t.Encode("x", false), (Ids{259, ByteId('x')}));
  // é is two bytes in UTF-8, so two byte pieces.
  EXPECT_EQ(t.Encode("\xC3\xA9", false), (Ids{259, ByteId(0xC3), ByteId(0xA9)}));
  // Invalid UTF-8 is still encoded, one piece per stray byte.
  EXPECT_EQ(t.Encode("\xFF", false), (Ids{259, ByteId(0xFF)}));
}

TEST(TokenizerTest, AddsBeginOfSequenceOnRequest) {
  const Tokenizer t = Tiny();
  EXPECT_EQ(t.Encode("a", true), (Ids{1, 264}));
  EXPECT_EQ(t.Encode("", true), (Ids{1}));
  EXPECT_TRUE(t.Encode("", false).empty());
}

TEST(TokenizerTest, SpecialTokensOnlyMatchAfterASpace) {
  const Tokenizer t = Tiny();
  EXPECT_EQ(t.Encode("<s>", false), (Ids{1}));
  EXPECT_EQ(t.Encode("a <s> a", false), (Ids{264, 1, 264}));
  // Glued to the word before, it is plain text.
  const Ids glued = t.Encode("a<s>", false);
  EXPECT_EQ(glued.front(), 264);
  EXPECT_EQ(glued.size(), 4U);
}

TEST(TokenizerTest, DecodesBackToTheText) {
  const Tokenizer t = Tiny();
  for (const std::string text : {"abc", "ab ab", " a", "x y", "\xC3\xA9t\xC3\xA9", "  bc  "}) {
    const auto decoded = t.Decode(t.Encode(text, false));
    ASSERT_TRUE(decoded.ok()) << decoded.status().ToString();
    EXPECT_EQ(*decoded, text);
  }
}

TEST(TokenizerTest, BrokenBytesDecodeToReplacementCharacters) {
  const Tokenizer t = Tiny();
  const auto decoded = t.Decode(Ids{260, ByteId(0xC3), ByteId(0xC3), 261});
  ASSERT_TRUE(decoded.ok());
  EXPECT_EQ(*decoded,
            "a\xEF\xBF\xBD\xEF\xBF\xBD"
            "b");
}

TEST(TokenizerTest, DecodeRefusesIdsOutsideTheVocab) {
  const Tokenizer t = Tiny();
  EXPECT_FALSE(t.Decode(Ids{267}).ok());
  EXPECT_FALSE(t.Decode(Ids{-1}).ok());
}

TEST(TokenizerTest, RefusesWhatItCannotRunExactly) {
  TinyOptions pre;
  pre.pre_tokenizer = R"({"type": "ByteLevel"})";
  EXPECT_FALSE(Tokenizer::FromJson(TinyJson(pre)).ok());

  TinyOptions no_fallback;
  no_fallback.byte_fallback = false;
  EXPECT_FALSE(Tokenizer::FromJson(TinyJson(no_fallback)).ok());

  TinyOptions missing_byte;
  missing_byte.drop_byte_ff = true;
  EXPECT_FALSE(Tokenizer::FromJson(TinyJson(missing_byte)).ok());

  TinyOptions bad_merge;
  bad_merge.merges = R"(["a z"])";
  EXPECT_FALSE(Tokenizer::FromJson(TinyJson(bad_merge)).ok());

  TinyOptions three_pieces;
  three_pieces.merges = R"(["a b c"])";
  EXPECT_FALSE(Tokenizer::FromJson(TinyJson(three_pieces)).ok());

  TinyOptions stripping;
  stripping.extra_added =
      R"(, {"id": 263, "content": "ab", "single_word": false, "lstrip": true,
         "rstrip": false, "normalized": true, "special": true})";
  EXPECT_FALSE(Tokenizer::FromJson(TinyJson(stripping)).ok());

  EXPECT_FALSE(Tokenizer::FromJson("[]").ok());
  EXPECT_FALSE(Tokenizer::FromJson("{").ok());
}

TEST(TokenizerTest, AcceptsMergesWrittenAsPairs) {
  TinyOptions pairs;
  pairs.merges = R"([["a", "b"], ["▁", "a"], ["b", "c"], ["▁", "ab"]])";
  const auto t = Tokenizer::FromJson(TinyJson(pairs));
  ASSERT_TRUE(t.ok()) << t.status().ToString();
  EXPECT_EQ(t->Encode("abc", false), (Ids{265, 262}));
}

// The rest needs the real model's tokenizer.json, from scripts/fetch-model.sh.
const char* ModelDir() {
  const char* directory = std::getenv("EMBER_MODEL_DIR");
  if (directory == nullptr || *directory == '\0') return nullptr;
  return directory;
}

#define REQUIRE_MODEL_DIR()                                                                 \
  const char* model_dir = ModelDir();                                                       \
  if (model_dir == nullptr) {                                                               \
    const char* required = std::getenv("EMBER_REQUIRE_MODEL");                              \
    if (required != nullptr && std::string_view(required) == "1") {                         \
      FAIL() << "EMBER_REQUIRE_MODEL is 1 but EMBER_MODEL_DIR is not set";                  \
    }                                                                                       \
    GTEST_SKIP() << "set EMBER_MODEL_DIR to models/tinyllama-15M (scripts/fetch-model.sh)"; \
  }

TEST(TokenizerTest, MatchesHuggingFaceOnEveryReferenceString) {
  REQUIRE_MODEL_DIR();
  const auto t = Tokenizer::Load(std::string(model_dir) + "/tokenizer.json");
  ASSERT_TRUE(t.ok()) << t.status().ToString();
  EXPECT_EQ(t->vocab_size(), 32000U);

  std::ifstream in(EMBER_REFERENCE_DIR "/hf_tokens_tinyllama-15M.json", std::ios::binary);
  ASSERT_TRUE(in.good());
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto reference = Json::Parse(text);
  ASSERT_TRUE(reference.ok()) << reference.status().ToString();
  const Json* cases = reference->Find("cases");
  ASSERT_NE(cases, nullptr);
  ASSERT_EQ(cases->size(), 20U);
  for (const Json& c : cases->items()) {
    const std::string& input = c.Find("text")->as_string();
    SCOPED_TRACE(input);
    Ids expected;
    for (const Json& id : c.Find("ids")->items()) {
      expected.push_back(static_cast<std::int32_t>(*id.as_uint64()));
    }
    EXPECT_EQ(t->Encode(input, false), expected);
    const auto decoded = t->Decode(expected);
    ASSERT_TRUE(decoded.ok()) << decoded.status().ToString();
    EXPECT_EQ(*decoded, c.Find("decoded")->as_string());
  }
}

TEST(TokenizerTest, RandomTextRoundTripsThroughTheRealVocab) {
  REQUIRE_MODEL_DIR();
  const auto t = Tokenizer::Load(std::string(model_dir) + "/tokenizer.json");
  ASSERT_TRUE(t.ok()) << t.status().ToString();
  // Letters, digits, spaces, newlines, and multi-byte characters the vocab may or may not hold.
  const std::vector<std::string> alphabet = {"a",
                                             "e",
                                             "t",
                                             "Q",
                                             "z",
                                             "0",
                                             "7",
                                             " ",
                                             " ",
                                             "\n",
                                             ".",
                                             ",",
                                             "'",
                                             "\xC3\xA9",
                                             "\xC3\x9F",
                                             "\xE6\x97\xA5",
                                             "\xD0\x96",
                                             "\xF0\x9F\x94\xA5",
                                             "\xE0\xA4\xA8",
                                             "<",
                                             ">",
                                             "s"};
  std::mt19937 rng(20261109);
  std::uniform_int_distribution<std::size_t> pick(0, alphabet.size() - 1);
  std::uniform_int_distribution<int> length(0, 40);
  for (int trial = 0; trial < 2000; ++trial) {
    std::string text;
    for (int i = length(rng); i > 0; --i) text += alphabet[pick(rng)];
    // "<s>" right after a space is a special token by design, which does not round trip.
    if (text.find(" <s>") != std::string::npos || text.rfind("<s>", 0) == 0) continue;
    if (text.find(" </s>") != std::string::npos || text.rfind("</s>", 0) == 0) continue;
    const Ids ids = t->Encode(text, false);
    const auto decoded = t->Decode(ids);
    ASSERT_TRUE(decoded.ok());
    ASSERT_EQ(*decoded, text) << "trial " << trial;
  }
}

// Qwen3's byte-level tokenizer, when EMBER_QWEN3_DIR points at models/qwen3-0.6b.
const char* QwenDir() {
  const char* directory = std::getenv("EMBER_QWEN3_DIR");
  return directory == nullptr || *directory == '\0' ? nullptr : directory;
}

TEST(TokenizerTest, Qwen3MatchesHuggingFaceOnEveryReferenceString) {
  const char* dir = QwenDir();
  if (dir == nullptr) GTEST_SKIP() << "set EMBER_QWEN3_DIR to models/qwen3-0.6b";
  const auto t = Tokenizer::Load(std::string(dir) + "/tokenizer.json");
  ASSERT_TRUE(t.ok()) << t.status().ToString();
  EXPECT_EQ(t->kind(), Tokenizer::Kind::kByteLevel);
  EXPECT_EQ(t->vocab_size(), 151669U);
  EXPECT_EQ(t->bos_id(), -1);
  EXPECT_EQ(t->stop_ids(), (Ids{151645, 151643}));

  std::ifstream in(EMBER_REFERENCE_DIR "/hf_tokens_qwen3-0.6b.json", std::ios::binary);
  ASSERT_TRUE(in.good());
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto reference = Json::Parse(text);
  ASSERT_TRUE(reference.ok()) << reference.status().ToString();
  const Json* cases = reference->Find("cases");
  ASSERT_NE(cases, nullptr);
  ASSERT_EQ(cases->size(), 36U);
  for (const Json& c : cases->items()) {
    const std::string& input = c.Find("text")->as_string();
    SCOPED_TRACE(input);
    Ids expected;
    for (const Json& id : c.Find("ids")->items()) {
      expected.push_back(static_cast<std::int32_t>(*id.as_uint64()));
    }
    EXPECT_EQ(t->Encode(input, true), expected);  // Qwen has no begin token to add
    const auto decoded = t->Decode(expected);
    ASSERT_TRUE(decoded.ok()) << decoded.status().ToString();
    EXPECT_EQ(*decoded, c.Find("decoded")->as_string());
  }
}

TEST(TokenizerTest, Qwen3RandomTextRoundTrips) {
  const char* dir = QwenDir();
  if (dir == nullptr) GTEST_SKIP() << "set EMBER_QWEN3_DIR to models/qwen3-0.6b";
  const auto t = Tokenizer::Load(std::string(dir) + "/tokenizer.json");
  ASSERT_TRUE(t.ok()) << t.status().ToString();
  // Already in NFC, so the text should come back exactly.
  const std::vector<std::string> alphabet = {"a",
                                             "e",
                                             "T",
                                             "z",
                                             "0",
                                             "7",
                                             " ",
                                             " ",
                                             "  ",
                                             "\n",
                                             "\r\n",
                                             "\t",
                                             ".",
                                             ",",
                                             "'",
                                             "'s",
                                             "'LL",
                                             "!",
                                             "?",
                                             "#",
                                             "\xC3\xA9",
                                             "\xE6\x97\xA5",
                                             "\xD0\x96",
                                             "\xF0\x9F\x94\xA5",
                                             "\xE0\xA4\xA8",
                                             "\xC2\xA0",
                                             "<",
                                             "|",
                                             ">",
                                             "im",
                                             "_"};
  std::mt19937 rng(20261109);
  std::uniform_int_distribution<std::size_t> pick(0, alphabet.size() - 1);
  std::uniform_int_distribution<int> length(0, 40);
  for (int trial = 0; trial < 2000; ++trial) {
    std::string text;
    for (int i = length(rng); i > 0; --i) text += alphabet[pick(rng)];
    const auto decoded = t->Decode(t->Encode(text, false));
    ASSERT_TRUE(decoded.ok());
    ASSERT_EQ(*decoded, text) << "trial " << trial;
  }
}

}  // namespace
