#include "ember/json.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace {

using ember::Json;
using ember::StatusCode;

Json MustParse(std::string_view text) {
  auto parsed = Json::Parse(text);
  EXPECT_TRUE(parsed.ok()) << text << ": " << parsed.status().ToString();
  return parsed.ok() ? std::move(parsed).value() : Json();
}

bool Refused(std::string_view text) {
  const auto parsed = Json::Parse(text);
  return !parsed.ok() && parsed.status().code() == StatusCode::kInvalidFormat;
}

TEST(Json, ReadsEveryKind) {
  const Json json = MustParse(R"( {"a": null, "b": true, "c": false, "d": -1.5e2, "e": "x",
                                    "f": [1, [2]], "g": {}} )");
  ASSERT_TRUE(json.is_object());
  ASSERT_EQ(json.size(), 7U);
  EXPECT_TRUE(json.Find("a")->is_null());
  EXPECT_TRUE(json.Find("b")->as_bool());
  EXPECT_FALSE(json.Find("c")->as_bool());
  EXPECT_EQ(json.Find("d")->as_number(), -150.0);
  EXPECT_EQ(json.Find("e")->as_string(), "x");
  ASSERT_EQ(json.Find("f")->size(), 2U);
  EXPECT_EQ(json.Find("f")->items()[1].items()[0].as_number(), 2.0);
  EXPECT_TRUE(json.Find("g")->is_object());
  EXPECT_EQ(json.Find("missing"), nullptr);
}

TEST(Json, KeepsKeysInFileOrder) {
  const Json json = MustParse(R"({"z": 1, "a": 2, "m": 3})");
  ASSERT_EQ(json.keys().size(), 3U);
  EXPECT_EQ(json.keys()[0], "z");
  EXPECT_EQ(json.keys()[1], "a");
  EXPECT_EQ(json.keys()[2], "m");
}

TEST(Json, ReadsWholeNumbersExactlyPastTwoToThe53) {
  // 2^53 + 1 has no double, so only as_uint64 can say it.
  const Json json = MustParse("[9007199254740993, 18446744073709551615, 0]");
  EXPECT_EQ(json.items()[0].as_uint64(), std::uint64_t{9007199254740993U});
  EXPECT_EQ(json.items()[1].as_uint64(), UINT64_MAX);
  EXPECT_EQ(json.items()[2].as_uint64(), std::uint64_t{0});
}

TEST(Json, OnlyPlainNonNegativeIntegersHaveAnExactValue) {
  const Json json = MustParse("[-1, 1.0, 1e3, 18446744073709551616, \"1\"]");
  for (const Json& item : json.items()) EXPECT_FALSE(item.as_uint64().has_value());
}

TEST(Json, DecodesEscapesToUtf8) {
  const Json json = MustParse(R"("a\"\\\/\b\f\n\r\t\u00e9\u20ac\ud83d\ude00")");
  EXPECT_EQ(json.as_string(), "a\"\\/\b\f\n\r\t\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80");
}

TEST(Json, RefusesWhatTheGrammarDoesNotAllow) {
  for (const std::string_view text : {"",
                                      " ",
                                      "{",
                                      "[",
                                      "[1,]",
                                      "{\"a\":1,}",
                                      "{a:1}",
                                      "{\"a\" 1}",
                                      "01",
                                      "1.",
                                      ".5",
                                      "1e",
                                      "+1",
                                      "-",
                                      "tru",
                                      "nul",
                                      "\"abc",
                                      "\"\\x\"",
                                      "\"\\u12\"",
                                      "\"\\ud800\"",
                                      "\"\\udc00\"",
                                      "\"\\ud800\\u0041\"",
                                      "[1] [2]",
                                      "1e999",
                                      "NaN",
                                      "\"tab\there\"",
                                      "{\"a\":1}x"}) {
    EXPECT_TRUE(Refused(text)) << "accepted: " << text;
  }
}

TEST(Json, RefusesDuplicateKeys) {
  EXPECT_TRUE(Refused(R"({"a": 1, "a": 2})"));
  EXPECT_TRUE(MustParse(R"({"a": {"a": 1}})").is_object());
}

TEST(Json, RefusesNestingPastTheLimit) {
  const std::string ok = std::string(Json::kMaxDepth, '[') + std::string(Json::kMaxDepth, ']');
  EXPECT_TRUE(MustParse(ok).is_array());
  const std::string deep =
      std::string(Json::kMaxDepth + 1, '[') + std::string(Json::kMaxDepth + 1, ']');
  EXPECT_TRUE(Refused(deep));
  // Deep enough to overflow the stack if the limit were missing.
  EXPECT_TRUE(Refused(std::string(1'000'000, '[')));
}

TEST(Json, SaysWhereItGaveUp) {
  const auto parsed = Json::Parse("[1, x]");
  ASSERT_FALSE(parsed.ok());
  EXPECT_NE(parsed.status().message().find("byte 4"), std::string::npos)
      << parsed.status().message();
}

}  // namespace
