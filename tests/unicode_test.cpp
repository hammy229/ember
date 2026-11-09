#include "ember/unicode.h"

#include <gtest/gtest.h>

#include <string>

namespace {

namespace u = ember::unicode;

std::u32string Nfc(const std::u32string& text) { return u::Nfc(text); }

TEST(Unicode, DecodesAndEncodesUtf8) {
  const std::string text = "a\xC3\xA9\xE6\x97\xA5\xF0\x9F\x94\xA5";  // a é 日 🔥
  const std::u32string code = u::Decode(text);
  EXPECT_EQ(code, (std::u32string{U'a', 0xE9, 0x65E5, 0x1F525}));
  EXPECT_EQ(u::Encode(code), text);
}

TEST(Unicode, KeepsInvalidBytesAsThemselves) {
  const std::string text = "a\xFF\xC3(\xED\xA0\x80";  // a stray byte, a cut-off é, a surrogate
  const std::u32string code = u::Decode(text);
  EXPECT_EQ(code[1], u::kInvalidByteBase + 0xFF);
  EXPECT_FALSE(u::IsLetter(code[1]));
  EXPECT_FALSE(u::IsWhiteSpace(code[1]));
  EXPECT_EQ(u::Encode(code), text);
  EXPECT_EQ(u::Encode(u::Nfc(code)), text);
}

TEST(Unicode, ClassifiesLettersNumbersAndSpaces) {
  for (const char32_t c : {U'a', U'Z', char32_t{0xE9}, char32_t{0x65E5}, char32_t{0x5D0},
                           char32_t{0x1100}, char32_t{0x17F}}) {
    EXPECT_TRUE(u::IsLetter(c)) << std::hex << static_cast<unsigned>(c);
  }
  for (const char32_t c : {U'0', U'9', char32_t{0xB2}, char32_t{0x663}, char32_t{0x2167}}) {
    EXPECT_TRUE(u::IsNumber(c)) << std::hex << static_cast<unsigned>(c);
  }
  for (const char32_t c : {U' ', U'\t', U'\n', U'\r', char32_t{0xA0}, char32_t{0x3000},
                           char32_t{0x2003}, char32_t{0x85}}) {
    EXPECT_TRUE(u::IsWhiteSpace(c)) << std::hex << static_cast<unsigned>(c);
  }
  for (const char32_t c : {U'!', U'_', char32_t{0x301}, char32_t{0x1F525}, char32_t{0x200B}}) {
    EXPECT_FALSE(u::IsLetter(c) || u::IsNumber(c) || u::IsWhiteSpace(c))
        << std::hex << static_cast<unsigned>(c);
  }
}

TEST(Unicode, NfcComposes) {
  EXPECT_EQ(Nfc({U'e', 0x301}), (std::u32string{0xE9}));
  EXPECT_EQ(Nfc({U'A', 0x30A}), (std::u32string{0xC5}));
  // Hangul jamo compose by arithmetic.
  EXPECT_EQ(Nfc({0x1100, 0x1161, 0x11A8}), (std::u32string{0xAC01}));
  EXPECT_EQ(Nfc({0x1100, 0x1161}), (std::u32string{0xAC00}));
}

TEST(Unicode, NfcReordersMarksBeforeComposing) {
  // a + dot below (class 220) + circumflex (class 230), in either order, becomes ậ.
  EXPECT_EQ(Nfc({U'a', 0x323, 0x302}), (std::u32string{0x1EAD}));
  EXPECT_EQ(Nfc({U'a', 0x302, 0x323}), (std::u32string{0x1EAD}));
}

TEST(Unicode, NfcDecomposesWhatNfcNeverWrites) {
  // The Angstrom sign and the Kelvin sign are singletons: NFC replaces them.
  EXPECT_EQ(Nfc({0x212B}), (std::u32string{0xC5}));
  EXPECT_EQ(Nfc({0x212A}), (std::u32string{U'K'}));
  // A composition exclusion stays decomposed.
  EXPECT_EQ(Nfc({0x958}), (std::u32string{0x915, 0x93C}));
}

TEST(Unicode, NfcBlocksCompositionAcrossAStarterOrAnEqualClass) {
  // Two marks of the same class: the second cannot reach the base past the first.
  EXPECT_EQ(Nfc({U'a', 0x301, 0x301}), (std::u32string{0xE1, 0x301}));
  // A starter in between blocks the first letter, but the mark composes with the nearer one.
  EXPECT_EQ(Nfc({U'x', U'a', 0x301}), (std::u32string{U'x', 0xE1}));
}

TEST(Unicode, NfcLeavesAsciiAndComposedTextAlone) {
  const std::u32string plain = U"Hello, world! 123";
  EXPECT_EQ(Nfc(plain), plain);
  const std::u32string composed{0xE9, 0xC5, 0xAC01, 0x65E5};
  EXPECT_EQ(Nfc(composed), composed);
}

TEST(Unicode, LossyReplacesEachBrokenSequenceOnce) {
  EXPECT_EQ(u::Lossy("ok"), "ok");
  EXPECT_EQ(u::Lossy("a\xFF"
                     "b"),
            "a\xEF\xBF\xBD"
            "b");
  // A three-byte character cut after two bytes is one replacement, as in Rust.
  EXPECT_EQ(u::Lossy("\xE6\x97"
                     "a"),
            "\xEF\xBF\xBD"
            "a");
  // E0 cannot be followed by 80, so each byte is its own replacement.
  EXPECT_EQ(u::Lossy("\xE0\x80"), "\xEF\xBF\xBD\xEF\xBF\xBD");
  EXPECT_EQ(u::Lossy("\xF0\x9F\x94\xA5"), "\xF0\x9F\x94\xA5");
}

}  // namespace
