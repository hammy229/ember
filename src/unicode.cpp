#include "ember/unicode.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ember::unicode {

namespace {

struct CodeRange {
  char32_t first;
  char32_t last;
};
struct CombiningClass {
  char32_t code;
  std::uint8_t value;
};
struct Decomposition {
  char32_t code;
  std::uint16_t offset;
  std::uint8_t length;
};
struct Composition {
  char32_t first;
  char32_t second;
  char32_t composite;
};

#include "unicode_tables.inc"

constexpr char32_t kSBase = 0xAC00;
constexpr char32_t kLBase = 0x1100;
constexpr char32_t kVBase = 0x1161;
constexpr char32_t kTBase = 0x11A7;
constexpr char32_t kLCount = 19;
constexpr char32_t kVCount = 21;
constexpr char32_t kTCount = 28;
constexpr char32_t kNCount = kVCount * kTCount;
constexpr char32_t kSCount = kLCount * kNCount;

template <std::size_t N>
bool InRanges(const CodeRange (&table)[N], char32_t c) {
  const auto* end = table + N;
  const auto* it =
      std::upper_bound(table, end, c, [](char32_t v, const CodeRange& r) { return v < r.first; });
  return it != table && c <= (it - 1)->last;
}

std::uint8_t ClassOf(char32_t c) {
  const auto* end = std::end(kCombiningClasses);
  const auto* it = std::lower_bound(std::begin(kCombiningClasses), end, c,
                                    [](const CombiningClass& e, char32_t v) { return e.code < v; });
  return it != end && it->code == c ? it->value : 0;
}

void AppendDecomposed(char32_t c, std::u32string& out) {
  if (c >= kSBase && c < kSBase + kSCount) {
    const char32_t s = c - kSBase;
    out += kLBase + s / kNCount;
    out += kVBase + (s % kNCount) / kTCount;
    if (s % kTCount != 0) out += kTBase + s % kTCount;
    return;
  }
  const auto* end = std::end(kDecompositions);
  const auto* it = std::lower_bound(std::begin(kDecompositions), end, c,
                                    [](const Decomposition& e, char32_t v) { return e.code < v; });
  if (it != end && it->code == c) {
    out.append(kDecompositionPool + it->offset, it->length);
  } else {
    out += c;
  }
}

// The primary composite of two code points, or 0.
char32_t Pair(char32_t first, char32_t second) {
  if (first >= kLBase && first < kLBase + kLCount && second >= kVBase &&
      second < kVBase + kVCount) {
    return kSBase + ((first - kLBase) * kVCount + (second - kVBase)) * kTCount;
  }
  if (first >= kSBase && first < kSBase + kSCount && (first - kSBase) % kTCount == 0 &&
      second > kTBase && second < kTBase + kTCount) {
    return first + (second - kTBase);
  }
  const auto* end = std::end(kCompositions);
  const auto* it =
      std::lower_bound(std::begin(kCompositions), end, Composition{first, second, 0},
                       [](const Composition& a, const Composition& b) {
                         return a.first != b.first ? a.first < b.first : a.second < b.second;
                       });
  return it != end && it->first == first && it->second == second ? it->composite : 0;
}

// The valid character at bytes[i]: its length, or 0 with `consumed` set to how many bytes the one
// replacement covers.
std::size_t Scan(std::string_view bytes, std::size_t i, char32_t* code, std::size_t* consumed) {
  const auto at = [&](std::size_t k) { return static_cast<unsigned char>(bytes[i + k]); };
  const unsigned char lead = at(0);
  *consumed = 1;
  if (lead < 0x80) {
    *code = lead;
    return 1;
  }
  std::size_t length = 0;
  unsigned char low = 0x80;
  unsigned char high = 0xBF;
  if (lead >= 0xC2 && lead <= 0xDF) {
    length = 2;
  } else if (lead >= 0xE0 && lead <= 0xEF) {
    length = 3;
    if (lead == 0xE0) low = 0xA0;
    if (lead == 0xED) high = 0x9F;
  } else if (lead >= 0xF0 && lead <= 0xF4) {
    length = 4;
    if (lead == 0xF0) low = 0x90;
    if (lead == 0xF4) high = 0x8F;
  } else {
    return 0;
  }
  char32_t value = lead & (length == 2 ? 0x1F : length == 3 ? 0x0F : 0x07);
  for (std::size_t k = 1; k < length; ++k) {
    if (i + k >= bytes.size()) return 0;
    const unsigned char b = at(k);
    const unsigned char lo = k == 1 ? low : 0x80;
    const unsigned char hi = k == 1 ? high : 0xBF;
    if (b < lo || b > hi) return 0;
    value = (value << 6) | (b & 0x3F);
    *consumed = k + 1;
  }
  *code = value;
  return length;
}

}  // namespace

std::u32string Decode(std::string_view utf8) {
  std::u32string out;
  out.reserve(utf8.size());
  for (std::size_t i = 0; i < utf8.size();) {
    char32_t code = 0;
    std::size_t consumed = 0;
    const std::size_t length = Scan(utf8, i, &code, &consumed);
    if (length > 0) {
      out += code;
      i += length;
    } else {
      out += kInvalidByteBase + static_cast<unsigned char>(utf8[i]);
      ++i;
    }
  }
  return out;
}

std::string Encode(std::u32string_view text) {
  std::string out;
  out.reserve(text.size());
  for (const char32_t c : text) {
    if (c >= kInvalidByteBase) {
      out += static_cast<char>(c - kInvalidByteBase);
    } else if (c < 0x80) {
      out += static_cast<char>(c);
    } else if (c < 0x800) {
      out += static_cast<char>(0xC0 | (c >> 6));
      out += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
      out += static_cast<char>(0xE0 | (c >> 12));
      out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (c & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (c >> 18));
      out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (c & 0x3F));
    }
  }
  return out;
}

bool IsLetter(char32_t c) { return c < kInvalidByteBase && InRanges(kLetters, c); }
bool IsNumber(char32_t c) { return c < kInvalidByteBase && InRanges(kNumbers, c); }

bool IsWhiteSpace(char32_t c) {
  return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
         c == 0x3000;
}

std::u32string Nfc(std::u32string_view text) {
  // Nothing below U+0300 decomposes or combines, so plain ASCII and Latin-1 text skips the work.
  if (std::all_of(text.begin(), text.end(), [](char32_t c) { return c < 0x300; })) {
    return std::u32string(text);
  }
  std::u32string decomposed;
  decomposed.reserve(text.size());
  for (const char32_t c : text) {
    if (c >= kInvalidByteBase) {
      decomposed += c;
    } else {
      AppendDecomposed(c, decomposed);
    }
  }
  // Canonical ordering: a stable sort of each run of non-starters by combining class.
  for (std::size_t i = 0; i < decomposed.size();) {
    if (ClassOf(decomposed[i]) == 0) {
      ++i;
      continue;
    }
    std::size_t j = i;
    while (j < decomposed.size() && ClassOf(decomposed[j]) != 0) ++j;
    std::stable_sort(decomposed.begin() + static_cast<std::ptrdiff_t>(i),
                     decomposed.begin() + static_cast<std::ptrdiff_t>(j),
                     [](char32_t a, char32_t b) { return ClassOf(a) < ClassOf(b); });
    i = j;
  }
  // Composition: a character joins the last starter unless something in between is a starter or
  // has a combining class at least its own.
  std::u32string out;
  out.reserve(decomposed.size());
  std::size_t starter = std::u32string::npos;
  int last = -1;
  for (const char32_t c : decomposed) {
    const int cls = ClassOf(c);
    if (starter != std::u32string::npos && (last == -1 || (last != 0 && last < cls))) {
      const char32_t composite = Pair(out[starter], c);
      if (composite != 0) {
        out[starter] = composite;
        continue;
      }
    }
    if (cls == 0) {
      starter = out.size();
      last = -1;
    } else {
      last = cls;
    }
    out += c;
  }
  return out;
}

std::string Lossy(std::string_view bytes) {
  std::string out;
  out.reserve(bytes.size());
  for (std::size_t i = 0; i < bytes.size();) {
    char32_t code = 0;
    std::size_t consumed = 0;
    const std::size_t length = Scan(bytes, i, &code, &consumed);
    if (length > 0) {
      out.append(bytes.substr(i, length));
      i += length;
    } else {
      out += "\xEF\xBF\xBD";
      i += consumed;
    }
  }
  return out;
}

}  // namespace ember::unicode
