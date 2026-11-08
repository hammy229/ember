#include "ember/json.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "ember/check.h"

namespace ember {

class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : text_(text) {}

  StatusOr<Json> ParseDocument() {
    SkipWhitespace();
    EMBER_ASSIGN_OR_RETURN(Json value, ParseValue(0));
    SkipWhitespace();
    if (position_ != text_.size()) return Error("unexpected text after the value");
    return value;
  }

 private:
  Status Error(std::string_view what) const {
    return Status::InvalidFormat("JSON: " + std::string(what) + " at byte " +
                                 std::to_string(position_));
  }

  bool AtEnd() const { return position_ >= text_.size(); }
  char Peek() const { return text_[position_]; }

  void SkipWhitespace() {
    while (!AtEnd()) {
      const char c = Peek();
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') return;
      ++position_;
    }
  }

  bool Consume(std::string_view word) {
    if (text_.substr(position_, word.size()) != word) return false;
    position_ += word.size();
    return true;
  }

  StatusOr<Json> ParseValue(std::size_t depth) {
    if (AtEnd()) return Error("expected a value");
    Json value;
    switch (Peek()) {
      case '{':
        return ParseObject(depth);
      case '[':
        return ParseArray(depth);
      case '"': {
        value.kind_ = Json::Kind::kString;
        EMBER_ASSIGN_OR_RETURN(value.string_, ParseString());
        return value;
      }
      case 't':
      case 'f':
        value.kind_ = Json::Kind::kBool;
        if (Consume("true")) {
          value.bool_ = true;
          return value;
        }
        if (Consume("false")) return value;
        return Error("expected true or false");
      case 'n':
        if (Consume("null")) return value;
        return Error("expected null");
      default:
        return ParseNumber();
    }
  }

  StatusOr<Json> ParseObject(std::size_t depth) {
    if (depth >= Json::kMaxDepth) return Error("nested too deeply");
    ++position_;  // '{'
    Json object;
    object.kind_ = Json::Kind::kObject;
    SkipWhitespace();
    if (!AtEnd() && Peek() == '}') {
      ++position_;
      return object;
    }
    while (true) {
      SkipWhitespace();
      if (AtEnd() || Peek() != '"') return Error("expected a key in quotes");
      EMBER_ASSIGN_OR_RETURN(std::string key, ParseString());
      if (object.Find(key) != nullptr) return Error("duplicate key \"" + key + "\"");
      SkipWhitespace();
      if (AtEnd() || Peek() != ':') return Error("expected ':' after a key");
      ++position_;
      SkipWhitespace();
      EMBER_ASSIGN_OR_RETURN(Json member, ParseValue(depth + 1));
      object.keys_.push_back(std::move(key));
      object.items_.push_back(std::move(member));
      SkipWhitespace();
      if (AtEnd()) return Error("unterminated object");
      if (Peek() == '}') {
        ++position_;
        return object;
      }
      if (Peek() != ',') return Error("expected ',' or '}' in an object");
      ++position_;
    }
  }

  StatusOr<Json> ParseArray(std::size_t depth) {
    if (depth >= Json::kMaxDepth) return Error("nested too deeply");
    ++position_;  // '['
    Json array;
    array.kind_ = Json::Kind::kArray;
    SkipWhitespace();
    if (!AtEnd() && Peek() == ']') {
      ++position_;
      return array;
    }
    while (true) {
      SkipWhitespace();
      EMBER_ASSIGN_OR_RETURN(Json item, ParseValue(depth + 1));
      array.items_.push_back(std::move(item));
      SkipWhitespace();
      if (AtEnd()) return Error("unterminated array");
      if (Peek() == ']') {
        ++position_;
        return array;
      }
      if (Peek() != ',') return Error("expected ',' or ']' in an array");
      ++position_;
    }
  }

  // Four hex digits after a \u.
  std::optional<std::uint32_t> ParseHex4() {
    if (text_.size() - position_ < 4) return std::nullopt;
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = text_[position_++];
      value <<= 4;
      if (c >= '0' && c <= '9') {
        value |= static_cast<std::uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        value |= static_cast<std::uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        value |= static_cast<std::uint32_t>(c - 'A' + 10);
      } else {
        return std::nullopt;
      }
    }
    return value;
  }

  static void AppendUtf8(std::uint32_t code_point, std::string& out) {
    if (code_point < 0x80) {
      out.push_back(static_cast<char>(code_point));
    } else if (code_point < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
      out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else if (code_point < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
      out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
      out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    }
  }

  StatusOr<std::string> ParseString() {
    ++position_;  // the opening quote
    std::string out;
    while (true) {
      if (AtEnd()) return Error("unterminated string");
      const char c = text_[position_++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 0x20) return Error("control character in a string");
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (AtEnd()) return Error("unterminated escape");
      switch (text_[position_++]) {
        case '"':
          out.push_back('"');
          break;
        case '\\':
          out.push_back('\\');
          break;
        case '/':
          out.push_back('/');
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          std::optional<std::uint32_t> unit = ParseHex4();
          if (!unit) return Error("bad \\u escape");
          std::uint32_t code_point = *unit;
          if (code_point >= 0xDC00 && code_point <= 0xDFFF) return Error("lone low surrogate");
          if (code_point >= 0xD800 && code_point <= 0xDBFF) {
            // A high surrogate only means something followed by a low one.
            if (!Consume("\\u")) return Error("lone high surrogate");
            const std::optional<std::uint32_t> low = ParseHex4();
            if (!low || *low < 0xDC00 || *low > 0xDFFF) return Error("lone high surrogate");
            code_point = 0x10000 + ((code_point - 0xD800) << 10) + (*low - 0xDC00);
          }
          AppendUtf8(code_point, out);
          break;
        }
        default:
          return Error("unknown escape");
      }
    }
  }

  static bool IsDigit(char c) { return c >= '0' && c <= '9'; }

  StatusOr<Json> ParseNumber() {
    const std::size_t start = position_;
    if (!AtEnd() && Peek() == '-') ++position_;
    if (AtEnd() || !IsDigit(Peek())) return Error("expected a value");
    // A leading zero must stand alone, so 012 is not a number.
    if (Peek() == '0') {
      ++position_;
    } else {
      while (!AtEnd() && IsDigit(Peek())) ++position_;
    }
    bool whole = true;
    if (!AtEnd() && Peek() == '.') {
      whole = false;
      ++position_;
      if (AtEnd() || !IsDigit(Peek())) return Error("expected a digit after '.'");
      while (!AtEnd() && IsDigit(Peek())) ++position_;
    }
    if (!AtEnd() && (Peek() == 'e' || Peek() == 'E')) {
      whole = false;
      ++position_;
      if (!AtEnd() && (Peek() == '+' || Peek() == '-')) ++position_;
      if (AtEnd() || !IsDigit(Peek())) return Error("expected a digit in the exponent");
      while (!AtEnd() && IsDigit(Peek())) ++position_;
    }

    const std::string lexeme(text_.substr(start, position_ - start));
    Json value;
    value.kind_ = Json::Kind::kNumber;
    // strtod needs a terminated string, which is why the lexeme was copied out. The grammar above
    // already proved it is a number, so strtod reads all of it.
    value.number_ = std::strtod(lexeme.c_str(), nullptr);
    if (!std::isfinite(value.number_)) return Error("number out of range");

    if (whole && lexeme[0] != '-') {
      std::uint64_t exact = 0;
      bool fits = true;
      for (const char digit : lexeme) {
        const auto d = static_cast<std::uint64_t>(digit - '0');
        if (exact > (UINT64_MAX - d) / 10) {
          fits = false;
          break;
        }
        exact = exact * 10 + d;
      }
      if (fits) value.uint64_ = exact;
    }
    return value;
  }

  std::string_view text_;
  std::size_t position_ = 0;
};

StatusOr<Json> Json::Parse(std::string_view text) { return JsonParser(text).ParseDocument(); }

bool Json::as_bool() const {
  EMBER_CHECK(is_bool(), "read a JSON value as a bool that is not one");
  return bool_;
}

double Json::as_number() const {
  EMBER_CHECK(is_number(), "read a JSON value as a number that is not one");
  return number_;
}

const std::string& Json::as_string() const {
  EMBER_CHECK(is_string(), "read a JSON value as a string that is not one");
  return string_;
}

std::optional<std::uint64_t> Json::as_uint64() const {
  if (!is_number()) return std::nullopt;
  return uint64_;
}

std::span<const Json> Json::items() const { return items_; }

std::span<const std::string> Json::keys() const { return keys_; }

const Json* Json::Find(std::string_view key) const {
  EMBER_CHECK(is_object(), "looked up a key in a JSON value that is not an object");
  for (std::size_t i = 0; i < keys_.size(); ++i) {
    if (keys_[i] == key) return &items_[i];
  }
  return nullptr;
}

}  // namespace ember
