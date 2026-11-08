#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ember/status_or.h"

namespace ember {

// Just enough JSON to read a safetensors header and a model's config.json, both of which may come
// from a file nobody checked. It follows RFC 8259 strictly and refuses anything else, rather than
// guessing what a broken file meant:
//
// - Nesting deeper than kMaxDepth is refused, so a file of a million '[' cannot overflow the stack.
// - An object with the same key twice is refused, because two readers of one file could then
//   disagree on which value counts.
// - Strings must be valid escapes, and \u escapes must pair up surrogates. They come out as UTF-8.
class Json {
 public:
  enum class Kind : std::uint8_t { kNull, kBool, kNumber, kString, kArray, kObject };

  static constexpr std::size_t kMaxDepth = 64;

  static StatusOr<Json> Parse(std::string_view text);

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::kNull; }
  bool is_bool() const { return kind_ == Kind::kBool; }
  bool is_number() const { return kind_ == Kind::kNumber; }
  bool is_string() const { return kind_ == Kind::kString; }
  bool is_array() const { return kind_ == Kind::kArray; }
  bool is_object() const { return kind_ == Kind::kObject; }

  // These read the value as one kind, and asking a value of another kind is a bug that aborts.
  // Check the kind first when reading a file.
  bool as_bool() const;
  double as_number() const;
  const std::string& as_string() const;

  // The number exactly, if it was written as a whole number from 0 to 2^64 - 1 with no fraction or
  // exponent. Byte offsets can pass 2^53, where a double stops counting every integer, so they are
  // read this way rather than through as_number().
  std::optional<std::uint64_t> as_uint64() const;

  // An array's items, or an object's values in the order the file wrote them.
  std::span<const Json> items() const;
  // An object's keys, in the same order as items().
  std::span<const std::string> keys() const;
  std::size_t size() const { return items().size(); }

  // A member of an object, or null if there is none. Asking a value that is not an object is a bug.
  const Json* Find(std::string_view key) const;

 private:
  friend class JsonParser;

  Kind kind_ = Kind::kNull;
  bool bool_ = false;
  double number_ = 0;
  std::optional<std::uint64_t> uint64_;
  std::string string_;
  std::vector<std::string> keys_;
  std::vector<Json> items_;
};

}  // namespace ember
