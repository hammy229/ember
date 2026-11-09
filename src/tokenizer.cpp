#include "ember/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ember/json.h"
#include "ember/mapped_file.h"

namespace ember {

namespace {

constexpr std::string_view kSpacePiece = "\xE2\x96\x81";  // U+2581, SentencePiece's space

// How many bytes the UTF-8 character starting at text[i] takes, or 0 if the bytes there are not a
// valid character (a stray continuation byte, an overlong form, a surrogate, or past U+10FFFF).
std::size_t Utf8Length(std::string_view text, std::size_t i) {
  const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(text[i + k]); };
  const auto continuation = [&](std::size_t k) {
    return i + k < text.size() && (byte(k) & 0xC0) == 0x80;
  };
  const unsigned char lead = byte(0);
  if (lead < 0x80) return 1;
  if (lead >= 0xC2 && lead <= 0xDF) return continuation(1) ? 2 : 0;
  if (lead >= 0xE0 && lead <= 0xEF) {
    if (!continuation(1) || !continuation(2)) return 0;
    if (lead == 0xE0 && byte(1) < 0xA0) return 0;   // overlong
    if (lead == 0xED && byte(1) >= 0xA0) return 0;  // surrogate
    return 3;
  }
  if (lead >= 0xF0 && lead <= 0xF4) {
    if (!continuation(1) || !continuation(2) || !continuation(3)) return 0;
    if (lead == 0xF0 && byte(1) < 0x90) return 0;   // overlong
    if (lead == 0xF4 && byte(1) >= 0x90) return 0;  // past U+10FFFF
    return 4;
  }
  return 0;
}

bool IsValidUtf8(std::string_view text) {
  for (std::size_t i = 0; i < text.size();) {
    const std::size_t length = Utf8Length(text, i);
    if (length == 0) return false;
    i += length;
  }
  return true;
}

std::string Normalize(std::string_view text) {
  std::string out(kSpacePiece);
  out.reserve(text.size() + 3);
  for (const char c : text) {
    if (c == ' ') {
      out += kSpacePiece;
    } else {
      out += c;
    }
  }
  return out;
}

// The byte a piece like "<0x0A>" stands for, or -1 if it is not one.
int ByteOfPiece(std::string_view piece) {
  if (piece.size() != 6 || piece.substr(0, 3) != "<0x" || piece[5] != '>') return -1;
  int value = 0;
  for (const char c : piece.substr(3, 2)) {
    value *= 16;
    if (c >= '0' && c <= '9') {
      value += c - '0';
    } else if (c >= 'A' && c <= 'F') {
      value += c - 'A' + 10;
    } else {
      return -1;
    }
  }
  return value;
}

Status Unsupported(std::string_view what) {
  return Status::InvalidFormat("tokenizer.json: " + std::string(what) +
                               " is not the Llama 2 tokenizer Ember supports");
}

const Json* Member(const Json& object, std::string_view key) {
  return object.is_object() ? object.Find(key) : nullptr;
}

bool IsString(const Json* value, std::string_view expected) {
  return value != nullptr && value->is_string() && value->as_string() == expected;
}

bool IsNullOrMissing(const Json* value) { return value == nullptr || value->is_null(); }

bool IsFalseOrMissing(const Json* value) {
  return value == nullptr || value->is_null() || (value->is_bool() && !value->as_bool());
}

// The normalizer must be exactly: prepend "▁", then replace " " with "▁".
bool IsLlamaNormalizer(const Json* normalizer) {
  if (normalizer == nullptr || !IsString(Member(*normalizer, "type"), "Sequence")) return false;
  const Json* steps = Member(*normalizer, "normalizers");
  if (steps == nullptr || !steps->is_array() || steps->size() != 2) return false;
  const Json& prepend = steps->items()[0];
  const Json& replace = steps->items()[1];
  if (!IsString(Member(prepend, "type"), "Prepend") ||
      !IsString(Member(prepend, "prepend"), kSpacePiece)) {
    return false;
  }
  const Json* pattern = Member(replace, "pattern");
  return IsString(Member(replace, "type"), "Replace") && pattern != nullptr &&
         IsString(Member(*pattern, "String"), " ") &&
         IsString(Member(replace, "content"), kSpacePiece);
}

// The decoder must be exactly: "▁" to " ", byte fallback, fuse, strip one leading space.
bool IsLlamaDecoder(const Json* decoder) {
  if (decoder == nullptr || !IsString(Member(*decoder, "type"), "Sequence")) return false;
  const Json* steps = Member(*decoder, "decoders");
  if (steps == nullptr || !steps->is_array() || steps->size() != 4) return false;
  const auto s = steps->items();
  const Json* pattern = Member(s[0], "pattern");
  const Json* start = Member(s[3], "start");
  const Json* stop = Member(s[3], "stop");
  return IsString(Member(s[0], "type"), "Replace") && pattern != nullptr &&
         IsString(Member(*pattern, "String"), kSpacePiece) &&
         IsString(Member(s[0], "content"), " ") && IsString(Member(s[1], "type"), "ByteFallback") &&
         IsString(Member(s[2], "type"), "Fuse") && IsString(Member(s[3], "type"), "Strip") &&
         IsString(Member(s[3], "content"), " ") && start != nullptr && start->as_uint64() == 1 &&
         stop != nullptr && stop->as_uint64() == 0;
}

}  // namespace

StatusOr<Tokenizer> Tokenizer::Load(const std::string& path) {
  EMBER_ASSIGN_OR_RETURN(const MappedFile file, MappedFile::Open(path));
  const auto bytes = file.bytes();
  return FromJson(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

StatusOr<Tokenizer> Tokenizer::FromJson(std::string_view tokenizer_json) {
  EMBER_ASSIGN_OR_RETURN(const Json root, Json::Parse(tokenizer_json));
  if (!root.is_object()) return Unsupported("a top level that is not an object");
  const Json* model = Member(root, "model");
  if (model == nullptr || !IsString(Member(*model, "type"), "BPE")) {
    return Unsupported("a model that is not BPE");
  }
  const Json* byte_fallback = Member(*model, "byte_fallback");
  if (byte_fallback == nullptr || !byte_fallback->is_bool() || !byte_fallback->as_bool()) {
    return Unsupported("BPE without byte fallback");
  }
  if (!IsNullOrMissing(Member(*model, "dropout"))) return Unsupported("BPE dropout");
  for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    const Json* value = Member(*model, key);
    if (!IsNullOrMissing(value) && !IsString(value, "")) return Unsupported(key);
  }
  if (!IsFalseOrMissing(Member(*model, "ignore_merges"))) return Unsupported("ignore_merges");
  if (!IsLlamaNormalizer(Member(root, "normalizer"))) return Unsupported("this normalizer");
  if (!IsNullOrMissing(Member(root, "pre_tokenizer"))) return Unsupported("a pre-tokenizer");
  if (!IsLlamaDecoder(Member(root, "decoder"))) return Unsupported("this decoder");

  Tokenizer t;
  const Json* vocab = Member(*model, "vocab");
  if (vocab == nullptr || !vocab->is_object() || vocab->size() == 0) {
    return Status::InvalidFormat("tokenizer.json: model.vocab must be a non-empty object");
  }
  const std::size_t n = vocab->size();
  if (n > (std::size_t{1} << 30)) return Status::InvalidFormat("tokenizer.json: vocab too big");
  t.pieces_.assign(n, std::string());
  std::vector<bool> seen(n, false);
  for (std::size_t i = 0; i < n; ++i) {
    const std::optional<std::uint64_t> id = vocab->items()[i].as_uint64();
    if (!id || *id >= n || seen[*id]) {
      return Status::InvalidFormat("tokenizer.json: vocab ids must be 0 to " +
                                   std::to_string(n - 1) + ", each used once");
    }
    seen[*id] = true;
    const std::string& piece = vocab->keys()[i];
    t.pieces_[*id] = piece;
    t.ids_.emplace(piece, static_cast<std::int32_t>(*id));
  }
  t.byte_ids_.fill(-1);
  for (std::size_t id = 0; id < n; ++id) {
    const int byte = ByteOfPiece(t.pieces_[id]);
    if (byte >= 0) t.byte_ids_[static_cast<std::size_t>(byte)] = static_cast<std::int32_t>(id);
  }
  for (const std::int32_t id : t.byte_ids_) {
    if (id < 0) return Status::InvalidFormat("tokenizer.json: a byte piece <0x..> is missing");
  }

  const Json* merges = Member(*model, "merges");
  if (merges == nullptr || !merges->is_array()) {
    return Status::InvalidFormat("tokenizer.json: model.merges must be an array");
  }
  for (std::size_t rank = 0; rank < merges->size(); ++rank) {
    const Json& merge = merges->items()[rank];
    std::string left;
    std::string right;
    if (merge.is_string()) {
      const std::string& text = merge.as_string();
      const std::size_t space = text.find(' ');
      if (space == std::string::npos || text.find(' ', space + 1) != std::string::npos) {
        return Status::InvalidFormat("tokenizer.json: merge \"" + text + "\" is not two pieces");
      }
      left = text.substr(0, space);
      right = text.substr(space + 1);
    } else if (merge.is_array() && merge.size() == 2 && merge.items()[0].is_string() &&
               merge.items()[1].is_string()) {
      left = merge.items()[0].as_string();
      right = merge.items()[1].as_string();
    } else {
      return Status::InvalidFormat("tokenizer.json: a merge is not two pieces");
    }
    const auto l = t.ids_.find(left);
    const auto r = t.ids_.find(right);
    const auto m = t.ids_.find(left + right);
    if (l == t.ids_.end() || r == t.ids_.end() || m == t.ids_.end()) {
      return Status::InvalidFormat("tokenizer.json: merge \"" + left + " " + right +
                                   "\" uses a piece that is not in the vocab");
    }
    // A pair listed twice keeps its first, highest priority rank, as Hugging Face does.
    t.merges_.emplace(PairKey(l->second, r->second),
                      Merge{static_cast<std::int32_t>(rank), m->second});
  }

  t.is_special_.assign(n, false);
  const Json* added = Member(root, "added_tokens");
  if (added != nullptr && !added->is_null()) {
    if (!added->is_array()) return Status::InvalidFormat("tokenizer.json: added_tokens");
    for (const Json& token : added->items()) {
      const Json* content = Member(token, "content");
      const Json* id_value = Member(token, "id");
      const std::optional<std::uint64_t> id =
          id_value != nullptr ? id_value->as_uint64() : std::nullopt;
      if (content == nullptr || !content->is_string() || content->as_string().empty() || !id ||
          *id >= n || t.pieces_[*id] != content->as_string()) {
        return Status::InvalidFormat("tokenizer.json: an added token does not match the vocab");
      }
      const Json* normalized = Member(token, "normalized");
      if (normalized == nullptr || !normalized->is_bool() || !normalized->as_bool() ||
          !IsFalseOrMissing(Member(token, "single_word")) ||
          !IsFalseOrMissing(Member(token, "lstrip")) ||
          !IsFalseOrMissing(Member(token, "rstrip"))) {
        return Unsupported("an added token that is not normalized, or strips or matches words");
      }
      t.specials_.push_back(
          Special{Normalize(content->as_string()), static_cast<std::int32_t>(*id)});
      t.is_special_[*id] = true;
      if (content->as_string() == "<s>") t.bos_id_ = static_cast<std::int32_t>(*id);
      if (content->as_string() == "</s>") t.eos_id_ = static_cast<std::int32_t>(*id);
    }
  }
  if (t.bos_id_ < 0 || t.eos_id_ < 0) {
    return Status::InvalidFormat("tokenizer.json: <s> and </s> must be added tokens");
  }
  return t;
}

void Tokenizer::EncodeSegment(std::string_view text, std::vector<std::int32_t>& out) const {
  if (text.empty()) return;

  // One symbol per character, or per byte where a character has no piece. They form a linked list
  // so merging two is constant time.
  struct Symbol {
    std::int32_t id;
    int prev;
    int next;
    bool alive;
  };
  std::vector<Symbol> symbols;
  symbols.reserve(text.size());
  const auto push = [&](std::int32_t id) {
    const int index = static_cast<int>(symbols.size());
    symbols.push_back(Symbol{id, index - 1, index + 1, true});
  };
  for (std::size_t i = 0; i < text.size();) {
    const std::size_t length = Utf8Length(text, i);
    if (length > 0) {
      const auto found = ids_.find(std::string(text.substr(i, length)));
      if (found != ids_.end()) {
        push(found->second);
      } else {
        for (std::size_t k = 0; k < length; ++k) {
          push(byte_ids_[static_cast<unsigned char>(text[i + k])]);
        }
      }
      i += length;
    } else {
      push(byte_ids_[static_cast<unsigned char>(text[i])]);
      ++i;
    }
  }
  symbols.back().next = -1;
  const auto at = [&](int index) -> Symbol& { return symbols[static_cast<std::size_t>(index)]; };

  // Candidate merges, best first: lowest rank, then leftmost, which is Hugging Face's order. An
  // entry goes stale when either side has since merged with something else; it is then skipped.
  struct Candidate {
    std::int32_t rank;
    int left;
    int right;
    std::int32_t left_id;
    std::int32_t right_id;
    std::int32_t merged;
    bool operator>(const Candidate& other) const {
      return rank != other.rank ? rank > other.rank : left > other.left;
    }
  };
  std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue;
  const auto consider = [&](int left, int right) {
    if (left < 0 || right < 0) return;
    const auto found = merges_.find(PairKey(at(left).id, at(right).id));
    if (found == merges_.end()) return;
    queue.push(Candidate{found->second.rank, left, right, at(left).id, at(right).id,
                         found->second.merged});
  };
  for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) consider(i, i + 1);

  while (!queue.empty()) {
    const Candidate c = queue.top();
    queue.pop();
    Symbol& left = at(c.left);
    Symbol& right = at(c.right);
    if (!left.alive || !right.alive || left.next != c.right || left.id != c.left_id ||
        right.id != c.right_id) {
      continue;
    }
    left.id = c.merged;
    right.alive = false;
    left.next = right.next;
    if (right.next >= 0) at(right.next).prev = c.left;
    consider(left.prev, c.left);
    consider(c.left, left.next);
  }
  for (int i = 0; i >= 0; i = at(i).next) out.push_back(at(i).id);
}

std::vector<std::int32_t> Tokenizer::Encode(std::string_view text, bool add_bos) const {
  std::vector<std::int32_t> out;
  if (add_bos) out.push_back(bos_id_);
  if (text.empty()) return out;
  const std::string normalized = Normalize(text);
  const std::string_view view(normalized);
  std::size_t segment_start = 0;
  for (std::size_t i = 0; i < view.size();) {
    const Special* match = nullptr;
    for (const Special& special : specials_) {
      if (view.substr(i, special.pattern.size()) == special.pattern &&
          (match == nullptr || special.pattern.size() > match->pattern.size())) {
        match = &special;
      }
    }
    if (match == nullptr) {
      ++i;
      continue;
    }
    EncodeSegment(view.substr(segment_start, i - segment_start), out);
    out.push_back(match->id);
    i += match->pattern.size();
    segment_start = i;
  }
  EncodeSegment(view.substr(segment_start), out);
  return out;
}

StatusOr<std::string> Tokenizer::Decode(std::span<const std::int32_t> ids) const {
  std::string text;
  std::string pending;  // bytes from a run of byte pieces, not yet known to be valid UTF-8
  std::size_t pending_count = 0;
  const auto flush = [&] {
    if (IsValidUtf8(pending)) {
      text += pending;
    } else {
      for (std::size_t k = 0; k < pending_count; ++k) text += "\xEF\xBF\xBD";  // U+FFFD
    }
    pending.clear();
    pending_count = 0;
  };
  for (const std::int32_t id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= pieces_.size()) {
      return Status::InvalidArgument("token id " + std::to_string(id) +
                                     " is outside the vocabulary of " +
                                     std::to_string(pieces_.size()));
    }
    const std::string& piece = pieces_[static_cast<std::size_t>(id)];
    const int byte = is_special_[static_cast<std::size_t>(id)] ? -1 : ByteOfPiece(piece);
    if (byte >= 0) {
      pending += static_cast<char>(byte);
      ++pending_count;
      continue;
    }
    flush();
    if (is_special_[static_cast<std::size_t>(id)]) {
      text += piece;
      continue;
    }
    for (std::size_t i = 0; i < piece.size();) {
      if (std::string_view(piece).substr(i, kSpacePiece.size()) == kSpacePiece) {
        text += ' ';
        i += kSpacePiece.size();
      } else {
        text += piece[i];
        ++i;
      }
    }
  }
  flush();
  if (!text.empty() && text.front() == ' ') text.erase(0, 1);
  return text;
}

}  // namespace ember
