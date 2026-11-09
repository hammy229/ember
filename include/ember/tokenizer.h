#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ember/status_or.h"

namespace ember {

// The Llama 2 tokenizer, read from the tokenizer.json Hugging Face saves next to a model: byte pair
// encoding over SentencePiece pieces, with unknown characters falling back to raw bytes.
//
// It turns text into exactly the ids Hugging Face's `tokenizers` gives, checked in CI against
// tests/reference/hf_tokens_tinyllama-15M.json:
//
// 1. Normalize: put "▁" in front and turn every space into "▁".
// 2. Split out special tokens (<s>, </s>, <unk>). Their text is normalized too, so "<s>" only
//    counts at the start of the text or right after a space, as in Hugging Face.
// 3. Split the rest into characters. A character with no piece of its own becomes one piece per
//    UTF-8 byte, <0x00> to <0xFF>.
// 4. Merge neighbouring pieces, always the pair whose merge rule comes first, until none apply.
//
// Any tokenizer.json that would need other steps is refused, rather than encoded slightly wrong.
class Tokenizer {
 public:
  static StatusOr<Tokenizer> FromJson(std::string_view tokenizer_json);
  static StatusOr<Tokenizer> Load(const std::string& path);

  // Text to ids. `add_bos` puts the begin-of-sequence id first, as the model expects at the start
  // of a prompt. Text that is not valid UTF-8 is still encoded, its stray bytes as byte pieces.
  std::vector<std::int32_t> Encode(std::string_view text, bool add_bos) const;

  // Ids back to text. Byte pieces that do not form valid UTF-8 become U+FFFD, one per byte. Fails
  // on an id outside the vocabulary.
  StatusOr<std::string> Decode(std::span<const std::int32_t> ids) const;

  std::size_t vocab_size() const { return pieces_.size(); }
  std::int32_t bos_id() const { return bos_id_; }
  std::int32_t eos_id() const { return eos_id_; }
  // The piece an id stands for, as the vocabulary writes it ("▁the", "<0x0A>", "<s>").
  const std::string& piece(std::int32_t id) const { return pieces_[static_cast<std::size_t>(id)]; }

 private:
  struct Merge {
    std::int32_t rank;
    std::int32_t merged;
  };
  struct Special {
    std::string pattern;  // the token's text after normalization, which is what gets matched
    std::int32_t id;
  };

  void EncodeSegment(std::string_view normalized, std::vector<std::int32_t>& out) const;

  static std::uint64_t PairKey(std::int32_t left, std::int32_t right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
           static_cast<std::uint32_t>(right);
  }

  std::vector<std::string> pieces_;
  std::unordered_map<std::string, std::int32_t> ids_;
  std::unordered_map<std::uint64_t, Merge> merges_;
  std::array<std::int32_t, 256> byte_ids_{};
  std::vector<Special> specials_;
  std::vector<bool> is_special_;
  std::int32_t bos_id_ = -1;
  std::int32_t eos_id_ = -1;
};

}  // namespace ember
