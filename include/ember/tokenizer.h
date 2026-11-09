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

// A byte pair encoding tokenizer read from the tokenizer.json Hugging Face saves next to a model.
// It turns text into exactly the ids Hugging Face's `tokenizers` gives, and supports the two kinds
// Ember's models use. Any tokenizer.json that would need other steps is refused, rather than
// encoded slightly wrong.
//
// Llama 2 (SentencePiece style), checked against tests/reference/hf_tokens_tinyllama-15M.json:
// 1. Normalize: put "▁" in front and turn every space into "▁".
// 2. Split out special tokens (<s>, </s>, <unk>). Their text is normalized too, so "<s>" only
//    counts at the start of the text or right after a space, as in Hugging Face.
// 3. Split the rest into characters. A character with no piece of its own becomes one piece per
//    UTF-8 byte, <0x00> to <0xFF>.
// 4. Merge neighbouring pieces, always the pair whose merge rule comes first, until none apply.
//
// Qwen3 (GPT-2 style byte level), checked against tests/reference/hf_tokens_qwen3-0.6b.json:
// 1. Split out added tokens (<|im_start|> and the rest) wherever they appear in the raw text.
// 2. Normalize the rest to Unicode NFC.
// 3. Split it into words, single digits, punctuation runs, and whitespace, by Qwen's regex.
// 4. Write each piece's UTF-8 bytes as the 256 printable stand-in characters GPT-2 uses, then
//    merge as above.
class Tokenizer {
 public:
  enum class Kind : std::uint8_t { kSentencePiece, kByteLevel };

  static StatusOr<Tokenizer> FromJson(std::string_view tokenizer_json);
  static StatusOr<Tokenizer> Load(const std::string& path);

  // Text to ids. `add_bos` puts the begin-of-sequence id first, as Llama expects at the start of a
  // prompt; Qwen has none, so it is ignored there. Text that is not valid UTF-8 is still encoded.
  std::vector<std::int32_t> Encode(std::string_view text, bool add_bos) const;

  // Ids back to text. Bytes that do not form valid UTF-8 become U+FFFD, as in Hugging Face. Fails
  // on an id outside the vocabulary.
  StatusOr<std::string> Decode(std::span<const std::int32_t> ids) const;

  Kind kind() const { return kind_; }
  std::size_t vocab_size() const { return pieces_.size(); }
  // -1 when the tokenizer has none.
  std::int32_t bos_id() const { return bos_id_; }
  std::int32_t eos_id() const { return eos_id_; }
  // The tokens that end a reply: </s> and <s> for Llama (small story models end a story with <s>),
  // <|im_end|> and <|endoftext|> for Qwen.
  const std::vector<std::int32_t>& stop_ids() const { return stop_ids_; }
  // The piece an id stands for, as the vocabulary writes it ("▁the", "<0x0A>", "Ġthe", "<s>").
  const std::string& piece(std::int32_t id) const { return pieces_[static_cast<std::size_t>(id)]; }

 private:
  struct Merge {
    std::int32_t rank;
    std::int32_t merged;
  };
  struct Special {
    std::string pattern;  // the text that gets matched
    std::int32_t id;
  };

  void EncodeSentencePiece(std::string_view normalized, std::vector<std::int32_t>& out) const;
  void EncodeByteLevel(std::string_view text, std::vector<std::int32_t>& out) const;
  void MergeSymbols(std::vector<std::int32_t> symbols, std::vector<std::int32_t>& out) const;

  static std::uint64_t PairKey(std::int32_t left, std::int32_t right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
           static_cast<std::uint32_t>(right);
  }

  Kind kind_ = Kind::kSentencePiece;
  std::vector<std::string> pieces_;
  std::unordered_map<std::string, std::int32_t> ids_;
  std::unordered_map<std::uint64_t, Merge> merges_;
  // SentencePiece: the id of <0xXX> for each byte. Byte level: the id of each byte's stand-in.
  std::array<std::int32_t, 256> byte_ids_{};
  // Byte level: the byte each stand-in character (by code point) means.
  std::unordered_map<char32_t, unsigned char> stand_in_bytes_;
  std::vector<Special> specials_;
  std::vector<bool> is_special_;
  std::vector<std::int32_t> stop_ids_;
  std::int32_t bos_id_ = -1;
  std::int32_t eos_id_ = -1;
};

}  // namespace ember
