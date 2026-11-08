#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ember/dtype.h"
#include "ember/shape.h"
#include "ember/status_or.h"
#include "ember/tensor.h"

namespace ember {

// Reads the safetensors format Hugging Face uses for weights:
//
//   8 bytes   N, the header length, little-endian
//   N bytes   a JSON object with one entry per tensor and an optional "__metadata__" of strings:
//             "name": {"dtype": "F16", "shape": [rows, columns], "data_offsets": [begin, end]}
//   the rest  raw tensor bytes, with offsets counted from here
//
// Nothing is copied: each tensor is a span into the bytes passed in, which must outlive this.
//
// Every number in the header is checked before it is used, because a file is untrusted input:
// - the header must fit in the file and be at most kMaxHeaderBytes,
// - each tensor's offsets must lie inside the data and match its shape and element type exactly,
// - tensors must cover the data with no gap and no overlap, as the format requires, so no byte can
//   belong to two tensors and no hidden bytes can ride along.
class Safetensors {
 public:
  // The reference implementation allows 100 MB. Ember allows 16 MB, which is still hundreds of
  // times the header of an 8B model and keeps a hostile header from eating a browser tab's memory.
  static constexpr std::size_t kMaxHeaderBytes = std::size_t{16} << 20;

  struct Entry {
    std::string name;
    DType dtype;
    Shape shape;
    std::span<const std::byte> bytes;
  };

  static StatusOr<Safetensors> Parse(std::span<const std::byte> file);

  // In the order of their bytes in the file.
  std::span<const Entry> tensors() const { return tensors_; }
  // Null if there is no tensor by that name.
  const Entry* Find(std::string_view name) const;

  // The string pairs under "__metadata__", in file order.
  std::span<const std::pair<std::string, std::string>> metadata() const { return metadata_; }

  // A view of the entry's bytes. It fails if the bytes are not aligned for the element type, which
  // the format does not promise. Reading f16 that way is fine, since F16ToF32 takes any alignment.
  static StatusOr<TensorView> View(const Entry& entry);

 private:
  std::vector<Entry> tensors_;
  std::vector<std::pair<std::string, std::string>> metadata_;
};

}  // namespace ember
