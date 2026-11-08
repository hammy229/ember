#pragma once

// Builds safetensors files in memory for tests, so no test needs real weights.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ember::testing {

// The raw format: an 8 byte little-endian length, the header text, then the data.
inline std::vector<std::byte> RawSafetensors(const std::string& header,
                                             const std::vector<std::byte>& data) {
  std::vector<std::byte> file(8 + header.size() + data.size());
  std::uint64_t length = header.size();
  for (std::size_t i = 0; i < 8; ++i) {
    file[i] = static_cast<std::byte>(length & 0xFF);
    length >>= 8;
  }
  std::memcpy(file.data() + 8, header.data(), header.size());
  if (!data.empty()) std::memcpy(file.data() + 8 + header.size(), data.data(), data.size());
  return file;
}

// Adds named tensors one after another and writes a valid file, padding the header with spaces so
// the data starts on an 8 byte boundary the way the Python library does.
class SafetensorsWriter {
 public:
  void AddF32(const std::string& name, const std::vector<std::size_t>& shape,
              const std::vector<float>& values) {
    Add(name, "F32", shape, values.data(), values.size() * sizeof(float));
  }

  // Raw f16 bit patterns.
  void AddF16(const std::string& name, const std::vector<std::size_t>& shape,
              const std::vector<std::uint16_t>& bits) {
    Add(name, "F16", shape, bits.data(), bits.size() * sizeof(std::uint16_t));
  }

  void AddMetadata(const std::string& key, const std::string& value) {
    metadata_ += (metadata_.empty() ? "" : ",") + Quote(key) + ":" + Quote(value);
  }

  std::vector<std::byte> Build() const {
    std::string header = "{";
    if (!metadata_.empty()) header += "\"__metadata__\":{" + metadata_ + "}";
    if (!metadata_.empty() && !entries_.empty()) header += ",";
    header += entries_ + "}";
    while ((8 + header.size()) % 8 != 0) header += ' ';
    return RawSafetensors(header, data_);
  }

 private:
  static std::string Quote(const std::string& text) { return "\"" + text + "\""; }

  void Add(const std::string& name, const std::string& dtype, const std::vector<std::size_t>& shape,
           const void* bytes, std::size_t size) {
    std::string dims;
    for (const std::size_t dim : shape) dims += (dims.empty() ? "" : ",") + std::to_string(dim);
    const std::size_t begin = data_.size();
    data_.resize(begin + size);
    if (size > 0) std::memcpy(data_.data() + begin, bytes, size);
    entries_ += (entries_.empty() ? "" : ",") + Quote(name) + ":{\"dtype\":\"" + dtype +
                "\",\"shape\":[" + dims + "],\"data_offsets\":[" + std::to_string(begin) + "," +
                std::to_string(data_.size()) + "]}";
  }

  std::string entries_;
  std::string metadata_;
  std::vector<std::byte> data_;
};

}  // namespace ember::testing
