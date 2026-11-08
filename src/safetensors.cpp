#include "ember/safetensors.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "ember/json.h"

namespace ember {

namespace {

constexpr std::size_t kLengthBytes = 8;

Status Bad(std::string_view what) {
  return Status::InvalidFormat("safetensors: " + std::string(what));
}

std::optional<DType> DTypeFromName(std::string_view name) {
  if (name == "F32") return DType::kF32;
  if (name == "F16") return DType::kF16;
  if (name == "BF16") return DType::kBF16;
  if (name == "I32") return DType::kI32;
  if (name == "I8") return DType::kI8;
  return std::nullopt;
}

// A JSON whole number that also fits in this machine's size_t, which is 32 bits in WebAssembly.
std::optional<std::size_t> AsSize(const Json& value) {
  const std::optional<std::uint64_t> exact = value.as_uint64();
  if (!exact || *exact > std::numeric_limits<std::size_t>::max()) return std::nullopt;
  return static_cast<std::size_t>(*exact);
}

struct Offsets {
  std::size_t begin;
  std::size_t end;
};

StatusOr<Safetensors::Entry> ParseEntry(const std::string& name, const Json& info,
                                        std::span<const std::byte> data, Offsets& offsets) {
  const std::string where = "tensor \"" + name + "\"";
  if (!info.is_object()) return Bad(where + " is not an object");

  const Json* dtype_json = info.Find("dtype");
  if (dtype_json == nullptr || !dtype_json->is_string()) return Bad(where + " has no dtype");
  const std::optional<DType> dtype = DTypeFromName(dtype_json->as_string());
  if (!dtype) return Bad(where + " has unsupported dtype " + dtype_json->as_string());

  const Json* shape_json = info.Find("shape");
  if (shape_json == nullptr || !shape_json->is_array()) return Bad(where + " has no shape");
  if (shape_json->size() > Shape::kMaxRank) return Bad(where + " has too many dimensions");
  std::array<std::size_t, Shape::kMaxRank> dims{};
  for (std::size_t i = 0; i < shape_json->size(); ++i) {
    const std::optional<std::size_t> dim = AsSize(shape_json->items()[i]);
    if (!dim) return Bad(where + " has a dimension that is not a size");
    dims[i] = *dim;
  }
  // Make refuses shapes whose element count would overflow, which is the check that matters most.
  StatusOr<Shape> shape =
      Shape::Make(std::span<const std::size_t>(dims.data(), shape_json->size()));
  if (!shape.ok()) return Bad(where + ": " + shape.status().message());

  const Json* offsets_json = info.Find("data_offsets");
  if (offsets_json == nullptr || !offsets_json->is_array() || offsets_json->size() != 2) {
    return Bad(where + " needs data_offsets [begin, end]");
  }
  const std::optional<std::size_t> begin = AsSize(offsets_json->items()[0]);
  const std::optional<std::size_t> end = AsSize(offsets_json->items()[1]);
  if (!begin || !end || *begin > *end || *end > data.size()) {
    return Bad(where + " has data_offsets outside the file");
  }
  // Shape::kMaxElements makes this product safe.
  const std::size_t needed = shape->element_count() * SizeOf(*dtype);
  if (*end - *begin != needed) {
    return Bad(where + " is " + std::to_string(*end - *begin) + " bytes, but " +
               std::string(Name(*dtype)) + shape->ToString() + " is " + std::to_string(needed));
  }
  if (info.size() != 3) return Bad(where + " has fields other than dtype, shape, data_offsets");

  offsets = {*begin, *end};
  return Safetensors::Entry{name, *dtype, *shape, data.subspan(*begin, needed)};
}

}  // namespace

StatusOr<Safetensors> Safetensors::Parse(std::span<const std::byte> file) {
  if (file.size() < kLengthBytes) return Bad("file is shorter than its 8 byte length");
  std::uint64_t header_length = 0;
  for (std::size_t i = kLengthBytes; i > 0; --i) {
    header_length = (header_length << 8) | static_cast<std::uint64_t>(file[i - 1]);
  }
  if (header_length > kMaxHeaderBytes) return Bad("header is larger than 16 MB");
  if (header_length > file.size() - kLengthBytes)
    return Bad("header runs past the end of the file");

  const auto header_size = static_cast<std::size_t>(header_length);
  const std::string_view header(reinterpret_cast<const char*>(file.data() + kLengthBytes),
                                header_size);
  const std::span<const std::byte> data = file.subspan(kLengthBytes + header_size);

  StatusOr<Json> json = Json::Parse(header);
  if (!json.ok()) return Bad(json.status().message());
  if (!json->is_object()) return Bad("header is not a JSON object");

  Safetensors result;
  std::vector<Offsets> offsets;
  for (std::size_t i = 0; i < json->size(); ++i) {
    const std::string& name = json->keys()[i];
    const Json& value = json->items()[i];
    if (name == "__metadata__") {
      if (!value.is_object()) return Bad("__metadata__ is not an object");
      for (std::size_t j = 0; j < value.size(); ++j) {
        if (!value.items()[j].is_string()) return Bad("__metadata__ values must be strings");
        result.metadata_.emplace_back(value.keys()[j], value.items()[j].as_string());
      }
      continue;
    }
    Offsets range{};
    EMBER_ASSIGN_OR_RETURN(Entry entry, ParseEntry(name, value, data, range));
    result.tensors_.push_back(std::move(entry));
    offsets.push_back(range);
  }

  // Sort by where each tensor starts, then each must start exactly where the last one ended.
  std::vector<std::size_t> order(offsets.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return offsets[a].begin != offsets[b].begin ? offsets[a].begin < offsets[b].begin
                                                : offsets[a].end < offsets[b].end;
  });
  std::size_t covered = 0;
  for (const std::size_t i : order) {
    if (offsets[i].begin != covered) {
      return Bad(offsets[i].begin < covered
                     ? "tensor \"" + result.tensors_[i].name + "\" overlaps another"
                     : "there is a gap before tensor \"" + result.tensors_[i].name + "\"");
    }
    covered = offsets[i].end;
  }
  if (covered != data.size()) return Bad("bytes after the last tensor belong to nothing");

  std::vector<Entry> sorted;
  sorted.reserve(order.size());
  for (const std::size_t i : order) sorted.push_back(std::move(result.tensors_[i]));
  result.tensors_ = std::move(sorted);
  return result;
}

const Safetensors::Entry* Safetensors::Find(std::string_view name) const {
  for (const Entry& entry : tensors_) {
    if (entry.name == name) return &entry;
  }
  return nullptr;
}

StatusOr<TensorView> Safetensors::View(const Entry& entry) {
  return TensorView::Wrap(entry.bytes, entry.dtype, entry.shape);
}

}  // namespace ember
