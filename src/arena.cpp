#include "ember/arena.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace ember {

namespace {

constexpr std::size_t kMaxSize = std::numeric_limits<std::size_t>::max();

}  // namespace

Arena::Arena(Options options) : options_(options) {
  EMBER_CHECK(options_.chunk_bytes > 0, "an arena chunk needs at least one byte");
}

Arena::~Arena() {
  for (AlignedBuffer& chunk : chunks_) EMBER_UNPOISON(chunk.data(), chunk.size());
}

std::byte* Arena::TryBump(std::size_t bytes, std::size_t alignment) {
  if (chunks_.empty()) return nullptr;
  AlignedBuffer& chunk = chunks_[chunk_index_];

  // Align the address, not the offset, because a large alignment can exceed the chunk's own.
  const auto base = reinterpret_cast<std::uintptr_t>(chunk.data());
  const std::uintptr_t cursor = base + offset_ + (offset_ == 0 ? 0 : kRedzoneBytes);
  const auto mask = static_cast<std::uintptr_t>(alignment - 1);
  const std::uintptr_t aligned = (cursor + mask) & ~mask;
  const std::size_t start = static_cast<std::size_t>(aligned - base);

  // Written so neither side can overflow. A start past the end means the padding alone did not fit.
  if (start > chunk.size() || bytes > chunk.size() - start) return nullptr;

  used_bytes_ += (start - offset_) + bytes;
  peak_bytes_ = std::max(peak_bytes_, used_bytes_);
  offset_ = start + bytes;
  EMBER_UNPOISON(chunk.data() + start, bytes);
  return chunk.data() + start;
}

Status Arena::Grow(std::size_t bytes, std::size_t alignment) {
  if (bytes > kMaxSize - alignment) return Status::OutOfMemory("request is too large to align");
  const std::size_t needed = bytes + alignment;

  const std::size_t room = options_.max_bytes - capacity_bytes_;
  if (needed > room) return Status::OutOfMemory("the arena memory limit was reached");
  // Prefer a full chunk, but near the limit settle for what the request needs.
  const std::size_t chunk_bytes = std::max(needed, std::min(options_.chunk_bytes, room));

  EMBER_ASSIGN_OR_RETURN(AlignedBuffer chunk,
                         AlignedBuffer::Allocate(chunk_bytes, kDefaultAlignment));
  // Nothing in a chunk may be touched until an allocation hands it out.
  EMBER_POISON(chunk.data(), chunk.size());
  chunk_index_ = chunks_.size();
  offset_ = 0;
  chunks_.push_back(std::move(chunk));
  capacity_bytes_ += chunk_bytes;
  return Status::Ok();
}

StatusOr<std::byte*> Arena::Allocate(std::size_t bytes, std::size_t alignment) {
  if (!std::has_single_bit(alignment) || alignment > kMaxAlignment) {
    return Status::InvalidArgument("arena alignment must be a power of two up to 4096");
  }
  if (std::byte* const memory = TryBump(bytes, alignment)) return memory;

  // After a rewind, later chunks are free again. Use them before asking the system for more.
  const std::size_t saved_index = chunk_index_;
  const std::size_t saved_offset = offset_;
  while (chunk_index_ + 1 < chunks_.size()) {
    ++chunk_index_;
    offset_ = 0;
    if (std::byte* const memory = TryBump(bytes, alignment)) return memory;
  }

  if (Status grown = Grow(bytes, alignment); !grown.ok()) {
    // Walking the chunks must not cost anything if the request fails in the end.
    chunk_index_ = saved_index;
    offset_ = saved_offset;
    return grown;
  }
  std::byte* const memory = TryBump(bytes, alignment);
  EMBER_CHECK(memory != nullptr, "a new arena chunk must fit the request it was sized for");
  return memory;
}

void Arena::Rewind(const Mark& mark) {
  const bool is_behind_or_here = mark.chunk_index < chunk_index_ ||
                                 (mark.chunk_index == chunk_index_ && mark.offset <= offset_);
  EMBER_CHECK(is_behind_or_here, "rewound an arena to a mark that is ahead of it");
  // The freed memory goes back to being poisoned, so a tensor that outlives its scope crashes on
  // its next read instead of quietly seeing whatever is allocated there next.
  if constexpr (EMBER_ASAN) {
    for (std::size_t i = mark.chunk_index; i <= chunk_index_ && i < chunks_.size(); ++i) {
      const std::size_t from = i == mark.chunk_index ? mark.offset : 0;
      EMBER_POISON(chunks_[i].data() + from, chunks_[i].size() - from);
    }
  }
  chunk_index_ = mark.chunk_index;
  offset_ = mark.offset;
  used_bytes_ = mark.used_bytes;
}

void Arena::Reset() { Rewind(Mark{}); }

}  // namespace ember
