#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "ember/aligned_buffer.h"
#include "ember/internal/asan.h"
#include "ember/status_or.h"

namespace ember {

// Hands out memory by moving a pointer forward, and frees it all at once.
//
// One forward pass of a model makes many short-lived tensors that all die together. Calling the
// system allocator for each one costs time and scatters them around memory. An arena turns each
// allocation into a little arithmetic, keeps the tensors close together, and needs no individual
// frees.
//
// Memory lives in chunks that are never moved, so a pointer stays valid until the arena is reset or
// destroyed, even as the arena grows. Not thread safe: give each thread its own arena.
class Arena {
 public:
  // Cache line size on the targets we care about, and enough for any SIMD register up to AVX-512.
  static constexpr std::size_t kDefaultAlignment = 64;
  // A larger alignment than a page is a bug in the caller, not a real need.
  static constexpr std::size_t kMaxAlignment = 4096;
  static constexpr std::size_t kDefaultChunkBytes = std::size_t{1} << 20;
  // A gap between allocations in sanitizer builds, so writing past the end of a tensor lands on
  // poisoned memory and not silently on the next tensor. Normal builds waste nothing.
  static constexpr std::size_t kRedzoneBytes = EMBER_ASAN ? 32 : 0;

  struct Options {
    // How much to ask the system for at a time. Bigger requests get a chunk of their own size.
    std::size_t chunk_bytes = kDefaultChunkBytes;
    // Allocations fail with kOutOfMemory rather than let chunks add up past this. Set it for the
    // browser, where the tab should get an error and not be killed.
    std::size_t max_bytes = std::numeric_limits<std::size_t>::max();
  };

  Arena() : Arena(Options{}) {}
  explicit Arena(Options options);

  // Tensors point into the chunks, and a moved arena would leave its position behind. Nothing needs
  // either, so neither is allowed.
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&&) = delete;
  Arena& operator=(Arena&&) = delete;
  ~Arena();

  // The memory is not zeroed. Asking for zero bytes gives a valid, aligned pointer that must not be
  // read or written. Fails if alignment is not a power of two up to kMaxAlignment, or if the memory
  // limit or the system says no.
  StatusOr<std::byte*> Allocate(std::size_t bytes, std::size_t alignment = kDefaultAlignment);

  // A saved position in the arena.
  struct Mark {
    std::size_t chunk_index = 0;
    std::size_t offset = 0;
    std::size_t used_bytes = 0;
  };

  Mark mark() const { return Mark{chunk_index_, offset_, used_bytes_}; }

  // Frees everything allocated since the mark and keeps the chunks to reuse. Pointers handed out
  // after the mark are dead. Marks must be undone newest first: rewinding to a position ahead of
  // the current one is a bug and aborts.
  void Rewind(const Mark& mark);

  // Frees everything and keeps the chunks. peak_bytes() survives, because its job is to report
  // the most this arena ever needed.
  void Reset();

  // Bytes handed out since the last reset, counting alignment padding.
  std::size_t used_bytes() const { return used_bytes_; }
  // The most used_bytes() has ever been. This tells you how to size the arena for a workload.
  std::size_t peak_bytes() const { return peak_bytes_; }
  // Bytes held from the system across all chunks.
  std::size_t capacity_bytes() const { return capacity_bytes_; }
  std::size_t chunk_count() const { return chunks_.size(); }

 private:
  // Tries to fit the request in the current chunk. Returns null if it does not fit.
  std::byte* TryBump(std::size_t bytes, std::size_t alignment);
  // Adds a chunk that is certain to fit the request and makes it current.
  Status Grow(std::size_t bytes, std::size_t alignment);

  Options options_;
  std::vector<AlignedBuffer> chunks_;
  std::size_t chunk_index_ =
      0;                    // The chunk being bumped. Only meaningful when chunks_ is not empty.
  std::size_t offset_ = 0;  // First free byte of the current chunk.
  std::size_t used_bytes_ = 0;
  std::size_t peak_bytes_ = 0;
  std::size_t capacity_bytes_ = 0;
};

// Rewinds the arena when the scope ends, so scratch tensors in a function free themselves.
class ArenaScope {
 public:
  [[nodiscard]] explicit ArenaScope(Arena& arena) : arena_(arena), mark_(arena.mark()) {}
  ~ArenaScope() { arena_.Rewind(mark_); }

  ArenaScope(const ArenaScope&) = delete;
  ArenaScope& operator=(const ArenaScope&) = delete;
  ArenaScope(ArenaScope&&) = delete;
  ArenaScope& operator=(ArenaScope&&) = delete;

 private:
  Arena& arena_;
  Arena::Mark mark_;
};

}  // namespace ember
