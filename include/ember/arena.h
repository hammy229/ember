#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "ember/aligned_buffer.h"
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
  ~Arena() = default;

  // The memory is not zeroed. Asking for zero bytes gives a valid, aligned pointer that must not be
  // read or written. Fails if alignment is not a power of two up to kMaxAlignment, or if the memory
  // limit or the system says no.
  StatusOr<std::byte*> Allocate(std::size_t bytes, std::size_t alignment = kDefaultAlignment);

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

}  // namespace ember
