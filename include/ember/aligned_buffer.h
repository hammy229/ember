#pragma once

#include <cstddef>

#include "ember/status_or.h"

namespace ember {

// One block of heap memory that starts on a chosen byte boundary, freed automatically.
//
// SIMD loads run fastest on data that sits on a cache line, and two threads writing to the same
// line slow each other down, so tensor memory needs a stronger promise than plain new gives.
class AlignedBuffer {
 public:
  // An empty buffer: data() is null and size() is 0.
  AlignedBuffer() = default;

  // Alignment must be a power of two. Fails instead of throwing when memory runs out. The memory is
  // not zeroed. A size of zero still gives a unique, aligned, non-null pointer.
  static StatusOr<AlignedBuffer> Allocate(std::size_t bytes, std::size_t alignment);

  ~AlignedBuffer();
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  AlignedBuffer(AlignedBuffer&& other) noexcept;
  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

  std::byte* data() { return data_; }
  const std::byte* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  AlignedBuffer(std::byte* data, std::size_t size, std::size_t alignment)
      : data_(data), size_(size), alignment_(alignment) {}

  void Free() noexcept;

  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t alignment_ = 1;
};

}  // namespace ember
