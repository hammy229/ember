#include "ember/aligned_buffer.h"

#include <bit>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>

namespace ember {

StatusOr<AlignedBuffer> AlignedBuffer::Allocate(std::size_t bytes, std::size_t alignment) {
  if (!std::has_single_bit(alignment)) {
    return Status::InvalidArgument("alignment must be a power of two");
  }
  // No object can be bigger than this, and pointer subtraction across it would overflow.
  constexpr auto kMaxObjectBytes =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (bytes > kMaxObjectBytes) return Status::OutOfMemory("request is larger than any object");

  void* const memory = ::operator new(bytes, std::align_val_t{alignment}, std::nothrow);
  if (memory == nullptr) return Status::OutOfMemory("the system refused the allocation");
  return AlignedBuffer(static_cast<std::byte*>(memory), bytes, alignment);
}

AlignedBuffer::~AlignedBuffer() { Free(); }

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      alignment_(std::exchange(other.alignment_, 1)) {}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
  if (this != &other) {
    Free();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    alignment_ = std::exchange(other.alignment_, 1);
  }
  return *this;
}

void AlignedBuffer::Free() noexcept {
  if (data_ != nullptr) ::operator delete(data_, std::align_val_t{alignment_});
  data_ = nullptr;
  size_ = 0;
}

}  // namespace ember
