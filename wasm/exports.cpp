#include <emscripten/emscripten.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "ember/arena.h"
#include "ember/shape.h"
#include "ember/tensor.h"
#include "ember/version.h"

namespace {

// Things that only WebAssembly can prove, because the native tests run with a 64-bit size_t.
// Returns 0 if everything holds, or the number of the first check that failed.
int RunMemorySelfCheck() {
  using ember::Arena;
  using ember::DType;
  using ember::Shape;
  using ember::Tensor;

  if (sizeof(std::size_t) != 4) return 1;

  // 2^16 by 2^16 wraps to zero in 32 bits, and would be a perfectly fine shape on 64 bits.
  if (Shape::Make({std::size_t{1} << 16, std::size_t{1} << 16}).ok()) return 2;

  Arena::Options options;
  options.chunk_bytes = 4096;
  options.max_bytes = std::size_t{1} << 20;
  Arena arena(options);

  // The memory limit turns a request too big for the tab into an error.
  if (arena.Allocate(std::size_t{1} << 21).ok()) return 3;

  const auto block = arena.Allocate(10);
  if (!block.ok()) return 4;
  if (reinterpret_cast<std::uintptr_t>(*block) % Arena::kDefaultAlignment != 0) return 5;

  const std::array<float, 6> values{0, 1, 2, 3, 4, 5};
  const auto shape = Shape::Make({2, 3});
  if (!shape.ok()) return 6;
  const auto source = Tensor::FromValues<float>(arena, *shape, values);
  if (!source.ok()) return 7;
  const auto swapped = source->Transpose(0, 1);
  if (!swapped.ok()) return 8;
  if (swapped->at<float>(2, 1) != 5.0F) return 9;

  const auto copy = Tensor::CopyOf(arena, *swapped);
  if (!copy.ok()) return 10;
  if (!copy->is_contiguous() || copy->data<float>()[1] != 3.0F) return 11;

  // Rewinding gives the same memory back, so repeated work does not grow the tab's memory.
  const Arena::Mark mark = arena.mark();
  const std::size_t capacity = arena.capacity_bytes();
  for (int round = 0; round < 100; ++round) {
    const auto scratch = Tensor::Zeros(arena, DType::kF32, *shape);
    if (!scratch.ok()) return 12;
    arena.Rewind(mark);
  }
  if (arena.capacity_bytes() != capacity) return 13;
  return 0;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE const char* ember_version() { return ember::version(); }

EMSCRIPTEN_KEEPALIVE int ember_memory_self_check() { return RunMemorySelfCheck(); }

}  // extern "C"
