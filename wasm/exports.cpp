#include <emscripten/emscripten.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <span>

#include "ember/arena.h"
#include "ember/ops.h"
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

  // Sizes near the top of a 32-bit size_t, where padding could wrap around. Nothing here limits the
  // arena, so these have to be stopped by the overflow checks themselves.
  Arena unlimited;
  if (unlimited.Allocate(0xFFFFFFFFU).ok()) return 14;
  if (unlimited.Allocate(0x80000000U).ok()) return 15;
  if (unlimited.used_bytes() != 0) return 16;

  // A large count under the limit of about 2^29 still works, and one over it does not.
  if (!Shape::Make({std::size_t{1} << 14, std::size_t{1} << 14}).ok()) return 17;
  if (Shape::Make({std::size_t{1} << 15, std::size_t{1} << 15}).ok()) return 18;
  return 0;
}

// The math ops on 32-bit WebAssembly, where the compiler and libm differ from the native build.
// Returns 0 if everything holds, or the number of the first check that failed.
int RunOpsSelfCheck() {
  using ember::Arena;
  using ember::Shape;
  using ember::Tensor;

  Arena arena;
  const auto make = [&](std::initializer_list<std::size_t> dims, std::span<const float> values) {
    return Tensor::FromValues<float>(arena, *Shape::Make(dims), values);
  };

  const std::array<float, 6> a_values{1, 2, 3, 4, 5, 6};
  const std::array<float, 6> b_values{7, 8, 9, 10, 11, 12};
  const std::array<float, 4> zeros{0, 0, 0, 0};
  const auto a = make({2, 3}, a_values);
  const auto b = make({3, 2}, b_values);
  const auto out = make({2, 2}, zeros);
  if (!a.ok() || !b.ok() || !out.ok()) return 1;

  if (!ember::MatMul(*a, *b, *out).ok()) return 2;
  if (out->at<float>(0, 0) != 58.0F || out->at<float>(1, 1) != 154.0F) return 3;

  // Through a transpose, the case attention uses.
  const auto flipped = a->Transpose(0, 1);
  if (!flipped.ok() || !ember::MatMul(*a, *flipped, *out).ok()) return 4;
  if (out->at<float>(0, 1) != 32.0F || out->at<float>(1, 1) != 77.0F) return 5;

  // An output on top of an input has to be refused here too.
  if (ember::MatMul(*out, *out, *out).ok()) return 6;

  // These logits overflow a float if softmax ever computes e^x directly.
  const std::array<float, 3> logits_values{1001, 1002, 1003};
  const auto logits = make({3}, logits_values);
  if (!logits.ok() || !ember::Softmax(*logits, 0, *logits).ok()) return 7;
  const float total = logits->at<float>(0) + logits->at<float>(1) + logits->at<float>(2);
  if (total < 0.9999F || total > 1.0001F) return 8;
  if (logits->at<float>(2) < 0.665F || logits->at<float>(2) > 0.666F) return 9;

  const std::array<float, 2> bad_values{1, std::numeric_limits<float>::quiet_NaN()};
  const auto bad = make({2}, bad_values);
  if (!bad.ok() || ember::Softmax(*bad, 0, *bad).ok()) return 10;
  return 0;
}

const float* AttentionFrame(int tokens, int features, float phase) {
  using ember::Arena;
  using ember::DType;
  using ember::Shape;
  using ember::Tensor;

  constexpr int kMaxSide = 128;
  if (tokens < 1 || tokens > kMaxSide || features < 1 || features > kMaxSide) return nullptr;
  const auto token_count = static_cast<std::size_t>(tokens);
  const auto feature_count = static_cast<std::size_t>(features);

  static Arena arena;
  static const Arena::Mark start = arena.mark();
  arena.Rewind(start);

  const auto shape = Shape::Make({token_count, feature_count});
  const auto scores_shape = Shape::Make({token_count, token_count});
  if (!shape.ok() || !scores_shape.ok()) return nullptr;
  const auto queries = Tensor::Uninitialized(arena, DType::kF32, *shape);
  const auto keys = Tensor::Uninitialized(arena, DType::kF32, *shape);
  const auto scores = Tensor::Uninitialized(arena, DType::kF32, *scores_shape);
  if (!queries.ok() || !keys.ok() || !scores.ok()) return nullptr;

  for (std::size_t t = 0; t < token_count; ++t) {
    for (std::size_t f = 0; f < feature_count; ++f) {
      const auto tf = static_cast<float>(t);
      const auto ff = static_cast<float>(f);
      queries->at<float>(t, f) =
          std::sin(0.31F * (tf + 1.0F) * (ff + 1.0F) / 3.0F + phase * (0.5F + 0.07F * ff));
      keys->at<float>(t, f) =
          std::cos(0.23F * (tf + 1.0F) * (ff + 2.0F) / 3.0F - phase * (0.4F + 0.05F * ff));
    }
  }

  const auto flipped = keys->Transpose(0, 1);
  if (!flipped.ok()) return nullptr;
  const auto copied = Tensor::CopyOf(arena, *flipped);
  if (!copied.ok()) return nullptr;
  if (!ember::MatMul(*queries, *copied, *scores).ok()) return nullptr;

  // Sharper than the 1 / sqrt(features) in real attention, so the grid shows more contrast.
  const float sharpen = 4.0F / std::sqrt(static_cast<float>(features));
  if (!ember::Scale(*scores, sharpen, *scores).ok()) return nullptr;
  const float hidden = -std::numeric_limits<float>::infinity();
  for (std::size_t t = 0; t < token_count; ++t) {
    for (std::size_t later = t + 1; later < token_count; ++later) {
      scores->at<float>(t, later) = hidden;
    }
  }
  if (!ember::Softmax(*scores, 1, *scores).ok()) return nullptr;
  return scores->data<float>();
}

int SoftmaxRowForPage(const float* logits, float* out, int count) {
  using ember::DType;
  using ember::Shape;
  using ember::Tensor;
  using ember::TensorView;

  if (count < 1) return 1;
  const auto shape = Shape::Make({static_cast<std::size_t>(count)});
  if (!shape.ok()) return 1;
  const auto input = TensorView::Wrap(static_cast<const void*>(logits), DType::kF32, *shape);
  const auto output = Tensor::Wrap(out, DType::kF32, *shape);
  if (!input.ok() || !output.ok()) return 1;
  return ember::Softmax(*input, 0, *output).ok() ? 0 : 1;
}

float MatMulBench(int size, int calls) {
  using ember::Arena;
  using ember::DType;
  using ember::Shape;
  using ember::Tensor;

  const float failed = std::numeric_limits<float>::quiet_NaN();
  constexpr int kMaxSize = 2048;
  if (size < 1 || size > kMaxSize || calls < 1) return failed;

  static Arena arena;
  static const Arena::Mark start = arena.mark();
  static int prepared_size = 0;
  static Tensor a;
  static Tensor b;
  static Tensor out;
  if (size != prepared_size) {
    prepared_size = 0;
    arena.Rewind(start);
    const auto side = static_cast<std::size_t>(size);
    const auto shape = Shape::Make({side, side});
    if (!shape.ok()) return failed;
    const auto made_a = Tensor::Uninitialized(arena, DType::kF32, *shape);
    const auto made_b = Tensor::Uninitialized(arena, DType::kF32, *shape);
    const auto made_out = Tensor::Uninitialized(arena, DType::kF32, *shape);
    if (!made_a.ok() || !made_b.ok() || !made_out.ok()) return failed;
    a = *made_a;
    b = *made_b;
    out = *made_out;
    std::mt19937 random(12345);
    std::uniform_real_distribution<float> value(-1.0F, 1.0F);
    for (std::size_t i = 0; i < a.element_count(); ++i) {
      a.data<float>()[i] = value(random);
      b.data<float>()[i] = value(random);
    }
    prepared_size = size;
  }
  for (int call = 0; call < calls; ++call) {
    if (!ember::MatMul(a, b, out).ok()) return failed;
  }
  return out.data<float>()[0];
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE const char* ember_version() { return ember::version(); }

EMSCRIPTEN_KEEPALIVE int ember_memory_self_check() { return RunMemorySelfCheck(); }

EMSCRIPTEN_KEEPALIVE int ember_ops_self_check() { return RunOpsSelfCheck(); }

// One frame of the website's coal grid. It makes up queries and keys from sines, so no model is
// involved, then runs them through what attention will do: copy the transposed keys, MatMul, scale,
// hide the future with negative infinity, and Softmax each row. Returns tokens x tokens floats, row
// by row, valid until the next call, or null on bad sizes or an engine error.
EMSCRIPTEN_KEEPALIVE const float* ember_attention_frame(int tokens, int features, float phase) {
  return AttentionFrame(tokens, features, phase);
}

// Softmax of one row from the page. Returns 0, or 1 if the engine refused the row.
EMSCRIPTEN_KEEPALIVE int ember_softmax_row(const float* logits, float* out, int count) {
  return SoftmaxRowForPage(logits, out, count);
}

// Multiplies two random size x size matrices `calls` times, for the page to time. Returns one
// element of the answer, so the work cannot be thrown away, or NaN on bad sizes or an engine error.
EMSCRIPTEN_KEEPALIVE float ember_matmul_bench(int size, int calls) {
  return MatMulBench(size, calls);
}

}  // extern "C"
