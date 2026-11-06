// Times the math kernels on one thread. It prints only what it measures on this machine. Use
// scripts/bench.sh, which builds the release preset and records the machine next to the numbers.
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>

#include "bench_args.h"
#include "ember/arena.h"
#include "ember/ops.h"
#include "ember/shape.h"
#include "ember/status.h"
#include "ember/tensor.h"
#include "ember/timer.h"

namespace {

using ember::Arena;
using ember::DType;
using ember::Shape;
using ember::Status;
using ember::StatusOr;
using ember::Tensor;

constexpr std::size_t kMaxRepeats = 100000;
// Each run does size^3 work, so anything bigger is a typo, not a benchmark.
constexpr std::size_t kMaxSize = 4096;

volatile float g_sink = 0.0F;

void Require(const Status& status) {
  if (status.ok()) return;
  std::cerr << status.ToString() << "\n";
  std::abort();
}

template <typename T>
T Require(StatusOr<T> value) {
  Require(value.status());
  return *std::move(value);
}

Tensor RandomMatrix(Arena& arena, std::size_t size, std::mt19937& random) {
  const Tensor matrix =
      Require(Tensor::Uninitialized(arena, DType::kF32, *Shape::Make({size, size})));
  std::uniform_real_distribution<float> value(-1.0F, 1.0F);
  float* const first = matrix.data<float>();
  for (std::size_t i = 0; i < matrix.element_count(); ++i) first[i] = value(random);
  return matrix;
}

// The product as most people first write it. It is here only to compare against. The inner loop is
// a running total, which the compiler may not reorder into SIMD without permission to change the
// rounding, and it jumps a whole row of b on every step. So the gap to MatMul is both of those
// effects together, and this benchmark cannot separate them.
void TextbookMatMul(const float* a, const float* b, float* out, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    for (std::size_t j = 0; j < size; ++j) {
      float total = 0.0F;
      for (std::size_t k = 0; k < size; ++k) total += a[i * size + k] * b[k * size + j];
      out[i * size + j] = total;
    }
  }
}

// Times fn after changing the input, so the compiler cannot reuse an earlier answer, and reads the
// output, so it cannot throw the work away. The fastest run is the steadiest number on a busy
// machine, because other programs can only slow a run down.
template <typename Fn>
void Measure(const std::string& name, const Tensor& input, const Tensor& output, double operations,
             const char* unit, std::size_t repeats, Fn&& fn) {
  std::size_t run = 0;
  const ember::TimingSummary timing = ember::Repeat(
      [&] {
        input.data<float>()[0] = static_cast<float>(run++ % 2);
        fn();
        g_sink = output.data<float>()[output.element_count() - 1];
      },
      repeats, 1);
  std::cout << std::left << std::setw(40) << name << std::right << std::fixed
            << std::setprecision(3) << " min " << std::setw(10) << timing.min_ms << " ms  median "
            << std::setw(10) << timing.median_ms << " ms  " << std::setprecision(2) << std::setw(8)
            << operations / (timing.min_ms * 1e6) << " " << unit << "\n";
}

int Usage() {
  std::cerr << "usage: ember_bench_ops [repeats, 1 to " << kMaxRepeats << "] [matrix size, 1 to "
            << kMaxSize << "]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t repeats = 20;
  std::size_t size = 512;
  if (argc > 3) return Usage();
  if (argc > 1) {
    const auto parsed = ember_tools::ParseCount(argv[1], kMaxRepeats);
    if (!parsed) return Usage();
    repeats = *parsed;
  }
  if (argc > 2) {
    const auto parsed = ember_tools::ParseCount(argv[2], kMaxSize);
    if (!parsed) return Usage();
    size = *parsed;
  }

  Arena arena;
  std::mt19937 random(12345);
  const Tensor a = RandomMatrix(arena, size, random);
  const Tensor b = RandomMatrix(arena, size, random);
  const Tensor out = RandomMatrix(arena, size, random);
  const Tensor b_transposed = Require(b.Transpose(0, 1));

  const double count = static_cast<double>(size);
  const double elements = count * count;
  const double product_operations = 2.0 * count * count * count;  // One multiply and one add each.
  const std::string dims = std::to_string(size) + "x" + std::to_string(size);

  std::cout << "repeats: " << repeats << "\n";
  Measure("matmul " + dims, a, out, product_operations, "billion math ops/s", repeats,
          [&] { Require(ember::MatMul(a, b, out)); });
  Measure("matmul " + dims + ", transposed", a, out, product_operations, "billion math ops/s",
          repeats, [&] { Require(ember::MatMul(a, b_transposed, out)); });
  // The same product again, paying for one copy first so the inner loop reads memory in a line.
  const ember::Arena::Mark before_copy = arena.mark();
  Measure("matmul " + dims + ", transposed, copied", a, out, product_operations,
          "billion math ops/s", repeats, [&] {
            const Tensor copied = Require(Tensor::CopyOf(arena, b_transposed));
            Require(ember::MatMul(a, copied, out));
            arena.Rewind(before_copy);
          });
  Measure("textbook loops " + dims, a, out, product_operations, "billion math ops/s", repeats,
          [&] { TextbookMatMul(a.data<float>(), b.data<float>(), out.data<float>(), size); });
  Measure("add " + dims, a, out, elements, "billion elements/s", repeats,
          [&] { Require(ember::Add(a, b, out)); });
  Measure("softmax " + dims + ", each row", a, out, elements, "billion elements/s", repeats,
          [&] { Require(ember::Softmax(a, 1, out)); });
  Measure("softmax " + dims + ", each column", a, out, elements, "billion elements/s", repeats,
          [&] { Require(ember::Softmax(a, 0, out)); });
  return 0;
}
