// Tiny benchmark harness. It times a placeholder workload (summing a float
// array) to exercise the repeat-and-report loop. Real kernels get added here
// as they exist. It prints only what it measures on this machine.
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ember/timer.h"

namespace {

volatile float g_sink = 0.0F;

float SumAll(const std::vector<float>& v) {
  float total = 0.0F;
  for (float x : v) total += x;
  return total;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t repeats = 20;
  std::size_t size = 1U << 22;
  if (argc > 1) repeats = std::strtoul(argv[1], nullptr, 10);
  if (argc > 2) size = std::strtoul(argv[2], nullptr, 10);
  if (repeats == 0 || size == 0) {
    std::cerr << "usage: ember_bench [repeats > 0] [elements > 0]\n";
    return 2;
  }

  std::vector<float> data(size, 1.0F);
  const ember::TimingSummary s = ember::Repeat([&] { g_sink = SumAll(data); }, repeats, 2);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "bench: sum of " << size << " floats, " << s.count << " runs\n";
  std::cout << "  min    " << s.min_ms << " ms\n";
  std::cout << "  median " << s.median_ms << " ms\n";
  std::cout << "  mean   " << s.mean_ms << " ms\n";
  std::cout << "  max    " << s.max_ms << " ms\n";
  return 0;
}
