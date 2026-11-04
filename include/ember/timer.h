#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

namespace ember {

// Wall-clock timer on the steady clock, in milliseconds.
class Stopwatch {
 public:
  Stopwatch() : start_(Clock::now()) {}

  void Reset() { start_ = Clock::now(); }

  double ElapsedMs() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

struct TimingSummary {
  std::size_t count = 0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double mean_ms = 0.0;
  double median_ms = 0.0;
};

// Summarizes timings given in milliseconds. An empty input gives all zeros.
inline TimingSummary Summarize(std::vector<double> samples_ms) {
  TimingSummary out;
  if (samples_ms.empty()) return out;
  std::sort(samples_ms.begin(), samples_ms.end());
  const std::size_t n = samples_ms.size();
  out.count = n;
  out.min_ms = samples_ms.front();
  out.max_ms = samples_ms.back();
  out.mean_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / static_cast<double>(n);
  out.median_ms = (n % 2 == 1) ? samples_ms[n / 2] : (samples_ms[n / 2 - 1] + samples_ms[n / 2]) / 2.0;
  return out;
}

// Calls fn `warmup` times untimed, then `repeats` times timed, and summarizes the timed runs.
template <typename Fn>
TimingSummary Repeat(Fn&& fn, std::size_t repeats, std::size_t warmup = 1) {
  for (std::size_t i = 0; i < warmup; ++i) fn();
  std::vector<double> samples;
  samples.reserve(repeats);
  for (std::size_t i = 0; i < repeats; ++i) {
    Stopwatch sw;
    fn();
    samples.push_back(sw.ElapsedMs());
  }
  return Summarize(std::move(samples));
}

}  // namespace ember
