// Times a real model end to end: how long loading takes, how long one forward pass takes at several
// sequence lengths, and how much memory it all needs. scripts/results.sh runs it and saves the
// output with the machine and commit, so every number on the site and in the README can be traced
// back to a run anyone can repeat.
//
// usage: ember_bench_model MODEL_DIR [repeats]

#include <sys/resource.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "bench_args.h"
#include "ember/arena.h"
#include "ember/forward.h"
#include "ember/internal/asan.h"
#include "ember/llama.h"
#include "ember/timer.h"

namespace {

int Usage() {
  std::cerr << "usage: ember_bench_model MODEL_DIR [repeats from 1 to 1000]\n";
  return 2;
}

// The most memory this process has held at once, in bytes.
double PeakResidentBytes() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
  return static_cast<double>(usage.ru_maxrss);  // bytes on macOS
#else
  return static_cast<double>(usage.ru_maxrss) * 1024.0;  // kilobytes on Linux
#endif
}

double Megabytes(double bytes) { return bytes / (1024.0 * 1024.0); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) return Usage();
  std::size_t repeats = 5;
  if (argc == 3) {
    const auto parsed = ember_tools::ParseCount(argv[2], 1000);
    if (!parsed) return Usage();
    repeats = *parsed;
  }
#if EMBER_ASAN || !defined(NDEBUG)
  std::cerr << "refusing to time a debug or sanitized build; use the release preset\n";
  return 1;
#endif
  const std::string directory = argv[1];

  // Loading twice keeps the first, cold read from the disk out of the number. The OS caches the
  // file after that, which is also what a second run of any program sees.
  std::vector<double> load_samples;
  std::unique_ptr<ember::LlamaModel> model;
  for (std::size_t i = 0; i < repeats + 1; ++i) {
    // Freed first, so two copies of the weights never count toward the peak memory below.
    model.reset();
    const ember::Stopwatch stopwatch;
    auto loaded = ember::LlamaModel::Load(directory);
    const double ms = stopwatch.ElapsedMs();
    if (!loaded.ok()) {
      std::cerr << "error: " << loaded.status().ToString() << '\n';
      return 1;
    }
    if (i > 0) load_samples.push_back(ms);
    model = std::move(loaded).value();
  }
  const ember::TimingSummary load = ember::Summarize(load_samples);
  const ember::LlamaConfig& c = model->config();

  std::printf("model:      %zu parameters, %zu layers, hidden %zu, vocab %zu\n",
              model->parameter_count(), c.layer_count, c.hidden_size, c.vocab_size);
  std::printf("weights:    %.1f MB in f32 after widening\n",
              Megabytes(static_cast<double>(model->parameter_count()) * 4.0));
  std::printf("repeats:    %zu, median shown, single thread\n\n", repeats);
  std::printf("load        median %9.2f ms   min %9.2f ms\n\n", load.median_ms, load.min_ms);
  std::printf("%-8s %14s %14s %16s %14s\n", "tokens", "forward ms", "min ms", "tokens/s",
              "scratch MB");

  ember::Arena arena;
  for (const std::size_t length :
       {std::size_t{1}, std::size_t{8}, std::size_t{32}, std::size_t{128}, c.max_positions}) {
    if (length > c.max_positions) continue;
    std::vector<std::int32_t> tokens(length);
    // Spread across the vocabulary. The values do not change the work done, only the answer.
    for (std::size_t i = 0; i < length; ++i) {
      tokens[i] = static_cast<std::int32_t>((i * 7919 + 1) % c.vocab_size);
    }
    bool failed = false;
    std::size_t scratch = 0;
    const ember::TimingSummary forward = ember::Repeat(
        [&] {
          arena.Reset();
          const auto logits = ember::Forward(*model, tokens, arena);
          if (!logits.ok()) failed = true;
          scratch = std::max(scratch, arena.peak_bytes());
        },
        repeats);
    if (failed) {
      std::cerr << "error: forward pass failed at " << length << " tokens\n";
      return 1;
    }
    std::printf("%-8zu %14.2f %14.2f %16.1f %14.1f\n", length, forward.median_ms, forward.min_ms,
                static_cast<double>(length) / (forward.median_ms / 1000.0),
                Megabytes(static_cast<double>(scratch)));
    if (length == c.max_positions) break;
  }
  std::printf("\npeak memory of the whole process: %.1f MB\n", Megabytes(PeakResidentBytes()));
  return 0;
}
