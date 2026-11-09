#include "ember/threads.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ember {

namespace {

// Set on every thread while it runs a piece, so a nested ParallelFor runs inline instead of waiting
// on workers that are busy with the outer one.
thread_local bool tl_inside = false;

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
constexpr bool kHaveThreads = false;
#else
constexpr bool kHaveThreads = true;
#endif

// One call's work: the body, how many pieces, and the counter every thread takes pieces from. It
// lives on the caller's stack.
struct Job {
  const std::function<void(std::size_t, std::size_t)>* body;
  std::size_t count;
  std::size_t pieces;
  std::atomic<std::size_t> next{0};
};

// Generation calls the pool about 200 times a token, so handing out work has to be cheap. Between
// jobs, workers spin for a short while watching for the next one, and only then sleep on a
// condition variable. The hand-off itself is lock-free:
//
// - The caller publishes the job, bumps `generation_`, and wakes any sleepers.
// - A worker marks itself busy before it looks at the job, and the caller, after taking the job
//   back, waits until no worker is busy. With sequentially consistent atomics a worker either sees
//   the job while the caller still waits for it, or sees no job at all, so nobody touches a Job
//   after its caller has returned.
class Pool {
 public:
  ~Pool() { Resize(1); }

  void Resize(std::size_t threads) {
    threads = std::max<std::size_t>(threads, 1);
    if (!kHaveThreads) threads = 1;
    if (threads == workers_.size() + 1) return;
    stopping_.store(true);
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      generation_.fetch_add(1);
    }
    wake_.notify_all();
    for (std::thread& worker : workers_) worker.join();
    workers_.clear();
    stopping_.store(false);
    // Each worker starts from the generation as it is now, read here rather than in the new
    // thread: a thread that starts late must not mistake a later stop signal for its starting
    // point.
    const std::size_t start = generation_.load();
    for (std::size_t i = 1; i < threads; ++i) {
      workers_.emplace_back([this, start] { Work(start); });
    }
  }

  std::size_t size() const { return workers_.size() + 1; }

  void Run(std::size_t count, std::size_t grain,
           const std::function<void(std::size_t, std::size_t)>& body) {
    const std::size_t threads = size();
    // A few pieces per thread balances uneven cores without much counter traffic.
    std::size_t pieces = std::min(threads * 4, (count + grain - 1) / grain);
    pieces = std::max<std::size_t>(pieces, 1);
    if (threads == 1 || pieces == 1 || tl_inside) {
      const bool outer = !tl_inside;
      tl_inside = true;
      body(0, count);
      if (outer) tl_inside = false;
      return;
    }
    Job job{&body, count, pieces};
    job_.store(&job);
    generation_.fetch_add(1);
    if (sleeping_.load() > 0) {
      const std::lock_guard<std::mutex> lock(mutex_);
      wake_.notify_all();
    }
    TakePieces(job);
    job_.store(nullptr);
    while (busy_.load() != 0) {
      Pause();
    }
  }

 private:
  static void Pause() {
#if defined(__aarch64__) || defined(__arm__)
    asm volatile("yield");
#elif defined(__x86_64__) || defined(__i386__)
    asm volatile("pause");
#endif
  }

  static void TakePieces(Job& job) {
    const bool outer = !tl_inside;
    tl_inside = true;
    while (true) {
      const std::size_t piece = job.next.fetch_add(1, std::memory_order_relaxed);
      if (piece >= job.pieces) break;
      (*job.body)(job.count* piece / job.pieces, job.count * (piece + 1) / job.pieces);
    }
    if (outer) tl_inside = false;
  }

  void Work(std::size_t seen) {
    while (true) {
      if (stopping_.load()) return;
      // Spin a little, since the next job usually comes within microseconds.
      bool woke = false;
      for (int spin = 0; spin < 20000; ++spin) {
        if (generation_.load() != seen) {
          woke = true;
          break;
        }
        Pause();
      }
      if (!woke) {
        std::unique_lock<std::mutex> lock(mutex_);
        sleeping_.fetch_add(1);
        wake_.wait(lock, [&] { return generation_.load() != seen || stopping_.load(); });
        sleeping_.fetch_sub(1);
      }
      seen = generation_.load();
      if (stopping_.load()) return;
      busy_.fetch_add(1);
      if (Job* job = job_.load(); job != nullptr) TakePieces(*job);
      busy_.fetch_sub(1);
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::atomic<std::size_t> generation_{0};
  std::atomic<std::size_t> sleeping_{0};
  std::atomic<std::size_t> busy_{0};
  std::atomic<bool> stopping_{false};
  std::atomic<Job*> job_{nullptr};
};

Pool& ThePool() {
  static Pool pool;
  return pool;
}

}  // namespace

void SetThreadCount(std::size_t count) { ThePool().Resize(count); }

std::size_t ThreadCount() { return ThePool().size(); }

void ParallelFor(std::size_t count, std::size_t grain,
                 const std::function<void(std::size_t begin, std::size_t end)>& body) {
  if (count == 0) return;
  ThePool().Run(count, std::max<std::size_t>(grain, 1), body);
}

}  // namespace ember
