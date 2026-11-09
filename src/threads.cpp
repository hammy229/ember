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

// One call's work: the body, how many pieces, and the counters every thread shares. It lives on the
// caller's stack, and the caller waits until no worker holds it before returning.
struct Job {
  const std::function<void(std::size_t, std::size_t)>* body;
  std::size_t count;
  std::size_t pieces;
  std::atomic<std::size_t> next{0};
  std::size_t attached = 0;  // workers holding this job, guarded by the pool's mutex
};

// Workers sleep on a condition variable between jobs. Each takes the next piece from the job's
// shared counter, so fast cores take more pieces than slow ones (the M4 mixes performance and
// efficiency cores).
class Pool {
 public:
  ~Pool() { Resize(1); }

  void Resize(std::size_t threads) {
    threads = std::max<std::size_t>(threads, 1);
    if (!kHaveThreads) threads = 1;
    if (threads == workers_.size() + 1) return;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      ++generation_;
    }
    wake_.notify_all();
    for (std::thread& worker : workers_) worker.join();
    workers_.clear();
    stopping_ = false;
    for (std::size_t i = 1; i < threads; ++i) workers_.emplace_back([this] { Work(); });
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
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      job_ = &job;
      ++generation_;
    }
    wake_.notify_all();
    TakePieces(job);
    std::unique_lock<std::mutex> lock(mutex_);
    // No new worker can attach once job_ is cleared, and the ones attached are finishing pieces.
    job_ = nullptr;
    done_.wait(lock, [&] { return job.attached == 0; });
  }

 private:
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

  void Work() {
    std::size_t seen = 0;
    while (true) {
      Job* job = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return generation_ != seen; });
        seen = generation_;
        if (stopping_) return;
        job = job_;
        if (job == nullptr) continue;
        ++job->attached;
      }
      TakePieces(*job);
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        --job->attached;
      }
      done_.notify_all();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  std::size_t generation_ = 0;
  bool stopping_ = false;
  Job* job_ = nullptr;
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
