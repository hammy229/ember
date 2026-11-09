#pragma once

#include <cstddef>
#include <functional>

namespace ember {

// How many threads the engine's heavy loops use, including the caller's. Starts at 1. Setting it
// starts or stops worker threads, so do it between calls, not during one. WebAssembly builds have
// no threads and always run on one.
void SetThreadCount(std::size_t count);
std::size_t ThreadCount();

// Calls body(begin, end) on pieces of [0, count) that cover every index exactly once, spread across
// the engine's threads, and returns when all are done. Pieces are at least `grain` long, so tiny
// loops are not split. A call from inside a body runs on the calling thread alone.
//
// Which thread runs which piece changes from call to call, so a body must not depend on it: each
// index's work has to be the same wherever it runs. Then the results are identical for any number
// of threads, which tests/threads_test.cpp checks for the engine's own uses.
void ParallelFor(std::size_t count, std::size_t grain,
                 const std::function<void(std::size_t begin, std::size_t end)>& body);

}  // namespace ember
