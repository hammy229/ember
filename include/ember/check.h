#pragma once

// Checks for bugs in the program, as opposed to bad input.
//
// Bad input (a corrupt model file, a shape that does not fit in memory) comes back as a Status,
// because the caller can do something about it. A broken promise between two pieces of our own code
// (reading a float tensor as ints) has no sensible recovery, so it prints where it happened and
// aborts.

namespace ember::internal {

[[noreturn]] void CheckFailed(const char* file, int line, const char* condition,
                              const char* message);

}  // namespace ember::internal

// Always on. Use it where the cost does not matter, like at the start of an operation.
#define EMBER_CHECK(condition, message)                                          \
  do {                                                                           \
    if (!(condition)) [[unlikely]] {                                             \
      ::ember::internal::CheckFailed(__FILE__, __LINE__, #condition, (message)); \
    }                                                                            \
  } while (false)

// Off in release builds. Use it in hot paths, like reading one element of a tensor. The sizeof
// keeps the condition compiled, so it cannot rot, but never runs it.
#ifdef NDEBUG
#define EMBER_DCHECK(condition, message) \
  do {                                   \
    (void)sizeof(!(condition));          \
    (void)sizeof(message);               \
  } while (false)
#else
#define EMBER_DCHECK(condition, message) EMBER_CHECK(condition, message)
#endif
