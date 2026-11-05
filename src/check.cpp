#include "ember/check.h"

#include <cstdio>
#include <cstdlib>

namespace ember::internal {

void CheckFailed(const char* file, int line, const char* condition, const char* message) {
  std::fprintf(stderr, "%s:%d: check failed: %s: %s\n", file, line, condition, message);
  std::abort();
}

}  // namespace ember::internal
