#!/usr/bin/env bash
# Measures how much of the engine the test suite runs, with clang's source-based coverage, and
# prints a line and branch summary for each file in src/. The report lands in
# build/coverage/report.txt.
#
#   ./scripts/coverage.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
build="build/coverage"

# Apple's tools come through xcrun. Elsewhere, use whatever llvm-profdata and llvm-cov are first on
# the path, which must match the clang that compiled the tests.
if command -v xcrun >/dev/null && xcrun --find llvm-cov >/dev/null 2>&1; then
  profdata=(xcrun llvm-profdata)
  cov=(xcrun llvm-cov)
  compiler="$(xcrun --find clang++)"
else
  profdata=(llvm-profdata)
  cov=(llvm-cov)
  compiler="${CXX:-clang++}"
fi

flags="-fprofile-instr-generate -fcoverage-mapping"
cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER="$compiler" \
  -DCMAKE_CXX_FLAGS="$flags" -DCMAKE_EXE_LINKER_FLAGS="$flags" >/dev/null
cmake --build "$build" --target ember_tests >/dev/null

rm -f "$build"/*.profraw
LLVM_PROFILE_FILE="$build/tests-%p.profraw" "$build/ember_tests" --gtest_brief=1 >/dev/null
"${profdata[@]}" merge -sparse "$build"/*.profraw -o "$build/tests.profdata"
"${cov[@]}" report "$build/ember_tests" -instr-profile="$build/tests.profdata" \
  -ignore-filename-regex='(_deps|tests|build)/' "$root/src" | tee "$build/report.txt"
