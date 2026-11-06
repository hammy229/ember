#!/usr/bin/env bash
# Builds the release preset, runs the kernel benchmark at several sizes, and prints the machine and
# commit next to the numbers, so a result can be traced back to what produced it.
#
#   ./scripts/bench.sh [repeats]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
repeats="${1:-10}"

cmake --preset release >/dev/null
cmake --build --preset release --target ember_bench_ops >/dev/null

compiler="$(grep -m1 '^CMAKE_CXX_COMPILER:' build/release/CMakeCache.txt | cut -d= -f2)"
flags="$(grep -m1 '^CMAKE_CXX_FLAGS_RELEASE:' build/release/CMakeCache.txt | cut -d= -f2)"
local_changes=""
git diff --quiet HEAD || local_changes=" (with local changes)"

echo "commit:   $(git rev-parse --short HEAD)$local_changes"
echo "compiler: $("$compiler" --version | head -1)"
echo "flags:    $flags"
if [[ "$(uname)" == "Darwin" ]]; then
  echo "cpu:      $(sysctl -n machdep.cpu.brand_string)"
  echo "memory:   $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)) GB"
else
  echo "cpu:      $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
  echo "memory:   $(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024)) GB"
fi
echo "threads:  1"
echo

for size in 64 256 1024; do
  ./build/release/ember_bench_ops "$repeats" "$size"
  echo
done
