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

cached() { grep -m1 "^$1:" build/release/CMakeCache.txt | cut -d= -f2-; }

# An old build directory can keep settings the preset does not name, and the numbers would look
# normal while coming from the wrong build.
if [[ "$(cached CMAKE_BUILD_TYPE)" != "Release" || "$(cached EMBER_SANITIZE)" == "ON" ]]; then
  echo "build/release is not a plain release build, delete it and run this again" >&2
  exit 1
fi

local_changes=""
[[ -z "$(git status --porcelain)" ]] || local_changes=" (with local changes)"

echo "commit:   $(git rev-parse --short HEAD)$local_changes"
echo "compiler: $("$(cached CMAKE_CXX_COMPILER)" --version | head -1)"
echo "flags:    $(cached CMAKE_CXX_FLAGS_RELEASE)"
if [[ "$(uname)" == "Darwin" ]]; then
  echo "cpu:      $(sysctl -n machdep.cpu.brand_string)"
  echo "memory:   $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)) GB"
else
  # Linux on ARM has no model name line in /proc/cpuinfo.
  echo "cpu:      $(lscpu | grep -m1 'Model name' | cut -d: -f2- | xargs) ($(uname -m))"
  echo "memory:   $(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024)) GB"
fi
echo "threads:  1"
echo

for size in 64 256 1024; do
  ./build/release/ember_bench_ops "$repeats" "$size"
  echo
done
