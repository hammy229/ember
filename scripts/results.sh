#!/usr/bin/env bash
# Produces every number Ember claims, in one run, and saves them with the machine and commit that
# produced them. Nothing goes in the README or on the site unless it came out of this script.
#
#   ./scripts/results.sh                   writes docs/results/<date>-<cpu>.md
#   ./scripts/results.sh --print           prints the report instead of saving it
#   ./scripts/results.sh --before COMMIT   also times the model at an older commit, side by side
#
# Needs CMake, Ninja, a C++20 compiler, Python 3 with NumPy, and clang's coverage tools.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
model="models/tinyllama-15M"
scratch="$(mktemp -d)"
print=false
before=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --print) print=true ;;
    --before)
      before="$(git rev-parse --short "$2")"
      shift
      ;;
    *)
      echo "usage: scripts/results.sh [--print] [--before COMMIT]" >&2
      exit 2
      ;;
  esac
  shift
done
cleanup() {
  [[ -z "$before" ]] || git worktree remove --force "$scratch/before" 2>/dev/null || true
  rm -rf "$scratch"
}
trap cleanup EXIT

# The older commit is built in a separate worktree, so the working tree is never touched.
if [[ -n "$before" ]]; then
  git worktree add --detach "$scratch/before" "$before" >/dev/null 2>&1
  cmake -S "$scratch/before" -B "$scratch/before/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DEMBER_BUILD_TESTS=OFF >/dev/null
  cmake --build "$scratch/before/build" --target ember_bench_model >/dev/null
fi

./scripts/fetch-model.sh >/dev/null
cmake --preset release >/dev/null && cmake --build --preset release >/dev/null
cmake --preset debug >/dev/null && cmake --build --preset debug >/dev/null

cached() { grep -m1 "^$1:" build/release/CMakeCache.txt | cut -d= -f2-; }
if [[ "$(cached CMAKE_BUILD_TYPE)" != "Release" || "$(cached EMBER_SANITIZE)" == "ON" ]]; then
  echo "build/release is not a plain release build, delete it and run this again" >&2
  exit 1
fi

if [[ "$(uname)" == "Darwin" ]]; then
  cpu="$(sysctl -n machdep.cpu.brand_string)"
  memory="$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)) GB"
  one_thread=(env VECLIB_MAXIMUM_THREADS=1)
else
  cpu="$(lscpu | grep -m1 'Model name' | cut -d: -f2- | xargs) ($(uname -m))"
  memory="$(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024)) GB"
  one_thread=(env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1)
fi
local_changes=""
[[ -z "$(git status --porcelain)" ]] || local_changes=" (with local changes)"

# Token ids for real sentences, so the agreement check runs on text the model was trained for.
# 5 tokens: "Once upon a time". 23 tokens: "Once upon a time, there was a little girl named Lily. She
# wanted to play with her little dog". 128 tokens: spread across the whole vocabulary.
short=(1 9038 2501 263 931)
sentence=(1 9038 2501 263 931 29892 727 471 263 2217 7826 4257 365 2354 29889 2296 5131 304 1708 411 902 2217 11203)
long=()
for ((i = 0; i < 128; i++)); do long+=($(((i * 7919 + 1) % 32000))); done

agreement() {
  local name="$1"
  shift
  ./build/release/ember logits "$model" --out "$scratch/$name.f32" "$@" >/dev/null
  echo "### $name ($# tokens)"
  echo
  echo '```'
  python3 scripts/reference_logits.py "$model" "$scratch/$name.f32" "$@"
  echo '```'
  echo
}

report() {
  echo "# Ember results, $(date +%Y-%m-%d)"
  echo
  echo "Made by \`scripts/results.sh\`. Rerun it to check any number here on your own machine."
  echo
  echo '```'
  echo "commit:   $(git rev-parse --short HEAD)$local_changes"
  echo "compiler: $("$(cached CMAKE_CXX_COMPILER)" --version | head -1)"
  echo "flags:    $(cached CMAKE_CXX_FLAGS_RELEASE)"
  echo "cpu:      $cpu"
  echo "memory:   $memory"
  echo "numpy:    $(python3 -c 'import numpy; print(numpy.__version__)')"
  echo "model:    nickypro/tinyllama-15M, pinned by scripts/fetch-model.sh"
  echo '```'
  echo
  echo "## 1. Correctness: Ember against a separate NumPy Llama"
  echo
  echo "Every one of the 32,000 logits at every position, compared with \`scripts/reference_logits.py\`."
  echo
  agreement "Once upon a time" "${short[@]}"
  agreement "a full sentence" "${sentence[@]}"
  agreement "128 tokens across the vocabulary" "${long[@]}"
  echo "## 2. Speed and memory"
  echo
  echo "One thread, \`ember_bench_model\`, median of 10 runs."
  echo
  echo '```'
  ./build/release/ember_bench_model "$model" 10
  echo '```'
  echo
  if [[ -n "$before" ]]; then
    echo "## 2b. The same benchmark at commit $before, built the same way"
    echo
    echo '```'
    "$scratch/before/build/ember_bench_model" "$model" 10
    echo '```'
    echo
  fi
  echo "## 3. Baseline: the same model in NumPy at 128 tokens"
  echo
  echo "NumPy hands matrix products to the system BLAS, a library tuned by hand for this CPU."
  echo
  echo '```'
  echo "all threads: $(python3 scripts/reference_logits.py --time "$model" "$scratch/128 tokens across the vocabulary.f32" "${long[@]}" | head -1)"
  echo "one thread:  $("${one_thread[@]}" python3 scripts/reference_logits.py --time "$model" "$scratch/128 tokens across the vocabulary.f32" "${long[@]}" | head -1)"
  echo '```'
  echo
  echo "## 4. Tests"
  echo
  echo '```'
  ctest --preset debug 2>&1 | grep -E "tests passed"
  echo '```'
  echo
  echo "## 5. Coverage of src/ by the test suite"
  echo
  echo '```'
  ./scripts/coverage.sh | grep -E "^TOTAL" | awk '{print $1, "lines", $(NF-3), "branches", $NF}'
  echo '```'
}

if $print; then
  report
else
  slug="$(echo "$cpu" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')"
  out="docs/results/$(date +%Y-%m-%d)-$slug.md"
  mkdir -p docs/results
  report >"$out"
  echo "wrote $out"
fi
