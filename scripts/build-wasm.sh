#!/usr/bin/env bash
# Compiles the engine to WebAssembly and copies it into the website.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

emcmake cmake -S "$root" -B "$root/build/wasm" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DEMBER_BUILD_TESTS=OFF
cmake --build "$root/build/wasm"

mkdir -p "$root/web/public/wasm"
cp "$root/build/wasm/ember.js" "$root/build/wasm/ember.wasm" "$root/web/public/wasm/"
