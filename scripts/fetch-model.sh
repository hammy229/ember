#!/usr/bin/env bash
# Downloads the 15M parameter Llama model Ember is tested on into models/, which git ignores.
#
# The model is nickypro/tinyllama-15M on Hugging Face (MIT license), Andrej Karpathy's TinyStories
# model converted to the Hugging Face Llama format. The revision and every file's SHA-256 are pinned,
# so a changed or corrupted download fails here instead of somewhere deep in the engine.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="$root/models/tinyllama-15M"
revision="a97e01fa7d54c088df9dc1b53d581a863ce08cd6"
base="https://huggingface.co/nickypro/tinyllama-15M/resolve/$revision"

files=(
  "config.json 894c283ab2d3373b3dd4b88a1a868f4292c5cab5f0e75976fbdf993eccb24766"
  "model.safetensors b9e8d4614e24c89e99502000294af0aab73e9266029357377578e0a504b7f8d9"
  "tokenizer.json 8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1"
)

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

mkdir -p "$dir"
for entry in "${files[@]}"; do
  read -r name expected <<<"$entry"
  path="$dir/$name"
  if [[ ! -f "$path" || "$(sha256 "$path")" != "$expected" ]]; then
    echo "downloading $name"
    curl --fail --location --silent --show-error -o "$path.part" "$base/$name"
    mv "$path.part" "$path"
  fi
  actual="$(sha256 "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "error: $name has SHA-256 $actual, expected $expected" >&2
    rm -f "$path"
    exit 1
  fi
done
echo "model ready in $dir"
