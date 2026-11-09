#!/usr/bin/env bash
# Downloads a model Ember is tested on into models/, which git ignores.
#
#   ./scripts/fetch-model.sh              the 15M parameter TinyStories Llama (30 MB), used by CI
#   ./scripts/fetch-model.sh qwen3-0.6b   Qwen3 0.6B (1.5 GB)
#
# tinyllama-15M is nickypro/tinyllama-15M on Hugging Face (MIT license), Andrej Karpathy's
# TinyStories model converted to the Hugging Face Llama format. qwen3-0.6b is Qwen/Qwen3-0.6B
# (Apache 2.0). The revision and every file's SHA-256 are pinned, so a changed or corrupted
# download fails here instead of somewhere deep in the engine.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="${1:-tinyllama-15M}"
case "$name" in
  tinyllama-15M)
    revision="a97e01fa7d54c088df9dc1b53d581a863ce08cd6"
    base="https://huggingface.co/nickypro/tinyllama-15M/resolve/$revision"
    files=(
      "config.json 894c283ab2d3373b3dd4b88a1a868f4292c5cab5f0e75976fbdf993eccb24766"
      "model.safetensors b9e8d4614e24c89e99502000294af0aab73e9266029357377578e0a504b7f8d9"
      "tokenizer.json 8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1"
    )
    ;;
  qwen3-0.6b)
    revision="c1899de289a04d12100db370d81485cdf75e47ca"
    base="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/$revision"
    files=(
      "config.json 660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
      "generation_config.json 2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"
      "tokenizer.json aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
      "tokenizer_config.json d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
      "model.safetensors f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
    )
    ;;
  *)
    echo "usage: scripts/fetch-model.sh [tinyllama-15M | qwen3-0.6b]" >&2
    exit 2
    ;;
esac
dir="$root/models/$name"

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
