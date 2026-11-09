# Ember against llama.cpp

Made by `scripts/compare_llamacpp.py`. Do not edit by hand.

Qwen3 0.6B in 8 bits on both, CPU only, 6 threads, median of 3 runs, on macOS-27.2-arm64-arm-64bit-Mach-O. llama.cpp build 12110 (cd959bae9) with no GPU layers.

| | Ember | llama.cpp | Ember as a share of llama.cpp |
|---|---|---|---|
| Writing, tokens a second | 58.0 | 126.9 | 46% |
| Reading a 121 token prompt, tokens a second | 257.9 | 642.9 | 40% |

Likely reasons, not yet measured one by one: llama.cpp does its 8-bit dot products with integer SIMD instructions and blocks its matrix multiplies for the cache, while Ember widens each 8-bit value to f32 and multiplies one row at a time.
