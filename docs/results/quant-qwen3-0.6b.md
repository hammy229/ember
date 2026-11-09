# 8-bit weights on qwen3-0.6b

Made by `scripts/quant_report.py`. Do not edit by hand.

The same model loaded in f32 and with `--q8`, compared at all 207 positions of four short texts.

| Text | Positions | Mean KL | Largest KL | Same top pick | Top-5 overlap |
|---|---|---|---|---|---|
| story | 60 | 0.01157 | 0.5301 | 95.0% | 94.7% |
| encyclopedia | 44 | 0.01190 | 0.4373 | 88.6% | 96.4% |
| code | 46 | 0.00563 | 0.2343 | 100.0% | 97.0% |
| math | 57 | 0.01459 | 0.5579 | 93.0% | 95.1% |
| all | 207 | 0.01115 | 0.5579 | 94.2% | 95.7% |

Generating 128 tokens with 6 threads, median of 3 runs, on macOS-27.2-arm64-arm-64bit-Mach-O:

| Weights | Tokens a second | Peak memory |
|---|---|---|
| f32 | 16.9 | 3423 MB |
| 8-bit | 55.2 | 2377 MB |

KL divergence is in nats. The embedding table and the norms stay f32. Every matrix used in the forward pass, including the output layer, is 8-bit with one scale per 32 weights, and the input to each one is rounded to 8 bits the same way, so the multiply can use integer dot products, as llama.cpp does for its Q8_0 format. That second rounding costs more accuracy than the first: with f32 inputs (commit 33fd481) the same texts gave a mean KL of 0.00264 and the same pick 97.1% of the time, at 45.4 tokens a second.
