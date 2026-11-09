# 8-bit weights on qwen3-0.6b

Made by `scripts/quant_report.py`. Do not edit by hand.

The same model loaded in f32 and with `--q8`, compared at all 207 positions of four short texts.

| Text | Positions | Mean KL | Largest KL | Same top pick | Top-5 overlap |
|---|---|---|---|---|---|
| story | 60 | 0.00255 | 0.0927 | 98.3% | 97.0% |
| encyclopedia | 44 | 0.00205 | 0.0578 | 95.5% | 96.8% |
| code | 46 | 0.00187 | 0.0790 | 100.0% | 96.5% |
| math | 57 | 0.00381 | 0.0961 | 94.7% | 95.8% |
| all | 207 | 0.00264 | 0.0961 | 97.1% | 96.5% |

Generating 128 tokens with 6 threads, median of 3 runs, on macOS-27.2-arm64-arm-64bit-Mach-O:

| Weights | Tokens a second | Peak memory |
|---|---|---|
| f32 | 27.9 | 3555 MB |
| 8-bit | 45.4 | 2377 MB |

KL divergence is in nats. The embedding table and the norms stay f32; every matrix used in the forward pass, including the output layer, is 8-bit with one scale per 32 weights.
