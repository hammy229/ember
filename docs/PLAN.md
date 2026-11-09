# Ember plan: Nov 4 to Nov 20, 2026

Ember is an LLM inference engine written from scratch in C++20, with no ML framework. It runs natively and in the browser through WebAssembly.

**Fri, Nov 13 is publish day: the site goes live and the resume gets updated.** The repo went public early, on Nov 8. The goal is every feature working and tested by the end of Nov 12. Anything that is not goes out later as an update, never half-done in the post.

## Timeline

| Date | Goal | Status |
|---|---|---|
| Wed, Nov 4 | Repo, build presets, tests, CI, WebAssembly build, site skeleton | Done |
| Thu, Nov 5 | Tensors and memory allocator | Done |
| Fri, Nov 6 | Matrix operations, softmax, first benchmark | Done |
| Sat, Nov 7 | Load model weights with memory mapping (15M parameter tiny model), fuzz the loader against corrupt files | Done |
| Sun, Nov 8 | Attention and the full forward pass | Done |
| Mon, Nov 9 | **Milestone 1: logits match Hugging Face**, checked in CI from a saved reference | Done |
| Mon, Nov 9 (night) | Tokenizer, sampling, KV cache, text generation, per-token uncertainty (entropy and top-2 margin) | |
| Tue, Nov 10 | **Milestone 2: text generation in the browser**, private deploy; Qwen3 0.6B on the Mac, NEON SIMD with before-and-after benchmarks | |
| Wed, Nov 11 | Same model against llama.cpp, multithreading, 8-bit quantization and quality report; `ember route` and the routing eval, run overnight | |
| Thu, Nov 12 | MedData due 2pm CST. Autograd with gradients checked, native and in-browser training, attention maps and logit lens, routing view, final benchmarks, reviews, demo video | |
| Fri, Nov 13 | **Publish day:** site live, resume update, launch post with the routing result | |
| Nov 14 to 20 | Fixes, and anything that missed Thursday ships as an update | |

## Scope

Out of scope for this window: CUDA backend, continuous batching, runtime LoRA loading, and a separate Dear ImGui app. The dashboard lives on the website.

**Quantization quality report:** a script anyone can rerun that compares the quantized model to the full model on logit error, KL divergence, and top-1 agreement, next to speed and memory.

**Confidence-gated routing:** local-first routers send each request to the user's own hardware first and escalate to the cloud only when needed, but the hard part is knowing when local is not good enough. An engine that owns its logits can tell from inside: `ember route` tracks per-token entropy and top-2 margin while generating and escalates as soon as the model gets unsure. The eval is a script anyone can rerun on a public benchmark (GSM8K): accuracy against the share of requests kept local, next to random and prompt-length routers at the same share. Escalated questions are scored with the benchmark's answer key, so no paid API is needed, and the write-up says so. The result gets published even if entropy turns out to be a weak signal. It needs a real model, so it waits for Qwen3 0.6B on Nov 10.

**Training:** starts only once Milestone 2 and the router work. Otherwise that time goes to them first.

Stretch goals, only if ahead of schedule: 4-bit quantization and Qwen3-8B on the Mac.

Optional: the mortality pipeline comparison against 0.804 AUC. It needs lab approval and nothing else depends on it.

**Minimum to publish on Nov 13:** Milestones 1 and 2 plus a benchmark page.

**If behind, drop in this order:** the routing view in the browser, then the attention maps and logit lens, then training in the browser (keep native training), then all training, then the quantization report. Never drop the browser demo.

## What each kind of reader should find

Ember is one deep project, not a pile of small ones. A reader should be able to check each claim without trusting the author.

| Reader | What they look for | Where Ember shows it |
|---|---|---|
| Inference or ML systems | Modern C++, attention and KV cache, quantization, profiling, comparison to llama.cpp | The engine, the quantization report, benchmarks against llama.cpp |
| Software engineering | Tests, sanitizers, CI, handling bad input, clear design decisions | Random and fuzz tests, sanitizer builds, `docs/memory.md` |
| ML engineering | Getting the math right, proof the model is correct, training | Logits matching Hugging Face in CI, gradients checked numerically, attention maps |
| AI engineering | Something people can use, with measured quality | The browser demo, the quantization report's quality numbers, the routing eval's cost and accuracy curve |

**The llama.cpp comparison gets published even if Ember loses.** Nobody expects a two week engine to beat it. An honest table of how far behind it is, and why, is the point.

## Rules

- Benchmarks come from a script anyone can rerun, with the hardware listed. No number goes on the site or resume until it has been measured.
- Model weights and patient data never go in this repo.
- Free tools and free tiers only.
