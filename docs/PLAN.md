# Ember plan: Nov 4 to Nov 20, 2026

Ember is an LLM inference engine written from scratch in C++20, with no ML framework. It runs natively and in the browser through WebAssembly.

**Nov 19 is demo day, resume update day, and publish day.** The repo and site stay private until then.

## Timeline

| Date | Goal | Status |
|---|---|---|
| Wed, Nov 4 | Repo, build presets, tests, CI, WebAssembly build, site skeleton | Done |
| Thu, Nov 5 | Tensors and memory allocator | Done |
| Fri, Nov 6 | Matrix operations, softmax, first benchmark | Done |
| Sat, Nov 7 | Load model weights with memory mapping (15M parameter tiny model), fuzz the loader against corrupt files | Done |
| Sun, Nov 8 | Attention and the full forward pass | Done |
| Mon, Nov 9 | **Milestone 1: logits match Hugging Face**, checked in CI from a saved reference | |
| Tue, Nov 10 | Tokenizer, sampling, KV cache, text generation | |
| Wed, Nov 11 | **Milestone 2: text generation in the browser**, first private site | |
| Thu, Nov 12 | Light day, MedData is due at 2pm CST. Catch-up only | |
| Fri, Nov 13 | Run Qwen3 0.6B on the Mac, NEON SIMD with before-and-after benchmarks, same model against llama.cpp | |
| Sat, Nov 14 | Multithreading, 8-bit quantization, quantization quality report | |
| Sun, Nov 15 | Autograd and optimizer, with gradients checked numerically | |
| Mon, Nov 16 | Train a tiny character-level transformer natively, final benchmarks | |
| Tue, Nov 17 | **Milestone 3: training runs live in the browser**, attention maps and logit lens, benchmark and architecture pages | |
| Wed, Nov 18 | **Feature freeze:** ECC code and security reviews, demo video, rehearsal | |
| Thu, Nov 19 | **Demo day, resume update, publish** | |
| Fri, Nov 20 | Fixes only, no new features | |

## Scope

Out of scope for this window: CUDA backend, continuous batching, runtime LoRA loading, and a separate Dear ImGui app. The dashboard lives on the website.

**Quantization quality report:** a script anyone can rerun that compares the quantized model to the full model on logit error, KL divergence, and top-1 agreement, next to speed and memory.

**Training:** starts Nov 15 only if Milestone 2 is done by the end of Nov 12. Otherwise the days go to catching up first.

Stretch goals, only if ahead of schedule: 4-bit quantization and Qwen3-8B on the Mac.

Optional: the mortality pipeline comparison against 0.804 AUC. It needs lab approval and nothing else depends on it.

**Minimum to publish on Nov 19:** Milestones 1 and 2 plus a benchmark page.

**If behind, drop in this order:** the attention maps and logit lens, then training in the browser (keep native training), then all training, then the quantization report. Never drop the browser demo.

## What each kind of reader should find

Ember is one deep project, not a pile of small ones. A reader should be able to check each claim without trusting the author.

| Reader | What they look for | Where Ember shows it |
|---|---|---|
| Inference or ML systems | Modern C++, attention and KV cache, quantization, profiling, comparison to llama.cpp | The engine, the quantization report, benchmarks against llama.cpp |
| Software engineering | Tests, sanitizers, CI, handling bad input, clear design decisions | Random and fuzz tests, sanitizer builds, `docs/memory.md` |
| ML engineering | Getting the math right, proof the model is correct, training | Logits matching Hugging Face in CI, gradients checked numerically, attention maps |
| AI engineering | Something people can use, with measured quality | The browser demo, the quantization report's quality numbers |

**The llama.cpp comparison gets published even if Ember loses.** Nobody expects a two week engine to beat it. An honest table of how far behind it is, and why, is the point.

## Rules

- Benchmarks come from a script anyone can rerun, with the hardware listed. No number goes on the site or resume until it has been measured.
- Model weights and patient data never go in this repo.
- Free tools and free tiers only.
