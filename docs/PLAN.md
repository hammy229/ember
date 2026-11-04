# Ember plan: Nov 4 to Nov 20, 2026

Ember is an LLM inference engine written from scratch in C++20, with no ML framework. It runs natively and in the browser through WebAssembly.

**Nov 19 is demo day, resume update day, and publish day.** The repo and site stay private until then.

## Timeline

| Date | Goal | Status |
|---|---|---|
| Wed, Nov 4 | Repo, build presets, tests, CI, WebAssembly build, site skeleton | Done |
| Thu, Nov 5 | Tensors and memory allocator | |
| Fri, Nov 6 | Matrix operations, softmax, first benchmark | |
| Sat, Nov 7 | Load model weights with memory mapping (15 MB tiny model) | |
| Sun, Nov 8 | Attention and the full forward pass | |
| Mon, Nov 9 | **Milestone 1: logits match Hugging Face** | |
| Tue, Nov 10 | Tokenizer, sampling, KV cache, text generation | |
| Wed, Nov 11 | **Milestone 2: text generation in the browser**, first private site | |
| Thu, Nov 12 | Light day, MedData is due at 2pm CST. Catch-up only | |
| Fri, Nov 13 | Run Qwen3 0.6B on the Mac | |
| Sat, Nov 14 | NEON SIMD and multithreading, with before-and-after benchmarks | |
| Sun, Nov 15 | 8-bit and 4-bit quantization | |
| Mon, Nov 16 | **Milestone 3: Qwen3-8B runs on the Mac**, final benchmarks | |
| Tue, Nov 17 | "Inside the engine" visuals, benchmark and architecture pages | |
| Wed, Nov 18 | **Feature freeze:** ECC code and security reviews, demo video, rehearsal | |
| Thu, Nov 19 | **Demo day, resume update, publish** | |
| Fri, Nov 20 | Fixes only, no new features | |

## Scope

Out of scope for this window: CUDA backend, continuous batching, runtime LoRA loading, and a separate Dear ImGui app. The dashboard lives on the website.

Optional: the mortality pipeline comparison against 0.804 AUC. It needs lab approval and nothing else depends on it.

**Minimum to publish on Nov 19:** Milestones 1 and 2 plus a benchmark page.

**If behind, drop in this order:** 4-bit quantization, then the visuals, then Qwen3-8B. Never drop the browser demo.

## Rules

- Benchmarks come from a script anyone can rerun, with the hardware listed. No number goes on the site or resume until it has been measured.
- Model weights and patient data never go in this repo.
- Free tools and free tiers only.
