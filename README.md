# Ember

**An LLM engine written from scratch in C++20, with no ML framework. It runs Llama and Qwen3 models from Hugging Face, writes text, knows when it is unsure, trains small transformers with its own autograd, and compiles to WebAssembly so all of it also runs in a browser tab.**

[![CI](https://github.com/hammy229/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/hammy229/ember/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Live demo: [ember-rosy-three.vercel.app](https://ember-rosy-three.vercel.app)**

**Links:** [Results](docs/results/2026-11-11-apple-m4.md) · [Against llama.cpp](docs/results/llamacpp-qwen3-0.6b.md) · [8-bit report](docs/results/quant-qwen3-0.6b.md) · [How loading works](docs/loader.md) · [How the forward pass works](docs/forward.md) · [Plan](docs/PLAN.md) · [Arham Doshi](https://arhamdoshi.dev) · [LinkedIn](https://linkedin.com/in/arhamdoshi)

![Ember's site: a triangular grid of coals from pale ash to deep ember, each one an attention weight the engine is computing live in the browser](docs/screenshots/home.png)

The grid is not an animation. The browser is running queries and keys through the real `MatMul` and `Softmax`, compiled from C++, and each coal shows its value on hover.

## Results

### On a real model: Qwen3 0.6B

Each row names the script or test that produced it; the numbers are in the linked results files, with the machine.

| Claim | Measured | How it is checked |
|---|---|---|
| It runs Qwen3 exactly as Hugging Face does | Largest difference 1.0e-4 across the top 10 logits at all 73 positions of three inputs (a sentence, a chat-format math question, code), with the same next token at all 73 | `tests/hf_reference_test.cpp` against `scripts/hf_reference_qwen3.py`, with `EMBER_QWEN3_DIR` set (the 1.5 GB model is not downloaded in CI) |
| Its tokenizers match Hugging Face's | The same token ids on 20 hard strings for Llama (CI) and 36 for Qwen3: NFC normalization, emoji, Hindi, Thai, CRLF, contractions, chat markup | `tests/tokenizer_test.cpp` against `scripts/hf_tokens.py`; Unicode 16 tables made by `scripts/gen_unicode_tables.py`, which checks its own NFC against Python's on every code point |
| 8-bit weights make it faster and smaller | 58.9 tokens a second against 26.2 in f32, peak memory 2377 MB against 3545 MB, with 6 threads on an Apple M4 | [`scripts/quant_report.py`](docs/results/quant-qwen3-0.6b.md), median of 3 runs |
| What 8 bits cost, stated plainly | Mean KL divergence 0.011 nats from the f32 model, the same top pick 94.2% of the time over 207 positions | Same report; with f32 inputs to the 8-bit matrices it was 0.0026 and 97.1%, at 45 tokens a second |
| How far behind llama.cpp it is | 46% of llama.cpp's speed writing (58.0 against 126.9 tokens a second) and 40% reading a prompt, same model, same 8-bit format, CPU only, 6 threads | [`scripts/compare_llamacpp.py`](docs/results/llamacpp-qwen3-0.6b.md), published as measured. It was 33% and 19% before integer dot products and a lock-free thread pool |
| It knows when it is likely wrong | On all 1319 GSM8K test questions, the highest per-token entropy in an answer spots Qwen3's wrong answers with AUROC 0.740, against 0.687 for question length alone (gap +0.053, 95% CI [+0.023, +0.094]). With the threshold set on one half and tested on the other, escalating 28.9% of questions caught 51.2% of the mistakes, and accuracy on the rest rose from 61.2% to 73.3% | [`scripts/route_eval.py`](docs/results/route-gsm8k.md), pinned dataset revision, bootstrap intervals; no cloud model was called, so this is not a combined system's score |
| Threads change speed, not answers | Results identical to the bit for any number of threads | `tests/threads_test.cpp`, and a ThreadSanitizer job in CI |
| It can learn, not just run | Every autograd operation and a whole transformer's gradients match finite differences in double precision | `tests/autograd_test.cpp`, `tests/char_model_test.cpp` |

![Routing section of the site: Qwen3 0.6B gets 62.1% of GSM8K right alone; escalating 28.9% of held-out questions catches 51.2% of its mistakes and lifts accuracy on the rest to 73.3%; max entropy scores AUROC 0.740 against 0.687 for question length and 0.478 for random](docs/screenshots/route.png)

### On the 15M parameter TinyStories Llama

Every number here comes from one run of [`scripts/results.sh`](scripts/results.sh), saved in [docs/results/2026-11-11-apple-m4.md](docs/results/2026-11-11-apple-m4.md) with the commit, compiler, and machine. The SIMD row is from the [Nov 8 run](docs/results/2026-11-08-apple-m4.md), which timed the older commit too. Rerun it and you get your own copy.

| Claim | Measured | How it is checked |
|---|---|---|
| It matches Hugging Face | Largest logit difference 3.0e-5 from `transformers`' own logits, with the same top token at all 156 positions tested | `tests/hf_reference_test.cpp` against logits saved by `scripts/hf_reference.py`, run in CI on Linux, under sanitizers, and on macOS |
| Every logit is right | Largest difference 4.2e-5 from a separate NumPy Llama, across all 32,000 logits at the same 156 positions | `scripts/reference_logits.py` on three inputs from 5 to 128 tokens |
| It runs a real model | 15,191,712 parameters, 846 tokens a second on one thread, 147 MB of memory for the whole process | `ember_bench_model` on the TinyStories Llama |
| Speed work is measured, not guessed | Spelling out the SIMD registers in `Linear` took one token from 1.98 ms to 1.23 ms (1.6x) with logits identical to the bit | `results.sh --before 08c2e73` builds the old commit and times both |
| It is honest about where it stands | NumPy with Apple's hand-tuned BLAS runs 128 tokens in 20.3 ms. Ember takes 154.2 ms, 7.6x slower | Same script, same machine, same tokens |
| Bad input cannot crash it | Every truncation and thousands of corrupted headers fail cleanly under AddressSanitizer, and libFuzzer runs on every push | `tests/loader_corruption_test.cpp`, `fuzz/`, CI |
| The tests reach the code | 431 tests, covering 91.9% of lines and 74.5% of branches in `src/`. The gap is mostly `tokenizer.cpp` (70% of lines), whose Qwen3 tests need the 1.5 GB model, which the coverage run does not load | `scripts/coverage.sh` |

![Model section of the site: logits within 3.0e-5 of Hugging Face's, the same next token at 156 of 156 positions, 846 tokens a second, and a bar chart where NumPy runs 128 tokens in 20 ms against Ember's 154 ms](docs/screenshots/model.png)

These rows are from the Nov 11 run, on one thread and f32 weights; threads and 8 bits are in the Qwen3 rows above.

## Why it matters

Most people who use a language model send their words to someone else's server. That costs money for every request, needs a connection, and means the text leaves the device, which rules it out for things like medical notes or schoolwork that should stay private. An engine small enough to run on a laptop, and soon in a browser tab, does none of that. The model runs where the data already is, for free.

Doing that well is mostly engineering: loading weights safely from files nobody checked, getting the math exactly right, fitting in a browser's memory, and making it fast without guessing. Ember is a from-scratch attempt at all of it, with the evidence in the repo.

## How it works

1. **Load.** `MappedFile` maps `model.safetensors` read-only. A strict JSON parser reads the header, every offset and shape is checked, and f16 or bf16 weights are widened to f32 exactly, or quantized straight to 8 bits a row at a time. ([docs/loader.md](docs/loader.md))
2. **Tokenize.** Llama's SentencePiece-style BPE and Qwen3's byte-level BPE, with NFC normalization and Qwen's split regex written out by hand, both matching Hugging Face's ids.
3. **Compute.** Embedding, then per layer RMS norm, attention with RoPE, grouped-query heads, and Qwen3's per-head query and key norms, and a SwiGLU MLP, then the output layer. A KV cache keeps each layer's keys and values so a new token costs one row. ([docs/forward.md](docs/forward.md), [docs/ops.md](docs/ops.md))
4. **Generate and route.** Greedy, temperature, top-k, and top-p sampling, and for every token the entropy and top-2 margin of the model's choice. `ember route` uses them to stop mid-answer and hand the request to a bigger model once the local one gets unsure.
5. **Go fast.** Matrix rows split across a thread pool, and 8-bit weights multiplied with the CPU's integer dot product instructions.
6. **Learn.** A reverse-mode autograd and AdamW train a small character-level transformer from random weights, natively or in the browser.
7. **Run anywhere.** The same C++ compiles natively and to WebAssembly, and the site calls it from the browser: writing, training, and attention maps all run in the visitor's tab.

Design choices that matter:

| Choice | Why |
|---|---|
| `Status` and `StatusOr` instead of exceptions | Every failure is a value the caller must handle, and ignoring one is a compile error |
| Read-only `TensorView` for weights | Writing to a mapped weight would crash, so it does not compile |
| Overlap checks on every op | Writing over an input you are still reading gives a silent wrong answer, so it is refused |
| Copy transposed keys before `MatMul` | Measured 11.7x faster at 1024 x 1024 than reading them sideways ([docs/ops.md](docs/ops.md)) |
| Explicit SIMD vector types in `Linear` | Small rewrites swung the auto-vectorizer's speed by more than half, so the registers are spelled out |

![Softmax with 1000 added to every score: the chances hold, because Ember subtracts each row's largest score before calling exp](docs/screenshots/softmax.png)

On the site, visitors can drag the softmax scores and try to break it. Adding 1000 to every score would overflow the plain formula into NaN, and hiding every word makes the engine return an error rather than an answer.

## Tech stack

C++20, CMake and Ninja, GoogleTest, libFuzzer, AddressSanitizer and UBSan, clang source coverage, Emscripten (WebAssembly), Next.js and TypeScript for the site, Python and NumPy for the reference checks, GitHub Actions on Linux and macOS.

## Getting started

You need CMake 3.24+, Ninja, and a C++20 compiler.

```sh
cmake --preset release && cmake --build --preset release
./scripts/fetch-model.sh                       # 15M parameter test model into models/, which git ignores
./build/release/ember inspect models/tinyllama-15M
./build/release/ember logits models/tinyllama-15M 1 9038 2501 263 931   # "Once upon a time"
./build/release/ember generate models/tinyllama-15M --stats "Once upon a time"

./scripts/fetch-model.sh qwen3-0.6b             # Qwen3 0.6B, 1.5 GB
./build/release/ember generate models/qwen3-0.6b --chat --q8 --threads 6 "Why is the sky blue?"
./build/release/ember route models/qwen3-0.6b --threads 6 --signal "max entropy" --threshold 2.272105 "A hard question"   # threshold from docs/results/route-gsm8k.md, f32 only
./build/release/ember train data/text/shakespeare.txt --steps 1000   # any text file
```

The site needs Node 24+ and Emscripten:

```sh
./scripts/build-wasm.sh
cd web && npm ci && npm run dev
```

To use Ember inside another CMake project, add it with `add_subdirectory`. Its tests and `-Werror` stay off unless you turn on `EMBER_BUILD_TESTS` and `EMBER_WERROR`.

## Project structure

```
include/ember/   public headers: tensors, arena, ops, layers, loader, forward pass, tokenizer,
                 sampling, generation, router, 8-bit weights, threads, autograd, character model
src/             the engine
tests/           GoogleTest suites, reference implementations, corruption tests
fuzz/            libFuzzer targets for the JSON parser and the whole loader
tools/           the ember CLI and the benchmarks
scripts/         fetch models, benchmarks, coverage, reference logits and tokens, the routing
                 eval, the 8-bit report, the llama.cpp comparison
wasm/            exports for the WebAssembly build
web/             the Next.js site
docs/            one design doc per layer, the plan, saved benchmarks and results
```

## Testing

```sh
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
cmake --preset sanitize && cmake --build --preset sanitize && ctest --preset sanitize
./scripts/coverage.sh       # line and branch coverage of src/
./scripts/results.sh        # every number in this README, saved with the machine and commit
```

CI runs on every push: a clang-format check, release, AddressSanitizer plus UBSan, and ThreadSanitizer builds with GCC, a release build with Apple clang on macOS, a minute of fuzzing per target, the WebAssembly build with its smoke test, and the site build. The forward pass is checked against a separate double-precision implementation, and breaking RoPE, the attention scale, or the causal mask on purpose makes those tests fail. The Linux, sanitizer, and macOS jobs also download the real model and check its logits against the ones Hugging Face's `transformers` produced, and fail rather than skip if the download is missing.

![Benchmark section: MatMul at 65 ms against 828 ms for textbook loops at 1024 x 1024, then a table timing the engine in the visitor's browser](docs/screenshots/speed.png)

The bars come from [one saved run of `scripts/bench.sh`](docs/benchmarks/2026-11-06-m4.txt). The orange column is measured live in whatever browser opens the page.

## Limitations and next steps

- llama.cpp is still about twice as fast on the same CPU. Likely reasons, not yet measured one by one: its matrix multiplies are blocked for the cache, and it runs a prompt as one large multiply where Ember goes row by row.
- Rounding the inputs to 8 bits costs more accuracy than rounding the weights; a scale per smaller block, or f32 inputs for the most sensitive layers, are the next things to try.
- Llama and Qwen3 models in one safetensors file. Llama 3's scaled RoPE and sliding-window attention are refused rather than run wrong.
- The routing threshold was calibrated on one model, one benchmark, f32 weights, and greedy decoding. Math answers may make uncertainty easier to read than open-ended chat, and 8-bit weights would need their own threshold.
- The browser runs the 15M model only; Qwen3 0.6B needs more memory than a 32-bit WebAssembly tab should take.
- The dated plan, with what is done and what is next, is in [docs/PLAN.md](docs/PLAN.md).

## License

MIT, see [LICENSE](LICENSE). Built by [Arham Doshi](https://arhamdoshi.dev).
