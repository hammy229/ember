# Ember

An LLM inference engine written from scratch in C++20, with no ML framework. It runs natively and in the browser through WebAssembly.

![Ember's site: a triangular grid of coals from pale ash to deep ember, each one an attention weight the engine is computing live in the browser](docs/screenshots/home.png)

The grid is not an animation. The browser is running made-up queries and keys through the real `MatMul` and `Softmax`, compiled from C++, and each coal shows its real value on hover.

![Softmax with 1000 added to every score: the chances hold, because Ember subtracts each row's largest score before calling exp](docs/screenshots/softmax.png)

Visitors can drag the scores and try to break it. Adding 1000 to every score would overflow the plain formula into NaN, and hiding every word makes the engine return an error rather than an answer.

![Benchmark section: MatMul at 65 ms against 828 ms for textbook loops at 1024 x 1024, then a table timing the engine in the visitor's browser](docs/screenshots/speed.png)

The bars come from [one saved run of `scripts/bench.sh`](docs/benchmarks/2026-11-06-m4.txt). The orange column is measured live in whatever browser opens the page, and here that was Chrome on an Apple M4.

![Model section: logits within 4.2e-5 of a separate Llama, 815 tokens a second, and a bar chart where NumPy runs 128 tokens in 21 ms against Ember's 158 ms](docs/screenshots/model.png)

The real-model results, with the baseline Ember still loses to drawn in slate. They come from [the saved results run](docs/results/2026-11-08-apple-m4.md).

**Status:** Ember loads a real Llama model from Hugging Face's safetensors format and runs the full forward pass from token ids to logits. On the 15M parameter TinyStories model its logits agree with a separate NumPy Llama to within 4.2e-5, and after "Once upon a time" it picks a comma. The numbers are in [Results](#results). It has no tokenizer or KV cache yet, so it does not generate text. The loader treats model files as untrusted and is fuzzed in CI. See [the plan](docs/PLAN.md), [how memory works](docs/memory.md), [how the math works](docs/ops.md), [how loading works](docs/loader.md), and [how the forward pass works](docs/forward.md).

## Results

Every number here comes from one run of [`scripts/results.sh`](scripts/results.sh), saved in [docs/results/2026-11-08-apple-m4.md](docs/results/2026-11-08-apple-m4.md) with the commit, compiler, and machine. Rerun it and you get your own copy.

| Claim | Measured | How it is checked |
|---|---|---|
| The math is right | Largest logit difference 4.2e-5 from a separate NumPy Llama, across all 32,000 logits, with the same top token at all 156 positions tested | `scripts/reference_logits.py` on three inputs from 5 to 128 tokens |
| It runs a real model | 15,191,712 parameters, 815 tokens a second on one thread, 147 MB of memory for the whole process | `ember_bench_model` on the TinyStories Llama |
| Speed work is measured, not guessed | Spelling out the SIMD registers in `Linear` took one token from 1.98 ms to 1.23 ms (1.6x) with logits identical to the bit | `results.sh --before 08c2e73` builds the old commit and times both |
| It is honest about where it stands | NumPy with Apple's hand-tuned BLAS runs 128 tokens in 21.0 ms. Ember takes 158.1 ms, 7.5x slower | Same script, same machine, same tokens |
| Bad input cannot crash it | Every truncation and thousands of corrupted headers fail cleanly under AddressSanitizer, and libFuzzer runs on every push | `tests/loader_corruption_test.cpp`, `fuzz/`, CI |
| The tests reach the code | 326 tests, covering 97.8% of lines and 86.0% of branches in `src/` | `scripts/coverage.sh` |

The gap to NumPy is the to-do list: hand-written NEON kernels, more than one thread, and 8-bit weights are on [the plan](docs/PLAN.md), each with a before and after from the same script.

## Why it matters

Most people who use a language model send their words to someone else's server. That costs money for every request, needs a connection, and means the text leaves the device, which rules it out for things like medical notes or schoolwork that should stay private. An engine small enough to run on a laptop, and soon in a browser tab, does none of that. The model runs where the data already is, for free.

Doing that well is mostly engineering: loading weights safely from files nobody checked, getting the math exactly right, fitting in a browser's memory, and making it fast without guessing. Ember is a from-scratch attempt at all of it, with the evidence in the repo.

## How it is engineered

| What to look for | Where it is |
|---|---|
| Modern C++ without exceptions | `Status` and `StatusOr` for every failure, `[[nodiscard]]`, `std::span`, concepts, RAII for mapped files and arenas |
| Memory you can reason about | One arena per forward pass, scoped scratch that frees itself, no allocation per op ([docs/memory.md](docs/memory.md)) |
| Treating input as hostile | Bounds, overflow, alignment, and overlap checks on every file and tensor ([docs/loader.md](docs/loader.md)) |
| Testing beyond happy paths | Reference implementations, property tests, corruption tests, fuzzing, and mutation checks that a broken RoPE, scale, or mask fails the suite ([docs/forward.md](docs/forward.md)) |
| CI that means something | Format check, release and AddressSanitizer plus UBSan builds, fuzzing, the WebAssembly build, and the site, on every push |
| Performance with receipts | Benchmarks that refuse debug builds, record the machine, and compare commits ([docs/ops.md](docs/ops.md), [docs/results](docs/results)) |
| Writing it down | A design doc per layer, explaining what was chosen, what it cost, and what was measured |

## Build

You need CMake 3.24+, Ninja, and a C++20 compiler.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
./build/debug/ember
```

Use the `sanitize` preset to run the tests with AddressSanitizer and UBSan, `release` for an optimized build, or `fuzz` for the libFuzzer targets ([docs/loader.md](docs/loader.md) says how).

To use Ember inside another CMake project, add it with `add_subdirectory`. Its tests and `-Werror` stay off unless you turn on `EMBER_BUILD_TESTS` and `EMBER_WERROR`.

To run a real model, download the 15M parameter test model (it goes in `models/`, which git ignores) and ask for the likeliest next tokens:

```sh
./scripts/fetch-model.sh
cmake --preset release && cmake --build --preset release
./build/release/ember inspect models/tinyllama-15M
./build/release/ember logits models/tinyllama-15M 1 9038 2501 263 931
```

`ember_bench [repeats] [elements]` times a placeholder workload and compares the arena to the system allocator on your machine. Use the `release` preset for numbers worth reading.

`./scripts/bench.sh [repeats]` builds the `release` preset, times the math kernels at several sizes, and prints the commit, compiler, flags, and hardware next to the numbers.

## Website

You need Node 24+ and Emscripten. The first command compiles the engine to WebAssembly and copies it into the site.

```sh
./scripts/build-wasm.sh
cd web
npm ci
npm run dev
```

After building, `node tests/wasm_smoke.mjs` checks that the WebAssembly build loads, reports the right version, passes its 32-bit memory and math checks, and gives the site correct answers.
