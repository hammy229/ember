# Ember

An LLM inference engine written from scratch in C++20, with no ML framework. It runs natively and in the browser through WebAssembly.

![Ember's site: a triangular grid of glowing coals, each one an attention weight the engine is computing live in the browser](docs/screenshots/home.png)

The grid is not an animation. The browser is running made-up queries and keys through the real `MatMul` and `Softmax`, compiled from C++, and each coal shows its real value on hover.

![Softmax with 1000 added to every score: the chances hold, because Ember subtracts each row's largest score before calling exp](docs/screenshots/softmax.png)

Visitors can drag the scores and try to break it. Adding 1000 to every score would overflow the plain formula into NaN, and hiding every word makes the engine return an error rather than an answer.

![Benchmark section: MatMul at 65 ms against 828 ms for textbook loops at 1024 x 1024, then a table timing the engine in the visitor's browser](docs/screenshots/speed.png)

The bars come from [one saved run of `scripts/bench.sh`](docs/benchmarks/2026-11-06-m4.txt). The orange column is measured live in whatever browser opens the page, and here that was Chrome on an Apple M4.

**Status:** Ember loads a real Llama model from Hugging Face's safetensors format and runs the full forward pass from token ids to logits. On the 15M parameter TinyStories model its logits agree with a separate NumPy Llama to within 4.2e-5, and after "Once upon a time" it picks a comma. It has no tokenizer or KV cache yet, so it does not generate text. The loader treats model files as untrusted and is fuzzed in CI. See [the plan](docs/PLAN.md), [how memory works](docs/memory.md), [how the math works](docs/ops.md), [how loading works](docs/loader.md), and [how the forward pass works](docs/forward.md).

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
