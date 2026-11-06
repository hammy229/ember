# Ember

An LLM inference engine written from scratch in C++20, with no ML framework. It runs natively and in the browser through WebAssembly.

![Ember's site: a triangular grid of glowing coals, each one an attention weight the engine is computing live in the browser](docs/screenshots/home.png)

The grid is not an animation. The browser is running made-up queries and keys through the real `MatMul` and `Softmax`, compiled from C++, and each coal shows its real value on hover.

![Softmax with 1000 added to every score: the chances hold, because Ember subtracts each row's largest score before calling exp](docs/screenshots/softmax.png)

Visitors can drag the scores and try to break it. Adding 1000 to every score would overflow the plain formula into NaN, and hiding every word makes the engine return an error rather than an answer.

![Benchmark section: MatMul at 65 ms against 828 ms for textbook loops at 1024 x 1024, then a table timing the engine in the visitor's browser](docs/screenshots/speed.png)

The bars come from [one saved run of `scripts/bench.sh`](docs/benchmarks/2026-11-06-m4.txt). The orange column is measured live in whatever browser opens the page, and here that was Chrome on an Apple M4.

**Status:** the memory layer and the core math are done: tensors, an arena allocator, matrix multiplication, softmax, and sanitizer support. The engine does not load a model or generate text yet. See [the plan](docs/PLAN.md), [how memory works](docs/memory.md), and [how the math works](docs/ops.md).

## Build

You need CMake 3.24+, Ninja, and a C++20 compiler.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
./build/debug/ember
```

Use the `sanitize` preset to run the tests with AddressSanitizer and UBSan, or `release` for an optimized build.

To use Ember inside another CMake project, add it with `add_subdirectory`. Its tests and `-Werror` stay off unless you turn on `EMBER_BUILD_TESTS` and `EMBER_WERROR`.

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
