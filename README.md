# Ember

An LLM inference engine written from scratch in C++20, with no ML framework. It runs natively and in the browser through WebAssembly.

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

After building, `node tests/wasm_smoke.mjs` checks that the WebAssembly build loads, reports the right version, and passes its 32-bit memory checks.
