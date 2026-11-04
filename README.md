# Ember

An LLM inference engine written from scratch in C++20, with no ML framework. It runs natively and in the browser through WebAssembly.

**Status:** early setup. The engine does not generate text yet. See [the plan](docs/PLAN.md).

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

`ember_bench [repeats] [elements]` times a placeholder workload on your machine.

## Website

You need Node 24+ and Emscripten. The first command compiles the engine to WebAssembly and copies it into the site.

```sh
./scripts/build-wasm.sh
cd web
npm ci
npm run dev
```

After building, `node tests/wasm_smoke.mjs` checks that the WebAssembly build loads and reports the right version.
