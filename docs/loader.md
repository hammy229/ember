# How Ember loads a model

A model is two files Hugging Face downloads side by side: `config.json`, which says the shape of the network, and `model.safetensors`, which holds the weights. Both come from the internet, so the loader treats every byte of them as untrusted. This page explains what it checks, why, and how that was tested.

## The pieces

| Piece | File | Job |
|---|---|---|
| `MappedFile` | `include/ember/mapped_file.h` | Maps a file read-only into memory. Nothing is read until it is touched |
| `Json` | `include/ember/json.h` | A strict JSON parser, just big enough for a header and a config |
| `Safetensors` | `include/ember/safetensors.h` | Checks the header and gives each tensor as a span of the file |
| `LlamaConfig`, `LlamaModel` | `include/ember/llama.h` | Reads the config, finds every weight by name, checks its shape, widens it to f32 |

`./scripts/fetch-model.sh` downloads the 15M parameter TinyStories Llama (`nickypro/tinyllama-15M`, MIT license) into `models/`, which git ignores, and checks a pinned SHA-256 for each file. Then:

```sh
./build/release/ember inspect models/tinyllama-15M
```

## Memory mapping

Mapping the file means opening a model costs almost nothing, and the operating system reads pages in as the weights are used. The pages are read-only, and the loader only ever hands them out as `TensorView`, so writing to a weight is a compile error instead of a crash.

f32 weights are used straight from the mapping with no copy. The test model stores f16, and every op works in f32, so its weights are widened once at load time into memory the model owns. Every f16 and bf16 value has an exact f32 value, so widening loses nothing. `F16ToF32` is checked against the definition on all 65,536 bit patterns.

## What a bad file cannot do

| A file that tries to | Is stopped by |
|---|---|
| Claim a header longer than the file | The length is checked against the file size before anything is read |
| Send a huge header to eat memory | Headers over 16 MB, JSON over 2^20 values, and nesting over 64 levels are refused |
| Make the parser slow | Duplicate keys are found with a hash set, so checking is linear, not quadratic |
| Point a tensor past the end | Each tensor's offsets must lie inside the data |
| Make a shape's size overflow | `Shape::Make` refuses any element count that could overflow, including on 32-bit WebAssembly, where `[65536, 65536]` would wrap to zero |
| Claim a shape bigger than its bytes | A tensor's byte count must equal its shape times its element size, exactly |
| Hide extra bytes, or share bytes between tensors | Tensors must cover the data with no gap and no overlap, as the format requires |
| Give the same key twice | Refused, because two readers could disagree on which value counts |
| Put a tensor on a misaligned address | Reading an f32 there is undefined behavior, so f32 is copied somewhere aligned first |
| Be a different architecture | Every expected weight must exist with the shape the config implies, and any weight nobody asked for is an error |
| Use settings Ember does not implement | RoPE scaling, biases, and activations other than SiLU are refused, so the model never runs slightly wrong |

The config's numbers get the same treatment: every size must be a whole number in a sane range, the heads must divide evenly, and `rope_theta` is read from either the old top-level field or the newer `rope_parameters` block, since reading only one would quietly run a newer model with the wrong base.

Numbers are read with `std::from_chars`, not `strtod`, because `strtod` follows the process locale and would read `1.5` as `1` in a locale that writes decimals with a comma.

## How it was tested

- **Unit tests** build files in memory with `tests/safetensors_writer.h`, including bad files for the cases in the table above.
- **Corruption tests** (`tests/loader_corruption_test.cpp`) take a valid model and try every possible truncation, thousands of random byte changes to the header, every single digit swap in the header, and random edits to the config. Each must fail cleanly or load a model of the right shape. CI runs them under AddressSanitizer and UBSan, so a read one byte past the end fails the build.
- **Fuzzing** with libFuzzer (`fuzz/`): one target feeds random text to the JSON parser and the config reader, another feeds random files to the whole loader. CI fuzzes each for a minute on every push and keeps any input that crashes. To fuzz locally:

```sh
CXX=clang++ cmake --preset fuzz     # needs a clang with libFuzzer
cmake --build --preset fuzz
./build/fuzz/ember_fuzz_seeds build/fuzz/seeds
./build/fuzz/ember_fuzz_safetensors -max_total_time=60 build/fuzz/seeds/safetensors
```

On macOS, Apple's clang has no libFuzzer. Homebrew's `llvm` does, but its headers need its own libc++, so add `-DCMAKE_EXE_LINKER_FLAGS="-L$(brew --prefix llvm)/lib/c++ -Wl,-rpath,$(brew --prefix llvm)/lib/c++"` to the configure step.

## Limits

- One `model.safetensors` per model. Models split across several files are refused with a clear error for now.
- Weights must be f32, f16, or bf16.
- If another program shortens the file while Ember has it mapped, reading past the new end kills the process with SIGBUS. That is how memory mapping works everywhere.
