# How Ember manages memory

An LLM spends most of its time on a few loops over big arrays. What decides speed is where those arrays live and how often they get allocated. This page explains the choices in `include/ember/`, and what each one gave up.

## The problem

One forward pass makes many tensors: attention scores, head outputs, activations. All of them die when the pass ends, and the next token makes the same ones again. So the pattern is many allocations, all the same lifetime, repeated forever. A general allocator is built for the opposite case, where anything can be freed at any time.

## Arena allocation

`Arena` holds big chunks of memory. Allocating moves a pointer forward. Freeing rewinds that pointer, which frees everything allocated since in one step.

| Choice | Why | What it costs |
|---|---|---|
| Bump pointer instead of `new` per tensor | Allocation is a little arithmetic, and related tensors land next to each other | Nothing can be freed alone, only everything newer than a mark |
| Chunks are never moved | A tensor's pointer stays valid as the arena grows. A single growing buffer would have to move, and break every tensor | Up to one request's worth of space wasted at the end of each chunk |
| Rewind keeps the chunks | Generating many tokens holds no more memory than generating one | Memory is not given back until the arena is destroyed |
| 64 byte alignment by default | That is a cache line, and enough for any SIMD register up to AVX-512 | Up to 63 bytes of padding per allocation |
| One arena per thread, no locks | Locks would cost more than the allocation they protect | Threads cannot share an arena |
| A memory limit | In a browser tab, running out should be an error the page can show, not a killed tab | One more number to choose |

### Is it worth it?

`ember_bench` allocates and frees the same 100,000 blocks of mixed sizes both ways. These are the medians of 100 runs from three separate runs of the program, on an Apple M4 with a release build:

| | Per allocation and free |
|---|---|
| Arena | 3.2 to 3.3 ns |
| System allocator through `AlignedBuffer` | 34.6 to 34.9 ns |

So the arena was about 10.7 times faster here. That is one machine and one allocation pattern, and it times the allocator alone. It says nothing yet about the speed of inference. Run it yourself with `ember_bench`.

## Catching the bugs an arena creates

Arenas have a known weakness: they hide their own bugs from memory tools. A tensor that outlives its scope still points at memory the system considers valid, so AddressSanitizer sees nothing wrong, and the tensor quietly reads whatever gets allocated there next.

So sanitizer builds tell AddressSanitizer what the arena knows:

- Memory that has not been handed out, or has been rewound, is poisoned. Touching it stops the program at that exact line.
- Blocks get a poisoned gap between them, so running off the end of a tensor is caught instead of landing in its neighbor.

Normal builds compile all of this away and add no gap. `tests/arena_test.cpp` has tests that make each of these mistakes on purpose and expect the crash.

`tests/arena_random_test.cpp` then runs long random sequences of allocations and rewinds with fixed seeds, and checks the arena against a simple model of what should be alive.

## Tensors do not own memory

A tensor is a small copyable handle: a pointer, an element type, a shape, and strides. The numbers belong to an arena or to a memory-mapped model file. Slicing and transposing only change the handle, so they cost a few words and no memory. Attention transposes constantly, and copying there would add up.

There are two handle types. `TensorView` can only read and `Tensor` can write. Weights mapped from a file sit on read-only pages, where a stray write kills the process, and handing them out as `TensorView` makes that a compile error.

The cost of not owning memory is that a handle can dangle, and the type system will not stop it. That is what the poisoning above is for.

## Errors

There are two kinds, and they are handled differently on purpose.

- **Bad input** comes back as a `Status` or `StatusOr`, because the caller can do something about it. The engine does not use exceptions.
- **Bugs in our own code**, like reading a float tensor as ints, abort through `EMBER_CHECK` with the file and line. There is nothing sensible to recover to. Checks that run per element, like tensor bounds, are `EMBER_DCHECK` and only exist in debug builds.

## Untrusted sizes

Sizes in a model file cannot be trusted. On 32-bit WebAssembly, a shape of 65536 by 65536 multiplies to exactly zero, so a naive engine would allocate almost nothing and then read far past it. `Shape::Make` is the one place a shape gets built, and it rejects anything whose element count times the largest element size would overflow. Every later size calculation can rely on that, so a `Shape` that exists is safe to multiply.

The native tests run with a 64-bit `size_t` and cannot see this kind of bug. So the WebAssembly build runs its own checks, and `tests/wasm_smoke.mjs` fails if any of them do.
