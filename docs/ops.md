# How Ember does math

A forward pass is mostly a few operations in a loop. This page explains the choices in `include/ember/ops.h`, what they cost, and what was measured. All numbers below come from [one run of `scripts/bench.sh`](benchmarks/2026-11-06-m4.txt) on an Apple M4, on one thread, using the fastest of 10 runs. Rerun the script for numbers on your machine.

## What exists

| Op | What it does | Can write over an input? |
|---|---|---|
| `Add`, `Multiply` | Element by element | Yes |
| `Scale` | Every element times one number | Yes |
| `MatMul` | Matrix product, with leading dimensions as a batch | No |
| `Softmax` | Scores to chances along any dimension | Yes |

All of them take f32 only and follow strides, so a slice or a transpose works without a copy. There is no broadcasting: mismatched shapes are an error, because guessing what was meant hides bugs.

## Loop order in MatMul

The textbook loops go row, column, middle. `MatMul` goes row, middle, column. Both add the same numbers in the same order, so the answers match apart from rounding, and `ember_bench_ops` checks that they do before it times anything.

The difference is the inner loop. The textbook version jumps a whole row of `b` on every step and keeps a running total, which the compiler may not turn into SIMD without permission to change the rounding. `MatMul` adds one row to another in a straight line, which it can.

| Size | Textbook loops | `MatMul` | Faster by |
|---|---|---|---|
| 64x64 | 0.0857 ms | 0.0179 ms | 4.8x |
| 256x256 | 9.4875 ms | 1.0364 ms | 9.2x |
| 1024x1024 | 828.4239 ms | 65.1702 ms | 12.7x |

The gap grows with size, which fits the cache getting more important, but this benchmark cannot separate the cache from SIMD. There is no hand-written SIMD yet, only what the compiler does at `-O3`.

## A transposed input is the slow case

Attention multiplies queries by transposed keys, so this is not a corner case. A transposed `b` gives the right answer, but its inner loop has to jump through memory again, and it runs about as slowly as the textbook loops. One copy first fixes that, because the copy costs size² and the product costs size³.

| Size | Transposed `b` | Copied first | Faster by | Copied first, compared to a plain `MatMul` |
|---|---|---|---|---|
| 64x64 | 0.0725 ms | 0.0320 ms | 2.3x | 1.8x slower |
| 256x256 | 9.7449 ms | 1.2607 ms | 7.7x | 1.2x slower |
| 1024x1024 | 810.2957 ms | 69.0571 ms | 11.7x | 1.1x slower |

The copy won at every size measured, so the attention code should use `Tensor::CopyOf` on its transposed keys. This also says a stride is not free. `Softmax` shows the same thing at 1024x1024, taking 1.8497 ms over each row and 8.7200 ms over each column.

## Overlapping memory

An op that writes while it reads can silently give a wrong answer if the two share memory, so the overlap is checked on every call.

| Case | Result | Why |
|---|---|---|
| `out` is apart from every input | Works | Nothing to go wrong |
| `Add`, `Multiply`, `Scale`, or `Softmax` with `out` exactly on an input | Works | Each element is read before it is overwritten |
| Those ops with `out` partly on an input, like a slice shifted by one | Refused | It would read numbers it had already overwritten |
| A square matrix added into its own transpose | Refused | Same memory and same first element, but not the same order |
| `MatMul` with `out` touching an input at all | Refused | Each input element is read many times, so even an exact match is wrong |

The check compares memory ranges, not elements, so two columns of one matrix count as overlapping even though they share no element. It is the same check `Tensor::CopyFrom` uses, with the same trade: being too careful costs a copy, and being wrong costs a silent bug.

## Softmax

| Choice | Why | What it costs |
|---|---|---|
| Subtract each row's largest score first | The answer does not change, and e^1000 no longer overflows into NaN. The largest term becomes exactly 1, so the total cannot be zero either | One more pass over the row |
| Sum in a double | Adding a million tiny values in a float loses a visible share of the total to rounding | Converting each value |
| Negative infinity gets exactly zero | This is how attention hides a token | Nothing |
| NaN, positive infinity, and fully hidden rows are errors | A broken model should stop here, not sample from garbage | A check per element, and rows before the bad one are already written when it fails |

At 0.57 billion elements a second over rows, `Softmax` is about 18 times slower per element than `Add` at 1024x1024, and it calls `std::exp` once per element. Softmax has not been profiled yet, so that call is a place to look and not a proven cause.

## What the numbers do not say

- They come from one machine and one compiler, and they are native. The site times the WebAssembly build in the visitor's own browser, but no such run is saved here.
- The inputs are random dense matrices, not real attention workloads. There is no model to run yet.
- `add` at 64x64 and 256x256 fits in the fastest caches, so those rows do not predict speed on large tensors.
- They set a starting point for NEON SIMD on Nov 13, not a claim about being fast. No comparison to any other engine exists yet.
