# How Ember runs a forward pass

The forward pass turns a list of token ids into logits: for each position, a score for every token in the vocabulary that could come next. `Forward` in `include/ember/forward.h` does it for any Llama-architecture model the loader accepts.

```sh
./scripts/fetch-model.sh
./build/release/ember logits models/tinyllama-15M 1 9038 2501 263 931
```

Those ids are the start-of-text token and "Once upon a time". The likeliest next token is 29892, which is a comma.

## One layer

Every layer updates a running sum called the residual stream, one row per token:

```
x = x + Attention(RmsNorm(x))
x = x + Down(Silu(Gate(RmsNorm(x))) * Up(RmsNorm(x)))
```

After the last layer comes one more `RmsNorm` and a `Linear` into the vocabulary. The pieces live in `include/ember/layers.h`:

| Piece | What it does | A detail that matters |
|---|---|---|
| `Embed` | One row of the embedding table per token | An id outside the vocabulary is an error, not a read past the end |
| `RmsNorm` | Divides each row by its root mean square, then scales | Scales first and multiplies by the weight after, in the same order as Hugging Face |
| `Linear` | `x` times a PyTorch-layout weight `[out, in]` | Reads each weight row in a straight line, so it is not `MatMul` with a transposed view |
| `ApplyRope` | Rotates query and key pairs by an angle that grows with position | Pairs element `j` with `j + head_dim / 2`, the layout Hugging Face uses, not neighbors |
| `CausalAttention` | Each token mixes in the values of itself and earlier tokens | Built on `MatMul` and `Softmax` from `ops.h` |
| `Silu` | `x / (1 + e^-x)` | Stays finite at both extremes |

## Attention

For each head, attention computes `softmax(q k^T / sqrt(head_dim)) v`, with the future hidden. The choices:

- **Keys are copied once per key head.** `q k^T` needs the keys turned sideways, and a sideways matrix makes `MatMul` about ten times slower ([docs/ops.md](ops.md)). One copy per key head is reused by every query head that shares it.
- **The future gets negative infinity.** `Softmax` turns it into a chance of exactly zero. Each token can always see itself, so no row is all negative infinity, which `Softmax` would refuse.
- **Grouped-query attention works.** With fewer key heads than query heads, query head `h` uses key head `h / (heads / kv_heads)`, as Hugging Face does.
- **More keys than queries is allowed.** Query `i` sees keys up to `i + keys - queries`. A full forward pass has as many of each, but the KV cache will pass new queries against old keys, and this is already tested.
- **Scratch memory comes from the arena** and goes back when the function returns, so attention allocates nothing per call once the arena has grown.

`Forward` allocates the logits first and everything else inside an `ArenaScope`, so when it returns the arena holds only the logits.

## How it was checked

Four ways, from cheapest to strongest:

1. **Each piece against its definition.** `tests/layers_test.cpp` checks every layer against the formula written out in double precision, plus properties: RoPE at position 0 changes nothing, and a query and key rotated by positions 7 and 3 score the same as at 104 and 100.
2. **The whole pass against a second implementation.** `tests/forward_test.cpp` runs random tiny models (tied and untied, f32 and f16, grouped-query attention, one to eight tokens) through `Forward` and through a separate Llama written with plain loops in double. They must agree within `1e-4`. Changing the sign of one term in RoPE makes three of the four comparisons fail, so the test does catch a real bug. It also checks that adding tokens never changes an earlier row's logits.
3. **The real model against NumPy.** `scripts/reference_logits.py` is a third Llama, written in float32 NumPy from Hugging Face's modeling code. On the TinyStories model it agreed with Ember to within `4.2e-5` on three inputs from 5 to 128 tokens, across all 32,000 logits and with the same top token at every position. The largest logit was about 25. The full run is saved in [docs/results/2026-11-08-apple-m4.md](results/2026-11-08-apple-m4.md), and one input can be checked by hand with:

```sh
./build/release/ember logits models/tinyllama-15M --out /tmp/logits.f32 1 9038 2501 263 931
python3 scripts/reference_logits.py models/tinyllama-15M /tmp/logits.f32 1 9038 2501 263 931
```

All three were written for this project, so they could share a misreading of the architecture. The fourth check was not:

4. **The real model against Hugging Face.** `scripts/hf_reference.py` runs the same model and the same 156 positions through `transformers` itself (float32, eager attention, versions pinned in the script) and saves part of each row in `tests/reference/`: the 10 highest logits, every 500th logit, and the logsumexp, mean, and root mean square of all 32,000. `tests/hf_reference_test.cpp` runs Ember on those tokens and compares within `1e-4`. The largest difference is `3.0e-5` on the release build ([docs/results/2026-11-09-apple-m4.md](results/2026-11-09-apple-m4.md)). CI downloads the model and runs this on GCC for x86, under AddressSanitizer and UBSan, and on Apple clang for ARM, and prints each platform's difference in the log. Doubling RoPE's theta, a 10x larger RMS norm eps, or shifting only the keys by one position each make it fail. (Shifting queries and keys together passes, correctly: RoPE only sees the distance between them.)

```sh
./scripts/fetch-model.sh
EMBER_MODEL_DIR=models/tinyllama-15M ./build/release/ember_tests --gtest_filter='HfReferenceTest.*'
```

The saved file cannot see one unlikely token with a wrong logit that is neither in the top 10 nor on the every-500th grid. The NumPy check compares every logit, so the two together cover it. PyTorch is needed only to remake the file (`uv run scripts/hf_reference.py`), never in CI.

## Not done yet

- There is no KV cache, so generating each new token reprocesses the whole sequence.
- There is no tokenizer, so the CLI takes token ids.
- Everything runs on one thread. `Linear` uses 4-wide SIMD registers through the compiler's vector types, but there are no hand-tuned kernels yet, and NumPy with Apple's BLAS is still 7.5x faster on the same model ([results](results/2026-11-09-apple-m4.md)).
