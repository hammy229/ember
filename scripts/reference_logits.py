#!/usr/bin/env python3
"""Checks Ember's logits against a second Llama written in NumPy.

Ember's logits for a token sequence come from `ember logits MODEL_DIR --out FILE TOKENS...`. This
script runs the same tokens through its own Llama, written from the Hugging Face modeling code in
float32 NumPy, and prints the largest difference. It reads safetensors by hand, so it needs only
NumPy, not PyTorch.

This is a cross-check between two implementations written by the same project. The independent
check against Hugging Face's own transformers is scripts/hf_reference.py, which saves its logits,
and tests/hf_reference_test.cpp, which CI runs against them. That one keeps a sample of each row,
while this one compares every logit, so each covers the other's blind spot.

With --time, it also times its own forward pass, as a baseline for Ember's speed. NumPy hands
matrix products to the system's BLAS library, which is tuned by hand and may use several threads,
so set VECLIB_MAXIMUM_THREADS=1 (macOS) or OPENBLAS_NUM_THREADS=1 (Linux) for a one thread
comparison.

usage: scripts/reference_logits.py [--time] MODEL_DIR LOGITS_FILE TOKEN_ID...
"""

import json
import statistics
import struct
import sys
import time

import numpy as np

DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": None}


def load_safetensors(path):
    with open(path, "rb") as f:
        data = f.read()
    (length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + length])
    body = memoryview(data)[8 + length :]
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        raw = body[begin:end]
        if info["dtype"] == "BF16":
            bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            array = bits.view(np.float32)
        else:
            array = np.frombuffer(raw, dtype=np.dtype(DTYPES[info["dtype"]]).newbyteorder("<"))
        tensors[name] = array.astype(np.float32).reshape(info["shape"])
    return tensors


def rms_norm(x, weight, eps):
    variance = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return weight * (x / np.sqrt(variance + np.float32(eps)))


def rope(x, theta):
    # x is [tokens, heads, head_dim]; Hugging Face rotates j with j + head_dim / 2.
    tokens, _, dim = x.shape
    inv_freq = 1.0 / (np.float32(theta) ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)))
    angles = np.outer(np.arange(tokens, dtype=np.float32), inv_freq).astype(np.float32)
    angles = np.concatenate([angles, angles], axis=-1)[:, None, :]
    half = dim // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * np.cos(angles) + rotated * np.sin(angles)


def forward(config, w, tokens):
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads") or heads
    head_dim = config.get("head_dim") or hidden // heads
    eps = config.get("rms_norm_eps", 1e-6)
    theta = config.get("rope_theta", 10000.0)
    n = len(tokens)

    x = w["model.embed_tokens.weight"][tokens]
    mask = np.triu(np.full((n, n), -np.inf, dtype=np.float32), k=1)
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = rms_norm(x, w[p + "input_layernorm.weight"], eps)
        q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(n, heads, head_dim)
        k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(n, kv_heads, head_dim)
        v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(n, kv_heads, head_dim)
        q, k = rope(q, theta), rope(k, theta)
        group = heads // kv_heads
        k = np.repeat(k, group, axis=1)
        v = np.repeat(v, group, axis=1)
        scores = np.einsum("qhd,khd->hqk", q, k) / np.float32(np.sqrt(head_dim)) + mask
        scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        scores /= scores.sum(axis=-1, keepdims=True)
        attended = np.einsum("hqk,khd->qhd", scores, v).reshape(n, heads * head_dim)
        x = x + attended @ w[p + "self_attn.o_proj.weight"].T
        h = rms_norm(x, w[p + "post_attention_layernorm.weight"], eps)
        gate = h @ w[p + "mlp.gate_proj.weight"].T
        up = h @ w[p + "mlp.up_proj.weight"].T
        x = x + ((gate / (1 + np.exp(-gate))) * up) @ w[p + "mlp.down_proj.weight"].T
    x = rms_norm(x, w["model.norm.weight"], eps)
    head = w["model.embed_tokens.weight"] if config.get("tie_word_embeddings") else w["lm_head.weight"]
    return (x @ head.T).astype(np.float32)


def main():
    args = sys.argv[1:]
    timed = bool(args) and args[0] == "--time"
    if timed:
        args = args[1:]
    if len(args) < 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    model_dir, logits_path = args[0], args[1]
    tokens = [int(t) for t in args[2:]]
    with open(f"{model_dir}/config.json") as f:
        config = json.load(f)
    weights = load_safetensors(f"{model_dir}/model.safetensors")
    expected = forward(config, weights, tokens)
    if timed:
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            forward(config, weights, tokens)
            samples.append((time.perf_counter() - start) * 1000)
        print(f"numpy forward pass median {statistics.median(samples):.2f} ms over 5 runs")
    actual = np.fromfile(logits_path, dtype="<f4").reshape(expected.shape)

    vocab = {}
    try:
        with open(f"{model_dir}/tokenizer.json") as f:
            vocab = {i: piece for piece, i in json.load(f)["model"]["vocab"].items()}
    except (OSError, KeyError):
        pass

    worst = float(np.max(np.abs(actual - expected)))
    scale = float(np.max(np.abs(expected)))
    same_top = int(np.sum(actual.argmax(axis=-1) == expected.argmax(axis=-1)))
    print(f"tokens {len(tokens)}, vocab {expected.shape[1]}")
    print(f"largest logit difference {worst:.3g} (largest logit {scale:.3g})")
    print(f"same top token at {same_top} of {len(tokens)} positions")
    top = int(expected[-1].argmax())
    print(f"next token after the last one: {top} {vocab.get(top, '')!r}")
    return 0 if worst < 1e-3 * max(1.0, scale) and same_top == len(tokens) else 1


if __name__ == "__main__":
    sys.exit(main())
