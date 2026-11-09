#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy==2.5.4",
#   "tokenizers==0.23.3",
#   "torch==2.15.0",
#   "transformers==5.22.2",
# ]
# ///
"""Saves Hugging Face's own logits for the test model, so CI can check Ember against them.

This is Milestone 1 (docs/PLAN.md): the independent check. Ember, its double precision test
reference, and scripts/reference_logits.py were all written by this project, so they could share a
misreading of the architecture. This script runs the model through `transformers` itself, in float32
with eager attention, and writes what it got to tests/reference/hf_logits_tinyllama-15M.json. The C++
test tests/hf_reference_test.cpp runs Ember on the same tokens and compares.

A full row of logits is 32,000 floats, so the file keeps, for each position:

- the 10 highest logits and their token ids, which is what sampling cares about,
- every 500th logit (64 of them), spread across the whole vocabulary,
- the log of the sum of exp over all 32,000 logits, which moves if any single logit is wrong.

Run it with uv, which installs the pinned versions above into a cache the first time:

    ./scripts/fetch-model.sh && uv run scripts/hf_reference.py

PyTorch is only needed to make the file. CI reads the saved file, so it never installs PyTorch.

usage: scripts/hf_reference.py [MODEL_DIR] [OUT_FILE]
"""

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import tokenizers
import torch
import transformers
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 10
STRIDE = 500

# The same inputs scripts/results.sh uses, so the NumPy check and this one cover the same ground.
SENTENCE = "Once upon a time, there was a little girl named Lily. She wanted to play with her little dog"
INPUTS = [
    ("once upon a time", [1, 9038, 2501, 263, 931]),
    (
        "sentence",
        [1, 9038, 2501, 263, 931, 29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296,
         5131, 304, 1708, 411, 902, 2217, 11203],
    ),
    ("128 tokens across the vocabulary", [(i * 7919 + 1) % 32000 for i in range(128)]),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded(values):
    # Nine significant digits round-trips a float32 exactly.
    return [float(f"{float(v):.9g}") for v in values]


def main():
    args = sys.argv[1:]
    if len(args) > 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    model_dir = Path(args[0]) if args else ROOT / "models" / "tinyllama-15M"
    out_path = Path(args[1]) if len(args) > 1 else ROOT / "tests" / "reference" / "hf_logits_tinyllama-15M.json"

    # The hardcoded sentence ids must be what the model's own tokenizer makes of the text, so the
    # tokenizer that comes next can be checked against this same pair.
    tokenizer = tokenizers.Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    encoded = tokenizer.encode(SENTENCE).ids
    if encoded != INPUTS[1][1]:
        print(f"error: the tokenizer gives {encoded} for the sentence", file=sys.stderr)
        return 1

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    records = []
    with torch.no_grad():
        for name, tokens in INPUTS:
            logits = model(torch.tensor([tokens])).logits[0].to(torch.float64).numpy()
            positions = []
            for row in logits:
                top = np.argsort(-row, kind="stable")[:TOP_K]
                peak = row.max()
                positions.append({
                    "top_ids": [int(i) for i in top],
                    "top_logits": rounded(row[top]),
                    "sampled": rounded(row[::STRIDE]),
                    "logsumexp": float(f"{peak + math.log(np.exp(row - peak).sum()):.12g}"),
                })
            records.append({"name": name, "tokens": tokens, "positions": positions})

    reference = {
        "about": "Logits from Hugging Face transformers, made by scripts/hf_reference.py",
        "model": "nickypro/tinyllama-15M",
        "model_sha256": sha256(model_dir / "model.safetensors"),
        "vocab_size": int(model.config.vocab_size),
        "top_k": TOP_K,
        "stride": STRIDE,
        "sentence": SENTENCE,
        "made_with": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tokenizers": tokenizers.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "float32",
            "attention": "eager",
        },
        "inputs": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(reference, f, separators=(",", ":"))
        f.write("\n")
    total = sum(len(r["tokens"]) for r in records)
    print(f"wrote {out_path.relative_to(ROOT)}: {len(records)} inputs, {total} positions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
