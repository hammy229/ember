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
"""Saves Hugging Face's own logits for Qwen3 0.6B, so Ember can be checked against them.

The same idea as scripts/hf_reference.py, for the first real model: run it through `transformers`
in float32 with eager attention and keep, for every position of each input, the 10 highest logits
with their token ids and the log of the sum of exp over all 151,936 logits. The token ids are saved
too, so the check does not depend on Ember's tokenizer.

    ./scripts/fetch-model.sh qwen3-0.6b && uv run scripts/hf_reference_qwen3.py

tests/hf_reference_test.cpp compares against the file when EMBER_QWEN3_DIR is set.

usage: scripts/hf_reference_qwen3.py [MODEL_DIR] [OUT_FILE]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 10

INPUTS = [
    ("plain sentence", "The capital of France is", False),
    (
        "chat math question",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
        "in May. How many clips did Natalia sell altogether in April and May?",
        True,
    ),
    ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return", False),
]


def main() -> None:
    args = sys.argv[1:]
    model_dir = Path(args[0]) if args else ROOT / "models" / "qwen3-0.6b"
    out_path = Path(args[1]) if len(args) > 1 else ROOT / "tests" / "reference" / "hf_logits_qwen3-0.6b.json"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    records = []
    with torch.no_grad():
        for name, text, chat in INPUTS:
            if chat:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            tokens = tokenizer(text, add_special_tokens=False).input_ids
            logits = model(torch.tensor([tokens])).logits[0].double().numpy()
            positions = []
            for row in logits:
                top = np.argsort(-row, kind="stable")[:TOP_K]
                largest = row.max()
                positions.append(
                    {
                        "top_ids": [int(i) for i in top],
                        "top_logits": [round(float(row[i]), 6) for i in top],
                        "logsumexp": round(float(largest + np.log(np.exp(row - largest).sum())), 6),
                    }
                )
            records.append({"name": name, "text": text, "tokens": tokens, "positions": positions})
    doc = {
        "about": "Logits from Hugging Face transformers, made by scripts/hf_reference_qwen3.py",
        "model": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "vocab_size": int(model.config.vocab_size),
        "top_k": TOP_K,
        "made_with": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "float32",
            "attention": "eager",
        },
        "inputs": records,
    }
    out_path.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {sum(len(r['tokens']) for r in records)} positions to {out_path}")


if __name__ == "__main__":
    main()
