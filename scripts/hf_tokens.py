#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "tokenizers==0.23.3",
# ]
# ///
"""Saves Hugging Face's own token ids for a set of strings, so CI can check Ember's tokenizer.

The strings are picked to hit the hard cases: leading and repeated spaces, newlines, punctuation,
digits, accented letters, scripts the vocabulary barely covers (which fall back to raw bytes), and
emoji. The C++ test tests/tokenizer_test.cpp encodes each one and must get exactly these ids, then
decode them back to exactly the original text.

    ./scripts/fetch-model.sh && uv run scripts/hf_tokens.py

usage: scripts/hf_tokens.py [MODEL_DIR] [OUT_FILE]
"""

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

STRINGS = [
    "Once upon a time, there was a little girl named Lily. She wanted to play with her little dog",
    "Hello world",
    " Hello",
    "  two leading spaces",
    "trailing space ",
    "many     spaces   inside",
    "line one\nline two\n\nline four",
    "tabs\tand\ttabs",
    "Numbers: 0 1 42 3.14159 2026-11-09",
    "Punctuation!?.,;:'\"()[]{}<>@#$%^&*-_=+/\\|~`",
    "naïve café résumé",
    "Straße über Größe",
    "日本語のテキスト",
    "Привет, мир",
    "नमस्ते दुनिया",
    "emoji 🔥🚀 and 👩‍💻",
    "a",
    "",
    "<s> is not special in plain text",
    "unbelievable antidisestablishmentarianism",
]


def main() -> None:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "models/tinyllama-15M")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "tests/reference/hf_tokens_tinyllama-15M.json")
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    cases = []
    for text in STRINGS:
        ids = tok.encode(text, add_special_tokens=False).ids
        decoded = tok.decode(ids, skip_special_tokens=False)
        cases.append({"text": text, "ids": ids, "decoded": decoded})
    doc = {
        "about": "Token ids from Hugging Face tokenizers, made by scripts/hf_tokens.py",
        "model": "nickypro/tinyllama-15M",
        "made_with": {"tokenizers": "0.23.3"},
        "add_special_tokens": False,
        "cases": cases,
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
