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
    ./scripts/fetch-model.sh qwen3-0.6b && uv run scripts/hf_tokens.py qwen3-0.6b

usage: scripts/hf_tokens.py [tinyllama-15M | qwen3-0.6b]
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

# More for byte-level tokenizers like Qwen3's, aimed at each branch of its split regex and at NFC.
BYTE_LEVEL_STRINGS = [
    "I'm sure they'll say it's fine, we've DONE it, YOU'RE right, she'd go, I'M",
    "ſhe's ſ'S long s",
    "12345 + 678 = 13023, pi is 3.14159",
    "x   y\t\tz   ",
    "end with spaces   ",
    "a\r\nb\n\n\nc\r\n  \n d",
    "   leading spaces then words",
    " no-break　ideographic em space",
    "é decomposed, é composed, 각 jamo, Å",
    "!!! ??? ... --- `` ;; @@ '' \"\"",
    " !hello ?world",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "text<|endoftext|>more text<|im_end|>",
    "def f(x):\n    return x ** 2  # square\n\nprint(f(3))",
    "مرحبا שלום สวัสดี",
    "\U0001F600\U0001F680 mixed\U0001F44D\U0001F3FD",
]


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "tinyllama-15M"
    model_dir = Path("models") / name
    out = Path(f"tests/reference/hf_tokens_{name}.json")
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    byte_level = tok.pre_tokenizer is not None
    cases = []
    for text in STRINGS + (BYTE_LEVEL_STRINGS if byte_level else []):
        ids = tok.encode(text, add_special_tokens=False).ids
        decoded = tok.decode(ids, skip_special_tokens=False)
        cases.append({"text": text, "ids": ids, "decoded": decoded})
    doc = {
        "about": "Token ids from Hugging Face tokenizers, made by scripts/hf_tokens.py",
        "model": name,
        "made_with": {"tokenizers": "0.23.3"},
        "add_special_tokens": False,
        "cases": cases,
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
