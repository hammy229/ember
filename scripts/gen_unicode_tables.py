#!/usr/bin/env -S uv run --script --python 3.14
# /// script
# requires-python = ">=3.14"
# ///
"""Writes src/unicode_tables.inc, the Unicode data the Qwen3 tokenizer needs.

Hugging Face's tokenizers library is Rust, and the crates it uses for regex classes and NFC
normalization carry Unicode 16.0 tables. Python 3.14's unicodedata is also Unicode 16.0, so tables
made from it match. The script refuses to run on any other version.

It writes:
- letter (\\p{L}) and number (\\p{N}) code point ranges, for the pre-tokenizer's regex classes,
- canonical combining classes, full canonical decompositions, and primary composition pairs, for
  NFC.

Before writing, it runs NFC built from these tables over every code point and over random strings,
and stops if any result differs from unicodedata.normalize. The C++ side is a direct port of that
same algorithm.

    uv run scripts/gen_unicode_tables.py
"""

import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "unicode_tables.inc"
MAX = 0x110000
S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT


def ranges(predicate):
    out, start = [], None
    for cp in range(MAX):
        if predicate(cp):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAX - 1))
    return out


def canonical_mapping(cp):
    d = unicodedata.decomposition(chr(cp))
    if not d or d.startswith("<"):
        return None
    return [int(x, 16) for x in d.split()]


def full_decomposition(cp):
    mapping = canonical_mapping(cp)
    if mapping is None:
        return [cp]
    out = []
    for part in mapping:
        out.extend(full_decomposition(part))
    return out


ccc = {cp: unicodedata.combining(chr(cp)) for cp in range(MAX) if unicodedata.combining(chr(cp))}
decompositions = {}
for cp in range(MAX):
    if S_BASE <= cp < S_BASE + S_COUNT:
        continue  # Hangul syllables are done by arithmetic
    if canonical_mapping(cp) is not None:
        decompositions[cp] = full_decomposition(cp)
compositions = {}
for cp in range(MAX):
    mapping = canonical_mapping(cp)
    if mapping is None or len(mapping) != 2 or S_BASE <= cp < S_BASE + S_COUNT:
        continue
    # Composition exclusions and non-starter decompositions never appear in NFC output.
    if unicodedata.normalize("NFC", chr(cp)) != chr(cp):
        continue
    compositions[(mapping[0], mapping[1])] = cp


def decompose(text):
    out = []
    for ch in text:
        cp = ord(ch)
        if S_BASE <= cp < S_BASE + S_COUNT:
            s = cp - S_BASE
            out.append(L_BASE + s // N_COUNT)
            out.append(V_BASE + (s % N_COUNT) // T_COUNT)
            if s % T_COUNT:
                out.append(T_BASE + s % T_COUNT)
        else:
            out.extend(decompositions.get(cp, [cp]))
    # Canonical ordering: a stable sort of each run of non-starters by combining class.
    i = 0
    while i < len(out):
        if ccc.get(out[i], 0) == 0:
            i += 1
            continue
        j = i
        while j < len(out) and ccc.get(out[j], 0) != 0:
            j += 1
        out[i:j] = sorted(out[i:j], key=lambda c: ccc[c])
        i = j
    return out


def pair(first, second):
    """The primary composite of two code points, or None."""
    if L_BASE <= first < L_BASE + L_COUNT and V_BASE <= second < V_BASE + V_COUNT:
        return S_BASE + ((first - L_BASE) * V_COUNT + (second - V_BASE)) * T_COUNT
    if (S_BASE <= first < S_BASE + S_COUNT and (first - S_BASE) % T_COUNT == 0
            and T_BASE < second < T_BASE + T_COUNT):
        return first + (second - T_BASE)
    return compositions.get((first, second))


def compose(cps):
    """Canonical composition from UAX #15: a character joins the last starter unless something in
    between is a starter or has a combining class at least its own."""
    out = []
    starter = None
    last = -1  # combining class of the last character kept after the starter, -1 if none
    for cp in cps:
        c = ccc.get(cp, 0)
        if starter is not None and (last == -1 or (last != 0 and last < c)):
            composite = pair(out[starter], cp)
            if composite is not None:
                out[starter] = composite
                continue
        if c == 0:
            starter = len(out)
            last = -1
        else:
            last = c
        out.append(cp)
    return out


def nfc(text):
    return "".join(chr(c) for c in compose(decompose(text)))


def check():
    for cp in range(MAX):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        if nfc(ch) != unicodedata.normalize("NFC", ch):
            sys.exit(f"NFC mismatch on U+{cp:04X}")
    marks = [cp for cp in ccc][:400] + [0x301, 0x300, 0x308, 0x327, 0x323, 0x302]
    bases = [ord(c) for c in "aeiouAEIOUnNcCsSzZ"] + [0x1100, 0x1161, 0x11A8, 0xAC00, 0x3B1, 0x415]
    rng = random.Random(20261109)
    for _ in range(200000):
        s = "".join(chr(rng.choice(bases if rng.random() < 0.5 else marks)) for _ in range(rng.randint(1, 6)))
        if nfc(s) != unicodedata.normalize("NFC", s):
            sys.exit(f"NFC mismatch on {[hex(ord(c)) for c in s]}")


def emit_ranges(name, items):
    body = ",\n".join(f"    {{0x{a:X}, 0x{b:X}}}" for a, b in items)
    return f"constexpr CodeRange {name}[] = {{\n{body},\n}};\n"


def main():
    if unicodedata.unidata_version != "16.0.0":
        sys.exit(f"need Unicode 16.0.0 to match Hugging Face tokenizers, have {unicodedata.unidata_version}")
    check()
    letters = ranges(lambda cp: unicodedata.category(chr(cp)).startswith("L"))
    numbers = ranges(lambda cp: unicodedata.category(chr(cp)).startswith("N"))
    pool, entries = [], []
    for cp in sorted(decompositions):
        entries.append((cp, len(pool), len(decompositions[cp])))
        pool.extend(decompositions[cp])
    parts = [
        "// Generated by scripts/gen_unicode_tables.py from Unicode 16.0.0. Do not edit by hand.\n",
        emit_ranges("kLetters", letters),
        emit_ranges("kNumbers", numbers),
        "constexpr CombiningClass kCombiningClasses[] = {\n"
        + ",\n".join(f"    {{0x{cp:X}, {c}}}" for cp, c in sorted(ccc.items()))
        + ",\n};\n",
        "constexpr Decomposition kDecompositions[] = {\n"
        + ",\n".join(f"    {{0x{cp:X}, {off}, {n}}}" for cp, off, n in entries)
        + ",\n};\n",
        "constexpr char32_t kDecompositionPool[] = {\n"
        + ",\n".join("    " + ", ".join(f"0x{c:X}" for c in pool[i : i + 8]) for i in range(0, len(pool), 8))
        + ",\n};\n",
        "constexpr Composition kCompositions[] = {\n"
        + ",\n".join(f"    {{0x{a:X}, 0x{b:X}, 0x{c:X}}}" for (a, b), c in sorted(compositions.items()))
        + ",\n};\n",
    ]
    OUT.write_text("".join(parts))
    print(f"wrote {OUT}: {len(letters)} letter ranges, {len(numbers)} number ranges, "
          f"{len(ccc)} combining classes, {len(entries)} decompositions, {len(compositions)} compositions")


if __name__ == "__main__":
    main()
