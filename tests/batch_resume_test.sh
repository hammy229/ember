#!/bin/sh
# Checks that `ember batch --resume` keeps finished answers, drops a line cut off mid-write,
# and answers only the prompts that are left. Needs the tiny model in EMBER_MODEL_DIR; skips
# (exit 77) without it unless EMBER_REQUIRE_MODEL is 1, as CI sets.
set -eu
ember=$1
if [ -z "${EMBER_MODEL_DIR:-}" ]; then
  if [ "${EMBER_REQUIRE_MODEL:-}" = "1" ]; then
    echo "EMBER_REQUIRE_MODEL is 1 but EMBER_MODEL_DIR is not set"
    exit 1
  fi
  echo "set EMBER_MODEL_DIR to models/tinyllama-15M (scripts/fetch-model.sh)"
  exit 77
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
printf '%s\n' '{"id": 0, "prompt": "Once upon a time"}' '{"id": "b", "prompt": "The dog"}' \
  '{"id": 2, "prompt": "She said"}' > "$work/in.jsonl"

"$ember" batch "$EMBER_MODEL_DIR" "$work/in.jsonl" "$work/full.jsonl" --tokens 8 2>/dev/null

# A run stopped after the first answer, halfway through writing the second.
head -n 1 "$work/full.jsonl" > "$work/out.jsonl"
printf '{"id": "b", "text": "cut' >> "$work/out.jsonl"
"$ember" batch "$EMBER_MODEL_DIR" "$work/in.jsonl" "$work/out.jsonl" --tokens 8 --resume \
  2>/dev/null

# Greedy decoding gives the same answers apart from the timing field.
strip() { sed 's/"ms": [^,]*,//' "$1"; }
if [ "$(strip "$work/out.jsonl")" != "$(strip "$work/full.jsonl")" ]; then
  echo "resumed output differs from a full run:"
  cat "$work/out.jsonl"
  exit 1
fi

# Resuming a finished file answers nothing new.
"$ember" batch "$EMBER_MODEL_DIR" "$work/in.jsonl" "$work/out.jsonl" --tokens 8 --resume \
  2>/dev/null
[ "$(wc -l < "$work/out.jsonl")" -eq 3 ] || { echo "a finished file grew"; exit 1; }
echo "resume ok"
