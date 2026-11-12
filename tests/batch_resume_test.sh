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
fail() {
  echo "$1"
  cat "$work/err" 2>/dev/null || true
  exit 1
}
batch() {
  out=$1
  shift
  "$ember" batch "$EMBER_MODEL_DIR" "$work/in.jsonl" "$work/$out" --tokens 8 "$@" 2>"$work/err"
}
# Greedy decoding gives the same answers apart from the timing field.
strip() { sed 's/"ms": [^,]*,//' "$1"; }
same() { [ "$(strip "$work/$1")" = "$(strip "$work/full.jsonl")" ]; }

# Id 2 appears twice and is answered once.
printf '%s\n' '{"id": 0, "prompt": "Once upon a time"}' '{"id": "b", "prompt": "The dog"}' \
  '{"id": 2, "prompt": "She said"}' '{"id": 2, "prompt": "She said"}' > "$work/in.jsonl"
batch full.jsonl || fail "the full run failed"
[ "$(wc -l < "$work/full.jsonl")" -eq 3 ] || fail "a repeated id was answered twice"

# A run stopped after the first answer, halfway through writing the second.
head -n 1 "$work/full.jsonl" > "$work/cut.jsonl"
printf '{"id": "b", "text": "cut' >> "$work/cut.jsonl"
batch cut.jsonl --resume || fail "resuming after a cut line failed"
same cut.jsonl || fail "resuming after a cut line differs from a full run"

# Stopped right before the newline: the answer is whole and kept.
head -n 2 "$work/full.jsonl" > "$work/whole.jsonl"
printf '%s' "$(tail -n 1 "$work/full.jsonl")" >> "$work/whole.jsonl"
batch whole.jsonl --resume || fail "resuming after a missing newline failed"
same whole.jsonl || fail "a whole last line without its newline was lost or repeated"

# Resuming a finished file changes nothing.
batch cut.jsonl --resume || fail "resuming a finished file failed"
same cut.jsonl || fail "resuming a finished file changed it"

# A bad line in the middle is not something stopping a run can cause, so it is refused.
{ echo 'not json'; cat "$work/full.jsonl"; } > "$work/bad.jsonl"
cp "$work/bad.jsonl" "$work/bad-before.jsonl"
if batch bad.jsonl --resume; then fail "a bad middle line was accepted"; fi
grep -q "line 1 is not a finished answer" "$work/err" || fail "unexpected error"
cmp -s "$work/bad.jsonl" "$work/bad-before.jsonl" || fail "a refused file was changed"
# The output cannot be the input, which writing would erase.
cp "$work/in.jsonl" "$work/in-before.jsonl"
if "$ember" batch "$EMBER_MODEL_DIR" "$work/in.jsonl" "$work/in.jsonl" 2>"$work/err"; then
  fail "writing over the input was accepted"
fi
cmp -s "$work/in.jsonl" "$work/in-before.jsonl" || fail "the input was changed"

# A prompt too long for the context is skipped, the rest are answered, and the exit code says so.
long=$(awk 'BEGIN { for (i = 0; i < 400; i++) printf "the dog ran. " }')
printf '%s\n' "{\"id\": \"long\", \"prompt\": \"$long\"}" '{"id": 0, "prompt": "Once upon a time"}' \
  > "$work/in.jsonl"
if batch skip.jsonl; then fail "a skipped prompt still exited 0"; fi
grep -q 'skipped id "long"' "$work/err" || fail "the skipped prompt was not named"
[ "$(wc -l < "$work/skip.jsonl")" -eq 1 ] || fail "the prompt after a bad one was not answered"
echo "resume ok"
