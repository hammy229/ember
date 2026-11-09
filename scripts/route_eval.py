#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy==2.5.4",
#   "pyarrow==22.0.0",
# ]
# ///
"""Does the local model know when it is wrong? Measures confidence-gated routing on GSM8K.

A local-first router answers on the user's own hardware and sends a request to the cloud only when
it must. The hard part is deciding when. Ember owns every logit, so while it writes an answer it
can track how unsure the model is (entropy and top-2 margin per token, src/sampling.cpp). This
script checks whether that signal finds the answers the local model gets wrong.

1. Downloads the GSM8K test set at a pinned revision into build/route/.
2. Runs every question through Qwen3 0.6B with `ember batch` (greedy, chat format, thinking off).
3. Marks each answer right or wrong against the answer key.
4. Scores each answer's uncertainty several ways, and compares them with two baselines a router
   could use without logits: random choice, and the length of the question.

The main numbers are:
- AUROC: the chance that a wrong answer is scored more uncertain than a right one (0.5 is random).
- Errors caught: escalate the most uncertain X% of questions, and count what share of the local
  model's wrong answers that sends to the cloud. Random escalation catches X%.
- A held-out check: the threshold is picked on the first half of the questions for a 30% escalation
  rate, then applied unchanged to the second half.

No cloud model is called, so nothing here claims what a cloud model would have answered. Writes
docs/results/route-gsm8k.json and .md.

    ./scripts/fetch-model.sh qwen3-0.6b && cmake --build --preset release
    uv run scripts/route_eval.py [--limit N] [--threads N]
"""

import argparse
import json
import math
import random
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "build" / "route"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
DATA_URL = f"https://huggingface.co/datasets/openai/gsm8k/resolve/{REVISION}/main/test-00000-of-00001.parquet"
INSTRUCTION = "\nSolve it step by step, then end with: The answer is <number>."
ESCALATE = 0.30
EARLY_TOKENS = 32


def number(text):
    text = text.replace(",", "").replace("$", "")
    try:
        value = float(text)
    except ValueError:
        return None
    return value


def extract(answer):
    match = re.findall(r"answer is[^0-9\-]*(-?[\d,]*\.?\d+)", answer, flags=re.IGNORECASE)
    if match:
        return number(match[-1])
    numbers = re.findall(r"-?[\d,]*\.?\d+", answer)
    return number(numbers[-1]) if numbers else None


def auroc(scores, wrong):
    """Chance a wrong answer scores higher than a right one, ties counted half."""
    pos = [s for s, w in zip(scores, wrong) if w]
    neg = [s for s, w in zip(scores, wrong) if not w]
    if not pos or not neg:
        return float("nan")
    ranks = np.argsort(np.argsort(np.array(pos + neg), kind="stable"), kind="stable").astype(float)
    # Average ranks for ties.
    values = np.array(pos + neg)
    for v in np.unique(values):
        idx = values == v
        ranks[idx] = ranks[idx].mean()
    rank_sum = ranks[: len(pos)].sum() + len(pos)  # 1-based
    return float((rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def caught(scores, wrong, share):
    """Share of wrong answers among the most uncertain `share` of questions."""
    n = len(scores)
    k = int(round(share * n))
    order = np.argsort(-np.array(scores), kind="stable")[:k]
    total_wrong = sum(wrong)
    return sum(wrong[i] for i in order) / total_wrong if total_wrong else float("nan")


def bootstrap(fn, n, seed=0, rounds=1000):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        value = fn(idx)
        if not math.isnan(value):
            out.append(value)
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="first N questions only")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    data = WORK / "gsm8k-test.parquet"
    if not data.exists():
        urllib.request.urlretrieve(DATA_URL, data)
    table = pq.read_table(data).to_pylist()
    if args.limit:
        table = table[: args.limit]
    gold = [number(row["answer"].split("####")[-1].strip()) for row in table]

    prompts = WORK / "prompts.jsonl"
    answers = WORK / "answers.jsonl"
    with prompts.open("w") as f:
        for i, row in enumerate(table):
            f.write(json.dumps({"id": i, "prompt": row["question"] + INSTRUCTION}) + "\n")
    if not args.analyze_only:
        subprocess.run(
            [str(ROOT / "build/release/ember"), "batch", str(ROOT / "models/qwen3-0.6b"),
             str(prompts), str(answers), "--chat", "--threads", str(args.threads),
             "--tokens", str(args.tokens)],
            check=True,
        )
    results = [json.loads(line) for line in answers.open()]
    results.sort(key=lambda r: r["id"])
    if len(results) != len(table):
        sys.exit(f"have {len(results)} answers for {len(table)} questions")

    wrong, rows = [], []
    for r, g in zip(results, gold):
        got = extract(r["text"])
        is_wrong = got is None or g is None or abs(got - g) > 1e-6
        wrong.append(is_wrong)
        entropy = np.array([t[1] for t in r["tokens"]] or [0.0])
        margin = np.array([t[2] for t in r["tokens"]] or [1.0])
        rows.append({
            "mean entropy": float(entropy.mean()),
            "max entropy": float(entropy.max()),
            "lowest margin": float(-margin.min()),
            "share of tokens with margin under 0.5": float((margin < 0.5).mean()),
            f"mean entropy, first {EARLY_TOKENS} tokens only": float(entropy[:EARLY_TOKENS].mean()),
            "question length (baseline)": float(r["prompt_tokens"]),
            "ms": r["ms"],
            "tokens": len(r["tokens"]),
        })
    n = len(rows)
    rng = random.Random(0)
    random_scores = [rng.random() for _ in range(n)]
    signals = [k for k in rows[0] if k not in ("ms", "tokens")]
    report = {"questions": n, "local_accuracy": round(1 - sum(wrong) / n, 4),
              "escalate": ESCALATE, "signals": {}}
    for name in signals + ["random (baseline)"]:
        scores = random_scores if name == "random (baseline)" else [row[name] for row in rows]
        report["signals"][name] = {
            "auroc": round(auroc(scores, wrong), 3),
            "auroc_95": bootstrap(lambda idx: auroc([scores[i] for i in idx], [wrong[i] for i in idx]), n),
            "errors_caught": round(caught(scores, wrong, ESCALATE), 3),
            "errors_caught_95": bootstrap(lambda idx: caught([scores[i] for i in idx], [wrong[i] for i in idx], ESCALATE), n),
        }

    # Held out: pick the threshold on the first half, apply it to the second.
    best = max(signals[:-1], key=lambda k: report["signals"][k]["auroc"])
    half = n // 2
    calib = sorted((rows[i][best] for i in range(half)), reverse=True)
    threshold = calib[int(round(ESCALATE * half)) - 1]
    test = range(half, n)
    sent = [i for i in test if rows[i][best] >= threshold]
    test_wrong = [i for i in test if wrong[i]]
    kept = [i for i in test if i not in set(sent)]
    report["held_out"] = {
        "signal": best,
        "threshold": round(threshold, 4),
        "questions": len(test),
        "escalated_share": round(len(sent) / len(test), 3),
        "errors_caught": round(sum(wrong[i] for i in sent) / len(test_wrong), 3) if test_wrong else None,
        "local_accuracy_all": round(1 - len(test_wrong) / len(test), 3),
        "local_accuracy_kept": round(1 - sum(wrong[i] for i in kept) / len(kept), 3) if kept else None,
    }
    report["speed"] = {
        "threads": args.threads,
        "mean_seconds_per_question": round(float(np.mean([r["ms"] for r in rows])) / 1000, 2),
        "mean_tokens_per_answer": round(float(np.mean([r["tokens"] for r in rows])), 1),
    }
    report["setup"] = {
        "model": "Qwen/Qwen3-0.6B c1899de, f32 weights, greedy, chat format with thinking off",
        "data": f"openai/gsm8k test, revision {REVISION}",
        "instruction": INSTRUCTION.strip(),
        "max_new_tokens": args.tokens,
    }

    out = ROOT / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "route-gsm8k.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = [
        "# Does the local model know when it is wrong?",
        "",
        "Made by `scripts/route_eval.py`. Do not edit by hand.",
        "",
        f"Qwen3 0.6B running in Ember answered {n} GSM8K test questions and got "
        f"{report['local_accuracy']:.1%} right. For each answer, Ember scored how unsure the model "
        "was from its own logits. A good signal scores the wrong answers as more unsure.",
        "",
        f"| Signal | AUROC (95% CI) | Wrong answers caught by escalating {ESCALATE:.0%} (95% CI) |",
        "|---|---|---|",
    ]
    for name, s in report["signals"].items():
        lines.append(f"| {name} | {s['auroc']:.3f} [{s['auroc_95'][0]:.3f}, {s['auroc_95'][1]:.3f}] | "
                     f"{s['errors_caught']:.1%} [{s['errors_caught_95'][0]:.1%}, {s['errors_caught_95'][1]:.1%}] |")
    h = report["held_out"]
    lines += [
        "",
        "AUROC is the chance that a wrong answer scores as more unsure than a right one; 0.5 is no "
        f"better than chance. Escalating {ESCALATE:.0%} at random catches about {ESCALATE:.0%} of "
        "the wrong answers.",
        "",
        "## Held out",
        "",
        f"The threshold on {h['signal']} was set on the first half of the questions to escalate "
        f"{ESCALATE:.0%}, then used unchanged on the other {h['questions']}. It escalated "
        f"{h['escalated_share']:.1%} of them and caught {h['errors_caught']:.1%} of the local "
        f"model's wrong answers. Accuracy on the questions kept local went from "
        f"{h['local_accuracy_all']:.1%} (keeping all) to {h['local_accuracy_kept']:.1%}.",
        "",
        "## Limits",
        "",
        "- No cloud model was called, so this measures whether the signal finds the local model's "
        "mistakes, not what a combined system would score.",
        "- One model, one benchmark, greedy decoding. Short math answers may make uncertainty "
        "easier to read than open-ended chat.",
        f"- Speed: {report['speed']['mean_seconds_per_question']} s per question on average with "
        f"{args.threads} threads, {report['speed']['mean_tokens_per_answer']} tokens per answer.",
        "",
    ]
    (out / "route-gsm8k.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
