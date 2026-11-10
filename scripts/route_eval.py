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
    uv run scripts/route_eval.py [--limit N] [--threads N] [--resume] [--analyze-only]
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


# Signals ember route decides on exactly as the full answer would (see include/ember/route.h).
DEPLOYABLE = ("max entropy", "lowest margin")


def number(text):
    text = text.replace(",", "").replace("$", "")
    try:
        value = float(text)
    except ValueError:
        return None
    return value


def extract(answer):
    """The number after the last "answer is", or failing that the last number in the text."""
    match = re.findall(r"answer is[^\d\n\-]{0,12}(-?[\d,]*\.?\d+)", answer, flags=re.IGNORECASE)
    if match:
        return number(match[-1])
    numbers = re.findall(r"-?[\d,]*\.?\d+", answer)
    return number(numbers[-1]) if numbers else None


def auroc(scores, wrong):
    """Chance a wrong answer scores higher than a right one, ties counted half."""
    values = np.asarray(scores, dtype=float)
    w = np.asarray(wrong, dtype=bool)
    n_wrong, n_right = int(w.sum()), int((~w).sum())
    if not n_wrong or not n_right:
        return float("nan")
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]  # 1-based, ties averaged
    return float((ranks[w].sum() - n_wrong * (n_wrong + 1) / 2) / (n_wrong * n_right))


def caught(scores, wrong, share):
    """Share of wrong answers among the most uncertain `share` of questions."""
    n = len(scores)
    k = int(round(share * n))
    order = np.argsort(-np.array(scores), kind="stable")[:k]
    total_wrong = sum(wrong)
    return sum(wrong[i] for i in order) / total_wrong if total_wrong else float("nan")


def bootstrap(fn, n, seed=0, rounds=2000):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        value = fn(idx)
        if not math.isnan(value):
            out.append(value)
    assert len(out) >= 0.99 * rounds, "too many resamples had no wrong or no right answers"
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="first N questions only")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="keep the answers already in build/route and finish the rest")
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    data = WORK / "gsm8k-test.parquet"
    if not data.exists():
        urllib.request.urlretrieve(DATA_URL, data)
    table = pq.read_table(data).to_pylist()
    if args.limit:
        table = table[: args.limit]
    gold = [number(row["answer"].split("####")[-1].strip()) for row in table]
    assert all(g is not None and math.isfinite(g) for g in gold), "unreadable gold answer"

    prompts = WORK / "prompts.jsonl"
    answers = WORK / "answers.jsonl"
    with prompts.open("w") as f:
        for i, row in enumerate(table):
            f.write(json.dumps({"id": i, "prompt": row["question"] + INSTRUCTION}) + "\n")
    if not args.analyze_only:
        subprocess.run(
            [str(ROOT / "build/release/ember"), "batch", str(ROOT / "models/qwen3-0.6b"),
             str(prompts), str(answers), "--chat", "--threads", str(args.threads),
             "--tokens", str(args.tokens)] + (["--resume"] if args.resume else []),
            check=True,
        )
    results = [json.loads(line) for line in answers.open()]
    results.sort(key=lambda r: r["id"])
    if len(results) != len(table):
        sys.exit(f"have {len(results)} answers for {len(table)} questions")

    wrong, rows = [], []
    fallback = truncated = 0
    for r, g in zip(results, gold):
        got = extract(r["text"])
        fallback += not re.search(r"answer is", r["text"], flags=re.IGNORECASE)
        truncated += r["stop"] == 0  # Generation::Stop::kMaxTokens
        is_wrong = got is None or abs(got - g) > 1e-6
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
              "escalate": ESCALATE, "answers_without_answer_is": fallback,
              "answers_cut_off_at_max_tokens": truncated, "signals": {}}
    for name in signals + ["random (baseline)"]:
        scores = random_scores if name == "random (baseline)" else [row[name] for row in rows]
        report["signals"][name] = {
            "auroc": round(auroc(scores, wrong), 3),
            "auroc_95": bootstrap(lambda idx: auroc([scores[i] for i in idx], [wrong[i] for i in idx]), n),
            "errors_caught": round(caught(scores, wrong, ESCALATE), 3),
            "errors_caught_95": bootstrap(lambda idx: caught([scores[i] for i in idx], [wrong[i] for i in idx], ESCALATE), n),
        }

    # Is the best logit signal better than just counting the question's tokens? The best of the
    # logit signals is picked on this same data, so each resample picks again; otherwise the
    # interval would be too kind to the winner.
    logit_signals = [k for k in signals if "baseline" not in k]
    length = np.array([row["question length (baseline)"] for row in rows])
    columns = {k: np.array([row[k] for row in rows]) for k in logit_signals}
    wrong_array = np.array(wrong)
    top = max(logit_signals, key=lambda k: auroc(columns[k], wrong))

    def gap(idx):
        w = wrong_array[idx]
        return max(auroc(columns[k][idx], w) for k in logit_signals) - auroc(length[idx], w)

    report["versus_length"] = {
        "signal": top,
        "candidates": len(logit_signals),
        "auroc_gain": round(auroc(columns[top], wrong) - auroc(length, wrong), 3),
        "auroc_gain_95": bootstrap(gap, n),
    }

    # Held out: the signal and threshold are both chosen on the first half, then used unchanged on
    # the second. GSM8K's test file is not sorted by difficulty, so the split is in file order.
    # Only the two signals that never go down are candidates: `ember route` stops on them
    # mid-answer and still makes the decision scored here, which is not true of the averages.
    half = n // 2
    if half < 2:
        sys.exit("need at least 4 questions for the held-out split")
    first, second = list(range(half)), list(range(half, n))
    best = max(DEPLOYABLE, key=lambda k: auroc(columns[k][first], wrong_array[first]))
    calib = np.sort(columns[best][first])[::-1]
    k = max(1, int(round(ESCALATE * half)))
    # The logged scores have 6 significant digits, and the router compares the unrounded float.
    # Putting the threshold halfway between the k-th score and the next lower one means no score
    # sits on the boundary, so ember route makes the same call as this script.
    lower = calib[calib < calib[k - 1]]
    threshold = float((calib[k - 1] + lower[0]) / 2) if lower.size else float(calib[k - 1])
    scores_2 = columns[best][second]
    wrong_2 = wrong_array[second]

    def held(idx):
        sent = scores_2[idx] >= threshold
        w = wrong_2[idx]
        return {
            "caught": w[sent].sum() / w.sum() if w.sum() else float("nan"),
            "kept_accuracy": 1 - w[~sent].mean() if (~sent).any() else float("nan"),
        }

    point = held(np.arange(len(second)))
    sent = scores_2 >= threshold
    report["held_out"] = {
        "signal": best,
        "threshold": threshold,
        "questions": len(second),
        "calibration_escalated_share": round(float((calib >= threshold).mean()), 3),
        "escalated_share": round(float(sent.mean()), 3),
        "errors_caught": round(float(point["caught"]), 3),
        "errors_caught_95": bootstrap(lambda idx: held(idx)["caught"], len(second)),
        "local_accuracy_all": round(float(1 - wrong_2.mean()), 3),
        "local_accuracy_kept": round(float(point["kept_accuracy"]), 3),
        "local_accuracy_kept_95": bootstrap(lambda idx: held(idx)["kept_accuracy"], len(second)),
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
        "## Better than question length?",
        "",
    ]
    v = report["versus_length"]
    low, high = v["auroc_gain_95"]
    verdict = ("The whole interval is above zero, so the logits find mistakes better than length "
               "alone does." if low > 0 else
               "The whole interval is below zero, so length alone does better here." if high < 0
               else "The interval includes zero, so this run cannot say the logits beat length.")
    lines += [
        f"Longer questions are harder, so question length alone is a real baseline. The best of "
        f"the {v['candidates']} logit signals, {v['signal']}, picked on this same data, differs "
        f"from it by {v['auroc_gain']:+.3f} AUROC (95% CI [{low:+.3f}, {high:+.3f}], paired "
        f"bootstrap that picks the best signal again in every resample). {verdict}",
        "",
        "## Held out",
        "",
        f"The signal ({h['signal']}) and its threshold were picked on the first {n - h['questions']} "
        f"questions to escalate {ESCALATE:.0%} (it escalated {h['calibration_escalated_share']:.1%} "
        f"there), then used unchanged on the other {h['questions']}. There it escalated "
        f"{h['escalated_share']:.1%} and caught {h['errors_caught']:.1%} "
        f"[{h['errors_caught_95'][0]:.1%}, {h['errors_caught_95'][1]:.1%}] of the local model's "
        f"wrong answers, against about {h['escalated_share']:.0%} for escalating at random. "
        f"Accuracy on the {1 - h['escalated_share']:.1%} kept local was "
        f"{h['local_accuracy_kept']:.1%} [{h['local_accuracy_kept_95'][0]:.1%}, "
        f"{h['local_accuracy_kept_95'][1]:.1%}], against {h['local_accuracy_all']:.1%} for keeping "
        "everything local.",
        "",
        "Only max entropy and lowest margin were candidates here, because they never go down as an "
        "answer goes on, so `ember route` can stop the moment one crosses the threshold and still "
        "make the same call as the full answer. The threshold sits halfway between two logged "
        "scores, so rounding in the log cannot flip a question. To run it:",
        "",
        "```sh",
        f"ember route models/qwen3-0.6b --signal \"{h['signal']}\" --threshold {h['threshold']:.9g} "
        "\"your question\"",
        "```",
        "",
        "## Limits",
        "",
        "- No cloud model was called, so this measures whether the signal finds the local model's "
        "mistakes, not what a combined system would score.",
        "- One model, one benchmark, greedy decoding. Short math answers may make uncertainty "
        "easier to read than open-ended chat. GSM8K is public, so it may be in the model's "
        "training data.",
        f"- {report['answers_cut_off_at_max_tokens']} answers hit the {args.tokens} token cap and "
        f"{report['answers_without_answer_is']} never said \"The answer is\"; those were scored on "
        "their last number. Both count toward the local accuracy above.",
        "- f32 weights. With --q8 the entropies shift a little, so this threshold is for f32.",
        f"- Speed: {report['speed']['mean_seconds_per_question']} s per question on average with "
        f"{args.threads} threads, {report['speed']['mean_tokens_per_answer']} tokens per answer.",
        "",
    ]
    (out / "route-gsm8k.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
