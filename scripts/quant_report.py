#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy==2.5.4",
# ]
# ///
"""What do 8-bit weights cost? Compares a model loaded in f32 with the same model loaded with --q8.

Quality, on a few kinds of text, at every position:
- KL divergence from the f32 model's next-token distribution to the 8-bit one's (0 means identical),
- how often both pick the same most likely token,
- how many of the f32 model's top 5 tokens are also in the 8-bit model's top 5.

Cost, unless --skip-speed: tokens per second generating 128 tokens greedily (median of 3 runs) and
the peak memory of the whole process, from /usr/bin/time. Run that part on an otherwise idle
machine, since anything else running changes the speed.

Writes docs/results/quant-<model>.json and .md.

    uv run scripts/quant_report.py qwen3-0.6b [--threads 6] [--skip-speed]
"""

import argparse
import json
import platform
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMBER = ROOT / "build" / "release" / "ember"

TEXTS = {
    "story": "Once upon a time, there was a little girl named Lily. She loved to play outside in the "
    "park with her dog, Max. One day, Lily and Max found a big red ball under a tree. Lily wanted "
    "to keep the ball, but her mom said they should look for its owner.",
    "encyclopedia": "The Roman Empire was the period of ancient Roman civilization after the fall "
    "of the Roman Republic. At its height it ruled the lands around the Mediterranean Sea, and its "
    "capital, Rome, was the largest city in the world.",
    "code": "def merge_sort(items):\n    if len(items) <= 1:\n        return items\n    middle = "
    "len(items) // 2\n    left = merge_sort(items[:middle])\n    right = merge_sort(items[middle:])\n",
    "math": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
    "in May. How many clips did Natalia sell altogether in April and May? First, in May she sold "
    "48 / 2 = 24 clips.",
}


def run(*args):
    return subprocess.run([str(EMBER), *map(str, args)], check=True, capture_output=True, text=True)


def logits(model_dir, ids, q8, threads):
    with tempfile.NamedTemporaryFile(suffix=".f32") as f:
        run("logits", model_dir, "--out", f.name, "--threads", threads, *(["--q8"] if q8 else []), *ids)
        return np.fromfile(f.name, dtype="<f4").reshape(len(ids), -1).astype(np.float64)


def log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def speed(model_dir, q8, threads):
    rates, peaks = [], []
    for _ in range(3):
        out = subprocess.run(
            ["/usr/bin/time", "-l", str(EMBER), "generate", str(model_dir), "--tokens", "128",
             "--threads", str(threads), *(["--q8"] if q8 else []), "The history of the Roman Empire"],
            check=True, capture_output=True, text=True,
        )
        rates.append(float(re.search(r"\(([0-9.]+) tokens/s", out.stderr).group(1)))
        peaks.append(int(re.search(r"(\d+)\s+maximum resident set size", out.stderr).group(1)))
    return {"tokens_per_second": round(statistics.median(rates), 1),
            "peak_memory_mb": round(statistics.median(peaks) / 2**20)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="qwen3-0.6b")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--skip-speed", action="store_true")
    args = parser.parse_args()
    model_dir = ROOT / "models" / args.model

    rows = {}
    all_kl, all_same, all_top5 = [], [], []
    for name, text in TEXTS.items():
        ids = run("tokenize", model_dir, text).stdout.split()
        full = log_softmax(logits(model_dir, ids, False, args.threads))
        small = log_softmax(logits(model_dir, ids, True, args.threads))
        kl = (np.exp(full) * (full - small)).sum(axis=1)
        same = full.argmax(axis=1) == small.argmax(axis=1)
        top_full = np.argsort(-full, axis=1)[:, :5]
        top_small = np.argsort(-small, axis=1)[:, :5]
        top5 = np.array([len(set(a) & set(b)) / 5 for a, b in zip(top_full, top_small)])
        rows[name] = {"positions": len(ids), "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                      "same_pick": float(same.mean()), "top5_overlap": float(top5.mean())}
        all_kl.extend(kl)
        all_same.extend(same)
        all_top5.extend(top5)

    report = {
        "model": args.model,
        "positions": len(all_kl),
        "kl_mean": float(np.mean(all_kl)),
        "kl_max": float(np.max(all_kl)),
        "same_pick": float(np.mean(all_same)),
        "top5_overlap": float(np.mean(all_top5)),
        "texts": rows,
        "machine": platform.platform(),
    }
    if not args.skip_speed:
        report["threads"] = args.threads
        report["f32"] = speed(model_dir, False, args.threads)
        report["q8"] = speed(model_dir, True, args.threads)

    out = ROOT / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"quant-{args.model}.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = [
        f"# 8-bit weights on {args.model}",
        "",
        "Made by `scripts/quant_report.py`. Do not edit by hand.",
        "",
        f"The same model loaded in f32 and with `--q8`, compared at all {report['positions']} "
        "positions of four short texts.",
        "",
        "| Text | Positions | Mean KL | Largest KL | Same top pick | Top-5 overlap |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        lines.append(f"| {name} | {r['positions']} | {r['kl_mean']:.5f} | {r['kl_max']:.4f} | "
                     f"{r['same_pick']:.1%} | {r['top5_overlap']:.1%} |")
    lines.append(f"| all | {report['positions']} | {report['kl_mean']:.5f} | {report['kl_max']:.4f} | "
                 f"{report['same_pick']:.1%} | {report['top5_overlap']:.1%} |")
    if "f32" in report:
        lines += [
            "",
            f"Generating 128 tokens with {args.threads} threads, median of 3 runs, on {report['machine']}:",
            "",
            "| Weights | Tokens a second | Peak memory |",
            "|---|---|---|",
            f"| f32 | {report['f32']['tokens_per_second']} | {report['f32']['peak_memory_mb']} MB |",
            f"| 8-bit | {report['q8']['tokens_per_second']} | {report['q8']['peak_memory_mb']} MB |",
        ]
    lines += [
        "",
        "KL divergence is in nats. The embedding table and the norms stay f32; every matrix used in "
        "the forward pass, including the output layer, is 8-bit with one scale per 32 weights.",
        "",
    ]
    (out / f"quant-{args.model}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
