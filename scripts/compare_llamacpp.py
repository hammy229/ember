#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = []
# ///
"""Ember against llama.cpp on the same model, the same 8-bit format, and the same CPU threads.

llama.cpp is the reference for running models on a laptop, tuned for years by many people. This
measures how far behind a from-scratch engine is, and the result is published whichever way it
goes.

- Model: Qwen3 0.6B. llama.cpp runs Qwen's own Q8_0 GGUF (Qwen/Qwen3-0.6B-GGUF 23749fe); Ember
  loads the safetensors with --q8. Both store weights as 8-bit values with one scale per 32.
- CPU only: llama.cpp gets -ngl 0, so it does not use the Mac's GPU, and both get the same threads.
- Generation: tokens a second writing 128 tokens (llama-bench tg128; `ember generate --tokens 128`,
  whose short prompt is included in its time).
- Prompt: tokens a second reading a prompt of about 128 tokens (llama-bench pp; Ember's first
  generation step, which runs the whole prompt and only the last row of the output layer, as
  llama.cpp does).
- Median of 3 runs each.

If another process is using the CPU, --pause-pid freezes it (SIGSTOP) for the measurements and
lets it go on (SIGCONT) afterwards.

    ./scripts/fetch-model.sh qwen3-0.6b && cmake --build --preset release
    uv run scripts/compare_llamacpp.py [--threads 6] [--pause-pid PID]
"""

import argparse
import json
import os
import platform
import re
import signal
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBER = ROOT / "build" / "release" / "ember"
MODEL = ROOT / "models" / "qwen3-0.6b"
GGUF = ROOT / "models" / "qwen3-0.6b-gguf" / "Qwen3-0.6B-Q8_0.gguf"
GGUF_URL = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf"
PARAGRAPH = ("The Roman Empire was the period of ancient Roman civilization after the fall of the "
             "Roman Republic. At its height it ruled the lands around the Mediterranean Sea. ")


def ember(*args):
    return subprocess.run([str(EMBER), *map(str, args)], check=True, capture_output=True, text=True)


def ember_generation(threads):
    rates = []
    for _ in range(3):
        out = ember("generate", MODEL, "--q8", "--threads", threads, "--tokens", 128,
                    "The history of the Roman Empire")
        rates.append(float(re.search(r"\(([0-9.]+) tokens/s", out.stderr).group(1)))
    return statistics.median(rates)


def ember_prompt(threads, prompt):
    rates, count = [], 0
    for _ in range(3):
        out = ember("generate", MODEL, "--q8", "--threads", threads, "--tokens", 1, prompt)
        match = re.search(r"1 tokens in ([0-9.]+) ms .*prompt of (\d+) included", out.stderr)
        ms, count = float(match.group(1)), int(match.group(2))
        rates.append(count / ms * 1000)
    return statistics.median(rates), count


def llama(threads, prompt_tokens):
    out = subprocess.run(
        ["llama-bench", "-m", str(GGUF), "-p", str(prompt_tokens), "-n", "128", "-t", str(threads),
         "-ngl", "0", "-r", "3", "-o", "json"],
        check=True, capture_output=True, text=True,
    )
    rows = json.loads(out.stdout)
    prompt = next(r for r in rows if r["n_prompt"] > 0)
    gen = next(r for r in rows if r["n_gen"] > 0)
    return prompt["avg_ts"], gen["avg_ts"], rows[0].get("build_commit", "?"), rows[0].get("build_number", "?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--pause-pid", type=int, default=0)
    args = parser.parse_args()
    if not GGUF.exists():
        GGUF.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-fL", "-o", str(GGUF), GGUF_URL], check=True)

    prompt = PARAGRAPH * 4
    if args.pause_pid:
        os.kill(args.pause_pid, signal.SIGSTOP)
    try:
        e_gen = ember_generation(args.threads)
        e_prompt, prompt_tokens = ember_prompt(args.threads, prompt)
        l_prompt, l_gen, commit, build = llama(args.threads, prompt_tokens)
    finally:
        if args.pause_pid:
            os.kill(args.pause_pid, signal.SIGCONT)

    report = {
        "model": "Qwen3 0.6B, 8-bit with a scale per 32 weights (Ember --q8, llama.cpp Q8_0)",
        "threads": args.threads,
        "machine": platform.platform(),
        "llama_cpp": {"build": build, "commit": commit, "gpu_layers": 0},
        "prompt_tokens": prompt_tokens,
        "generation_tokens_per_second": {"ember": round(e_gen, 1), "llama.cpp": round(l_gen, 1)},
        "prompt_tokens_per_second": {"ember": round(e_prompt, 1), "llama.cpp": round(l_prompt, 1)},
    }
    out = ROOT / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "llamacpp-qwen3-0.6b.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = [
        "# Ember against llama.cpp",
        "",
        "Made by `scripts/compare_llamacpp.py`. Do not edit by hand.",
        "",
        f"Qwen3 0.6B in 8 bits on both, CPU only, {args.threads} threads, median of 3 runs, on "
        f"{report['machine']}. llama.cpp build {build} ({commit}) with no GPU layers.",
        "",
        "| | Ember | llama.cpp | Ember as a share of llama.cpp |",
        "|---|---|---|---|",
        f"| Writing, tokens a second | {e_gen:.1f} | {l_gen:.1f} | {e_gen / l_gen:.0%} |",
        f"| Reading a {prompt_tokens} token prompt, tokens a second | {e_prompt:.1f} | {l_prompt:.1f} | "
        f"{e_prompt / l_prompt:.0%} |",
        "",
        "Likely reasons, not yet measured one by one: llama.cpp does its 8-bit dot products with "
        "integer SIMD instructions and blocks its matrix multiplies for the cache, while Ember widens "
        "each 8-bit value to f32 and multiplies one row at a time.",
        "",
    ]
    (out / "llamacpp-qwen3-0.6b.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
