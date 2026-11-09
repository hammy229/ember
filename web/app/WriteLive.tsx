"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { loadEngine, type Engine, type PickedToken } from "./engine";

// The same pinned files scripts/fetch-model.sh downloads, checked against the same hashes, so the
// model in this tab is byte for byte the one CI tests against Hugging Face.
const REVISION = "a97e01fa7d54c088df9dc1b53d581a863ce08cd6";
const BASE = `https://huggingface.co/nickypro/tinyllama-15M/resolve/${REVISION}`;
const FILES = [
  { name: "config.json", sha256: "894c283ab2d3373b3dd4b88a1a868f4292c5cab5f0e75976fbdf993eccb24766" },
  {
    name: "model.safetensors",
    sha256: "b9e8d4614e24c89e99502000294af0aab73e9266029357377578e0a504b7f8d9",
  },
  { name: "tokenizer.json", sha256: "8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1" },
] as const;
const MODEL_MB = 32;

// How long to compute before letting the page draw, so typing and scrolling stay smooth.
const FRAME_BUDGET_MS = 12;

// Entropy in nats at which a token gets each shade. Most picks in a story sit well under 1.
const SHADES = [0.5, 1.25, 2.25];

type Phase =
  | { kind: "idle" }
  | { kind: "downloading"; received: number; total: number }
  | { kind: "ready" }
  | { kind: "writing" }
  | { kind: "failed"; message: string };

type Written = { text: string; tokens: PickedToken[] };

async function download(onProgress: (received: number) => void): Promise<Uint8Array[]> {
  let received = 0;
  return Promise.all(
    FILES.map(async (file) => {
      const response = await fetch(`${BASE}/${file.name}`);
      if (!response.ok || !response.body) {
        throw new Error(`could not download ${file.name} (HTTP ${response.status})`);
      }
      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let length = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        length += value.length;
        received += value.length;
        onProgress(received);
      }
      const bytes = new Uint8Array(length);
      let offset = 0;
      for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.length;
      }
      const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
      const hex = Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
      if (hex !== file.sha256) throw new Error(`${file.name} does not match its pinned hash`);
      return bytes;
    }),
  );
}

function shade(entropy: number): number {
  return SHADES.filter((limit) => entropy >= limit).length;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// Splits the full text into the prompt and one piece per token, by the byte offsets the engine
// reported.
function pieces(written: Written): { prompt: string; tokens: string[] } {
  const bytes = encoder.encode(written.text);
  const starts = written.tokens.map((t) => Math.min(t.start, bytes.length));
  const prompt = decoder.decode(bytes.subarray(0, starts[0] ?? bytes.length));
  const tokens = starts.map((start, i) =>
    decoder.decode(bytes.subarray(start, i + 1 < starts.length ? starts[i + 1] : bytes.length)),
  );
  return { prompt, tokens };
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function WriteLive() {
  const [engine, setEngine] = useState<Engine | null>(null);
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [prompt, setPrompt] = useState("Once upon a time");
  const [temperature, setTemperature] = useState(0.8);
  const [written, setWritten] = useState<Written | null>(null);
  const [speed, setSpeed] = useState<number | null>(null);
  const [threshold, setThreshold] = useState(2.5);
  const stopRef = useRef(false);

  useEffect(() => {
    let current = true;
    loadEngine().then(
      (loaded) => {
        if (current) setEngine(loaded);
      },
      () => {
        if (current) setPhase({ kind: "failed", message: "The engine did not load in this browser." });
      },
    );
    return () => {
      current = false;
      stopRef.current = true;
    };
  }, []);

  const loadModel = useCallback(async () => {
    if (!engine) return;
    setPhase({ kind: "downloading", received: 0, total: MODEL_MB * 1e6 });
    try {
      const [config, weights, tokenizer] = await download((received) =>
        setPhase({ kind: "downloading", received, total: MODEL_MB * 1e6 }),
      );
      engine.loadModel(config, weights, tokenizer);
      setPhase({ kind: "ready" });
    } catch (error) {
      setPhase({ kind: "failed", message: message(error) });
    }
  }, [engine]);

  const write = useCallback(() => {
    if (!engine) return;
    stopRef.current = false;
    let promptText: string;
    try {
      promptText = engine.startGeneration(prompt, {
        maxTokens: 200,
        temperature,
        topP: 0.95,
        seed: Math.floor(Math.random() * 2 ** 31),
      });
    } catch (error) {
      setPhase({ kind: "failed", message: message(error) });
      return;
    }
    const tokens: PickedToken[] = [];
    let text = promptText;
    let computeMs = 0;
    setWritten({ text, tokens: [] });
    setSpeed(null);
    setPhase({ kind: "writing" });

    const frame = () => {
      const began = performance.now();
      let finished = stopRef.current;
      try {
        while (!finished && performance.now() - began < FRAME_BUDGET_MS) {
          const step = engine.step();
          if (step.done) {
            finished = true;
            break;
          }
          text = step.text;
          tokens.push(step.token);
        }
      } catch (error) {
        setPhase({ kind: "failed", message: message(error) });
        return;
      }
      computeMs += performance.now() - began;
      setWritten({ text, tokens: tokens.slice() });
      if (computeMs > 0 && tokens.length > 0) setSpeed((tokens.length * 1000) / computeMs);
      if (finished || stopRef.current) {
        setPhase({ kind: "ready" });
        return;
      }
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }, [engine, prompt, temperature]);

  const split = written ? pieces(written) : null;
  // The router in src/route.cpp with its "max entropy" signal: hand off at the first token whose
  // entropy reaches the threshold.
  const handoff = written ? written.tokens.findIndex((t) => t.entropy >= threshold) : -1;
  const unsure = written ? written.tokens.filter((t) => shade(t.entropy) === 3).length : 0;
  const writing = phase.kind === "writing";

  return (
    <div className="write">
      {phase.kind === "idle" || phase.kind === "downloading" ? (
        <div className="write-load">
          <button
            type="button"
            className="primary"
            onClick={loadModel}
            disabled={!engine || phase.kind === "downloading"}
          >
            {phase.kind === "downloading"
              ? `Downloading ${Math.min(99, Math.round((phase.received / phase.total) * 100))}%`
              : `Load the model (${MODEL_MB} MB)`}
          </button>
          <p>
            A 15M parameter Llama trained on children&apos;s stories, fetched from Hugging Face at a
            pinned revision and checked against its SHA-256 before the engine touches it.
          </p>
        </div>
      ) : null}

      {phase.kind === "ready" || phase.kind === "writing" ? (
        <form
          className="write-controls"
          onSubmit={(event) => {
            event.preventDefault();
            if (writing) stopRef.current = true;
            else write();
          }}
        >
          <label className="write-prompt">
            <span>Start the story</span>
            <input
              value={prompt}
              maxLength={200}
              onChange={(event) => setPrompt(event.target.value)}
              disabled={writing}
            />
          </label>
          <label className="lab-temperature">
            <span>Temperature</span>
            <input
              type="range"
              min={0}
              max={1.5}
              step={0.05}
              value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value))}
              disabled={writing}
            />
            <span className="lab-temperature-value">{temperature.toFixed(2)}</span>
          </label>
          <div className="lab-actions">
            <button
              type="submit"
              className={writing ? undefined : "primary"}
              disabled={!writing && prompt.trim() === ""}
            >
              {writing ? "Stop" : "Write"}
            </button>
          </div>
        </form>
      ) : null}

      {phase.kind === "failed" ? <p className="lab-note is-refused">{phase.message}</p> : null}

      {split && written ? (
        <>
          <p className="write-text">
            <span className="write-given">{split.prompt}</span>
            {split.tokens.map((piece, i) => {
              const t = written.tokens[i];
              return (
                <span key={i}>
                  {i === handoff ? (
                    <span className="handoff" title="A router with this threshold hands off here">
                      hand off
                    </span>
                  ) : null}
                  <span
                    className={`tok tok-${shade(t.entropy)}`}
                    title={`entropy ${t.entropy.toFixed(2)} nats, margin ${t.margin.toFixed(2)}`}
                  >
                    {piece}
                  </span>
                </span>
              );
            })}
            {writing ? <span className="write-caret" aria-hidden="true" /> : null}
          </p>
          <div className="write-meta">
            <span>
              {written.tokens.length} tokens
              {speed !== null ? `, ${Math.round(speed)} a second in this tab` : ""}
            </span>
            <span className="write-legend" aria-hidden="true">
              sure <i className="tok tok-0" />
              <i className="tok tok-1" />
              <i className="tok tok-2" />
              <i className="tok tok-3" /> unsure
            </span>
          </div>
          <label className="lab-temperature write-threshold">
            <span>Hand off at entropy</span>
            <input
              type="range"
              min={0.5}
              max={5}
              step={0.05}
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />
            <span className="lab-temperature-value">{threshold.toFixed(2)}</span>
          </label>
          <p className="lab-note">
            {handoff < 0
              ? `No token reached ${threshold.toFixed(2)} nats, so a router with this threshold keeps the whole answer local.`
              : `A local-first router with this threshold would stop at token ${handoff + 1} of ${written.tokens.length} and hand the request to a bigger model, having spent only those tokens locally.`}
          </p>
          {!writing && written.tokens.length > 0 ? (
            <p className="lab-note">
              {unsure === 0
                ? "The model was fairly sure of every token here."
                : `The darkest ${unsure === 1 ? "token is" : `${unsure} tokens are`} where the model was least sure, with an entropy over ${SHADES[2]} nats. A local-first router could hand the request to a bigger model at moments like these. Hover a token for its numbers.`}
            </p>
          ) : null}
          <p className="sr-only" aria-live="polite">
            {writing ? "" : written.text}
          </p>
        </>
      ) : null}
    </div>
  );
}
