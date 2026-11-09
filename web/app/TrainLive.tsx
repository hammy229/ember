"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { loadEngine, type Engine } from "./engine";

// Tiny Shakespeare, public domain, from a pinned commit of Andrej Karpathy's char-rnn repository.
const TEXT_URL =
  "https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt";
const TEXT_MB = 1.1;
const STEPS = 1000;
const CHECK_EVERY = 50;
const FRAME_BUDGET_MS = 30;

type Point = { step: number; train: number; heldOut: number };
type Phase =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "training" }
  | { kind: "done" }
  | { kind: "failed"; message: string };

function Chart({ points, guessing }: { points: Point[]; guessing: number }) {
  const width = 560;
  const height = 220;
  const pad = { left: 40, right: 12, top: 12, bottom: 28 };
  const top = Math.ceil(guessing);
  const x = (step: number) => pad.left + (step / STEPS) * (width - pad.left - pad.right);
  const y = (loss: number) =>
    pad.top + (1 - Math.max(0, Math.min(loss, top)) / top) * (height - pad.top - pad.bottom);
  const line = (key: "train" | "heldOut") =>
    points.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p[key]).toFixed(1)}`).join("");
  const ticks = Array.from({ length: top + 1 }, (_, i) => i);
  return (
    <svg
      className="train-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Training and held-out loss falling over the steps"
    >
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pad.left} x2={width - pad.right} y1={y(t)} y2={y(t)} className="train-grid" />
          <text x={pad.left - 8} y={y(t) + 4} className="train-tick" textAnchor="end">
            {t}
          </text>
        </g>
      ))}
      <line
        x1={pad.left}
        x2={width - pad.right}
        y1={y(guessing)}
        y2={y(guessing)}
        className="train-guess"
      />
      <text x={width - pad.right} y={y(guessing) - 6} className="train-tick" textAnchor="end">
        guessing
      </text>
      <path d={line("train")} className="train-line" />
      <path d={line("heldOut")} className="train-line is-held" />
      <text x={pad.left} y={height - 6} className="train-tick">
        0
      </text>
      <text x={width - pad.right} y={height - 6} className="train-tick" textAnchor="end">
        {STEPS} steps
      </text>
    </svg>
  );
}

export function TrainLive() {
  const [engine, setEngine] = useState<Engine | null>(null);
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [points, setPoints] = useState<Point[]>([]);
  const [sample, setSample] = useState("");
  const [info, setInfo] = useState<{ parameters: number; vocab: number } | null>(null);
  const [rate, setRate] = useState<number | null>(null);
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

  const train = useCallback(async () => {
    if (!engine) return;
    stopRef.current = false;
    setPhase({ kind: "loading" });
    let text: string;
    try {
      const response = await fetch(TEXT_URL);
      if (!response.ok) throw new Error(`could not download the text (HTTP ${response.status})`);
      text = await response.text();
    } catch (error) {
      setPhase({ kind: "failed", message: error instanceof Error ? error.message : String(error) });
      return;
    }
    const seed = Math.floor(Math.random() * 2 ** 31);
    let started: { parameters: number; vocab: number };
    try {
      started = engine.startTraining(text, seed);
    } catch (error) {
      setPhase({ kind: "failed", message: error instanceof Error ? error.message : String(error) });
      return;
    }
    setInfo(started);
    const first = engine.heldOutLoss();
    setPoints([{ step: 0, train: first, heldOut: first }]);
    setSample(engine.sample(240, 0.8, seed));
    setPhase({ kind: "training" });

    let step = 0;
    let lossSum = 0;
    let lossCount = 0;
    let computeMs = 0;
    const frame = () => {
      if (stopRef.current) {
        setPhase({ kind: "done" });
        return;
      }
      const began = performance.now();
      while (step < STEPS && performance.now() - began < FRAME_BUDGET_MS) {
        lossSum += engine.trainSteps(1);
        lossCount += 1;
        step += 1;
        if (step % CHECK_EVERY === 0) {
          const point = { step, train: lossSum / lossCount, heldOut: engine.heldOutLoss() };
          lossSum = 0;
          lossCount = 0;
          setPoints((all) => [...all, point]);
          if (step % (CHECK_EVERY * 4) === 0 || step === STEPS) {
            setSample(engine.sample(240, 0.8, seed + step));
          }
        }
      }
      computeMs += performance.now() - began;
      if (computeMs > 0) setRate((step * 1000) / computeMs);
      if (step >= STEPS) {
        setPhase({ kind: "done" });
        return;
      }
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }, [engine]);

  const last = points[points.length - 1];
  const guessing = info ? Math.log(info.vocab) : 4.2;
  const busy = phase.kind === "loading" || phase.kind === "training";

  return (
    <div className="train">
      <div className="write-load">
        <button
          type="button"
          className={busy ? undefined : "primary"}
          disabled={!engine || phase.kind === "loading"}
          onClick={() => {
            if (busy) stopRef.current = true;
            else void train();
          }}
        >
          {phase.kind === "loading"
            ? "Loading the text"
            : phase.kind === "training"
              ? "Stop"
              : phase.kind === "done"
                ? "Train a new one"
                : `Train from scratch (${TEXT_MB} MB of text)`}
        </button>
        <p>
          A {info ? info.parameters.toLocaleString() : "111,040"} parameter transformer starts from
          random weights and learns Shakespeare one character at a time. Every gradient is computed
          by Ember&apos;s own autograd, in this tab.
        </p>
      </div>

      {phase.kind === "failed" ? <p className="lab-note is-refused">{phase.message}</p> : null}

      {last ? (
        <>
          <Chart points={points} guessing={guessing} />
          <div className="write-meta">
            <span>
              step {last.step} of {STEPS}, held-out loss {last.heldOut.toFixed(2)} nats per character
              {rate ? `, ${rate.toFixed(0)} steps a second` : ""}
            </span>
            <span className="train-legend">
              <i className="train-key" /> training <i className="train-key is-held" /> held out
            </span>
          </div>
          <pre className="train-sample">{sample.trimStart()}</pre>
          <p className="lab-note">
            The sample is the model writing on its own after {last.step} steps. Held-out loss is
            measured on the last tenth of the text, which it never trains on, so a falling line
            means it is learning the language, not memorizing.
          </p>
        </>
      ) : null}
    </div>
  );
}
