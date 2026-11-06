"use client";

import { useEffect, useRef, useState } from "react";
import { loadEngine, type Engine } from "./engine";

const TOKENS = 32;
const FEATURES = 16;
const FRAME_GAP_MS = 33;
const PHASE_PER_MS = 0.00028;
const IGNITE_MS = 1400;
const STILL_PHASE = 2.2;

// Dark coal to white heat, one family of hues, so brightness alone reads as size.
const RAMP: [number, [number, number, number]][] = [
  [0, [38, 29, 24]],
  [0.25, [92, 30, 14]],
  [0.55, [194, 65, 12]],
  [0.8, [255, 107, 44]],
  [1, [255, 214, 150]],
];

function buildLookup(): string[] {
  const steps: string[] = [];
  for (let i = 0; i < 256; i += 1) {
    const t = i / 255;
    let upper = 1;
    while (upper < RAMP.length - 1 && RAMP[upper][0] < t) upper += 1;
    const [t0, c0] = RAMP[upper - 1];
    const [t1, c1] = RAMP[upper];
    const mix = (t - t0) / (t1 - t0);
    const channel = (index: number) => Math.round(c0[index] + (c1[index] - c0[index]) * mix);
    steps.push(`rgb(${channel(0)},${channel(1)},${channel(2)})`);
  }
  return steps;
}

type Status =
  | { kind: "loading" }
  | { kind: "running"; version: string; checksPassed: boolean }
  | { kind: "failed" };

type Hover = { row: number; column: number; chance: number | null };

export function CoalField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const latest = useRef<Float32Array | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "loading" });
  const [frameMs, setFrameMs] = useState<number | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    const lookup = buildLookup();
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let engine: Engine | null = null;
    let stopped = false;
    let visible = true;
    let animation = 0;
    let lastFrame = 0;
    let phase = STILL_PHASE;
    let ignitedAt = 0;
    let timeTotal = 0;
    let timeCount = 0;

    const size = () => {
      const scale = Math.min(window.devicePixelRatio || 1, 2);
      const box = canvas.getBoundingClientRect();
      canvas.width = Math.round(box.width * scale);
      canvas.height = Math.round(box.height * scale);
    };

    // Each row is scaled to its own hottest cell, because a row's chances shrink as it sees more
    // tokens, and the late rows would otherwise all look dark. The hover readout shows real values.
    const draw = (grid: Float32Array | null, heat: number) => {
      const { width, height } = canvas;
      context.clearRect(0, 0, width, height);
      const pitch = Math.min(width, height) / TOKENS;
      const gap = Math.max(1, pitch * 0.09);
      const cell = pitch - gap;
      const corner = Math.min(cell * 0.18, 3);
      for (let row = 0; row < TOKENS; row += 1) {
        let hottest = 0;
        if (grid) {
          for (let column = 0; column <= row; column += 1) {
            hottest = Math.max(hottest, grid[row * TOKENS + column]);
          }
        }
        for (let column = 0; column <= row; column += 1) {
          const relative = grid && hottest > 0 ? grid[row * TOKENS + column] / hottest : 0;
          const level = Math.round(Math.pow(relative, 1.3) * heat * 255);
          context.fillStyle = lookup[level];
          context.shadowBlur = level > 170 ? pitch * 0.9 * (level / 255) : 0;
          context.shadowColor = "rgba(255, 107, 44, 0.75)";
          context.shadowOffsetY = level > 170 ? gap : 0;
          context.beginPath();
          context.roundRect(column * pitch, row * pitch, cell, cell, corner);
          context.fill();
        }
      }
      context.shadowBlur = 0;
      context.shadowOffsetY = 0;
    };

    const compute = () => {
      if (!engine) return null;
      const started = performance.now();
      const grid = engine.attentionFrame(TOKENS, FEATURES, phase);
      timeTotal += performance.now() - started;
      timeCount += 1;
      if (timeCount === 30) {
        setFrameMs(timeTotal / timeCount);
        timeTotal = 0;
        timeCount = 0;
      }
      latest.current = grid;
      return grid;
    };

    const tick = (now: number) => {
      if (stopped) return;
      animation = requestAnimationFrame(tick);
      if (!visible || now - lastFrame < FRAME_GAP_MS) return;
      const elapsed = lastFrame === 0 ? 0 : Math.min(now - lastFrame, 100);
      lastFrame = now;
      phase += elapsed * PHASE_PER_MS;
      if (ignitedAt === 0) ignitedAt = now;
      const progress = Math.min((now - ignitedAt) / IGNITE_MS, 1);
      draw(compute(), 1 - Math.pow(2, -10 * progress));
    };

    size();
    draw(null, 0);
    const observer = new ResizeObserver(() => {
      size();
      draw(latest.current, latest.current ? 1 : 0);
    });
    observer.observe(canvas);
    const inView = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
    });
    inView.observe(canvas);

    loadEngine().then(
      (loaded) => {
        if (stopped) return;
        engine = loaded;
        setStatus({
          kind: "running",
          version: loaded.version,
          checksPassed: loaded.memoryChecksPassed && loaded.mathChecksPassed,
        });
        if (reduceMotion) {
          draw(compute(), 1);
          setFrameMs(timeCount > 0 ? timeTotal / timeCount : null);
        } else {
          animation = requestAnimationFrame(tick);
        }
      },
      () => {
        if (!stopped) setStatus({ kind: "failed" });
      },
    );

    return () => {
      stopped = true;
      cancelAnimationFrame(animation);
      observer.disconnect();
      inView.disconnect();
    };
  }, []);

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const pitch = Math.min(box.width, box.height) / TOKENS;
    const row = Math.floor((event.clientY - box.top) / pitch);
    const column = Math.floor((event.clientX - box.left) / pitch);
    if (row < 0 || row >= TOKENS || column < 0 || column >= TOKENS) {
      setHover(null);
      return;
    }
    const value = column <= row ? latest.current?.[row * TOKENS + column] : undefined;
    setHover({ row, column, chance: value ?? null });
  };

  return (
    <figure className="coal">
      <canvas
        ref={canvasRef}
        className="coal-canvas"
        role="img"
        aria-label="A triangular grid of glowing cells, each one an attention weight the engine is computing live."
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHover(null)}
      />
      <figcaption>
        <p className="coal-hover" aria-live="off">
          {hover === null && "Point at a coal to read its number."}
          {hover !== null &&
            hover.chance !== null &&
            `Token ${hover.row + 1} gives token ${hover.column + 1} ${(hover.chance * 100).toFixed(2)}% of its attention.`}
          {hover !== null &&
            hover.chance === null &&
            hover.column > hover.row &&
            `Token ${hover.row + 1} cannot see token ${hover.column + 1}, which comes later. Its chance is exactly 0.`}
          {hover !== null &&
            hover.chance === null &&
            hover.column <= hover.row &&
            "The engine has not computed this frame yet."}
        </p>
        <p>
          Made-up queries and keys through the real <code>MatMul</code> and <code>Softmax</code>,
          with each row brightened to its own hottest cell.
        </p>
        <p className="coal-status" role="status">
          {status.kind === "loading" && "Loading the engine"}
          {status.kind === "failed" &&
            "The engine could not load in this browser, so nothing here is live. The rest of the page still reads fine."}
          {status.kind === "running" && (
            <>
              <span>engine {status.version}</span>
              <span>{status.checksPassed ? "self-checks passed" : "self-checks FAILED"}</span>
              <span>{frameMs === null ? "timing a frame" : `${frameMs.toFixed(2)} ms a frame`}</span>
            </>
          )}
        </p>
      </figcaption>
    </figure>
  );
}
