"use client";

import { useEffect, useRef, useState } from "react";
import { loadEngine, type Engine } from "./engine";

const TOKENS = 32;
const FEATURES = 16;
const FRAME_GAP_MS = 33;
const PHASE_PER_MS = 0.00028;
const IGNITE_MS = 1400;
const STILL_PHASE = 2.2;
const SLOW_DRAW_MS = 10;
const FRAMES_PER_SAMPLE = 30;
const READOUT_EVERY_FRAMES = 6;

// Cold ash to charred ember, one family of hues, so on a light page darkness alone reads as size.
const RAMP: [number, [number, number, number]][] = [
  [0, [236, 230, 221]],
  [0.3, [246, 199, 165]],
  [0.6, [239, 122, 58]],
  [0.85, [194, 65, 12]],
  [1, [124, 45, 18]],
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
  | { kind: "running"; version: string; checksPassed: boolean; still: boolean }
  | { kind: "failed" };

type Cell = { row: number; column: number };
type Readout = Cell & { chance: number | null };

function describe(readout: Readout): string {
  const { row, column, chance } = readout;
  if (column > row) {
    return `Token ${row + 1} cannot see token ${column + 1}, which comes later. Its chance is exactly 0.`;
  }
  if (chance === null) return "The engine has not computed this frame yet.";
  return `Token ${row + 1} gives token ${column + 1} ${(chance * 100).toFixed(2)}% of its attention.`;
}

function describeFrame(ms: number): string {
  // Browsers round their clocks, so a very short call can measure as zero.
  return ms < 0.1 ? "under 0.1 ms to compute a frame" : `${ms.toFixed(2)} ms to compute a frame`;
}

export function CoalField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const latest = useRef<Float32Array | null>(null);
  const selected = useRef<Cell | null>(null);
  const repaint = useRef<() => void>(() => {});
  const [status, setStatus] = useState<Status>({ kind: "loading" });
  const [frameMs, setFrameMs] = useState<number | null>(null);
  const [readout, setReadout] = useState<Readout | null>(null);
  const [announcement, setAnnouncement] = useState("");

  const read = (cell: Cell): Readout => ({
    ...cell,
    chance:
      cell.column <= cell.row ? (latest.current?.[cell.row * TOKENS + cell.column] ?? null) : null,
  });

  const select = (cell: Cell | null) => {
    selected.current = cell;
    setReadout(cell ? read(cell) : null);
    repaint.current();
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) {
      setStatus({ kind: "failed" });
      return;
    }

    const lookup = buildLookup();
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let engine: Engine | null = null;
    let stopped = false;
    let visible = true;
    let animation = 0;
    let lastFrame = 0;
    let phase = STILL_PHASE;
    let ignitedAt = 0;
    let heat = 0;
    let computeTotal = 0;
    let drawTotal = 0;
    let frames = 0;
    let glow = true;

    const size = () => {
      const scale = Math.min(window.devicePixelRatio || 1, 2);
      const box = canvas.getBoundingClientRect();
      canvas.width = Math.round(box.width * scale);
      canvas.height = Math.round(box.height * scale);
    };

    // Each row is scaled to its own hottest cell, because a row's chances shrink as it sees more
    // tokens, and the late rows would otherwise all look dark. The readout shows real values.
    const draw = () => {
      const grid = latest.current;
      const { width, height } = canvas;
      context.clearRect(0, 0, width, height);
      const pitch = Math.min(width, height) / TOKENS;
      const gap = Math.max(1, pitch * 0.09);
      const cell = pitch - gap;
      const corner = Math.min(cell * 0.18, 3);
      context.shadowColor = "rgba(194, 65, 12, 0.3)";
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
          const glowing = glow && level > 170;
          context.fillStyle = lookup[level];
          context.shadowBlur = glowing ? pitch * 0.9 * (level / 255) : 0;
          context.shadowOffsetY = glowing ? gap : 0;
          context.beginPath();
          // Older browsers have no roundRect, and a square coal is better than a crash.
          if ("roundRect" in context) {
            context.roundRect(column * pitch, row * pitch, cell, cell, corner);
          } else {
            (context as CanvasRenderingContext2D).rect(column * pitch, row * pitch, cell, cell);
          }
          context.fill();
        }
      }
      context.shadowBlur = 0;
      context.shadowOffsetY = 0;
      const chosen = selected.current;
      if (chosen) {
        const line = Math.max(1.5, pitch * 0.09);
        context.strokeStyle = "#1c1917";
        context.lineWidth = line;
        context.strokeRect(
          chosen.column * pitch - line / 2,
          chosen.row * pitch - line / 2,
          cell + line,
          cell + line,
        );
      }
    };
    repaint.current = draw;

    const compute = () => {
      if (!engine) return;
      const started = performance.now();
      latest.current = engine.attentionFrame(TOKENS, FEATURES, phase);
      computeTotal += performance.now() - started;
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
      heat = 1 - Math.pow(2, -10 * progress);
      compute();
      const drawStarted = performance.now();
      draw();
      drawTotal += performance.now() - drawStarted;
      frames += 1;
      if (frames % READOUT_EVERY_FRAMES === 0 && selected.current) {
        const cell = selected.current;
        setReadout({
          ...cell,
          chance:
            cell.column <= cell.row
              ? (latest.current?.[cell.row * TOKENS + cell.column] ?? null)
              : null,
        });
      }
      if (frames === FRAMES_PER_SAMPLE) {
        setFrameMs(computeTotal / frames);
        // Canvas glow is the slow part on a phone, so give it up before the grid starts to stutter.
        if (drawTotal / frames > SLOW_DRAW_MS) glow = false;
        computeTotal = 0;
        drawTotal = 0;
        frames = 0;
      }
    };

    size();
    draw();
    const observer = new ResizeObserver(() => {
      size();
      draw();
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
          still: reduceMotion,
        });
        if (reduceMotion) {
          heat = 1;
          compute();
          draw();
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
      repaint.current = () => {};
      cancelAnimationFrame(animation);
      observer.disconnect();
      inView.disconnect();
    };
  }, []);

  const cellAt = (event: React.PointerEvent<HTMLCanvasElement>): Cell | null => {
    const box = event.currentTarget.getBoundingClientRect();
    const pitch = Math.min(box.width, box.height) / TOKENS;
    const row = Math.floor((event.clientY - box.top) / pitch);
    const column = Math.floor((event.clientX - box.left) / pitch);
    if (row < 0 || row >= TOKENS || column < 0 || column >= TOKENS) return null;
    return { row, column };
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLCanvasElement>) => {
    const moves: Record<string, Cell> = {
      ArrowUp: { row: -1, column: 0 },
      ArrowDown: { row: 1, column: 0 },
      ArrowLeft: { row: 0, column: -1 },
      ArrowRight: { row: 0, column: 1 },
    };
    if (event.key === "Escape") {
      select(null);
      setAnnouncement("");
      return;
    }
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    const from = selected.current ?? { row: TOKENS / 2, column: TOKENS / 4 };
    const row = Math.min(TOKENS - 1, Math.max(0, from.row + move.row));
    const column = Math.min(row, Math.max(0, from.column + move.column));
    const cell = selected.current ? { row, column } : from;
    select(cell);
    setAnnouncement(describe(read(cell)));
  };

  return (
    <figure className="coal">
      <canvas
        ref={canvasRef}
        className="coal-canvas"
        role="img"
        tabIndex={0}
        aria-label="A triangular grid of glowing cells, each one an attention weight the engine is computing live. Use the arrow keys to read a cell, and Escape to stop."
        onPointerMove={(event) => {
          if (event.pointerType === "mouse") select(cellAt(event));
        }}
        onPointerDown={(event) => select(cellAt(event))}
        onPointerLeave={(event) => {
          if (event.pointerType === "mouse") select(null);
        }}
        onBlur={() => select(null)}
        onKeyDown={onKeyDown}
      />
      <span className="sr-only" aria-live="polite">
        {announcement}
      </span>
      <figcaption>
        <p className="coal-hover">
          {readout ? describe(readout) : "Point at a coal or tap one to read its number."}
        </p>
        <p>
          Made-up queries and keys through the real <code>MatMul</code> and <code>Softmax</code>,
          with each row scaled to its own hottest cell.
        </p>
        <div className="coal-readout">
          <p className="coal-status" role="status">
            {status.kind === "loading" && "Loading the engine"}
            {status.kind === "failed" &&
              "The engine could not load, so nothing here is live. Reload to try again. The rest of the page still reads fine."}
            {status.kind === "running" && (
              <>
                <span className="coal-version">engine {status.version}</span>{" "}
                <span>{status.checksPassed ? "self-checks passed" : "self-checks FAILED"}</span>
              </>
            )}
          </p>
          {status.kind === "running" && (
            <p className="coal-status">
              {status.still
                ? "one still frame, because reduced motion is on"
                : frameMs === null
                  ? "timing a frame"
                  : describeFrame(frameMs)}
            </p>
          )}
        </div>
      </figcaption>
    </figure>
  );
}
