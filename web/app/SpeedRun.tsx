"use client";

import { useEffect, useRef, useState } from "react";
import { loadEngine } from "./engine";

// From docs/benchmarks/2026-11-06-m4.txt, made by scripts/bench.sh. Fastest of 10 runs.
const SAVED = [
  { size: 64, nativeMs: 0.0179 },
  { size: 256, nativeMs: 1.0364 },
  { size: 1024, nativeMs: 65.1702 },
];

const MIN_RUN_MS = 30;
const RUNS = 5;
const SLOW_RUNS = 2;
const SLOW_CALL_MS = 200;
// 1024 is 64 times the work of 256. Past this, it would freeze the tab for seconds at a time.
const TOO_SLOW_AT_256_MS = 60;

type Phase =
  | { kind: "idle" }
  | { kind: "running"; size: number | null }
  | { kind: "done"; message: string }
  | { kind: "failed" };

function formatMs(ms: number) {
  if (ms >= 100) return `${ms.toFixed(0)} ms`;
  if (ms >= 1) return `${ms.toFixed(1)} ms`;
  return `${ms.toFixed(3)} ms`;
}

const nextTask = () => new Promise((resolve) => setTimeout(resolve, 0));

export function SpeedRun() {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [results, setResults] = useState<Record<number, number>>({});
  const [skipped, setSkipped] = useState(false);
  const alive = useRef(true);
  const busy = useRef(false);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const run = async () => {
    if (busy.current) return;
    busy.current = true;
    setResults({});
    setSkipped(false);
    setPhase({ kind: "running", size: null });
    try {
      const engine = await loadEngine();
      const measured: Record<number, number> = {};
      for (const { size } of SAVED) {
        if (!alive.current) return;
        if (size === 1024 && measured[256] > TOO_SLOW_AT_256_MS) {
          setSkipped(true);
          continue;
        }
        setPhase({ kind: "running", size });
        await nextTask();
        if (!engine.matmul(size, 1)) throw new Error("the engine reported an error");

        // Browsers round their clocks, so short kernels repeat until the clock can see them.
        const probeStart = performance.now();
        engine.matmul(size, 1);
        const probe = Math.max(performance.now() - probeStart, 0.05);
        const calls = Math.max(1, Math.ceil(MIN_RUN_MS / probe));

        // Each run blocks the tab, so a slow size gets fewer of them.
        const runs = probe > SLOW_CALL_MS ? SLOW_RUNS : RUNS;
        let fastest = Infinity;
        for (let i = 0; i < runs; i += 1) {
          await nextTask();
          if (!alive.current) return;
          const started = performance.now();
          engine.matmul(size, calls);
          fastest = Math.min(fastest, (performance.now() - started) / calls);
        }
        measured[size] = fastest;
        setResults({ ...measured });
      }
      const largest = measured[1024] !== undefined ? 1024 : 256;
      if (alive.current) {
        setPhase({
          kind: "done",
          message: `Done. ${largest} x ${largest} took ${formatMs(measured[largest])} in this browser.`,
        });
      }
    } catch {
      if (alive.current) setPhase({ kind: "failed" });
    } finally {
      busy.current = false;
    }
  };

  const running = phase.kind === "running";

  return (
    <div className="run">
      <table>
        <caption className="sr-only">
          Time for one matrix multiply, in this browser and in a saved native run
        </caption>
        <thead>
          <tr>
            <th scope="col">Matrix size</th>
            <th scope="col">This browser, WebAssembly</th>
            <th scope="col">Apple M4, native, saved</th>
          </tr>
        </thead>
        <tbody>
          {SAVED.map(({ size, nativeMs }) => (
            <tr key={size}>
              <th scope="row">
                {size} x {size}
              </th>
              <td className={results[size] === undefined ? "run-empty" : "run-live"}>
                {results[size] !== undefined
                  ? formatMs(results[size])
                  : phase.kind === "running" && phase.size === size
                    ? "timing"
                    : skipped && size === 1024
                      ? "skipped, too slow here"
                      : "not run"}
              </td>
              <td>{formatMs(nativeMs)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="run-action">
        <button
          type="button"
          className="primary"
          aria-disabled={running}
          onClick={run}
        >
          {running ? "Timing" : phase.kind === "done" ? "Time it again" : "Time it on this machine"}
        </button>
        <div>
          <p>
            Fastest of up to 5 runs, on one thread. The tab can pause for a moment at the largest
            size. The WebAssembly build has no SIMD yet, and two different machines are not a fair
            race.
          </p>
          <p className="run-result" role="status">
            {phase.kind === "failed" && "The engine could not run here, so there is no timing."}
            {phase.kind === "done" && phase.message}
          </p>
        </div>
      </div>
    </div>
  );
}
