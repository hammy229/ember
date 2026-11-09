"use client";

import { useCallback, useEffect, useState } from "react";
import { loadEngine, type Engine, type Inside } from "./engine";
import { ensureModel, MODEL_MB } from "./model";

const MAX_TOKENS = 24;

// Pale ash to deep ember, the same ramp as the grid at the top of the page.
function heat(value: number): string {
  const v = Math.max(0, Math.min(1, value));
  const from = [242, 238, 231];
  const to = [124, 45, 18];
  const mix = from.map((f, i) => Math.round(f + (to[i] - f) * Math.sqrt(v)));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}

function show(token: string): string {
  if (token === "\n") return "\\n";
  return token.replace(/ /g, "·");
}

export function InsideLive() {
  const [engine, setEngine] = useState<Engine | null>(null);
  const [prompt, setPrompt] = useState("Once upon a time, there was a little girl named Lily. She");
  const [inside, setInside] = useState<Inside | null>(null);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let current = true;
    loadEngine().then(
      (loaded) => {
        if (current) setEngine(loaded);
      },
      () => {
        if (current) setStatus("The engine did not load in this browser.");
      },
    );
    return () => {
      current = false;
    };
  }, []);

  const look = useCallback(async () => {
    if (!engine) return;
    setBusy(true);
    try {
      await ensureModel(engine, (received, total) =>
        setStatus(`Downloading the model, ${Math.min(99, Math.round((received / total) * 100))}%`),
      );
      setStatus("");
      setInside(engine.inspect(prompt, MAX_TOKENS));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [engine, prompt]);

  const n = inside?.tokens.length ?? 0;
  const weight = (i: number, j: number) =>
    inside ? inside.attention[((layer * inside.heads + head) * n + i) * n + j] : 0;

  return (
    <div className="inside">
      <form
        className="write-controls"
        onSubmit={(event) => {
          event.preventDefault();
          void look();
        }}
      >
        <label className="write-prompt">
          <span>Text to look inside (the first {MAX_TOKENS} tokens)</span>
          <input value={prompt} maxLength={200} onChange={(event) => setPrompt(event.target.value)} />
        </label>
        <div className="lab-actions">
          <button type="submit" className="primary" disabled={!engine || busy || prompt.trim() === ""}>
            {inside ? "Look again" : "Look inside"}
          </button>
        </div>
        <p className="lab-note">
          {status ||
            (inside
              ? ""
              : `Needs the same ${MODEL_MB} MB model as the section above, downloaded once.`)}
        </p>
      </form>

      {inside ? (
        <>
          <div className="inside-pickers">
            <div>
              <span>Layer</span>
              {Array.from({ length: inside.layers }, (_, l) => (
                <button key={l} type="button" className={l === layer ? "is-on" : undefined}
                  onClick={() => setLayer(l)}>
                  {l + 1}
                </button>
              ))}
            </div>
            <div>
              <span>Head</span>
              {Array.from({ length: inside.heads }, (_, h) => (
                <button key={h} type="button" className={h === head ? "is-on" : undefined}
                  onClick={() => setHead(h)}>
                  {h + 1}
                </button>
              ))}
            </div>
          </div>

          <div className="inside-attention" style={{ gridTemplateColumns: `auto repeat(${n}, 1fr)` }}
            onMouseLeave={() => setHover(null)}>
            <span />
            {inside.tokens.map((t, j) => (
              <span key={`top-${j}`} className="inside-label is-top">{show(t)}</span>
            ))}
            {inside.tokens.map((t, i) => (
              <div key={`row-${i}`} className="inside-row">
                <span className="inside-label">{show(t)}</span>
                {inside.tokens.map((_, j) => (
                  <span key={j} className="inside-cell"
                    style={{ background: j > i ? "transparent" : heat(weight(i, j)) }}
                    onMouseEnter={() => setHover({ i, j })} />
                ))}
              </div>
            ))}
          </div>
          <p className="write-meta">
            <span>
              {hover && hover.j <= hover.i
                ? `"${show(inside.tokens[hover.i])}" puts ${(weight(hover.i, hover.j) * 100).toFixed(1)}% of its attention on "${show(inside.tokens[hover.j])}"`
                : `Layer ${layer + 1}, head ${head + 1}. Each row is a token, and its cells show where it looks. Point at one.`}
            </span>
          </p>

          <h3 className="inside-title">What it would say next, after each layer</h3>
          <div className="inside-lens-wrap">
            <table className="inside-lens">
              <thead>
                <tr>
                  <th />
                  {inside.tokens.map((t, j) => (
                    <th key={j}>{show(t)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: inside.layers }, (_, l) => (
                  <tr key={l}>
                    <th>{l + 1 === inside.layers ? "output" : `layer ${l + 1}`}</th>
                    {inside.tokens.map((_, j) => {
                      const [word, p] = inside.lens[l * n + j];
                      return (
                        <td key={j} style={{ background: heat(p), color: p > 0.45 ? "var(--paper)" : "var(--ink)" }}
                          title={`${(p * 100).toFixed(1)}%`}>
                          {show(word)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="lab-note">
            The logit lens runs each layer&apos;s half-finished state through the model&apos;s output
            layer early. Early layers guess common words; the right next word usually settles in the
            last few. Darker means more confident.
          </p>
        </>
      ) : null}
    </div>
  );
}
