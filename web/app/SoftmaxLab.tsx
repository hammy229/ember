"use client";

import { useEffect, useMemo, useState } from "react";
import { loadEngine, type Engine } from "./engine";

type Word = { word: string; score: number; hidden: boolean };

const START: Word[] = [
  { word: "glowing", score: 2.4, hidden: false },
  { word: "warm", score: 1.8, hidden: false },
  { word: "burning", score: 0.9, hidden: false },
  { word: "dark", score: -0.6, hidden: false },
  { word: "singing", score: -2.5, hidden: false },
];

const HUGE = 1000;
const SUMMARY_DELAY_MS = 500;

export function SoftmaxLab() {
  const [engine, setEngine] = useState<Engine | null>(null);
  const [failed, setFailed] = useState(false);
  const [words, setWords] = useState(START);
  const [temperature, setTemperature] = useState(1);
  const [huge, setHuge] = useState(false);
  const [summary, setSummary] = useState("");

  useEffect(() => {
    let current = true;
    loadEngine().then(
      (loaded) => {
        if (current) setEngine(loaded);
      },
      () => {
        if (current) setFailed(true);
      },
    );
    return () => {
      current = false;
    };
  }, []);

  const chances = useMemo(() => {
    if (!engine) return null;
    return engine.softmax(
      words.map((word) =>
        word.hidden ? -Infinity : word.score / temperature + (huge ? HUGE : 0),
      ),
    );
  }, [engine, words, temperature, huge]);

  const refused = engine !== null && chances === null;

  let top: { word: string; chance: number } | null = null;
  if (chances) {
    const best = chances.indexOf(Math.max(...chances));
    top = { word: words[best].word, chance: chances[best] };
  }
  const topWord = top?.word ?? null;
  const topChance = top?.chance ?? null;

  // One calm line for screen readers, once the dragging stops, rather than every number on change.
  useEffect(() => {
    const timer = setTimeout(() => {
      if (refused) setSummary("The engine refused this row.");
      else if (topWord !== null && topChance !== null) {
        setSummary(`${topWord} leads at ${(topChance * 100).toFixed(1)}%.`);
      }
    }, SUMMARY_DELAY_MS);
    return () => clearTimeout(timer);
  }, [refused, topWord, topChance]);

  const update = (index: number, change: Partial<Word>) =>
    setWords((all) => all.map((word, i) => (i === index ? { ...word, ...change } : word)));

  return (
    <div className="lab">
      <p className="lab-sentence">
        The coals were still{" "}
        {topWord ? (
          <span className="lab-answer">{topWord}</span>
        ) : (
          <span className="lab-blank" aria-label="blank" role="img" />
        )}
      </p>

      <span className="sr-only" aria-live="polite">
        {summary}
      </span>

      <div className="lab-rows">
        {words.map((word, index) => {
          const chance = chances?.[index] ?? 0;
          return (
            <div className={word.hidden ? "lab-row is-hidden" : "lab-row"} key={word.word}>
              <span className="lab-word">{word.word}</span>
              <div className="lab-score">
                <input
                  type="range"
                  aria-label={`Score for ${word.word}`}
                  aria-valuetext={word.score.toFixed(1)}
                  min={-6}
                  max={6}
                  step={0.1}
                  value={word.score}
                  disabled={word.hidden}
                  onChange={(event) => update(index, { score: Number(event.target.value) })}
                />
                <span className="lab-score-value" aria-hidden="true">
                  {word.score.toFixed(1)}
                </span>
              </div>
              <span className="lab-bar" aria-hidden="true">
                <span
                  style={{
                    transform: `scaleX(${chance})`,
                    background: `color-mix(in oklab, var(--ember), var(--ember-strong) ${Math.round(chance * 100)}%)`,
                  }}
                />
              </span>
              <span className="lab-chance">{chances ? `${(chance * 100).toFixed(1)}%` : "n/a"}</span>
              <button
                type="button"
                className={word.hidden ? "lab-hide is-on" : "lab-hide"}
                onClick={() => update(index, { hidden: !word.hidden })}
              >
                {word.hidden ? "Show" : "Hide"}
                <span className="sr-only"> {word.word}</span>
              </button>
            </div>
          );
        })}
      </div>

      <div className="lab-temperature">
        <label htmlFor="lab-temperature">Temperature</label>
        <input
          id="lab-temperature"
          type="range"
          aria-valuetext={temperature.toFixed(2)}
          min={0.1}
          max={3}
          step={0.05}
          value={temperature}
          onChange={(event) => setTemperature(Number(event.target.value))}
        />
        <span className="lab-temperature-value" aria-hidden="true">
          {temperature.toFixed(2)}
        </span>
      </div>

      <div className="lab-actions">
        <button type="button" className={huge ? "is-on" : undefined} onClick={() => setHuge(!huge)}>
          {huge ? `Take ${HUGE} back off every score` : `Add ${HUGE} to every score`}
        </button>
        <button
          type="button"
          onClick={() => setWords((all) => all.map((word) => ({ ...word, hidden: true })))}
        >
          Hide every word
        </button>
        <button
          type="button"
          onClick={() => {
            setWords(START);
            setTemperature(1);
            setHuge(false);
          }}
        >
          Reset
        </button>
      </div>

      <p className={refused ? "lab-note is-refused" : "lab-note"} role="status">
        {failed && "The engine could not load, so these bars cannot move. Reload to try again."}
        {!failed && !engine && "Loading the engine."}
        {refused &&
          "The engine refused this row and returned an error. With every word hidden there is nothing left to choose, and a made-up answer would be worse than none."}
        {engine && !refused && huge &&
          `Every score just went up by ${HUGE}. e^${HUGE} does not fit in a float, so the plain formula would give NaN here. Ember subtracts each row's largest score first, so the chances hold.`}
        {engine && !refused && !huge &&
          "The scores are made up so you can drag them. The chances are real: each change runs C++ Softmax in this tab."}
      </p>
    </div>
  );
}
