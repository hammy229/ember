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

export function SoftmaxLab() {
  const [engine, setEngine] = useState<Engine | null>(null);
  const [failed, setFailed] = useState(false);
  const [words, setWords] = useState(START);
  const [temperature, setTemperature] = useState(1);
  const [huge, setHuge] = useState(false);

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
  const update = (index: number, change: Partial<Word>) =>
    setWords((all) => all.map((word, i) => (i === index ? { ...word, ...change } : word)));

  return (
    <div className="lab">
      <p className="lab-sentence">
        The coals were still <span className="lab-blank" aria-hidden="true" />
      </p>

      <div className="lab-rows">
        {words.map((word, index) => {
          const chance = chances?.[index] ?? 0;
          return (
            <div className={word.hidden ? "lab-row is-hidden" : "lab-row"} key={word.word}>
              <span className="lab-word">{word.word}</span>
              <label className="lab-score">
                <span className="sr-only">Score for {word.word}</span>
                <input
                  type="range"
                  min={-6}
                  max={6}
                  step={0.1}
                  value={word.score}
                  disabled={word.hidden}
                  onChange={(event) => update(index, { score: Number(event.target.value) })}
                />
                <output>{word.score.toFixed(1)}</output>
              </label>
              <span className="lab-bar" aria-hidden="true">
                <span style={{ transform: `scaleX(${chance})` }} />
              </span>
              <output className="lab-chance">
                {chances ? `${(chance * 100).toFixed(1)}%` : "n/a"}
              </output>
              <button
                type="button"
                className="lab-hide"
                aria-pressed={word.hidden}
                aria-label={`${word.hidden ? "Show" : "Hide"} ${word.word}`}
                onClick={() => update(index, { hidden: !word.hidden })}
              >
                {word.hidden ? "Show" : "Hide"}
              </button>
            </div>
          );
        })}
      </div>

      <label className="lab-temperature">
        <span>Temperature</span>
        <input
          type="range"
          min={0.1}
          max={3}
          step={0.05}
          value={temperature}
          onChange={(event) => setTemperature(Number(event.target.value))}
        />
        <output>{temperature.toFixed(2)}</output>
      </label>

      <div className="lab-actions">
        <button type="button" aria-pressed={huge} onClick={() => setHuge(!huge)}>
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
        {failed && "The engine could not load in this browser, so these bars cannot move."}
        {!failed && !engine && "Loading the engine."}
        {refused &&
          "The engine refused this row and returned an error. With every word hidden there is nothing left to choose, and a made-up answer would be worse than none."}
        {engine && !refused && huge &&
          `Every score just went up by ${HUGE}. e^${HUGE} does not fit in a float, so the plain formula would give NaN here. Ember subtracts each row's largest score first, so the chances hold.`}
        {engine && !refused && !huge &&
          "Scores are made up, since no model is loaded yet. The chances are real: each change runs C++ Softmax in this tab."}
      </p>
    </div>
  );
}
