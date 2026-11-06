"use client";

import { useEffect, useState } from "react";
import { loadEngine, type Engine } from "./engine";

function verdict(passed: boolean) {
  return passed ? "passed" : "FAILED";
}

export function LiveChecks() {
  const [engine, setEngine] = useState<Engine | null>(null);
  const [failed, setFailed] = useState(false);

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

  return (
    <div className="checks">
      <p>
        The source is not public yet. For now, this is what you can check: the engine ran these on
        itself in this tab when the page loaded.
      </p>
      <dl>
        <div>
          <dt>32-bit memory checks</dt>
          <dd>{failed ? "could not run" : engine ? verdict(engine.memoryChecksPassed) : "running"}</dd>
        </div>
        <div>
          <dt>Math checks</dt>
          <dd>{failed ? "could not run" : engine ? verdict(engine.mathChecksPassed) : "running"}</dd>
        </div>
      </dl>
    </div>
  );
}
