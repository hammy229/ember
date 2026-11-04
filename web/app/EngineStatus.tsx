"use client";

import { useEffect, useState } from "react";

type EmberModule = {
  _ember_version(): number;
  UTF8ToString(pointer: number): string;
};

type Status =
  | { kind: "loading" }
  | { kind: "ready"; version: string }
  | { kind: "failed"; message: string };

const ENGINE_URL = "/wasm/ember.js";

async function loadVersion(): Promise<string> {
  const { default: createModule } = (await import(
    /* turbopackIgnore: true */ /* webpackIgnore: true */ ENGINE_URL
  )) as { default: () => Promise<EmberModule> };
  const engine = await createModule();
  return engine.UTF8ToString(engine._ember_version());
}

export function EngineStatus() {
  const [status, setStatus] = useState<Status>({ kind: "loading" });

  useEffect(() => {
    let current = true;
    loadVersion().then(
      (version) => {
        if (current) setStatus({ kind: "ready", version });
      },
      (error: unknown) => {
        if (current) setStatus({ kind: "failed", message: String(error) });
      },
    );
    return () => {
      current = false;
    };
  }, []);

  return (
    <p className="status" role="status">
      {status.kind === "loading" && "Loading the engine..."}
      {status.kind === "ready" && `C++ engine ${status.version} loaded through WebAssembly.`}
      {status.kind === "failed" && `The engine could not load: ${status.message}`}
    </p>
  );
}
