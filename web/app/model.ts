// Downloads the tiny model once and hands it to the engine, for every section that needs it.

import type { Engine } from "./engine";

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

export const MODEL_MB = 32;

type Listener = (received: number, total: number) => void;

let loading: Promise<void> | null = null;
const listeners = new Set<Listener>();
let loaded = false;

export function modelLoaded(): boolean {
  return loaded;
}

async function download(): Promise<Uint8Array[]> {
  let received = 0;
  const total = MODEL_MB * 1e6;
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
        for (const listener of listeners) listener(received, total);
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

// Downloads and loads the model the first time; later calls share the same promise.
export function ensureModel(engine: Engine, onProgress?: Listener): Promise<void> {
  if (onProgress) listeners.add(onProgress);
  loading ??= download()
    .then(([config, weights, tokenizer]) => {
      engine.loadModel(config, weights, tokenizer);
      loaded = true;
    })
    .catch((error: unknown) => {
      loading = null;
      throw error;
    });
  return loading.finally(() => {
    if (onProgress) listeners.delete(onProgress);
  });
}
