// The one place the site talks to the WebAssembly build. Everything shown as computed on this page
// comes through here, so it really is the C++ engine doing the math.

type EmberModule = {
  HEAPF32: Float32Array;
  UTF8ToString(pointer: number): string;
  _malloc(bytes: number): number;
  _free(pointer: number): void;
  _ember_version(): number;
  _ember_memory_self_check(): number;
  _ember_ops_self_check(): number;
  _ember_attention_frame(tokens: number, features: number, phase: number): number;
  _ember_softmax_row(logits: number, out: number, count: number): number;
  _ember_matmul_bench(size: number, calls: number): number;
};

export type Engine = {
  version: string;
  memoryChecksPassed: boolean;
  mathChecksPassed: boolean;
  /** A tokens x tokens grid, row by row. Copied, so it stays valid after the next call. */
  attentionFrame(tokens: number, features: number, phase: number): Float32Array | null;
  /** Chances for one row of scores, or null if the engine refused the row. */
  softmax(scores: readonly number[]): number[] | null;
  /** Runs MatMul. Returns false if the engine reported an error. */
  matmul(size: number, calls: number): boolean;
};

const ENGINE_URL = "/wasm/ember.js";

function wrap(module: EmberModule): Engine {
  return {
    version: module.UTF8ToString(module._ember_version()),
    memoryChecksPassed: module._ember_memory_self_check() === 0,
    mathChecksPassed: module._ember_ops_self_check() === 0,
    attentionFrame(tokens, features, phase) {
      const pointer = module._ember_attention_frame(tokens, features, phase);
      if (pointer === 0) return null;
      const start = pointer >> 2;
      // Read the heap fresh on every call, because it moves when WebAssembly memory grows.
      return module.HEAPF32.slice(start, start + tokens * tokens);
    },
    softmax(scores) {
      const bytes = scores.length * 4;
      let input = 0;
      let output = 0;
      try {
        input = module._malloc(bytes);
        output = module._malloc(bytes);
        if (input === 0 || output === 0) return null;
        module.HEAPF32.set(scores, input >> 2);
        if (module._ember_softmax_row(input, output, scores.length) !== 0) return null;
        const start = output >> 2;
        return Array.from(module.HEAPF32.subarray(start, start + scores.length));
      } finally {
        module._free(input);
        module._free(output);
      }
    },
    matmul(size, calls) {
      return !Number.isNaN(module._ember_matmul_bench(size, calls));
    },
  };
}

let loading: Promise<Engine> | null = null;

export function loadEngine(): Promise<Engine> {
  loading ??= import(/* turbopackIgnore: true */ /* webpackIgnore: true */ ENGINE_URL)
    .then(async ({ default: createModule }: { default: () => Promise<EmberModule> }) =>
      wrap(await createModule()),
    )
    .catch((error: unknown) => {
      // Forget a failed load, so a passing network problem does not last until a full reload.
      loading = null;
      throw error;
    });
  return loading;
}
