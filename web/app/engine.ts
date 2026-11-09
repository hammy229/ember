// The one place the site talks to the WebAssembly build. Everything shown as computed on this page
// comes through here, so it really is the C++ engine doing the math.

type EmberModule = {
  HEAPF32: Float32Array;
  HEAPU8: Uint8Array;
  UTF8ToString(pointer: number): string;
  stringToUTF8(text: string, pointer: number, maxBytes: number): void;
  lengthBytesUTF8(text: string): number;
  _malloc(bytes: number): number;
  _free(pointer: number): void;
  _ember_version(): number;
  _ember_memory_self_check(): number;
  _ember_ops_self_check(): number;
  _ember_attention_frame(tokens: number, features: number, phase: number): number;
  _ember_softmax_row(logits: number, out: number, count: number): number;
  _ember_matmul_bench(size: number, calls: number): number;
  _ember_load_model(
    config: number,
    configBytes: number,
    weights: number,
    weightBytes: number,
    tokenizer: number,
    tokenizerBytes: number,
  ): number;
  _ember_generate_start(
    prompt: number,
    maxTokens: number,
    temperature: number,
    topP: number,
    seed: number,
  ): number;
  _ember_generate_step(): number;
  _ember_generated_text(): number;
  _ember_last_token(): number;
  _ember_last_error(): number;
};

/** One token the engine picked, and how sure the model was when it picked it. */
export type PickedToken = {
  id: number;
  /** Entropy of the next-token distribution, in nats. */
  entropy: number;
  /** Top probability minus the runner-up's. */
  margin: number;
  topProbability: number;
  /** Where this token's text starts in the full text, in UTF-8 bytes. */
  start: number;
};

export type Step = { done: true } | { done: false; text: string; token: PickedToken };

export type GenerationSettings = {
  maxTokens: number;
  temperature: number;
  topP: number;
  seed: number;
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
  /** Hands the three model files to the engine. Throws with the engine's own error message. */
  loadModel(config: Uint8Array, weights: Uint8Array, tokenizer: Uint8Array): void;
  /** Starts a new generation and gives back the prompt's text as the engine decoded it. */
  startGeneration(prompt: string, settings: GenerationSettings): string;
  /** Picks the next token. Throws with the engine's own error message. */
  step(): Step;
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
    loadModel(config, weights, tokenizer) {
      const pointers: number[] = [];
      try {
        for (const bytes of [config, weights, tokenizer]) {
          const pointer = module._malloc(Math.max(bytes.length, 1));
          if (pointer === 0) throw new Error("the tab ran out of memory for the model");
          pointers.push(pointer);
          module.HEAPU8.set(bytes, pointer);
        }
        const code = module._ember_load_model(
          pointers[0],
          config.length,
          pointers[1],
          weights.length,
          pointers[2],
          tokenizer.length,
        );
        if (code !== 0) throw new Error(module.UTF8ToString(module._ember_last_error()));
      } finally {
        // The engine keeps its own copy, so these can go.
        for (const pointer of pointers) module._free(pointer);
      }
    },
    startGeneration(prompt, settings) {
      const bytes = module.lengthBytesUTF8(prompt) + 1;
      const pointer = module._malloc(bytes);
      try {
        module.stringToUTF8(prompt, pointer, bytes);
        const code = module._ember_generate_start(
          pointer,
          settings.maxTokens,
          settings.temperature,
          settings.topP,
          settings.seed,
        );
        if (code < 0) throw new Error(module.UTF8ToString(module._ember_last_error()));
      } finally {
        module._free(pointer);
      }
      return module.UTF8ToString(module._ember_generated_text());
    },
    step() {
      const code = module._ember_generate_step();
      if (code < 0) throw new Error(module.UTF8ToString(module._ember_last_error()));
      if (code === 0) return { done: true };
      const start = module._ember_last_token() >> 2;
      const [id, entropy, margin, topProbability, offset] = module.HEAPF32.slice(start, start + 5);
      return {
        done: false,
        text: module.UTF8ToString(module._ember_generated_text()),
        token: { id, entropy, margin, topProbability, start: offset },
      };
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
