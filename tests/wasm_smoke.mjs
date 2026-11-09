// Smoke test: loads the WebAssembly build in Node and checks the engine version.
// Run after ./scripts/build-wasm.sh with: node tests/wasm_smoke.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";
import assert from "node:assert/strict";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const cmake = readFileSync(path.join(root, "CMakeLists.txt"), "utf8");
const expected = /project\(ember VERSION ([0-9.]+)/.exec(cmake)?.[1];
assert.ok(expected, "could not read the project version from CMakeLists.txt");

const { default: createEmber } = await import(
  pathToFileURL(path.join(root, "build", "wasm", "ember.js")).href
);
const ember = await createEmber();
const version = ember.UTF8ToString(ember._ember_version());

assert.equal(version, expected);

// Zero means every check passed. Anything else is the number of the check in wasm/exports.cpp.
assert.equal(ember._ember_memory_self_check(), 0);
assert.equal(ember._ember_ops_self_check(), 0);

// The coal grid: every row adds up to 1, and tokens never see the future.
const side = 16;
const frame = ember._ember_attention_frame(side, 8, 1.5);
assert.notEqual(frame, 0);
const grid = ember.HEAPF32.subarray(frame / 4, frame / 4 + side * side);
for (let row = 0; row < side; row += 1) {
  let total = 0;
  for (let column = 0; column < side; column += 1) {
    const value = grid[row * side + column];
    if (column > row) assert.equal(value, 0);
    total += value;
  }
  assert.ok(Math.abs(total - 1) < 1e-4, `row ${row} adds up to ${total}`);
}
assert.equal(ember._ember_attention_frame(0, 8, 0), 0);
assert.equal(ember._ember_attention_frame(16, 129, 0), 0);

// The softmax row the page lets people drag.
const softmaxRow = (scores) => {
  const bytes = scores.length * 4;
  const input = ember._malloc(bytes);
  const output = ember._malloc(bytes);
  ember.HEAPF32.set(scores, input / 4);
  const code = ember._ember_softmax_row(input, output, scores.length);
  const result = Array.from(ember.HEAPF32.subarray(output / 4, output / 4 + scores.length));
  ember._free(input);
  ember._free(output);
  return { code, result };
};
const known = softmaxRow([1, 2, 3]);
assert.equal(known.code, 0);
assert.ok(Math.abs(known.result[2] - 0.66524096) < 1e-5);
assert.equal(softmaxRow([0, -Infinity]).result[1], 0);
assert.equal(softmaxRow([-Infinity, -Infinity]).code, 1);
assert.equal(softmaxRow([1, NaN]).code, 1);

// The timing the page runs. Memory grows here, so the page must read HEAPF32 afresh after it.
assert.ok(Number.isFinite(ember._ember_matmul_bench(64, 1)));
assert.ok(Number.isFinite(ember._ember_matmul_bench(1024, 1)));
assert.ok(Number.isNaN(ember._ember_matmul_bench(0, 1)));
assert.ok(Number.isNaN(ember._ember_matmul_bench(2049, 1)));
assert.equal(softmaxRow([1, 2, 3]).code, 0);

console.log(`wasm smoke test passed: ember ${version}`);

// Text generation with the real tiny model, when it has been fetched (scripts/fetch-model.sh). The
// browser build must write exactly what the native build writes for the same greedy prompt.
const modelDir = process.env.EMBER_MODEL_DIR;
if (modelDir) {
  const copyIn = (bytes) => {
    const pointer = ember._malloc(Math.max(bytes.length, 1));
    ember.HEAPU8.set(bytes, pointer);
    return pointer;
  };
  const files = ["config.json", "model.safetensors", "tokenizer.json"].map((name) =>
    readFileSync(path.join(modelDir, name)),
  );
  const pointers = files.map(copyIn);
  const loaded = ember._ember_load_model(
    pointers[0], files[0].length, pointers[1], files[1].length, pointers[2], files[2].length,
  );
  pointers.forEach((p) => ember._free(p));
  assert.equal(loaded, 0, ember.UTF8ToString(ember._ember_last_error()));

  const prompt = "Once upon a time";
  const promptBytes = ember.lengthBytesUTF8(prompt) + 1;
  const promptPointer = ember._malloc(promptBytes);
  ember.stringToUTF8(prompt, promptPointer, promptBytes);
  const promptTokens = ember._ember_generate_start(promptPointer, 40, 0, 1, 0);
  ember._free(promptPointer);
  assert.equal(promptTokens, 5, ember.UTF8ToString(ember._ember_last_error()));

  let steps = 0;
  let code;
  while ((code = ember._ember_generate_step()) === 1) {
    steps += 1;
    const last = ember.HEAPF32.subarray(ember._ember_last_token() / 4, ember._ember_last_token() / 4 + 5);
    assert.ok(last[1] >= 0, "entropy is never negative");
    assert.ok(last[2] >= 0 && last[2] <= 1, "the margin is a difference of probabilities");
  }
  assert.equal(code, 0, ember.UTF8ToString(ember._ember_last_error()));
  assert.equal(steps, 40);
  assert.equal(
    ember.UTF8ToString(ember._ember_generated_text()),
    "Once upon a time, there was a little girl named Lily. She loved to play outside in the " +
      "sunshine. One day, she saw a big, red ball in the sky. It was the sun!",
  );
  console.log("wasm generation matches the native build");
}
