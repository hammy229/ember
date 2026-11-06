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
