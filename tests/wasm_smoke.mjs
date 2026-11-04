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
console.log(`wasm smoke test passed: ember ${version}`);
