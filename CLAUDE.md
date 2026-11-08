# Ember

C++20 LLM inference engine with no ML framework, plus a Next.js site in `web/` that runs it through WebAssembly. The dated plan is in `docs/PLAN.md`; keep its status column current.

## Commits

- Conventional commits, one logical change each, in plain language.
- Author: Arham Doshi <doshi.arham22@gmail.com>, GitHub account hammy229.
- No Co-authored-by or generated-by lines.

## Checks before a commit

- Format: `git ls-files '*.h' '*.cpp' | xargs uvx --from clang-format==18.1.8 clang-format -i`.
- C++: `cmake --build --preset debug && ctest --preset debug`, and the same with `sanitize`.
- WebAssembly and site: `./scripts/build-wasm.sh`, then `node tests/wasm_smoke.mjs`, then `npm run build` in `web/`.

## Rules

- The repo has been public since Nov 8, 2026. The site is not deployed yet; that and the demo are Nov 19.
- Never commit model weights or patient data.
- Never put a number on the site or in docs unless it was measured by a script in this repo.
- No em or en dashes in written copy.
