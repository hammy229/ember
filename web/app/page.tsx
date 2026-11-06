import { CoalField } from "./CoalField";
import { Rail } from "./Rail";
import { SoftmaxLab } from "./SoftmaxLab";
import { SpeedRun } from "./SpeedRun";

// From docs/benchmarks/2026-11-06-m4.txt, made by scripts/bench.sh. Fastest of 10 runs at 1024 x 1024.
const PRODUCT_TIMES = [
  { name: "MatMul", ms: 65.1702 },
  { name: "MatMul, transposed input copied first", ms: 69.0571 },
  { name: "MatMul, transposed input", ms: 810.2957 },
  { name: "Textbook loops", ms: 828.4239 },
];
const SLOWEST = Math.max(...PRODUCT_TIMES.map((row) => row.ms));

const PROOF = [
  {
    title: "It refuses to be quietly wrong",
    body: "Writing a matrix product over its own input would corrupt it with no error, so MatMul returns one instead. A NaN stops at Softmax rather than reaching a sampler.",
  },
  {
    title: "Machines check it on every push",
    body: "CI builds with GCC and runs the tests under AddressSanitizer and UBSan. UBSan caught a real null pointer bug in MatMul on Nov 6, before that code was ever pushed.",
  },
  {
    title: "It gets checked against the definition",
    body: "MatMul runs against a plain product written straight from the formula, at random sizes, through every mix of transposed inputs and outputs. The browser build checks its own answers too.",
  },
];

const PLAN = [
  { when: "Nov 4 to 6", what: "Build setup, tensors, memory allocator, matrix math, softmax, first benchmark", done: true },
  { when: "Nov 7", what: "Load model weights with memory mapping, fuzz the loader against corrupt files", done: false },
  { when: "Nov 8 to 9", what: "Attention and the full forward pass. Logits must match Hugging Face, checked in CI", done: false },
  { when: "Nov 10 to 11", what: "Tokenizer, sampling, KV cache. Real text generation in this browser", done: false },
  { when: "Nov 13 to 14", what: "NEON SIMD, threads, 8-bit quantization. The same model against llama.cpp, published even if Ember loses", done: false },
  { when: "Nov 15 to 17", what: "If generation lands on time: autograd, then training a tiny transformer live in the browser", done: false },
  { when: "Nov 19", what: "Demo day. The code goes public", done: false },
];

export default function Home() {
  return (
    <>
      <Rail />
      <main>
        <section className="hero" id="grid">
          <div className="hero-text">
            <h1>An LLM engine, written from scratch, running in this tab.</h1>
            <p className="lead">
              Ember is C++20 with no ML framework, compiled to WebAssembly. The grid here is not an
              animation. Your browser is computing it right now, with the same math attention runs on.
            </p>
            <p className="hero-honest">
              It does not load a model or generate text yet. Tensors, memory, and the core math are
              done. Text generation is the next milestone.
            </p>
            <p className="hero-links">
              <a className="arrow" href="#softmax">
                Drag the math yourself
              </a>
              <a className="arrow" href="#speed">
                Time it on your machine
              </a>
            </p>
          </div>
          <CoalField />
        </section>

        <section className="band split" id="softmax">
          <div className="band-head">
            <h2>Softmax turns scores into chances. Try to break it.</h2>
            <p>
              A language model scores every possible next word, and this one step decides how those
              scores become odds. Raise the temperature and the odds even out. Lower it and one word
              takes over.
            </p>
          </div>
          <SoftmaxLab />
        </section>

        <section className="band" id="speed">
          <div className="band-wide">
            <h2>The order of three loops is worth a factor of twelve.</h2>
            <p className="band-lead">
              MatMul and the textbook loops add the same numbers in the same order, but only MatMul
              reads memory in a straight line. Attention multiplies by a transposed matrix, which
              breaks that line, so the first benchmark changed how Ember will build attention: copy
              first.
            </p>
          </div>

          <figure className="bars">
            <figcaption>Time for one 1024 x 1024 multiply, Apple M4, one thread, shorter is better</figcaption>
            <dl>
              {PRODUCT_TIMES.map((row) => (
                <div key={row.name}>
                  <dt>{row.name}</dt>
                  <dd>
                    <span className="bar" style={{ width: `${(row.ms / SLOWEST) * 88}%` }} />
                    <span className="bar-value">{row.ms.toFixed(0)} ms</span>
                  </dd>
                </div>
              ))}
            </dl>
          </figure>

          <SpeedRun />
          <p className="foot">
            Saved numbers come from one run of the repo&apos;s benchmark script, with no hand-written SIMD
            yet. They are a starting point, not a claim of being fast. No comparison to another engine
            exists yet.
          </p>
        </section>

        <section className="band" id="proof">
          <div className="band-wide">
            <h2>Built so that you can check it instead of trusting it.</h2>
          </div>
          <div className="columns">
            {PROOF.map((item) => (
              <article key={item.title}>
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="band split" id="plan">
          <div className="band-head">
            <h2>What is finished, and what comes next.</h2>
            <p>These dates are the plan, not promises. A lit coal means finished and checked.</p>
          </div>
          <ol className="plan">
            {PLAN.map((row) => (
              <li key={row.when} className={row.done ? "is-done" : undefined}>
                <span className="plan-dot" aria-hidden="true" />
                <span className="plan-when">{row.when}</span>
                <span className="plan-what">
                  {row.what}
                  {row.done && <span className="sr-only"> (finished)</span>}
                </span>
              </li>
            ))}
          </ol>
        </section>

        <footer className="close">
          <p className="close-line">The code goes public on November 19.</p>
          <p className="close-by">
            Made by{" "}
            <a className="plain" href="https://arhamdoshi.dev">
              Arham Doshi
            </a>
          </p>
        </footer>
      </main>
    </>
  );
}
