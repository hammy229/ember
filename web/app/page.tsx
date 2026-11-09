import { CoalField } from "./CoalField";
import { LiveChecks } from "./LiveChecks";
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

// From docs/results/2026-11-08-apple-m4.md, made by scripts/results.sh. Median of the runs, one
// forward pass of 128 tokens through the 15M parameter TinyStories Llama, Apple M4.
const MODEL_TIMES = [
  { name: "NumPy with Apple's tuned BLAS", ms: 21.03, baseline: true },
  { name: "Ember, SIMD spelled out", ms: 158.14, baseline: false },
  { name: "Ember, one commit earlier", ms: 259.32, baseline: false },
];
const MODEL_SLOWEST = Math.max(...MODEL_TIMES.map((row) => row.ms));

// From docs/results/2026-11-09-apple-m4.md, the next run of the same script. Each figure says what
// produced it.
const MODEL_FACTS = [
  { value: "3.0e-5", label: "largest difference from Hugging Face's own logits, checked on every push" },
  { value: "156 / 156", label: "positions where Ember and Hugging Face pick the same next token" },
  { value: "814", label: "tokens a second on one thread, with 147 MB for the whole process" },
  { value: "97.8%", label: "of the engine's lines run by its 328 tests" },
];

const PROOF = [
  {
    title: "It refuses to be quietly wrong",
    body: "Writing a matrix product over its own input would corrupt it with no error, so MatMul returns one instead. A NaN stops at Softmax rather than reaching a sampler.",
  },
  {
    title: "Machines check every change to main",
    body: "CI builds with GCC and runs the tests under AddressSanitizer and UBSan. UBSan caught a real null pointer bug in MatMul on Nov 6, before that code was ever pushed.",
  },
  {
    title: "It gets checked against the definition",
    body: "Every layer and the whole model run against a second version written straight from the formulas. Breaking RoPE, the attention scale, or the causal mask on purpose makes those tests fail, so they catch real bugs.",
  },
  {
    title: "Model files are treated as hostile",
    body: "A downloaded file can lie about its own sizes. Every truncation of a model and thousands of corrupted headers have to fail cleanly under AddressSanitizer, and a fuzzer attacks the loader on every push.",
  },
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
              On a laptop it now loads a 15M parameter Llama and runs it all the way to logits. It
              does not generate text yet, and no model runs in this tab yet. Text generation is the
              next milestone.
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
            <h2>The order of three loops is worth more than a factor of twelve.</h2>
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
                    <span
                      className="bar"
                      style={{ width: `calc((100% - 76px) * ${row.ms / SLOWEST})` }}
                    />
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

        <section className="band" id="model">
          <div className="band-wide">
            <h2>A real model, checked against Hugging Face.</h2>
            <p className="band-lead">
              Ember loads a 15M parameter Llama trained on children&apos;s stories and runs it to the
              last logit. After &ldquo;Once upon a time&rdquo; it picks a comma, and so does Hugging
              Face&apos;s own library, with scores that agree to within 0.00003. A separate Llama
              written in NumPy agrees with every one of the 32,000.
            </p>
          </div>

          <dl className="facts">
            {MODEL_FACTS.map((fact) => (
              <div key={fact.value}>
                <dt>{fact.value}</dt>
                <dd>{fact.label}</dd>
              </div>
            ))}
          </dl>

          <figure className="bars">
            <figcaption>One forward pass of 128 tokens, Apple M4, shorter is better</figcaption>
            <dl>
              {MODEL_TIMES.map((row) => (
                <div key={row.name}>
                  <dt>{row.name}</dt>
                  <dd>
                    <span
                      className={row.baseline ? "bar is-baseline" : "bar"}
                      style={{ width: `calc((100% - 76px) * ${row.ms / MODEL_SLOWEST})` }}
                    />
                    <span className="bar-value">{row.ms.toFixed(0)} ms</span>
                  </dd>
                </div>
              ))}
            </dl>
          </figure>
          <p className="foot">
            NumPy is still 7.5 times faster, because it hands the math to a library tuned by hand for
            this chip. Writing the SIMD registers out by hand made Ember 1.6 times faster with
            answers identical to the bit. Every number here comes from one run of the repo&apos;s
            results script, which anyone can rerun.
          </p>
        </section>

        <section className="band" id="proof">
          <div className="band-wide">
            <h2>Built so that you can check it instead of trusting it.</h2>
          </div>
          <LiveChecks />
          <div className="columns">
            {PROOF.map((item) => (
              <article key={item.title}>
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </article>
            ))}
          </div>
        </section>

        <footer className="close">
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
