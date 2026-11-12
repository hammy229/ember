import { Bars } from "./Bars";
import { CoalField } from "./CoalField";
import { DemoReel } from "./DemoReel";
import { InsideLive } from "./InsideLive";
import { LiveChecks } from "./LiveChecks";
import { Rail } from "./Rail";
import { SoftmaxLab } from "./SoftmaxLab";
import { SpeedRun } from "./SpeedRun";
import { TrainLive } from "./TrainLive";
import { WriteLive } from "./WriteLive";

// From docs/benchmarks/2026-11-06-m4.txt, made by scripts/bench.sh. Fastest of 10 runs at 1024 x 1024.
const PRODUCT_TIMES = [
  { name: "MatMul", value: 65.1702 },
  { name: "MatMul, transposed input copied first", value: 69.0571 },
  { name: "MatMul, transposed input", value: 810.2957 },
  { name: "Textbook loops", value: 828.4239 },
].map((row) => ({ ...row, label: `${row.value.toFixed(0)} ms` }));

// From docs/results/2026-11-11-apple-m4.md, made by scripts/results.sh. Median of the runs, one
// forward pass of 128 tokens through the 15M parameter TinyStories Llama, Apple M4.
const MODEL_TIMES = [
  { name: "NumPy with Apple's tuned BLAS", value: 20.26, baseline: true },
  { name: "Ember", value: 154.17 },
].map((row) => ({ ...row, label: `${row.value.toFixed(0)} ms` }));

// From the same run, except "156 / 156": the same-pick check is asserted by
// tests/hf_reference_test.cpp, which the run passes. Speed is the one-token row, the cost of writing
// each new token.
// The 1.6x SIMD figure in this section's footnote is from docs/results/2026-11-08-apple-m4.md,
// made by results.sh --before 08c2e73, which built and timed both commits.
const MODEL_FACTS = [
  {
    value: "3.0e-5",
    label: "largest difference from Hugging Face's own logits, checked on every push",
  },
  {
    value: "156 / 156",
    label: "positions where Ember and Hugging Face pick the same next token",
  },
  {
    value: "846",
    label: "tokens a second writing on one thread, with 147 MB for the whole process",
  },
  { value: "91.9%", label: "of the engine's lines run by its 431 tests" },
];

// From the README's Qwen3 rows: tests/hf_reference_test.cpp against scripts/hf_reference_qwen3.py,
// and docs/results/quant-qwen3-0.6b.md made by scripts/quant_report.py.
const QWEN_FACTS = [
  {
    value: "1.0e-4",
    label:
      "largest difference from Hugging Face across the top 10 logits, same next token at 73 of 73 positions",
  },
  {
    value: "2.2x",
    label: "faster with 8-bit weights: 58.9 tokens a second against 26.2 in f32, 6 threads",
  },
  { value: "2377 MB", label: "peak memory in 8 bits, against 3545 MB in f32" },
  {
    value: "94.2%",
    label: "of positions where 8 bits picks the same next token as f32, mean KL 0.011 nats",
  },
];

// From docs/results/llamacpp-qwen3-0.6b.md, made by scripts/compare_llamacpp.py. Qwen3 0.6B in 8
// bits on both, CPU only, 6 threads, median of 3 runs, Apple M4.
const LLAMACPP_WRITE = [
  { name: "llama.cpp", value: 126.9, baseline: true },
  { name: "Ember", value: 58.0 },
].map((row) => ({ ...row, label: `${row.value.toFixed(0)} tok/s` }));
const LLAMACPP_READ = [
  { name: "llama.cpp", value: 642.9, baseline: true },
  { name: "Ember", value: 257.9 },
].map((row) => ({ ...row, label: `${row.value.toFixed(0)} tok/s` }));

// From docs/results/route-gsm8k.md, made by scripts/route_eval.py on all 1319 GSM8K test questions.
const ROUTE_AUROC = [
  { name: "Highest entropy in the answer", value: 0.74 },
  { name: "Question length", value: 0.687, baseline: true },
  { name: "Random", value: 0.478, baseline: true },
].map((row) => ({ ...row, label: row.value.toFixed(3) }));
const ROUTE_FACTS = [
  {
    value: "62.1%",
    label: "of GSM8K questions Qwen3 0.6B gets right on its own, in Ember",
  },
  {
    value: "28.9%",
    label: "of held-out questions escalated by a threshold set on the other half",
  },
  {
    value: "51.2%",
    label: "of the local model's mistakes caught by those escalations, 95% CI 45.0 to 57.5",
  },
  {
    value: "73.3%",
    label: "accuracy on the questions kept local, against 61.2% keeping everything",
  },
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
              animation. Your browser is computing it right now, with the same math attention runs
              on.
            </p>
            <p className="hero-honest">
              It loads a 15M parameter Llama and writes stories with it, in this tab, on your
              machine. The model is small, so the stories are simple. The engine underneath is the
              point.
            </p>
            <p className="hero-links">
              <a className="arrow" href="#write">
                Watch it write
              </a>
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

        <section className="band" id="demo">
          <div className="band-wide">
            <h2>See it run first.</h2>
            <p className="band-lead">
              Each clip is this page, recorded in headless Chrome. The training clip plays at 4x
              speed, and nothing else is changed. Every section below lets you do it yourself.
            </p>
          </div>
          <DemoReel />
        </section>

        <section className="band split" id="write">
          <div className="band-head">
            <h2>It writes here, and shows where it was unsure.</h2>
            <p>
              Every word below comes from the C++ engine running in your browser, with no server.
              Each token is shaded by the entropy of the model&apos;s choice: pale where it was
              sure, dark where many next words looked about as good.
            </p>
            <p>
              That signal is free to an engine that owns its logits, and it is what a local-first
              router needs to decide when to hand a request to a bigger model.
            </p>
          </div>
          <WriteLive />
        </section>

        <section className="band" id="inside">
          <div className="band-head">
            <h2>Look inside while it reads.</h2>
            <p>
              The engine can record what it does on the way: every attention head&apos;s weights,
              and what the model would predict if it stopped after each layer. Both come from the
              same forward pass that writes the stories above.
            </p>
          </div>
          <InsideLive />
        </section>

        <section className="band split" id="train">
          <div className="band-head">
            <h2>Train one from scratch, here.</h2>
            <p>
              Ember also learns. Its own autograd computes every gradient, checked against finite
              differences in its tests, and AdamW updates the weights. Press train and a small
              transformer goes from random noise to something shaped like Shakespeare.
            </p>
          </div>
          <TrainLive />
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

          <Bars
            caption="Time for one 1024 x 1024 multiply, Apple M4, one thread, shorter is better"
            rows={PRODUCT_TIMES}
          />

          <SpeedRun />
          <p className="foot">
            Saved numbers come from one run of the repo&apos;s benchmark script, before any
            hand-written SIMD. They are a starting point, not a claim of being fast. The comparison
            with llama.cpp is further down.
          </p>
        </section>

        <section className="band" id="model">
          <div className="band-wide">
            <h2>A real model, checked against Hugging Face.</h2>
            <p className="band-lead">
              Ember loads a 15M parameter Llama trained on children&apos;s stories and runs it to
              the last logit. After &ldquo;Once upon a time&rdquo; it picks a comma, and so does
              Hugging Face&apos;s own library, with scores that agree to within 0.00003. A separate
              Llama written in NumPy agrees with every one of the 32,000.
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

          <Bars
            caption="One forward pass of 128 tokens, Apple M4, shorter is better"
            rows={MODEL_TIMES}
          />
          <p className="foot">
            NumPy is still 7.6 times faster, because it hands the math to a library tuned by hand
            for this chip. Writing the SIMD registers out by hand made Ember 1.6 times faster with
            answers identical to the bit, in the Nov 8 run, which timed both commits. Every number
            here comes from the repo&apos;s results script, which anyone can rerun.
          </p>
        </section>

        <section className="band" id="qwen">
          <div className="band-wide">
            <h2>A 0.6B model, next to llama.cpp.</h2>
            <p className="band-lead">
              The same engine runs Qwen3 0.6B on the laptop, matches Hugging Face on it, and loads
              it in 8 bits. llama.cpp, the engine most local AI runs on, is still more than twice as
              fast. Here is by how much.
            </p>
          </div>

          <dl className="facts">
            {QWEN_FACTS.map((fact) => (
              <div key={fact.value}>
                <dt>{fact.value}</dt>
                <dd>{fact.label}</dd>
              </div>
            ))}
          </dl>

          <Bars
            caption="Writing, tokens a second, Qwen3 0.6B in 8 bits, CPU only, 6 threads, Apple M4, longer is better"
            rows={LLAMACPP_WRITE}
          />
          <Bars
            caption="Reading a 121 token prompt, tokens a second, same setup, longer is better"
            rows={LLAMACPP_READ}
          />
          <p className="foot">
            Ember runs at 46% of llama.cpp&apos;s speed writing and 40% reading, with the same model
            in the same 8-bit format. It was 33% and 19% before integer dot products and a lock-free
            thread pool. The 58.0 here and the 58.9 above are two runs by two scripts on the same
            build. The browser runs only the 15M model, because Qwen3 needs more memory than a
            WebAssembly tab should take.
          </p>
        </section>

        <section className="band" id="route">
          <div className="band-wide">
            <h2>It can tell when it is likely wrong.</h2>
            <p className="band-lead">
              Qwen3 0.6B answered all 1319 GSM8K test questions in Ember. The highest per-token
              entropy in each answer, read straight from the logits, flags the wrong ones better
              than question length does, so a router can keep easy questions local and send the rest
              to a bigger model.
            </p>
          </div>

          <dl className="facts">
            {ROUTE_FACTS.map((fact) => (
              <div key={fact.value}>
                <dt>{fact.value}</dt>
                <dd>{fact.label}</dd>
              </div>
            ))}
          </dl>

          <Bars
            caption="How well each signal spots a wrong answer (AUROC, bars start at 0, 0.5 is chance), longer is better"
            rows={ROUTE_AUROC}
          />
          <p className="foot">
            The gap over question length is +0.053 AUROC, 95% CI +0.023 to +0.094. No cloud model
            was called: escalated questions are scored with the answer key, so this measures whether
            the signal finds the local model&apos;s mistakes, not what a combined system would
            score. One model, one benchmark, greedy decoding.
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
