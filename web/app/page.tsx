import { EngineStatus } from "./EngineStatus";

export default function Home() {
  return (
    <main>
      <h1>Ember</h1>
      <p>An LLM inference engine written from scratch in C++, running in your browser.</p>
      <EngineStatus />
      <p className="note">Early setup. Ember does not generate text yet.</p>
    </main>
  );
}
