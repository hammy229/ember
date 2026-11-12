"use client";

import { useEffect, useRef } from "react";

const CLIPS = [
  {
    name: "write",
    href: "#write",
    title: "It writes, and shades each token",
    body: "The 15M model writes a story in real time. Then the hand-off slider moves where a router would stop and pass the request on.",
  },
  {
    name: "train",
    href: "#train",
    title: "It trains from random weights",
    body: "1000 steps of the 111,040 parameter model learning Shakespeare. This one clip plays at 4x speed.",
  },
  {
    name: "inside",
    href: "#inside",
    title: "It shows where each head looks",
    body: "One sentence, clicking through layers and heads. Every cell is a real attention weight from the forward pass.",
  },
  {
    name: "softmax",
    href: "#softmax",
    title: "Softmax holds at huge scores",
    body: "Temperature evens the odds out or lets one word take over. Adding 1000 to every score leaves the chances where they were.",
  },
];

// Plays a clip only while it is on screen, and never on its own for people who asked for less motion.
function Clip({ name }: { name: string }) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      video.controls = true;
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) void video.play().catch(() => {});
        else video.pause();
      },
      { threshold: 0.4 },
    );
    observer.observe(video);
    return () => observer.disconnect();
  }, []);

  return (
    <video
      ref={ref}
      src={`/demo/${name}.mp4`}
      poster={`/demo/${name}.jpg`}
      muted
      loop
      playsInline
      preload="none"
      aria-hidden="true"
      tabIndex={-1}
    />
  );
}

export function DemoReel() {
  return (
    <div className="reel">
      {CLIPS.map((clip) => (
        <figure key={clip.name}>
          <div className="reel-frame">
            <Clip name={clip.name} />
          </div>
          <figcaption>
            <h3>{clip.title}</h3>
            <p>{clip.body}</p>
            <a className="arrow" href={clip.href}>
              Try it below
            </a>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
