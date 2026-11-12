"use client";

import { useEffect, useState } from "react";

const LINKS = [
  { href: "#grid", label: "Live grid" },
  { href: "#demo", label: "Demo" },
  { href: "#write", label: "Write" },
  { href: "#inside", label: "Inside" },
  { href: "#train", label: "Train" },
  { href: "#softmax", label: "Softmax" },
  { href: "#speed", label: "Speed" },
  { href: "#model", label: "Model" },
  { href: "#qwen", label: "Qwen3" },
  { href: "#route", label: "Route" },
  { href: "#proof", label: "Proof" },
];

export function Rail() {
  const [current, setCurrent] = useState(LINKS[0].href);

  useEffect(() => {
    // Only a thin band across the middle of the window counts, so one section is current at a time.
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setCurrent(`#${entry.target.id}`);
        }
      },
      { rootMargin: "-45% 0px -50% 0px" },
    );
    for (const link of LINKS) {
      const section = document.querySelector(link.href);
      if (section) observer.observe(section);
    }
    return () => observer.disconnect();
  }, []);

  return (
    <nav className="rail" aria-label="Sections">
      <a className="rail-mark" href="#grid" aria-label="Ember, back to the top">
        <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">
          <rect x="2" y="2" width="6" height="6" rx="1.5" fill="var(--ember)" />
          <rect x="2" y="9" width="6" height="6" rx="1.5" fill="var(--ember-ink)" />
          <rect x="9" y="9" width="6" height="6" rx="1.5" fill="var(--ember)" />
          <rect x="2" y="16" width="6" height="6" rx="1.5" fill="var(--ember-strong)" />
          <rect x="9" y="16" width="6" height="6" rx="1.5" fill="var(--ember-ink)" />
          <rect x="16" y="16" width="6" height="6" rx="1.5" fill="var(--ember-pale)" />
        </svg>
        <span className="rail-name">Ember</span>
      </a>
      <ul>
        {LINKS.map((link) => (
          <li key={link.href}>
            <a href={link.href} aria-current={current === link.href ? "location" : undefined}>
              <span>{link.label}</span>
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
