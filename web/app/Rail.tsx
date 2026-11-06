const LINKS = [
  { href: "#grid", label: "Live grid" },
  { href: "#softmax", label: "Softmax" },
  { href: "#speed", label: "Speed" },
  { href: "#proof", label: "Proof" },
];

export function Rail() {
  return (
    <nav className="rail" aria-label="Sections">
      <a className="rail-mark" href="#grid" aria-label="Ember, back to the top">
        <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">
          <rect x="2" y="2" width="6" height="6" rx="1.5" fill="var(--ember)" />
          <rect x="2" y="9" width="6" height="6" rx="1.5" fill="var(--ember-deep)" />
          <rect x="9" y="9" width="6" height="6" rx="1.5" fill="var(--ember)" />
          <rect x="2" y="16" width="6" height="6" rx="1.5" fill="var(--ember-low)" />
          <rect x="9" y="16" width="6" height="6" rx="1.5" fill="var(--ember-deep)" />
          <rect x="16" y="16" width="6" height="6" rx="1.5" fill="var(--ember-hot)" />
        </svg>
        <span className="rail-name">Ember</span>
      </a>
      <ul>
        {LINKS.map((link) => (
          <li key={link.href}>
            <a href={link.href}>
              <span>{link.label}</span>
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
