---
name: Ember
description: Kiln paper. A warm light page around a grid the engine is computing live.
colors:
  paper: "#faf8f4"
  paper-deep: "#f2eee7"
  rule: "#e4ddd2"
  rule-strong: "#c9bfb1"
  ink: "#1c1917"
  graphite: "#6b6259"
  ember: "#e8590c"
  ember-ink: "#b8410a"
  ember-strong: "#9a3412"
  ember-pale: "#fde3d2"
  slate: "#4f6478"
typography:
  display: { fontFamily: "Spectral, Georgia, serif", fontSize: "clamp(44px, min(4.5vw, 8vh), 88px)", fontWeight: 300, lineHeight: 1.04, letterSpacing: "-0.025em" }
  headline: { fontFamily: "Spectral, Georgia, serif", fontSize: "clamp(32px, 3.5vw, 56px)", fontWeight: 300, lineHeight: 1.08, letterSpacing: "-0.02em" }
  title: { fontFamily: "Spectral, Georgia, serif", fontSize: "23px", fontWeight: 400, lineHeight: 1.25 }
  body: { fontFamily: "Inter Tight Variable, system-ui, sans-serif", fontSize: "17px", fontWeight: 400, lineHeight: 1.6 }
  data: { fontFamily: "JetBrains Mono Variable, ui-monospace, monospace", fontSize: "14px", fontWeight: 400, lineHeight: 1.45 }
rounded:
  mark: "4px"
  control: "3px"
  chip: "2px"
spacing:
  rail: "72px"
  gutter: "clamp(24px, 4vw, 64px)"
  band: "clamp(80px, 11vw, 160px)"
components:
  button-primary: { backgroundColor: "{colors.ember-ink}", textColor: "{colors.paper}", rounded: "{rounded.control}", height: "46px", padding: "0 22px" }
  button-primary-hover: { backgroundColor: "{colors.ember-strong}", textColor: "{colors.paper}", rounded: "{rounded.control}" }
  button-quiet: { textColor: "{colors.ink}", rounded: "{rounded.control}", height: "42px", padding: "0 16px" }
---

# Design System: Ember

## Overview

Ember's site is a light page around one live object. The layout comes from the author's Tink and MedData sites: serif headlines, hairlines instead of cards, and a fixed side rail. The palette is its own, kiln paper: a warm page like a lab notebook, near-black ink, one ember accent for what Ember measured, and slate for whatever it is compared against. It replaced a dark "banked coals" palette on Nov 8, when the author asked for a near-white background.

The hero grid is not an illustration. Each cell is a value the C++ engine computed in the visitor's browser, so the page's main image is also its main evidence.

## Colors

- **Paper** `#faf8f4` is the page. **Paper deep** `#f2eee7` is the empty track of a bar.
- **Ink** `#1c1917` is primary text. **Graphite** `#6b6259` is secondary text (5.6:1 on paper).
- **Rule** `#e4ddd2` divides rows and sections. **Rule strong** `#c9bfb1` opens a list or table, and draws slider tracks.
- **Ember** `#e8590c` fills what is live or measured: bars and the softmax bars. It is too light for text on paper, so text in the accent color uses **Ember ink** `#b8410a` (5.2:1): timings, passing checks, the softmax answer, hover and focus, and the one primary button, which carries paper-colored text.
- **Ember strong** `#9a3412` is hover on the primary button, refusals, and the big measured figures.
- **Slate** `#4f6478` is the baseline Ember is compared with, so a reader can tell the two apart without reading labels.
- **The grid** runs from cold ash `#ece6dd` through peach and ember to charred `#7c2d12`. On a light page darker means more attention, so the eye reads weight, not glare.

## Typography

- **Spectral** for anything titled or said: headlines, the softmax sentence and its words.
- **Inter Tight** for reading and controls.
- **JetBrains Mono** only for data: timings, percentages, engine status. Never as decoration.

All three are self-hosted through Fontsource, because hosted font fetches have flaked in this author's CI.

## Layout

- Desktop has a 72px fixed rail with rotated labels. It widens to 180px on hover or focus and turns the labels upright. The current section's label is ink.
- Below 1024px the rail is a 56px top bar, and the grid moves up under the headline, so it stays in the first screen on a phone.
- Sections are bands separated by one hairline. Groups are columns divided by hairlines, not cards.
- Between 641px and 1399px, each softmax row gives its slider a full line, so the slider stays wide enough to use.

## Elevation and motion

The page is flat. The only glow is a faint warm shadow on the hottest coals, and it turns off automatically if drawing gets slow. The one authored moment is the grid igniting on load. Everything else is a short state change. With reduced motion on, the grid draws one still frame and says so.

## Do

- Show the engine working rather than describing it.
- Label anything made up, and say plainly what Ember cannot do yet.
- Keep numbers traceable: a repo script measured them, or the visitor's browser just did.
- Theme browser surfaces: selection, focus ring, scrollbar, slider.

## Don't

- Don't use ember for decoration. If it is orange, it is live, measured, or under the pointer. If it is slate, it is the thing Ember is measured against.
- Don't add kickers above headings, section numbers, or cards.
- Don't use mono as a costume for "technical".
- Don't announce dates or link the repo before publish day.
