---
name: Ember
description: Banked coals. A dark page around a grid the engine is computing live.
colors:
  soot: "#15110e"
  char: "#1f1915"
  line: "#362d26"
  line-strong: "#5c4d41"
  ash: "#ede5d8"
  smoke: "#a79c8e"
  ember: "#ff6b2c"
  ember-hot: "#ffd696"
  ember-deep: "#c2410c"
  ember-low: "#5c1e0e"
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
  button-primary: { backgroundColor: "{colors.ember}", textColor: "{colors.soot}", rounded: "{rounded.control}", height: "46px", padding: "0 22px" }
  button-primary-hover: { backgroundColor: "{colors.ember-hot}", textColor: "{colors.soot}", rounded: "{rounded.control}" }
  button-quiet: { textColor: "{colors.ash}", rounded: "{rounded.control}", height: "42px", padding: "0 16px" }
---

# Design System: Ember

## Overview

Ember's site is a dark page around one live object. The layout comes from the author's Tink and MedData sites: serif headlines, hairlines instead of cards, and a fixed side rail. The palette is its own, banked coals: soot, ash, and one ember orange.

The hero grid is not an illustration. Each cell is a value the C++ engine computed in the visitor's browser, so the page's main image is also its main evidence.

## Colors

- **Soot** `#15110e` is the page. **Char** `#1f1915` is the empty track of a bar.
- **Ash** `#ede5d8` is primary text. **Smoke** `#a79c8e` is secondary text.
- **Line** `#362d26` divides rows and sections. **Line strong** `#5c4d41` opens a list or table, and draws slider tracks.
- **Ember** `#ff6b2c` is for what is live or measured: bars, timings, passing checks, the softmax answer. It also marks hover and focus, and the one primary button.
- **Ember hot, deep, and low** are the heat ramp from the grid, also used for softmax bars, where more chance burns hotter.

## Typography

- **Spectral** for anything titled or said: headlines, the softmax sentence and its words.
- **Inter Tight** for reading and controls.
- **JetBrains Mono** only for data: timings, percentages, engine status. Never as decoration.

All three are self-hosted through Fontsource, because hosted font fetches have flaked in this author's CI.

## Layout

- Desktop has a 72px fixed rail with rotated labels. It widens to 180px on hover or focus and turns the labels upright. The current section's label is ash.
- Below 1024px the rail is a 56px top bar, and the grid moves up under the headline, so it stays in the first screen on a phone.
- Sections are bands separated by one hairline. Groups are columns divided by hairlines, not cards.
- Between 641px and 1399px, each softmax row gives its slider a full line, so the slider stays wide enough to use.

## Elevation and motion

The page is flat. The only glow is on hot coals, and it turns off automatically if drawing gets slow. The one authored moment is the grid igniting on load. Everything else is a short state change. With reduced motion on, the grid draws one still frame and says so.

## Do

- Show the engine working rather than describing it.
- Label anything made up, and say plainly what Ember cannot do yet.
- Keep numbers traceable: a repo script measured them, or the visitor's browser just did.
- Theme browser surfaces: selection, focus ring, scrollbar, slider.

## Don't

- Don't use ember for decoration. If it is orange, it is live, measured, or under the pointer.
- Don't add kickers above headings, section numbers, or cards.
- Don't use mono as a costume for "technical".
- Don't announce dates or link the repo before publish day.
