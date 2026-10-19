# Doodle UI kit for the Offramp dashboard

User-chosen look: the brag-styles `doodle` set ("a hand-drawn pitch deck come alive"), translated from video to a
responsive web app. This file is the binding visual spec for `web/`; it overrides the PRD's "dark mode default"
(the user asked for this look explicitly).

Source set: `/Users/mac/.claude/plugins/cache/brag-scaled/brag-scaled/0.8.0-scaled/skills/brag-styles/sets/doodle/`
(`set.md`, `frames/f1-f4.webp` for composition, `ref/engine.html` for hand-drawn path generators — lift the
helpers, never its copy; all words and numbers come from Offramp).

## What carries over
- **Palette (CSS custom properties on `:root`).** The user asked for MORE colour than the set's single lime:
  a full pack of highlighter pens and sticky notes, every colour carrying one meaning, used consistently on
  every screen. (This deliberately overrides the set's "one highlighter colour" rule.)
  - Paper and pens: `--paper #f0ede8`, `--ink #0a0a0a`, `--muted #5f5c57` (darkened from #6f6c66 for 4.5:1),
    `--bluepen #1f4fd1` (links, live log timestamps), `--redpen #b3261e` (failure marks, crosses).
  - Highlighters (backgrounds behind ink text only — never as text colour on paper):
    `--hl-lime #c8ff00` MIGRATE / pass / success; `--hl-yellow #ffe94d` tier Lightning;
    `--hl-cyan #6fe3ff` tier Super; `--hl-violet #c9a7ff` tier Ultra; `--hl-pink #ff8fc7` KEEP_CLOSED / fail;
    `--hl-orange #ffb05c` MANUAL_REVIEW / warnings / budget.
  - Sticky notes (pastel tints for cards and panels): `--note-yellow #fff6b0`, `--note-pink #ffd9ec`,
    `--note-cyan #d4f6ff`, `--note-lime #edffb8`, `--note-violet #ebdfff`, `--note-orange #ffe2c2`.
    Rotate sticky notes slightly (±0.6–1.2°) and give each a strip of "tape" or a hand-drawn pin.
  - Tier colours are the same everywhere: stepper, site cards, heatmap headers, legend, report bars.
  - Heatmap: cells tinted by parity relative to the site's threshold — below threshold = pink tint scaled by
    distance, at/above = lime tint — with the parity number always printed in the cell (JetBrains Mono) and a
    hand-drawn circle on the chosen tier.
  - Colour is never the only signal: every chip and cell also has a word, number or icon. Ink text on any
    highlighter or note must reach 4.5:1 (all listed fills do with `#0a0a0a`).
- **Dark variant ("blackboard")** under `prefers-color-scheme: dark` and a toggle: `--paper #161616`,
  `--ink #f0ede8`, `--muted #b5b0a8`, `--bluepen #8fb0ff`, `--redpen #ff8a80`; highlighters and notes keep
  their hues (notes darken to ~22% tints of the highlighter on #161616 with light ink; text placed ON a
  highlighter stays `#0a0a0a`).
- **Type** via `next/font/google` (self-hosted at build): **Archivo 800** for claims and headings, **Patrick Hand**
  for handwritten annotations, labels, notes and chips, **JetBrains Mono** for every number, id and code path.
- **Texture:** 60px grid at 7% ink plus a fine paper grain, as one static background layer (pre-rendered SVG or
  PNG data URI). Never animated.
- **Shapes:** hand-drawn rough rectangles for cards and buttons (irregular `border-radius` like
  `255px 15px 225px 15px / 15px 225px 15px 255px` or SVG rough paths), 2–3px ink strokes, slight rotation
  (±0.6°) on sticky-note style elements only. No clip art; draw with lines.

## Signature move as UI feedback
Strokes draw themselves (`pathLength=1`, animated `stroke-dashoffset`) when state changes:
- stage stepper: a hand-drawn check draws on each completed stage; the current stage gets a pencil circle;
- tier heatmap: the chosen tier cell gets a hand-drawn circle; failing cells get a red-pen cross-hatch;
- verdicts: MIGRATE rows get a highlighter swipe under the tier name; KEEP_CLOSED gets an ink stamp
  "NOT MIGRATED · 71.0% < 95%" (real numbers from the run) with a short thud-style scale animation;
- new site cards spring in (spring easing, ~300 ms); a highlighter swipe lands under the headline saving number
  on the report.
- Optional ink "boil" (SVG turbulence displacement at ~10 fps) only on decorative strokes, never on text.

## Web rules (the video frame rules do NOT apply)
- Ignore the video rules: 1920x1080 frame, 28px minimum text, 120px safe area, `window.ready`.
- `prefers-reduced-motion: reduce` → no draw-on, no boil, no springs: render final states.
- Visible focus ring (highlighter outline + ink), keyboard reachable controls, 4.5:1 text contrast in both themes.
- 375px layout: no horizontal page scroll; wide tables/heatmap scroll inside their own container.
- No infinite animations on clickable elements.
- Badges "Nebius Token Factory" and "NVIDIA Nemotron" as hand-lettered stamp/sticker badges in the header and on
  every verdict chip (text only, no copied logos).
- When re-skinning existing work, grep `web/` for the previous palette's hex values and replace them all
  (including fixtures, SVGs and inline styles).
