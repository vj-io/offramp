// Hand-drawn primitives for the doodle kit. Path generators are lifted from the brag-styles
// doodle engine (line, circle, check, cross) with a seeded RNG so server and client draw the
// same wobble. Strokes use pathLength=1 and the `.draw` class to draw themselves on mount.
import type { CSSProperties, ReactNode } from "react";
import type { Decision, Tier } from "@/lib/types";

export const TIER_HL: Record<Tier, string> = {
  lightning: "var(--hl-yellow)",
  super: "var(--hl-cyan)",
  ultra: "var(--hl-violet)",
};
export const TIER_NOTE: Record<Tier, string> = {
  lightning: "var(--note-yellow)",
  super: "var(--note-cyan)",
  ultra: "var(--note-violet)",
};
export const DECISION_HL: Record<Decision, string> = {
  MIGRATE: "var(--hl-lime)",
  KEEP_CLOSED: "var(--hl-pink)",
  MANUAL_REVIEW: "var(--hl-orange)",
};
export const DECISION_NOTE: Record<Decision, string> = {
  MIGRATE: "var(--note-lime)",
  KEEP_CLOSED: "var(--note-pink)",
  MANUAL_REVIEW: "var(--note-orange)",
};

function rng(seed: string) {
  let s = 7;
  for (const c of seed) s = (s * 31 + c.charCodeAt(0)) % 2147483647;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}
const f1 = (v: number) => v.toFixed(1);

function generators(seed: string) {
  const rnd = rng(seed);
  const J = (a: number) => (rnd() - 0.5) * 2 * a;
  const P = (x: number, y: number) => `${f1(x)} ${f1(y)}`;
  const line = (x1: number, y1: number, x2: number, y2: number, b = 3) =>
    `M${P(x1 + J(1.5), y1 + J(1.5))} Q${P((x1 + x2) / 2 + J(b), (y1 + y2) / 2 + J(b))} ${P(x2 + J(1.5), y2 + J(1.5))}`;
  const circle = (cx: number, cy: number, rx: number, ry: number, turn = 1.12, a0 = -2.2) => {
    let d = "";
    for (let i = 0; i <= 48; i++) {
      const a = a0 + (i / 48) * Math.PI * 2 * turn;
      const k = 1 + J(0.03) + (i / 48) * 0.06;
      d += `${i ? " L" : "M"}${P(cx + Math.cos(a) * rx * k, cy + Math.sin(a) * ry * k)}`;
    }
    return d;
  };
  return { line, circle, rnd };
}

// Fixed shapes in a 100x100 box, scaled to whatever they overlay.
export const circlePath = (seed: string) => generators(seed).circle(50, 50, 47, 44);
export const checkPath = (seed: string) => {
  const { line } = generators(seed);
  return `${line(12, 52, 40, 84, 2)} ${line(40, 84, 90, 14, 4)}`;
};
export const crossPath = (seed: string) => {
  const { line } = generators(seed);
  return `${line(16, 16, 84, 84, 4)} ${line(84, 16, 16, 84, 4)}`;
};
export const hatchPath = (seed: string) => {
  const { line } = generators(seed);
  return [0, 25, 50, 75, 100, 125].map((x) => line(x - 25, 100, x + 15, 0, 2)).join(" ");
};

/** An SVG overlay stretched over its (relative) parent; draws `d` once on mount.
 *  The wrapper span takes the inset classes (an <svg> would keep its intrinsic size). */
export function Ink({
  d,
  className = "",
  width = 2.5,
  clip = false,
}: {
  d: string;
  className?: string;
  width?: number;
  clip?: boolean;
}) {
  return (
    <span aria-hidden="true" className={`pointer-events-none absolute ${className}`}>
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        className={`block h-full w-full ${clip ? "overflow-hidden" : "overflow-visible"}`}
      >
        <path
          d={d}
          pathLength={1}
          className="ink-stroke draw"
          style={{ strokeWidth: width, vectorEffect: "non-scaling-stroke" }}
        />
      </svg>
    </span>
  );
}

/** Inline icon-size stroke (check or cross) that draws itself. */
export function InkIcon({ d, label, className = "h-[1em] w-[1em]" }: { d: string; label?: string; className?: string }) {
  return (
    <svg
      viewBox="0 0 100 100"
      className={`inline-block shrink-0 overflow-visible ${className}`}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <path d={d} pathLength={1} className="ink-stroke draw" style={{ strokeWidth: 12 }} />
    </svg>
  );
}

/** Highlighter swipe behind ink text. `mark` is one of the highlighter CSS vars. */
export function Swipe({ mark, children }: { mark: string; children: ReactNode }) {
  return (
    <span className="swipe" style={{ "--mark": mark } as CSSProperties}>
      {children}
    </span>
  );
}

/** Sticky note: pastel fill, tilt and tape. */
export function Note({
  note,
  tilt = 0,
  className = "",
  children,
  as: Tag = "div",
}: {
  note: string;
  tilt?: number;
  className?: string;
  children: ReactNode;
  as?: "div" | "section" | "article" | "li";
}) {
  return (
    <Tag className={`note ${className}`} style={{ "--note": note, "--tilt": `${tilt}deg` } as CSSProperties}>
      {children}
    </Tag>
  );
}

/** Ink stamp for KEEP_CLOSED: lands with a short thud. */
export function Stamp({ children, rotate = -4, big = false }: { children: ReactNode; rotate?: number; big?: boolean }) {
  return (
    <span
      className={`stamp hl-chip inline-block self-start border-[3px] border-[var(--on-hl)] font-mono font-bold tracking-wide max-w-full ${big ? "px-3 py-1.5 text-base sm:text-xl" : "px-2.5 py-1 text-sm"}`}
      style={{ "--mark": "var(--hl-pink)", rotate: `${rotate}deg` } as CSSProperties}
    >
      {children}
    </span>
  );
}

/** Small tilt that alternates by index so a grid of notes looks pinned by hand. */
export const tiltFor = (i: number) => [-1, 0.7, -0.6, 1.1, -0.9, 0.6][i % 6];
