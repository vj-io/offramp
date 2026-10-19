import type { CSSProperties } from "react";
import { linkParts } from "@/lib/format";

// Hand-lettered sticker badges, text only (no copied logos).
export function BrandBadges({ small = false }: { small?: boolean }) {
  const size = small ? "px-1.5 py-0 text-[0.8rem]" : "px-2.5 py-0.5 text-base";
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <span className={`sketch bg-note-plain font-hand leading-snug text-ink ${size}`} style={{ rotate: "-1.5deg" }}>
        Nebius Token Factory
      </span>
      <span className={`sketch bg-note-plain font-hand leading-snug text-ink ${size}`} style={{ rotate: "1.2deg" }}>
        NVIDIA Nemotron
      </span>
    </span>
  );
}

/** Plain text with only its http(s) URLs turned into links. */
export function LinkedText({ text }: { text: string }) {
  return (
    <>
      {linkParts(text).map((p, i) =>
        p.href ? (
          <a key={i} href={p.href} target="_blank" rel="noreferrer" className="break-all text-bluepen underline underline-offset-2">
            {p.text}
          </a>
        ) : (
          <span key={i}>{p.text}</span>
        ),
      )}
    </>
  );
}

/** GC21: deprecation notice for the source model. Orange = warning; the words carry the meaning. */
export function SunsetRisk({ note }: { note: string }) {
  return (
    <span className="flex flex-col items-start gap-1">
      <span className="hl-chip sketch px-2 font-hand text-base" style={{ "--mark": "var(--hl-orange)" } as CSSProperties}>
        Sunset risk
      </span>
      <span className="font-hand text-base leading-snug">
        <LinkedText text={note} />
      </span>
    </span>
  );
}
