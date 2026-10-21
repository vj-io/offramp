import { ImageResponse } from "next/og";
import { headline } from "@/lib/format";
import { applyEvents, initialRunState, parseRecording, toSnapshot } from "@/lib/run-state";
import recording from "@/public/replay/demo-events.json";

// The link-preview card, drawn at build time in the dashboard's doodle look. Its numbers come from the recorded
// demo run (bundled at build) through the report's own headline(), so they match what the report shows.
export const alt = "Offramp: which closed-LLM calls can move to NVIDIA Nemotron on Nebius Token Factory";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const INK = "#0a0a0a";
const PAPER = "#f0ede8";
const LIME = "#c8ff00";

function demoLead(): string[] | null {
  const snap = toSnapshot(applyEvents(initialRunState, parseRecording(recording as unknown)?.events ?? []));
  return snap.verdicts.length ? headline(snap.verdicts, snap.totals).lead.split(" · ") : null;
}

// Google Fonts, subset to the card's text, as next/font fetches them at build; the default font if offline.
// "use cache" keeps the fetches in the build's prerender, so the card stays a static file.
async function font(family: string, weight: number, text: string) {
  "use cache";
  try {
    const q = `family=${family.replaceAll(" ", "+")}:wght@${weight}&text=${encodeURIComponent(text)}`;
    const css = await (await fetch(`https://fonts.googleapis.com/css2?${q}`)).text();
    const url = css.match(/src: url\((.+?)\) format\('(opentype|truetype)'\)/)?.[1];
    if (!url) return null;
    return { name: family, data: await (await fetch(url)).arrayBuffer(), weight: weight as 400 | 800, style: "normal" as const };
  } catch {
    return null;
  }
}

const sticker = (rotate: number) => ({
  display: "flex",
  padding: "4px 14px",
  border: `3px solid ${INK}`,
  borderRadius: "255px 15px 225px 15px / 15px 225px 15px 255px",
  background: "#faf8f4",
  fontSize: 30,
  transform: `rotate(${rotate}deg)`,
});

export default async function Image() {
  const lead = demoLead() ?? ["Which closed-LLM calls can move to Nemotron?"];
  const tagline = "Held-out evals first, then a pull request.";
  const stickers = ["Nebius Token Factory", "NVIDIA Nemotron"];
  const hand = [...lead, tagline, ...stickers].join(" ");
  const fonts = (await Promise.all([font("Archivo", 800, "Offramp"), font("Patrick Hand", 400, hand)])).filter(
    (f) => f !== null,
  );
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "56px 72px",
          color: INK,
          backgroundColor: PAPER,
          backgroundImage:
            "linear-gradient(rgba(10,10,10,0.07) 2px, transparent 2px), linear-gradient(90deg, rgba(10,10,10,0.07) 2px, transparent 2px)",
          backgroundSize: "60px 60px",
          fontFamily: "Patrick Hand",
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-end", gap: 16 }}>
          <span style={{ fontFamily: "Archivo", fontWeight: 800, fontSize: 64, letterSpacing: -2 }}>Offramp</span>
          <svg width="96" height="48" viewBox="0 0 60 30" style={{ marginBottom: 10 }}>
            <path d="M2 26 L34 26 Q46 26 52 8" fill="none" stroke={INK} strokeWidth="3.5" strokeLinecap="round" />
            <path d="M44 9 L52 6 L55 14" fill="none" stroke={INK} strokeWidth="3.5" strokeLinecap="round" />
          </svg>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", fontSize: 66, lineHeight: 1.15 }}>
          {lead.map((part, i) => (
            <div key={i} style={{ display: "flex" }}>
              <span
                style={
                  part.startsWith("est. ")
                    ? { background: LIME, padding: "0 10px", borderRadius: "6px 18px 8px 16px" }
                    : {}
                }
              >
                {part}
              </span>
              {i < lead.length - 1 && <span style={{ color: "#5f5c57", padding: "0 14px" }}>·</span>}
            </div>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontSize: 34, color: "#5f5c57" }}>{tagline}</span>
          <div style={{ display: "flex", gap: 14 }}>
            <span style={sticker(-2)}>{stickers[0]}</span>
            <span style={sticker(1.5)}>{stickers[1]}</span>
          </div>
        </div>
      </div>
    ),
    { ...size, fonts },
  );
}
