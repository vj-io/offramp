import type { CSSProperties, ReactNode } from "react";
import Link from "next/link";
import type { RunSource } from "@/lib/run-state";
import { Note } from "./doodle";

/** Says where the numbers on screen come from. Fixture data is always flagged. */
export function SourceLabel({ source, runId, apiUrl }: { source: RunSource; runId: string | null; apiUrl: string }) {
  if (source === "fixture")
    return (
      <p
        className="hl-chip sketch inline-block self-start px-3 py-1 font-hand text-lg"
        style={{ "--mark": "var(--hl-orange)" } as CSSProperties}
      >
        Fixture data: made-up numbers for building the dashboard, not a measured run.
      </p>
    );
  if (source === "replay")
    return (
      <p className="font-hand text-lg text-muted">
        Replay of recorded run <span className="font-mono text-sm text-ink">{runId ?? "…"}</span>, played at 4× speed.
      </p>
    );
  return (
    <p className="font-hand text-lg text-muted">
      Live from <span className="font-mono text-sm text-ink">{apiUrl}</span>
    </p>
  );
}

export function NoticeNote({ title, children, tone = "plain" }: { title: string; children?: ReactNode; tone?: "plain" | "warn" }) {
  return (
    <Note note={tone === "warn" ? "var(--note-orange)" : "var(--note-plain)"} tilt={-0.6} className="max-w-xl p-5 pt-6">
      <h2 className="text-2xl font-extrabold tracking-tight">{title}</h2>
      {children && <div className="mt-2 font-hand text-lg leading-snug">{children}</div>}
    </Note>
  );
}

export function NoRecording() {
  return (
    <NoticeNote title="No recorded demo run yet">
      <p>
        The demo replay plays back a real Offramp run, and that run has not been recorded. Start a run on your own
        repo instead, or check back once the recording is published.
      </p>
      <p className="mt-3">
        <Link href="/" className="text-bluepen underline underline-offset-4">
          Start a run
        </Link>
      </p>
    </NoticeNote>
  );
}
