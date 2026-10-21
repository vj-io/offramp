import type { CSSProperties, ReactNode } from "react";
import Link from "next/link";
import { prLabel, repoName, TIER_SHORT } from "@/lib/format";
import { replayEntry, type RunSource } from "@/lib/run-state";
import { TIER_ORDER, type Tier } from "@/lib/types";
import { Note } from "./doodle";

/** Says where the numbers on screen come from. Fixture data is always flagged. */
export function SourceLabel({
  source,
  runId,
  apiUrl,
  report = false,
}: {
  source: RunSource;
  runId: string | null;
  apiUrl: string;
  report?: boolean;
}) {
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
        {report ? "Report of" : "Replay of"} recorded run <span className="font-mono text-sm text-ink wrap-anywhere">{runId ?? "…"}</span>
      </p>
    );
  return (
    <p className="font-hand text-lg text-muted">
      Live from <span className="font-mono text-sm text-ink">{apiUrl}</span>
    </p>
  );
}

/** The target repo and the run's pull request, as links; nothing until the run has recorded them. */
export function RunLinks({ repoUrl, prUrl }: { repoUrl: string | null; prUrl: string | null }) {
  if (!repoUrl && !prUrl) return null;
  const link = "text-bluepen underline underline-offset-4";
  return (
    <p className="flex flex-wrap items-baseline gap-x-3 gap-y-1 font-hand text-lg">
      {repoUrl && (
        <span className="min-w-0 wrap-anywhere">
          Repo{" "}
          {/^https:\/\//.test(repoUrl) ? (
            <a href={repoUrl} target="_blank" rel="noreferrer" className={`font-mono text-sm ${link}`}>
              {repoName(repoUrl)}
            </a>
          ) : (
            <span className="font-mono text-sm">{repoName(repoUrl)}</span>
          )}
        </span>
      )}
      {prUrl && (
        <a href={prUrl} target="_blank" rel="noreferrer" className={link}>
          {prLabel(prUrl)}
        </a>
      )}
    </p>
  );
}

/** The tier policy a recording ran under: the allow-list's words for it, and the tiers its config allowed. */
export function PolicyNote({ replay, tiers }: { replay: string | null; tiers: Tier[] | null }) {
  const policy = replayEntry(replay)?.policy;
  const capped = tiers && tiers.length < TIER_ORDER.length ? tiers : null;
  if (!policy && !capped) return null;
  return (
    <p
      className="hl-chip sketch inline-block self-start px-3 py-1 font-hand text-lg"
      style={{ "--mark": "var(--hl-violet)" } as CSSProperties}
    >
      {policy && <>Team policy: {policy}. </>}
      {capped && <>Tiers this run could use: {capped.map((t) => TIER_SHORT[t]).join(", ")}.</>}
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

export function NoRecording({ replay }: { replay: string | null }) {
  const policy = replayEntry(replay)?.policy;
  return (
    <NoticeNote title="This recording isn't published yet">
      {policy && <p className="mb-2">Team policy: {policy}.</p>}
      <p>
        This page plays back a real Offramp run, and that run has not been recorded and published here yet. Check
        back soon{replay !== "demo" ? ", or watch the recorded demo run meanwhile" : ""}.
      </p>
      <p className="mt-3 flex flex-wrap gap-x-4">
        {replay !== "demo" && (
          <Link href="/runs/demo?replay=demo" className="text-bluepen underline underline-offset-4">
            Watch the demo run
          </Link>
        )}
        <Link href="/" className="text-bluepen underline underline-offset-4">
          Home
        </Link>
      </p>
    </NoticeNote>
  );
}
