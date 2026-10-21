import type { CSSProperties } from "react";
import { DECISION_NAME, pct100, resultBounds, scoreText, STAGE_NAME, TIER_NAME, TIER_SHORT } from "@/lib/format";
import type {
  Adapter,
  CallSite,
  CaseSummary,
  RunEvent,
  Stage,
  TierResult,
  Verdict,
} from "@/lib/types";

const SHOWN = 200;

function describe(e: RunEvent, sites: Record<string, CallSite>): string {
  const p = e.payload;
  switch (e.type) {
    case "stage":
      return `Stage: ${STAGE_NAME[p.stage as Stage]}${p.detail ? `. ${p.detail}` : ""}`;
    case "site": {
      const s = p as unknown as CallSite;
      return `Found ${s.id} in ${s.file}:${s.lines[0]} (${s.source_model ?? "unknown model"})`;
    }
    case "case_summary": {
      const c = p as unknown as CaseSummary;
      return `${c.site_id}: ${c.dev} dev + ${c.holdout} holdout cases, ${c.references} references`;
    }
    case "tier_result": {
      const r = p as unknown as TierResult;
      const task = sites[r.site_id]?.contract?.task_type;
      const bounds = resultBounds(r, task);
      return `${r.site_id} on ${TIER_SHORT[r.tier]} (${r.split}, adapter v${r.adapter_version}): ${scoreText(r, task)}${bounds ? ` (${bounds})` : ""} ${r.passed ? "pass" : "fail"}`;
    }
    case "iteration": {
      const a = p as unknown as Adapter;
      return `${a.site_id}: adapter v${a.version}${a.tier ? ` for ${TIER_SHORT[a.tier]}` : ""}. ${a.change_log.join("; ")}`;
    }
    case "verdict": {
      const v = p as unknown as Verdict;
      if (v.decision === "MIGRATE" && v.tier) return `${v.site_id}: migrate to ${TIER_NAME[v.tier]}`;
      return `${v.site_id}: ${DECISION_NAME[v.decision].toLowerCase()}${v.parity_pct !== null ? `, best parity ${pct100(v.parity_pct)}` : ""}`;
    }
    case "pr":
      return `Pull request opened: ${p.title}`;
    case "log":
      return String(p.message);
    case "error":
      return `Failed at ${STAGE_NAME[p.stage as Stage] ?? p.stage}: ${p.reason}`;
    case "done":
      return "Run finished";
  }
}

function level(e: RunEvent): "info" | "warn" | "error" {
  if (e.type === "error") return "error";
  if (e.type === "log" && (e.payload.level === "warn" || e.payload.level === "error")) return e.payload.level;
  return "info";
}

// `quiet` while a replay plays: a hundred events in seconds would flood a screen reader's live region.
export function LogStream({ events, sites, quiet = false }: { events: RunEvent[]; sites: Record<string, CallSite>; quiet?: boolean }) {
  const recent = events.slice(-SHOWN).reverse();
  if (recent.length === 0) return <p className="font-hand text-lg text-muted">Nothing logged yet.</p>;
  return (
    <ol role="log" aria-live={quiet ? "off" : "polite"} tabIndex={0} aria-label="Run log, newest first" className="max-h-96 overflow-y-auto pr-1 font-mono text-[0.8rem] leading-relaxed">
      {recent.map((e) => {
        const lvl = level(e);
        return (
          <li key={e.seq} className="flex gap-2 border-b border-dashed border-muted/40 py-1">
            <time dateTime={e.ts} className="shrink-0 text-bluepen">
              {e.ts.slice(11, 19)}
            </time>
            {lvl !== "info" && (
              <span
                className="hl-chip h-fit shrink-0 px-1 font-hand text-sm"
                style={{ "--mark": lvl === "warn" ? "var(--hl-orange)" : "var(--hl-pink)" } as CSSProperties}
              >
                {lvl}
              </span>
            )}
            <span className={`min-w-0 break-words ${lvl === "error" ? "text-redpen" : ""}`}>{describe(e, sites)}</span>
          </li>
        );
      })}
    </ol>
  );
}
