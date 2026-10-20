// Folds the run's SSE events (Global Constraints 12-14) into dashboard state.
// Every id-bearing event is an upsert, and events are deduplicated by `seq`, so a reconnect that
// replays the stream from the start, or a producer that re-emits a result, never duplicates UI.
import {
  STAGE_ORDER,
  TIER_ORDER,
  type Adapter,
  type CallSite,
  type CaseSummary,
  type RunConfig,
  type RunEvent,
  type RunReport,
  type RunSnapshot,
  type RunTotals,
  type Stage,
  type Tier,
  type TierResult,
  type Verdict,
} from "./types";

export interface RunState {
  runId: string | null;
  lastSeq: number;
  stage: Stage | null;
  lastActiveStage: Stage | null; // last stage before FAILED
  failure: { stage: Stage; reason: string } | null;
  config: RunConfig | null; // from the QUEUED stage payload (GC13); null for older recordings
  sites: Record<string, CallSite>; // insertion order = scanner order
  cases: Record<string, CaseSummary>;
  tierResults: TierResult[]; // latest emission last
  iterations: Adapter[];
  verdicts: Record<string, Verdict>;
  pr: { url: string; title: string } | null;
  totals: RunTotals | null;
  events: RunEvent[];
}

export const initialRunState: RunState = {
  runId: null,
  lastSeq: -1,
  stage: null,
  lastActiveStage: null,
  failure: null,
  config: null,
  sites: {},
  cases: {},
  tierResults: [],
  iterations: [],
  verdicts: {},
  pr: null,
  totals: null,
  events: [],
};

const resultKey = (r: TierResult) => `${r.site_id}|${r.tier}|${r.adapter_version}|${r.split}`;
const adapterKey = (a: Adapter) => `${a.site_id}|${a.tier}|${a.version}`;

function upsertLast<T>(list: T[], item: T, key: (x: T) => string): T[] {
  const k = key(item);
  return [...list.filter((x) => key(x) !== k), item];
}

export function applyEvent(s: RunState, e: RunEvent): RunState {
  if (e.seq <= s.lastSeq) return s;
  const p = e.payload;
  const next: RunState = { ...s, runId: s.runId ?? e.run_id, lastSeq: e.seq, events: [...s.events, e] };
  const stage = (e.type === "stage" ? (p.stage as Stage) : e.stage) ?? s.stage;
  next.stage = stage;
  if (stage !== "FAILED") next.lastActiveStage = stage;

  switch (e.type) {
    case "stage":
      if (p.config && typeof p.config === "object") next.config = p.config as RunConfig;
      if (stage === "FAILED" && !s.failure)
        next.failure = { stage: s.lastActiveStage ?? "QUEUED", reason: String(p.detail ?? "Run failed") };
      break;
    case "site": {
      const site = p as unknown as CallSite;
      next.sites = { ...s.sites, [site.id]: site };
      break;
    }
    case "case_summary": {
      const c = p as unknown as CaseSummary;
      next.cases = { ...s.cases, [c.site_id]: c };
      break;
    }
    case "tier_result":
      next.tierResults = upsertLast(s.tierResults, p as unknown as TierResult, resultKey);
      break;
    case "iteration": {
      const a = { ...(p as unknown as Adapter) };
      a.site_id ??= e.call_site_id ?? "";
      next.iterations = upsertLast(s.iterations, a, adapterKey);
      break;
    }
    case "verdict": {
      const v = p as unknown as Verdict;
      next.verdicts = { ...s.verdicts, [v.site_id]: v };
      break;
    }
    case "pr":
      next.pr = { url: String(p.url), title: String(p.title ?? "") };
      break;
    case "error":
      next.stage = "FAILED";
      next.failure = { stage: (p.stage as Stage) ?? s.lastActiveStage ?? "QUEUED", reason: String(p.reason ?? "") };
      break;
    case "done":
      next.totals = (p.totals as RunTotals) ?? null;
      break;
  }
  return next;
}

export const applyEvents = (s: RunState, es: RunEvent[]) => es.reduce(applyEvent, s);

export type StageStatus = "done" | "current" | "pending" | "failed" | "skipped";

// The stage names are past tense (CLONED = clone finished): the latest reached stage is
// "current", everything before it is done. A run that finished without a `pr` event opened no PR
// (--no-pr, or nothing to commit): that stage is "skipped", never ticked.
export function stageProgress(s: RunState): { stage: Stage; status: StageStatus }[] {
  if (s.stage === "DONE")
    return STAGE_ORDER.map((stage) => ({ stage, status: stage === "PR_OPENED" && !s.pr ? "skipped" : "done" }));
  const failed = s.stage === "FAILED";
  const at = STAGE_ORDER.indexOf((failed ? s.failure?.stage : s.stage) as (typeof STAGE_ORDER)[number]);
  return STAGE_ORDER.map((stage, i) => ({
    stage,
    status: i < at ? "done" : i === at ? (failed ? "failed" : "current") : "pending",
  }));
}

export interface SiteView {
  site: CallSite;
  cases: CaseSummary | null;
  perTier: Record<Tier, TierResult | null>; // latest result per tier
  currentTier: Tier | null;
  iterations: number;
  verdict: Verdict | null;
}

export function siteView(s: RunState, siteId: string): SiteView {
  const results = s.tierResults.filter((r) => r.site_id === siteId);
  const perTier = Object.fromEntries(
    TIER_ORDER.map((t) => [t, results.findLast((r) => r.tier === t) ?? null]),
  ) as Record<Tier, TierResult | null>;
  const verdict = s.verdicts[siteId] ?? null;
  const currentTier = verdict?.tier ?? verdict?.best_tier ?? results.at(-1)?.tier ?? null;
  // The router iterates on each tier it tries; like Verdict.iterations, count the current tier only.
  const onTier = s.iterations.filter((a) => a.site_id === siteId && a.tier === currentTier && a.version > 0);
  return {
    site: s.sites[siteId],
    cases: s.cases[siteId] ?? null,
    perTier,
    currentTier,
    iterations: verdict ? verdict.iterations : onTier.length,
    verdict,
  };
}

export function toSnapshot(s: RunState): RunSnapshot {
  return {
    run_id: s.runId ?? "",
    config: s.config,
    stage: s.stage ?? "QUEUED",
    sites: Object.values(s.sites),
    tier_results: s.tierResults,
    verdicts: Object.values(s.verdicts),
    totals: s.totals,
    artifacts: s.pr ? { pr_url: s.pr.url } : {},
    case_summaries: Object.values(s.cases),
  };
}

// Global Constraint 14: 4x speed, each recorded gap capped at 1.5 s before the speed-up.
export function replayDelays(es: RunEvent[]): number[] {
  return es.map((e, i) =>
    i === 0 ? 0 : Math.min(Math.max(0, Date.parse(e.ts) - Date.parse(es[i - 1].ts)), 1500) / 4,
  );
}

/** Events from a replay file (array or {events}). Fixture data is refused (null), so the dev
 *  fixture can never play as the recorded demo even if someone copies it to /replay/. */
export function parseRecording(body: unknown): RunEvent[] | null {
  const events = Array.isArray(body) ? body : (body as { events?: unknown })?.events;
  if (!Array.isArray(events)) throw new Error("The replay file is not a recorded Offramp event log.");
  const isFixture =
    (body as { fixture?: unknown }).fixture === true ||
    events.some((e: Partial<RunEvent>) => String(e?.run_id ?? "").startsWith("fixture-"));
  return isFixture ? null : (events as RunEvent[]);
}

export type RunSource = "api" | "replay" | "fixture";

/** What "Download JSON" saves: the report, marked when it was built from fixture data. */
export const downloadPayload = (report: RunReport, source: RunSource) =>
  source === "fixture" ? { fixture: true, ...report } : report;

export function pickSource(q: URLSearchParams): RunSource {
  if (q.get("live") === "1") return "api";
  if (q.get("replay") === "demo") return "replay";
  if (q.get("fixture") === "dev") return "fixture";
  return "api";
}
