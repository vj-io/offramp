import type { CaseSummary, Decision, ReferenceAudit, RunConfig, Stage, TaskType, Tier, TierResult, Verdict } from "./types";

export const TIER_NAME: Record<Tier, string> = {
  lightning: "Nemotron 3.5 Lightning",
  super: "Nemotron 3 Super",
  ultra: "Nemotron 3 Ultra",
};
export const TIER_SHORT: Record<Tier, string> = { lightning: "Lightning", super: "Super", ultra: "Ultra" };

export const STAGE_NAME: Record<Stage, string> = {
  QUEUED: "Queued",
  CLONED: "Cloned",
  SCANNED: "Scanned",
  BASELINED: "References",
  EVALUATING: "Evaluating",
  ITERATING: "Iterating",
  VERDICTS: "Verdicts",
  PATCHED: "Patched",
  PR_OPENED: "PR opened",
  DONE: "Done",
  FAILED: "Failed",
};

// The stepper's label: a finished run that opened no PR says so (stageProgress marks that stage "skipped").
export const stageLabel = (stage: Stage, skipped: boolean) =>
  skipped && stage === "PR_OPENED" ? "PR not opened" : STAGE_NAME[stage];

export const DECISION_NAME: Record<Decision, string> = {
  MIGRATE: "Migrate",
  KEEP_CLOSED: "Not migrated",
  MANUAL_REVIEW: "Manual review",
};

export const TASK_NAME: Record<TaskType, string> = {
  classification: "classification",
  json_extraction: "JSON extraction",
  tool_call: "tool call",
  free_text: "free text",
  reasoning: "reasoning",
};

export const pct = (fraction: number) => `${(fraction * 100).toFixed(1)}%`;
export const pct100 = (value: number) => `${value.toFixed(1)}%`;
export const usd = (n: number) => `$${n > 0 && n < 0.01 ? n.toFixed(4) : n.toFixed(2)}`;
export const ms = (n: number) => (Math.abs(n) >= 1000 ? `${(n / 1000).toFixed(1)} s` : `${Math.round(n)} ms`);
export const signedMs = (n: number) => `${n > 0 ? "+" : n < 0 ? "−" : "±"}${ms(Math.abs(n))}`;

// TierResult.score is a parity fraction, except free_text where it is the judge mean on 1..5.
export const scoreText = (r: TierResult, task: TaskType | undefined) =>
  task === "free_text" ? `${r.score.toFixed(1)}/5` : pct(r.score);
export const scoreFraction = (r: TierResult, task: TaskType | undefined) =>
  task === "free_text" ? r.score / 5 : r.score;

// The pass bar the run applied, or null when this run's config is not known (never a guess).
export function thresholdFor(task: TaskType | undefined, config: RunConfig | null): number | null {
  if (!config || !task || task === "free_text") return null;
  return task === "reasoning" ? config.reasoning_threshold : config.threshold;
}

export function keepClosedStamp(v: Verdict, task: TaskType | undefined, config: RunConfig | null): string {
  if (v.parity_pct === null) return "NOT MIGRATED";
  const bar = thresholdFor(task, config);
  return bar === null
    ? `NOT MIGRATED · best parity ${pct100(v.parity_pct)}`
    : `NOT MIGRATED · ${pct100(v.parity_pct)} < ${Math.round(bar * 100)}%`;
}

// Heatmap tint: lime when the result passed, pink otherwise, deeper the further below the bar.
// Mixed with white, not paper, so ink #0a0a0a stays readable on it in both themes.
// The bar is used only for the tint depth; when the config is unknown the schema defaults set
// the depth scale and no threshold number is shown.
export function parityTint(r: TierResult, task: TaskType | undefined, config: RunConfig | null): string {
  if (r.passed) return "color-mix(in srgb, var(--hl-lime) 70%, #fff)";
  const bar = task === "free_text" ? 0.8 : (thresholdFor(task, config) ?? (task === "reasoning" ? 0.9 : 0.95));
  const depth = Math.min(1, Math.max(0, (bar - scoreFraction(r, task)) / 0.3));
  return `color-mix(in srgb, var(--hl-pink) ${Math.round(30 + depth * 70)}%, #fff)`;
}

// p50 latency cell. "Before" was never measured under Amendment A (delta null): show "—", never a guess.
export function latencyText(deltaMs: number | null | undefined, afterP50: number | null): string {
  if (deltaMs != null && afterP50 != null) return `${ms(afterP50 - deltaMs)} → ${ms(afterP50)}`;
  if (afterP50 != null) return `— → ${ms(afterP50)}`;
  return deltaMs != null ? signedMs(deltaMs) : "—";
}

// $/1k cell. Only a migrated call has an "after": the others stay on their source model, so it shows "—" there.
export function costText(v: Pick<Verdict, "decision" | "cost_before_per_1k" | "cost_after_per_1k">): string {
  const before = v.cost_before_per_1k;
  if (before == null) return "—";
  if (v.decision !== "MIGRATE") return `${usd(before)} → —`;
  return v.cost_after_per_1k != null ? `${usd(before)} → ${usd(v.cost_after_per_1k)}` : `${usd(before)}, unchanged`;
}

// "How parity was measured", labelled exactly as the run reports it; absent parts are omitted.
// reference_methods keys are the labels the PR uses too: "reference:*" methods, and anything else is a closed
// model whose answers stand in for references (GC27a, free text).
const isClosed = (method: string) => !method.startsWith("reference:");

export function referencesText(c: CaseSummary): string {
  const entries = Object.entries(c.reference_methods ?? {});
  if (entries.length > 0 && entries.every(([m]) => isClosed(m)))
    return `${c.references} closed references (${entries.map(([m]) => m).join(", ")})`;
  const methods = entries.map(([m, n]) => `${isClosed(m) ? `closed ${m}` : m} ${n}`);
  return `${c.references} references${methods.length ? ` (${methods.join(", ")})` : ""}`;
}

export function auditText(a: ReferenceAudit | undefined): string | null {
  if (!a || (a.checked === 0 && a.human_verified === 0)) return null;
  const counts = `${a.checked} checked, ${a.agreed} agreed, ${a.dropped} dropped, ${a.human_verified} human-verified`;
  return a.auditor ? `${counts}, auditor ${a.auditor}` : counts;
}

// GC27(c): the closed reference's parity on the same references, only when the verdict carries it.
export function closedReferenceText(v: Verdict): string | null {
  if (!v.closed_reference_model || v.closed_reference_parity_pct == null) return null;
  return `${v.closed_reference_model} scores ${pct100(v.closed_reference_parity_pct)} on the same references`;
}

// GC27(c) when the closed model also audited the site's references: the verdict names it with no parity,
// because its agreement with the references it kept is by construction. A note, never a comparison row.
export function closedByConstructionText(v: Verdict): string | null {
  if (!v.closed_reference_model || v.closed_reference_parity_pct != null) return null;
  return `${v.closed_reference_model} is not compared: it audited these references, so it agrees with every one kept by construction`;
}

// Split a note into text and http(s) links; anything else (bare domains) stays plain text.
export function linkParts(text: string): { text: string; href?: string }[] {
  return text
    .split(/(https?:\/\/[^\s,)]+)/)
    .filter(Boolean)
    .map((part) => (/^https?:\/\//.test(part) ? { text: part, href: part } : { text: part }));
}
