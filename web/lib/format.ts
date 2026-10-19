import type { Decision, RunConfig, Stage, TaskType, Tier, TierResult, Verdict } from "./types";

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
  BASELINED: "Baselined",
  EVALUATING: "Evaluating",
  ITERATING: "Iterating",
  VERDICTS: "Verdicts",
  PATCHED: "Patched",
  PR_OPENED: "PR opened",
  DONE: "Done",
  FAILED: "Failed",
};

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
