import tables from "./offramp-tables.json";
import type {
  CallSite,
  CaseSummary,
  Decision,
  NotEvaluated,
  ReferenceAudit,
  RunConfig,
  RunTotals,
  Stage,
  TaskType,
  Tier,
  TierResult,
  Verdict,
} from "./types";

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

// "9/9, ≥70%": holdout cases passing / evaluated and the Wilson 95% lower bound of that rate, rounded down so it
// is never overstated (as the PR writes it). JSON extraction's figure is a field match, so its count says "pass".
// null for recordings made before Task 19, which keep the figure alone.
export function boundsText(
  passed: number | null | undefined,
  cases: number | null | undefined,
  lower: number | null | undefined,
  task?: TaskType,
): string | null {
  if (!cases || passed == null || lower == null) return null;
  return `${passed}/${cases}${task === "json_extraction" ? " pass" : ""}, ≥${Math.floor(lower)}%`;
}
export const verdictBounds = (v: Verdict, task?: TaskType) =>
  boundsText(v.holdout_passed, v.holdout_cases, v.parity_lower_pct, task);
export const resultBounds = (r: TierResult, task?: TaskType) =>
  task === "free_text"
    ? r.cases
      ? `n=${r.cases}`
      : null
    : boundsText(r.cases != null ? r.cases - r.failing_case_ids.length : null, r.cases, r.parity_lower_pct, task);

// "100.0% (9/9, ≥70%)", or "100.0%" when the recording has no counts.
export function parityText(v: Verdict, task?: TaskType): string {
  if (v.parity_pct == null) return "—";
  const b = verdictBounds(v, task);
  return b ? `${pct100(v.parity_pct)} (${b})` : pct100(v.parity_pct);
}

// Free text: "mean 4.54, lowest 3.0, n=8"; with `failed`, each figure under its bar says so ("mean 3.95 < 4.0").
// Two decimals, never one: rounding 2.96 to "3.0" would hide a miss of the 3.0 floor.
const judgeNum = (x: number) => x.toFixed(2).replace(/0$/, "");
export function judgeText(
  v: Pick<Verdict, "judge_mean" | "judge_min" | "holdout_cases">,
  failed = false,
): string | null {
  if (v.judge_mean == null) return null;
  const fig = (name: string, x: number, bar: number) =>
    `${name} ${judgeNum(x)}${failed && x < bar ? ` < ${bar.toFixed(1)}` : ""}`;
  const parts = [fig("mean", v.judge_mean, tables.judge_floor)];
  if (v.judge_min != null) parts.push(fig("lowest", v.judge_min, tables.judge_case_floor));
  if (v.holdout_cases) parts.push(`n=${v.holdout_cases}`);
  return parts.join(", ");
}

// A kept site's figure and only the cause its verdict recorded (failed_gate, ported from pr_writer._kept). A
// verdict recorded without the gate states its figures and no comparison: a cause is never inferred.
export function keptText(v: Verdict, task: TaskType | undefined, config: RunConfig | null): string {
  if (v.failed_gate === "budget") {
    // holdout figures only when the best adapter's holdout ran (and failed) before the budget stopped the run
    if (v.parity_pct == null) return "not measured: the run's budget stopped it";
    const figures = (task === "free_text" ? judgeText(v, true) : null) ?? parityText(v, task);
    return `${figures}: holdout failed; the run's budget stopped the rest`;
  }
  if (v.parity_pct == null) return "not measured";
  const judged = task === "free_text" && (v.failed_gate === "judge_mean" || v.failed_gate === "case_floor");
  const text = (task === "free_text" ? judgeText(v, judged) : null) ?? parityText(v, task);
  const missed = v.failed_gate ? (tables.missed as Record<string, string>)[v.failed_gate] : undefined;
  if (missed) return text + missed;
  if (v.failed_gate !== "threshold") return text;
  const bar = thresholdFor(task, config);
  return bar === null ? `${text}, below the parity bar` : `${text} < ${Math.round(bar * 100)}%`;
}

export const keepClosedStamp = (v: Verdict, task: TaskType | undefined, config: RunConfig | null) =>
  v.parity_pct == null && v.failed_gate !== "budget" ? "NOT MIGRATED" : `NOT MIGRATED · ${keptText(v, task, config)}`;

// A manual-review verdict as the sentence the PR gives it (pr_writer.manual_reason, same tables), plus the raw
// reason to fold under it. Built from the patch stage's recorded failure kind, never from the reason's text;
// without that record (older recordings) the reason shows as it is and raw is null.
export function manualReason(v: Verdict): { sentence: string; raw: string | null } {
  const f = v.patch_failure;
  if (!f) return { sentence: v.reason, raw: null };
  const shown = f.last ?? f.kind;
  const alone = tables.alone as Record<string, string>;
  const table = !f.model_edit && shown in alone ? alone : (tables.cause as Record<string, string>);
  const tests = f.tests ?? [];
  const files = (f.files ?? []).map((p) => `\`${p}\``).join(", ") || "a file";
  const named = `the test${tests.length !== 1 ? "s" : ""} ` + tests.map((t) => `\`${t}\``).join(", ");
  // Function replacers: a name holding "$&" or "$1" must come out as written, as Python's str.format leaves it.
  const cause = (table[shown] ?? "").replaceAll("{files}", () => files).replaceAll("{tests}", () => named);
  const ended = (tables.ended as Record<string, string>)[f.kind];
  let sentence: string;
  if (ended) {
    const n = f.tries ?? 0;
    const tries = n ? `after ${n} ${n === 1 ? "try" : "tries"}` : "before the model replied";
    sentence = ended.replace("{tries}", () => tries) + (cause ? `; ${cause}` : "");
  } else sentence = cause.charAt(0).toUpperCase() + cause.slice(1);
  return { sentence: `${sentence}.`, raw: v.reason };
}

// A manual-review note for the page: the PR's sentence when the patch stage recorded its failure kind. A
// recording without the kind (made before Task 18) shows a short reason as it is; a long one (a check dump) goes
// behind the fold under one line that claims only what any patch-stage downgrade means.
const SHORT_REASON = 160;
export function reviewNote(v: Verdict): { sentence: string; raw: string | null } {
  const port = manualReason(v);
  if (port.raw !== null || (v.reason.length <= SHORT_REASON && !v.reason.includes("\n"))) return port;
  const sentence =
    v.tier != null
      ? "Offramp's automatic patch for this site failed, so it shipped no change."
      : "Offramp stopped at this site without a patch.";
  return { sentence, raw: v.reason };
}

export const PRICE_METHOD: Record<string, string> = tables.price_methods;
// pr_writer.BOUNDS: what the bracket after a parity figure means, and how to narrow it.
export const BOUNDS_NOTE: string = tables.bounds;

// The pipeline's log line for a call it found but did not evaluate (pipeline.NOT_EVALUATED), read back.
const NOT_EVALUATED = new RegExp(
  "^" +
    tables.not_evaluated
      .replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      .replace("\\{id\\}", "(?<id>\\S+)")
      .replace("\\{kind\\}", "(?<kind>\\S+)")
      .replace("\\{file\\}", "(?<file>.+)") +
    "$",
);
export function parseNotEvaluated(message: string): NotEvaluated | null {
  const g = NOT_EVALUATED.exec(message)?.groups;
  return g ? { id: g.id, kind: g.kind, file: g.file } : null;
}
export const notEvaluatedText = (n: NotEvaluated) => `found, not evaluated (${n.kind} calls aren't chat calls)`;

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

// The two "after" cells of the evidence table. Only a migrated call has an after: the others stay on their source
// model, so latency is "—" and cost keeps its before and shows "—" after.
export function afterCells(
  v: Pick<Verdict, "decision" | "latency_delta_ms" | "cost_before_per_1k" | "cost_after_per_1k">,
  afterP50: number | null,
): { latency: string; cost: string } {
  const before = v.cost_before_per_1k;
  const after = v.cost_after_per_1k;
  const migrated = v.decision === "MIGRATE";
  return {
    latency: migrated ? latencyText(v.latency_delta_ms, afterP50) : "—",
    cost:
      before == null
        ? "—"
        : !migrated
          ? `${usd(before)} → —`
          : after != null
            ? `${usd(before)} → ${usd(after)}`
            : `${usd(before)}, unchanged`,
  };
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

export function auditText(a: ReferenceAudit | undefined, withAuditor = true): string | null {
  if (!a || (a.checked === 0 && a.human_verified === 0)) return null;
  const counts = `${a.checked} checked, ${a.agreed} agreed, ${a.dropped} dropped, ${a.human_verified} human-verified`;
  return a.auditor && withAuditor ? `${counts}, auditor ${a.auditor}` : counts;
}

// The closed model that audited references, said once for every site it audited rather than once per site;
// "It" when the sentence before already named that model (`named`).
export function byConstructionLines(verdicts: Verdict[], named: string | null = null): string[] {
  const groups = new Map<string, string[]>();
  for (const v of verdicts)
    if (closedByConstructionText(v)) groups.set(v.closed_reference_model!, [...(groups.get(v.closed_reference_model!) ?? []), v.site_id]);
  return [...groups].map(
    ([model, ids]) =>
      `${model === named ? "It" : model} is not compared on ${ids.join(", ")}: it audited ${ids.length === 1 ? "that site's" : "those sites'"} references, so it agrees with every one kept by construction`,
  );
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

// Reference QA: free-text references the judge scored below 3. Shown on the report and in the PR, never excluded.
export function weakReferencesText(v: Verdict): string | null {
  const weak = Object.entries(v.weak_references ?? {}).sort(([a], [b]) => a.localeCompare(b));
  if (weak.length === 0) return null;
  const cases = weak.map(([id, score]) => `${id}, ${Number(score.toFixed(2))}/5`).join("; ");
  return `reference judged weak: ${cases} (below 3; kept in the evaluation)`;
}

// Split a note into text and http(s) links; anything else (bare domains) stays plain text.
export function linkParts(text: string): { text: string; href?: string }[] {
  return text
    .split(/(https?:\/\/[^\s,)]+)/)
    .filter(Boolean)
    .map((part) => (/^https?:\/\//.test(part) ? { text: part, href: part } : { text: part }));
}

// ---------------------------------------------------------------- the report's headline

// "−82%" for a saving, "+5%" for a rise; null when nothing to compare.
function change(before: number, after: number): string | null {
  if (!(before > 0)) return null;
  const p = Math.round((1 - after / before) * 100);
  return p > 0 ? `−${p}%` : p < 0 ? `+${-p}%` : "±0%";
}

const plural = (n: number, one: string, many: string) => (n === 1 ? one : many);

/** The report's lead, from the verdicts alone: how many sites reached parity (MIGRATE, or a manual review the
 *  patch stage downgraded, which keeps its passing tier), how many migrated, and the estimated cost change on
 *  those; then the run totals' figure with its scope. The totals sum only the sites with a cost estimate (a
 *  "before", and an "after" too when migrated; pipeline.totals), so the scope names how many those are when
 *  not all. Never hard-coded. */
export function headline(verdicts: Verdict[], totals: RunTotals | null): { lead: string; scope: string | null } {
  const n = verdicts.length;
  const migrated = verdicts.filter((v) => v.decision === "MIGRATE");
  const reached = verdicts.filter((v) => v.decision !== "KEEP_CLOSED" && v.tier != null && v.parity_pct != null);
  const m = migrated.length;
  let lead = `${reached.length} of ${n} call ${plural(n, "site", "sites")} reached parity on Nemotron (held-out cases) · ${m || "none"} migrated`;
  const priced = migrated.every((v) => v.cost_before_per_1k != null && v.cost_after_per_1k != null);
  const sum = (k: "cost_before_per_1k" | "cost_after_per_1k") => migrated.reduce((a, v) => a + (v[k] ?? 0), 0);
  const onMigrated = m && priced ? change(sum("cost_before_per_1k"), sum("cost_after_per_1k")) : null;
  if (onMigrated) lead += ` · est. ${onMigrated} on ${m === n ? `all ${n}` : `${plural(m, "that one", `those ${m}`)}`}`;

  const all = totals ? change(totals.est_cost_before_per_1k, totals.est_cost_after_per_1k) : null;
  if (!totals || !all || m === n) return { lead, scope: null };
  const inTotals = verdicts.filter(
    (v) => v.cost_before_per_1k != null && (v.decision !== "MIGRATE" || v.cost_after_per_1k != null),
  );
  const k = inTotals.length;
  const review = inTotals.filter((v) => v.decision === "MANUAL_REVIEW").length;
  const kept = inTotals.filter((v) => v.decision === "KEEP_CLOSED").length;
  const why = [
    review ? `${review} ${plural(review, "stays", "stay")} for review` : "",
    kept ? `${kept} ${plural(kept, "stays on its source model", "stay on their source models")}` : "",
  ].filter(Boolean);
  const money = `${usd(totals.est_cost_before_per_1k)} → ${usd(totals.est_cost_after_per_1k)} per 1,000 calls, est.`;
  const across = k === n ? `across all ${n}` : `across the ${k} priced ${plural(k, "site", "sites")}`;
  let scope = `${all} ${across} (${money})`;
  if (why.length) scope += `, since ${why.join(" and ")}`;
  if (k < n) scope += `; the other ${n - k} ${plural(n - k, "has", "have")} no cost estimate`;
  return { lead, scope };
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "2026-11-30" -> "30 Nov 2026", read as a calendar date (no time zone shift). */
export const dayText = (iso: string) => {
  const [y, mo, d] = iso.split("-").map(Number);
  return `${d} ${MONTHS[mo - 1]} ${y}`;
};

/** The model's name as its deprecation notice spells it ("Claude Sonnet 4.5" for claude-sonnet-4-5), else the id. */
export function spelledName(model: string, notice: string | null | undefined): string {
  const id = model.split("/").pop() ?? model;
  const parts = id.split(/[-_.\s]+/).filter(Boolean).map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const found = notice?.match(new RegExp(`\\b${parts.join("[\\s._-]+")}(?![\\w]|\\.\\d)`, "i"));
  return found ? found[0] : model;
}

/** "Claude Sonnet 4.5 retires 30 Nov 2026: cs_02, cs_03", one line per date and model, soonest first; "retired"
 *  once the date has passed. Only verdicts that recorded a retirement date (retires_on) count. */
export function sunsetLines(verdicts: Verdict[], sites: Record<string, CallSite>, today: string): string[] {
  const groups = new Map<string, { day: string; name: string; ids: string[] }>();
  for (const v of verdicts) {
    if (!v.retires_on) continue;
    const model = sites[v.site_id]?.source_model;
    const name = model ? spelledName(model, v.sunset_risk) : "The source model";
    const key = `${v.retires_on}|${name}`;
    if (!groups.has(key)) groups.set(key, { day: v.retires_on, name, ids: [] });
    groups.get(key)!.ids.push(v.site_id);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([, g]) => `${g.name} ${g.day < today ? "retired" : "retires"} ${dayText(g.day)}: ${g.ids.join(", ")}`);
}

/** "owner/repo" for a GitHub URL, else the URL or path as given. */
export function repoName(url: string): string {
  const m = url.match(/^https:\/\/github\.com\/([^/\s]+\/[^/\s]+?)(?:\.git)?\/?$/);
  return m ? m[1] : url;
}

/** "PR #1" for a GitHub pull request URL, else "the pull request". */
export const prLabel = (url: string) => {
  const m = url.match(/\/pull\/(\d+)/);
  return m ? `PR #${m[1]}` : "the pull request";
};

/** A unified diff split per file. A file starts at a "--- " line directly followed by "+++ " (a removed line
 *  reading "-- x" is not a header); `added` and `removed` count the hunk lines, not the headers. */
export function splitDiff(diff: string): { path: string; lines: string[]; added: number; removed: number }[] {
  const lines = diff.replace(/\n$/, "").split("\n");
  const files: { path: string; lines: string[]; added: number; removed: number }[] = [];
  for (let i = 0; i < lines.length; i++) {
    const header = lines[i].startsWith("--- ") && lines[i + 1]?.startsWith("+++ ");
    if (header) files.push({ path: lines[i + 1].slice(4).replace(/^b\//, ""), lines: [], added: 0, removed: 0 });
    else if (files.length === 0) files.push({ path: "", lines: [], added: 0, removed: 0 });
    const f = files[files.length - 1];
    const body = f.lines.length >= 2 || !f.path; // past this file's two header lines
    f.lines.push(lines[i]);
    if (body && !header && lines[i].startsWith("+")) f.added += 1;
    else if (body && !header && lines[i].startsWith("-")) f.removed += 1;
  }
  return files;
}
