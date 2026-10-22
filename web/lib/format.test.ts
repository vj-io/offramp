import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import cases from "./offramp-cases.json";
import {
  auditText,
  boundsText,
  byConstructionLines,
  dayText,
  headline,
  judgeText,
  keepClosedStamp,
  keptText,
  manualReason,
  notEvaluatedText,
  parityText,
  parseNotEvaluated,
  prLabel,
  repoName,
  resultBounds,
  reviewNote,
  spelledName,
  splitDiff,
  sunsetLines,
} from "./format";
import { applyEvents, initialRunState, parseRecording, runReducer, toSnapshot } from "./run-state";
import type { CallSite, PatchFailure, RunConfig, TaskType, TierResult, Verdict } from "./types";

// The recorded demo run as the hosted site serves it: no field it lacks is assumed. Tests marked "tripwire" pin
// today's recording (run 20261022-014146107-57cca9, PR vj-io/offramp-demo-app#3) and change when it is re-recorded.
const demo = toSnapshot(
  applyEvents(
    initialRunState,
    parseRecording(JSON.parse(readFileSync(join(__dirname, "../public/replay/demo-events.json"), "utf8")))!.events,
  ),
);

// The real-repo recording (?replay=oss): run 20261022-153359255-7bea7c on a fork of mnemox-ai/idea-reality-mcp.
const oss = toSnapshot(
  applyEvents(
    initialRunState,
    parseRecording(JSON.parse(readFileSync(join(__dirname, "../public/replay/oss-events.json"), "utf8")))!.events,
  ),
);

describe("the open-source recording", () => {
  it("tripwire: names its target repo, its PR on the fork and what moved", () => {
    expect(repoName(oss.config!.repo_url)).toBe("vj-io/idea-reality-mcp");
    expect(oss.artifacts.pr_url).toBe("https://github.com/vj-io/idea-reality-mcp/pull/1");
    expect(headline(oss.verdicts, oss.totals).lead).toBe(
      "2 of 5 call sites reached parity on Nemotron (held-out cases) · 2 migrated · est. −78% on those 2",
    );
  });
});

const V = (over: Partial<Verdict>): Verdict => ({
  site_id: "cs_01",
  decision: "MIGRATE",
  tier: "lightning",
  model: null,
  parity_pct: 100,
  best_tier: null,
  best_score: null,
  iterations: 0,
  latency_delta_ms: null,
  cost_before_per_1k: 10,
  cost_after_per_1k: 1,
  reason: "r",
  ...over,
});
const config = { threshold: 0.95, reasoning_threshold: 0.9 } as RunConfig;

describe("report headline", () => {
  it("tripwire: leads the recorded demo with what is true in its own data", () => {
    expect(headline(demo.verdicts, demo.totals)).toEqual({
      lead: "6 of 6 call sites reached parity on Nemotron (held-out cases) · 5 migrated · est. −95% on those 5",
      scope: "−95% across all 6 ($49.24 → $2.65 per 1,000 calls, est.), since 1 stays for review",
    });
  });
  it("names every scope: all migrated, none migrated, kept and review together", () => {
    const totals = { est_cost_before_per_1k: 30, est_cost_after_per_1k: 12 } as never;
    expect(headline([V({}), V({ site_id: "cs_02" })], totals)).toEqual({
      lead: "2 of 2 call sites reached parity on Nemotron (held-out cases) · 2 migrated · est. −90% on all 2",
      scope: null,
    });
    const kept = V({ site_id: "cs_02", decision: "KEEP_CLOSED", tier: null, best_tier: "ultra", parity_pct: 71 });
    const review = V({ site_id: "cs_03", decision: "MANUAL_REVIEW" });
    const stopped = V({ site_id: "cs_04", decision: "MANUAL_REVIEW", tier: null, parity_pct: null });
    expect(headline([V({}), kept, review, stopped], totals)).toEqual({
      lead: "2 of 4 call sites reached parity on Nemotron (held-out cases) · 1 migrated · est. −90% on that one",
      scope: "−60% across all 4 ($30.00 → $12.00 per 1,000 calls, est.), since 2 stay for review and 1 stays on its source model",
    });
    expect(headline([kept], null)).toEqual({ lead: "0 of 1 call site reached parity on Nemotron (held-out cases) · none migrated", scope: null });
  });
  it("leaves out a cost figure it cannot compute from every migrated site", () => {
    expect(headline([V({ cost_after_per_1k: null })], null).lead).toBe(
      "1 of 1 call site reached parity on Nemotron (held-out cases) · 1 migrated",
    );
  });
});

describe("sunset line", () => {
  const sites = {
    cs_02: { source_model: "claude-sonnet-4-5" },
    cs_03: { source_model: "claude-sonnet-4-5" },
    cs_04: { source_model: "gpt-4o" },
  } as unknown as Record<string, CallSite>;
  const notice = "Claude Sonnet 4.5 is deprecated and retires on November 30, 2026. (https://x)";
  it("groups by date and model, spelled as the notice spells it, soonest first", () => {
    const vs = [
      V({ site_id: "cs_02", retires_on: "2026-11-30", sunset_risk: notice }),
      V({ site_id: "cs_03", retires_on: "2026-11-30", sunset_risk: notice }),
      V({ site_id: "cs_04", retires_on: "2026-10-01", sunset_risk: "a notice without the name" }),
      V({ site_id: "cs_05", sunset_risk: notice }), // no recorded date: no line
    ];
    expect(sunsetLines(vs, sites, "2026-10-21")).toEqual([
      "gpt-4o retired 1 Oct 2026: cs_04",
      "Claude Sonnet 4.5 retires 30 Nov 2026: cs_02, cs_03",
    ]);
  });
  it("spells a model only from a whole-name match", () => {
    expect(spelledName("claude-sonnet-4-5", notice)).toBe("Claude Sonnet 4.5");
    expect(spelledName("claude-sonnet-4-5", "Claude Sonnet 4.50 is new")).toBe("claude-sonnet-4-5");
    expect(spelledName("openai/gpt-4o", "GPT-4o retires")).toBe("GPT-4o");
    expect(dayText("2026-11-30")).toBe("30 Nov 2026");
  });
  it("tripwire: names the recorded demo's retiring source model and its sites", () => {
    const byId = Object.fromEntries(demo.sites.map((s) => [s.id, s]));
    expect(sunsetLines(demo.verdicts, byId, "2026-10-21")).toEqual(["Claude Sonnet 4.5 retires 30 Nov 2026: cs_02, cs_03"]);
  });
});

describe("parity bounds", () => {
  it("adds count and Wilson lower bound when recorded, rounded down", () => {
    expect(parityText(V({ holdout_cases: 9, holdout_passed: 9, parity_lower_pct: 70.08 }))).toBe("100.0% (9/9, ≥70%)");
    expect(parityText(V({ holdout_cases: 8, holdout_passed: 8, parity_lower_pct: 67.5 }), "json_extraction")).toBe(
      "100.0% (8/8 pass, ≥67%)",
    );
    expect(parityText(V({}))).toBe("100.0%"); // older recordings keep the figure alone
    expect(boundsText(null, 5, 40)).toBeNull();
  });
  it("tier results: pass count from the failing cases; free text says n", () => {
    const r = { cases: 5, failing_case_ids: ["a"], parity_lower_pct: 37.55, score: 0.8 } as TierResult;
    expect(resultBounds(r, "classification")).toBe("4/5, ≥37%");
    expect(resultBounds({ ...r, cases: 8 }, "free_text")).toBe("n=8");
    expect(resultBounds({ ...r, cases: undefined, parity_lower_pct: undefined }, "classification")).toBeNull();
  });
  it("free text: mean, lowest and n, at two decimals so a miss is never rounded away", () => {
    expect(judgeText({ judge_mean: 4.54, judge_min: 3, holdout_cases: 8 })).toBe("mean 4.54, lowest 3.0, n=8");
    expect(judgeText({ judge_mean: 3.95, judge_min: 2.96, holdout_cases: 8 }, true)).toBe(
      "mean 3.95 < 4.0, lowest 2.96 < 3.0, n=8",
    );
    expect(judgeText({ judge_mean: null, judge_min: null, holdout_cases: 8 })).toBeNull();
  });
});

describe("KEEP_CLOSED: only the recorded gate", () => {
  const kept = V({ decision: "KEEP_CLOSED", tier: null, parity_pct: 71.4, holdout_cases: 7, holdout_passed: 5, parity_lower_pct: 35.9 });
  it("says each recorded gate and no other", () => {
    expect(keptText({ ...kept, failed_gate: "threshold" }, "reasoning", config)).toBe("71.4% (5/7, ≥35%) < 90%");
    expect(keptText({ ...kept, failed_gate: "threshold" }, "reasoning", null)).toBe("71.4% (5/7, ≥35%), below the parity bar");
    expect(keptText({ ...kept, failed_gate: "dev" }, "reasoning", config)).toBe("71.4% (5/7, ≥35%); holdout passed, dev did not");
    expect(keptText({ ...kept, failed_gate: "budget" }, "reasoning", config)).toBe(
      "71.4% (5/7, ≥35%): holdout failed; the run's budget stopped the rest",
    );
    expect(keptText({ ...kept, parity_pct: null, failed_gate: "budget" }, "reasoning", config)).toBe(
      "not measured: the run's budget stopped it",
    );
    const ft = { ...kept, judge_mean: 3.95, judge_min: 3.2, holdout_cases: 8 };
    expect(keptText({ ...ft, failed_gate: "judge_mean" }, "free_text", config)).toBe("mean 3.95 < 4.0, lowest 3.2, n=8");
    expect(keptText({ ...ft, failed_gate: "budget" }, "free_text", config)).toBe(
      "mean 3.95 < 4.0, lowest 3.2, n=8: holdout failed; the run's budget stopped the rest",
    );
    expect(keptText({ ...ft, failed_gate: "reference_margin" }, "free_text", config)).toBe(
      "mean 3.95, lowest 3.2, n=8, more than 0.3 below the references' judge mean",
    );
  });
  it("a verdict recorded without the gate states its figures with no comparison", () => {
    expect(keptText(kept, "reasoning", config)).toBe("71.4% (5/7, ≥35%)");
    expect(keepClosedStamp({ ...kept, parity_pct: null }, "reasoning", config)).toBe("NOT MIGRATED");
    expect(keepClosedStamp({ ...kept, parity_pct: null, failed_gate: "budget" }, "reasoning", config)).toBe(
      "NOT MIGRATED · not measured: the run's budget stopped it",
    );
  });
});

// offramp-cases.json is pr_writer's own output (tests/unit/test_web_tables.py writes and checks it): the port must
// say exactly what the PR says, joining text and names holding "$&", "$1" or "$$" included.
type ManualCase = { failure: PatchFailure; reason: string; expected: [string, string | null] };
type KeptCase = { verdict: Partial<Verdict>; task: TaskType; expected: string };

describe("the port says what pr_writer says (Python-generated cases)", () => {
  it.each(cases.manual_reason as ManualCase[])("manual_reason %#", ({ failure, reason, expected }) => {
    expect(manualReason(V({ decision: "MANUAL_REVIEW", reason, patch_failure: failure }))).toEqual({
      sentence: expected[0],
      raw: expected[1],
    });
  });
  it.each(cases.kept as KeptCase[])("_kept %#", ({ verdict, task, expected }) => {
    expect(keptText(V({ decision: "KEEP_CLOSED", tier: null, ...verdict }), task, config)).toBe(expected);
  });
  it("covers names with replacement patterns", () => {
    expect((cases.manual_reason as ManualCase[]).some((c) => c.expected[0].includes("$&"))).toBe(true);
  });
});

describe("manual-review note on the page", () => {
  it("tripwire: the recorded demo's one review site gives the PR's sentence for its recorded kind", () => {
    const review = demo.verdicts.filter((v) => v.decision === "MANUAL_REVIEW");
    expect(review.map((v) => [v.site_id, v.patch_failure?.kind])).toEqual([["cs_01", "search_miss"]]);
    expect(reviewNote(review[0])).toEqual(manualReason(review[0]));
    expect(reviewNote(review[0])).toEqual({
      sentence: "The model's edit to `tests/test_triage.py` quotes lines matching no single place there, so none applied.",
      raw: review[0].reason,
    });
  });
  it("a short reason without a recorded kind shows as it is; a recorded kind gives the PR's sentence", () => {
    const short = "patch not composed: Offramp could not rewrite this call; a model-written call is not shipped";
    expect(reviewNote(V({ decision: "MANUAL_REVIEW", reason: short }))).toEqual({ sentence: short, raw: null });
    const long = V({ decision: "MANUAL_REVIEW", tier: null, reason: "x".repeat(200) });
    expect(reviewNote(long)).toEqual({ sentence: "Offramp stopped at this site without a patch.", raw: long.reason });
    const kind = V({ decision: "MANUAL_REVIEW", reason: "raw", patch_failure: { kind: "repo_red" } });
    expect(reviewNote(kind)).toEqual(manualReason(kind));
  });
});

describe("calls found but not evaluated", () => {
  it("tripwire: reads the recorded demo's embedding call back from its log", () => {
    expect(demo.not_evaluated).toEqual([{ id: "emb_01", kind: "embedding", file: "helphub/search/embed.py" }]);
    expect(notEvaluatedText(demo.not_evaluated![0])).toBe("found, not evaluated (embedding calls aren't chat calls)");
  });
  it("matches only the pipeline's exact line", () => {
    expect(parseNotEvaluated("emb_02: embedding call in a b/c.py, not evaluated (chat calls only)")).toEqual({
      id: "emb_02",
      kind: "embedding",
      file: "a b/c.py",
    });
    expect(parseNotEvaluated("emb_02: embedding call in c.py, not evaluated")).toBeNull();
    expect(parseNotEvaluated("Stage: Scanned")).toBeNull();
  });
  it("restart clears them with the rest of the run", () => {
    expect(runReducer(applyEvents(initialRunState, []), "reset")).toBe(initialRunState);
  });
});

describe("how parity was measured, said once", () => {
  const auditor = "claude-sonnet-5.5 (via Claude Code; not the repo's source model)";
  it("names the auditing model once for all the sites it audited", () => {
    const vs = ["cs_01", "cs_02", "cs_04"].map((id) =>
      V({ site_id: id, closed_reference_model: auditor, closed_reference_parity_pct: id === "cs_04" ? 90 : null }),
    );
    expect(byConstructionLines(vs)).toEqual([
      `${auditor} is not compared on cs_01, cs_02: it audited those sites' references, so it agrees with every one kept by construction`,
    ]);
    expect(byConstructionLines(vs, auditor)[0]).toMatch(/^It is not compared on cs_01, cs_02: /);
    const a = { checked: 24, agreed: 21, dropped: 3, human_verified: 0, auditor };
    expect(auditText(a, false)).toBe("24 checked, 21 agreed, 3 dropped, 0 human-verified");
  });
});

describe("links and diffs", () => {
  it("names the repo and the PR", () => {
    expect(repoName("https://github.com/vj-io/offramp-demo-app")).toBe("vj-io/offramp-demo-app");
    expect(repoName("https://github.com/o/r.git/")).toBe("o/r");
    expect(repoName("/tmp/local-repo")).toBe("/tmp/local-repo");
    expect(prLabel("https://github.com/vj-io/offramp-demo-app/pull/1")).toBe("PR #1");
    expect(prLabel("https://example.com/x")).toBe("the pull request");
  });
  it("splits a diff per file and counts its lines, not its headers", () => {
    const diff =
      "--- a/app.py\n+++ b/app.py\n@@ -1,3 +1,3 @@\n-a\n--- sql comment\n+b\n ctx\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+d\n";
    expect(splitDiff(diff).map(({ path, added, removed, lines }) => [path, added, removed, lines.length])).toEqual([
      ["app.py", 1, 2, 7],
      ["new.py", 1, 0, 4],
    ]);
  });
});
