import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  boundsText,
  dayText,
  headline,
  judgeText,
  keepClosedStamp,
  keptText,
  manualReason,
  parityText,
  prLabel,
  repoName,
  resultBounds,
  spelledName,
  splitDiff,
  sunsetLines,
} from "./format";
import { applyEvents, initialRunState, parseRecording, toSnapshot } from "./run-state";
import type { CallSite, PatchFailure, RunConfig, TierResult, Verdict } from "./types";

// The recorded demo run as the hosted site serves it: no field it lacks is assumed.
const demo = toSnapshot(
  applyEvents(
    initialRunState,
    parseRecording(JSON.parse(readFileSync(join(__dirname, "../public/replay/demo-events.json"), "utf8")))!.events,
  ),
);

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
  it("leads the recorded demo with what is true in its own data", () => {
    expect(headline(demo.verdicts, demo.totals)).toEqual({
      lead: "6 of 6 call sites reached parity on Nemotron · 3 migrated · est. −82% on those 3",
      scope: "−11% across all 6 ($49.24 → $43.86 per 1,000 calls, est.), since 3 stay for review",
    });
  });
  it("names every scope: all migrated, none migrated, kept and review together", () => {
    const totals = { est_cost_before_per_1k: 30, est_cost_after_per_1k: 12 } as never;
    expect(headline([V({}), V({ site_id: "cs_02" })], totals)).toEqual({
      lead: "2 of 2 call sites reached parity on Nemotron · 2 migrated · est. −90% on all 2",
      scope: null,
    });
    const kept = V({ site_id: "cs_02", decision: "KEEP_CLOSED", tier: null, best_tier: "ultra", parity_pct: 71 });
    const review = V({ site_id: "cs_03", decision: "MANUAL_REVIEW" });
    const stopped = V({ site_id: "cs_04", decision: "MANUAL_REVIEW", tier: null, parity_pct: null });
    expect(headline([V({}), kept, review, stopped], totals)).toEqual({
      lead: "2 of 4 call sites reached parity on Nemotron · 1 migrated · est. −90% on that one",
      scope: "−60% across all 4 ($30.00 → $12.00 per 1,000 calls, est.), since 2 stay for review and 1 stays on its source model",
    });
    expect(headline([kept], null)).toEqual({ lead: "0 of 1 call site reached parity on Nemotron · none migrated", scope: null });
  });
  it("leaves out a cost figure it cannot compute from every migrated site", () => {
    expect(headline([V({ cost_after_per_1k: null })], null).lead).toBe(
      "1 of 1 call site reached parity on Nemotron · 1 migrated",
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
  it("adds nothing for the recorded demo, which predates retires_on", () => {
    const byId = Object.fromEntries(demo.sites.map((s) => [s.id, s]));
    expect(sunsetLines(demo.verdicts, byId, "2026-10-21")).toEqual([]);
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
    expect(keptText({ ...kept, failed_gate: "budget" }, "reasoning", config)).toBe("not measured: the run's budget stopped it");
    const ft = { ...kept, judge_mean: 3.95, judge_min: 3.2, holdout_cases: 8 };
    expect(keptText({ ...ft, failed_gate: "judge_mean" }, "free_text", config)).toBe("mean 3.95 < 4.0, lowest 3.2, n=8");
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

// The same cases as tests/unit/test_pr_render.py::test_each_recorded_failure_kind_reads_as_one_plain_sentence.
const EDITED = "with the model's test edits applied,";
const ALONE = "with Offramp's changes alone,";
const GREEN = "Its patch passes this repo's checks on its own, but";
const NEVER = "and Offramp never ships a patch that deletes a test.";
const SHIPS = "code that ships changed (`app/x.py`), and only Offramp's code may write it.";
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const M = true;
const CASES: [PatchFailure, string][] = [
  [{ kind: "search_miss", files: ["tests/t.py"], model_edit: M }, "The model's edit to `tests/t.py` quotes lines matching no single place there, so none applied."],
  [{ kind: "deletes_tests", files: ["tests/t.py"], tests: ["tests/t.py::test_o3"], model_edit: M }, `The model's edit deleted the test \`tests/t.py::test_o3\`, ${NEVER}`],
  [{ kind: "locked_file", files: ["app/x.py"], model_edit: M }, "The model tried to edit `app/x.py`, which ships exactly as Offramp's code wrote it."],
  [{ kind: "not_editable", files: ["../x.py"], model_edit: M }, "The model's edit to `../x.py` is not allowed for this site."],
  [{ kind: "does_not_parse", files: ["tests/t.py"], model_edit: M }, "The model's edit left `tests/t.py` unparseable."],
  [{ kind: "checks_failed", model_edit: M }, `${cap(EDITED)} this repo's checks still failed.`],
  [{ kind: "backstop", files: ["app/x.py"], model_edit: M }, `${cap(EDITED)} ${SHIPS}`],
  [{ kind: "probe_mismatch", model_edit: M }, `${cap(EDITED)} the call does not send the request that passed the evaluation.`],
  [{ kind: "probe_mismatch" }, "Offramp's own rewrite does not send the request that passed the evaluation."],
  [{ kind: "merge_conflict", files: ["tests/t.py"] }, `${GREEN} it conflicts with another migrated site's patch in \`tests/t.py\`.`],
  [{ kind: "not_together" }, `${GREEN} not merged with the other migrated sites' patches.`],
  [{ kind: "merged_drift" }, `${GREEN} on the merged tree its code or its evaluated request changed.`],
  [{ kind: "repo_red" }, "This repo's own checks fail before any migration patch, so no patch could be checked."],
  [{ kind: "budget_stop", last: "checks_failed", tries: 0 }, `The run's model budget ran out before the model replied at this site's patch; ${ALONE} this repo's checks fail.`],
  [{ kind: "budget_stop", last: "deletes_tests", tries: 2, model_edit: M, tests: ["t.py::test_a"] }, `The run's model budget ran out after 2 tries at this site's patch; the model's edit deleted the test \`t.py::test_a\`, ${NEVER}`],
  [{ kind: "no_edit_block", last: "checks_failed", tries: 4 }, `The model's last reply had no complete edit, so nothing was applied; ${ALONE} this repo's checks fail.`],
  [{ kind: "no_change", last: "backstop", tries: 2, model_edit: M, files: ["app/x.py"] }, `The model's last repair repeated an edit it had already tried; ${EDITED} ${SHIPS}`],
];

describe("manual-review reason (pr_writer.manual_reason)", () => {
  it.each(CASES)("%o reads as one plain sentence, raw reason folded", (failure, sentence) => {
    expect(manualReason(V({ decision: "MANUAL_REVIEW", reason: "raw | output", patch_failure: failure }))).toEqual({
      sentence,
      raw: "raw | output",
    });
  });
  it("without a recorded failure the reason shows as it is (the recorded demo)", () => {
    const v = demo.verdicts.find((x) => x.decision === "MANUAL_REVIEW")!;
    expect(manualReason(v)).toEqual({ sentence: v.reason, raw: null });
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
