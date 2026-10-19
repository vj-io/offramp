import { describe, expect, it } from "vitest";
import fixtureFile from "../fixtures/dev-events.json";
import {
  applyEvents,
  downloadPayload,
  initialRunState,
  parseRecording,
  pickSource,
  replayDelays,
  siteView,
  stageProgress,
  toSnapshot,
} from "./run-state";
import {
  auditText,
  closedReferenceText,
  keepClosedStamp,
  latencyText,
  linkParts,
  referencesText,
  thresholdFor,
} from "./format";
import { STAGE_ORDER, type CaseSummary, type RunEvent, type Verdict } from "./types";

const events = fixtureFile.events as unknown as RunEvent[];
const final = applyEvents(initialRunState, events);
const ofType = (t: RunEvent["type"]) => events.filter((e) => e.type === t);

describe("dev fixture covers the brief", () => {
  it("is labelled as fixture data", () => {
    expect(fixtureFile.fixture).toBe(true);
    expect(fixtureFile.note).toMatch(/made-up numbers/);
  });
  it("has 6 sites, a KEEP_CLOSED reasoning site, a 2-iteration pass, all tiers and a PR", () => {
    expect(ofType("site")).toHaveLength(6);
    const kept = Object.values(final.verdicts).filter((v) => v.decision === "KEEP_CLOSED");
    expect(kept).toHaveLength(1);
    expect(final.sites[kept[0].site_id].contract?.task_type).toBe("reasoning");
    const twoIter = Object.values(final.verdicts).find((v) => v.decision === "MIGRATE" && v.iterations === 2);
    expect(twoIter).toBeDefined();
    expect(new Set(final.tierResults.map((r) => r.tier))).toEqual(new Set(["lightning", "super", "ultra"]));
    expect(ofType("pr")).toHaveLength(1);
  });
  it("uses the amended shapes: references, null latency, est. cost notes, a multi-tier iteration site", () => {
    for (const e of ofType("case_summary")) expect(e.payload).toHaveProperty("references");
    const verdicts = Object.values(final.verdicts);
    expect(verdicts.every((v) => v.latency_delta_ms === null)).toBe(true);
    expect(verdicts.every((v) => v.cost_before_note?.startsWith("est.:"))).toBe(true);
    expect(verdicts.some((v) => v.sunset_risk)).toBe(true);
    const tiersIterated = new Set(final.iterations.filter((a) => a.site_id === "cs_05").map((a) => a.tier));
    expect(tiersIterated.size).toBeGreaterThan(1);
    expect(ofType("stage")[0].payload.config).toBeDefined();
  });
});

describe("applyEvents over the fixture", () => {
  it("collects every site, verdict, PR and the done totals", () => {
    expect(Object.keys(final.sites)).toEqual(ofType("site").map((e) => e.call_site_id));
    expect(Object.keys(final.verdicts).sort()).toEqual(ofType("verdict").map((e) => e.call_site_id).sort());
    expect(final.pr).toEqual(ofType("pr")[0].payload);
    expect(final.totals).toEqual(ofType("done")[0].payload.totals);
    expect(final.stage).toBe("DONE");
    expect(final.failure).toBeNull();
    expect(final.lastSeq).toBe(events[events.length - 1].seq);
    expect(final.config?.threshold).toBe(0.95);
  });

  it("ignores events it has already seen (reconnect replays from seq 0)", () => {
    const twice = applyEvents(final, events);
    expect(twice).toEqual(final);
    expect(twice.events).toHaveLength(events.length);
  });

  it("upserts re-emitted tier results instead of appending duplicates", () => {
    const keys = ofType("tier_result").map((e) => {
      const p = e.payload as { site_id: string; tier: string; adapter_version: number; split: string };
      return `${p.site_id}|${p.tier}|${p.adapter_version}|${p.split}`;
    });
    expect(keys.length).toBeGreaterThan(new Set(keys).size); // the fixture re-emits at least one
    expect(final.tierResults).toHaveLength(new Set(keys).size);
  });

  it("re-emitted verdict for a site replaces the earlier one", () => {
    const v = ofType("verdict")[0];
    const changed = { ...v, seq: final.lastSeq + 1, payload: { ...v.payload, reason: "re-measured" } };
    const next = applyEvents(final, [changed]);
    expect(next.verdicts[v.call_site_id!].reason).toBe("re-measured");
    expect(Object.keys(next.verdicts)).toHaveLength(Object.keys(final.verdicts).length);
  });
});

describe("siteView", () => {
  it("reports the site that passed after two iterations on its chosen tier", () => {
    const v = Object.values(final.verdicts).find((x) => x.iterations === 2)!;
    const view = siteView(final, v.site_id);
    expect(view.iterations).toBe(2);
    expect(view.currentTier).toBe(v.tier);
    const r = view.perTier[v.tier!]!;
    expect(r.passed).toBe(true);
    expect(r.split).toBe("holdout"); // latest result for the tier wins
  });

  it("KEEP_CLOSED site shows its best tier and failing results on every tier", () => {
    const v = Object.values(final.verdicts).find((x) => x.decision === "KEEP_CLOSED")!;
    const view = siteView(final, v.site_id);
    expect(view.currentTier).toBe(v.best_tier);
    for (const r of Object.values(view.perTier)) expect(r?.passed).toBe(false);
  });

  it("counts iterations on the current tier only, matching Verdict.iterations", () => {
    const v = final.verdicts.cs_05;
    const all = final.iterations.filter((a) => a.site_id === "cs_05").length;
    expect(all).toBeGreaterThan(v.iterations); // the fixture iterates on several tiers
    expect(siteView(final, "cs_05").iterations).toBe(v.iterations);
    const before = applyEvents(initialRunState, events.filter((e) => e.seq < ofType("verdict")[0].seq));
    const view = siteView(before, "cs_05");
    const onTier = before.iterations.filter((a) => a.site_id === "cs_05" && a.tier === view.currentTier).length;
    expect(view.iterations).toBe(onTier);
    expect(view.iterations).toBeLessThan(all);
  });

  it("before any tier result, current tier is null and chips are empty", () => {
    const early = applyEvents(initialRunState, events.filter((e) => e.type === "site"));
    const view = siteView(early, "cs_01");
    expect(view.currentTier).toBeNull();
    expect(Object.values(view.perTier).every((r) => r === null)).toBe(true);
  });
});

describe("stageProgress", () => {
  it("marks every stage done when the run is DONE", () => {
    expect(stageProgress(final).map((s) => s.status)).toEqual(STAGE_ORDER.map(() => "done"));
  });

  it("marks reached stages done and the latest one current", () => {
    const upTo = events.findIndex((e) => e.type === "stage" && e.payload.stage === "EVALUATING");
    const mid = applyEvents(initialRunState, events.slice(0, upTo + 1));
    const statuses = stageProgress(mid);
    const i = STAGE_ORDER.indexOf("EVALUATING");
    expect(statuses[i].status).toBe("current");
    expect(statuses.slice(0, i).every((s) => s.status === "done")).toBe(true);
    expect(statuses.slice(i + 1).every((s) => s.status === "pending")).toBe(true);
  });

  it("an error event marks the failing stage and keeps the reason", () => {
    const upTo = events.findIndex((e) => e.type === "stage" && e.payload.stage === "BASELINED");
    const base = applyEvents(initialRunState, events.slice(0, upTo + 1));
    const err: RunEvent = {
      run_id: "r",
      seq: base.lastSeq + 1,
      ts: "2026-10-19T10:05:00Z",
      stage: "FAILED",
      type: "error",
      call_site_id: null,
      payload: { stage: "BASELINED", reason: "baseline fixtures missing" },
    };
    const failed = applyEvents(base, [err]);
    expect(failed.stage).toBe("FAILED");
    expect(failed.failure).toEqual({ stage: "BASELINED", reason: "baseline fixtures missing" });
    const statuses = stageProgress(failed);
    expect(statuses[STAGE_ORDER.indexOf("BASELINED")].status).toBe("failed");
    expect(statuses[STAGE_ORDER.indexOf("EVALUATING")].status).toBe("pending");
  });
});

describe("toSnapshot", () => {
  it("rebuilds the RunSnapshot shape from the event log", () => {
    const snap = toSnapshot(final);
    expect(snap.run_id).toBe(events[0].run_id);
    expect(snap.stage).toBe("DONE");
    expect(snap.sites).toHaveLength(6);
    expect(snap.verdicts).toHaveLength(ofType("verdict").length);
    expect(snap.artifacts.pr_url).toBe(final.pr?.url);
    expect(snap.totals).toEqual(final.totals);
    expect(snap.case_summaries).toHaveLength(ofType("case_summary").length);
  });
});

describe("replayDelays (Global Constraint 14)", () => {
  it("caps gaps at 1.5 s and plays at 4x", () => {
    const at = (s: number) => ({ ...events[0], ts: new Date(Date.UTC(2026, 9, 19, 10, 0, s)).toISOString() });
    expect(replayDelays([at(0), at(1), at(10), at(10)])).toEqual([0, 250, 375, 0]);
  });
  it("finishes the fixture in under 90 s", () => {
    const total = replayDelays(events).reduce((a, b) => a + b, 0);
    expect(total).toBeLessThan(90_000);
  });
});

describe("parseRecording (replay file)", () => {
  it("accepts an array or {events}", () => {
    const real = events.map((e) => ({ ...e, run_id: "run-abc" }));
    expect(parseRecording(real)).toEqual(real);
    expect(parseRecording({ events: real })).toEqual(real);
  });
  it("refuses fixture data, so it can never play as the recorded demo", () => {
    expect(parseRecording(fixtureFile)).toBeNull();
    expect(parseRecording(events)).toBeNull(); // run ids start with "fixture-"
    expect(parseRecording({ events: [{ ...events[0], run_id: "run-1" }], fixture: true })).toBeNull();
  });
  it("throws on a body that is not an event log", () => {
    expect(() => parseRecording({ hello: 1 })).toThrow();
  });
});

describe("downloadPayload", () => {
  const report = { snapshot: toSnapshot(final), pr_body_markdown: null };
  it("marks fixture downloads as fixture data", () => {
    expect(downloadPayload(report, "fixture")).toMatchObject({ fixture: true });
    expect(downloadPayload(report, "replay")).not.toHaveProperty("fixture");
  });
});

describe("pickSource", () => {
  const src = (q: string) => pickSource(new URLSearchParams(q));
  it("chooses replay, fixture or the API, with ?live=1 forcing the API", () => {
    expect(src("")).toBe("api");
    expect(src("replay=demo")).toBe("replay");
    expect(src("fixture=dev")).toBe("fixture");
    expect(src("replay=demo&live=1")).toBe("api");
  });
});

describe("KEEP_CLOSED stamp", () => {
  const v = Object.values(final.verdicts).find((x) => x.decision === "KEEP_CLOSED") as Verdict;
  it("uses the run's own threshold for the task type", () => {
    expect(thresholdFor("reasoning", final.config)).toBe(0.9);
    expect(thresholdFor("classification", final.config)).toBe(0.95);
    expect(keepClosedStamp(v, "reasoning", final.config)).toBe("NOT MIGRATED · 71.0% < 90%");
  });
  it("never shows a threshold the run did not record", () => {
    expect(thresholdFor("reasoning", null)).toBeNull();
    expect(keepClosedStamp(v, "reasoning", null)).toBe("NOT MIGRATED · best parity 71.0%");
    expect(keepClosedStamp({ ...v, parity_pct: null }, "reasoning", final.config)).toBe("NOT MIGRATED");
  });
});

describe("evidence text", () => {
  it("latency: before is never invented", () => {
    expect(latencyText(null, 412)).toBe("— → 412 ms");
    expect(latencyText(-30, 380)).toBe("410 ms → 380 ms");
    expect(latencyText(null, null)).toBe("—");
    expect(latencyText(40, null)).toBe("+40 ms");
  });
  it("references and audit come from the case summary, and an empty audit is omitted", () => {
    const c = ofType("case_summary").find((e) => e.call_site_id === "cs_01")!.payload as unknown as CaseSummary;
    expect(referencesText(c)).toBe("40 references (reference:seed 40)");
    expect(auditText(c.audit)).toBe("40 checked, 39 agreed, 1 dropped, 0 human-verified, auditor claude-sonnet-5.5 (via Claude Code)");
    const oracle = ofType("case_summary").find((e) => e.call_site_id === "cs_05")!.payload as unknown as CaseSummary;
    expect(auditText(oracle.audit)).toBeNull();
    expect(referencesText({ ...c, reference_methods: {} })).toBe("40 references");
  });
  it("closed-reference comparison only when the verdict carries it", () => {
    expect(closedReferenceText(final.verdicts.cs_05)).toBe(
      "claude-sonnet-5.5 (via Claude Code; not the repo's source model) scores 88.0% on the same references",
    );
    expect(closedReferenceText(final.verdicts.cs_01)).toBeNull();
  });
  it("links only http(s) URLs in notes", () => {
    expect(linkParts("est. (see https://example.com/p, fetched) or openai.com/x")).toEqual([
      { text: "est. (see " },
      { text: "https://example.com/p", href: "https://example.com/p" },
      { text: ", fetched) or openai.com/x" },
    ]);
  });
});
