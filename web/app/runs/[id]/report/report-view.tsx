"use client";

import type { CSSProperties } from "react";
import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { BrandBadges, LinkedText, SunsetBadge, SunsetRisk } from "@/components/badges";
import { DECISION_NOTE, Note, Stamp, Swipe, TIER_HL } from "@/components/doodle";
import { FilePath } from "@/components/file-path";
import { NoRecording, NoticeNote, SourceLabel } from "@/components/run-notices";
import {
  afterCells,
  auditText,
  closedByConstructionText,
  closedReferenceText,
  keepClosedStamp,
  ms,
  pct100,
  referencesText,
  signedMs,
  TASK_NAME,
  thresholdFor,
  TIER_NAME,
  TIER_SHORT,
  usd,
} from "@/lib/format";
import { downloadPayload, pickSource, type RunSource } from "@/lib/run-state";
import type { CallSite, CaseSummary, RunReport, TierResult, Verdict } from "@/lib/types";
import { API_URL, useReport } from "@/lib/use-run";

export function ReportView() {
  const { id } = useParams<{ id: string }>();
  const query = useSearchParams().toString();
  return <ReportBody key={`${id}?${query}`} id={id} source={pickSource(new URLSearchParams(query))} query={query} />;
}

function ReportBody({ id, source, query }: { id: string; source: RunSource; query: string }) {
  const status = useReport(id, source);
  if (status.kind === "loading") return <p className="font-hand text-xl text-muted">Loading the report…</p>;
  if (status.kind === "missing") return <NoRecording />;
  if (status.kind === "error")
    return (
      <NoticeNote title="The report could not be loaded" tone="warn">
        {status.message}
      </NoticeNote>
    );
  return <Report report={status.report} id={id} source={source} query={query} />;
}

// A kept call's sunset notice already sits in its card above the table; only the others get a footnote under it.
const hasSunsetFootnote = (v: Verdict) => v.sunset_risk && v.decision !== "KEEP_CLOSED";

function holdoutResult(results: TierResult[], v: Verdict) {
  return results.findLast((r) => r.site_id === v.site_id && r.tier === v.tier && r.split === "holdout") ?? null;
}

function download(report: RunReport, source: RunSource) {
  const json = JSON.stringify(downloadPayload(report, source), null, 2);
  const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: `offramp-run-${report.snapshot.run_id}.json` });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000); // revoking synchronously can cancel the download
}

const EVIDENCE_COLUMNS = [
  "Site",
  "File",
  "Task",
  "Before",
  "After",
  "Parity vs reference (holdout)",
  "p50 latency (after)",
  "$/1k calls before (est.) → after",
];

function Report({ report, id, source, query }: { report: RunReport; id: string; source: RunSource; query: string }) {
  const snap = report.snapshot;
  const sites: Record<string, CallSite> = Object.fromEntries(snap.sites.map((s) => [s.id, s]));
  const order = (v: Verdict) => snap.sites.findIndex((s) => s.id === v.site_id);
  const verdicts = [...snap.verdicts].sort((a, b) => order(a) - order(b));
  const migrated = verdicts.filter((v) => v.decision === "MIGRATE");
  const kept = verdicts.filter((v) => v.decision === "KEEP_CLOSED");
  const review = verdicts.filter((v) => v.decision === "MANUAL_REVIEW");
  const costNotes = verdicts.filter((v) => v.cost_before_note);
  const sunsetNotes = verdicts.filter(hasSunsetFootnote);
  const totals = snap.totals;
  const saving =
    totals && totals.est_cost_before_per_1k > 0
      ? Math.round((1 - totals.est_cost_after_per_1k / totals.est_cost_before_per_1k) * 100)
      : null;
  const liveHref = `/runs/${encodeURIComponent(id)}${query ? `?${query}` : ""}`;

  if (verdicts.length === 0)
    return (
      <NoticeNote title="No verdicts yet">
        The report fills in once the run reaches the Verdicts stage.{" "}
        <Link href={liveHref} className="text-bluepen underline underline-offset-4">
          Back to the live run
        </Link>
      </NoticeNote>
    );

  return (
    <div className="flex flex-col gap-12">
      <section className="flex flex-col gap-4">
        <SourceLabel source={source} runId={snap.run_id} apiUrl={API_URL} />
        <h1 className="max-w-4xl text-4xl leading-[1.02] font-extrabold tracking-tight sm:text-6xl">
          {migrated.length} of {verdicts.length} LLM calls move to NVIDIA Nemotron on Nebius Token Factory.
        </h1>
        <p className="max-w-3xl font-hand text-2xl leading-snug">
          {kept.length > 0 && (
            <>
              {kept.length} {kept.length === 1 ? "stays on its source model" : "stay on their source models"}: Nemotron
              did not reach the parity bar on held-out reference cases.{" "}
            </>
          )}
          {review.length > 0 && <>{review.length} need a manual review. </>}
          {totals && saving !== null && (
            <>
              Estimated cost per 1,000 calls goes from{" "}
              <span className="font-mono text-xl">{usd(totals.est_cost_before_per_1k)}</span> (est.) to{" "}
              <span className="font-mono text-xl">{usd(totals.est_cost_after_per_1k)}</span>:{" "}
              <Swipe mark="var(--hl-lime)">
                <span className="font-mono text-xl font-bold">{saving > 0 ? `−${saving}%` : `${-saving}% more`}</span>
              </Swipe>
            </>
          )}
        </p>
        <div className="flex flex-wrap items-center gap-3">
          {snap.artifacts.pr_url ? (
            <a
              href={snap.artifacts.pr_url}
              target="_blank"
              rel="noreferrer"
              className="sketch bg-ink px-4 py-2 font-hand text-xl text-paper hover:opacity-90"
            >
              Open the pull request
            </a>
          ) : (
            <span className="font-hand text-lg text-muted">This run did not open a pull request.</span>
          )}
          <button
            type="button"
            onClick={() => download(report, source)}
            className="sketch cursor-pointer px-4 py-2 font-hand text-xl hover:bg-note-plain"
          >
            Download JSON
          </button>
          <Link href={liveHref} className="font-hand text-lg text-bluepen underline underline-offset-4">
            Back to the live run
          </Link>
        </div>
      </section>

      {kept.length > 0 && (
        <section aria-labelledby="kept-h" className="flex flex-col gap-6">
          <h2 id="kept-h" className="text-3xl font-extrabold tracking-tight">
            What Offramp did not change
          </h2>
          <ul className="grid gap-6 md:grid-cols-2">
            {kept.map((v, i) => {
              const site = sites[v.site_id];
              const closedRef = closedReferenceText(v);
              return (
                <Note key={v.site_id} as="li" note={DECISION_NOTE.KEEP_CLOSED} tilt={i % 2 ? 0.8 : -0.8} className="flex flex-col gap-3 p-5 pt-6">
                  <Stamp rotate={-3} big>
                    {keepClosedStamp(v, site?.contract?.task_type, snap.config)}
                  </Stamp>
                  <p className="font-mono text-sm">
                    <span className="font-bold">{v.site_id}</span> {site?.file}
                  </p>
                  <p className="text-xl font-extrabold tracking-tight">
                    Stays on <span className="font-mono text-lg">{site?.source_model ?? "its source model"}</span>
                  </p>
                  <p className="font-hand text-lg leading-snug">{v.reason}</p>
                  {v.best_tier && (
                    <p className="font-hand text-base text-muted">
                      Best try: {TIER_NAME[v.best_tier]}
                      {v.iterations > 0 && (
                        <>
                          {" "}after <span className="font-mono text-sm">{v.iterations}</span> adapter iteration
                          {v.iterations === 1 ? "" : "s"}
                        </>
                      )}
                    </p>
                  )}
                  {closedRef && <p className="font-hand text-lg leading-snug">For comparison: {closedRef}.</p>}
                  {v.sunset_risk && <SunsetRisk note={v.sunset_risk} />}
                  <BrandBadges small />
                </Note>
              );
            })}
          </ul>
        </section>
      )}

      <section aria-labelledby="evidence-h">
        <h2 id="evidence-h" className="mb-4 text-3xl font-extrabold tracking-tight">
          Evidence
        </h2>
        <div role="region" tabIndex={0} aria-label="Evidence table, scrolls sideways" className="relative overflow-x-auto pb-2">
          <table className="w-full min-w-[60rem] border-collapse text-left">
            <thead className="font-hand text-lg">
              <tr className="border-b-2 border-ink">
                {EVIDENCE_COLUMNS.map((h) => (
                  <th key={h} scope="col" className="px-2 py-2 align-bottom font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {verdicts.map((v) => (
                <EvidenceRow key={v.site_id} v={v} site={sites[v.site_id]} results={snap.tier_results} report={report} />
              ))}
            </tbody>
          </table>
        </div>
        {costNotes.length > 0 && (
          <div className="mt-4 max-w-4xl">
            <h3 className="font-hand text-lg">Cost before is an estimate. Price sources:</h3>
            <ul className="mt-1 flex flex-col gap-1 font-mono text-xs leading-relaxed text-muted">
              {costNotes.map((v) => (
                <li key={v.site_id} id={`cost-note-${v.site_id}`}>
                  <span className="font-bold text-ink">{v.site_id}</span> <LinkedText text={v.cost_before_note!} />
                </li>
              ))}
            </ul>
          </div>
        )}
        {sunsetNotes.length > 0 && (
          <div className="mt-4 max-w-4xl">
            <h3 className="font-hand text-lg">Sunset risk notices:</h3>
            <ul className="mt-1 flex flex-col gap-1 font-hand text-base leading-snug">
              {sunsetNotes.map((v) => (
                <li key={v.site_id} id={`sunset-note-${v.site_id}`}>
                  <span className="font-mono text-xs font-bold">{v.site_id}</span> <LinkedText text={v.sunset_risk!} />
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <HowMeasured summaries={snap.case_summaries} verdicts={verdicts} />

      <div className="grid gap-12 lg:grid-cols-2">
        <CostBars verdicts={verdicts} totals={totals} />
        <LatencyAfter verdicts={verdicts} results={snap.tier_results} />
      </div>

      <p className="font-hand text-lg text-muted">
        Every number on this page comes from run <span className="font-mono text-sm text-ink">{snap.run_id}</span>
        {source === "fixture" ? ", which is fixture data" : ""}. Costs before are estimates from published prices; latency
        before was not measured.
      </p>
    </div>
  );
}

function HowMeasured({ summaries, verdicts }: { summaries: CaseSummary[] | undefined; verdicts: Verdict[] }) {
  const closedRefs = verdicts.filter((v) => closedReferenceText(v));
  return (
    <section aria-labelledby="measured-h" className="max-w-4xl">
      <h2 id="measured-h" className="text-3xl font-extrabold tracking-tight">
        How parity was measured
      </h2>
      <p className="mt-2 font-hand text-xl leading-snug">
        Parity is agreement with reference answers whose ground truth is known, scored on held-out cases. The source
        model&apos;s own outputs were never recorded.
      </p>
      {summaries && summaries.length > 0 && (
        <ul className="mt-4 flex flex-col gap-3">
          {summaries.map((c) => {
            const audit = auditText(c.audit);
            const verdict = verdicts.find((v) => v.site_id === c.site_id);
            const byConstruction = verdict ? closedByConstructionText(verdict) : null;
            return (
              <li key={c.site_id} className="grid grid-cols-[3.5rem_minmax(0,1fr)] gap-x-3">
                <span className="font-mono text-sm font-bold">{c.site_id}</span>
                <span className="font-hand text-lg leading-snug">
                  {referencesText(c)}, <span className="font-mono text-sm">{c.dev}</span> dev /{" "}
                  <span className="font-mono text-sm">{c.holdout}</span> holdout.
                  {audit && <> Reference audit: {audit}.</>}
                  {byConstruction && <> {byConstruction}.</>}
                </span>
              </li>
            );
          })}
        </ul>
      )}
      {closedRefs.length > 0 && (
        <ul className="mt-4 flex flex-col gap-2">
          {closedRefs.map((v) => (
            <li key={v.site_id} className="grid grid-cols-[3.5rem_minmax(0,1fr)] gap-x-3">
              <span className="font-mono text-sm font-bold">{v.site_id}</span>
              <span className="font-hand text-lg leading-snug">Closed reference: {closedReferenceText(v)}.</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function EvidenceRow({ v, site, results, report }: { v: Verdict; site: CallSite | undefined; results: TierResult[]; report: RunReport }) {
  const task = site?.contract?.task_type;
  const r = holdoutResult(results, v);
  const bar = thresholdFor(task, report.snapshot.config);
  const isKept = v.decision === "KEEP_CLOSED";
  const after = afterCells(v, r?.metrics.p50_ms ?? null);
  return (
    <tr
      className="border-b border-dashed border-muted align-top"
      style={isKept ? { background: "var(--note-pink)" } : v.decision === "MANUAL_REVIEW" ? { background: "var(--note-orange)" } : undefined}
    >
      <th scope="row" className="px-2 py-3 font-mono text-sm font-bold">
        {v.site_id}
      </th>
      <td className="min-w-[9rem] px-2 py-3 font-mono text-sm wrap-anywhere">{site && <FilePath path={site.file} />}</td>
      <td className="px-2 py-3 font-hand text-lg">{task ? TASK_NAME[task] : "—"}</td>
      <td className="min-w-[10rem] px-2 py-3">
        <span className="flex flex-col items-start gap-2">
          <span className="font-mono text-sm wrap-anywhere">{site?.source_model ?? "—"}</span>
          {v.sunset_risk && (
            <span className="flex flex-wrap items-center gap-x-2">
              <SunsetBadge />
              {hasSunsetFootnote(v) && (
                <a
                  href={`#sunset-note-${v.site_id}`}
                  aria-label={`Sunset risk note for ${v.site_id}`}
                  className="font-hand text-base text-bluepen underline underline-offset-2"
                >
                  note
                </a>
              )}
            </span>
          )}
        </span>
      </td>
      <td className="px-2 py-3">
        {v.decision === "MIGRATE" && v.tier ? (
          <span className="flex flex-col items-start gap-0.5">
            <span className="text-lg font-extrabold tracking-tight whitespace-nowrap">
              <Swipe mark={TIER_HL[v.tier]}>{TIER_NAME[v.tier]}</Swipe>
            </span>
            {v.model && <span className="font-mono text-xs wrap-anywhere text-muted">{v.model}</span>}
          </span>
        ) : isKept ? (
          <Stamp rotate={-2}>NOT MIGRATED</Stamp>
        ) : (
          <span className="hl-chip sketch px-2 font-hand text-lg" style={{ "--mark": "var(--hl-orange)" } as CSSProperties}>
            Manual review
          </span>
        )}
      </td>
      <td className="px-2 py-3 font-mono text-sm">
        {v.parity_pct == null ? (
          "—"
        ) : isKept ? (
          bar !== null ? (
            `${pct100(v.parity_pct)} < ${Math.round(bar * 100)}%`
          ) : (
            <>
              {pct100(v.parity_pct)} <span className="font-hand text-base">below the bar</span>
            </>
          )
        ) : (
          <>
            {pct100(v.parity_pct)}
            {v.iterations > 0 && (
              <span className="font-hand text-base">
                {" "}
                after {v.iterations} adapter iteration{v.iterations === 1 ? "" : "s"}
              </span>
            )}
          </>
        )}
      </td>
      <td className="px-2 py-3 font-mono text-sm whitespace-nowrap">
        {after.latency}
      </td>
      <td className="px-2 py-3 font-mono text-sm whitespace-nowrap">
        {after.cost}
        {v.cost_before_note && (
          <a href={`#cost-note-${v.site_id}`} className="ml-1 font-hand text-base text-bluepen underline underline-offset-2">
            source
          </a>
        )}
      </td>
    </tr>
  );
}

function BarRow({ label, before, after, mark }: { label: string; before: number; after: number | null; mark: string | null }) {
  return (
    <li className="grid grid-cols-[3.5rem_minmax(0,1fr)] items-center gap-x-3 gap-y-1">
      <span className="row-span-2 font-mono text-sm font-bold">{label}</span>
      <span className="flex items-center gap-2">
        <span aria-hidden="true" className="sketch-box h-3.5 w-full max-w-[70%] border-2 bg-[repeating-linear-gradient(135deg,transparent_0_5px,color-mix(in_srgb,var(--ink)_25%,transparent)_5px_7px)]" />
        <span className="font-mono text-xs whitespace-nowrap">{usd(before)} est.</span>
      </span>
      <span className="flex items-center gap-2">
        {after === null || mark === null ? (
          <span className="font-hand text-base text-muted">unchanged, stays on its source model</span>
        ) : (
          <>
            <span
              aria-hidden="true"
              className="sketch-box h-3.5 border-2 bg-[var(--mark)]"
              style={{ "--mark": mark, width: `${Math.max(2, (after / before) * 70)}%` } as CSSProperties}
            />
            <span className="font-mono text-xs whitespace-nowrap">{usd(after)}</span>
          </>
        )}
      </span>
    </li>
  );
}

function CostBars({ verdicts, totals }: { verdicts: Verdict[]; totals: RunReport["snapshot"]["totals"] }) {
  const rows = verdicts.filter((v) => v.cost_before_per_1k != null && v.cost_before_per_1k > 0);
  return (
    <section aria-labelledby="cost-h">
      <h2 id="cost-h" className="text-3xl font-extrabold tracking-tight">
        Cost per 1,000 calls
      </h2>
      <p className="mt-1 mb-5 font-hand text-lg text-muted">
        Hatched bar: estimated cost on the source model, from its published price. Coloured bar: the Nemotron tier it
        moved to. Each row is scaled to its own before cost.
      </p>
      <ul className="flex flex-col gap-4">
        {rows.map((v) => (
          <BarRow
            key={v.site_id}
            label={v.site_id}
            before={v.cost_before_per_1k!}
            after={v.decision === "MIGRATE" ? v.cost_after_per_1k : null}
            mark={v.decision === "MIGRATE" && v.tier ? TIER_HL[v.tier] : null}
          />
        ))}
        {totals && totals.est_cost_before_per_1k > 0 && (
          <BarRow label="Total" before={totals.est_cost_before_per_1k} after={totals.est_cost_after_per_1k} mark="var(--hl-lime)" />
        )}
      </ul>
    </section>
  );
}

// Measured Nemotron p50 on the chosen tier's holdout run. "Before" was not measured, so there is no before bar.
function LatencyAfter({ verdicts, results }: { verdicts: Verdict[]; results: TierResult[] }) {
  const rows = verdicts.flatMap((v) => {
    const r = v.decision === "MIGRATE" ? holdoutResult(results, v) : null;
    return r && v.tier ? [{ v, tier: v.tier, p50: r.metrics.p50_ms }] : [];
  });
  const max = Math.max(1, ...rows.map((x) => x.p50));
  return (
    <section aria-labelledby="lat-h">
      <h2 id="lat-h" className="text-3xl font-extrabold tracking-tight">
        Latency on Nemotron, p50
      </h2>
      <p className="mt-1 mb-5 font-hand text-lg text-muted">
        Measured on each call&apos;s chosen tier over its holdout cases. Latency before was not measured.
      </p>
      {rows.length === 0 ? (
        <p className="font-hand text-lg text-muted">No call was migrated, so there is no Nemotron latency to show.</p>
      ) : (
        <ul className="flex flex-col gap-3">
          {rows.map(({ v, tier, p50 }) => {
            const d = v.latency_delta_ms;
            return (
              <li key={v.site_id} className="grid grid-cols-[3.5rem_minmax(0,1fr)] items-center gap-3">
                <span className="font-mono text-sm font-bold">{v.site_id}</span>
                <span className="flex flex-wrap items-center gap-2">
                  <span
                    aria-hidden="true"
                    className="sketch-box h-3.5 border-2 bg-[var(--mark)]"
                    style={{ "--mark": TIER_HL[tier], width: `${Math.max(4, (p50 / max) * 55)}%` } as CSSProperties}
                  />
                  <span className="font-mono text-sm">{ms(p50)}</span>
                  <span className="font-hand text-base">
                    on {TIER_SHORT[tier]}
                    {d != null && `, ${signedMs(d)} ${d < 0 ? "faster" : d > 0 ? "slower" : "same"}`}
                  </span>
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
