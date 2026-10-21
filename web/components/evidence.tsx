import { Fragment, type CSSProperties, type ReactNode } from "react";
import {
  afterCells,
  dayText,
  judgeText,
  keptText,
  notEvaluatedText,
  pct100,
  TASK_NAME,
  TIER_NAME,
  verdictBounds,
} from "@/lib/format";
import type { CallSite, NotEvaluated, RunConfig, TaskType, TierResult, Verdict } from "@/lib/types";
import { SunsetBadge } from "./badges";
import { Stamp, Swipe, TIER_HL } from "./doodle";
import { FilePath } from "./file-path";

// A kept call's sunset notice already sits in its card above the table; only the others get a footnote under it.
export const hasSunsetFootnote = (v: Verdict) => Boolean(v.sunset_risk) && v.decision !== "KEEP_CLOSED";

export function holdoutResult(results: TierResult[], v: Verdict) {
  return results.findLast((r) => r.site_id === v.site_id && r.tier === v.tier && r.split === "holdout") ?? null;
}

const COLUMNS = [
  "File",
  "Task",
  "Before",
  "After",
  "Parity vs reference (holdout)",
  "p50 latency (after)",
  "$/1k calls before (est.) → after",
] as const;
type Column = (typeof COLUMNS)[number];

// "note" and "source" footnote links, at least 24 px tall for a thumb (WCAG 2.5.8).
const FOOTNOTE = "inline-block py-0.5 font-hand text-base text-bluepen underline underline-offset-2";

// The parity cell: free text's judge figures, one unbroken figure per line where it wraps; else the parity
// figure with its holdout count and lower bound under it.
function ParityCell({ v, task }: { v: Verdict; task: TaskType | undefined }) {
  const judged = task === "free_text" ? judgeText(v) : null;
  if (judged)
    return judged.split(", ").map((part, i) => (
      <Fragment key={i}>
        {i > 0 && ", "}
        <span className="whitespace-nowrap">{part}</span>
      </Fragment>
    ));
  const bounds = verdictBounds(v, task);
  return (
    <>
      {pct100(v.parity_pct!)}
      {bounds && <span className="block text-xs whitespace-nowrap">({bounds})</span>}
    </>
  );
}

function cells(v: Verdict, site: CallSite | undefined, results: TierResult[], config: RunConfig | null, today: string): Record<Column, ReactNode> {
  const task = site?.contract?.task_type;
  const isKept = v.decision === "KEEP_CLOSED";
  const after = afterCells(v, holdoutResult(results, v)?.metrics.p50_ms ?? null);
  return {
    File: site ? <FilePath path={site.file} /> : "—",
    Task: task ? TASK_NAME[task] : "—",
    Before: (
      <span className="flex flex-col items-start gap-2">
        <span className="font-mono text-sm wrap-anywhere">{site?.source_model ?? "—"}</span>
        {v.sunset_risk && (
          <span className="flex flex-wrap items-center gap-x-2">
            <SunsetBadge />
            {v.retires_on && (
              <span className="font-hand text-base">
                {v.retires_on < today ? "retired" : "retires"} {dayText(v.retires_on)}
              </span>
            )}
            {hasSunsetFootnote(v) && (
              <a href={`#sunset-note-${v.site_id}`} aria-label={`Sunset risk note for ${v.site_id}`} className={FOOTNOTE}>
                note
              </a>
            )}
          </span>
        )}
      </span>
    ),
    After:
      v.decision === "MIGRATE" && v.tier ? (
        <span className="flex flex-col items-start gap-0.5">
          <span className="text-lg font-extrabold tracking-tight sm:whitespace-nowrap">
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
      ),
    "Parity vs reference (holdout)": isKept ? (
      keptText(v, task, config)
    ) : v.parity_pct == null ? (
      "—"
    ) : (
      <>
        <ParityCell v={v} task={task} />
        {v.iterations > 0 && (
          <span className="font-hand text-base">
            {" "}
            after {v.iterations} adapter iteration{v.iterations === 1 ? "" : "s"}
          </span>
        )}
      </>
    ),
    "p50 latency (after)": after.latency,
    "$/1k calls before (est.) → after": (
      <>
        {after.cost}
        {v.cost_before_note && (
          <a href={`#cost-note-${v.site_id}`} aria-label={`Price source for ${v.site_id}`} className={`ml-1 ${FOOTNOTE}`}>
            source
          </a>
        )}
      </>
    ),
  };
}

const rowTint = (v: Verdict): CSSProperties | undefined =>
  v.decision === "KEEP_CLOSED"
    ? { background: "var(--note-pink)" }
    : v.decision === "MANUAL_REVIEW"
      ? { background: "var(--note-orange)" }
      : undefined;

const CELL: Record<Column, string> = {
  File: "min-w-[9rem] font-mono text-sm wrap-anywhere",
  Task: "font-hand text-lg",
  Before: "min-w-[10rem]",
  After: "",
  "Parity vs reference (holdout)": "font-mono text-sm",
  "p50 latency (after)": "font-mono text-sm whitespace-nowrap",
  "$/1k calls before (est.) → after": "font-mono text-sm whitespace-nowrap",
};

// The same cell fonts on a phone card, without the table's widths and no-wrap.
const CARD: Record<Column, string> = { ...CELL, File: "font-mono text-sm", Before: "", "p50 latency (after)": "font-mono text-sm", "$/1k calls before (est.) → after": "font-mono text-sm" };

/** The evidence table; below 640 px one card per site, so a phone reaches every column without scrolling. Calls the
 *  run found but did not evaluate (an embedding call) get a row of their own, from the run's log. */
export function EvidenceTable({
  verdicts,
  sites,
  results,
  config,
  notEvaluated,
  today,
}: {
  verdicts: Verdict[];
  sites: Record<string, CallSite>;
  results: TierResult[];
  config: RunConfig | null;
  notEvaluated: NotEvaluated[];
  today: string;
}) {
  const rows = verdicts.map((v) => ({ v, c: cells(v, sites[v.site_id], results, config, today) }));
  const withBounds = verdicts.some((v) => verdictBounds(v));
  return (
    <>
      <div role="region" tabIndex={0} aria-label="Evidence table, scrolls sideways" className="relative hidden overflow-x-auto pb-2 sm:block">
        <table className="w-full min-w-[60rem] border-collapse text-left">
          <thead className="font-hand text-lg">
            <tr className="border-b-2 border-ink">
              {["Site", ...COLUMNS].map((h) => (
                <th key={h} scope="col" className="px-2 py-2 align-bottom font-normal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ v, c }) => (
              <tr key={v.site_id} className="border-b border-dashed border-muted align-top" style={rowTint(v)}>
                <th scope="row" className="px-2 py-3 font-mono text-sm font-bold">
                  {v.site_id}
                </th>
                {COLUMNS.map((col) => (
                  <td key={col} className={`px-2 py-3 ${CELL[col]}`}>
                    {c[col]}
                  </td>
                ))}
              </tr>
            ))}
            {notEvaluated.map((n) => (
              <tr key={n.id} className="border-b border-dashed border-muted align-top text-muted">
                <th scope="row" className="px-2 py-3 font-mono text-sm font-bold">
                  {n.id}
                </th>
                <td className={`px-2 py-3 ${CELL.File}`}>
                  <FilePath path={n.file} />
                </td>
                <td className="px-2 py-3 font-hand text-lg">{n.kind}</td>
                <td colSpan={COLUMNS.length - 2} className="px-2 py-3 font-hand text-lg">
                  {notEvaluatedText(n)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="flex flex-col gap-4 sm:hidden" aria-label="Evidence, one card per call site">
        {rows.map(({ v, c }) => (
          <li key={v.site_id} className="sketch-box min-w-0 p-4" style={rowTint(v) ?? { background: "var(--note-plain)" }}>
            <h3 className="font-mono text-sm font-bold">{v.site_id}</h3>
            <dl className="mt-2 grid grid-cols-[7.5rem_minmax(0,1fr)] gap-x-3 gap-y-2">
              {COLUMNS.map((col) => (
                <Fragment key={col}>
                  <dt className="font-hand text-base leading-snug text-muted">{col}</dt>
                  <dd className={`min-w-0 wrap-anywhere ${CARD[col]}`}>{c[col]}</dd>
                </Fragment>
              ))}
            </dl>
          </li>
        ))}
        {notEvaluated.map((n) => (
          <li key={n.id} className="sketch-box min-w-0 border-dashed p-4 text-muted">
            <h3 className="font-mono text-sm font-bold">{n.id}</h3>
            <p className="mt-1 font-mono text-sm wrap-anywhere">{n.file}</p>
            <p className="mt-1 font-hand text-lg">{notEvaluatedText(n)}</p>
          </li>
        ))}
      </ul>
      {withBounds && (
        <p className="mt-3 font-hand text-lg text-muted">
          In brackets: (passed/n, ≥ lower 95% bound).{" "}
          <a href="#bounds-note" className="text-bluepen underline underline-offset-2">
            What that bound says
          </a>
        </p>
      )}
    </>
  );
}
