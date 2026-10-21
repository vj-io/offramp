import type { CSSProperties } from "react";
import {
  closedReferenceText,
  keepClosedStamp,
  manualReason,
  referencesText,
  resultBounds,
  scoreText,
  TASK_NAME,
  TIER_NAME,
  TIER_SHORT,
} from "@/lib/format";
import type { SiteView } from "@/lib/run-state";
import { TIER_ORDER, type RunConfig, type TaskType, type Verdict } from "@/lib/types";
import { BrandBadges, CodeText, SunsetRisk } from "./badges";
import { checkPath, crossPath, DECISION_HL, DECISION_NOTE, InkIcon, Note, Stamp, TIER_HL, TIER_NOTE, tiltFor } from "./doodle";
import { FilePath } from "./file-path";

const mark = (m: string) => ({ "--mark": m }) as CSSProperties;

// A reason can hold error text, file paths and tokens with no spaces: wrap anywhere, never widen the card.
// Past this many characters it starts on 3 lines behind a native "more" toggle; the full text stays in the DOM.
// The text sits outside the <summary>, so the toggle's accessible name stays short; the chip is the `group`, and
// group-has-[details[open]] means its own <details> is open.
const LONG_REASON = 120;

// A recorded failure kind reads as the PR's plain sentence, with the raw reason folded under it; an older
// recording's reason shows as it is.
function ManualReviewNote({ verdict }: { verdict: Verdict }) {
  const { sentence, raw } = manualReason(verdict);
  const text = `Manual review: ${sentence}`;
  const chip = "hl-chip sketch max-w-full min-w-0 px-2.5 py-0.5 font-hand text-lg wrap-anywhere";
  if (raw !== null)
    return (
      <div className={chip} style={mark(DECISION_HL.MANUAL_REVIEW)}>
        <CodeText text={text} />
        <details className="text-base">
          <summary className="cursor-pointer underline underline-offset-2">Raw failure</summary>
          <p className="mt-1 font-mono text-xs leading-relaxed whitespace-pre-wrap">{raw.replaceAll(" | ", "\n")}</p>
        </details>
      </div>
    );
  if (text.length <= LONG_REASON)
    return (
      <span className={chip} style={mark(DECISION_HL.MANUAL_REVIEW)}>
        {text}
      </span>
    );
  return (
    <div className={`group ${chip}`} style={mark(DECISION_HL.MANUAL_REVIEW)}>
      <span className="line-clamp-3 group-has-[details[open]]:line-clamp-none">{text}</span>
      <details>
        <summary className="cursor-pointer list-none text-base underline underline-offset-2 [&::-webkit-details-marker]:hidden">
          <span className="group-has-[details[open]]:hidden">more</span>
          <span className="hidden group-has-[details[open]]:inline">less</span>
          <span className="sr-only"> of the manual review reason</span>
        </summary>
      </details>
    </div>
  );
}

export function VerdictChip({
  verdict,
  task,
  sourceModel,
  config,
}: {
  verdict: Verdict;
  task: TaskType | undefined;
  sourceModel: string | null;
  config: RunConfig | null;
}) {
  return (
    <div className="flex flex-col items-start gap-1.5">
      {verdict.decision === "MIGRATE" && verdict.tier && (
        <span className="hl-chip sketch px-2.5 py-0.5 font-hand text-lg" style={mark(DECISION_HL.MIGRATE)}>
          Migrate to {TIER_NAME[verdict.tier]}
        </span>
      )}
      {verdict.decision === "KEEP_CLOSED" && (
        <>
          <Stamp>{keepClosedStamp(verdict, task, config)}</Stamp>
          <span className="font-hand text-lg">Stays on {sourceModel ?? "its source model"}</span>
          {closedReferenceText(verdict) && (
            <span className="font-hand text-base leading-snug text-muted">{closedReferenceText(verdict)}</span>
          )}
        </>
      )}
      {verdict.decision === "MANUAL_REVIEW" && <ManualReviewNote verdict={verdict} />}
      {verdict.sunset_risk && <SunsetRisk note={verdict.sunset_risk} />}
      <BrandBadges small />
    </div>
  );
}

export function SiteCard({ view, index, config }: { view: SiteView; index: number; config: RunConfig | null }) {
  const { site, verdict, currentTier, perTier, iterations, cases } = view;
  const task = site.contract?.task_type;
  const note = verdict ? DECISION_NOTE[verdict.decision] : currentTier ? TIER_NOTE[currentTier] : "var(--note-plain)";
  return (
    <Note as="li" note={note} tilt={tiltFor(index)} className="spring-in flex min-w-0 flex-col gap-3 p-4 pt-5">
      <h3 className="font-mono text-sm wrap-anywhere">
        <span className="font-bold">{site.id}</span> <FilePath path={site.file} range={site.lines.join("–")} />
      </h3>
      <p className="font-hand text-lg leading-snug">
        {task ? TASK_NAME[task] : "unlabelled"} call, source model{" "}
        <span className="font-mono text-sm">{site.source_model ?? "unknown"}</span>
      </p>
      <ul className="flex flex-col gap-1.5" aria-label={`${site.id} results per tier`}>
        {TIER_ORDER.map((t) => {
          const r = perTier[t];
          const bounds = r && resultBounds(r, task);
          return (
            <li key={t} className="flex flex-wrap items-center gap-2">
              <span className="hl-chip w-[5.5rem] px-2 font-hand text-base" style={mark(TIER_HL[t])}>
                {TIER_SHORT[t]}
              </span>
              {r ? (
                <span
                  className="hl-chip inline-flex items-center gap-1.5 px-2 font-mono text-sm"
                  style={mark(r.passed ? "var(--hl-lime)" : "var(--hl-pink)")}
                >
                  <InkIcon d={(r.passed ? checkPath : crossPath)(`${site.id}${t}${r.adapter_version}`)} />
                  {scoreText(r, task)}
                  {bounds && <span className="text-xs">({bounds})</span>} {r.passed ? "pass" : "fail"}
                  <span className="font-hand text-sm">
                    {r.split}
                    {r.adapter_version > 0 ? `, v${r.adapter_version}` : ""}
                  </span>
                </span>
              ) : (
                <span className="font-hand text-base text-muted">not run</span>
              )}
            </li>
          );
        })}
      </ul>
      <p className="font-hand text-base leading-snug text-muted">
        {currentTier ? <>On {TIER_SHORT[currentTier]}</> : "Waiting for the first eval"}
        {iterations > 0 && (
          <>
            , <span className="font-mono text-sm">{iterations}</span> adapter iteration{iterations === 1 ? "" : "s"}
          </>
        )}
        {cases && (
          <>
            , <span className="font-mono text-sm">{cases.dev}</span> dev /{" "}
            <span className="font-mono text-sm">{cases.holdout}</span> holdout cases
          </>
        )}
      </p>
      {cases && (
        <p className="font-hand text-base leading-snug text-muted">
          Parity vs {referencesText(cases)}
          {cases.audit?.checked > 0 && (
            <>
              , audit {cases.audit.agreed} of {cases.audit.checked} agreed
            </>
          )}
        </p>
      )}
      {verdict && <VerdictChip verdict={verdict} task={task} sourceModel={site.source_model} config={config} />}
    </Note>
  );
}
