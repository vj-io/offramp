import type { CSSProperties } from "react";
import { keepClosedStamp, scoreText, TASK_NAME, TIER_NAME, TIER_SHORT } from "@/lib/format";
import type { SiteView } from "@/lib/run-state";
import { TIER_ORDER, type RunConfig, type TaskType, type Verdict } from "@/lib/types";
import { BrandBadges } from "./badges";
import { checkPath, crossPath, DECISION_HL, DECISION_NOTE, InkIcon, Note, Stamp, TIER_HL, TIER_NOTE, tiltFor } from "./doodle";

const mark = (m: string) => ({ "--mark": m }) as CSSProperties;

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
          <span className="font-hand text-lg">Stays on {sourceModel ?? "the original model"}</span>
        </>
      )}
      {verdict.decision === "MANUAL_REVIEW" && (
        <span className="hl-chip sketch px-2.5 py-0.5 font-hand text-lg" style={mark(DECISION_HL.MANUAL_REVIEW)}>
          Manual review: {verdict.reason}
        </span>
      )}
      <BrandBadges small />
    </div>
  );
}

export function SiteCard({ view, index, config }: { view: SiteView; index: number; config: RunConfig | null }) {
  const { site, verdict, currentTier, perTier, iterations, cases } = view;
  const task = site.contract?.task_type;
  const note = verdict ? DECISION_NOTE[verdict.decision] : currentTier ? TIER_NOTE[currentTier] : "var(--note-plain)";
  return (
    <Note as="li" note={note} tilt={tiltFor(index)} className="spring-in flex flex-col gap-3 p-4 pt-5">
      <h3 className="font-mono text-sm break-all">
        <span className="font-bold">{site.id}</span> {site.file}:{site.lines.join("–")}
      </h3>
      <p className="font-hand text-lg leading-snug">
        {task ? TASK_NAME[task] : "unlabelled"} call, recorded on{" "}
        <span className="font-mono text-sm">{site.source_model ?? "unknown model"}</span>
      </p>
      <ul className="flex flex-col gap-1.5" aria-label={`${site.id} results per tier`}>
        {TIER_ORDER.map((t) => {
          const r = perTier[t];
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
                  {scoreText(r, task)} {r.passed ? "pass" : "fail"}
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
      {verdict && <VerdictChip verdict={verdict} task={task} sourceModel={site.source_model} config={config} />}
    </Note>
  );
}
