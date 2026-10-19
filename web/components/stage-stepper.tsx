"use client";

import { useEffect, useRef } from "react";
import { STAGE_NAME } from "@/lib/format";
import type { StageStatus } from "@/lib/run-state";
import type { Stage } from "@/lib/types";
import { checkPath, circlePath, crossPath, Ink, InkIcon } from "./doodle";

const STATUS_TEXT: Record<StageStatus, string> = {
  done: "done",
  current: "current stage",
  pending: "not started",
  failed: "failed here",
};

export function StageStepper({
  stages,
  failureReason,
}: {
  stages: { stage: Stage; status: StageStatus }[];
  failureReason?: string;
}) {
  const region = useRef<HTMLDivElement>(null);
  const active = stages.find((s) => s.status === "current" || s.status === "failed")?.stage;
  // Keep the current stage in view on narrow screens (sideways only, never moves the page).
  useEffect(() => {
    const el = region.current?.querySelector<HTMLElement>(`[data-stage="${active}"]`);
    if (el && region.current) region.current.scrollLeft = el.offsetLeft - region.current.clientWidth / 2;
  }, [active]);
  return (
    <div>
      <div ref={region} role="region" tabIndex={0} aria-label="Run stages, scrolls sideways" className="relative -mx-4 overflow-x-auto px-4 pt-1 pb-2">
        <ol className="flex min-w-max items-start gap-1" aria-label="Run stages">
          {stages.map(({ stage, status }, i) => (
            <li
              key={stage}
              data-stage={stage}
              aria-current={status === "current" ? "step" : undefined}
              className="flex items-start gap-1"
            >
              {i > 0 && (
                <span
                  aria-hidden="true"
                  className={`mt-4 h-0 w-4 border-t-2 sm:w-6 ${status === "pending" ? "border-dashed border-muted" : "border-ink"}`}
                />
              )}
              <span className="flex w-[4.75rem] flex-col items-center gap-1 text-center">
                <span className="relative grid h-9 w-9 place-items-center font-mono text-sm">
                  {status === "done" ? (
                    <InkIcon d={checkPath(stage)} className="h-6 w-6" />
                  ) : status === "failed" ? (
                    <InkIcon d={crossPath(stage)} className="h-6 w-6 text-redpen" />
                  ) : (
                    <span className={status === "pending" ? "text-muted" : ""}>{i + 1}</span>
                  )}
                  {status === "current" && <Ink d={circlePath(stage)} className="inset-0" width={2.5} />}
                </span>
                <span
                  className={`font-hand text-base leading-tight ${status === "pending" ? "text-muted" : ""} ${status === "failed" ? "text-redpen" : ""}`}
                >
                  {STAGE_NAME[stage]}
                  <span className="sr-only">, {STATUS_TEXT[status]}</span>
                </span>
              </span>
            </li>
          ))}
        </ol>
      </div>
      {failureReason !== undefined && (
        <p role="alert" className="mt-2 font-hand text-lg text-redpen">
          Run failed: {failureReason}
        </p>
      )}
    </div>
  );
}
