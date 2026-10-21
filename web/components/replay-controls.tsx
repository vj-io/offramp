"use client";

import { useRef } from "react";
import { SPEEDS } from "@/lib/run-state";
import type { Playback } from "@/lib/use-run";

const BUTTON = "sketch cursor-pointer px-3 py-0.5 hover:bg-note-plain aria-disabled:cursor-default aria-disabled:opacity-45";

/** WCAG 2.2.2 for the auto-playing replay: pause, restart, speed and skip to the end, all plain buttons. They stay
 *  mounted when the replay ends (Pause and Skip are then aria-disabled, not removed), so keyboard focus never
 *  falls back to the page; Skip to end hands focus to Restart. */
export function ReplayControls({ playback, ended }: { playback: Playback; ended: boolean }) {
  const { paused, setPaused, speed, setSpeed, skipToEnd, restart } = playback;
  const restartButton = useRef<HTMLButtonElement>(null);
  return (
    <div className="flex flex-col gap-1">
      <div role="group" aria-label="Replay controls" className="flex flex-wrap items-center gap-2 font-hand text-lg">
        <button
          type="button"
          aria-disabled={ended}
          onClick={() => !ended && setPaused(!paused)}
          className={`${BUTTON} min-w-[5.5rem]`}
        >
          {paused || ended ? "Play" : "Pause"}
        </button>
        <button ref={restartButton} type="button" onClick={restart} className={BUTTON}>
          Restart
        </button>
        <span id="replay-speed" className="ml-1 text-muted">
          Speed
        </span>
        <span role="group" aria-labelledby="replay-speed" className="flex gap-1">
          {SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              aria-pressed={speed === s}
              onClick={() => setSpeed(s)}
              className={`${BUTTON} px-2.5 font-mono text-sm ${speed === s ? "bg-ink text-paper hover:bg-ink" : ""}`}
            >
              {s}×
            </button>
          ))}
        </span>
        <button
          type="button"
          aria-disabled={ended}
          onClick={() => {
            if (ended) return;
            skipToEnd();
            restartButton.current?.focus();
          }}
          className={BUTTON}
        >
          Skip to end
        </button>
      </div>
      <p role="status" className="font-hand text-base text-muted">
        {ended ? "Replay finished." : paused ? "Paused. Press Play to watch the run, or Skip to end for the result." : ""}
      </p>
    </div>
  );
}
