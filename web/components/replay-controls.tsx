"use client";

import { SPEEDS } from "@/lib/run-state";
import type { Playback } from "@/lib/use-run";

const BUTTON = "sketch cursor-pointer px-3 py-0.5 hover:bg-note-plain";

/** WCAG 2.2.2 for the auto-playing replay: pause, speed and skip to the end, all plain buttons. */
export function ReplayControls({ playback }: { playback: Playback }) {
  const { paused, setPaused, speed, setSpeed, skipToEnd } = playback;
  return (
    <div role="group" aria-label="Replay controls" className="flex flex-wrap items-center gap-2 font-hand text-lg">
      <button type="button" onClick={() => setPaused(!paused)} className={`${BUTTON} min-w-[5.5rem]`}>
        {paused ? "Play" : "Pause"}
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
      <button type="button" onClick={skipToEnd} className={BUTTON}>
        Skip to end
      </button>
    </div>
  );
}
