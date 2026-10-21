"use client";

import { useEffect, useReducer, useRef, useState } from "react";
import {
  applyEvents,
  initialRunState,
  parseRecording,
  replayDelays,
  replayEntry,
  toSnapshot,
  type Recording,
  type RunSource,
  type Speed,
} from "./run-state";
import { EVENT_TYPES, type RunEvent, type RunReport } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/** A recording for replay/fixture runs, or null when it has not been published yet (a missing replay
 *  file, or one holding fixture data). `replay` is the ?replay= name; names off the allow-list are refused. */
async function loadRecording(source: Exclude<RunSource, "api">, replay: string | null): Promise<Recording | null> {
  if (source === "fixture") {
    // Dev-only: the import is compiled out of production builds.
    if (process.env.NODE_ENV !== "production") {
      const file = await import("../fixtures/dev-events.json");
      return { events: file.default.events as unknown as RunEvent[], diffs: {} };
    }
    throw new Error("The dev fixture only loads in development builds (pnpm dev).");
  }
  const file = replayEntry(replay)?.file;
  if (!file) throw new Error(`There is no recording called “${replay}”.`);
  const res = await fetch(file, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Could not load ${file} (HTTP ${res.status}).`);
  return parseRecording(await res.json());
}

export type RunStatus =
  | { kind: "loading" } // fetching a recording
  | { kind: "missing" } // replay requested, nothing recorded yet
  | { kind: "error"; message: string }
  | { kind: "playing" } // replay/fixture playback in progress
  | { kind: "connecting" | "open" | "reconnecting" | "lost" } // SSE connection to the API
  | { kind: "ended" };

/** Live run state: SSE from the API, or a recording played back with pause, speed and skip to end. */
export function useRun(id: string, source: RunSource, replay: string | null) {
  const [state, dispatch] = useReducer(applyEvents, initialRunState);
  const [status, setStatus] = useState<RunStatus>({ kind: source === "api" ? "connecting" : "loading" });
  const [attempt, setAttempt] = useState(0);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState<Speed>(4);
  const lastSeq = useRef(-1); // survives effect restarts (route hidden and shown again)
  const recording = useRef<RunEvent[]>([]);
  const next = useRef(0); // index of the next recorded event to play

  useEffect(() => {
    if (source === "api") return;
    let cancelled = false;
    loadRecording(source, replay).then(
      (rec) => {
        if (cancelled) return;
        if (!rec) return setStatus({ kind: "missing" });
        recording.current = rec.events;
        next.current = rec.events.findIndex((e) => e.seq > lastSeq.current);
        if (next.current < 0) {
          next.current = rec.events.length;
          return setStatus({ kind: "ended" });
        }
        // WCAG 2.2.2: nothing moves on its own for someone who asked for less motion.
        if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) setPaused(true);
        setStatus({ kind: "playing" });
      },
      (err: Error) => !cancelled && setStatus({ kind: "error", message: err.message }),
    );
    return () => {
      cancelled = true;
    };
  }, [source, replay]);

  const running = status.kind === "playing" && !paused;
  useEffect(() => {
    if (!running) return;
    const events = recording.current;
    const delays = replayDelays(events, speed);
    let timer: ReturnType<typeof setTimeout>;
    const step = () => {
      lastSeq.current = events[next.current].seq;
      dispatch([events[next.current]]);
      next.current += 1;
      if (next.current >= events.length) setStatus({ kind: "ended" });
      else timer = setTimeout(step, delays[next.current]);
    };
    timer = setTimeout(step, delays[next.current]);
    return () => clearTimeout(timer);
  }, [running, speed]);

  const skipToEnd = () => {
    const rest = recording.current.slice(next.current);
    if (rest.length > 0) {
      lastSeq.current = rest[rest.length - 1].seq;
      dispatch(rest);
    }
    next.current = recording.current.length;
    setStatus({ kind: "ended" });
  };

  useEffect(() => {
    if (source !== "api") return;
    // EventSource reconnects by itself and sends Last-Event-ID; the reducer drops any repeat seq.
    const es = new EventSource(`${API_URL}/api/runs/${encodeURIComponent(id)}/events`);
    const onEvent = (m: Event) => {
      // "error" is both the run's error event (has data) and EventSource's connection error.
      if (!(m instanceof MessageEvent) || typeof m.data !== "string") {
        setStatus({ kind: es.readyState === EventSource.CLOSED ? "lost" : "reconnecting" });
        return;
      }
      const e = JSON.parse(m.data) as RunEvent;
      lastSeq.current = Math.max(lastSeq.current, e.seq);
      dispatch([e]);
      if (e.type === "done" || e.type === "error") {
        es.close();
        setStatus({ kind: "ended" });
      }
    };
    es.onopen = () => setStatus({ kind: "open" });
    for (const t of EVENT_TYPES) es.addEventListener(t, onEvent);
    return () => es.close();
  }, [id, source, attempt]);

  const retry = () => {
    setStatus({ kind: "connecting" });
    setAttempt((n) => n + 1);
  };
  return { state, status, retry, playback: { paused, setPaused, speed, setSpeed, skipToEnd } };
}

export type Playback = ReturnType<typeof useRun>["playback"];

export type ReportStatus =
  | { kind: "loading" }
  | { kind: "missing" }
  | { kind: "error"; message: string }
  | { kind: "ready"; report: RunReport };

/** Report data: the API's report for live runs, rebuilt from the whole recording otherwise. */
export function useReport(id: string, source: RunSource, replay: string | null): ReportStatus {
  const [status, setStatus] = useState<ReportStatus>({ kind: "loading" });
  useEffect(() => {
    let cancelled = false;
    const load: Promise<RunReport | null> =
      source === "api"
        ? fetch(`${API_URL}/api/runs/${encodeURIComponent(id)}/report`).then(async (res) => {
            if (!res.ok) throw new Error(`The Offramp API returned HTTP ${res.status} for this run's report.`);
            return (await res.json()) as RunReport;
          })
        : loadRecording(source, replay).then((rec) =>
            rec
              ? { snapshot: toSnapshot(applyEvents(initialRunState, rec.events)), pr_body_markdown: null, diffs: rec.diffs }
              : null,
          );
    load.then(
      (report) => !cancelled && setStatus(report ? { kind: "ready", report } : { kind: "missing" }),
      (err: Error) =>
        !cancelled &&
        setStatus({
          kind: "error",
          message: err instanceof TypeError ? `Could not reach the Offramp API at ${API_URL}.` : err.message,
        }),
    );
    return () => {
      cancelled = true;
    };
  }, [id, source, replay]);
  return status;
}
