"use client";

import { useEffect, useReducer, useRef, useState } from "react";
import { applyEvents, initialRunState, parseRecording, replayDelays, toSnapshot, type RunSource } from "./run-state";
import { EVENT_TYPES, type RunEvent, type RunReport } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/** Recorded events for replay/fixture runs, or null when no demo run has been recorded yet
 *  (a missing replay file, or one holding fixture data). */
async function loadRecording(source: Exclude<RunSource, "api">): Promise<RunEvent[] | null> {
  if (source === "fixture") {
    // Dev-only: the import is compiled out of production builds.
    if (process.env.NODE_ENV !== "production") {
      const file = await import("../fixtures/dev-events.json");
      return file.default.events as unknown as RunEvent[];
    }
    throw new Error("The dev fixture only loads in development builds (pnpm dev).");
  }
  const res = await fetch("/replay/demo-events.json", { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Could not load /replay/demo-events.json (HTTP ${res.status}).`);
  return parseRecording(await res.json());
}

export type RunStatus =
  | { kind: "loading" } // fetching a recording
  | { kind: "missing" } // replay requested, nothing recorded yet
  | { kind: "error"; message: string }
  | { kind: "playing" } // replay/fixture playback in progress
  | { kind: "connecting" | "open" | "reconnecting" | "lost" } // SSE connection to the API
  | { kind: "ended" };

/** Live run state: SSE from the API, or a recorded log played back at replay speed. */
export function useRun(id: string, source: RunSource) {
  const [state, dispatch] = useReducer(applyEvents, initialRunState);
  const [status, setStatus] = useState<RunStatus>({ kind: source === "api" ? "connecting" : "loading" });
  const [attempt, setAttempt] = useState(0);
  const lastSeq = useRef(-1); // survives effect restarts (route hidden and shown again)

  useEffect(() => {
    if (source === "api") return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    loadRecording(source).then(
      (events) => {
        if (cancelled) return;
        if (!events) return setStatus({ kind: "missing" });
        const delays = replayDelays(events);
        let i = events.findIndex((e) => e.seq > lastSeq.current);
        if (i < 0) return setStatus({ kind: "ended" });
        setStatus({ kind: "playing" });
        const step = () => {
          lastSeq.current = events[i].seq;
          dispatch([events[i]]);
          i += 1;
          if (i >= events.length) setStatus({ kind: "ended" });
          else timer = setTimeout(step, delays[i]);
        };
        timer = setTimeout(step, delays[i]);
      },
      (err: Error) => !cancelled && setStatus({ kind: "error", message: err.message }),
    );
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [source]);

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
  return { state, status, retry };
}

export type ReportStatus =
  | { kind: "loading" }
  | { kind: "missing" }
  | { kind: "error"; message: string }
  | { kind: "ready"; report: RunReport };

/** Report data: the API's report for live runs, rebuilt from the whole recording otherwise. */
export function useReport(id: string, source: RunSource): ReportStatus {
  const [status, setStatus] = useState<ReportStatus>({ kind: "loading" });
  useEffect(() => {
    let cancelled = false;
    const load: Promise<RunReport | null> =
      source === "api"
        ? fetch(`${API_URL}/api/runs/${encodeURIComponent(id)}/report`).then(async (res) => {
            if (!res.ok) throw new Error(`The Offramp API returned HTTP ${res.status} for this run's report.`);
            return (await res.json()) as RunReport;
          })
        : loadRecording(source).then((events) =>
            events
              ? { snapshot: toSnapshot(applyEvents(initialRunState, events)), pr_body_markdown: null }
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
  }, [id, source]);
  return status;
}
