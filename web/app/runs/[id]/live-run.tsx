"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { LogStream } from "@/components/log-stream";
import { ReplayControls } from "@/components/replay-controls";
import { NoRecording, NoticeNote, PolicyNote, RunLinks, SourceLabel } from "@/components/run-notices";
import { SiteCard } from "@/components/site-card";
import { StageStepper } from "@/components/stage-stepper";
import { TierHeatmap } from "@/components/tier-heatmap";
import { notEvaluatedText, repoName } from "@/lib/format";
import { pickSource, siteView, stageProgress, type RunSource } from "@/lib/run-state";
import { API_URL, useRun } from "@/lib/use-run";

export function LiveRun() {
  const { id } = useParams<{ id: string }>();
  const query = useSearchParams().toString();
  // Keyed so a different run or mode starts from a clean state.
  const q = new URLSearchParams(query);
  return <RunBody key={`${id}?${query}`} id={id} source={pickSource(q)} replay={q.get("replay")} query={query} />;
}

function RunBody({ id, source, replay, query }: { id: string; source: RunSource; replay: string | null; query: string }) {
  const { state, status, retry, playback } = useRun(id, source, replay);

  if (status.kind === "missing") return <NoRecording replay={replay} />;
  if (status.kind === "error") return <NoticeNote title="This run could not be loaded" tone="warn">{status.message}</NoticeNote>;

  const views = Object.keys(state.sites).map((siteId) => siteView(state, siteId));
  const hasVerdicts = Object.keys(state.verdicts).length > 0;
  const reportHref = `/runs/${encodeURIComponent(id)}/report${query ? `?${query}` : ""}`;

  return (
    <div className="flex flex-col gap-8">
      <section className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex min-w-0 flex-col gap-2">
          <PolicyNote replay={source === "replay" ? replay : null} tiers={state.config?.tiers ?? null} />
          <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl">
            {state.config ? (
              <>
                Offramp on{" "}
                <span className="block font-mono text-2xl font-bold wrap-anywhere sm:inline sm:text-3xl">
                  {repoName(state.config.repo_url)}
                </span>
              </>
            ) : (
              <>
                Run <span className="font-mono text-2xl font-bold sm:text-3xl">{state.runId ?? id}</span>
              </>
            )}
          </h1>
          <RunLinks repoUrl={state.config?.repo_url ?? null} prUrl={state.pr?.url ?? null} />
          <SourceLabel source={source} runId={state.runId} apiUrl={API_URL} />
          <ConnectionNote kind={status.kind} onRetry={retry} />
          {source !== "api" && (status.kind === "playing" || status.kind === "ended") && (
            <ReplayControls playback={playback} ended={status.kind === "ended"} />
          )}
        </div>
        {hasVerdicts && (
          <Link href={reportHref} className="sketch bg-ink px-4 py-2 font-hand text-xl text-paper hover:opacity-90">
            {state.stage === "DONE" ? "Open the report" : "Open the report so far"}
          </Link>
        )}
      </section>

      <section aria-labelledby="stages-h">
        <h2 id="stages-h" className="sr-only">
          Stages
        </h2>
        <StageStepper
          stages={stageProgress(state)}
          failureReason={state.stage === "FAILED" ? (state.failure?.reason ?? "") : undefined}
        />
      </section>

      <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,25rem)]">
        <section aria-labelledby="sites-h" className="min-w-0">
          <h2 id="sites-h" className="mb-5 text-2xl font-extrabold tracking-tight">
            Call sites
            {views.length > 0 && (
              <span className="ml-2 font-mono text-lg font-bold">
                <span className="sr-only">: </span>
                {views.length}
                {state.notEvaluated.length > 0 && (
                  <span className="font-hand text-base font-normal text-muted"> + {state.notEvaluated.length} not evaluated</span>
                )}
              </span>
            )}
          </h2>
          {views.length === 0 ? (
            <p className="font-hand text-xl text-muted">The scanner lists every OpenAI and Anthropic call once the repo is cloned.</p>
          ) : (
            <ul className="grid gap-6 sm:grid-cols-2">
              {views.map((v, i) => (
                <SiteCard key={v.site.id} view={v} index={i} config={state.config} />
              ))}
              {state.notEvaluated.map((n) => (
                <li key={n.id} className="sketch-box flex min-w-0 flex-col gap-1 border-dashed p-4 text-muted">
                  <h3 className="font-mono text-sm wrap-anywhere">
                    <span className="font-bold">{n.id}</span> {n.file}
                  </h3>
                  <p className="font-hand text-lg leading-snug">
                    {n.kind} call: {notEvaluatedText(n)}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </section>

        <div className="flex min-w-0 flex-col gap-10">
          <section aria-labelledby="heat-h">
            <h2 id="heat-h" className="mb-3 text-2xl font-extrabold tracking-tight">
              Parity by tier
            </h2>
            {views.length === 0 ? (
              <p className="font-hand text-lg text-muted">Fills in as each tier is evaluated, cheapest first.</p>
            ) : (
              <TierHeatmap views={views} config={state.config} />
            )}
          </section>
          <section aria-labelledby="log-h">
            <h2 id="log-h" className="mb-3 text-2xl font-extrabold tracking-tight">
              Log
            </h2>
            <LogStream events={state.events} sites={state.sites} quiet={status.kind === "playing"} />
          </section>
        </div>
      </div>
    </div>
  );
}

function ConnectionNote({ kind, onRetry }: { kind: string; onRetry: () => void }) {
  if (kind === "connecting" || kind === "reconnecting")
    return (
      <p className="font-hand text-lg text-muted" role="status">
        {kind === "connecting" ? "Connecting to the Offramp API." : "Reconnecting to the Offramp API."} No backend
        running?{" "}
        <Link href="/runs/demo?replay=demo" className="text-bluepen underline underline-offset-4">
          Watch the recorded demo run
        </Link>
      </p>
    );
  if (kind === "lost")
    return (
      <p className="flex flex-wrap items-center gap-3 font-hand text-lg text-redpen" role="alert">
        Lost the connection to the Offramp API at {API_URL}.
        <button type="button" onClick={onRetry} className="sketch cursor-pointer px-3 py-0.5 text-ink">
          Reconnect
        </button>
      </p>
    );
  if (kind === "loading") return <p className="font-hand text-lg text-muted">Loading the recording…</p>;
  return null;
}
