"use client";

import { useState, type CSSProperties, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { TIER_HL } from "@/components/doodle";
import { TIER_SHORT } from "@/lib/format";
import { TIER_ORDER, type RunConfig, type Tier } from "@/lib/types";
import { API_URL } from "@/lib/use-run";

const field = "sketch-box w-full bg-paper px-3 py-2 font-mono text-sm text-ink";
const label = "font-hand text-lg";

export function NewRunForm({ demoRepoUrl }: { demoRepoUrl: string }) {
  const router = useRouter();
  const [repo, setRepo] = useState("");
  // Number fields hold their text so clearing one never puts NaN in a controlled input.
  const [threshold, setThreshold] = useState("95");
  const [reasoning, setReasoning] = useState("90");
  const [tiers, setTiers] = useState<Tier[]>([...TIER_ORDER]);
  const [maxIterations, setMaxIterations] = useState("3");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function start(e: FormEvent) {
    e.preventDefault();
    if (tiers.length === 0) return setError("Pick at least one Nemotron tier.");
    const config: Omit<RunConfig, "open_pr" | "live"> = {
      repo_url: repo.trim(),
      threshold: Number(threshold) / 100,
      reasoning_threshold: Number(reasoning) / 100,
      tiers: TIER_ORDER.filter((t) => tiers.includes(t)),
      max_iterations: Number(maxIterations),
    };
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/runs`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(config),
      });
      if (res.status === 429) throw new Error("The API allows 30 new runs per minute from one address. Try again in a minute.");
      if (!res.ok) throw new Error(`The Offramp API returned HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
      const { run_id } = (await res.json()) as { run_id: string };
      router.push(`/runs/${encodeURIComponent(run_id)}`);
    } catch (err) {
      setBusy(false);
      setError(
        err instanceof TypeError
          ? `Could not reach the Offramp API at ${API_URL}. Start the backend, or watch the recorded demo run.`
          : (err as Error).message,
      );
    }
  }

  return (
    <form onSubmit={start} className="flex flex-col gap-5">
      <div className="flex flex-col gap-1.5">
        <label htmlFor="repo" className={label}>
          Python repository
        </label>
        <input
          id="repo"
          name="repo"
          required
          inputMode="url"
          autoComplete="off"
          spellCheck={false}
          placeholder="https://github.com/owner/repo"
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
          className={field}
        />
        <button
          type="button"
          onClick={() => setRepo(demoRepoUrl)}
          className="self-start font-hand text-lg text-bluepen underline underline-offset-4"
        >
          Use demo repo
        </button>
      </div>

      <fieldset className="flex flex-col gap-2">
        <legend className={label}>Parity needed to migrate</legend>
        <div className="grid grid-cols-2 gap-3">
          <label className="flex flex-col gap-1 font-hand text-base text-muted">
            Most calls, %
            <input type="number" min={50} max={100} step={0.5} required value={threshold} onChange={(e) => setThreshold(e.target.value)} className={field} />
          </label>
          <label className="flex flex-col gap-1 font-hand text-base text-muted">
            Reasoning calls, %
            <input type="number" min={50} max={100} step={0.5} required value={reasoning} onChange={(e) => setReasoning(e.target.value)} className={field} />
          </label>
        </div>
      </fieldset>

      <fieldset className="flex flex-col gap-2">
        <legend className={label}>Nemotron tiers to try, cheapest first</legend>
        <div className="flex flex-wrap gap-3">
          {TIER_ORDER.map((t) => (
            <label key={t} className="flex cursor-pointer items-center gap-2">
              <input
                type="checkbox"
                checked={tiers.includes(t)}
                onChange={(e) => setTiers(e.target.checked ? [...tiers, t] : tiers.filter((x) => x !== t))}
                className="h-5 w-5 accent-[var(--ink)]"
              />
              <span className="hl-chip px-2 font-hand text-lg" style={{ "--mark": TIER_HL[t] } as CSSProperties}>
                {TIER_SHORT[t]}
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      <label className="flex max-w-[12rem] flex-col gap-1.5">
        <span className={label}>Adapter iterations per tier</span>
        <input type="number" min={0} max={3} required value={maxIterations} onChange={(e) => setMaxIterations(e.target.value)} className={field} />
      </label>

      {error && (
        <p role="alert" className="font-hand text-lg text-redpen">
          {error}
        </p>
      )}
      <button
        type="submit"
        disabled={busy}
        className="sketch cursor-pointer self-start bg-ink px-5 py-2 font-hand text-xl text-paper hover:opacity-90 disabled:cursor-wait disabled:opacity-60"
      >
        {busy ? "Starting the run…" : "Start run"}
      </button>
    </form>
  );
}
