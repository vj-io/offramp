"use client";

import { useState, type FormEvent } from "react";
import { FilePath } from "@/components/file-path";
import { fileUrl, ratioText, runScan, tierName, type ScanResult } from "@/lib/scan";

const COLUMNS = ["Site", "File", "Provider and model", "Call shape", "Flags"];

export function ScanForm({ demoRepoUrl }: { demoRepoUrl: string }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);

  async function scan(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    const out = await runScan(url);
    setBusy(false);
    if (out.ok) setResult(out.result);
    else setError(out.message);
  }

  return (
    <div className="flex min-w-0 flex-col gap-6">
      <form onSubmit={scan} className="flex max-w-3xl flex-col gap-2">
        <label htmlFor="github-url" className="font-hand text-xl">
          Public GitHub repository
        </label>
        <div className="flex flex-wrap gap-3">
          <input
            id="github-url"
            name="github_url"
            required
            inputMode="url"
            autoComplete="off"
            spellCheck={false}
            placeholder="https://github.com/owner/repo"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="sketch-box min-w-0 flex-1 basis-64 bg-paper px-3 py-2 font-mono text-sm text-ink"
          />
          <button
            type="submit"
            disabled={busy}
            className="sketch cursor-pointer bg-ink px-5 py-2 font-hand text-xl text-paper hover:opacity-90 disabled:cursor-wait disabled:opacity-60"
          >
            {busy ? "Scanning…" : "Scan"}
          </button>
        </div>
        <div className="flex flex-wrap items-baseline gap-x-4">
          <button
            type="button"
            onClick={() => setUrl(demoRepoUrl)}
            className="font-hand text-lg text-bluepen underline underline-offset-4"
          >
            Use demo repo
          </button>
          <span className="font-hand text-base text-muted">
            Add /tree/&lt;branch&gt; for a branch. Up to 25 MB and 5,000 Python files.
          </span>
        </div>
      </form>

      <div aria-live="polite" className="min-w-0">
        {busy && <p className="font-hand text-xl text-muted">Downloading and reading the Python files (up to 20 s)…</p>}
        {error && (
          <p role="alert" className="font-hand text-xl text-redpen">
            {error}
          </p>
        )}
        {result && <ScanTable result={result} />}
      </div>
    </div>
  );
}

function ScanTable({ result }: { result: ScanResult }) {
  const name = result.repo.replace("https://github.com/", "");
  const at = result.commit ?? result.ref;
  const shown = result.commit ? result.commit.slice(0, 7) : result.ref;
  const refs = result.references;
  const tiers = refs.map((r) => tierName(r.tier)).join(" / ");
  const prices = refs.map((r) => `${tierName(r.tier)} $${r.input_usd_per_m.toFixed(2)}/$${r.output_usd_per_m.toFixed(2)}`);
  const statuses = [...new Set(refs.map((r) => r.price_status.replaceAll("-", " ")))].join(" and ");
  const guessed = result.rows.some((r) => r.shape_guessed);

  if (result.rows.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <h2 className="text-2xl font-extrabold tracking-tight">No call sites found in {name}.</h2>
        <p className="font-hand text-lg leading-snug">
          Read {result.python_files} Python files at <span className="font-mono text-sm">{shown}</span> and found
          no OpenAI, Anthropic, LiteLLM or LangChain call. Calls in notebooks, or made through a wrapper defined in
          another package, are not detected.
        </p>
        <Warnings warnings={result.warnings} total={result.warnings_total} />
      </div>
    );
  }

  return (
    <section aria-labelledby="scan-h" className="flex flex-col gap-3">
      <h2 id="scan-h" className="text-2xl font-extrabold tracking-tight">
        {result.rows_total} call {result.rows_total === 1 ? "site" : "sites"} in{" "}
        <a href={result.repo} className="underline underline-offset-4">
          {name}
        </a>
      </h2>
      <p className="font-hand text-lg text-muted">
        {result.python_files} Python files read at <span className="font-mono text-sm">{shown}</span>. The code
        was not run.
        {result.rows_total > result.rows.length && ` Showing the first ${result.rows.length.toLocaleString("en")}.`}
      </p>
      <div role="region" tabIndex={0} aria-label="Call sites, scrolls sideways" className="overflow-x-auto pb-2">
        <table className="w-full min-w-[48rem] border-collapse text-left">
          <thead className="font-hand text-lg">
            <tr className="border-b-2 border-ink">
              {[...COLUMNS, `List-price ratio vs ${tiers}*`].map((h) => (
                <th key={h} scope="col" className="px-2 py-2 align-bottom font-normal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.rows.map((r) => (
              <tr key={r.id} className="border-b border-ink/20 align-top">
                <th scope="row" className="px-2 py-2.5 font-mono text-sm font-bold">
                  {r.id}
                </th>
                <td className="min-w-[10rem] px-2 py-2.5 font-mono text-sm wrap-anywhere">
                  <a href={fileUrl(result.repo, at, r.file, r.line)} className="underline decoration-ink/30 underline-offset-4">
                    <FilePath path={r.file} range={String(r.line)} />
                  </a>
                </td>
                <td className="px-2 py-2.5 font-mono text-sm">
                  {r.sdk} {r.model ?? <span className="text-muted">(model set at runtime)</span>}
                </td>
                <td className="px-2 py-2.5 font-hand text-lg whitespace-nowrap">
                  {r.shape}
                  {r.shape_guessed && <span title="guessed from the prompt text">~</span>}
                </td>
                <td className="px-2 py-2.5 font-hand text-base">{r.flags.join(", ") || "—"}</td>
                <td className={`px-2 py-2.5 text-sm ${r.price_ratio ? "font-mono" : "font-hand text-base text-muted"}`}>
                  {ratioText(r)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="max-w-4xl text-sm leading-relaxed text-muted">
        * {result.ratio_label}: the source model&apos;s list price per token over each Nemotron
        tier&apos;s ({prices.join(", ")} per 1M input/output tokens; {statuses}, as of {refs[0]?.as_of}). A range is
        the input and the output token ratio; below 1× the source model lists cheaper.
        {guessed && " ~ guessed from the prompt text; a full run labels each call."}
      </p>
      <Warnings warnings={result.warnings} total={result.warnings_total} />
    </section>
  );
}

function Warnings({ warnings, total }: { warnings: string[]; total: number }) {
  if (warnings.length === 0) return null;
  return (
    <details className="font-mono text-xs text-muted">
      <summary className="cursor-pointer font-hand text-base">
        {total} {total === 1 ? "file" : "files"} skipped{total > warnings.length && `, first ${warnings.length} listed`}
      </summary>
      <ul className="mt-1 flex flex-col gap-0.5">
        {warnings.map((w) => (
          <li key={w}>{w}</li>
        ))}
      </ul>
    </details>
  );
}
