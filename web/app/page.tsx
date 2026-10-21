import type { CSSProperties } from "react";
import Link from "next/link";
import { Note, Swipe, TIER_HL } from "@/components/doodle";
import { TIER_NAME } from "@/lib/format";
import { isLocalApi, SCAN_CLI } from "@/lib/scan";
import { TIER_ORDER } from "@/lib/types";
import { NewRunForm } from "./new-run-form";

const demoRepo = process.env.OFFRAMP_DEMO_REPO ?? "vj-io/offramp-demo-app";
const demoRepoUrl = demoRepo.startsWith("http") ? demoRepo : `https://github.com/${demoRepo}`;
// The run form posts to the API, which only a local build has; the hosted build offers the keyless scan instead.
const localApi = isLocalApi(process.env.NEXT_PUBLIC_API_URL);

export default function NewRunPage() {
  return (
    <div className="grid items-start gap-12 pt-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] lg:pt-10">
      <section className="flex flex-col gap-7">
        <h1 className="text-5xl leading-[0.98] font-extrabold tracking-[-0.035em] sm:text-7xl">
          Prove which LLM calls can <Swipe mark="var(--hl-lime)">move to Nemotron.</Swipe>
        </h1>
        <p className="max-w-xl text-lg leading-relaxed font-medium">
          Offramp finds every OpenAI and Anthropic call in a Python repo and proves which ones NVIDIA Nemotron on Nebius
          Token Factory can take over, by checking them against reference answers with known ground truth. It opens a
          pull request for the calls that reach the parity bar. The rest stay on their source model, with the numbers
          that show why.
        </p>
        <div className="flex flex-col gap-2">
          <p className="font-hand text-xl text-muted">Each call tries the cheapest tier first:</p>
          <ol className="flex flex-wrap items-center gap-2">
            {TIER_ORDER.map((t, i) => (
              <li key={t} className="flex items-center gap-2">
                {i > 0 && (
                  <svg aria-hidden="true" viewBox="0 0 40 20" className="h-4 w-8 overflow-visible">
                    <path d="M2 11 Q20 6 36 10 M29 4 L37 10 L29 16" className="ink-stroke" strokeWidth="2.5" />
                  </svg>
                )}
                <span className="hl-chip px-2.5 py-0.5 font-hand text-lg" style={{ "--mark": TIER_HL[t] } as CSSProperties}>
                  {TIER_NAME[t]}
                </span>
              </li>
            ))}
          </ol>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <Link href="/runs/demo?replay=demo" className="sketch bg-ink px-5 py-2.5 font-hand text-2xl text-paper hover:opacity-90">
            Watch demo run (replay)
          </Link>
          <span className="font-hand text-lg text-muted">A recorded run. No backend or keys needed.</span>
        </div>
      </section>

      {localApi ? (
        <Note note="var(--note-plain)" tilt={0.6} as="section" className="p-6 pt-7 sm:p-8">
          <h2 className="mb-5 text-3xl font-extrabold tracking-tight">Start a run</h2>
          <NewRunForm demoRepoUrl={demoRepoUrl} />
        </Note>
      ) : (
        <Note note="var(--note-plain)" tilt={0.6} as="section" className="flex min-w-0 flex-col gap-4 p-6 pt-7 sm:p-8">
          <h2 className="text-3xl font-extrabold tracking-tight">Try it on your repo</h2>
          <p className="text-lg leading-relaxed">
            Paste a public GitHub repo and see every OpenAI and Anthropic call Offramp would test, with its list-price
            gap to Nemotron. No key, no sign-in.
          </p>
          <Link href="/scan" className="sketch self-start bg-ink px-5 py-2.5 font-hand text-2xl text-paper hover:opacity-90">
            Scan a repo
          </Link>
          <p className="font-hand text-lg text-muted">Or on your machine, same scan:</p>
          <code className="block bg-paper px-3 py-2 font-mono text-xs wrap-anywhere">{SCAN_CLI}</code>
          <p className="font-hand text-lg leading-snug text-muted">
            The recorded run, locally with no key: <code className="font-mono text-xs text-ink">make demo</code>. A full
            run measures parity with your own Token Factory key:{" "}
            <code className="font-mono text-xs text-ink">uv run offramp run &lt;repo&gt;</code>
          </p>
        </Note>
      )}
    </div>
  );
}
