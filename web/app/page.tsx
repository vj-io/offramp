import type { CSSProperties } from "react";
import Link from "next/link";
import { Note, Swipe, TIER_HL } from "@/components/doodle";
import { TIER_NAME } from "@/lib/format";
import { TIER_ORDER } from "@/lib/types";
import { NewRunForm } from "./new-run-form";

const demoRepo = process.env.OFFRAMP_DEMO_REPO ?? "vj-io/offramp-demo-app";
const demoRepoUrl = demoRepo.startsWith("http") ? demoRepo : `https://github.com/${demoRepo}`;

export default function NewRunPage() {
  return (
    <div className="grid items-start gap-12 pt-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] lg:pt-10">
      <section className="flex flex-col gap-7">
        <h1 className="text-5xl leading-[0.98] font-extrabold tracking-[-0.035em] sm:text-7xl">
          Prove which LLM calls can <Swipe mark="var(--hl-lime)">move to Nemotron.</Swipe>
        </h1>
        <p className="max-w-xl text-lg leading-relaxed font-medium">
          Offramp finds every OpenAI and Anthropic call in a Python repo, replays each one on NVIDIA Nemotron through
          Nebius Token Factory, and opens a pull request for the calls that match the original model. Calls that
          don&apos;t match stay where they are, with the numbers that show why.
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

      <Note note="var(--note-plain)" tilt={0.6} as="section" className="p-6 pt-7 sm:p-8">
        <h2 className="mb-5 text-3xl font-extrabold tracking-tight">Start a run</h2>
        <NewRunForm demoRepoUrl={demoRepoUrl} />
      </Note>
    </div>
  );
}
