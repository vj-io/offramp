import type { Metadata } from "next";
import Link from "next/link";
import { Note, Swipe } from "@/components/doodle";
import { SCAN_CLI } from "@/lib/scan";
import { ScanForm } from "./scan-form";

export const metadata: Metadata = {
  title: "Scan a repo · Offramp",
  description: "Lists the OpenAI and Anthropic call sites in a public Python repo. No key, no model calls.",
};

const demoRepo = process.env.OFFRAMP_DEMO_REPO ?? "vj-io/offramp-demo-app";
const demoRepoUrl = demoRepo.startsWith("http") ? demoRepo : `https://github.com/${demoRepo}`;

export default function ScanPage() {
  return (
    <div className="flex flex-col gap-8 pt-4 lg:pt-8">
      <section className="flex max-w-3xl flex-col gap-4">
        <h1 className="text-4xl leading-[1.02] font-extrabold tracking-[-0.03em] sm:text-6xl">
          Find the LLM calls <Swipe mark="var(--hl-lime)">in your repo.</Swipe>
        </h1>
        <p className="text-lg leading-relaxed font-medium">
          Paste a public GitHub repository. Offramp downloads its Python files, reads them without running them, and
          lists every OpenAI, Anthropic, LiteLLM and LangChain call it finds, with the call&apos;s shape and how its list
          price compares with Nemotron&apos;s. No key, no sign-in, no model calls.
        </p>
      </section>

      <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <ScanForm demoRepoUrl={demoRepoUrl} />

        <Note note="var(--note-yellow)" tilt={-0.8} as="section" className="flex min-w-0 flex-col gap-3 p-5 pt-6">
          <h2 className="font-hand text-2xl">A full run measures parity; this page only finds call sites.</h2>
          <p className="leading-relaxed">
            A full run generates test cases for each call, scores Nemotron against reference answers and opens a pull
            request for the calls that pass. It needs Docker and your own Token Factory key, so this site does not
            host it.
          </p>
          <Link href="/runs/demo?replay=demo" className="font-hand text-xl text-bluepen underline underline-offset-4">
            Watch the recorded demo run
          </Link>
          <p className="font-hand text-lg text-muted">Or scan a local checkout:</p>
          <code className="block overflow-x-auto bg-paper px-3 py-2 font-mono text-xs whitespace-nowrap">{SCAN_CLI}</code>
        </Note>
      </div>
    </div>
  );
}
