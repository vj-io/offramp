import { CodeText, LinkedText } from "@/components/badges";
import { DECISION_NOTE, Note } from "@/components/doodle";
import { dayText, judgeText, parityText, PRICE_METHOD, reviewNote, TIER_NAME } from "@/lib/format";
import type { CallSite, PriceEvidence, Verdict } from "@/lib/types";

export function ReviewNote({ v, site, tilt }: { v: Verdict; site: CallSite | undefined; tilt: number }) {
  const { sentence, raw } = reviewNote(v);
  return (
    <Note as="li" note={DECISION_NOTE.MANUAL_REVIEW} tilt={tilt} className="flex min-w-0 flex-col gap-2 p-5 pt-6">
      <p className="font-mono text-sm wrap-anywhere">
        <span className="font-bold">{v.site_id}</span> {site?.file}
      </p>
      {v.tier && v.parity_pct != null && (
        <p className="font-hand text-lg leading-snug">
          Reached parity on {TIER_NAME[v.tier]}:{" "}
          <span className="font-mono text-sm">{judgeText(v) ?? parityText(v, site?.contract?.task_type)}</span>.
        </p>
      )}
      {raw === null ? (
        <p className="font-hand text-lg leading-snug wrap-anywhere">{sentence}</p>
      ) : (
        <>
          <p className="font-hand text-lg leading-snug wrap-anywhere">
            <CodeText text={sentence} />
          </p>
          <details className="font-hand text-base">
            <summary className="cursor-pointer underline underline-offset-2">Raw failure</summary>
            <p className="mt-1 font-mono text-xs leading-relaxed whitespace-pre-wrap wrap-anywhere">
              {raw.replaceAll(" | ", "\n")}
            </p>
          </details>
        </>
      )}
    </Note>
  );
}

// Where each source model's "before" price came from (Task 19): the method, the page, the words on it, the day,
// and any disagreement. Sites sharing a model and its evidence share a card.
export function PriceEvidencePanel({ verdicts, sites }: { verdicts: Verdict[]; sites: Record<string, CallSite> }) {
  const groups = new Map<string, { ev: PriceEvidence; model: string | null; ids: string[] }>();
  for (const v of verdicts) {
    if (!v.price_evidence) continue;
    const model = sites[v.site_id]?.source_model ?? null;
    const key = JSON.stringify([model, v.price_evidence]);
    if (!groups.has(key)) groups.set(key, { ev: v.price_evidence, model, ids: [] });
    groups.get(key)!.ids.push(v.site_id);
  }
  if (groups.size === 0) return null;
  return (
    <section aria-labelledby="tavily-h" className="flex flex-col gap-4">
      <h2 id="tavily-h" className="text-3xl font-extrabold tracking-tight">
        Price checked with Tavily
      </h2>
      <p className="max-w-3xl font-hand text-lg text-muted">
        The source model&apos;s price behind each &ldquo;before&rdquo; cost: how Offramp found it, the page it cites
        and the words on that page.
      </p>
      <ul className="grid gap-4 md:grid-cols-2">
        {[...groups.values()].map(({ ev, model, ids }) => (
          <li key={ids.join()} className="sketch-box flex min-w-0 flex-col gap-2 bg-note-plain p-4">
            <p className="font-mono text-sm wrap-anywhere">
              <span className="font-bold">{ids.join(", ")}</span> {model}
            </p>
            <p className="font-hand text-lg">
              {PRICE_METHOD[ev.method] ?? ev.method}, {dayText(ev.date)}
            </p>
            {ev.query && (
              <p className="font-hand text-base">
                Query: <span className="font-mono text-xs wrap-anywhere">{ev.query}</span>
              </p>
            )}
            <p className="font-mono text-xs wrap-anywhere">
              <LinkedText text={ev.url} />
            </p>
            {ev.quote && (
              <blockquote className="border-l-4 border-[var(--hl-lime)] pl-3 font-mono text-xs leading-relaxed wrap-anywhere">
                {ev.quote}
              </blockquote>
            )}
            {ev.note && (
              <p className="font-hand text-base leading-snug wrap-anywhere">
                <span className="font-bold">Note:</span> <LinkedText text={ev.note} />
              </p>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
