import type { CSSProperties } from "react";
import { parityTint, scoreText, TIER_SHORT } from "@/lib/format";
import type { SiteView } from "@/lib/run-state";
import { TIER_ORDER, type RunConfig } from "@/lib/types";
import { circlePath, hatchPath, Ink, TIER_HL } from "./doodle";

export function TierHeatmap({ views, config }: { views: SiteView[]; config: RunConfig | null }) {
  return (
    <div>
      <div role="region" tabIndex={0} aria-label="Tier heatmap, scrolls sideways" className="relative overflow-x-auto pb-1">
        <table className="w-full min-w-[19rem] border-separate border-spacing-1.5 text-left">
          <caption className="sr-only">Parity of each call site on each Nemotron tier</caption>
          <thead>
            <tr>
              <th scope="col" className="font-hand text-base font-normal text-muted">
                Site
              </th>
              {TIER_ORDER.map((t) => (
                <th key={t} scope="col" className="font-normal">
                  <span className="hl-chip px-2 font-hand text-base" style={{ "--mark": TIER_HL[t] } as CSSProperties}>
                    {TIER_SHORT[t]}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {views.map(({ site, perTier, verdict }) => (
              <tr key={site.id}>
                <th scope="row" className="pr-1 font-mono text-sm font-bold">
                  {site.id}
                </th>
                {TIER_ORDER.map((t) => {
                  const r = perTier[t];
                  const chosen = verdict?.decision === "MIGRATE" && verdict.tier === t;
                  if (!r)
                    return (
                      <td key={t} className="sketch-box h-12 border-[1.5px] border-dashed border-muted text-center font-hand text-muted">
                        <span aria-hidden="true">–</span>
                        <span className="sr-only">not run</span>
                      </td>
                    );
                  return (
                    <td
                      key={t}
                      className="relative h-12 px-2 text-center text-on-hl"
                      style={{ background: parityTint(r, site.contract?.task_type, config) }}
                    >
                      {!r.passed && (
                        <Ink d={hatchPath(site.id + t)} width={1.5} clip className="inset-0 text-[var(--redpen-on-hl)] opacity-45" />
                      )}
                      {chosen && <Ink d={circlePath(site.id + t)} width={2.5} className="inset-0.5 text-on-hl" />}
                      <span className="relative block font-mono text-sm font-bold">
                        {scoreText(r, site.contract?.task_type)}
                      </span>
                      <span className="relative block font-hand text-sm leading-none">
                        {r.passed ? "pass" : "fail"}
                        {chosen ? ", chosen" : ""}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-2 font-hand text-base" aria-label="Heatmap legend">
        <li className="flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-4 w-6" style={{ background: "color-mix(in srgb, var(--hl-lime) 70%, #fff)" }} />
          passed parity
        </li>
        <li className="flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="inline-block h-4 w-10"
            style={{ background: "linear-gradient(90deg, color-mix(in srgb, var(--hl-pink) 30%, #fff), var(--hl-pink))" }}
          />
          below the bar, deeper is further below
        </li>
        <li className="flex items-center gap-1.5">
          <span aria-hidden="true" className="relative inline-block h-4 w-6">
            <Ink d={circlePath("legend")} width={2} className="inset-0" />
          </span>
          tier chosen
        </li>
        <li className="flex items-center gap-1.5">
          <span aria-hidden="true" className="relative inline-block h-4 w-6">
            <Ink d={hatchPath("legend")} width={1.5} clip className="inset-0 text-redpen" />
          </span>
          failed
        </li>
      </ul>
    </div>
  );
}
