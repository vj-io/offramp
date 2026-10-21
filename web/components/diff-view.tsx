import { splitDiff, TIER_SHORT } from "@/lib/format";
import type { Tier } from "@/lib/types";

// A diff line's look: added on lime, removed on pink (mixed with white, ink on top, readable in both themes, as
// the heatmap does), hunk headers in blue pen, file headers in bold.
function lineClass(line: string, header: boolean): string {
  if (header) return "font-bold";
  if (line.startsWith("@@")) return "text-bluepen";
  if (line.startsWith("+")) return "bg-[color-mix(in_srgb,var(--hl-lime)_55%,#fff)] text-on-hl";
  if (line.startsWith("-")) return "bg-[color-mix(in_srgb,var(--hl-pink)_45%,#fff)] text-on-hl";
  return "";
}

/** One migrated site's shipped change, folded, its call site's file first; the summary counts files and lines. */
export function DiffView({
  siteId,
  siteFile,
  tier,
  diff,
  open,
}: {
  siteId: string;
  siteFile: string | undefined;
  tier: Tier | null;
  diff: string;
  open?: boolean;
}) {
  const files = splitDiff(diff).sort((a, b) => Number(b.path === siteFile) - Number(a.path === siteFile));
  const added = files.reduce((n, f) => n + f.added, 0);
  const removed = files.reduce((n, f) => n + f.removed, 0);
  return (
    <details open={open} className="sketch-box min-w-0 bg-note-plain">
      <summary className="cursor-pointer px-4 py-2 font-hand text-lg wrap-anywhere">
        <span className="font-mono text-sm font-bold">{siteId}</span>
        {tier && <> to {TIER_SHORT[tier]}</>}: {files.length} file{files.length === 1 ? "" : "s"},{" "}
        <span className="font-mono text-sm">
          +{added} −{removed}
        </span>{" "}
        <span className="font-mono text-xs text-muted">{files.map((f) => f.path).join(" · ")}</span>
      </summary>
      <pre
        tabIndex={0}
        aria-label={`Diff for ${siteId}`}
        className="max-h-[32rem] overflow-auto border-t-2 border-dashed border-muted font-mono text-xs leading-relaxed"
      >
        <code className="block w-max min-w-full py-2">
          {files.flatMap((f, fi) =>
            f.lines.map((line, i) => (
              <span key={`${fi}-${i}`} className={`block px-4 ${lineClass(line, Boolean(f.path) && i < 2)}`}>
                {line || " "}
              </span>
            )),
          )}
        </code>
      </pre>
    </details>
  );
}
