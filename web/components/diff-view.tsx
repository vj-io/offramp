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

// Lines shown per file; past this the pull request has the rest (a whole-file test rewrite can run to thousands).
const MAX_LINES = 400;

/** One migrated site's shipped change, folded, its call site's file first; the summary counts files and lines. */
export function DiffView({
  siteId,
  siteFile,
  tier,
  diff,
  prUrl,
  open,
}: {
  siteId: string;
  siteFile: string | undefined;
  tier: Tier | null;
  diff: string;
  prUrl: string | null;
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
          {files.map((f, fi) => (
            <span key={fi} className="block">
              {f.lines.slice(0, MAX_LINES).map((line, i) => (
                <span key={i} className={`block px-4 ${lineClass(line, Boolean(f.path) && i < 2)}`}>
                  {line || " "}
                </span>
              ))}
              {f.lines.length > MAX_LINES && (
                <span className="block px-4 py-1 font-hand text-base">
                  … {f.lines.length - MAX_LINES} more lines of {f.path || "this file"}{" "}
                  {prUrl ? (
                    <a href={`${prUrl}/files`} target="_blank" rel="noreferrer" className="text-bluepen underline underline-offset-2">
                      in the pull request
                    </a>
                  ) : (
                    "in the pull request"
                  )}
                </span>
              )}
            </span>
          ))}
        </code>
      </pre>
    </details>
  );
}
