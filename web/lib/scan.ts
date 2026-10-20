// The hosted scan (POST /api/scan, web/api/scan.py). Shapes mirror offramp/scan_table.py and offramp/github_scan.py.

export const SCAN_CLI = "uvx --from git+https://github.com/vj-io/offramp offramp scan .";

export type ScanRow = {
  id: string;
  file: string;
  line: number;
  sdk: string;
  model: string | null;
  shape: string;
  shape_guessed: boolean;
  streaming: boolean | null; // null: `stream=` is set at runtime
  flags: string[];
  // per Nemotron tier, cheapest first: source list price over the tier's, per input and per output token
  price_ratio: Record<string, { input: number; output: number }> | null;
  price_note: string | null;
};

export type TierPrice = {
  model: string;
  tier: string;
  input_usd_per_m: number;
  output_usd_per_m: number;
  price_status: string;
  as_of: string;
};

export type ScanResult = {
  repo: string;
  ref: string;
  commit: string | null;
  python_files: number;
  references: TierPrice[];
  ratio_label: string;
  rows: ScanRow[];
  rows_total: number; // rows is cut at 2,000
  warnings: string[];
  warnings_total: number;
};

/** The new-run form posts to the API, so it only shows when NEXT_PUBLIC_API_URL names a local one. */
export function isLocalApi(url: string | undefined): boolean {
  if (!url) return false;
  try {
    const host = new URL(url).hostname;
    return ["localhost", "127.0.0.1", "[::1]"].includes(host) || host.endsWith(".localhost");
  } catch {
    return false;
  }
}

const sig = (v: number) => String(Number(v.toPrecision(3)));

/** Per tier, cheapest first, the input–output ratio or one number when equal, as the CLI prints it:
 *  "50–62.5× / 10–16.7× / 3–5×". */
export function ratioText(row: ScanRow): string {
  if (!row.price_ratio) return row.price_note ?? "";
  return Object.values(row.price_ratio)
    .map(({ input, output }) => {
      const [lo, hi] = [Math.min(input, output), Math.max(input, output)];
      return lo === hi ? `${sig(lo)}×` : `${sig(lo)}–${sig(hi)}×`;
    })
    .join(" / ");
}

export const tierName = (tier: string) => tier.charAt(0).toUpperCase() + tier.slice(1);

/** A GitHub link to one line of a scanned file, each path segment encoded. */
export function fileUrl(repo: string, at: string, file: string, line: number): string {
  const path = file.split("/").map(encodeURIComponent).join("/");
  return `${repo}/blob/${encodeURIComponent(at).replaceAll("%2F", "/")}/${path}#L${line}`;
}

/** The message to show for a failed POST /api/scan. */
export function scanErrorText(status: number, body: unknown): string {
  const message = (body as { message?: unknown } | null)?.message;
  if (typeof message === "string" && message) {
    return status === 413 ? `${message} Run the scan on a local checkout instead: ${SCAN_CLI}` : message;
  }
  if (status === 404) return `The scan API is not running here (it is a Vercel function). Run the CLI instead: ${SCAN_CLI}`;
  return `The scan failed (HTTP ${status}). Try again, or run the CLI: ${SCAN_CLI}`;
}

export async function runScan(
  githubUrl: string,
  fetchImpl: typeof fetch = fetch,
): Promise<{ ok: true; result: ScanResult } | { ok: false; message: string }> {
  let res: Response;
  try {
    res = await fetchImpl("/api/scan", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ github_url: githubUrl.trim() }),
    });
  } catch {
    return { ok: false, message: "Could not reach the scan API. Check your connection and try again." };
  }
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok || !body || !Array.isArray((body as ScanResult).rows)) {
    return { ok: false, message: scanErrorText(res.status, body) };
  }
  return { ok: true, result: body as ScanResult };
}
