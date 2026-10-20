import { describe, expect, it } from "vitest";
import { isLocalApi, ratioText, runScan, SCAN_CLI, scanErrorText, type ScanRow } from "./scan";

const row = (over: Partial<ScanRow>): ScanRow => ({
  id: "cs_01",
  file: "app.py",
  line: 3,
  sdk: "openai",
  model: "gpt-4o-mini",
  shape: "free text",
  shape_guessed: true,
  streaming: false,
  flags: [],
  price_ratio: null,
  price_note: null,
  ...over,
});

const reply = (status: number, body: unknown) =>
  (async () => new Response(typeof body === "string" ? body : JSON.stringify(body), { status })) as typeof fetch;

describe("isLocalApi", () => {
  it("shows the run form only for a local API", () => {
    expect(isLocalApi("http://localhost:8000")).toBe(true);
    expect(isLocalApi("http://127.0.0.1:8000/")).toBe(true);
    expect(isLocalApi("http://[::1]:8000")).toBe(true);
    expect(isLocalApi("http://api.localhost:8000")).toBe(true);
    expect(isLocalApi(undefined)).toBe(false); // the hosted build sets no API URL
    expect(isLocalApi("")).toBe(false);
    expect(isLocalApi("https://offramp.example.com")).toBe(false);
    expect(isLocalApi("http://localhost.example.com")).toBe(false);
    expect(isLocalApi("not a url")).toBe(false);
  });
});

describe("ratioText", () => {
  it("prints one ratio when input and output match, both otherwise, else the reason there is none", () => {
    expect(ratioText(row({ price_ratio: { input: 2.5, output: 2.5 } }))).toBe("2.5x");
    expect(ratioText(row({ price_ratio: { input: 50, output: 62.5 } }))).toBe("50x in / 62.5x out");
    expect(ratioText(row({ price_note: "no list price on file" }))).toBe("no list price on file");
  });
});

describe("scan errors", () => {
  it("shows the API's own message, and points too-large repos at the CLI", () => {
    expect(scanErrorText(400, { error: "bad_url", message: "Give a public GitHub repository URL." })).toBe(
      "Give a public GitHub repository URL.",
    );
    const big = scanErrorText(413, { error: "too_large", message: "The repository archive is over 25 MB." });
    expect(big).toContain("over 25 MB");
    expect(big).toContain(SCAN_CLI);
  });

  it("explains a missing API (local dev without Vercel) and other failures", () => {
    expect(scanErrorText(404, null)).toContain("not running here");
    expect(scanErrorText(500, null)).toContain("HTTP 500");
  });

  it("runScan returns the result, or the message for an error or a non-JSON reply", async () => {
    const result = { repo: "https://github.com/a/b", rows: [], warnings: [] };
    expect(await runScan("https://github.com/a/b", reply(200, result))).toEqual({ ok: true, result });
    expect(await runScan("x", reply(429, { error: "rate_limited", message: "At most 6 scans a minute." }))).toEqual({
      ok: false,
      message: "At most 6 scans a minute.",
    });
    expect(await runScan("x", reply(502, "<html>Bad gateway</html>"))).toEqual({
      ok: false,
      message: scanErrorText(502, null),
    });
    const offline = (async () => {
      throw new TypeError("fetch failed");
    }) as typeof fetch;
    expect((await runScan("x", offline)).ok).toBe(false);
  });
});
