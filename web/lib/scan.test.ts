import { describe, expect, it } from "vitest";
import { fileUrl, isLocalApi, ratioText, runScan, SCAN_CLI, scanErrorText, type ScanRow } from "./scan";

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
  it("prints each tier's ratio, a range when input and output differ, else the reason there is none", () => {
    const sonnet = {
      lightning: { input: 50, output: 62.5 },
      super: { input: 10, output: 16.67 },
      ultra: { input: 3, output: 5 },
    };
    expect(ratioText(row({ price_ratio: sonnet }))).toBe("50–62.5× / 10–16.7× / 3–5×");
    const mini = {
      lightning: { input: 2.5, output: 2.5 },
      super: { input: 0.5, output: 0.67 },
      ultra: { input: 0.15, output: 0.2 },
    };
    expect(ratioText(row({ price_ratio: mini }))).toBe("2.5× / 0.5–0.67× / 0.15–0.2×");
    expect(ratioText(row({ price_note: "no list price on file" }))).toBe("no list price on file");
  });
});

describe("fileUrl", () => {
  it("encodes each path segment, so #, ? and spaces in names still link", () => {
    expect(fileUrl("https://github.com/a/b", "abc123", "src/my file#1.py", 7)).toBe(
      "https://github.com/a/b/blob/abc123/src/my%20file%231.py#L7",
    );
    expect(fileUrl("https://github.com/a/b", "feature/x", "a.py", 1)).toBe("https://github.com/a/b/blob/feature/x/a.py#L1");
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
    const complex = `This repo is too large or complex for the free hosted scan; run \`${SCAN_CLI}\` locally.`;
    expect(scanErrorText(413, { error: "too_large_or_complex", message: complex })).toBe(complex); // hint not repeated
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
