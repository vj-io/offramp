# Demo video script (under 3:00, narrated)

Target length 2:50, hard cap 3:00. Public on YouTube. The video must show the project working and say how Token Factory and Nemotron are used, so the narration says "Token Factory" and "Nemotron" out loud (the 0:20, 0:40 and 2:25 rows).

Rules for this script:

- **Only measured numbers.** Anything in square brackets, `[measured: ...]`, is read from the recorded run (`evals/results/*.json`, the exported replay events, the PR) when the video is made. Never type a number from memory. The table at the end says where each placeholder comes from.
- **Do not narrate a beat that did not happen in the recorded run.** If no site needed an iteration, replace the failure beat in the 0:40 row with the most informative measured failure instead (a cheaper tier's failing check). If no site was kept closed, say that every site passed; do not manufacture a failure.
- **No third-party logos or trademarks on screen.** The dashboard uses text badges ("Nebius Token Factory", "NVIDIA Nemotron"); keep it that way, and show no vendor logos in screenshots or the end card. That includes GitHub: never show the GitHub logo or page chrome. Show the pull request's body as markdown text (the `pr_body_markdown` field of the report JSON from `GET /api/runs/{id}/report`, in a plain text view) or crop a screenshot to the PR body only. **No copyrighted music:** none at all, or music synthesised in code.
- **Honest wording about parity:** it is agreement with references, because no closed-model key was used for the demo.

## Timeline

Speech is about 150 words per minute. The five rows carry roughly 310 words of narration (about 2:05 of speech) so every screen has room to be read.

| Time | Visual | Narration |
|---|---|---|
| 0:00 to 0:20 | Title card "Offramp", then the New Run screen with the repo field empty. Plain text only. | Swapping a closed LLM API for an open model sounds like a base URL change. The risk is silent regressions: broken JSON, mismatched tool schemas, wrong answers. Nobody can prove, call site by call site, that the open model is good enough. |
| 0:20 to 0:40 | Dashboard header with the "Nebius Token Factory" and "NVIDIA Nemotron" text badges. Click "Use demo repo" (the field fills), then hover "Watch demo run (replay)". | Offramp does the proving. Point it at a Python repo and it finds every closed-model call, then asks one question per call site: what is the cheapest NVIDIA Nemotron tier on Nebius Token Factory that passes? |
| 0:40 to 1:25 | Click "Watch demo run (replay)" and record `/runs/demo?replay=demo`. Stage stepper advances; site cards appear; the sites x tiers heatmap fills left to right (Lightning, Super, Ultra). Zoom on the card of `[measured: site]` turning red on `[measured: tier]` and the log line for the iteration. Cut dead time in editing; do not speed the text up past readable. | This is a replay of a real run. The scanner found `[measured: sites]` call sites. Each one runs against Nemotron Lightning, then Super, then Ultra, all through one OpenAI-compatible endpoint on Token Factory. Deterministic scorers check JSON, labels, tool calls and answers against references, on a holdout split the iterator never sees. Here `[measured: site]` fails on `[measured: tier]` with `[measured: failing check]`. Nemotron Ultra reads the failures and rewrites the adapter, and the rerun `[measured: outcome]`. |
| 1:25 to 2:25 | Report page `/runs/demo/report?replay=demo`: verdict table (MIGRATE rows with their tier, the `KEEP_CLOSED` row with its parity versus threshold), cost before and after labelled "est.", the sunset-risk badge if the run has one. Point at the "Open the pull request" link (do not follow it on camera). Then scroll the real pull request's body as text, with no site chrome: title, evidence table, "How parity was measured", diffs, "What Offramp did NOT change", and the sandbox test line. | Every site gets a verdict: migrate, with the tier that passed, or keep closed. `[measured: migrated count]` sites migrate. `[only if a site was kept closed:]` `[measured: kept site]` stays on its original model: parity against the reference was `[measured: parity]`, below the bar of `[measured: threshold]`. Offramp says no when it cannot prove yes. Parity means agreement with references, because no closed-model key was used, and the report says so. Costs are estimates: a published price, looked up with Tavily at run time, times tokens measured on Token Factory. Then the pull request: evidence table, how parity was measured, the diffs, and what Offramp did not change. The repo's own tests ran first, in a sandbox: `[measured: sandbox result]`. |
| 2:25 to 2:50 | Architecture diagram from the README (text only) with Token Factory highlighted, then the top of `LICENSE` as plain text ("MIT License"), then an end card: "Offramp", the repo URL as text, "make demo". No GitHub page or logo. | Every model call goes through Token Factory, three Nemotron tiers behind one endpoint. `[measured: where the eval matrix ran]`. The repo is MIT licensed, and make demo replays this with no keys. Offramp: prove it, then migrate. |

## Where each placeholder comes from

Read these from the artifacts committed after the live run (Task 14); the dashboard shows the same values.

| Placeholder | Source |
|---|---|
| `[measured: sites]` | Number of chat call sites in the run snapshot (`evals/results/`, `report.json`) |
| `[measured: site]`, `[measured: tier]`, `[measured: failing check]` | The first (site, tier) with a failed dev `TierResult` that a later iteration fixed: site id, tier, and the top `error_category` of its `failing_case_ids` |
| `[measured: outcome]` | That site's next `TierResult`: which tier passed holdout, after how many iterations |
| `[measured: migrated count]` | Verdicts with decision `MIGRATE`, said as "k of N" |
| `[only if a site was kept closed:]`, `[measured: kept site]`, `[measured: parity]`, `[measured: threshold]` | Say the whole sentence group only if the run has a `KEEP_CLOSED` verdict; then read the verdict's reason, "parity vs reference p% < t%". With no such verdict, cut it |
| `[measured: sandbox result]` | The sandbox line in the PR (the repo's own ruff and tests in the network-off container) |
| `[measured: where the eval matrix ran]` | The run artifact: say "the eval matrix ran as a Nebius AI Cloud serverless job" only if it did, otherwise "the eval matrix ran on a local worker", or leave the sentence out |

## Before recording

- Replay finishes in under 90 s and `?replay=demo` never calls Token Factory (it needs no key); the PR text is the real pull request body from the live run.
- Every number on screen and in the voice-over traces to a file above. Re-read the final cut against `evals/results/`.
- Final cut is under 3:00 (target 2:50), has captions, no logos, no music with a licence; uploaded as public on YouTube; the link goes in the Devpost form.
