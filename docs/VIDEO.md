# Demo video script (under 3:00, narrated)

Target length 2:50, hard cap 3:00. Public on YouTube. The video must show the project working and say how Token Factory and Nemotron are used, so the narration says "Token Factory" and "Nemotron" out loud (the 0:20, 0:40 and 2:25 rows).

Rules for this script:

- **Only measured numbers.** Every number below was read from the recorded run `20261020-124700775-6cb31e`: its replay events (`web/public/replay/demo-events.json`), its PR body (`offramp/api/replay/demo-events.pr.md`, the body of <https://github.com/vj-io/offramp-demo-app/pull/1>), `evals/results/*.json` in `vj-io/offramp-demo-app` at `76a7940`, and the live runs' budget counters listed in the README's Results. The table at the end gives each value's source. Never type a number from memory; if a value changes, re-read it from those files.
- **Do not narrate a beat that did not happen in the recorded run.** This run has no `KEEP_CLOSED` verdict: every site passed parity on some tier, so the script says that and has no kept-closed beat. It does have three manual-review verdicts, so the 1:25 row explains those instead. The failure beat in the 0:40 row is the run's first failed dev result that an iteration fixed.
- **No third-party logos or trademarks on screen.** The dashboard uses text badges ("Nebius Token Factory", "NVIDIA Nemotron"); keep it that way, and show no vendor logos in screenshots or the end card. That includes GitHub: never show the GitHub logo or page chrome. Show the pull request's body as markdown text (the `pr_body_markdown` field of the report JSON from `GET /api/runs/{id}/report`, the same text as `offramp/api/replay/demo-events.pr.md`, in a plain text view, or `docs/media/05-pr-body.png`) or crop a screenshot to the PR body only. **No copyrighted music:** none at all, or music synthesised in code.
- **Honest wording about parity:** it is agreement with references, because no closed-model key was used for the demo.
- **Name the scope of every cost figure:** −11% is all six sites (the report headline), −82% is the three migrated sites only (the PR title).

## Timeline

Speech is about 150 words per minute. The five rows carry 387 words of narration (about 2:35 of speech), inside the 2:50 target; no row's speech is longer than its time slot, and the 1:25 row is the tightest (143 words, about 57 s of its 60 s).

| Time | Visual | Narration |
|---|---|---|
| 0:00 to 0:20 | Title card "Offramp", then the New Run screen with the repo field empty. Plain text only. | Swapping a closed LLM API for an open model sounds like a base URL change. The risk is silent regressions: broken JSON, mismatched tool schemas, wrong answers. Nobody can prove, call site by call site, that the open model is good enough. |
| 0:20 to 0:40 | Dashboard header with the "Nebius Token Factory" and "NVIDIA Nemotron" text badges. Click "Use demo repo" (the field fills), then hover "Watch demo run (replay)". | Offramp does the testing. Point it at a Python repo and it finds every closed-model call, then asks one question per call site: what is the cheapest NVIDIA Nemotron tier on Nebius Token Factory that passes? |
| 0:40 to 1:25 | Click "Watch demo run (replay)" and record `/runs/demo?replay=demo`. The stage stepper advances, six call-site cards appear, and the "Parity by tier" grid fills (Lightning, Super, Ultra). Zoom on the cs_01 card: Lightning shows "91.7% fail, dev", then after the iteration "100.0% pass, holdout, v1". Then the cs_04 row of the grid: Lightning and Super fail at "dev, v3", Ultra passes. The replay runs about 15 s; slow it in editing rather than cutting beats, and keep text readable. | This is a replay of a real run. The scanner found six call sites. Each one runs against Nemotron Lightning, then Super, then Ultra, all through one OpenAI-compatible endpoint on Token Factory. Deterministic scorers check JSON, labels, tool calls and answers against references, on a holdout split the iterator never sees. Here the ticket classifier fails on Lightning: one of twelve dev tickets gets the wrong label. Nemotron Ultra reads the failure and adds two examples to the prompt, and the rerun passes all twelve dev and all nine holdout cases, on Lightning. The RAG answerer needed Ultra: three rewrites each on Lightning and Super fell short. |
| 1:25 to 2:25 | Report page `/runs/demo/report?replay=demo`: the headline "3 of 6 LLM calls move to NVIDIA Nemotron on Nebius Token Factory." and "$49.24 (est.) to $43.86: −11%", the evidence table (three Nemotron rows with their tier, three "Manual review" rows), the sunset-risk badge on cs_02, the cost bars labelled "est.". Point at "Open the pull request" (do not follow it on camera). Then scroll the pull request's body as text, with no site chrome: title, evidence table, "What changed", "What Offramp did NOT change", "How parity was measured". | Every site reached the parity bar on some Nemotron tier. Three migrate: the agent step to Super, the RAG answerer to Ultra, the summariser to Lightning. Three go to manual review, the classifier from earlier among them, because Nemotron's edits to their tests could not be verified. Parity means agreement with references, because no closed-model key was used. Costs are estimates: tokens measured on Token Factory, times a published price. Tavily looked that price up at run time, cached per day, for four of the six sites; the other two came from a checked-in price table. Across all six sites, the estimate falls eleven percent; across the three migrated sites, eighty-two. Then the pull request: evidence, the files it changed, and what Offramp did not change. The three patches passed the repo's own lint and tests in a network-off sandbox, alone and merged. |
| 2:25 to 2:50 | Architecture diagram from the README (text only) with Token Factory highlighted, then the top of `LICENSE` as plain text ("MIT License"), then an end card: "Offramp", the repo URL as text, "make demo". No GitHub page or logo. | Every model call goes through Token Factory, three Nemotron tiers behind one endpoint. The eval matrix ran on a local worker, and the four live runs behind this demo cost ninety-four cents by our counter at list prices, excluding case generation. The repo is MIT licensed, and make demo replays this with no keys. Offramp: test it, then migrate. |

## Where each number comes from

All values are from run `20261020-124700775-6cb31e` unless the row says otherwise; the dashboard shows the same values.

| Number in the script | Value | Source |
|---|---|---|
| Call sites | six (plus one embedding call, not evaluated) | replay event 2 (`SCANNED`, "6 call sites found") and event 9 |
| The failure beat | cs_01 (`helphub/inbox/triage.py`, ticket classifier) on Lightning, dev, adapter v0: 11 of 12 cases, 91.7%; `cs_01:002` `wrong_label`, expected `technical_support`, got `account_access` | `evals/results/cs_01.json`, first `tier_results` entry; replay event 39 |
| The fix | Ultra's adapter v1 added two few-shot examples; Lightning then passed 12 of 12 dev and 9 of 9 holdout (1 iteration) | replay events 40 to 42 (`change_log`); `evals/results/cs_01.json` |
| cs_04 needed Ultra | Lightning dev v0 to v3 and Super dev v0 to v3 all failed; Ultra v1 passed, judge mean 5.0 on 9 holdout cases | replay events 65 to 82; `evals/results/cs_04.json` |
| Migrated | 3 of 6: cs_02 to Super, cs_04 to Ultra, cs_06 to Lightning | `done` event totals; PR evidence table |
| Manual review | 3 of 6: cs_01 (test edit's search text not found), cs_03 (test edit fed an Anthropic `TextBlock` into the OpenAI fake), cs_05 (patch would delete a test) | replay events 97 to 99; PR "What Offramp did NOT change" |
| Kept closed | none | `done` event: `kept_closed: 0` |
| Cost, all six sites | $49.24 → $43.86 per 1,000 calls (est.), −11% | `done` event: `est_cost_before_per_1k`, `est_cost_after_per_1k`; report headline |
| Cost, migrated sites | $6.54 → $1.15 per 1,000 calls (est.), −82% | sum of the three migrated verdicts' `cost_before_per_1k` / `cost_after_per_1k`; PR title |
| Sandbox result | the three patches passed the repo's ruff and pytest in Docker (network off, 2 CPUs), alone and merged | PR "What changed", last two bullets |
| Where the eval matrix ran | a local worker, 18 cells | replay event 18: "eval matrix (18 cells) ran locally (nebius job: not configured)" |
| Price sources | Tavily for cs_01, cs_02, cs_03 and cs_06 (fetched 2026-10-20 and read from the per-day cache that run 3 wrote); `offramp/evals/pricing.yaml` for cs_04 (`gpt-4o`) and cs_05 (`o3`) | the verdicts' `cost_before_note` (replay events 90 to 95); README, Tavily bullet |
| Spend | $0.94 for the four live runs by Offramp's counter at list prices, excluding case generation; the recorded run itself replayed from cache at $0.00 | `budget.json` of runs `20261019-220847641-43d3df`, `20261019-222942730-2c0f0b`, `20261020-115200254-46f068`, `20261020-122354830-57c50b` (README, Results) |
| Replay length | about 15 s | measured in the browser on <https://offramp-seven.vercel.app/runs/demo?replay=demo> |

## Before recording

- The replay finishes in about 15 s and `?replay=demo` never calls Token Factory (it needs no key); the PR text is the real pull request body from the recorded run.
- Record after the web fix is live on <https://offramp-seven.vercel.app>. Before it, the manual-review notes of the cs_03 and cs_05 cards run past the card's right edge, the report's evidence table is cut at the right edge at 1500 px, and that table shows an "after" cost and latency for the three manual-review rows, which were not migrated. The 1:25 beat depends on that table.
- Every number on screen and in the voice-over traces to a row of the table above. Re-read the final cut against those files.
- Final cut is under 3:00 (target 2:50), has captions, no logos, no music with a licence; uploaded as public on YouTube; the link goes in the Devpost form.
