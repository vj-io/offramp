# Demo video script (under 3:00, narrated)

Target length 2:45, hard cap 3:00. Public on YouTube. The video must show the project working and say how Token Factory and Nemotron are used, so the narration says "Token Factory" and "Nemotron" out loud. The result and its number come first, in the opening 20 seconds.

Rules for this script:

- **Only measured numbers.** Every number spoken or shown traces to a row of [Where each number comes from](#where-each-number-comes-from). Never type a number from memory; if a value changes, re-read it from the source file.
- **Say what happened, including what went wrong.** The demo's sixth call went to manual review and the voice says why. The "no" beats are real runs, not staged.
- **Name the scope of every cost figure.** −95% is the five migrated demo sites (and, rounded, all six); the dollar figures are per 1,000 calls to each site, estimated.
- **Honest wording about parity:** it is agreement with reference answers on held-out cases, because no closed-model key was used.
- **No third-party logos or trademarks on screen.** The dashboard uses text badges ("Nebius Token Factory", "NVIDIA Nemotron"); keep it that way. Never show the GitHub logo or page header: show a pull request as its body text, cropped below the site header (as `docs/media/04-pull-request.png` is), and show CI results as an on-screen caption. **No copyrighted music:** none at all, or music synthesised in code.
- **Captions come from the narration.** Build the voice line and its caption from the same text, so they cannot drift apart.
- **Cut:** no architecture diagram walk-through and no licence screen. The end card carries the URLs as text.

## Timeline

Speech runs at about 150 words per minute. The seven rows carry 360 words (about 2:24 of speech) inside 2:45, and each row's speech fits its slot at that rate. The offline voice reads numbers more slowly than words, so the number-heavy rows (1:12 and 2:04) keep 3 to 5 seconds of slack. Measure each line's real length with the voice before fitting the cut, and trim words, never numbers' scope.

| Time | Visual | Narration |
|---|---|---|
| 0:00 to 0:17 | Cold open on the report of the final demo run (`/runs/demo/report?replay=demo`): the headline "6 of 6 call sites reached parity on Nemotron (held-out cases) · 5 migrated · est. −95% on those 5". Cut to the body of PR #3 as text (its title line and the evidence table), with the caption "Demo repo CI: 2 of 2 checks passed". | This pull request moves five of six LLM calls in a Python app onto NVIDIA Nemotron on Nebius Token Factory. Estimated cost on those five: ninety-five percent lower. The app's own tests pass. Offramp wrote it, with evidence for every call. |
| 0:17 to 0:36 | The home page: the headline "Test which LLM calls can move to Nemotron." and the Lightning → Super → Ultra chips. | Moving off a closed model looks like a base-URL change. The risk is silent: JSON that stops parsing, tool calls that drift, worse answers. Offramp checks each call on held-out cases and picks the cheapest Nemotron tier that passes: Lightning, then Super, then Ultra. |
| 0:36 to 1:12 | The replay (`/runs/demo?replay=demo`), one real frame per recorded event paced to the voice and labelled as such (the 1× replay runs 57 s): the stepper advances, six call-site cards appear, and the "Parity by tier" grid fills. Hold on the cs_04 card and the log line "cs_04: adapter v1 for Super" (the moment in `docs/media/01-replay-mid-run.png`). End on the finished grid with the chosen tiers circled. | This is a replay of the real run. Offramp found six call sites and built one hundred twenty-one test cases with reference answers, forty of them held out. Code scores JSON, labels and tool calls; Nemotron Ultra judges free text. Here the RAG answerer fails on Lightning. Three prompt rewrites by Ultra don't save it, and Super falls short too. On Ultra it passes, on all nine held-out cases. The grid fills in, with the cheapest passing tier circled for each call. |
| 1:12 to 1:46 | The report's evidence table: the five Nemotron rows, the cs_01 "Manual review" row, the "Sunset risk: Claude Sonnet 4.5 retires 30 Nov 2026" line, and the cost column labelled "est.". Then the cs_01 card's manual-review note. Then PR #3's body text again, with the caption "Demo repo CI: 2 of 2 checks passed". | Five calls migrate: two to Super, one to Ultra, two to Lightning. Across all six calls, estimated cost per thousand calls to each: forty-nine dollars before, two sixty-five after. Two of them ran on Claude Sonnet 4.5, which Tavily found retires on the thirtieth of November. One call goes to a person: its test edit would have dropped three asserts, so Offramp refused it. The repo's own CI passes on the pull request. |
| 1:46 to 2:04 | The no-Ultra replay (`/runs/demo?replay=no-ultra`): the violet "Team policy: a team that caps spend below Ultra" chip, the grid with an empty Ultra column, and the cs_04 card's "NOT MIGRATED" stamp. | Offramp also says no. Under a team policy of no Ultra, the RAG answerer stays on GPT-4o, because Lightning and Super both fall short on the dev cases. That is the gate doing its job. |
| 2:04 to 2:32 | The open-source replay (`/runs/demo?replay=oss`): the H1 "Offramp on vj-io/idea-reality-mcp" with the caption "A fork of mnemox-ai/idea-reality-mcp (MIT)", the "Latency gate" chip about the repo's own 30 s client timeout, the grid, then the report's "What Offramp did not change" cards (`docs/media/05-gate-says-no.png`). | On a real open-source repo we did not write, two of five calls move to Super. Two stay on Claude, because Super scored below the bar on held-out cases. Lightning was out: that day its median answer took over a minute, and the app's own client gives up after thirty seconds. The first runs here also found five bugs in Offramp, all now fixed. |
| 2:32 to 2:45 | End card, plain text: "Offramp", "offramp-seven.vercel.app", "github.com/vj-io/offramp", "Every Offramp model call: Nebius Token Factory · NVIDIA Nemotron". | Every Offramp model call ran on Token Factory, for four dollars thirteen of our twenty-five dollar grant. Scan your own repo for free at offramp-seven dot vercel dot app. |

## Where each number comes from

Run ids: the final demo run is `20261022-014146107-57cca9`, the no-Ultra run `20261022-015732734-b5aee2`, and the open-source run `20261022-153359255-7bea7c`. Their replays are `web/public/replay/{demo,no-ultra,oss}-events.json`; their run directories (`var/runs/<id>/`) are not committed.

| Number in the script | Value | Source |
|---|---|---|
| Call sites and migrations (demo) | 6 sites; 5 migrated (cs_02, cs_03 to Super; cs_04 to Ultra; cs_05, cs_06 to Lightning); 1 manual review (cs_01) | demo replay `done` event totals; PR #3 summary |
| "Six of six reached parity" (on screen) | 6 of 6, held-out cases | report headline, built from the verdicts in the demo replay |
| −95% on the 5 migrated sites | $49.21 → $2.61 per 1,000 calls to each site, est. | sum of the five MIGRATE verdicts' `cost_before_per_1k` / `cost_after_per_1k`; PR #3 title "est. −95% cost on those 5" |
| $49 before, $2.65 after | $49.24 → $2.65 per 1,000 calls to each site, est., all 6 sites (cs_01 counted at its source price) | demo replay `done` event: `est_cost_before_per_1k` 49.243, `est_cost_after_per_1k` 2.646; report subline |
| Test cases | 121 cases, 40 held out | PR #3 summary; demo `done` event `cases` |
| cs_04 needed Ultra | Lightning dev v0 to v3 failed (3 distinct rewrites); Super dev v0 and v1 failed, and its v2 and v3 repeated v1; Ultra v1 passed, judge mean 5.0 on 9 holdout cases | demo replay events 65 to 82; PR #3 evidence table |
| Two calls on Claude Sonnet 4.5, retiring 30 Nov 2026 | cs_02, cs_03; "retires on November 30, 2026" | PR #3 "Sunset risk" section (Tavily, platform.claude.com model page, 2026-10-22) |
| cs_01's refused edit | the edit "would drop" 3 asserts it never quoted | cs_01 verdict reason in the demo replay; PR #3 "What Offramp did NOT change" |
| CI on PR #3 | 2 of 2 checks passed | `gh pr view 3 -R vj-io/offramp-demo-app --json statusCheckRollup` (2026-10-22) |
| No-Ultra: cs_04 kept on gpt-4o | Lightning and Super failed dev; Super's best dev judge mean 4.45, more than 0.3 below the references' mean | no-Ultra replay, cs_04 verdict reason |
| Real repo: 2 of 5 migrate, 2 stay | cs_02, cs_04 to Super; cs_01 (Super holdout mean 4.28, more than 0.3 below the references), cs_05 (Super holdout mean 3.55 < 4.0) kept | open-source replay verdicts; PR <https://github.com/vj-io/idea-reality-mcp/pull/1> |
| Lightning "over a minute", app timeout 30 s | Lightning dev medians 64.3, 71.0, 76.7 and 97.5 s; the call's `AsyncAnthropic(timeout=30.0)` | open-source replay log lines "Lightning not iterated: dev median … s is more than twice the call's 30 s timeout" |
| Five bugs found on the real repo | duplicate cases across splits, Ultra budget spent on failing tiers, a degenerate adapter, latency against the call's timeout, a judge reply cut at `max_tokens` | README, "What the real repo taught us" |
| $4.13 of the $25 grant | $3.51 over 15 runs (`budget.json`) plus $0.62 recorded outside runs | README, "What it cost" |

## Before recording

- Record the dashboard from a local build (`pnpm --dir web dev`) or from <https://offramp-seven.vercel.app> once it is redeployed with the three recordings. Before a hosted take, check that `/replay/demo-events.json` holds run `20261022-014146107-57cca9` and that `?replay=no-ultra` and `?replay=oss` load.
- Play the replays at 1× for the shots that follow the run, and use "Skip to end" for the finished grid. Record at 1440 px wide or more, so the evidence table fits with its cost column visible.
- Every number on screen and in the voice traces to a row of the table above. Re-read the final cut against those files.
- Final cut: under 3:00 (target 2:45), captions on, no logos, no licensed music. Upload as public on YouTube and put the link in the Devpost form, the README and `docs/SUBMISSION.md`.
