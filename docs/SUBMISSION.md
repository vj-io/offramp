# Devpost submission text

Ready to paste into the Devpost form, one section per field. Every number comes from the recorded run `20261020-124700775-6cb31e` (the replay file `web/public/replay/demo-events.json`, the PR body `offramp/api/replay/demo-events.pr.md`, and `evals/results/*.json` in `vj-io/offramp-demo-app` at `e5baa78`) or from the budget counters of the live runs listed in the README's [Results](../README.md#results).

## Project name

Offramp

## Tagline

Tests each closed-LLM call in a Python repo against NVIDIA Nemotron on Nebius Token Factory, then opens a pull request for the calls that pass.

## Track

Coding and Agentic Engineering

## Inspiration

Moving a product off a closed LLM API sounds like a base-URL change, and teams have reasons to do it: price, vendor lock-in, and deprecations. In our demo app, Tavily found that one of the source models, `claude-sonnet-4-5`, retires on 30 November 2026. What stops teams is risk they cannot see: JSON that no longer parses, tool schemas that drift, answers that are slightly worse. Nobody wants to migrate on a vibe check, and nobody has time to check parity call site by call site. We wanted a tool that does the checking and hands over a pull request a reviewer can trust.

## What it does

Point Offramp at a Python repo. It:

1. **Scans** for every OpenAI, Anthropic, LiteLLM and LangChain call with a deterministic AST pass, and Nemotron 3 Super labels each call's contract: classification, JSON extraction, tool call, free text or reasoning.
2. **Builds an eval set per call site**, split into dev and holdout by a hash of the case id. Expected outputs are references with known ground truth: seed references (the expected label, record or tool call is chosen first, and Super writes an input that implies it), oracle references (computed by code), or references written by a different model family. Seed references are audited blind.
3. **Runs a tier matrix** on Nebius Token Factory: Nemotron 3.5 Lightning, then Nemotron 3 Super, then Nemotron 3 Ultra. Deterministic scorers check schemas, labels, tool names and arguments, and numeric answers; an Ultra judge scores free text.
4. **Iterates.** When a cheaper tier fails on dev, Nemotron 3 Ultra reads the failing cases and rewrites the adapter (prompt, few-shot examples, schema, reasoning switch), up to 3 times per tier. The Iterator never sees holdout cases.
5. **Decides** per call site: migrate to the cheapest tier that passes on holdout, keep the source model when no tier does, or hand the site to manual review.
6. **Patches and opens a PR.** Code rewrites each migrated call, and Nemotron edits only the tests and fakes that pinned the old provider. The repo's own ruff and pytest run in a network-off Docker sandbox, and any site whose patch cannot be verified goes to manual review with the exact reason. The PR carries an evidence table, how parity was measured, failing cases, prices with their sources, and a "What Offramp did NOT change" section.

A dashboard shows all of it live: stage stepper, call-site cards, a parity-by-tier grid, the log and a report page.

**On our demo repo** (a small support-desk app with six chat call sites on `gpt-4o-mini`, `gpt-4o`, `o3`, `claude-sonnet-4-5` and `claude-haiku-4-5`), Offramp evaluated 121 cases, 40 of them held out. Every site reached the parity bar on a Nemotron tier. Three were migrated in [the pull request](https://github.com/vj-io/offramp-demo-app/pull/1): the tool-calling agent step to Nemotron 3 Super, the RAG answerer to Nemotron 3 Ultra, and the thread summariser to Nemotron 3.5 Lightning. The other three went to manual review because Nemotron's test edits for them could not be verified: one edit's search text was not in the file, one fed an Anthropic block into the OpenAI test fake, and one patch would have deleted a test, which Offramp refuses. Estimated cost per 1,000 calls for the three migrated sites drops from $6.54 to $1.15 (−82%); across all six sites, with the manual-review sites still on their source models, it drops from $49.24 to $43.86 (−11%).

## How we built it

- **Backend:** Python 3.12 with FastAPI, Pydantic and SQLModel; a run is a state machine that streams its events over server-sent events.
- **Models:** every inference call goes through one client, the OpenAI Python SDK pointed at Nebius Token Factory. Nemotron 3 Super and Nemotron 3 Ultra are the agents (labelling, case generation, adapter, iterator, judge, patch writer, PR prose), and Nemotron 3.5 Lightning, Super and Ultra are the models under test. Structured calls use `temperature=0`, `json_schema` output, an explicit `enable_thinking` switch and a per-agent token cap.
- **Cache as cassettes:** every response is stored by a hash of model, messages and parameters. Cache hits are free, a run can be replayed exactly, and the test suite replays real Token Factory output without a key.
- **Budget guardrails:** caps per run on input and output tokens and on Ultra agent calls; hitting one keeps the affected sites on their source model instead of crashing.
- **Patching:** an AST rewrite writes the evaluated request into each call site as literal keyword arguments, and a check rejects any model edit to a non-test file. The target repo's own checks run in Docker with the network off.
- **Partners:** Tavily looks up each source model's published price and deprecation notices at run time, and a price is accepted only if the cited page prints it next to that exact model. LangSmith traces every Token Factory call and keeps each site's cases as a dataset, with public links in the PR. Claude Sonnet 5.5, run through Claude Code during the build, wrote the free-text references and blind-audited the seed references, so Nemotron never grades its own gold.
- **Dashboard:** Next.js and Tailwind CSS on Vercel's free tier. The demo replay runs entirely in the browser from a recorded event log.

## Challenges we ran into

- **No closed-model keys.** We had only hackathon credits, so the source models' own outputs could not be recorded. We redefined parity as agreement with references whose ground truth is known, labelled every reference with where it came from, and audited the seed references blind with a different model family.
- **The Ultra judge skipped a case.** In our first live demo run, an Ultra judge batch returned no score for one Japanese question, and a retry replayed the same cached reply, so the run failed. The judge now asks again about only the skipped cases.
- **Super was unreliable at writing shipped code to a spec.** Asked to rewrite call sites from the evaluated adapter, Super's patches were rejected on 5 of 6 sites: an unused few-shot constant, the old call left in place, a wrong model id, overlapping edits, a failing test. We moved the call-site rewrite into deterministic code and let the model edit tests only. Ultra repairs then broke the edit-block format three times (`FILES:` instead of `FILE:`, a missing `>>>>>>>`), and those sites went to manual review.
- **Nemotron specifics:** reasoning is on by default and its tokens count toward `max_tokens`; Lightning with `tool_choice="required"` looped until the token cap. Our `docs/FEEDBACK.md` has the details and evidence.
- **AI Cloud billing.** Running the eval matrix as a Nebius AI Cloud Serverless Job needed a payment card and a $25 top-up, separate from the Token Factory credits. We built the job path and tested it against a fake CLI, but it never ran on Nebius; the matrix runs on a local worker.
- **Small holdouts.** With 3 to 9 holdout cases per site, one miss moves parity by 11 to 33 points, so the report shows n for every site.

## Accomplishments that we're proud of

- An end-to-end run on our demo repo that ends in a real pull request: 6 call sites, 121 cases, 3 migrations, each with evidence, and 3 honest "a human should look at this" verdicts instead of patches we could not verify.
- Offramp writes every change to application code itself; Nemotron edits only tests and test fakes, and the PR lists the test files each model edit touched, for the reviewer. The three shipped patches passed the target repo's own ruff and pytest, alone and merged.
- Every number on screen and in the PR is rendered from run artifacts. Costs are labelled as estimates with their price source and date, and latency before migration is shown as not measured.
- The four live pipeline runs behind the demo cost $0.94 together on Token Factory, by our spend counter at list prices (unverified, because the price page is login-gated) and excluding case generation. One full live pipeline cost $0.83; each re-run after a fix cost $0.03 to $0.08, because the response cache served every unchanged call.
- Judges can watch the whole recorded run at <https://offramp-seven.vercel.app> with no login and no keys, or run `make demo` locally.

## What we learned

- Deterministic scorers should decide wherever they can. `json_schema` and tool calling worked on all three Nemotron tiers, so only the two free-text sites needed a judge.
- The cheapest tier is often enough, but not always: three sites passed on Lightning, two needed Super, and the RAG answerer needed Ultra after Lightning and Super each failed three adapter rewrites.
- Let models propose and code decide. Every allowance for model edits to shipped code reopened a way for unverified changes to slip through; restricting the model to tests closed it.
- Honest labels matter more than flattering numbers: "est.", "parity vs reference", "manual review" and the scope of every cost figure.

## What's next for Offramp

- Run the eval matrix as a Nebius AI Cloud Serverless Job once billing allows, and host the API.
- Human audit of references through Tendem (the audit script exists; the demo used none).
- Closed-model baselines for teams that have their source-model keys (the recorder exists and is tested with fake clients).
- A better repair loop for test edits, so fewer sites end in manual review.
- More call shapes: streaming calls (flagged for manual review today) and more SDK wrappers.

## Built with

Python, FastAPI, Pydantic, SQLModel, OpenAI Python SDK, Nebius Token Factory, NVIDIA Nemotron 3.5 Lightning, NVIDIA Nemotron 3 Super, NVIDIA Nemotron 3 Ultra, Next.js, React, TypeScript, Tailwind CSS, Docker, Tavily, LangSmith, Vercel, GitHub, uv, pnpm, Claude Code

## Testing instructions

- **Hosted demo, no login and no keys:** open <https://offramp-seven.vercel.app> and click "Watch demo run (replay)", or go straight to <https://offramp-seven.vercel.app/runs/demo?replay=demo>. The replay of the recorded run plays in about 15 seconds; then click "Open the report", or open <https://offramp-seven.vercel.app/runs/demo/report?replay=demo>.
- **The pull request the run opened:** <https://github.com/vj-io/offramp-demo-app/pull/1>.
- **Your own repo, no key:** paste a public GitHub URL at <https://offramp-seven.vercel.app/scan>, or run `uvx --from git+https://github.com/vj-io/offramp offramp scan .` in a checkout (needs uv). Both list the call sites, their call shapes and the list-price ratio to Nemotron Lightning, without importing or running the code. Measuring parity takes a full run with your own Token Factory key.
- **Locally,** with Python 3.12+ and uv, Node 20.9+ and pnpm, make and curl:

  ```bash
  git clone https://github.com/vj-io/offramp.git && cd offramp
  make demo
  ```

  `make demo` installs dependencies, starts the API on `:8000` and the dashboard on `:3000`, and opens the replay. No key is needed. `make test` runs the Python suite on recorded Token Factory responses, also without a key.

## Gallery images

1500x1000 PNGs in `docs/media/`, taken with Playwright from the deployed dashboard on 2026-10-20.

`05` is the real PR body rendered as markdown on a plain page, with no site chrome; the line above the title and the caption under it are ours, not part of the PR.

| File | Caption |
|---|---|
| `01-replay-mid-run.png` | Mid-run: the RAG answerer has failed on Lightning after three adapter rewrites, and Nemotron 3 Ultra is rewriting its adapter for Super. |
| `02-replay-done.png` | Done: the parity-by-tier grid with the chosen tier circled for each migrated call site. |
| `03-report-summary.png` | The report: 3 of 6 calls move to Nemotron; estimated cost across all six sites $49.24 → $43.86 per 1,000 calls (−11%). |
| `04-report-evidence.png` | The evidence table: before and after model, parity vs reference on holdout, p50 latency on Nemotron and estimated cost per 1,000 calls. The three manual-review rows show "—" for their after values, because those sites were not migrated. |
| `05-pr-body.png` | The pull request Offramp opened, shown as its markdown body; its −82% covers the three migrated sites only. |
| `06-call-site-card.png` | One call site up close: the RAG answerer fails on Lightning and Super after three adapter rewrites each, then passes on Ultra. The log beside it shows the recorded PR title; its est. −82% covers the 3 migrated sites only. |

## Did this project exist before the hackathon?

No. Offramp was created during the submission period: the first commit in `vj-io/offramp` is `13f3098` on 2026-10-19, and the demo app `vj-io/offramp-demo-app` was also first committed on 2026-10-19.
