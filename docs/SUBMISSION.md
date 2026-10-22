# Devpost submission text

Ready to paste into the Devpost form, one section per field. The demo numbers come from the final demo run `20261022-014146107-57cca9`: its replay (`web/public/replay/demo-events.json`), its pull request <https://github.com/vj-io/offramp-demo-app/pull/3>, and `evals/results/*.json` in `vj-io/offramp-demo-app` at `24b97e0`. The no-Ultra numbers come from run `20261022-015732734-b5aee2` (`web/public/replay/no-ultra-events.json`), and the open-source numbers from run `20261022-153359255-7bea7c` (`web/public/replay/oss-events.json`, PR <https://github.com/vj-io/idea-reality-mcp/pull/1>). Spend comes from each run's `budget.json`, listed in the README's [What it cost](../README.md#what-it-cost).

## Project name

Offramp

## Tagline

Tests each closed-LLM call in a Python repo against NVIDIA Nemotron on Nebius Token Factory, then opens a pull request for the calls that pass.

## Track

Coding and Agentic Engineering

## Inspiration

Moving a product off a closed LLM API sounds like a base-URL change, and teams have reasons to do it: price, vendor lock-in, and deprecations. Offramp's own Tavily lookup found that Claude Sonnet 4.5 retires on 30 November 2026, and two of our demo app's calls use it. A static scan of 30 popular public Python repos found 50 chat calls that name their model; 33 of them use a model its provider has retired or scheduled for shutdown. What stops teams from moving is risk they cannot see: JSON that no longer parses, tool schemas that drift, answers that are slightly worse. Nobody wants to migrate on a vibe check, and nobody has time to check parity call site by call site. We wanted a tool that does the checking and hands over a pull request a reviewer can trust.

## What it does

Point Offramp at a Python repo. It:

1. **Scans** for every OpenAI, Anthropic, LiteLLM and LangChain call with a deterministic AST pass, and Nemotron 3 Super labels each call's contract: classification, JSON extraction, tool call, free text or reasoning.
2. **Builds an eval set per call site**, split into dev and holdout by a hash of the case id. Expected outputs are reference answers: seeded (the expected label, record or tool call is chosen first, and Super writes an input that implies it), computed by code, or, for free text, written by Claude Sonnet 5.5, a different model family. Seed references are audited blind; the free-text ones were not audited.
3. **Runs a tier matrix** on Nebius Token Factory: Nemotron 3.5 Lightning, then Nemotron 3 Super, then Nemotron 3 Ultra. Deterministic scorers check schemas, labels, tool names and arguments, and numeric answers; an Ultra judge scores free text.
4. **Iterates.** When a cheaper tier fails on dev, Nemotron 3 Ultra reads the failing cases and rewrites the adapter (prompt, few-shot examples, schema, reasoning switch), up to 3 times per tier. The Iterator never sees holdout cases.
5. **Decides** per call site: migrate to the cheapest tier that passes dev and then holdout, keep the source model when no tier does, or hand the site to manual review. When the call sets a timeout, a tier must also answer within it.
6. **Patches and opens a PR.** Code rewrites each migrated call, and Nemotron edits only the tests and fakes that pinned the old provider. The repo's own ruff and pytest run in a network-off Docker sandbox, and any site whose patch cannot be verified goes to manual review with the exact reason. The PR carries an evidence table, how parity was measured, failing cases, prices with their sources, and a "What Offramp did NOT change" section.

A dashboard shows it live: stage stepper, call-site cards, a parity-by-tier grid, the log and a report page.

**On our demo app** (a small support-desk app with six chat calls on `gpt-4o-mini`, `gpt-4o`, `o3`, `claude-sonnet-4-5` and `claude-haiku-4-5`), Offramp evaluated 121 cases, 40 of them held out. All 6 call sites reached the parity bar on a Nemotron tier. 5 migrated in [one pull request](https://github.com/vj-io/offramp-demo-app/pull/3): two Claude Sonnet 4.5 calls to Super, the RAG answerer to Ultra, and the o3 refund planner and the Haiku summariser to Lightning. The demo repo's own CI passes on that PR. Estimated cost per 1,000 calls to each site falls from $49.21 to $2.61 on the 5 migrated sites (−95%), and from $49.24 to $2.65 across all 6 (−95%). The o3 site is most of the "before"; without it, the other five sites (four migrated, plus the ticket classifier at its source model's price) go from $9.74 to $1.46 (−85%). The sixth site, the ticket classifier, went to manual review: the model's edit to its test would have dropped three asserts, so Offramp refused it.

**The gate saying no.** Rerun with a team policy of "no Ultra", Offramp kept the RAG answerer on `gpt-4o`: Lightning and Super both failed the dev split, Super with a best judge mean of 4.45, more than 0.3 below the references' mean.

**On a real repo.** We forked [`mnemox-ai/idea-reality-mcp`](https://github.com/mnemox-ai/idea-reality-mcp), an MIT-licensed MCP server we did not write, and ran Offramp on the fork. Of its five async Anthropic calls, two migrated to Super ([PR on the fork](https://github.com/vj-io/idea-reality-mcp/pull/1); est. −78% on those two), two stay on Claude because Super fell below the bar on held-out cases, and one needs a person because its prompt is built at runtime. Lightning was out on every site it was tried on: its median answer took 64.3 to 97.5 s that day, and the repo's own client gives up after 30 s.

**Try it on your own code:** the scan needs no key. Run `uvx --from git+https://github.com/vj-io/offramp offramp scan .` or paste a GitHub URL at <https://offramp-seven.vercel.app/scan>.

## How we built it

- **Backend:** Python 3.12 with FastAPI, Pydantic and SQLModel; a run is a state machine that streams its events over server-sent events.
- **Models:** every inference call in Offramp's code goes through one client, the OpenAI Python SDK pointed at Nebius Token Factory. Nemotron 3 Super and Nemotron 3 Ultra are the agents (labelling, case generation, adapter, iterator, judge, patch writer, PR prose), and Nemotron 3.5 Lightning, Super and Ultra are the models under test. Structured calls use `temperature=0`, `json_schema` output, an explicit `enable_thinking` switch and a per-agent token cap.
- **Cache as cassettes:** every response is stored by a hash of model, messages and parameters. Cache hits are free, a finished run can be replayed from the cache without a new call, and the test suite replays real Token Factory output without a key. The exception is the open-source PR run `20261022-153359255-7bea7c`: a bug deleted judge replies it would need, and it is fixed for later runs.
- **Budget guardrails:** caps per run on input and output tokens and on Ultra agent calls; hitting one keeps the affected sites on their source model instead of crashing.
- **Patching:** an AST rewrite writes the evaluated request into each call site as literal keyword arguments, and a check rejects any model edit to a non-test file. The target repo's own checks run in Docker with the network off, on the repo's tracked files only.
- **Partners:** Tavily looks up each source model's published price and deprecation notices at run time, and a price is accepted only if the cited page prints it next to that exact model. LangSmith traces every Token Factory call and keeps each site's cases as a dataset. Claude Sonnet 5.5, run through Claude Code during the build, wrote the free-text references and blind-audited the seed references, so Nemotron never grades its own gold.
- **Dashboard:** Next.js and Tailwind CSS on Vercel's free tier. The recordings replay entirely in the browser, and `/scan` runs the static scanner in a Python function.

## Challenges we ran into

- **No closed-model keys.** We had only hackathon credits, so the source models' own outputs could not be recorded. We defined parity as agreement with reference answers (seeded, computed by code, or written by Claude Sonnet 5.5 for free text), labelled every reference with where it came from, and audited the seed references blind with a different model family.
- **Super was unreliable at writing shipped code to a spec.** Asked to rewrite call sites from the evaluated adapter, Super's patches were rejected on 5 of 6 sites. We moved the call-site rewrite into deterministic code and let the model edit tests only. That took the demo from 1 patched site to 3, and later fixes to the repair loop took it to 5.
- **A real repo broke our assumptions.** The first two runs on the open-source repo found five bugs in Offramp: repeated eval cases leaking across the dev and holdout splits, an Ultra budget spent on the wrong work, a degenerate adapter (one word repeated about 750 times) that passed the metric, parity measured without the call's own 30 s timeout, and a judge reply cut at `max_tokens` that the cache kept serving. We fixed each, with tests, before the run that opened the PR.
- **Nemotron and Token Factory specifics:** reasoning is on by default and its tokens count toward `max_tokens`; Lightning with `tool_choice="required"` looped until the token cap; and on 2026-10-22 Lightning answered about 50 times slower than on 2026-10-19 for replies of the same size. `docs/FEEDBACK.md` has priorities, evidence and repro steps.
- **AI Cloud billing.** Running the eval matrix as a Nebius AI Cloud Serverless Job needed a payment card and a $25 top-up, separate from the Token Factory credits. We built the job path and tested it against a fake CLI, but it never ran on Nebius.
- **Small holdouts.** With 3 to 11 holdout cases per site, one miss moves parity by 9 to 33 points, so the report shows n and a Wilson 95% lower bound for every site.

**Plan versus shipped.** Our PRD is public, so here is where we fell short of it:

| Planned | Shipped |
|---|---|
| A baseline recorded from the original closed model | No: parity is against references, because we had no OpenAI or Anthropic key. The recorder exists and is tested with fake clients |
| Paste a repo URL on the website, watch it live, get a PR | Partly: the website plays recordings and runs the keyless scan; live runs need a local API and your own key, because hosting them needs paid compute |
| The eval matrix as a Nebius Serverless Job | Built and tested against a fake CLI, not run on Nebius (AI Cloud needs separate billing) |
| One `KEEP_CLOSED` row in the main demo, from the o3 site | No: that site passed on Lightning in every demo run. `KEEP_CLOSED` shows in the no-Ultra run and on the real repo instead |
| Human audit of references through Tendem | Built, not used |
| Everything else (scan, tier matrix, iteration, verdicts, a real PR, Token Factory and Nemotron throughout) | Shipped |

## Accomplishments that we're proud of

- A demo run that ends in a real, green pull request: 6 call sites, 121 cases, 5 migrations, each with evidence, and one honest "a person should look at this" instead of a test edit we could not trust.
- A real open-source repo, where Offramp said no to two of five calls and said why, in numbers.
- Offramp writes every change to application code itself; Nemotron edits only tests and test fakes, and the PR lists the test files each model edit touched.
- Every number on screen and in the PR is rendered from run artifacts. Costs are labelled as estimates with their price source and date, and latency before migration is shown as not measured.
- All of it cost $4.13 of the $25 Token Factory grant by our spend counter: $3.51 over 15 pipeline runs, plus case generation and cassette recordings.
- Judges can watch all three recorded runs at <https://offramp-seven.vercel.app> with no login and no keys, or scan their own repo there.

## What we learned

- Deterministic scorers should decide wherever they can. `json_schema` and tool calling worked on all three Nemotron tiers, so only free-text sites needed a judge.
- The cheapest tier is often enough, but not always: on the demo, three sites passed on Lightning (one of them then went to manual review), two needed Super, and the RAG answerer needed Ultra after three adapter rewrites on Lightning and one on Super failed (Super's next two rewrites repeated that one).
- Let models propose and code decide. Every allowance for model edits to shipped code reopened a way for unverified changes to slip through; restricting the model to tests closed it.
- A real repo is worth more than a polished demo. Two runs on one found five bugs that our demo app never triggered.
- Honest labels matter more than flattering numbers: "est.", "parity vs reference", "manual review" and the scope of every cost figure.

## What's next for Offramp

- Run the eval matrix as a Nebius AI Cloud Serverless Job once billing allows, and host the API.
- Runtime capture for prompts built at runtime, which the static scan cannot render.
- Closed-model baselines for teams that have their source-model keys.
- Human audit of references through Tendem.
- Use the eval sets Offramp builds as fine-tuning data for the calls that stay on a closed model.

## Built with

Python, FastAPI, Pydantic, SQLModel, OpenAI Python SDK, Nebius Token Factory, NVIDIA Nemotron 3.5 Lightning, NVIDIA Nemotron 3 Super, NVIDIA Nemotron 3 Ultra, Next.js, React, TypeScript, Tailwind CSS, Docker, Tavily, LangSmith, Vercel, GitHub, uv, pnpm, Claude Code

## Testing instructions

- **Hosted, no login and no keys:** open <https://offramp-seven.vercel.app> and click "Watch demo run (replay)", or go straight to <https://offramp-seven.vercel.app/runs/demo?replay=demo>. At 4x speed the replay plays in about 14 seconds (the event gaps add up to 14.2 s); "Skip to end" jumps to the result. Then open the report: <https://offramp-seven.vercel.app/runs/demo/report?replay=demo>.
- **The gate saying no:** <https://offramp-seven.vercel.app/runs/demo?replay=no-ultra> (the same app with Ultra not allowed).
- **A real open-source repo:** <https://offramp-seven.vercel.app/runs/demo?replay=oss>, and the PR it opened on our fork: <https://github.com/vj-io/idea-reality-mcp/pull/1>.
- **The demo pull request:** <https://github.com/vj-io/offramp-demo-app/pull/3>, with the repo's own CI checks.
- **Your own repo, no key:** paste a public GitHub URL at <https://offramp-seven.vercel.app/scan>, or run `uvx --from git+https://github.com/vj-io/offramp offramp scan .` in a checkout (needs uv). Both list the call sites, their call shapes and the list-price ratio to each Nemotron tier, without importing or running the code. Measuring parity takes a full run with your own Token Factory key.
- **Locally,** with Python 3.12+ and uv, Node 20.9+ and pnpm, make and curl:

  ```bash
  git clone https://github.com/vj-io/offramp.git && cd offramp
  make demo
  ```

  `make demo` installs dependencies, starts the API on `:8000` and the dashboard on `:3000`, and opens the replay. No key is needed. `make test` runs the Python suite on recorded Token Factory responses, also without a key.

## Gallery images

1500x1000 PNGs in `docs/media/`, in this order. `00` and `07` were drawn as HTML and rendered with Playwright; `01`, `02`, `03`, `05` and `06` were taken with Playwright from the dashboard (local build, 2026-10-22); `04` is the GitHub pull request page, cropped below the site header.

| File | Caption |
|---|---|
| `00-cover.png` | Offramp on our demo app: 6 of 6 LLM calls reached parity on held-out cases, and 5 migrated to Nemotron in one pull request, est. −95%. |
| `01-replay-mid-run.png` | Mid-run: the RAG answerer has failed on Lightning after three adapter rewrites, and Nemotron 3 Ultra has just written adapter v1 for Super. |
| `02-replay-done.png` | Done: the parity-by-tier grid with the chosen tier circled for each migrated call site, and PR #3 linked at the top. |
| `03-report-summary.png` | The report: 5 of 6 calls migrated, est. −95%, the Claude Sonnet 4.5 retirement date, and the top of the evidence table with each site's tier, parity, latency and cost. |
| `04-pull-request.png` | The pull request Offramp opened on the demo repo: the retirement line, the summary and the evidence table. The repo's own CI passes on it. |
| `05-gate-says-no.png` | The gate saying no, on a real open-source repo: two calls stay on Claude because Super fell below the bar on held-out cases. |
| `06-real-repo.png` | The run on the open-source repo: two calls migrate to Super, two stay on Claude, one needs a person; Lightning was ruled out by the call's own 30 s timeout. |
| `07-architecture.png` | How Offramp works: every model call goes to Nebius Token Factory; models propose, code decides and writes the shipped code. |

## Did this project exist before the hackathon?

No. Offramp was created during the submission period: the first commit in `vj-io/offramp` is `13f3098` on 2026-10-19, and the demo app `vj-io/offramp-demo-app` was also first committed on 2026-10-19.
