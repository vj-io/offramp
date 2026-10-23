# Feedback: Nebius Token Factory, NVIDIA Nemotron, and the other services we used

Written while building Offramp (Nebius x NVIDIA Global AI Hackathon, 2026-10-19 onward). Offramp runs Nemotron 3 Super and Nemotron 3 Ultra on Token Factory as its agents, and Nemotron 3.5 Lightning, Super and Ultra as the models under test, so these notes come from real calls, not a reading of the docs. Each item says what we did, what broke or surprised us, how we worked around it, and what we would ask for.

Evidence paths are all in this repo. Recorded responses and tests replay offline (`uv run pytest -q`); where the only record of a live call is the project's build log, a plain-text extract is committed under `docs/evidence/`. Every probe row is a single call at `temperature=0`, so read the numbers as observations, not benchmarks.

## Top asks

Ranked by how many Token Factory users we think each one reaches. P1 means it breaks a correct client today; P2 means it costs time to find; P3 is polish.

| Priority | Ask | Who it affects | Item |
|---|---|---|---|
| P1 | Give reasoning its own budget, or say in `finish_reason` that reasoning used up `max_tokens` | every client that sends `max_tokens` with thinking on (the default) | [2](#2-reasoning-tokens-count-toward-max_tokens-and-an-exhausted-budget-returns-nothing) |
| P1 | Publish per-model latency, or a status page: Lightning answered ~58-token replies in a median 78 s on 2026-10-22, against 1.5 s on 2026-10-19 | anyone who calls Lightning with a client timeout | [19](#19-lightning-was-about-50-times-slower-on-2026-10-22-than-on-2026-10-19) |
| P2 | Stop forced tool calls on Lightning after one call, or reject `tool_choice="required"` with a 400 | agents that force a tool call | [4](#4-lightning-with-tool_choicerequired-loops-until-max_tokens) |
| P2 | Document `chat_template_kwargs.enable_thinking` with the request parameters | every structured-output client | [1](#1-reasoning-is-on-by-default-the-off-switch-is-a-chat-template-kwarg) |
| P2 | Say that a `json_schema` reply cut at `max_tokens` is not valid JSON, and support `maxLength` | batch judges and extractors | [20](#20-a-json_schema-reply-cut-at-max_tokens-is-invalid-json) |
| P3 | Make the price list public, or document the `pricing` field of `/v1/models?verbose=true` | cost tools, readers without an account | [7](#7-the-token-factory-price-page-is-login-gated) |
| P3 | Let hackathon credits fund a small AI Cloud job, or say up front that they cannot | hackathon participants | [14](#14-token-factory-credits-cannot-fund-ai-cloud-so-we-could-not-run-a-real-serverless-job) |

## Token Factory by the numbers

From [`docs/evidence/tf-production.md`](evidence/tf-production.md): 1,768 distinct chat completions that Offramp recorded between 2026-10-19 and 2026-10-22 (UTC), from 15 pipeline runs on two repos, their case generation, and the committed test cassettes. Latency is client-side wall time. Failed requests leave no record, so the error rate is unknown, not zero.

| Model | Calls | Latency p50 / p95 | `finish_reason: length` | Structured replies that were valid JSON |
|---|---|---|---|---|
| Lightning | 532 | 3.1 s / 110.8 s | 2 (0.4%) | 83 of 83 |
| Super | 583 | 3.1 s / 12.5 s | 3 (0.5%) | 197 of 197 |
| Ultra | 653 | 3.8 s / 11.0 s | 4 (0.6%) | 282 of 286 |

Lightning's p95 comes from the calls of 2026-10-22 (item 19). The 4 invalid Ultra replies all ended on `length`. These counts read Offramp's response cache, so they miss 5 Ultra judge replies from the open-source run `20261022-153359255-7bea7c` that did not match the rubric schema: Offramp asked each batch again with twice the room and deleted the first reply (item 20). Provider prompt-cache hits, from `prompt_cache_hit_tokens`: Ultra 409,088 of 2,566,410 input tokens (15.9%), Super 912 of 813,222 (0.1%), Lightning none.

## All items

| # | Item | Where it bit us | Our workaround |
|---|---|---|---|
| 1 | Reasoning is on by default, and the off switch is not a standard parameter | every structured call | always send `chat_template_kwargs.enable_thinking` |
| 2 | Reasoning tokens count toward `max_tokens` | empty content, `finish_reason=length` | budget at least 2000 tokens when thinking is on |
| 3 | Lightning put reasoning prose into `content` | classification probe | strip, use `content` only, never evaluate Lightning with thinking on unless chosen |
| 4 | Lightning with `tool_choice="required"` loops | every tool-call site | send `"auto"` on Lightning |
| 5 | `reasoning_effort="low"` does very different things per tier | reasoning budget | not used |
| 6 | Model ids are inconsistent and discovered by trial | config | verified once with `GET /v1/models` |
| 7 | The price page is login-gated | cost estimates | prices read from `GET /v1/models?verbose=true` |
| 8 | Super escapes code badly inside JSON | patch writing, fixture mapping | plain-text edit blocks, decode escapes |
| 9 | Ultra followed an instruction inside the input | blind reference audit | audit disagreements are dropped |
| 10 | Super invented a schema reference in a free-form JSON field | call-site labelling | schema comes from the code |
| 11 | An Ultra judge batch left a case unscored | first live demo run failed | re-ask about skipped cases alone |
| 12 | Super dropped spec details when writing call-site patches | patch stage, 5 of 6 sites | code writes the call; the model edits tests only |
| 13 | Ultra repair replies broke the edit-block format | 3 repairs on 2 sites | the two slips seen are now read as format only |
| – | Tavily answers picked other models' prices | price lookup | verify on the cited page |
| 14 | Token Factory credits cannot fund AI Cloud Serverless Jobs | eval matrix on Nebius | local worker, job path tested against a fake CLI |
| 15 | A fresh CLI profile lists no tenant, with no hint why | AI Cloud setup | open the console once |
| 16 | `iam v2 project list --all` fails with an invalid `parentId` | AI Cloud setup | none found |
| 17 | Console onboarding created a project in each of 9 regions | AI Cloud setup | none |
| 18 | Without a bucket, Serverless Jobs move data only through 64 KiB injected files and logs (per the CLI reference) | job path design | chunked input files, results as base64 lines in the log |
| 19 | Lightning was about 50 times slower on 2026-10-22 than on 2026-10-19 | every Lightning eval on the open-source repo | a tier must answer within the call's own timeout |
| 20 | A `json_schema` reply cut at `max_tokens` is invalid JSON | an Ultra judge batch on the open-source repo | ask once more with twice the room |

## Token Factory API behaviour

### 1. Reasoning is on by default; the off switch is a chat-template kwarg

Every tier reasons unless the request carries `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`. That is a chat-template kwarg, not an OpenAI parameter, so it has to go through `extra_body`; we verified it with a probe. With it, `reasoning_content` disappears from the reply (all three tiers, probe rows `*/think_off`). Completion tokens for the same one-line ticket were 170 (Ultra, default) versus 10 (Ultra, thinking off), but Super went the other way: 148 by default versus 381 with thinking off, because the answer came back as a long formatted explanation. We therefore never rely on "off" to mean "short" and cap output with `max_tokens` and a schema.

- Workaround: `offramp/tf/client.py` always sends `enable_thinking` explicitly, and the caller's reasoning level decides it (`false` for every structured agent call).
- Ask: list `enable_thinking` with the request parameters in the Token Factory docs, and say which models honour it.
- Evidence: `docs/plan/tf-probe.md` (rows `ultra/plain`, `ultra/think_off`, `super/plain`, `super/think_off`), `scripts/probe_tf.py`.

### 2. Reasoning tokens count toward `max_tokens`, and an exhausted budget returns nothing

Lightning with `response_format=json_schema`, `max_tokens=400` and thinking on returned `content: ""` with `finish_reason: "length"` and 400 completion tokens: the reasoning used the whole budget before the first JSON character. The same call on Super and Ultra finished inside the cap (144 and 120 tokens). A client that only reads `content` sees an empty string. It happened again on a longer task: Super as patch writer with thinking on hit our 16k cap on 2 of 5 call sites and did not fix any of the first round's failures, so we turned thinking off for that agent.

- Workaround: thinking off for structured calls; when thinking is on, at least 2000 tokens of budget.
- Ask: a separate reasoning budget (for example `max_reasoning_tokens`), or at least a `finish_reason` that says the reasoning ate the answer's budget, plus the partial `reasoning_content`.
- Evidence: `docs/plan/tf-probe.md` (rows `lightning/json_schema`, `super/json_schema`, `ultra/json_schema`); `docs/evidence/live-observations.md` (E1, the patch-writer rounds).
- Repro (one Lightning call, the probe's request from `scripts/probe_tf.py`; needs `NEBIUS_API_KEY`). On 2026-10-19 it printed `'' length 400`:

  ```bash
  uv run python - <<'EOF'
  import os
  from openai import OpenAI
  c = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1", api_key=os.environ["NEBIUS_API_KEY"])
  schema = {"name": "lbl", "strict": True, "schema": {"type": "object", "properties": {"label": {"type": "string", "enum": ["billing", "bug"]}}, "required": ["label"], "additionalProperties": False}}
  r = c.chat.completions.create(
      model="nvidia/Nemotron-3_5-Lightning", max_tokens=400, temperature=0,
      messages=[{"role": "user", "content": "Ticket: 'I was charged twice for invoice 991.' Classify."}],
      response_format={"type": "json_schema", "json_schema": schema},
  )
  print(repr(r.choices[0].message.content), r.choices[0].finish_reason, r.usage.completion_tokens)
  EOF
  ```

### 3. Lightning put reasoning prose into `content`

On the plain Lightning call (no reasoning flag, `max_tokens=400`), `content` began "Here's a thinking process: 1. **Analyze User Input:** ..." and `reasoning_content` was absent, while Super and Ultra returned their reasoning in `reasoning_content` and kept `content` clean. The same model with `enable_thinking=true` and `max_tokens=1500` returned a clean answer plus `reasoning_content`, so the leak may depend on the truncated default path; we did not isolate it further. `tool_calls` are affected too, see item 4.

- Workaround: strip `<think>...</think>`, build `ModelOutput.text` from `content` only, and never evaluate Lightning with thinking on unless the Adapter chose it.
- Ask: make Lightning's reasoning parser behave like Super's and Ultra's, so `content` is always the answer.
- Evidence: `docs/plan/tf-probe.md` (rows `lightning/plain`, `lightning/think_on`).

### 4. Lightning with `tool_choice="required"` loops until `max_tokens`

On the demo app's tool-calling agent step (cs_02), Lightning with `tool_choice="required"` returned 10 tool calls and stopped only at `max_tokens=512` (`finish_reason=length`), with raw `<tool_call><function=...>` markup also showing up in `content`. Adding `parallel_tool_calls=false` changed nothing. With `tool_choice="auto"` Lightning made one correct call, `lookup_order(HH-48213)`. Super with `"required"` made the same one correct call. The tool-call scorer compares the whole call list, so without the workaround every Lightning tool case would have failed for a reason unrelated to the model's judgement.

- Workaround: `build_request` sends `"auto"` to Lightning when the source code said `"required"`, and the PR flags the substitution (`lightning_tool_choice_auto`).
- Ask: fix the stop condition for forced tool calls on Lightning, or reject the combination with a 400.
- Evidence: `tests/agents/test_adapter.py::test_required_tool_choice_on_token_factory` (the looping and the no-effect of `parallel_tool_calls=false`, with the markup visible in two of the cassettes in `tests/fixtures/cassettes/adapter/`) and `test_adapted_requests_run_on_token_factory[cs_02]` (`"auto"` gives one correct call).
- Repro (sends the recorded request again, one Lightning call of about 900 input tokens; needs `NEBIUS_API_KEY`). The recorded reply printed `length 10 512`:

  ```bash
  uv run python - <<'EOF'
  import json, os, pathlib
  from openai import OpenAI
  path = "tests/fixtures/cassettes/adapter/e7dde4783f07a856afaf4ce54a6795418fb3e42c250aa469e6df377368446eba.json"
  req = json.loads(pathlib.Path(path).read_text())["request"]
  c = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1", api_key=os.environ["NEBIUS_API_KEY"])
  r = c.chat.completions.create(model=req["model"], messages=req["messages"], **req["params"])
  print(r.choices[0].finish_reason, len(r.choices[0].message.tool_calls or []), r.usage.completion_tokens)
  EOF
  ```

### 5. `reasoning_effort="low"` has a different effect on each tier

The parameter is accepted on all three, and the effect is inconsistent (completion tokens, `reasoning_effort="low"` versus `enable_thinking=true`, both with a 1500 cap): Ultra 221 versus 221 with the same opening text, Lightning 450 versus 464, Super 39 versus 161. Only Super got materially shorter.

- Workaround: we do not use `reasoning_effort`; reasoning is on or off.
- Ask: say which models honour effort levels, and return an error or a documented no-op where they do not.
- Evidence: `docs/plan/tf-probe.md` (rows `*/effort_low`, `*/think_on`).

### 6. Model ids are inconsistent and we had to discover them

The catalogue spells the three models we use three different ways: `nvidia/Nemotron-3_5-Lightning` (underscore in the version), `nvidia/nemotron-3-super-120b-a12b` (all lowercase) and `nvidia/Nemotron-3-Ultra-550b-a55b` (mixed case), while the same list has `nvidia/Nemotron-3.5-Super-VL` with a dot. Our first defaults came from third-party listings and were only confirmed by `GET /v1/models`. The catalogue also lists `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, which we did not use.

- Workaround: ids are configuration (`offramp/tf/tiers.py`, `OFFRAMP_MODEL_{LIGHTNING,SUPER,ULTRA}`).
- Ask: stable aliases such as `nemotron-3-super`, and one public table of id, tier, context length and supported features (`json_schema`, tools, reasoning switch) per model.
- Evidence: `docs/plan/tf-probe.md` ("Model ids confirmed via GET /v1/models").

### 7. The Token Factory price page is login-gated

`nebius.com/token-factory/prices` redirects to a login-gated organisation page, so a script, a judge or a reader without an account cannot read the price list there. The model API does carry the prices: `GET /v1/models?verbose=true` returns a `pricing` object per model, with `prompt` and `completion` in USD per token. On 2026-10-21 it listed exactly the three Nemotron prices our price file held ($0.06 / $0.24, $0.30 / $0.90 and $1 / $3 per 1M tokens). Offramp now reads it at run time, once a day (`offramp/tf/prices.py`), and labels the prices "Token Factory model API, <date>", with `offramp/evals/pricing.yaml` as the labelled fallback.

- Ask: document the `pricing` field of `/v1/models?verbose=true` (units, and whether it is the price a key is billed at), or make the price page public.
- Evidence: `offramp/tf/prices.py`, `tests/unit/test_tf_prices.py`, `offramp/evals/pricing.yaml` (the three tier entries, sourced to the API).

## What worked well on Token Factory

- **`json_schema` and function tools on all three tiers**, with `strict: true` schemas. A strict `OrderFields` schema on Lightning with thinking off extracted a German order email correctly. Evidence: `docs/plan/tf-probe.md` (rows `*/json_schema`, `*/tools`), `tests/agents/test_adapter.py::test_adapted_requests_run_on_token_factory[cs_03]` and its cassette.
- **Ultra as judge.** On 20 classification tickets we wrote, with a mix of correct, wrong and badly formatted answers, the Ultra judge agreed with the deterministic label scorer on 19 of 20, and its position-swap consistency was 100%. The one disagreement was `"Billing."`: the scorer forgives the trailing period, the judge took 2 points off for format. Evidence: `docs/evidence/judge-sanity.md` (reproduces the figures offline from `tests/fixtures/cassettes/judge/`), `tests/harness/test_judge_sanity.py`.
- **One OpenAI-compatible endpoint for all tiers**, so the tier matrix is three model ids and the client is the stock OpenAI SDK.
- **Prompt caching shows in `usage`.** Recorded responses carry `prompt_tokens_details.cached_tokens` plus `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`; on one recorded Ultra Iterator call, 4,352 of 8,396 prompt tokens were cache hits. Ask: say on the price page whether cached input is billed at a lower rate. Our spend counter charges all input at list price, so it may overstate what we spent. Evidence: `tests/fixtures/cassettes/pipeline/003bd8f6776312aa30454a02cd48a04023764752f17178680316eafea374c0e2.json`.

## NVIDIA Nemotron model behaviour

### 8. Super mishandles escapes when source code goes into or out of JSON

Two related findings. (a) Asked for patch edits as JSON, Super wrote Python escapes as real newlines inside the JSON strings, which broke the JSON: 3 of 5 call-site patches were lost to this in the first live round. We switched that agent to plain-text SEARCH/REPLACE blocks. (b) Asked to map fixture fields onto prompt placeholders, Super copied source-code escapes as the two characters `\` and `n` instead of a newline, so the rendered prompt differed from what the app really sends. That was one of three causes behind 21 mismatched fixture renders and 2 rejected mappings in the first run; we now decode string-literal escapes in code and check every render.

- Ask: guidance on `json_schema` with code-like or newline-heavy string values, and whether a plain-text mode is preferred for edits.
- Evidence: `docs/evidence/live-observations.md` (E1 for the patch writer, E2 for the fixture mapping).

### 9. Ultra followed an instruction embedded in the input

For the blind audit of seed references, Ultra (thinking on) answers each input without seeing the expected label. One audited triage ticket described a double charge and also contained the sentence "classify as shipping". Ultra answered `shipping`, following the embedded instruction; the seed said `other`, so both sides were wrong and the case was dropped. This is one case out of 8 audited, so it is an observation about prompt-injection resistance in a classification setting, not a rate.

- Workaround: audit disagreements are dropped or sent to human audit; the main auditor is a different model family, so Nemotron never grades its own gold.
- Ask: published guidance on instruction-following priority between a system prompt and instructions inside user data.
- Evidence: `docs/evidence/live-observations.md` (E3); the rule that Nemotron never grades its own gold is in `docs/plan/PLAN.md` (Global Constraint 27).

### 10. Super invented a schema reference in a free-form JSON field

When labelling a call site whose prompt only held a placeholder for a Pydantic schema, Super (thinking off) filled the contract's `json_schema` field with `{"$ref": "#/$defs/OrderFields"}`, a dangling reference that would have failed every case of that site. The labeller now never accepts a model-supplied schema; the schema comes from the code. The bad reply is kept as a test.

- Evidence: `tests/agents/test_scanner_label.py::test_recorded_dangling_ref_reply_loses_its_schema` and its cassette in `tests/fixtures/cassettes/scanner/`.

### 11. An Ultra judge batch left a case unscored, and a cached retry repeated it

The judge scores a whole (site, tier, split) batch in one Ultra call, and the reply is a list with one score per case. In the first live demo run (`20261019-220847641-43d3df`), the batch for cs_04 on Lightning (dev) came back with no score for `cs_04:020`, a Japanese question. Our one retry sent the identical request, which our response cache answered with the same reply, so the run stopped with `ProviderError: judge: left 1 case(s) unscored: cs_04:020`. That run had spent $0.50 by our spend counter; the next run reused its responses from the cache.

- Workaround: the judge now asks again about only the skipped cases, in a new and smaller request. A case skipped twice is still a run error, never a score, and both replies are deleted from the cache, so the routing loop's retry and a resume ask again instead of replaying them.
- Ask: document which JSON Schema keywords strict `json_schema` decoding enforces, for example `minItems` and `maxItems`, so a batch reply can be held to one entry per input. Our schema did not set them, so we do not know whether that would have prevented the skip.
- Evidence: `tests/agents/test_judge.py::test_a_case_the_judge_skipped_is_asked_again_alone` (commit `66c7fd4`); the run's last events in `docs/evidence/live-observations.md` (E5).

### 12. Super, writing call-site patches from a spec, dropped details of the spec

In the second live run (`20261019-222942730-2c0f0b`, whose evaluation the demo uses), Nemotron 3 Super wrote each migrated call site from the evaluated adapter: model id, `extra_body` with the thinking switch, few-shot constant, and the test edits. All 6 sites had passed parity, and 5 of the 6 patches were rejected, 4 by our patch checks and 1 by the repo's own tests. The few-shot constant was defined but not used (cs_01, cs_04), the old `messages.create` call stayed in place (cs_02), the model id was not the evaluated tier's and `extra_body` was missing (cs_04), two SEARCH blocks overlapped (cs_03), and a test that checks the prompt failed (cs_06). Only cs_05 was patched. Each rejection was a lost migration, not a wrong one, because the checks caught every bad patch.

- Workaround: code now writes the evaluated request into the call site (an AST rewrite, with the request as literal keyword arguments), and the model may edit only tests and test fakes. Every non-test file must match the code-written tree. With that change, the 2026-10-20 runs patched 3 sites. As shipped: after later patcher fixes (repairs applied to the files the model last saw, tolerated format slips, cleanup of imports Offramp's rewrite left unused), the final demo run `20261022-014146107-57cca9` patched 5 of 6.
- Ask: guidance on the edit format Super handles best (unified diff, SEARCH/REPLACE blocks, whole files). Ours was a guess.
- Evidence: `docs/evidence/live-observations.md` (E6, the rejection reasons quoted from that run's report).

### 13. Ultra repair replies broke the edit-block format

When the repo's checks reject a patch, Nemotron 3 Ultra (thinking off, `max_tokens=12000`) gets the failure and replies with new edit blocks. The prompt shows the format and says to begin with `FILE:`. A reply counts only if it holds complete `FILE:` / `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` blocks. Three replies did not, on the two sites that ended in manual review. The cs_01 repair in the fourth live run began with `FILES:` instead of `FILE:`. Both cs_03 repairs (third and fourth runs) ended their block with a bare `REPLACE` line instead of `>>>>>>> REPLACE`; each was one block whose SEARCH part (144 and 171 lines) started at the top of `tests/fakes.py`. All three ended with `finish_reason: "stop"` at 670, 1,826 and 2,924 completion tokens, far below the cap, so they were not truncated.

- Workaround, as shipped: the two slips seen (`FILES:`, and a bare `REPLACE` closing line) are read as format only, so the edit inside still applies and every check still runs on it. Any other reply with no complete block counts as a failed repair, and the site goes to manual review with that reason in the PR. In the 2026-10-20 runs we had no workaround.
- Ask: a constrained-output option for non-JSON formats (a regex or grammar `response_format`), so a fixed edit syntax can be enforced the way `json_schema` is.
- Evidence: the reasons in the PR body's "What Offramp did NOT change" section (`offramp/api/replay/demo-events.pr.md`); reply extracts in `docs/evidence/live-observations.md` (E7).

## Other services we used

### Tavily: answers and domain filters (for the Best Use of Tavily prize)

Offramp looks up each source model's price and deprecation notice with Tavily at run time (7 basic searches while building it, 5 kept as recordings).

- The `answer` field cannot be trusted for prices. In the recorded `o3` search (`tests/fixtures/tavily/o3.json`) Tavily's `answer` gives an output price of $80.00, which is `o3-pro`'s, and none of the 20 pages it returned prints `o3`'s own price: 6 are other o-series model pages (`o3-pro`, `o3-mini`, `o3-deep-research`), 13 are the o3/o4-mini launch post and its anchors, and the one pricing page came back as an excerpt that never mentions `o3`. A model extracting numbers can copy such a figure, so the extractor sees only the retrieved pages, and a price is accepted only if the cited page prints it against that exact model, not a similarly named one; otherwise we fall back to the price file and say so. In the 2026-10-20 run, `gpt-4o` and `o3` ended on that fallback because no returned page printed their own price. As shipped: when the search results print no such price, one Tavily Extract of the vendor's own model page follows, and its price may only confirm the checked-in one. In the final demo run (2026-10-22) all six source-model prices were checked this way: two by search, four by Extract.
- Ranking favours newer models: in the recorded `claude-sonnet-4-5` search (`tests/fixtures/tavily/claude-sonnet-4-5.json`), even with the model's table spelling in the query, 17 of the 20 results are Sonnet 5 pages and 2 are Sonnet 4.5 pages (ranked first). Our verifier has to tell them apart.
- `include_domains` matches subdomains, so `community.openai.com` (user posts, one with the wrong price) entered a real `gpt-4o-mini` search. We now send `exclude_domains` and also drop those results locally.
- Ask: an exact-domain option for `include_domains`, and the source passage behind each `answer`.
- Evidence: `tests/unit/test_tavily.py` (the search-request test and its subdomain comment; the recorded-search tests; the comment above `FELL_BACK` that counts the 7 searches), `tests/fixtures/tavily/*.json`.

### Toloka and Tendem

- The new Toloka platform (`platform.toloka.ai`, beta) has no API keys, so Offramp cannot call it; label checks would be a manual UI project. We left `TOLOKA_API_KEY` empty and did not integrate it. Evidence: `.env.example`.
- Tendem (by Toloka) worked through its MCP server (`https://mcp.tendem.ai/mcp`, streamable HTTP). The API key sits behind "Connect any MCP client with your own API key" on the agent page and is sent as `Authorization: ApiKey <key>`; `approve_task` spends credit, so every paid task needs a person's approval in our workflow. Evidence: `.env.example` (where the key is created and the header), `docs/evidence/live-observations.md` (E4).

## Nebius AI Cloud

First-hand notes from setting up AI Cloud for the eval matrix, observed during the build (2026-10-19, `nebius` CLI 0.12.292). They come from an interactive account setup, so there is nothing to replay; the repo's job path is built and tested against a fake `nebius` CLI and has not been run against a real account (see items 14 and 18).

### 14. Token Factory credits cannot fund AI Cloud, so we could not run a real Serverless Job

The Token Factory balance sits in an organisation (`aitenant-...`) that is a separate tenant from AI Cloud, and it cannot pay for Serverless Jobs. AI Cloud onboarding asks for a payment card and a $25 top-up. Our budget was hackathon credits only, so we did not run the eval matrix as a Serverless Job: the submit path (`nebius ai job ...`) is tested against a fake CLI on `PATH`, and the matrix runs on a local worker with the same output format.

- Ask: let hackathon credits (or the Token Factory balance) cover a small AI Cloud job, or give participants a job-sized AI Cloud grant; say up front in the hackathon materials that the two balances are separate.

### 15. A fresh CLI profile lists no tenant and does not say why

After `nebius profile create`, `nebius iam v2 tenant list` returned `{}`. Nothing hinted that the tenant is created on the first visit to the web console. Separately, `nebius iam tenant list` printed a deprecation warning pointing at `iam v2 tenant`.

- Ask: make the empty result explain itself (for example "no tenant yet: open the console once to create it").

### 16. `nebius iam v2 project list --all` fails instead of listing

`nebius iam v2 project list --all` failed with `'' is invalid for parentId` (InvalidArgument) instead of listing projects across tenants.

- Ask: either make `--all` work, or return an error that names the parent it needs.

### 17. Console onboarding created a default project in each of 9 regions

Onboarding through the console auto-created a default project in each of 9 regions.

- Ask: create one project in the chosen region, or ask which regions to create.

### 18. Serverless Jobs: what we built, and what we could not check

No Serverless Job ran on Nebius. We wrote the job path for the eval matrix (`offramp/evals/serverless_job/`, image `Dockerfile.eval`) from the CLI reference (`docs.nebius.com/cli/reference/ai/job/{create,get,logs,cancel}`) and `nebius ai job create --help`, and tested it against a fake `nebius` on `PATH`. AI Cloud billing needs a payment card and a $25 top-up, and this project spends only granted credits (item 14). Every demo run, the recorded one included, logged `eval matrix (18 cells) ran locally (nebius job: not configured)`. Building it still taught us three things:

- **Data in and out.** As we read the reference, a job receives files only through `--inject-file` (64 KiB per file) and, without a mounted bucket, returns data only through its logs. So the matrix and the cached responses it replays go in as a tar.gz cut into 48 KiB parts (at most 32, about 1.5 MB), and the results come back as base64 lines of 8,000 characters on stdout, below the 16 KiB at which container runtimes split a log line. Ask: a small output directory returned with the job, or larger injected files, so a short batch job needs no bucket.
- **Secrets.** `--env-secret NAME=SECRET_ID` needs the environment variable's name to equal the key inside the MysteryBox secret's payload; we found no other syntax, so a secret stored under a different key name cannot be mapped. Ask: a form that names the payload key, such as `NAME=SECRET_ID:payload_key`.
- **What only a real account can confirm.** Whether a real job accepts our flags as we use them (`--args` split on spaces, several `--inject-file`), the CPU platform and preset names (`cpu-d3`, `4vcpu-16gb`; the default is a GPU platform, and this work is network-bound), the field names of `nebius ai job get --format json`, and the log line format. Ask: a validate or dry-run mode for `nebius ai job create` that works before billing is set up, and an example CPU job in the docs.

Evidence: the module docstrings of `offramp/evals/serverless_job/submit.py` and `offramp/evals/serverless_job/transport.py`, `tests/integration/test_serverless_job.py` (the fake CLI), and the log line in `web/public/replay/demo-events.json` (event 18).

## Found on a real open-source repo (2026-10-22)

Offramp also ran on a fork of [`mnemox-ai/idea-reality-mcp`](https://github.com/mnemox-ai/idea-reality-mcp) (MIT). Each of its five Anthropic call sites builds `AsyncAnthropic(timeout=30.0)` inside the function, so every call has a 30 s budget in production. Two items come from those runs.

### 19. Lightning was about 50 times slower on 2026-10-22 than on 2026-10-19

On 2026-10-19, the demo app's Lightning eval calls with thinking off took a median 1.5 s (219 calls; median 441 input and 58 output tokens). On 2026-10-22, between 02:00 and 08:00 UTC, the open-source repo's Lightning eval calls with thinking off took a median 78.0 s (160 calls; median 439.5 input and 57.5 output tokens). The median sizes were nearly the same; the time was not. Per (site, adapter) cell, Lightning's dev median ran from 39.7 s to 109.1 s in the two runs that made those calls (`20261022-023459043-590031` and `20261022-065443547-22ed36`), and one cell's p95 was 485.7 s. Super and Ultra answered in a median 4.0 s and 4.7 s on the same day, with longer replies (median 172 and 426.5 output tokens).

The repo's own client gives up after 30 s, so at that speed Lightning could not serve these calls. Offramp now fails a tier whose holdout p50 is over the call's own timeout, and skips iterating a tier whose dev p50 is over twice it. In the final run, `20261022-153359255-7bea7c`, Lightning was not iterated on 4 sites (dev medians 64.3, 71.0, 76.7 and 97.5 s, measured on 2026-10-22 and served from the cache).

- Limits of the measurement: latency is the client's wall time, so it includes queueing in Offramp's own pool (at most 8 eval calls in flight, on both days) and the SDK's retries. Failed requests leave no record. It is one six-hour window, so we cannot say whether it lasted.
- Ask: per-model latency on a public status page, and a timing header on each response that splits queue time from generation time.
- Evidence: [`docs/evidence/live-observations.md`](evidence/live-observations.md) (E8, the per-day medians from the response cache); [`docs/evidence/tf-production.md`](evidence/tf-production.md) ("Run errors" lists the four "Lightning not iterated" lines); each migrated or kept site's line in <https://github.com/vj-io/idea-reality-mcp/pull/1>; `tests/agents/test_verdict.py::test_a_tier_whose_dev_median_is_over_twice_the_timeout_is_not_iterated`.
- Repro (three short calls, one per tier; needs `NEBIUS_API_KEY`). On 2026-10-19 the same Lightning request took 0.7 s for 60 tokens (`docs/plan/tf-probe.md`, row `lightning/think_off`):

  ```bash
  uv run python - <<'EOF'
  import os, time
  from openai import OpenAI
  c = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1", api_key=os.environ["NEBIUS_API_KEY"])
  msg = [{"role": "user", "content": "Ticket: 'I was charged twice for invoice 991.' Classify."}]
  for model in ["nvidia/Nemotron-3_5-Lightning", "nvidia/nemotron-3-super-120b-a12b", "nvidia/Nemotron-3-Ultra-550b-a55b"]:
      t = time.time()
      r = c.chat.completions.create(model=model, messages=msg, max_tokens=400, temperature=0,
                                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
      print(model, f"{time.time() - t:.1f} s", r.usage.completion_tokens, "tokens")
  EOF
  ```

### 20. A `json_schema` reply cut at `max_tokens` is invalid JSON

The Ultra judge scores a batch of cases in one call, with a `json_schema` response format (not strict) and thinking off. In run `20261022-065443547-22ed36`, the batch for the open-source repo's cs_05 Lightning dev cell (14 cases, 24,864 prompt tokens) stopped at `max_tokens` 2,500 with `finish_reason: "length"`. The prompt asks for a "note" of one short sentence per case. The 8 complete notes ran from 204 to 1,931 characters, and the JSON was cut inside the ninth entry. The reply did not parse, our response cache had stored it, and the cell was left unscored. It is the one `length` finish among the 141 recorded judge calls, but not the only judge reply that failed: in the later run `20261022-153359255-7bea7c`, 5 more judge batches were asked again with twice the room because the first reply did not match the rubric schema. Offramp had deleted those first replies from the cache, so we cannot say how they ended, and they are in none of our counts. At least 6 judge replies were off the rubric; 1 is recorded.

- Workaround: the judge asks once more with twice the `max_tokens`, as a new request, so the cache cannot serve the cut reply. A reply that does not parse stays cached while that retry succeeds, so a replay walks the same two requests; when the retry fails too, both are deleted, and the routing loop runs the cell once more, live; only if that fails as well does the run fail, and a resume asks again. Run `20261022-153359255-7bea7c` used an earlier version that deleted the first reply even when the retry succeeded.
- Ask: (a) say in the docs that a `length` finish voids the schema, or return an error rather than half a document; (b) support `maxLength` on string fields in `json_schema` decoding, so a note's length can be held. Our schema did not set `maxLength`, so we do not know whether it is honoured today. (c) Ultra wrote notes up to 1,931 characters where the prompt asked for one short sentence; guidance on length instructions would help.
- Evidence: `tests/fixtures/judge/truncated_reply_run_22ed36.json` (the cut reply's text), `tests/agents/test_judge.py::test_a_reply_cut_off_mid_json_is_asked_once_more_with_more_room`, [`docs/evidence/live-observations.md`](evidence/live-observations.md) (E9).
- Repro: the recorded request is not committed (its prompt holds the open-source repo's eval cases). Any `json_schema` call whose answer outgrows `max_tokens` shows the shape. We did not run this exact request:

  ```bash
  uv run python - <<'EOF'
  import json, os
  from openai import OpenAI
  c = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1", api_key=os.environ["NEBIUS_API_KEY"])
  item = {"type": "object", "properties": {"case": {"type": "integer"}, "note": {"type": "string"}}, "required": ["case", "note"]}
  schema = {"name": "batch", "schema": {"type": "object", "properties": {"items": {"type": "array", "items": item}}, "required": ["items"]}}
  r = c.chat.completions.create(
      model="nvidia/Nemotron-3-Ultra-550b-a55b", max_tokens=300, temperature=0,
      messages=[{"role": "user", "content": "For cases 1 to 30, write a three-sentence note on why each number is interesting."}],
      response_format={"type": "json_schema", "json_schema": schema},
      extra_body={"chat_template_kwargs": {"enable_thinking": False}},
  )
  print(r.choices[0].finish_reason)
  try:
      json.loads(r.choices[0].message.content)
      print("valid JSON")
  except json.JSONDecodeError as e:
      print("invalid JSON:", e)
  EOF
  ```
