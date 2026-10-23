## Summary

Offramp evaluated 6 LLM call sites against references (69 cases, 24 in the holdout split). 3 migrated to the cheapest Nemotron tier that passed parity on the holdout split; 3 kept on the original model.

## Evidence

| Site | File | Task | Before | After | Parity vs reference (holdout) | p50 latency (after) | $/1k calls before (est.) → after |
|---|---|---|---|---|---|---|---|
| cs_01 | `helphub/inbox/triage.py` | classification | gpt-4o-mini | **NOT MIGRATED** | 85.7% (6/7, ≥48%) < 95% | — | — |
| cs_02 | `helphub/ops/agent_step.py` | tool_call | claude-sonnet-4-5 | **NOT MIGRATED** | 50.0% (1/2, ≥9%) < 95% | — | — |
| cs_03 | `helphub/orders/extract.py` | json_extraction | claude-sonnet-4-5 | **NOT MIGRATED** | 71.4% (0/2 pass, ≥0%) < 95% | — | — |
| cs_04 | `helphub/qa/rag_answer.py` | free_text | gpt-4o | Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | 100.0% (5/5, ≥56%), judge mean 5/5, lowest case 5/5 | 1.0 s | $2.04 (est.) → $0.049 |
| cs_05 | `helphub/refunds/plan.py` | reasoning | o3 | Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | 100.0% (4/4, ≥51%) | 15.0 s | $41.58 (est.) → $1.25 |
| cs_06 | `helphub/summaries/summarize.py` | free_text | claude-haiku-4-5 | Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | 100.0% (4/4, ≥51%), judge mean 5/5, lowest case 5/5 | 2.6 s | $0.84 (est.) → $0.045 |

## What changed

- `helphub/offramp_client.py` added inside the `helphub` package, so it ships with it: OpenAI SDK client for Nebius Token Factory (`token_factory_client()`, reads `NEBIUS_API_KEY` and optional `NEBIUS_BASE_URL`) and the Nemotron tier constants `LIGHTNING`, `SUPER`, `ULTRA`. Every migrated call sends `chat_template_kwargs.enable_thinking` explicitly, since Nemotron reasons by default.
- `answer_question`, `plan_refund`, `summarize_thread` get the client from callers: callers were checked statically; lookups through reflection (getattr, importlib, sys.modules, vars/__dict__) were not, except the literal getattr/import_module forms Offramp refuses.
- Imports that Offramp's or the model's changes left unused were removed: `Anthropic` from `helphub/summaries/summarize.py`.
- **cs_04** `helphub/qa/rag_answer.py` → Nemotron 3.5 Lightning. `answer_question` now sends the request that passed parity on the holdout split: judge mean 5/5 over 5 cases, lowest case 5/5. Files: `helphub/cli.py`, `helphub/qa/rag_answer.py`. Tests updated (they pinned the old provider's request or fakes): `tests/test_call_sites.py`, `tests/test_cli.py`, `tests/test_rag_answer.py`. Request check: the code Offramp wrote, unchanged; not probed: no eval case's variables name its parameters (question, articles). 16 tests ran in test files that name `answer_question`: `tests/test_rag_answer.py`.
- **cs_05** `helphub/refunds/plan.py` → Nemotron 3.5 Lightning. `plan_refund` now sends the request that passed parity on the holdout split: parity vs reference 100.0% over 4 cases. Files: `helphub/cli.py`, `helphub/refunds/plan.py`. Tests updated (they pinned the old provider's request or fakes): `tests/test_call_sites.py`, `tests/test_cli.py`, `tests/test_refund_plan.py`. Request check: the code Offramp wrote, unchanged; not probed: no eval case's variables name its parameters (request). 46 tests ran in test files that name `plan_refund`: `tests/test_refund_plan.py`.
- **cs_06** `helphub/summaries/summarize.py` → Nemotron 3.5 Lightning. `summarize_thread` now sends the request that passed parity on the holdout split: judge mean 5/5 over 4 cases, lowest case 5/5. Files: `helphub/cli.py`, `helphub/summaries/summarize.py`. Tests updated (they pinned the old provider's request or fakes): `tests/fakes.py`, `tests/test_call_sites.py`, `tests/test_cli.py`, `tests/test_summarize.py`. Request check: the code Offramp wrote, unchanged; not probed: no eval case's variables name its parameters (messages). 13 tests ran in test files that name `summarize_thread`: `tests/test_summarize.py`.
- Offramp's code rewrote each migrated call (the evaluated request as written-out keyword arguments, the reply read the OpenAI way) and the callers that build its client, and that code ships as Offramp wrote it; Nemotron 3 Super edited only tests and their fakes (each site's test files are listed above) where they pinned the old provider, with repair rounds on Nemotron 3 Ultra (thinking off) when this repo's checks failed.
- The patches green on their own (cs_04, cs_05, cs_06) were then checked merged: all pass this repo's checks together.
- Each site's patch was checked with this repo's own `ruff check` and `pytest` in Docker (network off, 2 CPUs); a patch that failed them was dropped and its site marked for manual review.

## What Offramp did NOT change

- **cs_01** `helphub/inbox/triage.py` stays on gpt-4o-mini: The triage ticket function classifies support tickets but Lightning tier failed to match reference outputs on held-out cases, so the call remains on the current model. parity vs reference 85.7% < 95% on Lightning after 1 adapter iteration; cost per 1k calls $0.0543 before (est.). The triage ticket function remains on gpt-4o-mini because Lightning tier did not achieve sufficient parity; future work should refine the adapter to better capture technical_support boundaries. Adapter changes: Clarified technical_support description to explicitly include missing system emails/notifications; Added few-shot example for missing confirmation email -> technical_support; Added few-shot example for duplicate charge -> billing to reinforce boundary.
- **cs_02** `helphub/ops/agent_step.py` stays on claude-sonnet-4-5: The call site selects actions based on user input; none of the tested tiers matched reference behavior closely enough, so the current model remains. parity vs reference 50.0% < 95% on Lightning after 3 adapter iterations; cost per 1k calls $4.14 before (est.). The next_action function stays on claude-sonnet-4-5 as no tier reached acceptable parity; improving few-shot examples with clearer reasoning traces may help alignment. Adapter changes: Replaced few-shot assistant responses with natural-language reasoning that precedes tool calls, eliminating the conflicting text-only examples that taught the model to output tool names as plain text.; Kept the system prompt's explicit function-calling instruction as the primary behavioral anchor.. Rewrites never evaluated nor shipped: Ultra v2 refused before evaluation: identical to v1; no further rewrite asked on Ultra.
- **cs_03** `helphub/orders/extract.py` stays on claude-sonnet-4-5: The function extracts structured order data from emails but none of the tested tiers matched reference outputs on held‑out cases, so the current model remains. parity vs reference 71.4% < 95% on Lightning after 2 adapter iterations; cost per 1k calls $3.47 before (est.). The extract_order function keeps claude-sonnet-4-5 due to insufficient parity; ensuring strict separation between email headers and body content in prompts could improve extraction fidelity. Adapter changes: Clarified customer_email rule: only use From header when sender is the ordering customer; Clarified customer_name rule: use body/signature name, not From header unless it matches customer identity; Clarified order_date rule: order placement date, not email date; Added explicit prohibition against cross-email contamination; Kept reasoning off and max_tokens unchanged. Rewrites never evaluated nor shipped: Super v1 refused before evaluation: identical to v0; no further rewrite asked on Super.

## How parity was measured

The source models' own outputs were never recorded; parity is agreement with references.

Each site's cases are split once into dev (adapter iteration) and holdout (verdict only). Pass bar: 95% agreement with the references (90% for reasoning); free text: judge mean of at least 4.0 on a 1–5 rubric, no holdout case below 3.0. In brackets after a parity figure: holdout cases passing / evaluated (for JSON extraction, whose figure is the mean field match, the cases that pass whole), and the Wilson 95% lower bound of that pass rate; with few holdout cases, even 100% only shows, with 95% confidence, that the rate is above that bound. More holdout cases narrow it, and evaluating them costs cents on Nemotron 3.5 Lightning.
- cs_02: free-text arguments escalate_to_human.priority, escalate_to_human.reason checked for presence, not wording.

| Site | References | Cases (dev / holdout) | Reference audit (checked / agreed / dropped / human-verified, by) |
|---|---|---|---|
| cs_01 | reference:seed × 12 | 5 / 7 | 0 / 0 / 0 / 0 |
| cs_02 | reference:seed × 12 | 10 / 2 | 0 / 0 / 0 / 0 |
| cs_03 | reference:seed × 9 | 7 / 2 | 0 / 0 / 0 / 0 |
| cs_04 | — | 7 / 5 | 0 / 0 / 0 / 0 |
| cs_05 | reference:oracle × 12 | 8 / 4 | 0 / 0 / 0 / 0 |
| cs_06 | — | 8 / 4 | 0 / 0 / 0 / 0 |

Prices (USD per 1M input / output tokens):
- After: Nemotron 3.5 Lightning `nvidia/Nemotron-3_5-Lightning` $0.06 / $0.24 (pricing.yaml as of 2026-10-21; the Token Factory model API is not read in tests)
- Before, cs_04: est.: gpt-4o $2.50/$10.00 per 1M (developers.openai.com/api/docs/pricing, pricing.yaml as of 2026-10-19; fallback: offline run, Tavily not asked) x tokens measured on the Token Factory call (mean of 5 lightning holdout calls). Price by pricing.yaml fallback.
- Before, cs_05: est.: o3 $2.00/$8.00 per 1M (developers.openai.com/api/docs/pricing, pricing.yaml as of 2026-10-19; fallback: offline run, Tavily not asked) x tokens measured on the Token Factory call (mean of 4 lightning holdout calls). Price by pricing.yaml fallback.
- Before, cs_06: est.: claude-haiku-4-5 $1.00/$5.00 per 1M (platform.claude.com/docs/en/about-claude/pricing, pricing.yaml as of 2026-10-19; fallback: offline run, Tavily not asked) x tokens measured on the Token Factory call (mean of 4 lightning holdout calls). Price by pricing.yaml fallback.

<details><summary>Failing cases: cs_01 (Nemotron 3.5 Lightning, adapter v1, holdout split)</summary>

| Case | Error | Detail |
|---|---|---|
| cs_01:009 | wrong_label | expected 'general', got 'billing' |

</details>

<details><summary>Failing cases: cs_02 (Nemotron 3.5 Lightning, adapter v3, holdout split)</summary>

| Case | Error | Detail |
|---|---|---|
| cs_02:007 | wrong_tool | expected ['escalate_to_human'], got ['lookup_order'] |

</details>

<details><summary>Failing cases: cs_03 (Nemotron 3.5 Lightning, adapter v2, holdout split)</summary>

| Case | Error | Detail |
|---|---|---|
| cs_03:005 | wrong_value | field_match=0.714; wrong: customer_name, order_date |
| cs_03:009 | wrong_value | field_match=0.714; wrong: customer_name, customer_email |

</details>

Generated by Offramp · Nemotron 3 Ultra / Super / 3.5 Lightning via Nebius Token Factory · run test-demo-run
