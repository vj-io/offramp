## Summary

Offramp evaluated 6 LLM call sites against references (121 cases, 40 in the holdout split). 3 migrated to the cheapest Nemotron tier that passed parity on the holdout split; 0 kept on the original model; 3 need manual review.

## Evidence

| Site | File | Task | Before | After | Parity vs reference (holdout) | p50 latency (after) | $/1k calls before (est.) → after |
|---|---|---|---|---|---|---|---|
| cs_01 | `helphub/inbox/triage.py` | classification | gpt-4o-mini | **MANUAL REVIEW** | 100.0% | — | — |
| cs_02 | `helphub/ops/agent_step.py` | tool_call | claude-sonnet-4-5 (sunset risk) | Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) | 100.0% | 2.7 s | $3.59 (est.) → $0.33 |
| cs_03 | `helphub/orders/extract.py` | json_extraction | claude-sonnet-4-5 (sunset risk) | **MANUAL REVIEW** | 100.0% | — | — |
| cs_04 | `helphub/qa/rag_answer.py` | free_text | gpt-4o | Nemotron 3 Ultra (`nvidia/Nemotron-3-Ultra-550b-a55b`) | 100.0%, judge mean 5/5 (after 1 adapter iteration) | 3.0 s | $2.10 (est.) → $0.77 |
| cs_05 | `helphub/refunds/plan.py` | reasoning | o3 | **MANUAL REVIEW** | 100.0% | — | — |
| cs_06 | `helphub/summaries/summarize.py` | free_text | claude-haiku-4-5 | Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | 75.0%, judge mean 4.54/5 (after 1 adapter iteration) | 3.4 s | $0.84 (est.) → $0.046 |

## What changed

- `helphub/offramp_client.py` added inside the `helphub` package, so it ships with it: OpenAI SDK client for Nebius Token Factory (`token_factory_client()`, reads `NEBIUS_API_KEY` and optional `NEBIUS_BASE_URL`) and the Nemotron tier constants `LIGHTNING`, `SUPER`, `ULTRA`. Every migrated call sends `chat_template_kwargs.enable_thinking` explicitly, since Nemotron reasons by default.
- **cs_02** `helphub/ops/agent_step.py` → Nemotron 3 Super. The next_action function now uses Nemotron 3 Super with no adapter changes, as the Super tier matched reference behavior perfectly on the holdout set, ensuring the tool selection logic remains unchanged while reducing cost. Files: `helphub/cli.py`, `helphub/ops/agent_step.py`. Tests updated (they pinned the old provider's request or fakes): `tests/fakes.py`, `tests/test_agent_step.py`, `tests/test_call_sites.py`, `tests/test_cli.py`.
- **cs_04** `helphub/qa/rag_answer.py` → Nemotron 3 Ultra. The answer_question function switches to Nemotron 3 Ultra with updated prompting and few-shot examples to enforce inline citations and handoffs, preserving the RAG behavior while meeting reference quality on the holdout set. Adapter changes: Rewrote system prompt to enforce immediate inline citation after every fact and explicit missing-information handoff; Added three few-shot examples demonstrating inline citation, missing-information handoff, and multi-fact citation; Kept reasoning off and max_tokens unchanged. Files: `helphub/cli.py`, `helphub/qa/rag_answer.py`. Tests updated (they pinned the old provider's request or fakes): `tests/test_call_sites.py`, `tests/test_cli.py`, `tests/test_rag_answer.py`.
- **cs_06** `helphub/summaries/summarize.py` → Nemotron 3.5 Lightning. Adapter changes: Rewrote system prompt to explicitly forbid hallucination and require strict adherence to transcript-only facts; Added three few-shot examples covering empty thread, minimal thread, and multi-message thread with internal notes; Enforced exact output format: three labelled lines, no markdown, no extra text; Kept reasoning off to prevent leakage on Lightning tier; Maintained max_tokens at 400. Files: `helphub/cli.py`, `helphub/summaries/summarize.py`. Tests updated (they pinned the old provider's request or fakes): `tests/fakes.py`, `tests/test_call_sites.py`, `tests/test_cli.py`, `tests/test_summarize.py`.
- Offramp's code rewrote each migrated call (the evaluated request as written-out keyword arguments, the reply read the OpenAI way) and the callers that build its client, and that code ships as Offramp wrote it; Nemotron 3 Super edited only tests and their fakes (each site's test files are listed above) where they pinned the old provider, with repair rounds on Nemotron 3 Ultra (thinking off) when this repo's checks failed.
- The patches green on their own (cs_02, cs_04, cs_06) were then checked merged: all pass this repo's checks together.
- Each site's patch was checked with this repo's own `ruff check` and `pytest` in Docker (network off, 2 CPUs); a patch that failed them was dropped and its site marked for manual review.

## What Offramp did NOT change

- **cs_01** `helphub/inbox/triage.py` needs manual review: the reply holds no complete edit block; last failure: patch rejected (patch rules): search text not found in tests/test_triage.py: 'def test_request_contract():\n    client = FakeOpenAI("returns_refunds")\n\n    ass' The triage ticket classifier stays on gpt-4o-mini because the patch could not be applied due to missing search text in the test file; a reviewer must verify whether the intended few-shot additions can be manually integrated without breaking existing tests.
- **cs_03** `helphub/orders/extract.py` needs manual review: the reply holds no complete edit block; last failure: repo checks failed on the patch (docker): FAILED tests/test_extract.py::test_request_contract - pydantic_core._pydantic... | FAILED tests/test_extract.py::test_joins_multiple_text_blocks - pydantic_core... | FAILED tests/test_extract.py::test_every_fixture_email_is_sent_and_parsed[…] (13 cases) | E       pydantic_core._pydantic_core.ValidationError: 1 validation error for ChatCompletion | E       choices.0.message.content | E         Input should be a valid string [type=string_type, input_value=TextBlock(citations=None,...damaged"}', type='text'), input_type=TextBlock] | E           For further information visit https://errors.pydantic.dev/2.14/v/string_type | E       pydantic_core._pydantic_core.ValidationError: 1 validation error for ChatCompletion | E       choices.0.message.content The order extractor remains on claude-sonnet-4-5 due to persistent Pydantic validation errors in the extracted output; a reviewer must confirm that the added currency null instruction and German email example resolve the schema mismatches in the test suite.
- **cs_05** `helphub/refunds/plan.py` needs manual review: repair produced no change; last failure: patch rejected (patch rules): patch removes tests from tests/test_call_sites.py: ['test_openai_reasoning_site_follows_the_o3_parameter_rules'] The refund planner stays on o3 because the patch was rejected for removing a test; a reviewer should assess whether the parameter rules for reasoning can be preserved without deleting test coverage.

## How parity was measured

The source models' own outputs were never recorded; parity is agreement with references.
Reference outputs for cs_04, cs_06 were produced by `claude-sonnet-5.5 (via Claude Code; not the repo's source model)`.

Each site's cases are split once into dev (adapter iteration) and holdout (verdict only). Pass bar: 95% agreement with the references (90% for reasoning); free text: judge mean of at least 4.0 on a 1–5 rubric, and at least the references' judge mean − 0.3 where references hold text (cs_04, cs_06).
- cs_02: free-text arguments escalate_to_human.priority, escalate_to_human.reason checked for presence, not wording.

| Site | References | Cases (dev / holdout) | Reference audit (checked / agreed / dropped / human-verified, by) |
|---|---|---|---|
| cs_01 | reference:seed × 21 | 12 / 9 | 24 / 21 / 3 / 0 by claude-sonnet-5.5 (via Claude Code; not the repo's source model) |
| cs_02 | reference:seed × 15 | 10 / 5 | 20 / 15 / 5 / 0 by claude-sonnet-5.5 (via Claude Code; not the repo's source model) |
| cs_03 | reference:seed × 16 | 13 / 3 | 19 / 16 / 3 / 0 by claude-sonnet-5.5 (via Claude Code; not the repo's source model) |
| cs_04 | claude-sonnet-5.5 (via Claude Code; not the repo's source model) × 24 | 15 / 9 | 0 / 0 / 0 / 0 |
| cs_05 | reference:oracle × 24 | 18 / 6 | 0 / 0 / 0 / 0 |
| cs_06 | claude-sonnet-5.5 (via Claude Code; not the repo's source model) × 21 | 13 / 8 | 0 / 0 / 0 / 0 |

Prices (USD per 1M input / output tokens):
- After: Nemotron 3.5 Lightning `nvidia/Nemotron-3_5-Lightning` $0.06 / $0.24 (https://nebius.com/token-factory/prices, as of 2026-10-19, unverified list price)
- After: Nemotron 3 Super `nvidia/nemotron-3-super-120b-a12b` $0.30 / $0.90 (https://nebius.com/token-factory/prices, as of 2026-10-19, unverified list price)
- After: Nemotron 3 Ultra `nvidia/Nemotron-3-Ultra-550b-a55b` $1.00 / $3.00 (https://nebius.com/token-factory/prices, as of 2026-10-19, unverified list price)
- Before, cs_02: est.: claude-sonnet-4-5 $3.00/$15.00 per 1M (platform.claude.com/docs/en/models/sonnet-4-5, fetched via Tavily 2026-10-20) x tokens measured on the Token Factory call (mean of 5 super holdout calls)
- Before, cs_04: est.: gpt-4o $2.50/$10.00 per 1M (developers.openai.com/api/docs/pricing, pricing.yaml as of 2026-10-19; fallback: Tavily price not used (the cited page prints no unambiguous price for exactly this model)) x tokens measured on the Token Factory call (mean of 9 ultra holdout calls)
- Before, cs_06: est.: claude-haiku-4-5 $1.00/$5.00 per 1M (platform.claude.com/docs/zh-CN/models/haiku-4-5/overview, fetched via Tavily 2026-10-20) x tokens measured on the Token Factory call (mean of 8 lightning holdout calls)

Sunset risk (source-model deprecation notices):
- cs_02: Claude Sonnet 4.5 is deprecated: it is still functional but no longer recommended, and retires on November 30, 2026. (https://platform.claude.com/docs/en/models/sonnet-4-5)
- cs_03: Claude Sonnet 4.5 is deprecated: it is still functional but no longer recommended, and retires on November 30, 2026. (https://platform.claude.com/docs/en/models/sonnet-4-5)

LangSmith: [cs_01](https://smith.langchain.com/public/cd050a17-d5ea-40b2-8254-5b00e11f8ead/d) · [cs_02](https://smith.langchain.com/public/60cb6589-57b6-41bf-b678-cd9a43d49ed7/d) · [cs_03](https://smith.langchain.com/public/52a0b181-8e12-428a-b970-c283fcfe88ec/d) · [cs_04](https://smith.langchain.com/public/7ec2ae0d-a7dd-47e8-92be-bcb708dfb91f/d) · [cs_05](https://smith.langchain.com/public/b010a6a2-6964-4a89-80da-bd24fc56de14/d) · [cs_06](https://smith.langchain.com/public/903dda55-0791-4780-81c7-48a72d60cba6/d)

<details><summary>Failing cases: cs_06 (Nemotron 3.5 Lightning, adapter v1, holdout split)</summary>

| Case | Error | Detail |
|---|---|---|
| cs_06:015 | low_judge_score | judge 3.67 vs reference 5.00; candidate is A: Answer A is in Spanish, violating English requirement; Answer B is in English and accurate. |
| cs_06:020 | low_judge_score | judge 3.00 vs reference 1.00; candidate is A: Both answers invent facts not in evidence (customer names, amounts, plan details) and miss the actual thread content. |

</details>

Generated by Offramp · Nemotron 3 Ultra / Super / 3.5 Lightning via Nebius Token Factory · run 20261020-124700775-6cb31e
