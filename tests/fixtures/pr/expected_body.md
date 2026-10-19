## Summary

This PR moves five HelpHub call sites to NVIDIA Nemotron on Nebius Token Factory. The refund planner stays on its original model because it did not reach parity.

Offramp evaluated 6 LLM call sites against references (140 cases, 43 in the holdout split). 5 migrated to the cheapest Nemotron tier that passed parity on the holdout split; 1 kept on the original model.

## Evidence

| Site | File | Task | Before | After | Parity vs reference (holdout) | p50 latency (after) | $/1k calls before (est.) → after |
|---|---|---|---|---|---|---|---|
| cs_01 | `helphub/inbox/triage.py` | classification | gpt-4o-mini | Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | 97.1% (after 1 adapter iteration) | 410 ms | $0.047 (est.) → $0.019 |
| cs_02 | `helphub/ops/agent_step.py` | tool_call | claude-sonnet-4-5 (sunset risk) | Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) | 96.0% | 1.2 s | $3.00 (est.) → $0.26 |
| cs_03 | `helphub/orders/extract.py` | json_extraction | claude-sonnet-4-5 (sunset risk) | Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) | 96.4% | 1.6 s | $4.50 (est.) → $0.38 |
| cs_04 | `helphub/qa/rag_answer.py` | free_text | gpt-4o | Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) | 100.0%, judge mean 4.4/5 (after 2 adapter iterations) | 2.0 s | $4.50 (est.) → $0.50 |
| cs_05 | `helphub/refunds/plan.py` | reasoning | o3 | **NOT MIGRATED** | 71.4% < 90% | — | — |
| cs_06 | `helphub/summaries/summarize.py` | free_text | claude-haiku-4-5 | Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | 100.0%, judge mean 4.3/5 (after 1 adapter iteration) | 530 ms | $1.40 (est.) → $0.077 |

## What changed

- `offramp/client.py` added: OpenAI SDK client for Nebius Token Factory (`token_factory_client()`, reads `NEBIUS_API_KEY` and optional `NEBIUS_BASE_URL`) and the Nemotron tier constants `LIGHTNING`, `SUPER`, `ULTRA`. Every migrated call sends `chat_template_kwargs.enable_thinking` explicitly, since Nemotron reasons by default.
- **cs_01** `helphub/inbox/triage.py` → Nemotron 3.5 Lightning. Adapter changes: v1: user suffix restates the answer format (Lightning added a sentence before the label on dev).
- **cs_02** `helphub/ops/agent_step.py` → Nemotron 3 Super. Anthropic tool_use blocks are now read from OpenAI tool_calls. Adapter changes: v0: Anthropic tools -> OpenAI function tools; tool_choice any -> required. Files: `helphub/ops/agent_step.py`, `tests/test_agent_step.py`.
- **cs_03** `helphub/orders/extract.py` → Nemotron 3 Super. Adapter changes: v0: OrderFields schema enforced with response_format json_schema (strict).
- **cs_04** `helphub/qa/rag_answer.py` → Nemotron 3 Super. Adapter changes: v1: temperature 0.2 -> 0; v2: system prompt forbids citing ids that are not in the excerpts.
- **cs_06** `helphub/summaries/summarize.py` → Nemotron 3.5 Lightning. Adapter changes: v1: one worked handoff note as a few-shot example.
- Each site's patch was checked with this repo's own `ruff check` and `pytest` in Docker (network off, 2 CPUs); a patch that failed them was dropped and its site marked for manual review.

## What Offramp did NOT change

- **cs_05** `helphub/refunds/plan.py` stays on o3: parity vs reference 71.4% < 90%: multi-step proration with tiered seats still drifts by cents after 3 adapter iterations on Ultra. Try distilling the refund arithmetic before the next attempt. Adapter changes: v1: reasoning on; v2: worked example of tiered pricing; v3: explicit half-up rounding reminder.

## How parity was measured

The source models' own outputs were never recorded; parity is agreement with references.
Reference outputs for cs_04, cs_06 were produced by `claude-sonnet-5.5 (via Claude Code; not the repo's source model)`.

Each site's cases are split once into dev (adapter iteration) and holdout (verdict only). Pass bar: 95% agreement with the references (90% for reasoning); free text: judge mean of at least 4.0 on a 1–5 rubric.
- cs_02: free-text argument escalate_to_human.reason checked for presence, not wording.

| Site | References | Cases (dev / holdout) | Reference audit (checked / agreed / dropped / human-verified) |
|---|---|---|---|
| cs_01 | reference:seed × 25 | 18 / 7 | 26 / 25 / 1 / 0 |
| cs_02 | reference:seed × 25 | 17 / 8 | 26 / 25 / 1 / 0 |
| cs_03 | reference:seed × 22, reference:human × 2 | 16 / 8 | 25 / 22 / 1 / 2 |
| cs_04 | claude-sonnet-5.5 (via Claude Code; not the repo's source model) × 20 | 14 / 6 | — |
| cs_05 | reference:oracle × 26 | 18 / 8 | — |
| cs_06 | claude-sonnet-5.5 (via Claude Code; not the repo's source model) × 20 | 14 / 6 | — |

Prices (USD per 1M input / output tokens):
- After: Nemotron 3.5 Lightning `nvidia/Nemotron-3_5-Lightning` $0.06 / $0.24 (https://nebius.com/token-factory/prices, as of 2026-10-19, unverified list price)
- After: Nemotron 3 Super `nvidia/nemotron-3-super-120b-a12b` $0.30 / $0.90 (https://nebius.com/token-factory/prices, as of 2026-10-19, unverified list price)
- Before, cs_01: est.: gpt-4o-mini $0.15/$0.60 per 1M (developers.openai.com/api/docs/pricing, pricing.yaml as of 2026-10-19) x tokens measured on the Token Factory call
- Before, cs_02: est.: claude-sonnet-4-5 $3.00/$15.00 per 1M (platform.claude.com/docs/en/about-claude/pricing, pricing.yaml as of 2026-10-19) x tokens measured on the Token Factory call
- Before, cs_03: est.: claude-sonnet-4-5 $3.00/$15.00 per 1M (platform.claude.com/docs/en/about-claude/pricing, pricing.yaml as of 2026-10-19) x tokens measured on the Token Factory call
- Before, cs_04: est.: gpt-4o $2.50/$10.00 per 1M (developers.openai.com/api/docs/pricing, pricing.yaml as of 2026-10-19) x tokens measured on the Token Factory call
- Before, cs_06: est.: claude-haiku-4-5 $1.00/$5.00 per 1M (platform.claude.com/docs/en/about-claude/pricing, pricing.yaml as of 2026-10-19) x tokens measured on the Token Factory call

Sunset risk (source-model deprecation notices):
- cs_02: claude-sonnet-4-5 is listed as deprecated on platform.claude.com/docs/en/about-claude/pricing
- cs_03: claude-sonnet-4-5 is listed as deprecated on platform.claude.com/docs/en/about-claude/pricing

LangSmith: [project](https://smith.langchain.com/public/fixture-project/r) · [cs_05](https://smith.langchain.com/public/fixture-cs05-dev/r)

<details><summary>Failing cases: cs_01 (Nemotron 3.5 Lightning, adapter v1, holdout split)</summary>

| Case | Error | Detail |
|---|---|---|
| cs_01:017 | wrong_label | expected billing, got general |

</details>

<details><summary>Failing cases: cs_05 (Nemotron 3 Ultra, adapter v3, dev split)</summary>

| Case | Error | Detail |
|---|---|---|
| cs_05:004 | wrong_answer | expected 150.03, got 148.20 |
| cs_05:011 | wrong_answer | expected 41250, got 41251 |

</details>

Generated by Offramp · Nemotron 3 Ultra / Super / 3.5 Lightning via Nebius Token Factory · run fixture-run
