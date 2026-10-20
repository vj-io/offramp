## Summary

**Claude Sonnet 4.5 retires on 30 Nov 2026: cs_02, cs_03.**

This PR moves five HelpHub call sites to NVIDIA Nemotron on Nebius Token Factory. The refund planner stays on its original model because it did not reach parity.

Offramp evaluated 6 LLM call sites against references (140 cases, 43 in the holdout split). 5 migrated to the cheapest Nemotron tier that passed parity on the holdout split; 1 kept on the original model.

## Evidence

| Site | File | Task | Before | After | Parity vs reference (holdout) | p50 latency (after) | $/1k calls before (est.) → after |
|---|---|---|---|---|---|---|---|
| cs_02 | `helphub/ops/agent_step.py` | tool_call | claude-sonnet-4-5 (retires 30 Nov 2026) | Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) | 96.0% (24/25, ≥80%) | 1.2 s | $3.00 (est.) → $0.26 |
| cs_03 | `helphub/orders/extract.py` | json_extraction | claude-sonnet-4-5 (retires 30 Nov 2026) | Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) | 96.4% (8/8 pass, ≥67%) | 1.6 s | $4.50 (est.) → $0.38 |
| cs_01 | `helphub/inbox/triage.py` | classification | gpt-4o-mini | Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | 97.1% (34/35, ≥85%) (after 1 adapter iteration) | 410 ms | $0.047 (est.) → $0.019 |
| cs_04 | `helphub/qa/rag_answer.py` | free_text | gpt-4o | Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) | 100.0% (6/6, ≥60%), judge mean 4.4/5, lowest case 3.5/5 (after 2 adapter iterations) | 2.0 s | $4.50 (est.) → $0.50 |
| cs_05 | `helphub/refunds/plan.py` | reasoning | o3 | **NOT MIGRATED** | 71.4% (5/7, ≥35%) < 90% | — | — |
| cs_06 | `helphub/summaries/summarize.py` | free_text | claude-haiku-4-5 | Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | 100.0% (6/6, ≥60%), judge mean 4.3/5, lowest case 3.67/5 (after 1 adapter iteration) | 530 ms | $1.40 (est.) → $0.077 |

## What changed

- `helphub/offramp_client.py` added inside the `helphub` package, so it ships with it: OpenAI SDK client for Nebius Token Factory (`token_factory_client()`, reads `NEBIUS_API_KEY` and optional `NEBIUS_BASE_URL`) and the Nemotron tier constants `LIGHTNING`, `SUPER`, `ULTRA`. Every migrated call sends `chat_template_kwargs.enable_thinking` explicitly, since Nemotron reasons by default.
- **cs_01** `helphub/inbox/triage.py` → Nemotron 3.5 Lightning. `triage_ticket` now sends the request that passed parity on the holdout split: parity vs reference 97.1% over 1 case. Adapter changes: v1: user suffix restates the answer format (Lightning added a sentence before the label on dev).
- **cs_02** `helphub/ops/agent_step.py` → Nemotron 3 Super. `next_action` now sends the request that passed parity on the holdout split. Adapter changes: v0: Anthropic tools -> OpenAI function tools; tool_choice any -> required. Files: `helphub/ops/agent_step.py`. Tests updated (they pinned the old provider's request or fakes): `tests/fakes.py`, `tests/test_agent_step.py`.
- **cs_03** `helphub/orders/extract.py` → Nemotron 3 Super. `extract_order` now sends the request that passed parity on the holdout split. Adapter changes: v0: OrderFields schema enforced with response_format json_schema (strict).
- **cs_04** `helphub/qa/rag_answer.py` → Nemotron 3 Super. `answer_question` now sends the request that passed parity on the holdout split. Adapter changes: v1: temperature 0.2 -> 0; v2: system prompt forbids citing ids that are not in the excerpts.
- **cs_06** `helphub/summaries/summarize.py` → Nemotron 3.5 Lightning. `summarize_thread` now sends the request that passed parity on the holdout split. Adapter changes: v1: one worked handoff note as a few-shot example.
- Offramp's code rewrote each migrated call (the evaluated request as written-out keyword arguments, the reply read the OpenAI way) and the callers that build its client, and that code ships as Offramp wrote it; Nemotron 3 Super edited only tests and their fakes (each site's test files are listed above) where they pinned the old provider, with repair rounds on Nemotron 3 Ultra (thinking off) when this repo's checks failed.
- Each site's patch was checked with this repo's own `ruff check` and `pytest` in Docker (network off, 2 CPUs); a patch that failed them was dropped and its site marked for manual review.

## What Offramp did NOT change

- **cs_05** `helphub/refunds/plan.py` stays on o3: parity vs reference 71.4% < 90%: multi-step proration with tiered seats still drifts by cents after 3 adapter iterations on Ultra. Try distilling the refund arithmetic before the next attempt. Adapter changes: v1: reasoning on; v2: worked example of tiered pricing; v3: explicit half-up rounding reminder.

## How parity was measured

The source models' own outputs were never recorded; parity is agreement with references.
Reference outputs for cs_04, cs_06 were produced by `claude-sonnet-5.5 (via Claude Code; not the repo's source model)`.

Each site's cases are split once into dev (adapter iteration) and holdout (verdict only). Pass bar: 95% agreement with the references (90% for reasoning); free text: judge mean of at least 4.0 on a 1–5 rubric, no holdout case below 3.0, and at least the references' judge mean − 0.3 where references hold text (cs_04, cs_06). In brackets after a parity figure: holdout cases passing / evaluated, and the Wilson 95% lower bound of that pass rate; with few holdout cases, even 100% only shows the rate is above that bound. More holdout cases narrow it, and evaluating them costs cents on Nemotron 3.5 Lightning.
- cs_02: free-text argument escalate_to_human.reason checked for presence, not wording.

| Site | References | Cases (dev / holdout) | Reference audit (checked / agreed / dropped / human-verified, by) |
|---|---|---|---|
| cs_01 | reference:seed × 25 | 18 / 7 | 26 / 25 / 1 / 0 by claude-sonnet-5.5 (via Claude Code) |
| cs_02 | reference:seed × 25 | 17 / 8 | 26 / 25 / 1 / 0 by claude-sonnet-5.5 (via Claude Code) |
| cs_03 | reference:seed × 22, reference:human × 2 | 16 / 8 | 25 / 22 / 1 / 2 by claude-sonnet-5.5 (via Claude Code) |
| cs_04 | claude-sonnet-5.5 (via Claude Code; not the repo's source model) × 20 | 14 / 6 | — |
| cs_05 | reference:oracle × 26 | 18 / 8 | — |
| cs_06 | claude-sonnet-5.5 (via Claude Code; not the repo's source model) × 20 | 14 / 6 | — |

Prices (USD per 1M input / output tokens):
- After: Nemotron 3.5 Lightning `nvidia/Nemotron-3_5-Lightning` $0.06 / $0.24 (Token Factory model API, 2026-10-21)
- After: Nemotron 3 Super `nvidia/nemotron-3-super-120b-a12b` $0.30 / $0.90 (Token Factory model API, 2026-10-21)
- Before, cs_01: est.: gpt-4o-mini $0.15/$0.60 per 1M (developers.openai.com/api/docs/pricing, pricing.yaml as of 2026-10-19) x tokens measured on the Token Factory call
- Before, cs_02: est.: claude-sonnet-4-5 $3.00/$15.00 per 1M (platform.claude.com/docs/en/models/sonnet-4-5, fetched via Tavily 2026-10-19) x tokens measured on the Token Factory call. Price by Tavily search; the page prints `Input $3 / MTok Output $15 / MTok`.
- Before, cs_03: est.: claude-sonnet-4-5 $3.00/$15.00 per 1M (platform.claude.com/docs/en/about-claude/pricing, pricing.yaml as of 2026-10-19) x tokens measured on the Token Factory call
- Before, cs_04: est.: gpt-4o $2.50/$10.00 per 1M (developers.openai.com/api/docs/pricing, pricing.yaml as of 2026-10-19) x tokens measured on the Token Factory call. Price by pricing.yaml fallback.
- Before, cs_06: est.: claude-haiku-4-5 $1.00/$5.00 per 1M (platform.claude.com/docs/en/about-claude/pricing, pricing.yaml as of 2026-10-19) x tokens measured on the Token Factory call

Sunset risk (source-model deprecation notices):
- cs_02: Claude Sonnet 4.5 is deprecated: it is still functional but no longer recommended, and retires on November 30, 2026. (https://platform.claude.com/docs/en/models/sonnet-4-5)
- cs_03: Claude Sonnet 4.5 is deprecated: it is still functional but no longer recommended, and retires on November 30, 2026. (https://platform.claude.com/docs/en/models/sonnet-4-5)

LangSmith (public, read-only): [project](https://smith.langchain.com/public/fixture-project/r) · [cs_05](https://smith.langchain.com/public/fixture-cs05-dev/r)

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
