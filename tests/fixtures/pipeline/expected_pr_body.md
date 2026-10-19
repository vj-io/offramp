## Summary

This PR updates the answer_question function to use the Nemotron 3.5 Lightning model after verifying parity with reference outputs on held-out cases. All other call sites remain on their current models due to insufficient parity or manual-review requirements. The changes include updated model calls, adjusted test expectations, and modified files to reflect the migration while preserving behavior where applicable.

Offramp evaluated 6 LLM call sites against references (69 cases, 24 in the holdout split). 1 migrated to the cheapest Nemotron tier that passed parity on the holdout split; 3 kept on the original model; 2 need manual review.

## Evidence

| Site | File | Task | Before | After | Parity vs reference (holdout) | p50 latency (after) | $/1k calls before (est.) → after |
|---|---|---|---|---|---|---|---|
| cs_01 | `helphub/inbox/triage.py` | classification | gpt-4o-mini | **NOT MIGRATED** | 85.7% < 95% | — | — |
| cs_02 | `helphub/ops/agent_step.py` | tool_call | claude-sonnet-4-5 | **NOT MIGRATED** | 50.0% < 95% | — | — |
| cs_03 | `helphub/orders/extract.py` | json_extraction | claude-sonnet-4-5 | **NOT MIGRATED** | 71.4% < 95% | — | — |
| cs_04 | `helphub/qa/rag_answer.py` | free_text | gpt-4o | Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | 100.0%, judge mean 5/5 | 1.0 s | $2.04 (est.) → $0.049 |
| cs_05 | `helphub/refunds/plan.py` | reasoning | o3 | **MANUAL REVIEW** | 100.0% | — | — |
| cs_06 | `helphub/summaries/summarize.py` | free_text | claude-haiku-4-5 | **MANUAL REVIEW** | 100.0% | — | — |

## What changed

- `helphub/offramp_client.py` added inside the `helphub` package, so it ships with it: OpenAI SDK client for Nebius Token Factory (`token_factory_client()`, reads `NEBIUS_API_KEY` and optional `NEBIUS_BASE_URL`) and the Nemotron tier constants `LIGHTNING`, `SUPER`, `ULTRA`. Every migrated call sends `chat_template_kwargs.enable_thinking` explicitly, since Nemotron reasons by default.
- **cs_04** `helphub/qa/rag_answer.py` → Nemotron 3.5 Lightning. The answer_question function now uses Nemotron 3.5 Lightning after achieving perfect parity on held-out cases, with no adapter changes needed and corresponding test updates to reflect the new model output. Files: `helphub/cli.py`, `helphub/qa/rag_answer.py`. Tests updated (they pinned the old provider's request or fakes): `tests/test_call_sites.py`, `tests/test_cli.py`, `tests/test_rag_answer.py`.
- Each site's patch was checked with this repo's own `ruff check` and `pytest` in Docker (network off, 2 CPUs); a patch that failed them was dropped and its site marked for manual review.

## What Offramp did NOT change

- **cs_01** `helphub/inbox/triage.py` stays on gpt-4o-mini: The triage ticket function classifies support tickets but Lightning tier failed to match reference outputs on held-out cases, so the call remains on the current model. parity vs reference 85.7% < 95% on Lightning after 1 adapter iteration; cost per 1k calls $0.0543 before (est.). The triage ticket function stays on gpt-4o-mini because Lightning tier did not meet the required parity threshold on held-out cases, so no model change was made and the function continues to classify tickets as before. Adapter changes: Clarified technical_support description to explicitly include missing system emails/notifications; Added few-shot example for missing confirmation email -> technical_support; Added few-shot example for duplicate charge -> billing to reinforce boundary.
- **cs_02** `helphub/ops/agent_step.py` stays on claude-sonnet-4-5: The call site selects actions based on user input; none of the tested tiers matched reference behavior closely enough, so the current model remains. parity vs reference 50.0% < 95% on Lightning after 3 adapter iterations; cost per 1k calls $4.14 before (est.). The next_action function remains on claude-sonnet-4-5 as none of the tested tiers achieved sufficient alignment with reference behavior, preserving the current tool-selection logic. Adapter changes: Replaced few-shot assistant responses with natural-language reasoning that precedes tool calls, eliminating the conflicting text-only examples that taught the model to output tool names as plain text.; Kept the system prompt's explicit function-calling instruction as the primary behavioral anchor..
- **cs_03** `helphub/orders/extract.py` stays on claude-sonnet-4-5: The function extracts structured order data from emails but none of the tested tiers matched reference outputs on held‑out cases, so the current model remains. parity vs reference 71.4% < 95% on Lightning after 2 adapter iterations; cost per 1k calls $3.47 before (est.). The extract_order function keeps using claude-sonnet-4-5 since no tested tier reached the parity benchmark on held-out data, maintaining the existing email-to-order extraction process. Adapter changes: Clarified customer_email rule: only use From header when sender is the ordering customer; Clarified customer_name rule: use body/signature name, not From header unless it matches customer identity; Clarified order_date rule: order placement date, not email date; Added explicit prohibition against cross-email contamination; Kept reasoning off and max_tokens unchanged.
- **cs_05** `helphub/refunds/plan.py` needs manual review: patch rejected (adapter spec or patch rules): search text not found in helphub/refunds/plan.py: 'def plan_refund(request: RefundRequest) -> RefundPlan:\n    from helphub.offramp_' The plan_refund function requires manual review because the patch failed to apply due to missing expected text in the source file, so a contributor must verify the function's structure and intent before proceeding.
- **cs_06** `helphub/summaries/summarize.py` needs manual review: repair produced no change; last failure: repo checks failed on the patch (docker): E         + Mention internal notes only as "internal note"; their wording is for agents, never the customer. | E       assert 'You write ha...the customer.' == 'Issue: x.' | E          | E         - Issue: x. | E         + You write handoff notes for HelpHub support agents taking over a ticket. | E         + Summarise the thread in English, in at most 120 words, with three labelled parts: | E         + Issue, Tried so far, Next step. | E         + Quote order ids and amounts exactly as written. Skip greetings and pleasantries. The summarize_thread function needs manual inspection as the patch did not modify the file and automated checks failed, indicating a mismatch between the expected and actual summarization behavior that a reviewer must resolve.

## How parity was measured

The source models' own outputs were never recorded; parity is agreement with references.

Each site's cases are split once into dev (adapter iteration) and holdout (verdict only). Pass bar: 95% agreement with the references (90% for reasoning); free text: judge mean of at least 4.0 on a 1–5 rubric.
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
- After: Nemotron 3.5 Lightning `nvidia/Nemotron-3_5-Lightning` $0.06 / $0.24 (https://nebius.com/token-factory/prices, as of 2026-10-19, unverified list price)
- Before, cs_04: est.: gpt-4o $2.50/$10.00 per 1M (developers.openai.com/api/docs/pricing, pricing.yaml as of 2026-10-19; fallback: offline run, Tavily not asked) x tokens measured on the Token Factory call (mean of 5 lightning holdout calls)

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
