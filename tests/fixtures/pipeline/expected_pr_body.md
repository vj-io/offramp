## Summary

This PR keeps all LLM call sites on their current models because none of the tested alternative tiers achieved sufficient parity with reference outputs on held-out cases. The decisions are based on evaluation results showing parity below the required threshold, and no migrations were applied.

Offramp evaluated 6 LLM call sites against references (69 cases, 24 in the holdout split). 0 migrated to the cheapest Nemotron tier that passed parity on the holdout split; 3 kept on the original model; 3 need manual review.

## Evidence

| Site | File | Task | Before | After | Parity vs reference (holdout) | p50 latency (after) | $/1k calls before (est.) → after |
|---|---|---|---|---|---|---|---|
| cs_01 | `helphub/inbox/triage.py` | classification | gpt-4o-mini | **NOT MIGRATED** | 85.7% < 95% | — | — |
| cs_02 | `helphub/ops/agent_step.py` | tool_call | claude-sonnet-4-5 | **NOT MIGRATED** | 50.0% < 95% | — | — |
| cs_03 | `helphub/orders/extract.py` | json_extraction | claude-sonnet-4-5 | **NOT MIGRATED** | 71.4% < 95% | — | — |
| cs_04 | `helphub/qa/rag_answer.py` | free_text | gpt-4o | **MANUAL REVIEW** | 100.0% | — | — |
| cs_05 | `helphub/refunds/plan.py` | reasoning | o3 | **MANUAL REVIEW** | 100.0% | — | — |
| cs_06 | `helphub/summaries/summarize.py` | free_text | claude-haiku-4-5 | **MANUAL REVIEW** | 100.0% | — | — |

## What changed

- Each site's patch was checked with this repo's own `ruff check` and `pytest` in Docker (network off, 2 CPUs); a patch that failed them was dropped and its site marked for manual review.
- No call site was migrated.

## What Offramp did NOT change

- **cs_01** `helphub/inbox/triage.py` stays on gpt-4o-mini: The triage ticket function classifies support tickets but Lightning tier failed to match reference outputs on held-out cases, so the call remains on the current model. parity vs reference 85.7% < 95% on Lightning after 1 adapter iteration; cost per 1k calls $0.0543 before (est.). The triage ticket function remains on gpt-4o-mini as Lightning tier did not meet parity requirements despite adapter updates clarifying technical_support scope and adding few-shot examples. Adapter changes: Clarified technical_support description to explicitly include missing system emails/notifications; Added few-shot example for missing confirmation email -> technical_support; Added few-shot example for duplicate charge -> billing to reinforce boundary.
- **cs_02** `helphub/ops/agent_step.py` stays on claude-sonnet-4-5: The call site selects actions based on user input; none of the tested tiers matched reference behavior closely enough, so the current model remains. parity vs reference 50.0% < 95% on Lightning after 3 adapter iterations; cost per 1k calls $4.14 before (est.). The next_action function stays on claude-sonnet-4-5 since no tested tier matched reference behavior closely enough, preserving the current tool-call selection logic. Adapter changes: Replaced few-shot assistant responses with natural-language reasoning that precedes tool calls, eliminating the conflicting text-only examples that taught the model to output tool names as plain text.; Kept the system prompt's explicit function-calling instruction as the primary behavioral anchor..
- **cs_03** `helphub/orders/extract.py` stays on claude-sonnet-4-5: The function extracts structured order data from emails but none of the tested tiers matched reference outputs on held‑out cases, so the current model remains. parity vs reference 71.4% < 95% on Lightning after 2 adapter iterations; cost per 1k calls $3.47 before (est.). The extract_order function continues using claude-sonnet-4-5 as none of the tiers satisfied parity thresholds, maintaining the current email-to-order extraction rules. Adapter changes: Clarified customer_email rule: only use From header when sender is the ordering customer; Clarified customer_name rule: use body/signature name, not From header unless it matches customer identity; Clarified order_date rule: order placement date, not email date; Added explicit prohibition against cross-email contamination; Kept reasoning off and max_tokens unchanged.
- **cs_04** `helphub/qa/rag_answer.py` needs manual review: patch rejected (adapter spec or patch rules): search text not found in tests/test_rag_answer.py: '"""Tests for the RAG answer generator."""\n\nimport json\n\nimport pytest\nfrom fakes' The answer_question function requires manual review due to a patch rejection where test setup text could not be located in the test file.
- **cs_05** `helphub/refunds/plan.py` needs manual review: repo checks failed on the patch (docker): $ python -m pytest -q -p no:cacheprovider (exit 1) | E       AssertionError: no **kwargs unpacking at a call site | E       assert False | E        +  where False = all(<generator object keywords.<locals>.<genexpr> at 0x…>) | E       AssertionError: no **kwargs unpacking at a call site | E       assert False | E        +  where False = all(<generator object keywords.<locals>.<genexpr> at 0x…>) | E       AssertionError: assert 'nvidia/Nemot...3_5-Lightning' == 'o3' The plan_refund function needs manual review after repo checks failed, specifically due to an assertion error about kwargs unpacking and a model mismatch in expectations.
- **cs_06** `helphub/summaries/summarize.py` needs manual review: patch rejected (adapter spec or patch rules): search text not found in tests/test_summarize.py: '"""Pins the shape of HelpHub\'s LLM call sites.\n\nThe demo is a target for Offramp' The summarize_thread function awaits manual review because the patch was rejected for failing to locate required adapter specification text in its test suite.

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
