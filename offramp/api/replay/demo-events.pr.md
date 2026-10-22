## Summary

**Claude Sonnet 4.5 retires on 30 Nov 2026: cs_02, cs_03.**

Offramp evaluated 6 LLM call sites against references (121 cases, 40 in the holdout split). 5 migrated to the cheapest Nemotron tier that passed parity on the holdout split; 0 kept on the original model; 1 needs manual review.

## Evidence

| Site | File | Task | Before | After | Parity vs reference (holdout) | p50 latency (after) | $/1k calls before (est.) → after |
|---|---|---|---|---|---|---|---|
| cs_02 | `helphub/ops/agent_step.py` | tool_call | claude-sonnet-4-5 (retires 30 Nov 2026) | Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) | 100.0% (5/5, ≥56%) | 2.7 s | $3.59 (est.) → $0.33 |
| cs_03 | `helphub/orders/extract.py` | json_extraction | claude-sonnet-4-5 (retires 30 Nov 2026) | Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) | 100.0% (3/3 pass, ≥43%) (after 1 adapter iteration) | 2.4 s | $3.17 (est.) → $0.28 |
| cs_01 | `helphub/inbox/triage.py` | classification | gpt-4o-mini | **MANUAL REVIEW** | 100.0% (9/9, ≥70%) | — | — |
| cs_04 | `helphub/qa/rag_answer.py` | free_text | gpt-4o | Nemotron 3 Ultra (`nvidia/Nemotron-3-Ultra-550b-a55b`) | 100.0% (9/9, ≥70%), judge mean 5/5, lowest case 5/5 (after 1 adapter iteration) | 3.0 s | $2.10 (est.) → $0.77 |
| cs_05 | `helphub/refunds/plan.py` | reasoning | o3 | Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | 100.0% (6/6, ≥60%) | 19.4 s | $39.51 (est.) → $1.19 |
| cs_06 | `helphub/summaries/summarize.py` | free_text | claude-haiku-4-5 | Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | 75.0% (6/8, ≥40%), judge mean 4.54/5, lowest case 3/5 (after 1 adapter iteration) | 3.4 s | $0.84 (est.) → $0.046 |

## What changed

- `helphub/offramp_client.py` added inside the `helphub` package, so it ships with it: OpenAI SDK client for Nebius Token Factory (`token_factory_client()`, reads `NEBIUS_API_KEY` and optional `NEBIUS_BASE_URL`) and the Nemotron tier constants `LIGHTNING`, `SUPER`, `ULTRA`. Every migrated call sends `chat_template_kwargs.enable_thinking` explicitly, since Nemotron reasons by default.
- `next_action`, `extract_order`, `answer_question`, `plan_refund`, `summarize_thread` get the client from callers: callers were checked statically; lookups through reflection (getattr, importlib, sys.modules, vars/__dict__) were not, except the literal getattr/import_module forms Offramp refuses.
- Imports that Offramp's or the model's changes left unused were removed: `Anthropic` from `helphub/cli.py`; `Anthropic` from `helphub/ops/agent_step.py`; `Anthropic` from `helphub/orders/extract.py`; `Anthropic` from `helphub/summaries/summarize.py`.
- Test patches of those names were removed: 2 of `helphub.cli.Anthropic`.
- **cs_02** `helphub/ops/agent_step.py` → Nemotron 3 Super. `next_action` now sends the request that passed parity on the holdout split: parity vs reference 100.0% over 5 cases. Files: `helphub/cli.py`, `helphub/ops/agent_step.py`. Tests updated (they pinned the old provider's request or fakes): `tests/fakes.py`, `tests/test_agent_step.py`, `tests/test_call_sites.py`, `tests/test_cli.py`.
- **cs_03** `helphub/orders/extract.py` → Nemotron 3 Super. `extract_order` now sends the request that passed parity on the holdout split: parity vs reference 100.0% over 3 cases. Adapter changes: Added explicit instruction to set currency to null when no currency symbol or code appears in the email; Added a worked example showing correct null currency extraction from a German email with only numeric amounts. Files: `helphub/cli.py`, `helphub/orders/extract.py`. Tests updated (they pinned the old provider's request or fakes): `tests/fakes.py`, `tests/test_call_sites.py`, `tests/test_cli.py`, `tests/test_extract.py`.
- **cs_04** `helphub/qa/rag_answer.py` → Nemotron 3 Ultra. `answer_question` now sends the request that passed parity on the holdout split: judge mean 5/5 over 9 cases, lowest case 5/5. Adapter changes: Rewrote system prompt to enforce immediate inline citation after every fact and explicit missing-information handoff; Added three few-shot examples demonstrating inline citation, missing-information handoff, and multi-fact citation; Kept reasoning off and max_tokens unchanged. Files: `helphub/cli.py`, `helphub/qa/rag_answer.py`. Tests updated (they pinned the old provider's request or fakes): `tests/test_call_sites.py`, `tests/test_cli.py`, `tests/test_rag_answer.py`.
- **cs_05** `helphub/refunds/plan.py` → Nemotron 3.5 Lightning. `plan_refund` now sends the request that passed parity on the holdout split: parity vs reference 100.0% over 6 cases. Files: `helphub/cli.py`, `helphub/refunds/plan.py`. Tests updated (they pinned the old provider's request or fakes): `tests/test_call_sites.py`, `tests/test_cli.py`, `tests/test_refund_plan.py`.
- **cs_06** `helphub/summaries/summarize.py` → Nemotron 3.5 Lightning. `summarize_thread` now sends the request that passed parity on the holdout split: judge mean 4.54/5 over 8 cases, lowest case 3/5. Adapter changes: Rewrote system prompt to explicitly forbid hallucination and require strict adherence to transcript-only facts; Added three few-shot examples covering empty thread, minimal thread, and multi-message thread with internal notes; Enforced exact output format: three labelled lines, no markdown, no extra text; Kept reasoning off to prevent leakage on Lightning tier; Maintained max_tokens at 400. Files: `helphub/cli.py`, `helphub/summaries/summarize.py`. Tests updated (they pinned the old provider's request or fakes): `tests/fakes.py`, `tests/test_call_sites.py`, `tests/test_cli.py`, `tests/test_summarize.py`.
- Offramp's code rewrote each migrated call (the evaluated request as written-out keyword arguments, the reply read the OpenAI way) and the callers that build its client, and that code ships as Offramp wrote it; Nemotron 3 Super edited only tests and their fakes (each site's test files are listed above) where they pinned the old provider, with repair rounds on Nemotron 3 Ultra (thinking off) when this repo's checks failed.
- The patches green on their own (cs_02, cs_03, cs_04, cs_05, cs_06) were then checked merged: all pass this repo's checks together.
- Each site's patch was checked with this repo's own `ruff check` and `pytest` in Docker (network off, 2 CPUs); a patch that failed them was dropped and its site marked for manual review.

## What Offramp did NOT change

- **cs_01** `helphub/inbox/triage.py` needs manual review: The model's edit to `tests/test_triage.py` quotes lines matching no single place there, so none applied.

  <details><summary>Raw failure</summary>

  ```text
  patch rejected (patch rules): search text not found in tests/test_triage.py: '@pytest.mark.parametrize("ticket", TICKETS, ids=case_ids(TICKETS))\ndef test_ever'; applied by name, this edit would drop `assert triage_ticket(client, ticket["subject"], ticket["body"]) is Queue.SHIPPING`, `assert ticket["subject"] in user_message`, `assert ticket["body"] in user_message`, which your SEARCH never quoted: keep them, or quote them to change them
  ```

  </details>


## How parity was measured

The source models' own outputs were never recorded; parity is agreement with references.
Reference outputs for cs_04, cs_06 were produced by `claude-sonnet-5.5 (via Claude Code; not the repo's source model)`.

Each site's cases are split once into dev (adapter iteration) and holdout (verdict only). Pass bar: 95% agreement with the references (90% for reasoning); free text: judge mean of at least 4.0 on a 1–5 rubric, no holdout case below 3.0, and at least the references' judge mean − 0.3 where references hold text (cs_04, cs_06). In brackets after a parity figure: holdout cases passing / evaluated (for JSON extraction, whose figure is the mean field match, the cases that pass whole), and the Wilson 95% lower bound of that pass rate; with few holdout cases, even 100% only shows, with 95% confidence, that the rate is above that bound. More holdout cases narrow it, and evaluating them costs cents on Nemotron 3.5 Lightning.
- cs_02: free-text arguments escalate_to_human.priority, escalate_to_human.reason checked for presence, not wording.
- cs_06: reference judged weak: cs_06:020, 1/5 (below 3; kept in the evaluation).

| Site | References | Cases (dev / holdout) | Reference audit (checked / agreed / dropped / human-verified, by) |
|---|---|---|---|
| cs_01 | reference:seed × 21 | 12 / 9 | 24 / 21 / 3 / 0 by claude-sonnet-5.5 (via Claude Code; not the repo's source model) |
| cs_02 | reference:seed × 15 | 10 / 5 | 20 / 15 / 5 / 0 by claude-sonnet-5.5 (via Claude Code; not the repo's source model) |
| cs_03 | reference:seed × 16 | 13 / 3 | 19 / 16 / 3 / 0 by claude-sonnet-5.5 (via Claude Code; not the repo's source model) |
| cs_04 | claude-sonnet-5.5 (via Claude Code; not the repo's source model) × 24 | 15 / 9 | 0 / 0 / 0 / 0 |
| cs_05 | reference:oracle × 24 | 18 / 6 | 0 / 0 / 0 / 0 |
| cs_06 | claude-sonnet-5.5 (via Claude Code; not the repo's source model) × 21 | 13 / 8 | 0 / 0 / 0 / 0 |

Prices (USD per 1M input / output tokens):
- After: Nemotron 3.5 Lightning `nvidia/Nemotron-3_5-Lightning` $0.06 / $0.24 (Token Factory model API, 2026-10-22)
- After: Nemotron 3 Super `nvidia/nemotron-3-super-120b-a12b` $0.30 / $0.90 (Token Factory model API, 2026-10-22)
- After: Nemotron 3 Ultra `nvidia/Nemotron-3-Ultra-550b-a55b` $1.00 / $3.00 (Token Factory model API, 2026-10-22)
- Before, cs_02: est.: claude-sonnet-4-5 $3.00/$15.00 per 1M (platform.claude.com/docs/en/models/sonnet-4-5/overview, fetched via Tavily 2026-10-22) x tokens measured on the Token Factory call (mean of 5 super holdout calls). Price by Tavily search; the page prints `Input pricing : $3/ MTok Output pricing : $15`.
- Before, cs_03: est.: claude-sonnet-4-5 $3.00/$15.00 per 1M (platform.claude.com/docs/en/models/sonnet-4-5/overview, fetched via Tavily 2026-10-22) x tokens measured on the Token Factory call (mean of 3 super holdout calls). Price by Tavily search; the page prints `Input pricing : $3/ MTok Output pricing : $15`.
- Before, cs_04: est.: gpt-4o $2.50/$10.00 per 1M (developers.openai.com/api/docs/models/gpt-4o, confirmed via Tavily Extract 2026-10-22, as in pricing.yaml) x tokens measured on the Token Factory call (mean of 9 ultra holdout calls). Price by Tavily Extract, confirming pricing.yaml; the page prints `Input $2.50 Cached input $1.25 Output $10.00`.
- Before, cs_05: est.: o3 $2.00/$8.00 per 1M (developers.openai.com/api/docs/models/o3, confirmed via Tavily Extract 2026-10-22, as in pricing.yaml) x tokens measured on the Token Factory call (mean of 6 lightning holdout calls). Price by Tavily Extract, confirming pricing.yaml; the page prints `Input $2.00 Cached input $0.50 Output $8.00`.
- Before, cs_06: est.: claude-haiku-4-5 $1.00/$5.00 per 1M (platform.claude.com/docs/en/models/haiku-4-5, confirmed via Tavily Extract 2026-10-22, as in pricing.yaml) x tokens measured on the Token Factory call (mean of 8 lightning holdout calls). Price by Tavily Extract, confirming pricing.yaml; the page prints `Input pricing : $1/ MTok Output pricing : $5`.

Sunset risk (source-model deprecation notices):
- cs_02: Claude Sonnet 4.5 is deprecated: it is still functional but no longer recommended, and retires on November 30, 2026. (https://platform.claude.com/docs/en/models/sonnet-4-5/overview)
- cs_03: Claude Sonnet 4.5 is deprecated: it is still functional but no longer recommended, and retires on November 30, 2026. (https://platform.claude.com/docs/en/models/sonnet-4-5/overview)

LangSmith (private: members of the LangSmith workspace only): [cs_01](https://smith.langchain.com/o/3467d815-f9b6-4132-943b-93cb51ddcda2/datasets/cb1e7bd5-34b9-429a-a336-b4fea25cb420) · [cs_02](https://smith.langchain.com/o/3467d815-f9b6-4132-943b-93cb51ddcda2/datasets/e26543f1-4017-4653-9df7-c3dece7bf3a5) · [cs_03](https://smith.langchain.com/o/3467d815-f9b6-4132-943b-93cb51ddcda2/datasets/752abbb0-8eff-455d-b1b4-b6c0cc46d065) · [cs_04](https://smith.langchain.com/o/3467d815-f9b6-4132-943b-93cb51ddcda2/datasets/0cfb3485-650e-4b5f-a15e-ce51683cef37) · [cs_05](https://smith.langchain.com/o/3467d815-f9b6-4132-943b-93cb51ddcda2/datasets/bbe1f125-8c54-4989-a68b-0c1f7aa852e3) · [cs_06](https://smith.langchain.com/o/3467d815-f9b6-4132-943b-93cb51ddcda2/datasets/8e031a8e-9b38-495d-9c9d-a20b0740449e)

<details><summary>Failing cases: cs_06 (Nemotron 3.5 Lightning, adapter v1, holdout split)</summary>

| Case | Error | Detail |
|---|---|---|
| cs_06:015 | low_judge_score | judge 3.67 vs reference 5.00; candidate is A: Answer A is in Spanish, violating English requirement; Answer B is in English and accurate. |
| cs_06:020 | low_judge_score | judge 3.00 vs reference 1.00; candidate is A: Both answers invent facts not in evidence (customer names, amounts, plan details) and miss the actual thread content. |

</details>

Generated by Offramp · Nemotron 3 Ultra / Super / 3.5 Lightning via Nebius Token Factory · run 20261022-014146107-57cca9
