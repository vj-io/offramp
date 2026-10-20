# Token Factory by the numbers

Measured from the Token Factory calls Offramp actually made: 1,072 distinct recorded chat completions
(1,762,031 prompt tokens, 645,520 completion tokens) on the three tiers (Lightning =
`nvidia/Nemotron-3_5-Lightning`, Super = `nvidia/nemotron-3-super-120b-a12b`, Ultra =
`nvidia/Nemotron-3-Ultra-550b-a55b`), recorded between 2026-10-19 and 2026-10-20 (UTC). Nothing here is an
estimate or a re-run: every figure is computed by `scripts/tf_stats.py` from three kinds of files, and the raw
numbers are in [`tf-production.json`](tf-production.json).

- **Response cache** (`var/tf-cache`, one JSON per live call, not committed): the call's request, the raw
  response, the client-measured `latency_ms`, token usage and a timestamp.
- **Cassettes** (`tests/fixtures/cassettes`, committed): the same entry shape. 2 are hand-authored copies
  marked `"synthetic": true` (a recorded reply with its tail cut off, to test the repair retry); they are
  excluded. 5 cassettes were exact copies of another recorded call and are counted once.
- **Run files** (`var/runs/*`: `events.jsonl`, `budget.json`): no per-call data. They are used to attribute
  cache entries to pipeline runs by time window, to check the token totals, and to list errors.

A "call" is one response that Token Factory returned and Offramp recorded. A request replayed from the cache
is not counted again. Latency is the client's wall-clock time around the OpenAI SDK call (non-streaming), so
it includes the SDK's automatic retries and any queueing from Offramp's parallel eval threads. Percentiles are
nearest-rank: each is an actual recorded value. † marks a group of fewer than 20 calls, where p95 is close to
the maximum. A call's role is not recorded; it is read from the system prompt (the table in the script), and
`eval_target` is the migrated app's own call on the candidate tier. The per-run Ultra agent-call counts below
match `budget.json`, which checks the split between agent roles and `eval_target` for the Ultra calls.

## Where the calls come from

| Where the calls were recorded | Calls |
|---|---|
| Run `20261019-220847641-43d3df` (cache entries inside its time window) | 400 |
| Run `20261019-222942730-2c0f0b` (cache entries inside its time window) | 149 |
| Run `20261020-115200254-46f068` (cache entries inside its time window) | 15 |
| Run `20261020-122354830-57c50b` (cache entries inside its time window) | 3 |
| Run `20261020-124700775-6cb31e` (cache entries inside its time window) | 0 |
| Response cache, outside any run window (development and test calls) | 88 |
| Committed cassettes in `tests/fixtures/cassettes` (recorded by TFClient, not marked synthetic) | 417 |
| **Total** | **1,072** |

## By model

All recorded calls:

| Model | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|
| Lightning | 372 | 235,779 / 597 | 196,729 / 63 | 1.8 / 24.8 | 337 / 33 / 2 | 0.5% | 83 / 83 |
| Super | 352 | 611,295 / 731 | 215,556 / 79 | 2.5 / 16.0 | 293 / 56 / 3 | 0.9% | 143 / 143 |
| Ultra | 348 | 914,957 / 897 | 233,235 / 185 | 3.0 / 11.3 | 315 / 30 / 3 | 0.9% | 160 / 163 |

Only the calls inside the runs' time windows (567 calls):

| Model | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|
| Lightning | 243 | 140,340 / 554 | 128,910 / 64 | 1.8 / 22.5 | 233 / 10 / 0 | 0.0% | 52 / 52 |
| Super | 157 | 281,654 / 599 | 119,535 / 74 | 2.2 / 21.3 | 141 / 15 / 1 | 0.6% | 50 / 50 |
| Ultra | 167 | 383,846 / 602 | 108,585 / 107 | 3.0 / 11.0 | 157 / 10 / 0 | 0.0% | 69 / 69 |

Only the response cache (655 calls, no cassettes):

| Model | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|
| Lightning | 243 | 140,340 / 554 | 128,910 / 64 | 1.8 / 22.5 | 233 / 10 / 0 | 0.0% | 52 / 52 |
| Super | 211 | 471,440 / 669 | 167,224 / 88 | 2.8 / 21.3 | 193 / 15 / 3 | 1.4% | 86 / 86 |
| Ultra | 201 | 655,805 / 866 | 169,530 / 336 | 3.6 / 11.0 | 189 / 10 / 2 | 1.0% | 76 / 78 |

## By model and role

| Model | Role | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|---|
| Lightning | eval_target | 372 | 235,779 / 597 | 196,729 / 63 | 1.8 / 24.8 | 337 / 33 / 2 | 0.5% | 83 / 83 |
| Super | eval_target | 219 | 142,413 / 599 | 111,855 / 75 | 2.1 / 18.5 | 163 / 56 / 0 | 0.0% | 50 / 50 |
| Super | patcher | 40 | 327,610 / 7,436 | 82,941 / 882 | 6.5 / 33.6 | 37 / 0 / 3 | 7.5% | n/a |
| Super | case_gen | 39 | 54,543 / 1,215 | 16,397 / 341 | 3.3 / 9.5 | 39 / 0 / 0 | 0.0% | 39 / 39 |
| Super | scanner | 24 | 27,772 / 936 | 1,298 / 58 | 1.4 / 3.8 | 24 / 0 / 0 | 0.0% | 24 / 24 |
| Super | verdict | 17 | 5,975 / 348 | 596 / 34 | 1.5 / 7.9† | 17 / 0 / 0 | 0.0% | 17 / 17 |
| Super | pricing_extract | 9 | 46,103 / 5,048 | 829 / 73 | 2.3 / 4.6† | 9 / 0 / 0 | 0.0% | 9 / 9 |
| Super | pr_writer | 4 | 6,879 / 1,796 | 1,640 / 400 | 3.5 / 7.3† | 4 / 0 / 0 | 0.0% | 4 / 4 |
| Ultra | eval_target | 193 | 106,785 / 575 | 81,329 / 72 | 2.0 / 12.3 | 163 / 30 / 0 | 0.0% | 41 / 41 |
| Ultra | judge | 47 | 202,117 / 5,861 | 35,350 / 916 | 3.3 / 6.0 | 47 / 0 / 0 | 0.0% | 47 / 47 |
| Ultra | iterator | 42 | 220,890 / 4,702 | 33,610 / 653 | 3.6 / 5.3 | 41 / 0 / 1 | 2.4% | 41 / 42 |
| Ultra | patcher | 33 | 342,092 / 9,194 | 55,957 / 1,061 | 5.2 / 28.3 | 33 / 0 / 0 | 0.0% | n/a |
| Ultra | adapter | 18 | 26,540 / 1,134 | 4,924 / 265 | 2.1 / 5.6† | 18 / 0 / 0 | 0.0% | 18 / 18 |
| Ultra | case_gen | 14 | 15,701 / 937 | 20,063 / 1,026 | 4.9 / 11.8† | 12 / 0 / 2 | 14.3% | 12 / 14 |
| Ultra | reference_audit | 1 | 832 / 832 | 2,002 / 2,002 | 7.6 / 7.6† | 1 / 0 / 0 | 0.0% | 1 / 1 |

## Thinking on and off

Latency by the request's `enable_thinking` flag.

| Model | Thinking | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|---|
| Lightning | off | 336 | 214,201 / 567 | 17,598 / 57 | 1.6 / 4.5 | 301 / 33 / 2 | 0.6% | 83 / 83 |
| Lightning | on | 36 | 21,578 / 599 | 179,131 / 4,702 | 24.8 / 62.8 | 36 / 0 / 0 | 0.0% | n/a |
| Super | off | 282 | 541,557 / 723 | 112,558 / 71 | 2.0 / 10.6 | 253 / 26 / 3 | 1.1% | 129 / 129 |
| Super | on | 70 | 69,738 / 751 | 102,998 / 417 | 6.3 / 28.8 | 40 / 30 / 0 | 0.0% | 14 / 14 |
| Ultra | off | 321 | 898,524 / 952 | 159,507 / 97 | 2.7 / 7.3 | 288 / 30 / 3 | 0.9% | 159 / 162 |
| Ultra | on | 27 | 16,433 / 600 | 73,728 / 2,556 | 11.0 / 15.8 | 27 / 0 / 0 | 0.0% | 1 / 1 |

## The `length` finishes

8 of 1,072 calls (0.7%) ended on `finish_reason: length`: 2 Lightning eval_target, 3 Super patcher, 2 Ultra
case_gen, 1 Ultra iterator. 0 of them had thinking on (the reasoning-budget case in `docs/FEEDBACK.md` item
2); 2 used `tool_choice="required"` (item 4, Lightning looping on tool calls until the cap); the other 6 ran
to the cap with thinking off. The files record no cause. Where the reply had to be JSON, a `length` finish
leaves it truncated and invalid (next section).

| Model | Role | Thinking | max_tokens | Output tokens | tool_choice | response_format |
|---|---|---|---|---|---|---|
| Super | patcher | off | 8,000 | 8,000 | none | none |
| Super | patcher | off | 8,000 | 8,000 | none | none |
| Ultra | case_gen | off | 4,000 | 4,000 | none | json_schema |
| Ultra | case_gen | off | 4,000 | 4,000 | none | json_schema |
| Super | patcher | off | 12,000 | 12,000 | none | none |
| Lightning | eval_target | off | 512 | 512 | required | none |
| Lightning | eval_target | off | 512 | 512 | required | none |
| Ultra | iterator | off | 4,000 | 4,000 | none | json_schema |

## Validity, retries, errors, caching

- **JSON validity.** 386 of 389 calls that asked for `json_schema` or `json_object` returned JSON that parses
  and, for `json_schema`, satisfies the schema in the request (checked here with `jsonschema`; a `json_object`
  call is checked only for parsing as an object, because its schema is not in the request). The invalid
  replies: 3 with finish_reason `length`.
- **Repair retries.** 3 calls were repair retries ("Your previous reply is not valid"): 2 answered a synthetic
  reply (the excluded hand-cut ones, so not provider failures) and 1 a real reply that failed validation.
- **Run errors.** Runs that ended FAILED: `20261019-220847641-43d3df` at ITERATING: ProviderError: judge: left
  1 case(s) unscored: cs_04:020. Warning events: `20261019-220847641-43d3df`: cs_04 super dev: ProviderError:
  judge: left 1 case(s) unscored: cs_04:020; `20261019-220847641-43d3df`: cs_04 ultra dev: ProviderError:
  judge: left 1 case(s) unscored: cs_04:020; `20261019-220847641-43d3df`: cs_04: lightning dev: judge: left 1
  case(s) unscored: cs_04:020; retrying once.
- **Not recorded.** HTTP errors, 429s, timeouts and the SDK's automatic retries (up to 3 per call): the client
  caches only successful responses, so a failed request leaves no file. The provider error rate is unknown
  here, not zero.
- **Provider prompt-cache hits** (`prompt_cache_hit_tokens` in the response usage): Lightning 0 of 235,779
  (0.0%); Super 0 of 611,295 (0.0%); Ultra 143,616 of 914,957 (15.7%).

## Runs against their budgets

For each run, the cache entries inside its time window, summed, against what `budget.json` charged (tokens in,
tokens out, Ultra agent calls). All runs match. A run that replays cached calls charges only its new ones, so
later runs show few calls or none. `budget.json` also records spend: $0.94 across the 5 runs.

| Run | Calls | Tokens in (cache / budget) | Tokens out (cache / budget) | Ultra agent calls (cache / budget) | Budget spend | Match |
|---|---|---|---|---|---|---|
| `20261019-220847641-43d3df` | 400 | 355,556 / 355,556 | 237,923 / 237,923 | 26 / 26 | $0.50 | yes |
| `20261019-222942730-2c0f0b` | 149 | 333,721 / 333,721 | 106,577 / 106,577 | 30 / 30 | $0.33 | yes |
| `20261020-115200254-46f068` | 15 | 92,392 / 92,392 | 8,498 / 8,498 | 4 / 4 | $0.08 | yes |
| `20261020-122354830-57c50b` | 3 | 24,171 / 24,171 | 4,032 / 4,032 | 2 / 2 | $0.03 | yes |
| `20261020-124700775-6cb31e` | 0 | 0 / 0 | 0 / 0 | 0 / 0 | $0.00 | yes |

## What this does and doesn't show

It shows that the three Nemotron tiers on Token Factory returned 1,072 responses to Offramp's own calls
between 2026-10-19 and 2026-10-20: 945 `stop` finishes, 119 `tool_calls` (the normal end of a tool-use call)
and 8 `length`, and 386 of 389 structured-output replies were valid JSON. It shows where the latency tail
sits: with thinking off, p95 latency is Lightning 4.5 s, Super 10.6 s, Ultra 7.3 s; with thinking on, p50
latency is Lightning 24.8 s, Super 6.3 s, Ultra 11.0 s. It does not show uptime or an error rate (failed
requests are not recorded), throughput or behaviour under sustained load (the calls came from Offramp's
pipeline and tests in bursts, with at most 8 eval calls in flight), time to first token (nothing was
streamed), or cost per call (`budget.json` has run totals only). All of it comes from one demo app's call
sites, the cassettes are development and test recordings rather than customer traffic, the roles are inferred
from system prompts, and latency was measured on the client side only.
