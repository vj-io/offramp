# Token Factory by the numbers

**Scope: all 1,768 recorded calls: 1,349 from the response cache plus 419 from the committed cassettes, with
15 run directories for the run checks.**

Measured from the Token Factory calls Offramp actually made: 1,768 distinct recorded chat completions
(3,721,597 prompt tokens, 946,404 completion tokens) on the three tiers (Lightning =
`nvidia/Nemotron-3_5-Lightning`, Super = `nvidia/nemotron-3-super-120b-a12b`, Ultra =
`nvidia/Nemotron-3-Ultra-550b-a55b`), recorded between 2026-10-19 and 2026-10-22 (UTC). Nothing here is an
estimate or a re-run: every figure is computed by `scripts/tf_stats.py` from three kinds of files, and the raw
numbers are in [`tf-production.json`](tf-production.json).

- **Response cache** (`var/tf-cache`, one JSON per live call, not committed): the call's request, the raw
  response, the client-measured `latency_ms`, token usage and a timestamp.
- **Cassettes** (`tests/fixtures/cassettes`, committed): the same entry shape. 2 are hand-authored copies
  marked `"synthetic": true` (a recorded reply with its tail cut off, to test the repair retry); they are
  excluded. 6 cassettes were exact copies of another recorded call and are counted once.
- **Run files** (`var/runs/*`: `events.jsonl`, `budget.json`): no per-call data. They are used to attribute
  cache entries to pipeline runs by time window, to check the token totals, and to list errors.

A "call" is one response that Token Factory returned and Offramp recorded. A request replayed from the cache
is not counted again. Latency is the client's wall-clock time around the OpenAI SDK call (non-streaming), so
it includes the SDK's automatic retries, the one retry on the fallback key, and any queueing from Offramp's
parallel eval threads. Percentiles are nearest-rank: each is an actual recorded value. † marks a group of
fewer than 20 calls, where p95 is close to the maximum and one call moves the `length` share by 5 points or
more. A call's role is not recorded; it is read from the system prompt (the table in the script), and
`eval_target` is the migrated app's own call on the candidate tier. The per-run Ultra agent-call counts below
do not all match `budget.json`, which checks the split between agent roles and `eval_target` for the Ultra
calls.

## Reproduce

```
uv run python scripts/tf_stats.py --cache var/tf-cache --runs var/runs --format md \
  > docs/evidence/tf-production.md
uv run python scripts/tf_stats.py --cache var/tf-cache --runs var/runs \
  > docs/evidence/tf-production.json
```

`var/` is not committed, so a clean clone has no cache or run files and reproduces only the cassette part (419
calls; its scope line says so):

```
uv run python scripts/tf_stats.py --format md
```

## Where the calls come from

| Where the calls were recorded | Calls |
|---|---|
| Run `20261019-220847641-43d3df` (cache entries inside its time window) | 400 |
| Run `20261019-222942730-2c0f0b` (cache entries inside its time window) | 149 |
| Run `20261020-115200254-46f068` (cache entries inside its time window) | 15 |
| Run `20261020-122354830-57c50b` (cache entries inside its time window) | 3 |
| Run `20261020-124700775-6cb31e` (cache entries inside its time window) | 0 |
| Run `20261021-010850445-554f5f` (cache entries inside its time window) | 11 |
| Run `20261021-121928793-c6ae9e` (cache entries inside its time window) | 14 |
| Run `20261022-005412408-1240bf` (cache entries inside its time window) | 7 |
| Run `20261022-011012152-704127` (cache entries inside its time window) | 0 |
| Run `20261022-012608016-d7c85b` (cache entries inside its time window) | 0 |
| Run `20261022-014146107-57cca9` (cache entries inside its time window) | 0 |
| Run `20261022-015732734-b5aee2` (cache entries inside its time window) | 13 |
| Run `20261022-023459043-590031` (cache entries inside its time window) | 203 |
| Run `20261022-065443547-22ed36` (cache entries inside its time window) | 235 |
| Run `20261022-153359255-7bea7c` (cache entries inside its time window) | 181 |
| Response cache, not inside any recorded run window | 118 |
| Committed cassettes in `tests/fixtures/cassettes` (committed test and dev recordings, recorded by TFClient, not marked synthetic) | 419 |
| **Total** | **1,768** |

## By model

All recorded calls:

| Model | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|
| Lightning | 532 | 341,965 / 566 | 224,337 / 62 | 3.1 / 110.8 | 497 / 33 / 2 | 0.4% | 83 / 83 |
| Super | 583 | 813,222 / 617 | 275,131 / 94 | 3.1 / 12.5 | 524 / 56 / 3 | 0.5% | 197 / 197 |
| Ultra | 653 | 2,566,410 / 952 | 446,936 / 334 | 3.8 / 11.0 | 619 / 30 / 4 | 0.6% | 282 / 286 |

Only the calls inside the runs' time windows (1,231 calls):

| Model | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|
| Lightning | 403 | 246,526 / 458 | 156,518 / 61 | 3.9 / 117.4 | 393 / 10 / 0 | 0.0% | 52 / 52 |
| Super | 364 | 461,325 / 453 | 166,129 / 89 | 3.4 / 11.5 | 348 / 15 / 1 | 0.3% | 80 / 80 |
| Ultra | 464 | 2,015,798 / 834 | 317,093 / 370 | 4.2 / 10.7 | 453 / 10 / 1 | 0.2% | 184 / 185 |

Only the response cache (1,349 calls, no cassettes):

| Model | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|
| Lightning | 403 | 246,526 / 458 | 156,518 / 61 | 3.9 / 117.4 | 393 / 10 / 0 | 0.0% | 52 / 52 |
| Super | 441 | 671,569 / 599 | 226,421 / 111 | 3.4 / 12.5 | 423 / 15 / 3 | 0.7% | 139 / 139 |
| Ultra | 505 | 2,293,993 / 989 | 383,114 / 422 | 4.2 / 10.7 | 492 / 10 / 3 | 0.6% | 198 / 201 |

Only the committed cassettes (419 calls; what a clean clone reproduces):

| Model | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|
| Lightning | 129 | 95,439 / 675 | 67,819 / 62 | 1.9 / 36.6 | 104 / 23 / 2 | 1.6% | 31 / 31 |
| Super | 142 | 141,653 / 761 | 48,710 / 73 | 2.1 / 10.9 | 101 / 41 / 0 | 0.0% | 58 / 58 |
| Ultra | 148 | 272,417 / 944 | 63,822 / 90 | 2.4 / 11.7 | 127 / 20 / 1 | 0.7% | 84 / 85 |

## By model and role

| Model | Role | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|---|
| Lightning | eval_target | 532 | 341,965 / 566 | 224,337 / 62 | 3.1 / 110.8 | 497 / 33 / 2 | 0.4% | 83 / 83 |
| Super | eval_target | 396 | 247,864 / 465 | 154,214 / 87 | 3.3 / 11.3 | 340 / 56 / 0 | 0.0% | 50 / 50 |
| Super | case_gen | 57 | 68,685 / 1,030 | 28,713 / 362 | 3.6 / 14.0 | 57 / 0 / 0 | 0.0% | 57 / 57 |
| Super | patcher | 40 | 327,610 / 7,436 | 82,941 / 882 | 6.5 / 33.6 | 37 / 0 / 3 | 7.5% | n/a |
| Super | scanner | 29 | 34,088 / 986 | 1,585 / 58 | 1.4 / 4.6 | 29 / 0 / 0 | 0.0% | 29 / 29 |
| Super | verdict | 25 | 9,036 / 348 | 867 / 33 | 1.7 / 4.2 | 25 / 0 / 0 | 0.0% | 25 / 25 |
| Super | pricing_extract | 24 | 105,911 / 4,245 | 2,132 / 73 | 2.5 / 4.6 | 24 / 0 / 0 | 0.0% | 24 / 24 |
| Super | pr_writer | 12 | 20,028 / 1,798 | 4,679 / 378 | 3.9 / 7.3† | 12 / 0 / 0 | 0.0%† | 12 / 12 |
| Ultra | eval_target | 359 | 214,270 / 589 | 120,207 / 77 | 2.8 / 10.4 | 329 / 30 / 0 | 0.0% | 41 / 41 |
| Ultra | judge | 141 | 1,371,917 / 6,122 | 168,390 / 1,204 | 4.6 / 12.1 | 140 / 0 / 1 | 0.7% | 140 / 141 |
| Ultra | iterator | 57 | 409,131 / 6,779 | 55,956 / 739 | 4.2 / 10.3 | 56 / 0 / 1 | 1.8% | 56 / 57 |
| Ultra | patcher | 49 | 512,336 / 9,194 | 68,483 / 931 | 4.9 / 16.8 | 49 / 0 / 0 | 0.0% | n/a |
| Ultra | adapter | 25 | 35,987 / 1,291 | 6,759 / 265 | 2.3 / 4.4 | 25 / 0 / 0 | 0.0% | 25 / 25 |
| Ultra | case_gen | 21 | 21,937 / 937 | 25,139 / 1,026 | 4.9 / 10.3 | 19 / 0 / 2 | 9.5% | 19 / 21 |
| Ultra | reference_audit | 1 | 832 / 832 | 2,002 / 2,002 | 7.6 / 7.6† | 1 / 0 / 0 | 0.0%† | 1 / 1 |

## Thinking on and off

Latency by the request's `enable_thinking` flag, over all calls. Thinking is on for 133 calls, 132 of them
`eval_target` calls, while the thinking-off group also holds the agent roles that write long outputs (the
patcher's p95 is Super 33.6 s, Ultra 16.8 s). The two groups are therefore confounded by role, and p95 of one
is not comparable with p50 of the other.

| Model | Thinking | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|---|
| Lightning | off | 496 | 320,387 / 485 | 45,206 / 57 | 2.8 / 113.3 | 461 / 33 / 2 | 0.4% | 83 / 83 |
| Lightning | on | 36 | 21,578 / 599 | 179,131 / 4,702 | 24.8 / 62.8 | 36 / 0 / 0 | 0.0% | n/a |
| Super | off | 513 | 743,484 / 587 | 172,133 / 79 | 2.9 / 10.4 | 484 / 26 / 3 | 0.6% | 183 / 183 |
| Super | on | 70 | 69,738 / 751 | 102,998 / 417 | 6.3 / 28.8 | 40 / 30 / 0 | 0.0% | 14 / 14 |
| Ultra | off | 626 | 2,549,977 / 1,011 | 373,208 / 290 | 3.7 / 9.8 | 592 / 30 / 4 | 0.6% | 281 / 285 |
| Ultra | on | 27 | 16,433 / 600 | 73,728 / 2,556 | 11.0 / 15.8 | 27 / 0 / 0 | 0.0% | 1 / 1 |

The same split over `eval_target` calls only. The files do not say why thinking was on for a given call, so
this is still not a controlled comparison.

| Model | Thinking | Calls | Input tokens (total / p50) | Output tokens (total / p50) | Latency s (p50 / p95) | Finish: stop / tool_calls / length | `length` share | JSON valid |
|---|---|---|---|---|---|---|---|---|
| Lightning | off | 496 | 320,387 / 485 | 45,206 / 57 | 2.8 / 113.3 | 461 / 33 / 2 | 0.4% | 83 / 83 |
| Lightning | on | 36 | 21,578 / 599 | 179,131 / 4,702 | 24.8 / 62.8 | 36 / 0 / 0 | 0.0% | n/a |
| Super | off | 326 | 178,126 / 389 | 51,216 / 72 | 2.8 / 9.4 | 300 / 26 / 0 | 0.0% | 36 / 36 |
| Super | on | 70 | 69,738 / 751 | 102,998 / 417 | 6.3 / 28.8 | 40 / 30 / 0 | 0.0% | 14 / 14 |
| Ultra | off | 333 | 198,669 / 545 | 48,481 / 70 | 2.5 / 8.1 | 303 / 30 / 0 | 0.0% | 41 / 41 |
| Ultra | on | 26 | 15,601 / 600 | 71,726 / 2,556 | 11.0 / 15.8 | 26 / 0 / 0 | 0.0% | n/a |

## The `length` finishes

9 of 1,768 calls (0.5%) ended on `finish_reason: length`: 2 Lightning eval_target, 3 Super patcher, 2 Ultra
case_gen, 1 Ultra iterator, 1 Ultra judge. 0 of them had thinking on (the reasoning-budget case in
`docs/FEEDBACK.md` item 2); 2 used `tool_choice="required"` (item 4, Lightning looping on tool calls until the
cap); the other 7 ran to the cap with thinking off. The files record no cause. Where the reply had to be JSON,
a `length` finish leaves it truncated and invalid (next section).

| Model | Role | Source | Thinking | max_tokens | Output tokens | tool_choice | response_format |
|---|---|---|---|---|---|---|---|
| Super | patcher | cache | off | 8,000 | 8,000 | none | none |
| Super | patcher | cache | off | 8,000 | 8,000 | none | none |
| Ultra | case_gen | cache | off | 4,000 | 4,000 | none | json_schema |
| Ultra | judge | cache | off | 2,500 | 2,500 | none | json_schema |
| Ultra | case_gen | cache | off | 4,000 | 4,000 | none | json_schema |
| Super | patcher | cache | off | 12,000 | 12,000 | none | none |
| Lightning | eval_target | cassette | off | 512 | 512 | required | none |
| Lightning | eval_target | cassette | off | 512 | 512 | required | none |
| Ultra | iterator | cassette | off | 4,000 | 4,000 | none | json_schema |

## Validity, retries, errors, caching

- **JSON validity.** 562 of 566 calls that asked for `json_schema` or `json_object` returned JSON that parses
  and, for `json_schema`, satisfies the schema in the request (checked here with `jsonschema`; a `json_object`
  call is checked only for parsing as an object, because its schema is not in the request). The invalid
  replies: 4 with finish_reason `length`.
- **Repair retries.** 3 calls were repair retries ("Your previous reply is not valid"): 2 answered a synthetic
  reply (the excluded hand-cut ones, so not provider failures) and 1 a real reply that failed validation.
- **Judge replies deleted before this count.** 5 judge retries with twice the judge's `max_tokens` for their
  batch have no first reply recorded before them (`20261022-153359255-7bea7c`: 5). The judge sends such a
  retry only after the first reply to the same batch did not match the rubric schema, and Offramp deleted that
  first reply from the cache (where the same request was asked again later, live, the cache holds that later
  reply), so the replies that did not parse are in none of the counts here. (Offramp now keeps a first reply
  when the retry succeeds, and deletes both when it fails.)
- **Run errors.** Runs that ended FAILED: `20261019-220847641-43d3df` at ITERATING: ProviderError: judge: left
  1 case(s) unscored: cs_04:020. Warning events: `20261019-220847641-43d3df`: cs_04 super dev: ProviderError:
  judge: left 1 case(s) unscored: cs_04:020; `20261019-220847641-43d3df`: cs_04 ultra dev: ProviderError:
  judge: left 1 case(s) unscored: cs_04:020; `20261019-220847641-43d3df`: cs_04: lightning dev: judge: left 1
  case(s) unscored: cs_04:020; retrying once; `20261022-005412408-1240bf`: patch: removed 2 patch(es) of
  `helphub.cli.Anthropic`, which Offramp's rewrite removed; `20261022-011012152-704127`: patch: removed 2
  patch(es) of `helphub.cli.Anthropic`, which Offramp's rewrite removed; `20261022-012608016-d7c85b`: patch:
  removed 2 patch(es) of `helphub.cli.Anthropic`, which Offramp's rewrite removed;
  `20261022-014146107-57cca9`: patch: removed 2 patch(es) of `helphub.cli.Anthropic`, which Offramp's rewrite
  removed; `20261022-015732734-b5aee2`: patch: removed 2 patch(es) of `helphub.cli.Anthropic`, which Offramp's
  rewrite removed; `20261022-065443547-22ed36`: cs_05 lightning dev: ProviderError: judge: reply does not
  match the rubric schema; `20261022-153359255-7bea7c`: cs_01: Lightning not iterated: dev median 64.3 s is
  more than twice the call's 30 s timeout; `20261022-153359255-7bea7c`: cs_02: Lightning not iterated: dev
  median 76.7 s is more than twice the call's 30 s timeout; `20261022-153359255-7bea7c`: cs_04: Lightning not
  iterated: dev median 71.0 s is more than twice the call's 30 s timeout; `20261022-153359255-7bea7c`: cs_05:
  Lightning not iterated: dev median 97.5 s is more than twice the call's 30 s timeout. That failure is
  Offramp's own `ProviderError`: a judge reply that left a case unscored, not an HTTP error from Token Factory
  (`docs/FEEDBACK.md` item 11).
- **Not recorded.** HTTP errors, 429s, timeouts, the SDK's automatic retries (up to 3 per call) and the one
  retry on the fallback key: the client caches only successful responses, so a failed request leaves no file.
  The provider error rate is unknown here, not zero.
- **Provider prompt-cache hits** (`prompt_cache_hit_tokens` in the response usage): Lightning 0 of 341,965
  (0.0%); Super 912 of 813,222 (0.1%); Ultra 409,088 of 2,566,410 (15.9%).

## Runs against their budgets

For each run, the cache entries inside its time window, summed, against what `budget.json` charged (tokens in,
tokens out, Ultra agent calls). Not all runs match; see the `no` rows. A run that replays cached calls charges
only its new ones, so later runs show few calls or none. `budget.json` also records spend: $3.51 across the 15
runs.

| Run | Calls | Tokens in (cache / budget) | Tokens out (cache / budget) | Ultra agent calls (cache / budget) | Budget spend | Match |
|---|---|---|---|---|---|---|
| `20261019-220847641-43d3df` | 400 | 355,556 / 355,556 | 237,923 / 237,923 | 26 / 26 | $0.50 | yes |
| `20261019-222942730-2c0f0b` | 149 | 333,721 / 333,721 | 106,577 / 106,577 | 30 / 30 | $0.33 | yes |
| `20261020-115200254-46f068` | 15 | 92,392 / 92,392 | 8,498 / 8,498 | 4 / 4 | $0.08 | yes |
| `20261020-122354830-57c50b` | 3 | 24,171 / 24,171 | 4,032 / 4,032 | 2 / 2 | $0.03 | yes |
| `20261020-124700775-6cb31e` | 0 | 0 / 0 | 0 / 0 | 0 / 0 | $0.00 | yes |
| `20261021-010850445-554f5f` | 11 | 87,938 / 87,938 | 4,759 / 4,759 | 5 / 5 | $0.08 | yes |
| `20261021-121928793-c6ae9e` | 14 | 103,806 / 103,806 | 8,751 / 8,751 | 9 / 9 | $0.12 | yes |
| `20261022-005412408-1240bf` | 7 | 28,326 / 28,326 | 1,395 / 1,395 | 1 / 1 | $0.02 | yes |
| `20261022-011012152-704127` | 0 | 0 / 0 | 0 / 0 | 0 / 0 | $0.00 | yes |
| `20261022-012608016-d7c85b` | 0 | 0 / 0 | 0 / 0 | 0 / 0 | $0.00 | yes |
| `20261022-014146107-57cca9` | 0 | 0 / 0 | 0 / 0 | 0 / 0 | $0.00 | yes |
| `20261022-015732734-b5aee2` | 13 | 13,904 / 13,904 | 2,647 / 2,647 | 2 / 2 | $0.02 | yes |
| `20261022-023459043-590031` | 203 | 468,490 / 498,099 | 80,632 / 93,186 | 40 / 40 | $0.64 | **no** |
| `20261022-065443547-22ed36` | 235 | 479,940 / 479,940 | 93,654 / 93,654 | 40 / 40 | $0.61 | yes |
| `20261022-153359255-7bea7c` | 181 | 735,405 / 869,711 | 90,872 / 104,572 | 34 / 39 | $1.09 | **no** |

## What this does and doesn't show

It shows that the three Nemotron tiers on Token Factory returned 1,768 responses to Offramp's own calls
between 2026-10-19 and 2026-10-22: 1,640 `stop` finishes, 119 `tool_calls` (the normal end of a tool-use call)
and 9 `length`, and 562 of 566 structured-output replies were valid JSON. On `eval_target` calls alone, p50
latency with thinking off against on is Lightning 2.8 s vs 24.8 s, Super 2.8 s vs 6.3 s, Ultra 2.5 s vs 11.0
s, and p95 is Lightning 113.3 s vs 62.8 s, Super 9.4 s vs 28.8 s, Ultra 8.1 s vs 15.8 s; those are different
calls, not a controlled experiment. It does not show uptime or an error rate (failed requests are not
recorded), throughput or behaviour under sustained load (the calls came from Offramp's pipeline and tests in
bursts, with at most 8 eval calls in flight), time to first token (nothing was streamed), or cost per call
(`budget.json` has run totals only). All of it comes from the call sites of the few repos Offramp was pointed
at (a demo app and, in the cache since 2026-10-22, one open-source repo), the cassettes are committed test and
dev recordings rather than customer traffic, the roles are inferred from system prompts, and latency was
measured on the client side only.
