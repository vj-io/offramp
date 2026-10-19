# Offramp — Implementation Plan

**Spec (binding authority):** `docs/PRD.md` (Offramp PRD v2). Section numbers below (§N) refer to it.
**Contracts:** `offramp/schemas.py` — every component imports shapes from there; never redefine them.

Repo layout (two git repos, siblings on disk):

```
nvidia/offramp/            this repo — Python package `offramp/`, Next.js `web/`, tests, docs
nvidia/offramp-demo-app/   separate repo — the realistic target app Offramp migrates (§11 Day 1)
```

## Global Constraints

1. **Python 3.12, uv.** `uv sync` installs; `uv run pytest`, `uv run ruff check .` must pass. Package code lives
   under `offramp/` (`offramp/tf`, `offramp/agents`, `offramp/evals`, `offramp/api`). Tests under
   `tests/{unit,agents,harness,integration,e2e}`. Ruff config in `pyproject.toml` is fixed.
2. **Dependencies:** allowed runtime deps are those in `pyproject.toml` (openai, pydantic, sqlmodel, fastapi,
   uvicorn, httpx, pyyaml, jsonschema). `anthropic` only in the baseline recorder (optional extra).
   `langsmith` optional (import guarded). Adding any other dependency: append it to `dependencies` and say so in
   your report. Stdlib over new deps.
3. **Token Factory only.** All Nemotron calls go through `offramp.tf.client.TFClient` (Task 1) — OpenAI SDK
   pointed at `NEBIUS_BASE_URL` (default `https://api.tokenfactory.nebius.com/v1`) with `NEBIUS_API_KEY`.
   Never OpenRouter/HF. Agents never construct `openai.OpenAI` themselves.
4. **Tiers** (`offramp.schemas.Tier`): `lightning` → `super` → `ultra` (cheapest first). Model ids are config
   (`offramp/tf/tiers.py`, env overrides `OFFRAMP_MODEL_{LIGHTNING,SUPER,ULTRA}`). Defaults, to be verified
   against `GET /v1/models` in Task 14: lightning `nvidia/Nemotron-3_5-Lightning`, super
   `nvidia/nemotron-3-super-120b-a12b`, ultra `nvidia/Nemotron-3-Ultra-550b-a55b`.
5. **Agent tier allocation (§6.11):** Super — scanner labelling, case gen (bulk), verdict text, patch composer,
   PR writer. Ultra — case gen (adversarial), adapter, iterator, judge. Eval targets — lightning→super→ultra.
6. **Every LLM agent:** `temperature=0`; `response_format={"type":"json_schema",...}` when the tier supports it
   (flag in tiers.py), else `{"type":"json_object"}` + Pydantic validation + exactly 1 repair retry; per-agent
   `max_tokens` cap; all responses cached. Strip `<think>…</think>` and use `message.content` only (ignore
   `reasoning_content`) when building `ModelOutput.text`.
7. **Cache = cassette.** Content-addressed: `sha256(canonical_json({model, messages, params}))` → one JSON file
   per response under the cache dir. Live client writes on miss; `live=False` raises `CacheMiss` on miss.
   Agent contract tests point `TFClient(cache_dir=tests/fixtures/cassettes/<agent>, live=False)` at committed
   cassettes. Until a real key exists, cassettes may be hand-authored and MUST carry `"synthetic": true`;
   Task 14 re-records them live (`pytest --live`) and removes the flag.
8. **Cost guardrails (§10):** per-run token cap 1.5M input / 300k output (uncached calls only); Ultra *agent*
   calls (case_gen adversarial, adapter, iterator, judge) capped at 40 per run. Eval-target calls count toward
   token caps, not the Ultra-agent cap. Exceeding a cap raises `BudgetExceeded`; the pipeline turns that into
   `KEEP_CLOSED(reason="budget …")` for affected sites, never a crash. Cache hits are free.
9. **Honesty:** every number shown anywhere (UI, PR, README, video) comes from a measured run artifact
   (`evals/results/*.json`, run events). No invented numbers in shipped artifacts. Dev/test fixtures with made-up
   numbers live only under `tests/fixtures/` or `web/fixtures/` and are labelled as fixtures.
10. **Holdout:** `split = "holdout" if int(sha256(case.id).hexdigest(), 16) % 100 < 30 else "dev"`. The Iterator
    never receives holdout cases or holdout case ids (instrumented, tested). Final verdict uses holdout.
11. **Pass rules (§6.5)** — fraction of cases passing per site/tier/split:
    json_extraction: 100% schema-valid AND ≥95% field match (key-set + value equality vs baseline);
    classification: ≥95% label agreement; tool_call: ≥95% (tool name equal + normalised args equal);
    free_text: judge mean ≥ baseline_mean − 0.3 AND ≥ 4.0 (1–5 rubric); reasoning: ≥90% final-answer match
    (exact or numeric within 1e-6 relative). Thresholds configurable via `RunConfig` (default 0.95 / 0.90).
12. **Run stages** (`offramp.schemas.Stage`): QUEUED → CLONED → SCANNED → BASELINED → EVALUATING → ITERATING →
    VERDICTS → PATCHED → PR_OPENED → DONE | FAILED. Every transition emits a `RunEvent(type="stage")`.
13. **API (Task 12) — the web app's only contract with the backend:**
    - `POST /api/runs` body `RunConfig` → `{"run_id": str}`; `?replay=demo` → starts a replay run (no TF calls).
    - `GET /api/runs/{run_id}` → `RunSnapshot` `{run_id, config, stage, sites: CallSite[], tier_results:
      TierResult[], verdicts: Verdict[], totals: RunTotals, artifacts: {pr_url?, report_url?}}`.
    - `GET /api/runs/{run_id}/events` → `text/event-stream`; each event `id: <seq>`, `event: <type>`,
      `data: <RunEvent json>`; honours `Last-Event-ID`; ends after `done` or `error`.
    - `GET /api/runs/{run_id}/report` → JSON download `{snapshot, pr_body_markdown}`.
    - `GET /api/health` → `{"ok": true, "tf_configured": bool}`.
    - Rate limit 30 req/min per IP on `POST /api/runs` (429). CORS allow the web origin (env `OFFRAMP_WEB_ORIGIN`).
14. **Replay (§9):** `?replay=demo` plays a recorded event log at 4× speed, gaps capped at 1.5 s before
    speed-up, so a replay finishes < 90 s. Recorded log path: `offramp/api/replay/demo-events.jsonl` (API) and
    `web/public/replay/demo-events.json` (web, client-side fallback, same events). Both produced only by a real
    run (Task 14). Until then, replay shows an explicit "no recorded run yet" state.
15. **Secrets:** server-side only, from env (`.env` gitignored). Never log keys. No keys in tests or fixtures.
16. **Every task:** write tests first where logic is non-trivial; run them; commit on your branch with a clear
    message ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never dispatch subagents.
    Never touch files owned by another task (ownership listed per task). Tests that need an absent/empty
    resource must create that absence themselves (tmp dirs), never rely on tree state.

### Amendment A — no closed-model keys (user decision, 2026-10-19). Overrides anything above or below.

Credits available: Nebius Token Factory $25, LangSmith $100, Tavily $25, Toloka $50, Tendem (by Toloka) $50.
No OpenAI or Anthropic keys exist, so PRD G2's "baseline recorded from the original closed model" cannot be
produced for the demo. Never fabricate closed-model outputs, latencies or costs.

17. **Baselines are references** (`BaselineResult.source="reference"`, see `offramp/schemas.py`). Methods:
    `reference:seed` — the case generator picks the expected output first (label / tool name + args / extraction
    record) and asks Super to write a realistic input that implies it; expected = seed.
    `reference:oracle` — deterministic code computes the answer (cs_05 refund math; the demo repo's tests already
    hold an independent reference implementation — port it to `evals/reference/refund_oracle.py` in the demo repo).
    `reference:human` — written or verified by a human expert (Tendem), optional. References carry no
    latency/tokens/cost (None).
18. **Reference audit:** every `seed` reference gets a blind check — Ultra (thinking on) answers the input without
    seeing the seed; disagreements are dropped or sent to human audit. Record n checked / agreed / dropped /
    human-verified per site; show it in the report and PR.
19. **Parity = agreement with references** on holdout, same scorers and thresholds. Column header
    "Parity vs reference (holdout)".
20. **Free text (cs_04, cs_06):** judge (Ultra) scores 1–5 on an absolute rubric against the evidence (KB snippets
    or thread): groundedness, completeness, format. Pass: candidate mean ≥ 4.0; when human references exist, also
    candidate mean ≥ reference mean − 0.3 (references judged with the same rubric, position-swapped pairwise).
21. **Cost before is an estimate:** source-model published price from a Tavily runtime lookup (§6.10, now core;
    fallback `pricing.yaml`) × tokens measured on the Token Factory call for the same prompt. Shown as "est." with
    price source URL and date (`Verdict.cost_before_note`). Latency before is not measured: show "—".
    Tavily also returns deprecation notices → `Verdict.sunset_risk` badge.
22. **Closed-model recorder** (`scripts/record_baselines.py`) stays for teams that have keys (PRD G2); tested with
    fake SDK clients only; the demo never claims closed-model numbers.
23. **LangSmith** (when `LANGSMITH_API_KEY` set): trace every TF call under project `offramp-<run_id>`; upload each
    site's cases + references as a dataset and each (site, tier, split) evaluation as an experiment; share links go
    into the report and PR evidence section.
24. **Tendem** human audit via `scripts/audit_references.py` (builds the audit task text from flagged/sampled
    references; ingests verdicts → `verified_by: ["human:tendem"]`). Every paid task needs the user's explicit
    approval of its quoted price.
25. **Reasoning control (probe: `docs/plan/tf-probe.md`):** always send `chat_template_kwargs.enable_thinking`
    explicitly; `false` for every structured agent call unless the spec asks for reasoning. Reasoning tokens count
    toward `max_tokens`: when thinking is on, budget ≥ 2000. Lightning with thinking on leaks reasoning prose into
    `content`; never evaluate Lightning with thinking on unless the adapter explicitly chose it.
26. **TF spend:** $25 total. Dev runs reuse the cache; never re-run a full live matrix without need.

---

## Wave 1 — independent, run in parallel

### Task 1: Token Factory client, cache, tiers, budget, pricing, DB models

**Owns:** `offramp/tf/{__init__,client,cache,tiers,budget}.py`, `offramp/evals/pricing.yaml`,
`offramp/pricing.py`, `offramp/db.py`, `tests/unit/test_{cache,tf_client,budget,pricing,db}.py`.

- `tiers.py`: `TierInfo(tier, model_id, input_usd_per_m, output_usd_per_m, supports_json_schema: bool,
  reasoning_style)`; `get_tier(Tier) -> TierInfo`; env overrides per Global Constraint 4; prices read from
  `pricing.yaml`. Reasoning control: map `AdapterSpec.reasoning` to request params via one function
  `reasoning_params(tier, level) -> dict` (default: `extra_body={"chat_template_kwargs": {"enable_thinking":
  level != "off"}}`; exact mechanism verified in Task 14 — keep it in this one function).
- `pricing.yaml` (§6.5): timestamped (`as_of`, `source` URL per entry) prices for TF models (Ultra $1/$3,
  Super $0.30/$0.90, Lightning $0.06/$0.24 per 1M in/out) and closed source models used by the demo
  (gpt-4o-mini, gpt-4o, gpt-4.1-mini, o3, o4-mini, claude-sonnet-4-5, claude-haiku-4-5 — look up current
  published prices; record source URL). `pricing.py`: `price(model) -> (in, out)`, `cost_usd(model, tin,
  tout)`, `cost_per_1k_calls(model, mean_tin, mean_tout)`.
- `cache.py`: `cache_key(model, messages, params) -> str` (stable across dict key order), `Cache(dir)` with
  `get(key) -> dict|None`, `put(key, entry)`; entry stores request, raw response JSON, latency_ms, usage,
  created_at, optional `synthetic`. No TTL.
- `budget.py`: `Budget(max_in=1_500_000, max_out=300_000, max_ultra_agent_calls=40)`; `charge(role, tier,
  tin, tout)`; raises `BudgetExceeded`; `snapshot() -> dict`. Thread-safe.
- `client.py`: `TFClient(api_key=None, base_url=None, cache_dir, live=True, budget=None)` and
  `class LLM(Protocol)` with one method:
  `chat(*, role: str, tier: Tier, messages, max_tokens, temperature=0.0, response_model: type[BaseModel]|None=None,
  json_schema: dict|None=None, tools=None, tool_choice=None, reasoning="off", extra_body=None) -> ChatResult`.
  `ChatResult(output: ModelOutput, parsed: Any|None, model, tier, latency_ms, input_tokens, output_tokens,
  cost_usd, cached: bool, json_path: Literal["json_schema","json_object","none"], repaired: bool)`.
  Behaviour per Global Constraints 6–8: json_schema path when supported, else json_object + validation +
  one repair retry (send the invalid output + validation error back, ask for corrected JSON); parse
  `tool_calls` into `ModelOutput.tool_calls`; retries with backoff on 429/5xx/timeouts (max 3); budget
  charged only on uncached calls. Optional LangSmith tracing: if `LANGSMITH_TRACING=true` and `langsmith`
  importable, wrap the OpenAI client with `langsmith.wrappers.wrap_openai`. Also `FakeLLM` in
  `offramp/tf/fake.py` (owned here) for other tasks' unit tests: queue of canned `ChatResult`s/outputs keyed
  by role, records calls.
- `db.py` (§7): SQLModel tables Run, CallSite, EvalCase, Baseline, Adapter, TierResult, Verdict, Artifact with
  JSON columns as text; `engine(path)`, `init_db()`. Keep it thin — the API (Task 12) is the main user.
- Tests: cache hit/miss and key stability (key order, whitespace-insensitive JSON), TTL off; client via
  `httpx.MockTransport` (no network): json_schema path, json_object + repair retry on malformed first reply,
  `<think>` stripping, tool_call parsing, cache-only miss raises, budget accounting & Ultra cap; pricing $/1k
  arithmetic; db round trip in tmp sqlite.
- **Done when:** `uv run pytest tests/unit -q` green, ruff clean.

### Task 2: Demo repo `offramp-demo-app`

**Owns:** everything in `nvidia/offramp-demo-app/` (separate git repo, already `git init`ed on `main`).

Realistic support-ops SaaS ("HelpHub") Python package `helphub/` with exactly **6 chat call sites** + 1
embeddings call, prompts as module-level constants and f-strings (so the scanner can render templates):

| id (scan order) | file (exact path) | task | SDK / model | contract |
|---|---|---|---|---|
| cs_01 | `helphub/inbox/triage.py` | classify ticket into one of 6 queues | openai `gpt-4o-mini` | classification, labels enumerated in the prompt |
| cs_02 | `helphub/ops/agent_step.py` | choose next tool (lookup_order / issue_refund / escalate_to_human) | anthropic `claude-sonnet-4-5`, Anthropic `tools=` | tool_call |
| cs_03 | `helphub/orders/extract.py` | extract order/customer fields from email | anthropic `claude-sonnet-4-5` | json_extraction with a JSON schema (Pydantic model) |
| cs_04 | `helphub/qa/rag_answer.py` | answer from retrieved KB snippets | openai `gpt-4o` | free_text |
| cs_05 | `helphub/refunds/plan.py` | multi-step arithmetic refund/credit planning (prorations, tiered discounts, currency) | openai `o3` | reasoning, final numeric answer on last line `ANSWER: <number>` |
| cs_06 | `helphub/summaries/summarize.py` | summarise a ticket thread for handoff | anthropic `claude-haiku-4-5` | free_text |

The scanner assigns ids by sorted (path, line), so these exact paths produce the ids above (cs_03 = JSON
extract, cs_05 = reasoning canary, matching §8/§12.3). Exactly one chat call per file. Plus
`helphub/search/embed.py` with one `client.embeddings.create(...)` call (scanner flags it as `emb_01`;
Offramp never migrates it).

- Clients created via `OpenAI()` / `Anthropic()` at module level or injected; functions take plain inputs
  and return typed results. `helphub/kb/` small markdown KB (10–20 short articles) for RAG.
- cs_05 must be genuinely hard (multi-step proration with tiers, rounding rules) — it is the honest
  KEEP_CLOSED canary; do not make it trivial.
- `tests/`: pytest with fake clients (no network) covering each function's parsing/contract;
  `tests/fixtures/` real-looking sample inputs (tickets, emails, threads) — at least 10 per site, diverse:
  long/short, non-English (de, es, ja), malformed, empty, unicode. These are the "existing repo fixtures"
  the Case Generator prefers.
- `pyproject.toml` (uv, py3.12, deps openai + anthropic + pydantic, dev pytest+ruff), `README.md` (what it
  is, how to run with closed keys), `LICENSE` MIT, `.gitignore`.
- **Done when:** `uv run pytest -q` and `uv run ruff check .` green in the demo repo; committed on `main`.

### Task 3: Scanner (AST) + Super contract labelling

**Owns:** `offramp/agents/scanner.py`, `tests/fixtures/scanner/**`, `tests/unit/test_scanner.py`,
`tests/agents/test_scanner_label.py`, `tests/fixtures/cassettes/scanner/**`.

- `scan_repo(path) -> ScanResult` (deterministic, no LLM): walk `*.py` (skip venvs, `tests/`, hidden dirs),
  `ast` detect: `openai` (`OpenAI()/AsyncOpenAI()` client `.chat.completions.create`, `.responses.create`,
  `.embeddings.create`, module-level `openai.chat.completions.create`), `anthropic` (`Anthropic()`
  `.messages.create`), `litellm.completion`, LangChain `ChatOpenAI(...)`/`ChatAnthropic(...)` constructors.
  Track client variables through simple assignment (module and function scope, `self.client = OpenAI()`).
- Extract per §6.1: file, line span, enclosing function, model string, prompt source
  (literal/fstring/variable/dynamic), `tools`/`tool_choice`, `response_format`, `stream`, `max_tokens`,
  `temperature`, Anthropic `system=`. Resolve module-level constants (strings, dicts, lists, Pydantic
  `model_json_schema()` → mark `"<pydantic:ClassName>"`) by AST constant folding. Render `PromptTemplate`:
  f-strings become `{name}` placeholders, `variables` listed. Unresolvable → `prompt_source="dynamic"`, flag
  `NEEDS_RUNTIME_CAPTURE`. `stream=True` → flag `MANUAL_REVIEW`. Embeddings → `kind="embedding"`, flag
  `EMBEDDINGS`. Ids `cs_01..` over chat sites sorted by (file, line); embeddings get `emb_01..`.
- `label_contracts(sites, llm: LLM) -> list[CallSite]`: one Super call per chat site (role `scanner`,
  `response_model=Contract`), given code snippet + params; fills `contract` and `confidence`. Deterministic
  hints override the LLM where certain (tools present → tool_call; response_format json_schema → json_extraction).
- Fixtures: one file per SDK pattern + negatives (openai imported but unused, embeddings call, streaming call,
  dynamic prompt). Unit tests assert exact `CallSite` lists. Agent test uses `FakeLLM` from Task 1's
  interface — if `offramp/tf/fake.py` is not on your branch yet, write the test against a local minimal fake
  implementing the `LLM` protocol signature from this plan; the controller reconciles at merge.
- **Done when:** unit tests green; ruff clean.

### Task 4: Deterministic scorers

**Owns:** `offramp/evals/scorers/{__init__,json_schema,label,tool_call,reasoning,aggregate}.py`,
`tests/unit/test_scorers.py`.

- Each scorer: `score(case: EvalCase, baseline: ModelOutput, candidate: ModelOutput, contract: Contract) ->
  CaseScore` with `error_category` on failure.
  - json_schema: parse candidate text as JSON (tolerate ```json fences), `jsonschema.validate` against
    `contract.json_schema` (or schema inferred from baseline when absent), field match = fraction of baseline
    leaf keys with equal normalised values (strings trimmed/casefolded, numbers numeric-equal); case passes
    when valid and field match ≥ 0.95 (per-case), and the site rule (aggregate) also requires 100% valid.
  - label: normalise (strip, casefold, strip trailing punctuation, map to `contract.labels` when given).
  - tool_call: tool name equal; args equal after normalisation (key order, whitespace, number formatting
    `"5"`/`5`/`5.0`, case-insensitive enum strings).
  - reasoning: extract final answer (`ANSWER:` line, else last number); numeric relative tol 1e-6, else exact.
- `aggregate.py`: `site_pass(task_type, case_scores, thresholds, baseline_judge_mean=None) -> (passed, score)`
  implementing Global Constraint 11 (free_text branch takes judge scores computed elsewhere).
- Tests: table-driven, incl. edge cases from §12.1 (key order, whitespace, number formatting, numeric
  tolerance), plus threshold monotonicity property: raising thresholds never flips fail→pass.
- **Done when:** green; ruff clean.

### Task 5: Web dashboard (Next.js + Tailwind)

**Owns:** everything under `web/`.

- Next.js (App Router, TypeScript) + Tailwind, dark mode default, pnpm. Brand: "Offramp" with persistent
  header badges "Nebius Token Factory" and "NVIDIA Nemotron" (text badges, no copied logos).
- Screen 1 `/` New Run (§9): repo URL input, "Use demo repo" button (fills `OFFRAMP_DEMO_REPO` URL),
  thresholds (default 95% / reasoning 90%), tier set checkboxes (Lightning/Super/Ultra), max iterations (3),
  Start → `POST /api/runs` → navigate to `/runs/[id]`. Also a prominent "Watch demo run (replay)" → `/runs/demo?replay=demo`.
- Screen 2 `/runs/[id]` Live Run: stage stepper (the 10 stages QUEUED…DONE; FAILED shown as an error state), per-site cards (file,
  task type, current tier, pass/fail chips per tier, iteration count), tier heatmap (sites × tiers, colour =
  parity, accessible legend + numeric labels), live log stream, Nebius/Nemotron badge on every verdict chip.
- Screen 3 `/runs/[id]/report` Report: verdict table (same columns as PR §8), cost before/after bars per site
  and total, latency deltas, "Open PR" CTA (artifacts.pr_url), "Download JSON".
- Data: SSE client on `GET {NEXT_PUBLIC_API_URL}/api/runs/{id}/events` (EventSource, reconnect with
  Last-Event-ID) reducing `RunEvent`s into state; types mirror `offramp/schemas.py` (write `web/lib/types.ts`
  by hand from it). Replay: when `?replay=demo`, load `/replay/demo-events.json` and play client-side per
  Global Constraint 14 (no backend needed); if the file is absent, show "No recorded demo run yet". `?live=1`
  forces API mode.
- Dev fixture `web/fixtures/dev-events.json` (clearly fixture data, labelled) to build/test the reducer and
  screens; it is never served at `/replay/`.
- Tests: reducer unit tests (vitest) over the dev fixture; `pnpm build` passes; `pnpm lint` clean.
- Accessibility: keyboard reachable controls, colour not the only signal in heatmap/chips, ≥4.5:1 contrast.
- **Done when:** `pnpm install && pnpm test && pnpm build` green.

---

## Wave 2 — after Wave 1 merges

### Task 6: Adapter translation + Adapter Agent

**Owns:** `offramp/agents/adapter.py`, `offramp/agents/translate.py`, `tests/unit/test_translate.py`,
`tests/agents/test_adapter.py`, `tests/fixtures/cassettes/adapter/**`.

- `translate.py` (deterministic, golden-tested): Anthropic tools (`name, description, input_schema`) →
  OpenAI function tools; Anthropic `tool_choice` → OpenAI; `system=` → system message; `max_tokens` mapping;
  OpenAI `response_format` passthrough; Pydantic schema → strict json_schema. `spec_v0(site) -> AdapterSpec`
  and `build_request(spec, case, tier) -> dict` (messages + params for `TFClient.chat`), and
  `to_model_output(...)` helpers if needed.
- `adapter.py`: Adapter Agent (Ultra, role `adapter`): given site + contract + `spec_v0`, returns
  `Adapter(version=0, spec, notes, risk_flags)` — may set reasoning level from `contract.reasoning_heavy`,
  add output-format instructions, choose json_schema vs json_mode fallback per tier capability.
- **Done when:** golden tests green; agent contract test green on cassettes (incl. malformed-cassette repair path).

### Task 7: Case Generator with references, reference audit, baselines I/O, recorder

**Owns:** `offramp/agents/case_gen.py`, `offramp/agents/reference_audit.py`, `offramp/evals/baselines.py`,
`scripts/record_baselines.py`, `scripts/audit_references.py`, `tests/agents/test_case_gen.py`,
`tests/agents/test_reference_audit.py`, `tests/unit/test_baselines.py`, `tests/harness/test_baseline_integrity.py`,
`tests/fixtures/cassettes/{case_gen,reference_audit}/**`; in the demo repo: `evals/**` (cases, references,
manifest, `evals/reference/refund_oracle.py`).

Read Amendment A (Global Constraints 17–26) first: there are no closed-model keys; expected outputs are
references with ground truth known by construction.

- Case Generator (§6.2 + GC17): per site produce 20–30 `EvalCase` + one `BaselineResult(source="reference")` each.
  Sources, in order: repo fixtures under the target's `tests/fixtures/` that already carry an expected output
  (e.g. refund fixtures with `expected_answer`) → `reference:oracle`/fixture; then seed-first synthetic via Super
  (role `case_gen`, thinking off): pick the expected output first — classification label (balanced across
  `contract.labels`), tool name + args, extraction record (deterministic Python generator with a seeded RNG),
  refund scenario parameters (oracle computes the answer) — then ask Super to write a realistic input implying it,
  diverse in length, locale (en/de/es/ja), malformed, empty-ish, unicode; then 3–5 Ultra adversarial cases
  (role `case_gen_adversarial`). Free-text sites (cs_04, cs_06): inputs only (question + retrieved KB snippets /
  thread), `expected_kind="text"`, reference output text None unless a human reference exists.
  Render `input_messages` from `site.template`; split per GC10. Write `evals/cases/<site_id>.jsonl` and
  `evals/references/<site_id>.jsonl` into the target repo.
- `evals/reference/refund_oracle.py` in the demo repo: port the independent reference implementation from the demo
  repo's tests (do not import from tests/); cs_05 cases are generated from oracle parameters → narrative by Super.
- Reference audit (GC18, `reference_audit.py`): blind Ultra check (thinking on, role `reference_audit`) per seed
  reference using the site's own scorer; disagreements dropped (kept in `evals/references/<site>.dropped.jsonl`
  with the reason) or queued for human audit. Writes `evals/audit.json` {site: checked, agreed, dropped,
  human_verified}. Budget: these are Ultra agent calls — batch several cases per call to stay inside the 40-call cap
  (run once, offline, cached; it is dataset preparation, not a per-run step).
- `scripts/audit_references.py` (GC24): `export` writes a human-readable Tendem task (markdown table of sampled +
  flagged references, instructions, answer format) to `evals/human_audit/task.md`; `ingest <file>` applies expert
  verdicts → `verified_by += ["human:tendem"]` or drops. No network in this script.
- `baselines.py`: load/save `evals/references/<site_id>.jsonl` (also reads legacy `evals/baselines/`), manifest
  `evals/manifest.json` (sha256 per file, method counts, generated_at, generator model ids), `verify_manifest()`.
- `scripts/record_baselines.py` (GC22): unchanged purpose — teams with keys record closed-model baselines
  (`source="closed_model"`); fake SDK clients in tests; `--self-agreement`.
- Eval-target calls are never made here. Role names: never start with `eval` (reserved for eval targets).
- **Done when:** tests green on cassettes/fakes; then the controller runs `offramp prepare` live once on the demo
  repo (Task 14) to produce the committed `evals/`.

### Task 8: Eval Runner + Judge Agent

**Amendment A notes:** compare candidates against `BaselineResult` references (GC17–20). Eval-target calls use role
`eval_target` (any role starting with `eval` is excluded from the Ultra agent cap). Price with
`get_tier(tier)` prices, not `price(model_id)`. Judge per GC20: absolute rubric vs evidence; pairwise vs reference
only when a human reference exists. Send `enable_thinking` explicitly per `AdapterSpec.reasoning` (GC25).

**Owns:** `offramp/evals/runner.py`, `offramp/agents/judge.py`, `tests/unit/test_runner.py`,
`tests/agents/test_judge.py`, `tests/harness/test_judge_sanity.py`, `tests/fixtures/cassettes/judge/**`.

- `run_tier(site, cases, baselines, adapter, tier, client, split, thresholds) -> TierResult`: executes cases
  (thread pool, bounded concurrency 8), scores via Task 4 scorers / judge, computes p50/p95 latency, tokens,
  `$/1k calls` from pricing (Task 1).
- Judge (§6.5): Ultra, pairwise A/B then B/A (position swap), rubric from contract, **batched**: one call
  scores all case pairs of one (site, tier, split, ordering) — keeps Ultra agent calls within the cap.
  Returns per-case 1–5 for candidate and baseline + consistency. `judge_agreement()` on overlap sites with
  deterministic scores (§12.3: ≥85% agreement, ≥90% position-swap consistency).
- **Done when:** tests green on cassettes/fakes.

### Task 9: Iterator, Verdict, per-site routing loop

**Amendment A notes:** verdict `cost_before_per_1k` is an estimate (GC21): source-model price from
`offramp.integrations.tavily.lookup_source_model(model)` (Task 17; fallback `pricing.yaml`) × mean tokens measured
on the TF calls for the same prompts; fill `cost_before_note` and `sunset_risk`; `latency_delta_ms=None` (before
not measured). KEEP_CLOSED reason wording: "parity vs reference <p>% < <threshold>%".

**Owns:** `offramp/agents/iterator.py`, `offramp/agents/verdict.py`, `offramp/evals/route.py`,
`tests/agents/test_iterator.py`, `tests/agents/test_verdict.py`, `tests/harness/test_holdout_leakage.py`,
`tests/harness/test_threshold_monotonicity.py`, `tests/harness/test_known_fail_canary.py`,
`tests/fixtures/cassettes/{iterator,verdict}/**`.

- Iterator (§6.6): Ultra; input = adapter, contract, ≤10 failing **dev** cases, error categories; output new
  `Adapter(version+1, change_log)`. Type-level guard: accepts `DevCase` view only; instrumented (records every
  case id it saw) for the leakage test.
- `route.py`: per site, for tier in config.tiers: eval dev → if fail iterate ≤ max_iterations → if dev passes
  eval holdout → holdout pass ⇒ MIGRATE(tier) stop; else next tier. All fail ⇒ KEEP_CLOSED(best tier/score).
  Emits callbacks for events (tier_result, iteration). BudgetExceeded ⇒ KEEP_CLOSED("budget …").
- Verdict (§6.7): rule-based decision + Super one-sentence reason; fills cost/latency deltas from baselines vs
  chosen tier.
- Harness tests: leakage, monotonicity, known-fail canary (cs_05-like reasoning site with fixture outputs must
  be KEEP_CLOSED).

### Task 10: Patch Composer, sandbox, PR Writer, GitHub PR builder

**Amendment A notes:** PR evidence columns: Site | File | Task | Before | After | Parity vs reference (holdout) |
p50 latency (after) | $/1k calls before (est.) → after. Add a "How parity was measured" section: reference methods
and audit counts per site (`evals/audit.json`), price sources + dates, LangSmith share links when present, and the
sentence "No closed-model outputs were recorded; parity is agreement with references."

**Owns:** `offramp/agents/patcher.py`, `offramp/agents/pr_writer.py`, `offramp/github.py`,
`offramp/sandbox.py`, `offramp/templates/client.py.tmpl`, `tests/unit/test_pr_render.py`,
`tests/integration/test_patch_safety.py`, `tests/agents/test_{patcher,pr_writer}.py`,
`tests/fixtures/cassettes/{patcher,pr_writer}/**`, `tests/fixtures/pr/**`.

- Patch Composer (§6.8): copy repo to sandbox dir; add `offramp/client.py` in the target (TF client wrapper,
  tier constants, reads `NEBIUS_API_KEY`); for each MIGRATE site rewrite the call (Super, role `patcher`)
  implementing the winning `AdapterSpec` exactly; KEEP_CLOSED sites get comment
  `# offramp: kept on <model> — parity <p>% < <threshold>%`; validate with `ruff check` + the repo's `pytest`
  in Docker (`--network none`, 2 CPU, 10-min cap; local subprocess fallback when Docker unavailable, flagged);
  tests fail ⇒ that site downgraded to MANUAL_REVIEW and its patch dropped. Produces unified diff per site.
- PR Writer (§6.9): Super writes prose bits only; the evidence table, numbers and `<details>` blocks are
  rendered deterministically from `Verdict[]`/`TierResult[]` (template §8) — snapshot-tested.
- `github.py`: GitHub REST via httpx (`GITHUB_TOKEN`): create branch `offramp/<run_id>`, commit files (git
  data API), open PR, return URL. Test with `httpx.MockTransport`.
- Patch-safety integration test (§12.4): a patch that breaks a repo test ⇒ MANUAL_REVIEW, no PR change for that site.

### Task 11: Serverless eval job packaging

**Owns:** `offramp/evals/serverless_job/**`, `Dockerfile.eval`, `tests/integration/test_serverless_job.py`.

- Entrypoint `python -m offramp.evals.serverless_job.main --input matrix.json --output results.json` runs a
  matrix (`call_sites × tiers × cases`) with the runner; `submit.py` submits to Nebius Serverless Jobs via the
  `nebius` CLI when `NEBIUS_PROJECT_ID` + CLI present, polls, fetches results; else runs locally in-process
  (`concurrent.futures`) — same output format. Test: tiny matrix local vs "job" path (subprocess running the
  entrypoint) are byte-equal after sorting (§12.4).
- Nebius CLI facts (docs.nebius.com/cli/reference/ai/job/create, /serverless/jobs/manage): `nebius ai job create
  --name --image --container-command --args --env KEY=VAL --platform --preset --disk-size --volume --timeout`
  (+ registry credentials); `nebius ai job list`, `nebius ai job cancel <id>`, `nebius ai job delete <id>`;
  `nebius ai job run` packages a local script and streams logs (`--detach`). Jobs bill only while running.
  Results return via stdout logs or a mounted volume — pick logs (print one JSON line `OFFRAMP_RESULTS <json>`)
  unless a volume is simpler once tried. The CLI is not installed yet; build the submit path behind a
  `shutil.which("nebius")` check and test it with a fake CLI script on PATH.

---

### Task 17: Tavily + LangSmith integrations

**Owns:** `offramp/integrations/{__init__,tavily,langsmith}.py`, `tests/unit/test_tavily.py`,
`tests/unit/test_langsmith.py`, `tests/fixtures/tavily/**`.

- `tavily.py` (GC21, §6.10, "Best Use of Tavily"): `lookup_source_model(model_id) -> SourceModelInfo(model,
  input_usd_per_m, output_usd_per_m, price_source_url, fetched_at, deprecation: str|None, deprecation_url)` via
  Tavily Search REST (`POST https://api.tavily.com/search`, `TAVILY_API_KEY`, `include_answer`, domain filters for
  official pricing/deprecation pages: openai.com, platform.openai.com, anthropic.com, docs.anthropic.com,
  docs.claude.com) then Super (role `pricing_extract`, json_schema, thinking off) extracts numbers from the returned
  content; numbers must appear verbatim in the cited page content or the result is rejected (fallback
  `pricing.yaml`, noted). Cached on disk per (model, date). httpx MockTransport tests with recorded Tavily fixtures.
- `langsmith.py` (GC23): thin, import-guarded helpers: `tracing_enabled()`, `run_project(run_id)`,
  `upload_dataset(site, cases, references) -> url`, `log_experiment(site, tier, split, case_results) -> url`,
  `share_link(...)`. No-ops returning None without a key. Tests use a fake client object.

## Wave 3

### Task 12: Orchestrator API, CLI, end-to-end pipeline

**Owns:** `offramp/api/**`, `offramp/pipeline.py`, `offramp/cli.py`, `tests/integration/test_pipeline_demo_repo.py`,
`tests/integration/test_api.py`, `tests/e2e/**`.

- `pipeline.py`: the run state machine over all agents (clone → scan → label → cases/baselines load → adapter
  v0 → route per site → verdicts → patch → PR), persisting artifacts per stage to `OFFRAMP_DATA_DIR/runs/<id>/`
  so a run resumes from its last completed stage. Emits `RunEvent`s; writes `events.jsonl`, `report.json`,
  `evals/results/<site>.json`.
- FastAPI app per Global Constraint 13; background thread worker queue; SSE; SQLite via `offramp/db.py`;
  replay runs per Global Constraint 14; rate limit; CORS.
- CLI: `offramp scan <path>`, `offramp prepare <path>` (case gen into target repo), `offramp run <repo>
  [--no-pr] [--cache-only]`, `offramp serve`, `offramp export-replay <run_id>` (writes both replay files).
- Integration: full pipeline on the demo repo in cache-only mode with committed cassettes → 6 verdicts,
  cs_05 KEEP_CLOSED, PR body snapshot. Load test script: 5 concurrent replay runs, API p95 < 500 ms.

### Task 13: CI, packaging, docs

**Owns:** `.github/workflows/**`, `Makefile`, `Dockerfile`, `README.md`, `LICENSE`, `docs/FEEDBACK.md`,
`docs/VIDEO.md`, `.gitleaks.toml`.

- CI: ruff, pytest with coverage ≥70% on `offramp/` (`--cov=offramp --cov-fail-under=70`), web test+build,
  gitleaks. Nightly workflow `pytest --live` (skips without secret) for schema-drift alerts.
- `make demo`: one command — `uv sync`, start API + web, open replay. README: setup, "How Nemotron + Token
  Factory are used" (per-agent tier table), architecture diagram, judges' path (no keys needed).

---

## Wave 4 — needs real keys (controller + agents)

### Task 14: Live verification and the real demo run
Verify model ids/json_schema support/reasoning control; re-record all cassettes live; generate cases and
record closed-model baselines for the demo repo; run the live pipeline; export replay; open the real PR on
the demo repo; commit `evals/results/*.json`.

### Task 15: Deploy
Web on Vercel (replay works with no backend). API on Nebius (Serverless Endpoint or VM) with server-side TF
key + rate limit, else documented local `make demo`. Clean-machine check.

### Task 16: Demo video (≤3:00, narrated, measured numbers only) and submission materials.

## Devpost submission deliverables (from nebiusglobalaihackathon.devpost.com + /rules, read 2026-10-19)

Deadline: **30 Oct 2026 10:00 PT** (hard close; PRD targets 29 Oct). Judging 1–15 Dec 2026: hosted demo and
repo must stay up and free to use until 15 Dec. Judging criteria: Technological Implementation, Design (complete
product, not a PoC), Potential Impact, Quality of the Idea. Track: **Coding and Agentic Engineering**.

| # | Deliverable | Rule | Where it lives |
|---|---|---|---|
| D1 | Working project with a runtime call to the Token Factory inference API, ≥1 NVIDIA open model | required | this repo |
| D2 | Track selection: Coding and Agentic Engineering | required | Devpost form |
| D3 | Text description: what, why, how; features and functionality | required | `docs/SUBMISSION.md` |
| D4 | Working demo URL, free, no login (or creds in testing instructions) | required | Vercel web (replay works with no keys) + hosted API |
| D5 | Demo video < 3:00, public YouTube, shows the project working and how Token Factory + Nemotron are used; no copyrighted music, no third-party trademarks/logos | required | `docs/VIDEO.md` script + rendered MP4 (user uploads) |
| D6 | Public repo, OSI licence detectable in the GitHub About section | required | `vj-io/offramp` (flip public at submission) + `vj-io/offramp-demo-app` |
| D7 | README: setup, clear run guidance, how Nemotron is used, where Token Factory accelerated the workflow, other Nebius services used | required | `README.md` |
| D8 | Feedback on Token Factory, AI Cloud, NVIDIA tools/models (specific, incl. what broke) | required; "Most Valuable Feedback" prize | `docs/FEEDBACK.md` |
| D9 | "Existed before?" explanation | n/a: created during the period (first commit 2026-10-19) | Devpost form |
| D10 | Best Use of Tavily ($3k; 3 Tavily judges): meaningful runtime Tavily call | optional bonus | §6.10 pricing/deprecation lookup + sunset-risk badge |
| D11 | Gallery images / thumbnail | Devpost form | `docs/media/` screenshots of the real dashboard |
| D12 | City (IRL Builders & Brews), team representative | user's choice | Devpost form |
