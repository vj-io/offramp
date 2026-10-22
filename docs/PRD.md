# Offramp — PRD v2 (Narrowed Scope)
**Nebius x NVIDIA Global AI Hackathon · Coding & Agentic Engineering track**
**Status:** Build-ready · **Date:** 18 Oct 2026 · **Submit by:** 29 Oct 2026 (hard close 30 Oct, 10:00 PT)

> **As shipped (2026-10-22).** This is the plan as written on 18 October, kept as history. Notes marked "As shipped" below say where the build differs; the README's [Plan versus shipped](../README.md#plan-versus-shipped) table has the whole list. In short: parity is measured against references, not a recorded closed-model baseline; the Serverless Job path is built but never ran on Nebius; the hosted site plays recordings and scans repos, while live runs need a local API; and `KEEP_CLOSED` shows in the no-Ultra run and on a real open-source repo, not in the main demo. Where this page says Offramp "proves" parity, read "shows, on held-out cases".

---

## 0. One-line

> Offramp turns "we should try open models" into a merge-ready pull request: it finds every closed-LLM call in your repo, **proves** which ones NVIDIA Nemotron on Nebius Token Factory can take over (eval, latency, cost evidence), and leaves the rest alone when it can't prove parity.

---

## 1. Problem & Why Now

- Teams want to move off closed LLM APIs (cost, lock-in, data residency) but can't **prove** an open model is good enough per use case.
- Token Factory is OpenAI-compatible, so the swap is trivial — the **risk** is silent regressions: broken JSON contracts, mismatched tool schemas, reasoning-budget differences.
- Current practice: manual A/B in a notebook, or nothing. No per-call-site evidence, no audit trail.
- **Why now**
  - OpenAI hosted Evals go read-only 31 Oct 2026; API shuts 30 Nov 2026 → teams are re-platforming evals anyway.
  - *As shipped: we have no public source we can cite for this date, so the submission does not use it. Its why-now is the Claude Sonnet 4.5 retirement on 30 Nov 2026 that Offramp's Tavily lookup found on Anthropic's model page.*
  - Nemotron 3 Ultra/Super + 3.5 Lightning live on Token Factory at $1/$3, $0.30/$0.90, $0.06/$0.24 per 1M tokens.
  - Judges (12 of 23 from Nebius product/DevRel/GTM) want proof that Token Factory is a production destination.

---

## 2. Goals / Non-goals

### Goals (must ship)
- G1. Detect all closed-LLM call sites in a Python repo (openai + anthropic SDKs).
- G2. For each call site, build an eval set and record a **baseline from the original closed model** (fixtures committed; judges need no closed keys).
  - *As shipped: no baseline was recorded, because we had no OpenAI or Anthropic key. Parity is agreement with references (seeded, computed by code, or written by Claude Sonnet 5.5); the recorder exists and is tested with fake clients.*
- G3. Run a tier matrix (Lightning → Super → Ultra) and pick the **cheapest tier that passes** a parity threshold.
- G4. On failure, iterate (prompt/schema adapter rewrite by Ultra) up to N loops.
- G5. Emit an honest verdict per call site: `MIGRATE(tier)` or `KEEP_CLOSED(reason)`.
- G6. Open a real GitHub PR with diff + evidence table + failing cases attached.
- G7. Web dashboard: paste repo URL → live progress → PR link. Cached demo run that never breaks.
  - *As shipped: live runs work against a local API (`make demo`, `offramp serve`) with your own key. The hosted site plays three recorded runs and scans any public repo without a key.*
- G8. Make Token Factory + Nemotron **impossible to miss** (UI, README, Built With, video audio).

### Non-goals (explicitly out)
- Multi-language support (Python only).
- Fine-tuning / LoRA distillation (roadmap slide only; Nemotron fine-tunability on TF unverified).
- Migrating non-LLM infra (vector DBs, embeddings) — embeddings flagged, not migrated.
- Full CI integration / GitHub App marketplace listing.
- Supporting streaming-only call sites beyond detection + flag.

---

## 3. Users & Jobs

| Persona | Job | Success looks like |
|---|---|---|
| Backend lead at a startup with $5–50k/mo closed-API spend | "Cut spend without breaking prod" | PR with numbers she can show CTO |
| Platform/ML-infra engineer | "Standardise on open models, EU residency" | Per-call-site verdicts + reasons for the hold-outs |
| Hackathon judge (Nebius PMM / NVIDIA engineer) | "Does this prove Token Factory + Nemotron are production-ready?" | Honest table incl. one `KEEP_CLOSED` row |

*As shipped: the main demo run has no `KEEP_CLOSED` row (every site passed parity). The no-Ultra run keeps one site on `gpt-4o`, and the open-source run keeps two on Claude.*

---

## 4. Scope Decisions (from adversarial review)

| Critique | Decision |
|---|---|
| "It's just a `base_url` swap" | Headline = per-call-site **parity gate + tier routing**; demo shows tool-schema translation, JSON contract fix, reasoning-budget prompt fix |
| "Evals are self-graded / fake" | Deterministic checks first; LLM-judge only for free-text; held-out split; baselines recorded from real closed models; failures shown |
| "Judges can't run it" | Hosted demo, preloaded repo, cached run, server-side rate-limited TF key, one-command README |
| "Feels like a CLI bot" | Dashboard is the product; PR description is a designed artifact |
| "Credits blow up" | Bulk evals on Lightning; Ultra only for rewrites + judging; token caps; full response caching |

---

## 5. System Architecture

```
┌──────────────┐   repo URL    ┌─────────────────────────────────────────┐
│  Dashboard   │ ────────────▶ │  Orchestrator (FastAPI + job queue)     │
│  (Next.js)   │ ◀──── SSE ─── │  state machine per run                   │
└──────────────┘               └───────┬───────────┬─────────────┬───────┘
                                       │           │             │
                              ┌────────▼──┐  ┌─────▼─────┐  ┌────▼──────────┐
                              │ Scanner   │  │ Eval Job  │  │ PR Builder    │
                              │ (AST +    │  │ (Nebius   │  │ (GitHub API)  │
                              │  Super)   │  │ Serverless│  └───────────────┘
                              └───────────┘  │  Jobs)    │
                                             └─────┬─────┘
                                                   │ OpenAI-compatible calls
                                        ┌──────────▼───────────────┐
                                        │ Nebius Token Factory     │
                                        │  nemotron-3-ultra        │
                                        │  nemotron-3-super        │
                                        │  nemotron-3.5-lightning  │
                                        └──────────────────────────┘
```

### 5.1 Components
- **Dashboard** — Next.js + Tailwind. Views: New Run, Live Run (per-call-site cards, tier matrix heatmap), Report, PR link. Dark mode default; Nebius/Nemotron badges on every verdict.
- **Orchestrator** — FastAPI. Owns `Run` state machine, dispatches subagents, streams events via SSE, persists to SQLite (Postgres optional).
- **Subagents** — see §6. Each is a pure function `(input) -> output` over Token Factory with a fixed model tier, strict JSON output, retries, and caching.
- **Eval Runner** — executes matrix `call_sites × tiers × cases` as a **Nebius Serverless Job**; falls back to local worker if job API unavailable.
- **PR Builder** — applies patches to a branch, renders evidence markdown, opens PR via GitHub REST (PAT scoped to demo org).
- **Cache** — content-addressed (`sha256(model, messages, params)`) on disk; every TF response cached; demo run fully replayable offline.

### 5.2 Run state machine
`QUEUED → CLONED → SCANNED → BASELINED → EVALUATING → ITERATING → VERDICTS → PATCHED → PR_OPENED → DONE | FAILED(stage, reason)`
- Every transition emits an SSE event `{run_id, stage, call_site_id?, payload}`.
- Resumable from any stage via persisted artifacts.

---

## 6. Subagent Specifications

All subagents: OpenAI-compatible chat completions against `https://api.tokenfactory.nebius.com/v1`; `response_format={"type":"json_schema"}` where supported, else JSON-mode + Pydantic validation + 1 repair retry; `temperature=0`; token caps per agent; results cached.

### 6.1 Scanner Agent
- **Model:** deterministic AST (no LLM) + **Nemotron 3 Super** for contract labelling.
- **Input:** repo path.
- **Steps:**
  1. `ast` walk → find `openai.*`, `OpenAI().chat.completions.create`, `anthropic.*messages.create`, `litellm.completion`, LangChain `ChatOpenAI/ChatAnthropic` constructors.
  2. Extract: file, line span, model string, messages/prompt source (literal, f-string, variable), `tools`/`tool_choice`, `response_format`, `stream`, `max_tokens`, `temperature`.
  3. Super labels each site with `contract`.
- **Output (per call site):**
  ```json
  {"id":"cs_03","file":"app/extract.py","lines":[41,67],"sdk":"anthropic","source_model":"claude-sonnet-4-5",
   "contract":{"task_type":"json_extraction","output":"json_schema","tools":false,"streaming":false,
               "latency_budget_ms":1500,"reasoning_heavy":false},"prompt_ref":"PROMPT_EXTRACT","confidence":0.92}
  ```
- **Failure modes:** dynamic prompt construction → mark `prompt_source:"dynamic"`, request fixture capture at runtime; streaming → flag `MANUAL_REVIEW`.

### 6.2 Case Generator Agent
- **Model:** **Nemotron 3 Super** (bulk), **Ultra** for adversarial edge cases (3–5 per site).
- **Input:** call site + contract + any existing test fixtures found in repo.
- **Output:** 20–30 `EvalCase{id, input_messages, expected_kind, tags[]}` per site; split 70/30 dev/holdout (holdout hashed, never shown to Iterator).
- **Rules:** prefer real fixtures from repo tests; synthetic cases must be diverse (length, locale, malformed input, empty, unicode).

### 6.3 Baseline Recorder
- **Model:** the **original closed model** (run once by the team, offline; outputs committed as fixtures under `evals/baselines/`).
- **Output:** `BaselineResult{case_id, output, latency_ms, input_tokens, output_tokens, cost_usd}`.
- **Note:** Judges never need closed keys. Recorder is a script, not an LLM agent.

### 6.4 Adapter Agent
- **Model:** **Nemotron 3 Ultra**.
- **Job:** translate the call site to Token Factory semantics.
  - Anthropic `tools` → OpenAI function-calling schema; `tool_use` blocks → `tool_calls`.
  - Anthropic `system` param → system message; `max_tokens` mapping.
  - `response_format` → JSON schema or JSON-mode + validation.
  - Inject Nemotron reasoning control (`reasoning_effort` / thinking budget) based on `contract.reasoning_heavy`.
- **Output:** `Adapter{patch (unified diff), notes[], risk_flags[]}` — initial version before any eval.

### 6.5 Eval Runner (non-LLM + Judge)
- **Compute:** Nebius Serverless Job (`offramp-eval:<run_id>`), parallel over tiers.
  - *As shipped: built and tested against a fake `nebius` CLI; every run used the local worker, because AI Cloud billing is separate from the Token Factory credits.*
- **Target tiers:** `nemotron-3.5-lightning` → `nemotron-3-super` → `nemotron-3-ultra` (stop at first pass).
- **Scoring (deterministic first):**
  | Contract | Check | Pass rule |
  |---|---|---|
  | json_schema | `jsonschema.validate` + key-set equality vs baseline | 100% valid, ≥95% field match |
  | classification | exact label match vs baseline | ≥95% agreement |
  | tool_call | tool name == baseline; args semantic-equal (normalised) | ≥95% |
  | free_text | **Judge Agent** rubric 1–5 vs baseline | mean ≥ baseline − 0.3 and ≥4.0 |
  | reasoning | final-answer exact/numeric match | ≥90% |

  *As shipped: "baseline" became the references. Free text also needs no holdout case below 3.0, and when the call sets a timeout, the tier's holdout p50 must be within it.*
- **Judge Agent:** **Nemotron 3 Ultra**, pairwise with position-swap (A/B then B/A), rubric from contract; reports agreement with deterministic checks on overlap set (sanity).
- **Also records:** p50/p95 latency, tokens, `$/1k calls` from published TF prices (loaded from `pricing.yaml`, timestamped).
- **Output:** `TierResult{site_id, tier, pass, score, metrics, failing_case_ids[]}`.

### 6.6 Iterator Agent
- **Model:** **Nemotron 3 Ultra**.
- **Trigger:** cheapest tier fails on dev split.
- **Input:** adapter patch, contract, up to 10 failing cases (dev split only), error categories.
- **Action:** rewrite prompt / schema adapter / add few-shot / adjust reasoning budget → new `Adapter` version. Max **3 loops per tier**; if all tiers fail → `KEEP_CLOSED`.
- **Guard:** holdout split evaluated only after iteration converges; final verdict uses holdout.
- **Output:** `AdapterV{n}` + `change_log[]`.

### 6.7 Verdict Agent
- **Model:** rule-based + **Nemotron 3 Super** for the human-readable reason.
- **Rule:** `MIGRATE(tier)` = cheapest tier passing holdout; else `KEEP_CLOSED(reason, best_tier, best_score)`.
- **Output:** `Verdict{site_id, decision, tier?, parity_pct, latency_delta, cost_before, cost_after, reason}`.

### 6.8 Patch Composer
- **Model:** **Nemotron 3 Super** (code edits), validated by `ruff` + `pytest` run.
- **Job:** apply only `MIGRATE` adapters; add `offramp/client.py` (TF client wrapper with tier constants); leave `KEEP_CLOSED` sites untouched but add `# offramp: kept on <model> — parity 71% < 95%` comment.
- **Safety:** patches applied in a sandbox clone; repo tests must still pass or patch is downgraded to `MANUAL_REVIEW`. Stretch: run inside NVIDIA OpenShell.

### 6.9 PR Writer Agent
- **Model:** **Nemotron 3 Super**.
- **Output:** PR title, body (template §8), evidence table, collapsed `<details>` per site with failing cases, cost summary, "What Offramp did NOT change" section.

### 6.10 (Optional) Pricing/Deprecation Lookup
- **Tool:** Tavily runtime search → source-model price + deprecation notices → feeds `cost_before` and a "sunset risk" badge. Half a day; enables Tavily bonus eligibility.

### 6.11 Model-tier allocation summary
| Agent | Tier | Why |
|---|---|---|
| Scanner labelling, Case Gen (bulk), Verdict text, Patch Composer, PR Writer | Super | Cheap, structured, code-capable |
| Case Gen (adversarial), Adapter, Iterator, Judge | Ultra | Reasoning + long context |
| Eval target (default) | Lightning → Super → Ultra | Cheapest-passing routing is the product |

---

## 7. Data Model (SQLite via SQLModel)

- `Run(id, repo_url, commit_sha, stage, created_at, totals_json)`
- `CallSite(id, run_id, file, line_start, line_end, sdk, source_model, contract_json, prompt_ref)`
- `EvalCase(id, site_id, split, input_json, tags)`
- `Baseline(case_id, output_json, latency_ms, tokens_in, tokens_out, cost_usd)`
- `Adapter(id, site_id, version, patch, notes_json)`
- `TierResult(id, site_id, adapter_id, tier, split, pass, score, p50, p95, tokens_in, tokens_out, cost_per_1k, failing_case_ids_json)`
- `Verdict(site_id, decision, tier, parity_pct, cost_before, cost_after, latency_delta_ms, reason)`
- `Artifact(run_id, kind, path)` — PR url, report md, serverless job id

---

## 8. PR Output Spec

**Title:** `Offramp: migrate 5/6 LLM call sites to Nemotron on Nebius Token Factory (est. −83% cost)`

**Body template:**
```
## Summary
Offramp evaluated 6 closed-LLM call sites against recorded baselines (N=148 cases, 30% holdout).
5 migrated to the cheapest Nemotron tier that passed parity. 1 kept on the original model.

## Evidence
| Site | File | Task | Before | After | Parity (holdout) | p50 latency | $/1k calls before → after |
|---|---|---|---|---|---|---|---|
| cs_01 | app/triage.py | classification | gpt-4o-mini | nemotron-3.5-lightning | 97.8% | 410→380 ms | $0.21 → $0.03 |
| cs_03 | app/extract.py | json_extraction | claude-sonnet-4-5 | nemotron-3-super | 96.2% (after 2 adapter iterations) | 1.9→1.4 s | $3.10 → $0.48 |
| cs_05 | app/plan.py | reasoning | o3 | **NOT MIGRATED** | 71.0% < 95% | — | — |

## What changed
- `offramp/client.py` added (Token Factory client, tier constants)
- Per-site diffs below; adapter notes explain each prompt/schema change

## What Offramp did NOT change
- cs_05 stays on o3: multi-step arithmetic planning parity 71%. Suggested next step: LoRA distillation on Token Factory.

<details><summary>Failing cases (cs_05)</summary>…</details>
Generated by Offramp · Nemotron 3 Ultra/Super/3.5 Lightning via Nebius Token Factory · run abc123
```

---

## 9. Dashboard Spec

- **Screen 1 — New Run:** repo URL input, "Use demo repo" button, thresholds (default 95% / 90% reasoning), tier set, max iterations.
- **Screen 2 — Live Run:** stage stepper; per-site cards (file, task, current tier, pass/fail chips); tier heatmap (sites × tiers, colour = parity); live log stream; Token Factory + Nemotron badges in header.
- **Screen 3 — Report:** verdict table (same as PR), cost before/after bar, latency deltas, "Open PR" CTA, download JSON.
- **Demo safety:** `?replay=demo` serves cached events at 4× speed; never calls TF live during the video unless `?live=1`.

---

## 10. Infrastructure

| Need | Choice | Notes |
|---|---|---|
| LLM inference (mandatory) | Nebius Token Factory | `OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1` |
| Eval matrix compute | Nebius Serverless Jobs | container `offramp-eval`; fallback: local `rq` worker |
| App hosting | Nebius Serverless Endpoint (API) + Vercel (web) or single VM | TF call satisfies the gate; frontend host is free choice |
| Repo sandbox | Docker, network-off, 2 CPU, 10-min cap | Stretch: NVIDIA OpenShell |
| Secrets | server-side only; `.env` gitignored; git history audited | rate limit 30 req/min per IP |
| Credits | Claim `$25 TF grant` + Builders Program on Day 1 | budget: ≤$15 total for all dev runs; demo run ≈ $0.60 |

**Cost guardrails**
- Per-run token cap: 1.5M input / 300k output.
- Ultra calls capped at 40 per run.
- All responses cached; re-runs are free.

---

## 11. Build Plan (12 days, team of 3–4)

Roles: **A** = backend/orchestrator, **B** = evals/agents, **C** = frontend/PR/video, **D (optional)** = demo repo + docs + feedback.

| Day | Date | Deliverable | Owner | Done when |
|---|---|---|---|---|
| 1 | 19 Oct | Repo skeleton, TF client wrapper, cache, pricing.yaml, claim credits, CI (ruff+pytest) | A | `pytest` green; one Nemotron call cached |
| 1 | 19 Oct | Demo repo `offramp-demo-app` with 6 call sites (triage/classify, JSON extract, tool-call agent step, RAG answer, summary, hard reasoning), tests | D/B | repo runs locally with closed keys |
| 2 | 20 Oct | **Attend "Will It Ship" 9am PT**; Scanner AST + Super labelling | A | 6/6 sites detected on demo repo; unit tests |
| 2 | 20 Oct | Baseline Recorder script; record + commit baselines (≈150 cases) | B | fixtures committed; cost logged |
| 3 | 21 Oct | Case Generator + dev/holdout split; Adapter v0 | B | ≥20 cases/site; adapters compile |
| 4 | 22 Oct | Eval Runner (local) with deterministic scorers | B | TierResults for 6 sites × 3 tiers |
| 5 | 23 Oct | Judge Agent (pairwise, position-swap) + agreement check; Serverless Job packaging | B/A | job runs on Nebius; results identical to local |
| 6 | 24 Oct | Iterator loop + Verdict; make cs_05 honestly fail | B | at least one KEEP_CLOSED, one multi-iteration pass |
| 7 | 25 Oct | Patch Composer + sandbox test run; PR Writer; open real PR | A/C | PR visible on GitHub with evidence table |
| 8 | 26 Oct | Dashboard screens 1–3, SSE stream, replay mode | C | end-to-end click-through on demo repo |
| 9 | 27 Oct | Deploy; Tavily pricing lookup (optional); OpenShell stretch (optional) | A | public URL works from clean machine |
| 10 | 28 Oct | Video script + record (<3 min); README; LICENSE (MIT); feedback section | C/D | video public on YouTube |
| 11 | 29 Oct | Dry-run judging from clean laptop; fix; **submit** | all | Devpost submission confirmed |
| 12 | 30 Oct | Buffer only | — | — |

*As shipped: the day 5 job never ran on Nebius (see §6.5). The day 6 cs_05 failure did not happen: cs_05, scored against code-computed answers, passed on Lightning in every demo run that reached a verdict, and we did not rig it.*

**Daily ritual:** 15-min standup; "demo it" rule — every PR to main must keep the replay demo working.

---

## 12. Test Strategy

### 12.1 Unit tests (pytest, run in CI on every push)
- Scanner: fixture files per SDK pattern → expected `CallSite` list (incl. negatives: `openai` imported but unused, embeddings call, streaming call).
- Adapter translation: golden tests Anthropic tool schema → OpenAI function schema; `system` mapping; `max_tokens`.
- Scorers: jsonschema, label match, tool-arg normalisation (key order, whitespace, number formatting), numeric tolerance.
- Pricing: `$/1k` arithmetic from `pricing.yaml`.
- Cache: hit/miss, key stability, TTL off.
- PR renderer: snapshot test of markdown from a fixture `Verdict[]`.

### 12.2 Agent contract tests (recorded, replayed offline)
- Every subagent has `tests/agents/test_<agent>.py` using VCR-style cassettes of TF responses.
- Assert: output validates against Pydantic schema; repair-retry path works on a deliberately malformed cassette; token caps respected.
- Nightly `--live` flag re-records against Token Factory and diff-alerts on schema drift.

### 12.3 Eval-harness validity tests (the "evals aren't fake" tests)
- **Holdout leakage:** assert Iterator never receives holdout case ids (instrumented).
- **Judge sanity:** on sites with deterministic scores, judge agreement ≥ 85%; position-swap consistency ≥ 90%.
- **Baseline integrity:** fixture hashes match manifest; cost/latency fields present.
- **Threshold monotonicity:** raising threshold never flips KEEP_CLOSED → MIGRATE.
- **Known-fail canary:** cs_05 must produce `KEEP_CLOSED`; CI fails if it "passes" (guards against scorer bugs that over-pass).
  - *As shipped: the canary is `tests/harness/test_known_fail_canary.py`, a cs_05-like site whose fixture answers never reach 90% and must stay `KEEP_CLOSED`. The real cs_05 passes (see §11).*

### 12.4 Integration tests
- `test_pipeline_demo_repo.py`: full run against demo repo in replay mode → 6 verdicts, 5 MIGRATE, 1 KEEP_CLOSED, PR body snapshot.
  - *As shipped: the test replays a recorded run on its own small fixture dataset and pins that run's verdicts. The final demo run gave 5 MIGRATE and 1 MANUAL_REVIEW.*
- Serverless Job: submit tiny matrix, poll, compare to local runner output (byte-equal after sorting).
- Patch safety: inject a patch that breaks a repo test → expect `MANUAL_REVIEW`, no PR for that site.

### 12.5 End-to-end / demo rehearsal
- Clean-machine checklist (fresh clone, `make demo`, open URL) by a teammate who didn't build it — Day 9 and Day 11.
- Live-mode rehearsal with TF key on Day 10 (budget ≤ $1); replay fallback verified.
- Load test: 5 concurrent replay runs; API p95 < 500 ms.

### 12.6 Quality gates to submit
- CI green; coverage ≥ 70% on `offramp/` core.
- Replay demo completes in < 90 s.
- Evidence table numbers reproducible from `evals/results/*.json`.
- No secrets in repo (`gitleaks` in CI).

---

## 13. Demo Video Script (≤ 3:00)

*As shipped: superseded by [VIDEO.md](VIDEO.md), which uses only the recorded runs' numbers.*

| Time | Visual | Audio |
|---|---|---|
| 0:00–0:20 | Invoice-style closed-API bill; headline "OpenAI Evals read-only Oct 31" | Problem: spend + no proof open models are good enough |
| 0:20–1:30 | Live run on dashboard; Lightning fails JSON extraction; Ultra rewrites adapter; rerun passes | "Nemotron 3 Ultra on Nebius Token Factory rewrites the adapter…" |
| 1:30–2:30 | PR opens; evidence table; zoom on **NOT MIGRATED 71% < 95%** row | Honesty = trust; cheapest-passing tier routing |
| 2:30–3:00 | Who it's for; Nebius components used; roadmap (LoRA on TF) | CTA |

Rules: only measured numbers; say "Token Factory" and "Nemotron" out loud; no copyrighted music.

---

## 14. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `json_schema` response_format unsupported on some tiers | Med | JSON-mode + Pydantic + repair retry; adapter records which path used |
| Serverless Jobs onboarding friction | Med | Local worker fallback; mention attempt + feedback in submission |
| Teammate can't provision TF billing | Low-Med | Check Day 1; one shared key server-side |
| Ultra latency slows demo | Med | Replay mode for video; live mode only for 1 site |
| Scanner misses dynamic prompts | Med | Flag + manual fixture capture; show as "needs runtime capture" in UI |
| Demo repo looks toy | Med | Realistic domain (support-ops SaaS), real tests, 6 distinct contracts |
| Judges see it as PR bot | Med | Positioning + heatmap + honest row front and centre |

---

## 15. Submission Checklist

- [ ] Runtime call to Token Factory (no HF/OpenRouter); Nemotron 3 Ultra/Super/3.5 Lightning used
- [ ] Track: Coding & Agentic Engineering
- [ ] Description names Token Factory + Nemotron; Built With updated
- [ ] Public demo URL (replay works without keys) live through 15 Dec
- [ ] YouTube video < 3:00, public, tools named on audio
- [ ] Public GitHub repo, MIT license visible, README with setup + "how Nemotron/Token Factory used"
- [ ] No keys in repo or history (`gitleaks`)
- [ ] Feedback section: Token Factory, Serverless Jobs, Nemotron — specific, incl. what broke
- [ ] Optional: Tavily runtime call verified
- [ ] Team Representative set; all members eligible; submitted by 29 Oct

---

## 16. Open Questions (resolve by Day 2)

- Exact TF model ids and whether `json_schema` response_format is supported per tier → verify in TF catalogue Day 1.
- Serverless Jobs API availability on hackathon credits → test Day 1; decide fallback by Day 2.
- Demo repo hosted under team org vs personal → team org (PR must be visible to judges).
- Threshold defaults (95% / 90%) → confirm against baseline noise (run baseline twice, measure self-agreement).

---

## 17. Repo Layout

```
offramp/
  api/            FastAPI orchestrator, SSE, state machine
  agents/         scanner.py case_gen.py adapter.py judge.py iterator.py verdict.py patcher.py pr_writer.py
  evals/          scorers/ runner.py serverless_job/ pricing.yaml
  tf/             token_factory client, cache, tiers.py
  web/            Next.js dashboard
  demo-app/       (separate repo) offramp-demo-app
  tests/          unit/ agents/ harness/ integration/ e2e/
  docs/           PRD.md VIDEO.md FEEDBACK.md
```
