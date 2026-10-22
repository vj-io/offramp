"""Verdict Agent (PRD §6.7) and the per-site routing loop (offramp/evals/route.py, PRD §6.6).

Verdict: rule-based decision, cost estimate (GC21), closed reference row (GC27c) and the Super sentence, with
FakeLLM and then cassettes recorded live from Token Factory. Routing: stub runners and a stub Iterator.
"""

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import openai
import pytest

from offramp.agents.pr_writer import PRContext, render_pr
from offramp.agents.translate import UnresolvedCallSite
from offramp.agents.verdict import (
    VERDICT_MAX_TOKENS,
    VERDICT_ROLE,
    Evidence,
    VerdictReason,
    closed_reference_parity,
    decide,
    pricing_yaml_price,
)
from offramp.evals.errors import MissingReference, ProviderError
from offramp.evals.route import SiteRoute, route_site, secure_holdouts
from offramp.evals.runner import BudgetStop
from offramp.evals.scorers.aggregate import CANDIDATE_ERROR, NOT_JSON, site_pass
from offramp.integrations.tavily import SourceModelInfo
from offramp.schemas import (
    Adapter,
    AdapterSpec,
    BaselineResult,
    CallSite,
    CandidateResult,
    CaseScore,
    Contract,
    EvalCase,
    ModelOutput,
    PriceEvidence,
    PromptTemplate,
    RunConfig,
    Tier,
    TierMetrics,
    TierResult,
)
from offramp.tf import Budget, BudgetExceeded, CacheMiss, FakeLLM, TFClient
from offramp.tf.tiers import get_tier

CASSETTES = Path(__file__).parents[1] / "fixtures/cassettes/verdict"
NOW = datetime(2026, 10, 19, tzinfo=UTC)
CLAUDE = "claude-sonnet-5.5 (via Claude Code; not the repo's source model)"
OPENAI_PRICING = "https://developers.openai.com/api/docs/pricing"

SITE = CallSite(
    id="cs_01",
    file="helphub/inbox/triage.py",
    lines=[30, 41],
    sdk="openai",
    call="client.chat.completions.create",
    function="route_ticket",
    source_model="gpt-4o-mini",
    prompt_source="fstring",
    template=PromptTemplate(
        system="Route the ticket.", user="Subject: {subject}\n\n{body}", variables=["subject", "body"]
    ),
    contract=Contract(task_type="classification", output="label", labels=["billing", "shipping", "general"]),
)
REASONING = SITE.model_copy(
    update={
        "id": "cs_05",
        "file": "helphub/refunds/plan.py",
        "function": "plan_refund",
        "source_model": "o3",
        "contract": Contract(task_type="reasoning", output="text", reasoning_heavy=True),
    }
)
JSON_SITE = SITE.model_copy(
    update={
        "id": "cs_03",
        "file": "helphub/orders/extract.py",
        "function": "extract_order",
        "sdk": "anthropic",
        "source_model": "claude-sonnet-4-5",
        "contract": Contract(task_type="json_extraction", output="json"),
    }
)
FREE_TEXT = SITE.model_copy(
    update={
        "id": "cs_06",
        "file": "helphub/summaries/summarize.py",
        "function": "summarize_thread",
        "sdk": "anthropic",
        "source_model": "claude-haiku-4-5",
        "contract": Contract(task_type="free_text", output="text"),
    }
)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    for name in [f"OFFRAMP_MODEL_{t.name}" for t in Tier] + ["LANGSMITH_TRACING", "OFFRAMP_DATA_DIR"]:
        monkeypatch.delenv(name, raising=False)


def gc10(case_id: str) -> str:
    """Global Constraint 10, restated: the tests do not use the code under test to decide splits."""
    return "holdout" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100 < 30 else "dev"


def price(model: str) -> SourceModelInfo:
    return SourceModelInfo(
        model=model,
        input_usd_per_m=0.15,
        output_usd_per_m=0.60,
        price_source_url=OPENAI_PRICING,
        fetched_at=NOW,
        via="tavily",
    )


def scores(passes: list[bool], prefix: str = "cs_01", category: str = "wrong_label") -> list[CaseScore]:
    return [
        CaseScore(
            case_id=f"{prefix}:{i:03d}", passed=p, score=float(p), error_category=None if p else category
        )
        for i, p in enumerate(passes, 1)
    ]


def result(
    site: CallSite, tier: Tier, split: str, case_scores: list[CaseScore], version: int = 0
) -> TierResult:
    passed, score = site_pass(site.contract.task_type, case_scores, RunConfig(repo_url="x"), split=split)
    return TierResult(
        site_id=site.id,
        tier=tier,
        adapter_version=version,
        split=split,
        passed=passed,
        score=score,
        metrics=TierMetrics(
            p50_ms=700, p95_ms=1400, tokens_in=11600, tokens_out=87, cost_per_1k_calls_usd=0.0318
        ),
        failing_case_ids=[s.case_id for s in case_scores if not s.passed],
        case_scores=case_scores,
    )


def calls(tier: Tier, ok: int, tin: int, tout: int, failed: int = 0) -> list[CandidateResult]:
    good = [
        CandidateResult(
            case_id=f"c{i}",
            tier=tier,
            adapter_version=0,
            output=ModelOutput(text="billing"),
            latency_ms=700,
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=0.0,
        )
        for i in range(ok)
    ]
    bad = [
        CandidateResult(
            case_id=f"f{i}",
            tier=tier,
            adapter_version=0,
            output=None,
            error="APITimeoutError",
            latency_ms=120_000,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
        for i in range(failed)
    ]
    return good + bad


def evidence(
    site=SITE, tier=Tier.LIGHTNING, dev=None, holdout=None, n_calls=(29, 400, 3, 1), iterations=0, changes=()
):
    dev = dev or result(site, tier, "dev", scores([True] * 20))
    holdout = holdout or result(site, tier, "holdout", scores([True] * 29 + [False]))
    return Evidence(
        tier=tier,
        dev=dev,
        holdout=holdout,
        calls=calls(tier, *n_calls),
        iterations=iterations,
        changes=list(changes),
    )


def reason_llm(*replies) -> FakeLLM:
    return FakeLLM({VERDICT_ROLE: [r if not isinstance(r, str) else {"reason": r} for r in replies]})


# ---------------------------------------------------------------- the rule and the numbers (GC9, GC19, GC21)

COST_MIGRATE = "cost per 1k calls $0.0618 before (est.), $0.0318 after"  # evidence(): 0.0618 est., 0.0318
COST_KEEP = "cost per 1k calls $0.0618 before (est.)"
RULE_MIGRATE = (
    f"Lightning: parity vs reference 96.7% ≥ 95% on holdout after 0 adapter iterations; {COST_MIGRATE}."
)
GOOD = "Lightning returns the same queue labels as the references on the held-out tickets."


def keep(site, tier, holdout, **kw):
    return decide(
        site,
        evidence(site, tier, holdout=holdout, **kw),
        reason_llm(BudgetExceeded("x")),
        RunConfig(repo_url="x"),
        price_lookup=price,
    )


def test_migrate_fills_tier_parity_and_cost_from_measured_calls():
    v = decide(SITE, evidence(), reason_llm(GOOD), RunConfig(repo_url="x"), price_lookup=price)
    assert (v.site_id, v.decision, v.tier) == ("cs_01", "MIGRATE", Tier.LIGHTNING)
    assert v.model == get_tier(Tier.LIGHTNING).model_id
    assert v.parity_pct == 96.7  # 29/30 on holdout
    assert v.best_tier is None and v.best_score is None and v.iterations == 0
    assert v.latency_delta_ms is None  # "before" latency was never measured (GC21)
    assert v.cost_after_per_1k == 0.0318  # the holdout run's measured $/1k calls
    # 1000 x (400 x $0.15 + 3 x $0.60) / 1M: means over the 29 calls that returned (the failed one: no usage)
    assert v.cost_before_per_1k == pytest.approx(0.0618)
    assert v.sunset_risk is None and v.closed_reference_model is None
    # Super's qualitative sentence, then the code-built clause that carries every number
    assert v.reason == f"{GOOD} {RULE_MIGRATE}"


def test_cost_note_is_task17s_format_plus_the_calls_it_averages():
    v = decide(SITE, evidence(), reason_llm(GOOD), RunConfig(repo_url="x"), price_lookup=price)
    assert (
        v.cost_before_note
        == price("gpt-4o-mini").cost_before_note() + " (mean of 29 lightning holdout calls)"
    )
    assert v.cost_before_note.startswith(
        "est.: gpt-4o-mini $0.15/$0.60 per 1M (developers.openai.com/api/docs/"
    )
    assert "fetched via Tavily 2026-10-19" in v.cost_before_note


def test_a_lookups_pricing_yaml_fallback_keeps_its_reason():
    def lookup(model):
        return price(model).model_copy(
            update={"via": "pricing.yaml", "note": "fallback: Tavily price not used (no row for this model)"}
        )

    v = decide(SITE, evidence(), reason_llm(GOOD), RunConfig(repo_url="x"), price_lookup=lookup)
    assert "pricing.yaml as of 2026-10-19; fallback: Tavily price not used (no row for this model)" in (
        v.cost_before_note
    )


def test_the_default_lookup_is_task17s(monkeypatch):
    asked = []
    monkeypatch.setattr("offramp.agents.verdict.lookup_source_model", lambda m: asked.append(m) or price(m))
    decide(SITE, evidence(), reason_llm(GOOD), RunConfig(repo_url="x"))
    assert asked == ["gpt-4o-mini"]


def test_super_gets_qualitative_facts_and_no_numbers():
    llm = reason_llm(GOOD)
    decide(
        SITE,
        evidence(changes=["Pinned the reply to the bare queue name"]),
        llm,
        RunConfig(repo_url="x"),
        closed_parity=(CLAUDE, 88.0),
        price_lookup=price,
    )
    [call] = llm.calls
    assert (call["role"], call["tier"], call["reasoning"]) == (VERDICT_ROLE, Tier.SUPER, "off")
    assert call["response_model"] is VerdictReason and call["max_tokens"] == VERDICT_MAX_TOKENS
    text = call["messages"][1]["content"]
    assert not any(ch.isdigit() for ch in text)  # nothing to misattribute
    facts = json.loads(text)
    assert facts["decision"] == "MIGRATE" and facts["tier"] == "Lightning"
    # recorded live: told "cheaper tiers did not" for Lightning, Super wrote "lower-cost tiers failed"
    assert facts["outcome"].endswith("it is the cheapest tier") and "cheaper tiers" not in facts["outcome"]
    assert facts["adapter_changes"] == ["Pinned the reply to the bare queue name"]
    assert facts["call_site"] == {
        "file": "helphub/inbox/triage.py",
        "function": "route_ticket",
        "task_type": "classification",
    }


def test_keep_closed_reason_names_parity_and_threshold_and_never_rounds_up():
    holdout = result(
        REASONING, Tier.ULTRA, "holdout", scores([True] * 27 + [False] * 4, "cs_05", "wrong_answer")
    )
    dev = result(REASONING, Tier.ULTRA, "dev", scores([True] * 15 + [False] * 5, "cs_05", "wrong_answer"))
    v = decide(
        REASONING,
        evidence(REASONING, Tier.ULTRA, dev, holdout, iterations=3),
        reason_llm(BudgetExceeded("x")),
        RunConfig(repo_url="x"),
        price_lookup=price,
    )
    assert (v.decision, v.tier, v.model) == ("KEEP_CLOSED", None, None)
    assert (v.best_tier, v.best_score, v.iterations) == (Tier.ULTRA, 27 / 31, 3)
    assert v.parity_pct == 87.0  # 87.096...: floored, so a failing parity never displays at the threshold
    # reasoning: 90%; KEEP_CLOSED shows no "after" cost, the site stays where it is
    assert v.reason == f"parity vs reference 87.0% < 90% on Ultra after 3 adapter iterations; {COST_KEEP}."
    assert v.cost_after_per_1k is None and v.cost_before_per_1k == pytest.approx(0.0618)


def test_floored_parity_keeps_float_noise_from_dropping_a_tenth():
    holdout = result(
        REASONING, Tier.ULTRA, "holdout", scores([True] * 23 + [False] * 17, "cs_05", "wrong_answer")
    )
    assert 100 * holdout.score == 57.49999999999999  # 23/40, found by search: floor alone would give 57.4
    v = keep(REASONING, Tier.ULTRA, holdout)
    assert v.parity_pct == 57.5 and v.reason.startswith("parity vs reference 57.5% < 90%")


def test_threshold_in_the_reason_is_the_configured_one():
    holdout = result(SITE, Tier.ULTRA, "holdout", scores([True] * 18 + [False] * 2))
    v = decide(
        SITE,
        evidence(SITE, Tier.ULTRA, holdout=holdout),
        reason_llm(BudgetExceeded("x")),
        RunConfig(repo_url="x", threshold=0.925),
        price_lookup=price,
    )
    assert v.decision == "KEEP_CLOSED" and v.reason.startswith("parity vs reference 90.0% < 92.5% on Ultra")


def test_keep_closed_when_dev_failed_even_if_holdout_passed():
    dev = result(SITE, Tier.SUPER, "dev", scores([True] * 18 + [False] * 2))
    holdout = result(SITE, Tier.SUPER, "holdout", scores([True] * 10))
    v = decide(
        SITE,
        evidence(SITE, Tier.SUPER, dev, holdout, iterations=3),
        reason_llm(BudgetExceeded("x")),
        RunConfig(repo_url="x"),
        price_lookup=price,
    )
    assert v.decision == "KEEP_CLOSED" and v.parity_pct == 100.0 and v.best_tier == Tier.SUPER
    assert v.reason == (
        "dev parity vs reference 90.0% < 95% (holdout 100.0%) on Super after 3 adapter iterations; "
        f"{COST_KEEP}."
    )
    assert v.failed_gate == "dev"


@pytest.mark.parametrize(
    "holdout,gate",
    [
        (result(SITE, Tier.SUPER, "holdout", scores([True] * 7 + [False] * 3)), "threshold"),
        (
            result(FREE_TEXT, Tier.SUPER, "holdout", [CaseScore(case_id="h", passed=False, score=3.5)]),
            "judge_mean",
        ),
        (
            result(
                FREE_TEXT,
                Tier.SUPER,
                "holdout",
                [CaseScore(case_id=f"h{i}", passed=i > 0, score=[2.5, 5, 5, 5][i]) for i in range(4)],
            ),
            "case_floor",
        ),
        (result(SITE, Tier.SUPER, "holdout", scores([True] * 10)), None),  # migrated: nothing missed
    ],
)
def test_the_verdict_records_the_gate_its_holdout_missed(holdout, gate):
    site = FREE_TEXT if holdout.site_id == FREE_TEXT.id else SITE
    passing = [CaseScore(case_id="d", passed=True, score=5.0 if site is FREE_TEXT else 1.0)]
    v = keep(site, Tier.SUPER, holdout, dev=result(site, Tier.SUPER, "dev", passing))
    assert v.failed_gate == gate


def test_json_extraction_fails_on_validity_with_parity_above_threshold():
    rows = [CaseScore(case_id=f"cs_03:{i:03d}", passed=True, score=1.0) for i in range(19)]
    rows.append(CaseScore(case_id="cs_03:099", passed=False, score=0.0, error_category="schema_violation"))
    v = keep(JSON_SITE, Tier.SUPER, result(JSON_SITE, Tier.SUPER, "holdout", rows))
    assert v.decision == "KEEP_CLOSED" and v.parity_pct == 95.0
    assert v.reason.startswith(
        "parity vs reference 95.0% but 1/20 outputs not schema-valid (100% required) on Super"
    )


def test_json_extraction_fails_on_the_share_of_passing_cases():
    rows = [CaseScore(case_id=f"cs_03:{i:03d}", passed=True, score=1.0) for i in range(18)]
    rows += [
        CaseScore(case_id=f"cs_03:{i:03d}", passed=False, score=0.9, error_category="wrong_value")
        for i in (98, 99)
    ]
    v = keep(JSON_SITE, Tier.SUPER, result(JSON_SITE, Tier.SUPER, "holdout", rows))
    assert v.decision == "KEEP_CLOSED" and v.parity_pct == 99.0
    assert v.reason.startswith("parity vs reference 99.0% but only 90.0% of cases pass (< 95%) on Super")


def test_free_text_reason_names_the_judge_mean_and_parity_is_the_passing_share():
    rows = [CaseScore(case_id=f"cs_06:{i:03d}", passed=s >= 4, score=s) for i, s in enumerate([5, 4, 3, 2])]
    v = keep(FREE_TEXT, Tier.ULTRA, result(FREE_TEXT, Tier.ULTRA, "holdout", rows))
    assert v.decision == "KEEP_CLOSED" and v.parity_pct == 50.0 and v.best_score == 3.5
    assert v.reason.startswith("judge mean 3.50 < 4.0 (parity vs reference 50.0%) on Ultra")


def test_a_failing_judge_mean_is_floored_like_parity():
    rows = [CaseScore(case_id=f"cs_06:{i:03d}", passed=True, score=4.0) for i in range(249)]
    rows.append(CaseScore(case_id="cs_06:999", passed=False, score=3.0))
    holdout = result(FREE_TEXT, Tier.ULTRA, "holdout", rows)
    assert holdout.score == pytest.approx(3.996) and not holdout.passed  # f"{3.996:.2f}" would print 4.00
    assert keep(FREE_TEXT, Tier.ULTRA, holdout).reason.startswith("judge mean 3.99 < 4.0")


def test_free_text_below_the_reference_mean_margin_says_so():
    rows = [CaseScore(case_id=f"cs_06:{i:03d}", passed=True, score=4.0) for i in range(4)]
    holdout = result(FREE_TEXT, Tier.SUPER, "holdout", rows).model_copy(update={"passed": False})
    v = keep(FREE_TEXT, Tier.SUPER, holdout)
    assert v.reason.startswith("judge mean 4.00 more than 0.3 below the references' judge mean")


def test_a_free_text_case_below_the_per_case_floor_keeps_the_site_and_says_which():
    rows = [CaseScore(case_id=f"cs_06:{i:03d}", passed=True, score=5.0) for i in range(7)]
    rows.append(CaseScore(case_id="cs_06:015", passed=False, score=2.5))
    holdout = result(FREE_TEXT, Tier.LIGHTNING, "holdout", rows)
    assert holdout.score == pytest.approx(4.6875) and not holdout.passed

    v = keep(FREE_TEXT, Tier.LIGHTNING, holdout)

    assert v.decision == "KEEP_CLOSED"
    assert v.reason.startswith(
        "judge mean 4.68 ≥ 4.0, but holdout case cs_06:015 scored 2.50 < 3.0 (parity vs reference 87.5%) on "
        "Lightning"
    )


def test_a_failed_call_below_the_floor_is_named_as_a_failed_call_not_a_judged_score():
    rows = [CaseScore(case_id=f"cs_06:{i:03d}", passed=True, score=5.0) for i in range(7)]
    rows.append(CaseScore(case_id="cs_06:015", passed=False, score=1.0, error_category=CANDIDATE_ERROR))

    v = keep(FREE_TEXT, Tier.LIGHTNING, result(FREE_TEXT, Tier.LIGHTNING, "holdout", rows))

    assert "but holdout case cs_06:015 had no answer (its call failed) (parity" in v.reason


def test_a_reply_the_code_cannot_json_loads_below_the_floor_is_named_as_not_json():
    rows = [CaseScore(case_id=f"cs_06:{i:03d}", passed=True, score=5.0) for i in range(7)]
    rows.append(CaseScore(case_id="cs_06:015", passed=False, score=1.0, error_category=NOT_JSON))

    v = keep(FREE_TEXT, Tier.LIGHTNING, result(FREE_TEXT, Tier.LIGHTNING, "holdout", rows))

    assert "but holdout case cs_06:015 was not JSON, which the calling code parses (parity" in v.reason


def test_references_the_judge_scored_below_3_are_flagged_on_the_verdict_and_kept():
    rows = [
        CaseScore(case_id="cs_06:001", passed=True, score=5.0, reference_score=4.5),
        CaseScore(case_id="cs_06:020", passed=True, score=4.5, reference_score=1.0),
        CaseScore(case_id="cs_06:021", passed=True, score=4.5, reference_score=3.0),  # at 3: not weak
        CaseScore(case_id="cs_06:022", passed=True, score=4.5),  # judged without a reference
    ]
    holdout = result(FREE_TEXT, Tier.SUPER, "holdout", rows)
    dev = result(FREE_TEXT, Tier.SUPER, "dev", [CaseScore(case_id="cs_06:900", passed=True, score=4.5)])

    v = decide(FREE_TEXT, evidence(FREE_TEXT, Tier.SUPER, dev, holdout), reason_llm(BudgetExceeded("x")),
               RunConfig(repo_url="x"), price_lookup=price)  # fmt: skip

    assert v.decision == "MIGRATE" and v.parity_pct == 100.0  # shown, never excluded
    assert v.weak_references == {"cs_06:020": 1.0}


def test_migrate_free_text_parity_is_the_share_of_passing_cases():
    rows = [
        CaseScore(case_id=f"cs_06:{i:03d}", passed=s >= 4, score=s) for i, s in enumerate([5, 5, 4, 3.67])
    ]
    holdout = result(FREE_TEXT, Tier.SUPER, "holdout", rows)
    dev = result(FREE_TEXT, Tier.SUPER, "dev", [CaseScore(case_id="cs_06:900", passed=True, score=4.5)])
    v = decide(
        FREE_TEXT,
        evidence(FREE_TEXT, Tier.SUPER, dev, holdout),
        reason_llm(BudgetExceeded("x")),
        RunConfig(repo_url="x"),
        price_lookup=price,
    )
    assert v.decision == "MIGRATE" and v.parity_pct == 75.0
    assert v.reason == (
        "Super: judge mean 4.42 ≥ 4.0 (parity vs reference 75.0%) on holdout after 0 adapter iterations; "
        f"{COST_MIGRATE}."
    )
    # n, the mean and the floor case, and how sure 3 of 4 passing is
    assert (v.holdout_cases, v.holdout_passed, v.parity_lower_pct) == (4, 3, 30.0)  # Wilson 3/4: 0.3006
    assert (v.judge_mean, v.judge_min) == (4.42, 3.67)


def test_parity_carries_its_case_count_and_wilson_lower_bound():
    v = decide(SITE, evidence(), reason_llm(GOOD), RunConfig(repo_url="x"), price_lookup=price)
    assert (v.holdout_cases, v.holdout_passed) == (30, 29)
    assert v.parity_lower_pct == 83.3  # Wilson 95% of 29/30 = 0.8333..., rounded down to 0.1
    assert v.judge_mean is None and v.judge_min is None  # not free text
    perfect = result(SITE, Tier.LIGHTNING, "holdout", scores([True] * 3))
    assert keep(SITE, Tier.LIGHTNING, perfect).parity_lower_pct == 43.8  # 100% of 3 is only >= 43.8%


def test_a_failing_free_text_runs_judge_scores_are_rounded_down():
    rows = [CaseScore(case_id=f"cs_06:{i:03d}", passed=True, score=4.0) for i in range(249)]
    rows.append(CaseScore(case_id="cs_06:999", passed=False, score=2.999))
    v = keep(FREE_TEXT, Tier.ULTRA, result(FREE_TEXT, Tier.ULTRA, "holdout", rows))
    assert (v.judge_mean, v.judge_min) == (3.99, 2.99)


def test_the_clause_carries_the_closed_reference_row_and_an_unestimated_cost():
    v = decide(
        SITE.model_copy(update={"source_model": None}),
        evidence(),
        reason_llm(BudgetExceeded("x")),
        RunConfig(repo_url="x"),
        closed_parity=(CLAUDE, 88.0),
        price_lookup=price,
    )
    assert v.reason == (
        "Lightning: parity vs reference 96.7% ≥ 95% on holdout after 0 adapter iterations; cost per 1k calls "
        f"$0.0318 after (before not estimated); closed reference {CLAUDE} 88.0% on the same references."
    )


# ------------------------------------------------------------ the Super sentence is checked, never trusted


def test_a_keep_closed_sentence_without_numbers_is_used_before_the_clause():
    holdout = result(
        REASONING, Tier.ULTRA, "holdout", scores([True] * 7 + [False] * 3, "cs_05", "wrong_answer")
    )
    sentence = (
        "Refund planning stays on the current model: no Nemotron tier applied the policy arithmetic reliably"
    )
    v = decide(
        REASONING,
        evidence(REASONING, Tier.ULTRA, holdout=holdout, iterations=3),
        reason_llm(sentence),
        RunConfig(repo_url="x"),
        price_lookup=price,
    )
    assert v.reason == (
        f"{sentence}. parity vs reference 70.0% < 90% on Ultra after 3 adapter iterations; {COST_KEEP}."
    )


@pytest.mark.parametrize(
    "sentence",
    [
        # the review's probe: a real number (the closed row's 88.0) attached to the wrong quantity
        "Lightning reached 88.0% parity on held-out cases, so the call moves off gpt-4o-mini.",
        "Lightning matched the references on 95% of held-out cases.",
        "Lightning is 12x cheaper and matches the references.",
        "Lightning matched the references after three rounds of prompt changes.",  # number words too
        "Lightning matched the references on most cases.\nIt is cheaper.",  # not one sentence
        "",
    ],
)
def test_any_number_in_the_sentence_means_the_rule_sentence_alone(sentence):
    v = decide(
        SITE,
        evidence(),
        reason_llm(sentence),
        RunConfig(repo_url="x"),
        closed_parity=(CLAUDE, 88.0),
        price_lookup=price,
    )
    assert v.reason == (
        "Lightning: parity vs reference 96.7% ≥ 95% on holdout after 0 adapter iterations; "
        f"{COST_MIGRATE}; closed reference {CLAUDE} 88.0% on the same references."
    )


@pytest.mark.parametrize(
    "error",
    [
        BudgetExceeded("token cap reached"),
        openai.APIConnectionError(request=httpx.Request("POST", "https://tf")),
    ],
)
def test_a_failed_reason_call_falls_back_to_the_rule_sentence(error):
    v = decide(SITE, evidence(), reason_llm(error), RunConfig(repo_url="x"), price_lookup=price)
    assert v.decision == "MIGRATE" and v.reason == RULE_MIGRATE


def test_a_replay_cache_miss_is_not_hidden():
    with pytest.raises(CacheMiss):
        decide(
            SITE,
            evidence(),
            reason_llm(CacheMiss("no cassette")),
            RunConfig(repo_url="x"),
            price_lookup=price,
        )


# -------------------------------------------------------- cost before: Tavily lookup, pricing.yaml fallback


def test_sunset_risk_comes_from_the_lookups_deprecation_notice():
    def lookup(model):
        return price(model).model_copy(
            update={
                "deprecation": "FIXTURE: gpt-4o-mini retires on 2026-11-30",
                "deprecation_url": "https://example.com/dep",
            }
        )

    v = decide(
        SITE, evidence(), reason_llm(BudgetExceeded("x")), RunConfig(repo_url="x"), price_lookup=lookup
    )
    assert v.sunset_risk == "FIXTURE: gpt-4o-mini retires on 2026-11-30 (https://example.com/dep)"
    assert v.retires_on == date(2026, 11, 30)  # read from the notice, never written in by hand


def test_the_lookups_evidence_travels_on_the_verdict():
    def lookup(model):
        return price(model).model_copy(update={"query": "gpt-4o-mini pricing", "price_quote": "| $0.15 |"})

    v = decide(SITE, evidence(), reason_llm(GOOD), RunConfig(repo_url="x"), price_lookup=lookup)
    assert v.price_evidence == PriceEvidence(
        method="search", query="gpt-4o-mini pricing", url=OPENAI_PRICING, quote="| $0.15 |", date="2026-10-19"
    )
    assert v.retires_on is None
    offline = decide(
        SITE, evidence(), reason_llm(GOOD), RunConfig(repo_url="x"), price_lookup=pricing_yaml_price
    )
    assert offline.price_evidence.method == "yaml fallback" and offline.price_evidence.query is None


def test_the_offline_lookup_reads_pricing_yaml_and_says_so():
    p = pricing_yaml_price("gpt-4o-mini-2024-07-18")
    assert (p.input_usd_per_m, p.output_usd_per_m, p.via) == (0.15, 0.60, "pricing.yaml")
    v = decide(
        SITE,
        evidence(),
        reason_llm(BudgetExceeded("x")),
        RunConfig(repo_url="x"),
        price_lookup=pricing_yaml_price,
    )
    assert v.cost_before_per_1k == pytest.approx(0.0618)
    assert "pricing.yaml as of 2026-10-19; fallback: offline run, Tavily not asked" in v.cost_before_note
    assert OPENAI_PRICING.split("://")[1] in v.cost_before_note and v.sunset_risk is None


@pytest.mark.parametrize("model", ["acme-llm-9", None])
def test_no_known_price_means_no_cost_before(model):
    site = SITE.model_copy(update={"source_model": model})
    v = decide(
        site,
        evidence(site),
        reason_llm(BudgetExceeded("x")),
        RunConfig(repo_url="x"),
        price_lookup=pricing_yaml_price,
    )
    assert v.cost_before_per_1k is None and v.cost_after_per_1k == 0.0318
    assert "not estimated" in v.cost_before_note


def test_no_returned_call_means_no_cost_before():
    v = decide(
        SITE,
        evidence(n_calls=(0, 0, 0, 3)),
        reason_llm(BudgetExceeded("x")),
        RunConfig(repo_url="x"),
        price_lookup=price,
    )
    assert v.cost_before_per_1k is None and "not estimated" in v.cost_before_note


# ---------------------------------------------------------------- closed reference row (GC27c)


def case(cid: str, site_id: str = "cs_01") -> EvalCase:
    return EvalCase(
        id=cid,
        site_id=site_id,
        input_messages=[{"role": "user", "content": cid}],
        expected_kind="label",
        split=gc10(cid),
    )


def ref(cid: str, label: str, source: str = "reference", model: str = "reference:seed") -> BaselineResult:
    return BaselineResult(
        case_id=cid, source=source, model=model, output=ModelOutput(text=label), recorded_at=NOW
    )


def split_ids(split: str, n: int, prefix: str = "cs_01") -> list[str]:
    return [c for c in (f"{prefix}:{i:03d}" for i in range(1, 500)) if gc10(c) == split][:n]


def test_closed_reference_parity_is_scored_on_holdout_with_the_same_scorer():
    hold, dev = split_ids("holdout", 4), split_ids("dev", 2)
    cases = [case(c) for c in hold + dev]
    refs = {c: ref(c, "billing") for c in hold + dev}
    closed = {
        c: ref(c, "Billing.", "closed_model", CLAUDE) for c in hold[:3] + dev
    }  # label normalisation applies
    closed[hold[3]] = ref(hold[3], "shipping", "closed_model", CLAUDE)
    closed[dev[0]] = ref(dev[0], "general", "closed_model", CLAUDE)  # dev cases do not count
    assert closed_reference_parity(SITE, cases, refs, closed, RunConfig(repo_url="x")) == (CLAUDE, 75.0)


def test_closed_rows_standing_in_for_missing_references_are_not_scored_against_themselves():
    hold = split_ids("holdout", 3)
    closed = {c: ref(c, "billing", "closed_model", CLAUDE) for c in hold}
    refs = {hold[0]: ref(hold[0], "shipping"), hold[1]: closed[hold[1]], hold[2]: closed[hold[2]]}
    assert closed_reference_parity(SITE, [case(c) for c in hold], refs, closed, RunConfig(repo_url="x")) == (
        CLAUDE,
        0.0,
    )


def test_no_closed_reference_row_for_free_text_or_without_closed_outputs():
    hold = split_ids("holdout", 2, "cs_06")
    closed = {c: ref(c, "A summary.", "closed_model", CLAUDE) for c in hold}
    cases = [case(c, "cs_06") for c in hold]
    assert closed_reference_parity(FREE_TEXT, cases, closed, closed, RunConfig(repo_url="x")) is None
    assert closed_reference_parity(SITE, cases, {}, {}, RunConfig(repo_url="x")) is None


def test_closed_reference_row_goes_into_the_verdict():
    v = decide(
        SITE,
        evidence(),
        reason_llm(BudgetExceeded("x")),
        RunConfig(repo_url="x"),
        closed_parity=(CLAUDE, 88.0),
        price_lookup=price,
    )
    assert (v.closed_reference_model, v.closed_reference_parity_pct) == (CLAUDE, 88.0)


# ---------------------------------------------------------------- routing loop (stub runner, stub Iterator)


ROUTE_CASES = [case(f"cs_01:{i:03d}") for i in range(1, 41)]
DEV_N = sum(c.split == "dev" for c in ROUTE_CASES)
HOLD_N = len(ROUTE_CASES) - DEV_N
V0 = Adapter(site_id="cs_01", version=0, spec=AdapterSpec(max_tokens=16), notes=["v0"])


class Runner:
    """Stub run_tier: the share of passing cases per (tier, split[, version]); records every call."""

    def __init__(self, table: dict):
        self.table, self.calls, self.reused = table, [], []

    def __call__(
        self,
        site,
        cases,
        baselines,
        adapter,
        tier,
        client,
        split,
        thresholds,
        *,
        return_candidates=False,
        scored=None,
    ):
        assert return_candidates is True
        self.calls.append((Tier(tier), split, adapter.version))
        if scored is not None:  # run_tier(scored=...): the calls are replayed, the scores are not redone
            self.reused.append((Tier(tier), split, adapter.version))
        outcome = self.table.get((tier, split, adapter.version), self.table.get((tier, split)))
        if isinstance(outcome, list):  # one outcome per call, in order
            outcome = outcome.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "budget":
            return BudgetStop(
                site.id, tier, split, "budget: token cap reached (1500001/1500000 in, 0/300000 out)"
            ), []
        selected = [c for c in cases if c.split == split]
        k = round(outcome * len(selected))
        judged = site.contract.task_type == "free_text"  # judge scores: 5 for a passing case, 2 otherwise
        rows = [
            CaseScore(
                case_id=c.id,
                passed=i < k,
                score=(5.0 if i < k else 2.0) if judged else float(i < k),
                error_category=None if i < k else "wrong_label",
            )
            for i, c in enumerate(selected)
        ]
        passed, score = site_pass(site.contract.task_type, rows, thresholds)
        res = TierResult(
            site_id=site.id,
            tier=tier,
            adapter_version=adapter.version,
            split=split,
            passed=passed,
            score=score,
            metrics=TierMetrics(
                p50_ms=700, p95_ms=900, tokens_in=1, tokens_out=1, cost_per_1k_calls_usd=0.05
            ),
            failing_case_ids=[r.case_id for r in rows if not r.passed],
            case_scores=rows,
        )
        out = [
            CandidateResult(
                case_id=c.id,
                tier=tier,
                adapter_version=adapter.version,
                output=ModelOutput(text="x"),
                latency_ms=700,
                input_tokens=400,
                output_tokens=3,
                cost_usd=0.0,
            )
            for c in selected
        ]
        return (scored or res), out


class StubIterator:
    def __init__(self, give_up_at: int | None = None, error: BaseException | None = None):
        self.calls, self.give_up_at, self.error = [], give_up_at, error

    def __call__(self, adapter, site, failing, tier, error_counts):
        self.calls.append((Tier(tier), adapter.version, [c.id for c in failing], dict(error_counts)))
        if self.error:
            raise self.error
        if self.give_up_at == adapter.version + 1:
            return None
        return Adapter(
            site_id=site.id,
            version=adapter.version + 1,
            tier=tier,
            spec=adapter.spec,
            change_log=[f"stub change {adapter.version + 1}"],
        )


def route(
    table, site=SITE, iterator=None, llm=None, config=None, events=None, cases=ROUTE_CASES, **kw
) -> tuple:
    runner, iterator = Runner(table), iterator or StubIterator()
    llm = llm or reason_llm(*[BudgetExceeded("rule sentence only")] * 2)
    out = route_site(
        site,
        cases,
        {c.id: ref(c.id, "billing") for c in cases},
        V0,
        llm,
        config or RunConfig(repo_url="x"),
        iterator=iterator,
        runner=runner,
        price_lookup=price,
        on_event=None if events is None else (lambda t, p: events.append((t, p))),
        **kw,
    )
    return out, runner, iterator


L, S, U = Tier.LIGHTNING, Tier.SUPER, Tier.ULTRA


def matrix_cells(table: dict, site=SITE, cases=ROUTE_CASES) -> dict:
    """The eval matrix's cells for `table`: v0 on every tier's dev cases, keyed as route_site's `prior`."""
    runner = Runner(dict(table))
    config = RunConfig(repo_url="x")
    return {
        (t, 0, "dev"): runner(site, cases, {}, V0, t, None, "dev", config, return_candidates=True)[0]
        for t in (L, S, U)
        if (t, "dev") in table
    }


def test_migrates_to_the_cheapest_tier_passing_dev_and_holdout_and_stops():
    out, runner, it = route({(L, "dev"): 1.0, (L, "holdout"): 1.0})
    assert isinstance(out, SiteRoute) and out.verdict.decision == "MIGRATE" and out.verdict.tier == L
    assert runner.calls == [(L, "dev", 0), (L, "holdout", 0)] and it.calls == []
    assert out.adapter == V0 and [(r.tier, r.split) for r in out.results] == [(L, "dev"), (L, "holdout")]


def test_dev_failure_iterates_up_to_max_iterations_then_the_next_tier_starts_from_v0():
    # Lightning's best adapter is v1, so a loop that carried the best adapter over would start Super at v1
    out, runner, it = route({(L, "dev"): 0.5, (L, "dev", 1): 0.6, (S, "dev"): 1.0, (S, "holdout"): 1.0})
    assert runner.calls == [
        (L, "dev", 0),
        (L, "dev", 1),
        (L, "dev", 2),
        (L, "dev", 3),
        (S, "dev", 0),
        (S, "holdout", 0),
    ]
    assert [(t, v) for t, v, _, _ in it.calls] == [(L, 0), (L, 1), (L, 2)]
    assert out.verdict.decision == "MIGRATE" and out.verdict.tier == S and out.verdict.iterations == 0


def test_iterations_count_only_the_chosen_tier_and_the_passing_adapter_is_returned():
    out, runner, it = route({(L, "dev"): 0.5, (S, "dev"): 0.5, (S, "dev", 2): 1.0, (S, "holdout"): 1.0})
    assert out.verdict.tier == S and out.verdict.iterations == 2
    assert (out.adapter.version, out.adapter.tier, out.adapter.change_log) == (2, S, ["stub change 2"])
    assert runner.calls[-1] == (S, "holdout", 2)


def test_iterator_gets_at_most_ten_failing_dev_cases_and_all_their_error_counts():
    out, runner, it = route(
        {(L, "dev"): 0.0, (S, "dev"): 1.0, (S, "holdout"): 1.0},
        config=RunConfig(repo_url="x", max_iterations=1),
    )
    [(tier, version, shown, counts)] = it.calls
    dev_ids = [c.id for c in ROUTE_CASES if c.split == "dev"]
    assert shown == dev_ids[:10] and counts == {"wrong_label": DEV_N}


def test_holdout_failure_moves_on_without_iterating():
    out, runner, it = route({(L, "dev"): 1.0, (L, "holdout"): 0.5, (S, "dev"): 1.0, (S, "holdout"): 1.0})
    assert runner.calls == [(L, "dev", 0), (L, "holdout", 0), (S, "dev", 0), (S, "holdout", 0)]
    assert it.calls == [] and out.verdict.tier == S


def test_all_tiers_failing_keeps_closed_with_the_best_dev_tier_measured_on_holdout():
    out, runner, it = route({(L, "dev"): 0.5, (S, "dev"): 0.8, (U, "dev"): 0.7, (S, "holdout"): 0.75})
    v = out.verdict
    assert runner.calls[-1] == (S, "holdout", 0)  # best = first adapter reaching the top dev score
    # the iteration count is the measured adapter's (v0: none), though 3 were tried on Super
    assert v.decision == "KEEP_CLOSED" and v.best_tier == S and v.iterations == 0
    assert (DEV_N, HOLD_N) == (25, 15) and v.parity_pct == 73.3  # 11/15 = 73.33...
    assert v.reason.startswith("parity vs reference 73.3% < 95% on Super after 0 adapter iterations;")
    assert out.adapter.version == 0 and len(it.calls) == 9


def test_a_free_text_site_whose_holdout_passed_but_dev_never_did_is_kept_for_dev_and_says_so():
    """Review T19 N2, through route.py's "no tier qualified" path: the best dev adapter is measured on
    holdout, holdout passes, the site stays closed, and the PR row states that cause and no other."""
    table = {(L, "dev"): 0.5, (S, "dev"): 0.5, (U, "dev"): 0.5, (L, "holdout"): 1.0}
    out, runner, _ = route(table, site=FREE_TEXT)
    v = out.verdict
    assert runner.calls[-1] == (L, "holdout", 0)
    assert (v.decision, v.best_tier, v.parity_pct, v.failed_gate) == ("KEEP_CLOSED", L, 100.0, "dev")
    assert v.reason.startswith("dev judge mean 3.44 < 4.0")  # 12 of 25 dev cases at 5, 13 at 2
    ctx = PRContext(run_id="r", repo="o/r", config=RunConfig(repo_url="x"), sites=[FREE_TEXT], verdicts=[v])
    row = next(ln for ln in render_pr(ctx).body.splitlines() if ln.startswith("| cs_06 "))
    assert (
        f"| 100.0% ({HOLD_N}/{HOLD_N}, ≥" in row
        and "judge mean 5/5, lowest case 5/5; holdout passed, dev did not |" in row
    )
    assert "references' judge mean" not in row and "< " not in row


def test_keep_closed_counts_the_iterations_and_changes_of_the_best_adapter_only():
    llm = FakeLLM({VERDICT_ROLE: [{"reason": "Ticket routing stays on the current model."}]})
    table = {(L, "dev"): 0.5, (S, "dev"): 0.6, (S, "dev", 2): 0.8, (U, "dev"): 0.5, (S, "holdout"): 0.6}
    out, runner, it = route(table, llm=llm)
    v = out.verdict
    assert runner.calls[-1] == (S, "holdout", 2) and out.adapter.version == 2
    assert v.decision == "KEEP_CLOSED" and v.best_tier == S and v.iterations == 2
    facts = json.loads(llm.calls[0]["messages"][1]["content"])
    assert facts["adapter_changes"] == ["stub change 1", "stub change 2"]  # not stub change 3


def test_keep_closed_reuses_a_holdout_result_already_measured_for_the_best_adapter():
    out, runner, it = route({(L, "dev"): 1.0, (L, "holdout"): 0.6, (S, "dev"): 0.6, (U, "dev"): 0.6})
    assert runner.calls.count((L, "holdout", 0)) == 1 and runner.calls[-1][1] == "dev"
    assert (
        out.verdict.decision == "KEEP_CLOSED"
        and out.verdict.best_tier == L
        and out.verdict.parity_pct == 60.0
    )


def test_max_iterations_zero_never_calls_the_iterator():
    out, runner, it = route(
        {(L, "dev"): 0.5, (S, "dev"): 0.5, (U, "dev"): 0.5, (L, "holdout"): 0.5},
        config=RunConfig(repo_url="x", max_iterations=0),
    )
    assert it.calls == [] and out.verdict.decision == "KEEP_CLOSED" and out.verdict.iterations == 0


def test_tiers_follow_the_config():
    out, runner, it = route({(S, "dev"): 1.0, (S, "holdout"): 1.0}, config=RunConfig(repo_url="x", tiers=[S]))
    assert runner.calls == [(S, "dev", 0), (S, "holdout", 0)]


def test_configured_tiers_run_cheapest_first_and_once():
    table = {(L, "dev"): 1.0, (L, "holdout"): 1.0, (U, "dev"): 1.0, (U, "holdout"): 1.0}
    out, runner, it = route(table, config=RunConfig(repo_url="x", tiers=[U, L, L]))
    assert runner.calls == [(L, "dev", 0), (L, "holdout", 0)] and out.verdict.tier == L
    table = {(S, "dev"): 0.5, (U, "dev"): 0.5, (S, "holdout"): 0.5}
    out, runner, it = route(table, config=RunConfig(repo_url="x", tiers=[U, S, U], max_iterations=0))
    assert [t for t, split, _ in runner.calls if split == "dev"] == [S, U]


def test_reasoning_sites_pass_at_the_reasoning_threshold():
    table = {
        (L, "dev"): 0.92,
        (L, "holdout"): 0.92,
        (S, "dev"): 0.92,
        (S, "holdout"): 0.92,
        (U, "dev"): 0.92,
        (U, "holdout"): 0.92,
    }
    reasoning_cases = [c.model_copy(update={"site_id": "cs_05"}) for c in ROUTE_CASES]
    assert route(table, site=REASONING, cases=reasoning_cases)[0].verdict.decision == "MIGRATE"
    assert route(table, site=SITE)[0].verdict.decision == "KEEP_CLOSED"


def test_a_budget_stop_keeps_the_site_closed_with_the_budget_reason():
    out, runner, it = route({(L, "dev"): 0.5, (L, "dev", 1): "budget"})
    assert out.verdict.decision == "KEEP_CLOSED" and out.verdict.reason.startswith(
        "budget: token cap reached"
    )
    assert runner.calls == [(L, "dev", 0), (L, "dev", 1)]  # nothing after the stop
    assert out.verdict.parity_pct is None and out.adapter is None
    assert out.verdict.failed_gate == "budget"


def test_a_budget_stop_on_the_final_holdout_run_keeps_the_site_closed():
    out, runner, it = route(
        {(L, "dev"): 0.5, (S, "dev"): 0.5, (U, "dev"): 0.5, (L, "holdout"): "budget"},
        config=RunConfig(repo_url="x", max_iterations=0),
    )
    assert out.verdict.decision == "KEEP_CLOSED" and out.verdict.reason.startswith("budget: ")


def test_the_iterator_hitting_the_ultra_cap_keeps_the_site_closed():
    out, runner, it = route(
        {(L, "dev"): 0.5}, iterator=StubIterator(error=BudgetExceeded("Ultra agent call cap reached (40)"))
    )
    assert out.verdict.decision == "KEEP_CLOSED"
    assert out.verdict.reason == (
        "budget: Ultra agent call cap reached (40); best reached: Lightning dev, parity vs reference 48.0% < "
        "95% after 0 adapter iterations (holdout not measured)"
    )


# ---------------------------------------------------------------- the eval matrix's cells (`prior`)

ULTRA_CAP = BudgetExceeded("Ultra agent call cap reached (40)")


def test_cells_the_matrix_measured_are_reused_not_scored_again():
    """Run 20261022-023459043-590031, cs_01: the routing loop re-judged the Lightning v0 outputs the matrix
    had judged (4.42, then 3.44) and spent 2 Ultra calls on it. The matrix's TierResult is now the cell's
    result; the runner only replays the calls (cached) for the outputs the Iterator is shown."""
    table = {(L, "dev"): 0.5, (L, "dev", 1): 1.0, (L, "holdout"): 1.0}
    prior = matrix_cells(table)
    out, runner, it = route(table, prior=prior)
    assert runner.reused == [(L, "dev", 0)] and out.results[0] is prior[L, 0, "dev"]
    assert runner.calls == [(L, "dev", 0), (L, "dev", 1), (L, "holdout", 1)]  # v1 and holdout are new cells
    assert out.verdict.decision == "MIGRATE" and out.verdict.tier == L and out.verdict.iterations == 1


def test_a_tier_that_passed_dev_in_the_matrix_runs_its_holdout_before_a_cheaper_tier_is_iterated():
    """cs_04 of that run: Super (4.91) and Ultra (4.87) passed dev in the matrix, Lightning (4.40) did not.
    The loop iterated Lightning first, the Ultra cap stopped it, and a site with a dev-passing tier ended a
    budget stop. Now the cheapest dev-passing tier's holdout runs before the first Iterator call."""
    table = {(L, "dev"): 0.5, (S, "dev"): 1.0, (U, "dev"): 1.0, (S, "holdout"): 1.0}
    out, runner, it = route(table, prior=matrix_cells(table))
    assert runner.calls == [
        (L, "dev", 0),
        (S, "holdout", 0),  # secured before the Iterator's first call
        (L, "dev", 1),
        (L, "dev", 2),
        (L, "dev", 3),
        (S, "dev", 0),  # reused from the matrix; its holdout is not run again
    ]
    assert out.verdict.decision == "MIGRATE" and out.verdict.tier == S and out.adapter == V0
    assert "not fully tried" not in out.verdict.reason  # every cheaper tier was tried in full


def test_a_budget_stop_after_a_secured_verdict_keeps_it_and_says_cheaper_tiers_were_not_fully_tried():
    table = {(L, "dev"): 0.5, (S, "dev"): 1.0, (U, "dev"): 1.0, (S, "holdout"): 1.0}
    llm = FakeLLM({VERDICT_ROLE: [{"reason": "Idea report angles stay correct on Super."}]})
    out, runner, it = route(table, prior=matrix_cells(table), iterator=StubIterator(error=ULTRA_CAP), llm=llm)
    v = out.verdict
    assert runner.calls == [(L, "dev", 0), (S, "holdout", 0)]
    assert (v.decision, v.tier, v.failed_gate, v.iterations) == ("MIGRATE", S, None, 0)
    assert v.reason.endswith("; Lightning not fully tried (budget: Ultra agent call cap reached (40)).")
    facts = json.loads(llm.calls[0]["messages"][1]["content"])
    assert facts["outcome"].endswith("cheaper tiers were not fully tried before the run's budget ran out")


def test_a_cheaper_tier_that_passes_after_a_verdict_was_secured_wins():
    table = {
        (L, "dev"): 0.5,
        (L, "dev", 1): 1.0,
        (L, "holdout"): 1.0,
        (S, "dev"): 1.0,
        (S, "holdout"): 1.0,
    }
    out, runner, it = route(table, prior=matrix_cells(table))
    assert runner.calls == [(L, "dev", 0), (S, "holdout", 0), (L, "dev", 1), (L, "holdout", 1)]
    assert out.verdict.decision == "MIGRATE" and out.verdict.tier == L and out.verdict.iterations == 1


def test_a_secured_holdout_that_fails_moves_to_the_next_dev_passing_tier():
    table = {(L, "dev"): 0.5, (S, "dev"): 1.0, (S, "holdout"): 0.5, (U, "dev"): 1.0, (U, "holdout"): 1.0}
    out, runner, it = route(
        table, prior=matrix_cells(table), config=RunConfig(repo_url="x", max_iterations=1)
    )
    assert runner.calls == [
        (L, "dev", 0),
        (S, "holdout", 0),
        (U, "holdout", 0),
        (L, "dev", 1),
        (S, "dev", 0),  # passed dev, failed holdout: moves on without iterating, as before
        (U, "dev", 0),
    ]
    assert out.verdict.decision == "MIGRATE" and out.verdict.tier == U


def test_the_holdouts_that_secure_a_verdict_can_run_for_every_site_before_any_site_iterates():
    """Routing goes site by site: in that run cs_01 (no tier passed dev) iterated first and spent the cap
    before cs_04 was reached. The pipeline runs secure_holdouts for every site before routing any; routing
    then reuses those holdout cells like the matrix's."""
    table = {(L, "dev"): 0.5, (S, "dev"): 1.0, (S, "holdout"): 0.5, (U, "dev"): 1.0, (U, "holdout"): 1.0}
    prior, runner = matrix_cells(table), Runner(table)
    refs = {c.id: ref(c.id, "billing") for c in ROUTE_CASES}
    got = secure_holdouts(
        SITE, ROUTE_CASES, refs, V0, FakeLLM(), RunConfig(repo_url="x"), prior, runner=runner
    )
    assert runner.calls == [(S, "holdout", 0), (U, "holdout", 0)]  # cheapest first, until one passes
    assert {k: r.passed for k, r in got.items()} == {(S, 0, "holdout"): False, (U, 0, "holdout"): True}

    out, runner, it = route(table, prior=prior | got, config=RunConfig(repo_url="x", max_iterations=1))
    assert runner.reused == [
        (L, "dev", 0),
        (S, "holdout", 0),
        (U, "holdout", 0),
        (S, "dev", 0),
        (U, "dev", 0),
    ]
    assert out.verdict.decision == "MIGRATE" and out.verdict.tier == U


@pytest.mark.parametrize(
    "table, config",
    [
        (
            {(L, "dev"): 1.0, (S, "dev"): 1.0},
            RunConfig(repo_url="x"),
        ),  # routing runs Lightning's holdout first
        ({(L, "dev"): 0.5, (S, "dev"): 0.5, (U, "dev"): 0.5}, RunConfig(repo_url="x")),  # nothing passed dev
        ({(L, "dev"): 0.5, (S, "dev"): 1.0}, RunConfig(repo_url="x", max_iterations=0)),  # nothing to iterate
    ],
)
def test_no_holdout_is_secured_where_routing_would_not_iterate_before_it(table, config):
    runner = Runner(table)
    refs = {c.id: ref(c.id, "billing") for c in ROUTE_CASES}
    assert (
        secure_holdouts(SITE, ROUTE_CASES, refs, V0, FakeLLM(), config, matrix_cells(table), runner=runner)
        == {}
    )
    assert runner.calls == []


def test_a_budget_stop_while_securing_ends_it_quietly():
    table = {(L, "dev"): 0.5, (S, "dev"): 1.0, (S, "holdout"): "budget"}
    refs = {c.id: ref(c.id, "billing") for c in ROUTE_CASES}
    config = RunConfig(repo_url="x")
    assert (
        secure_holdouts(
            SITE, ROUTE_CASES, refs, V0, FakeLLM(), config, matrix_cells(table), runner=Runner(table)
        )
        == {}
    )


def test_a_budget_stop_before_any_holdout_names_the_best_tier_and_score_reached():
    """cs_01 and cs_05 of that run ended KEEP_CLOSED("budget ...") with best_tier, best_score null and
    iterations 0. The best adapter measured so far, the matrix's cells included, is now on the verdict."""
    table = {(L, "dev"): 0.5, (S, "dev"): 0.4, (U, "dev"): 0.6}
    out, runner, it = route(
        table, site=FREE_TEXT, prior=matrix_cells(table, FREE_TEXT), iterator=StubIterator(error=ULTRA_CAP)
    )
    v = out.verdict
    assert (v.decision, v.failed_gate, v.best_tier, v.iterations) == ("KEEP_CLOSED", "budget", U, 0)
    assert v.best_score == pytest.approx(3.8) and v.parity_pct is None and out.adapter is None
    assert v.reason == (
        "budget: Ultra agent call cap reached (40); best reached: Ultra dev, judge mean 3.80 < 4.0 "
        "(parity vs reference 60.0%) after 0 adapter iterations (holdout not measured)"
    )


def test_a_budget_stop_counts_the_iterations_of_the_best_adapter():
    table = {(L, "dev"): 0.5, (L, "dev", 1): 0.9, (L, "dev", 2): "budget"}
    out, runner, it = route(table)
    v = out.verdict
    assert (v.best_tier, v.best_score, v.iterations) == (L, 0.88, 1)  # v1: 22 of 25 dev cases
    assert (
        "best reached: Lightning dev, parity vs reference 88.0% < 95% after 1 adapter iteration " in v.reason
    )


def test_an_iterator_without_a_valid_adapter_ends_that_tiers_iterations():
    events = []
    out, runner, it = route(
        {(L, "dev"): 0.5, (S, "dev"): 1.0, (S, "holdout"): 1.0},
        iterator=StubIterator(give_up_at=2),
        events=events,
    )
    assert runner.calls[:3] == [(L, "dev", 0), (L, "dev", 1), (S, "dev", 0)]
    [log] = [p for t, p in events if t == "log"]
    assert log["level"] == "warn" and "no valid adapter" in log["message"] and "lightning" in log["message"]


@pytest.mark.parametrize(
    "error, words",
    [
        (MissingReference(["cs_01:004"]), "no reference"),
        (UnresolvedCallSite("call cannot be evaluated: <expr> in tools"), "<expr> in tools"),
    ],
)
def test_data_and_call_errors_send_the_site_to_manual_review(error, words):
    out, runner, it = route({(L, "dev"): error})
    assert out.verdict.decision == "MANUAL_REVIEW" and words in out.verdict.reason


def test_a_provider_error_is_retried_once_and_the_retry_counts():
    events = []
    judge_omission = ProviderError("judge: the reply left out case 3")
    out, runner, it = route({(L, "dev"): [judge_omission, 1.0], (L, "holdout"): 1.0}, events=events)
    assert runner.calls == [(L, "dev", 0), (L, "dev", 0), (L, "holdout", 0)]
    assert out.verdict.decision == "MIGRATE" and len(out.results) == 2  # the failed attempt left no result
    [log] = [p for t, p in events if t == "log"]
    assert log["level"] == "warn" and "retrying once" in log["message"] and "judge:" in log["message"]


def test_a_second_provider_error_is_a_run_error_never_a_verdict():
    events = []
    outage = ProviderError("8/8 calls timed out or failed with a 5xx: a provider outage, not a measurement")
    table = {(L, "dev"): 1.0, (L, "holdout"): [outage, outage, 1.0]}
    with pytest.raises(ProviderError, match="outage"):
        route(table, events=events)
    assert table[(L, "holdout")] == [1.0]  # exactly two attempts
    assert [t for t, _ in events] == ["tier_result", "log"]  # no verdict event, no measured holdout


@pytest.mark.parametrize(
    "site, cases, words",
    [
        (SITE.model_copy(update={"contract": None}), ROUTE_CASES, "no contract"),
        (SITE.model_copy(update={"flags": ["MANUAL_REVIEW"]}), ROUTE_CASES, "flagged MANUAL_REVIEW"),
        (SITE, [c for c in ROUTE_CASES if c.split == "dev"], "no holdout cases"),
        (SITE, [c for c in ROUTE_CASES if c.split == "holdout"], "no dev cases"),
    ],
)
def test_sites_that_cannot_be_evaluated_go_to_manual_review_without_calls(site, cases, words):
    out, runner, it = route({}, site=site, cases=cases)
    assert out.verdict.decision == "MANUAL_REVIEW" and words in out.verdict.reason and runner.calls == []


def test_events_in_order_with_json_payloads():
    events = []
    route({(L, "dev"): 0.5, (L, "dev", 1): 1.0, (L, "holdout"): 1.0}, events=events)
    assert [t for t, _ in events] == ["tier_result", "iteration", "tier_result", "tier_result", "verdict"]
    first = events[0][1]
    assert first["tier"] == "lightning" and first["split"] == "dev" and "case_scores" not in first
    assert events[1][1]["version"] == 1 and events[1][1]["change_log"] == ["stub change 1"]
    assert events[-1][1]["decision"] == "MIGRATE"
    json.dumps(events)  # plain JSON, ready for RunEvent.payload


def test_closed_references_fill_the_verdicts_comparison_row():
    hold = [c for c in ROUTE_CASES if c.split == "holdout"]
    closed = {c.id: ref(c.id, "billing", "closed_model", CLAUDE) for c in hold}
    out, *_ = route({(L, "dev"): 1.0, (L, "holdout"): 1.0}, closed=closed)
    assert (out.verdict.closed_reference_model, out.verdict.closed_reference_parity_pct) == (CLAUDE, 100.0)


# ---------------------------------------------------------------- cassettes recorded live from Token Factory


def recorded_verdicts(client) -> dict:
    """Verdicts with FIXTURE evidence (made-up measurements), each asking Super for its sentence.

    A KEEP_CLOSED and a MIGRATE on Super. The two Lightning MIGRATEs recorded with them (cs_01, cs_03) were
    dropped: their facts said "cheaper tiers did not", Super wrote "lower-cost tiers failed" of the cheapest
    tier, and the corrected facts change their keys (the fix round's 4 live Super calls were spent).
    """
    cs05_hold = result(
        REASONING, Tier.ULTRA, "holdout", scores([True] * 7 + [False] * 3, "cs_05", "wrong_answer")
    )
    cs05_dev = result(
        REASONING, Tier.ULTRA, "dev", scores([True] * 14 + [False] * 6, "cs_05", "wrong_answer")
    )
    free = [
        CaseScore(case_id=f"cs_06:{i:03d}", passed=True, score=s) for i, s in enumerate([5, 4.33, 4.67, 4])
    ]
    config = RunConfig(repo_url="x")
    return {
        "cs_05": decide(
            REASONING,
            evidence(
                REASONING,
                Tier.ULTRA,
                cs05_dev,
                cs05_hold,
                iterations=3,
                changes=["Added a worked example of the proration rule"],
            ),
            client,
            config,
            closed_parity=(CLAUDE, 90.0),
            price_lookup=price,
        ),
        "cs_06": decide(
            FREE_TEXT,
            evidence(
                FREE_TEXT,
                Tier.SUPER,
                result(FREE_TEXT, Tier.SUPER, "dev", free),
                result(FREE_TEXT, Tier.SUPER, "holdout", free),
            ),
            client,
            config,
            price_lookup=price,
        ),
    }


def test_verdicts_on_recorded_cassettes():
    verdicts = recorded_verdicts(TFClient(cache_dir=CASSETTES, live=False))
    # Super's sentences as recorded (re-recording may change these pins): qualitative, no numbers, accepted;
    # every number comes after them, from code.
    assert verdicts["cs_05"].decision == "KEEP_CLOSED" and verdicts["cs_05"].reason == (
        "The plan_refund function reasons about refund eligibility using policy rules but none of the tested "
        "tiers matched reference behavior on edge cases involving partial refunds. parity vs reference "
        "70.0% < 90% on Ultra after 3 adapter iterations; cost per 1k calls $0.0618 before (est.); "
        f"closed reference {CLAUDE} 90.0% on the same references."
    )
    assert verdicts["cs_06"].decision == "MIGRATE" and verdicts["cs_06"].reason == (
        "The summarize_thread function generates free-text thread summaries using the current model, and "
        "migration to Super is warranted because it matched reference outputs while lower-cost options fell "
        "short. Super: judge mean 4.50 ≥ 4.0 (parity vs reference 100.0%) on holdout after 0 adapter "
        "iterations; cost per 1k calls $0.0618 before (est.), $0.0318 after."
    )


@pytest.mark.live
def test_record_cassettes():
    """pytest --live: records the 2 Super calls above (hits are free)."""
    recorded_verdicts(TFClient(cache_dir=CASSETTES, live=True, budget=Budget(max_in=20_000, max_out=2_000)))
