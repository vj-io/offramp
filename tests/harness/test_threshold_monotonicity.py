"""Threshold monotonicity (PRD §12.3): raising thresholds never flips KEEP_CLOSED -> MIGRATE.

Scope: the measured outputs are fixed per (tier, split), so adapter iterations run but do not change them.
That is the property that catches pass-rule and routing bugs. With iterations that DO change outputs the PRD's
loop is path dependent (iterate only on a dev failure; evaluate holdout only after dev passes): at a low
threshold v0 can pass dev and fail holdout, so no v1 is ever made, while at a higher threshold v0 fails dev
and v1 may pass both. test_known_limitation_iteration_is_path_dependent pins that case so it stays visible.
"""

import hashlib
import random
from itertools import pairwise

import pytest

from offramp.evals.route import route_site
from offramp.evals.scorers.aggregate import site_pass
from offramp.schemas import (
    TIER_ORDER,
    Adapter,
    AdapterSpec,
    CallSite,
    CandidateResult,
    CaseScore,
    Contract,
    EvalCase,
    ModelOutput,
    PromptTemplate,
    RunConfig,
    Tier,
    TierMetrics,
    TierResult,
)
from offramp.tf import BudgetExceeded

FRACTIONS = [0.0, 0.5, 0.8, 0.9, 0.92, 0.95, 0.97, 1.0]
GRID = [0.5, 0.8, 0.9, 0.93, 0.95, 0.97, 1.0]


def gc10(case_id: str) -> str:
    return "holdout" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100 < 30 else "dev"


CASES = [
    EvalCase(
        id=f"cs_01:{i:03d}",
        site_id="cs_01",
        input_messages=[{"role": "user", "content": str(i)}],
        expected_kind="label",
        split=gc10(f"cs_01:{i:03d}"),
    )
    for i in range(1, 41)
]


def site(task: str) -> CallSite:
    output = {
        "classification": "label",
        "reasoning": "text",
        "tool_call": "tool_call",
        "json_extraction": "json",
    }
    return CallSite(
        id="cs_01",
        file="app.py",
        lines=[1, 2],
        sdk="openai",
        call="client.chat.completions.create",
        source_model="gpt-4o-mini",
        prompt_source="literal",
        template=PromptTemplate(user="x"),
        contract=Contract(task_type=task, output=output[task]),
    )


class Runner:
    """Stub run_tier with the real site rule: the share of passing cases per (tier, split[, version])."""

    def __init__(self, table: dict):
        self.table = table

    def __call__(
        self, site, cases, baselines, adapter, tier, client, split, thresholds, *, return_candidates=False
    ):
        share = self.table.get((tier, split, adapter.version), self.table.get((tier, split), 0.0))
        selected = [c for c in cases if c.split == split]
        k = round(share * len(selected))
        partial = (
            0.5 if site.contract.task_type == "json_extraction" else 0.0
        )  # field match of a failed record
        rows = [
            CaseScore(
                case_id=c.id,
                passed=i < k,
                score=1.0 if i < k else partial,
                error_category=None if i < k else "wrong_value",
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
            metrics=TierMetrics(p50_ms=1, p95_ms=1, tokens_in=1, tokens_out=1, cost_per_1k_calls_usd=0.01),
            failing_case_ids=[r.case_id for r in rows if not r.passed],
            case_scores=rows,
        )
        out = [
            CandidateResult(
                case_id=c.id,
                tier=tier,
                adapter_version=adapter.version,
                output=ModelOutput(text="x"),
                latency_ms=1,
                input_tokens=10,
                output_tokens=1,
                cost_usd=0.0,
            )
            for c in selected
        ]
        return res, out


def iterate(adapter, site, failing, tier, error_counts):
    return Adapter(
        site_id=site.id,
        version=adapter.version + 1,
        tier=tier,
        spec=adapter.spec.model_copy(update={"user_suffix": f" v{adapter.version + 1}"}),  # a real change
        change_log=["x"],
    )


def no_price(model: str):
    raise KeyError(model)


class NoReasonText:
    def chat(self, **kw):
        raise BudgetExceeded("no Super sentence in this test: the rule sentence stands")


def verdict(table: dict, task: str, t: float, max_iterations: int = 3):
    config = RunConfig(repo_url="x", threshold=t, reasoning_threshold=t, max_iterations=max_iterations)
    return route_site(
        site(task),
        CASES,
        {},
        Adapter(site_id="cs_01", version=0, spec=AdapterSpec()),
        NoReasonText(),
        config,
        iterator=iterate,
        runner=Runner(table),
        price_lookup=no_price,
    ).verdict


@pytest.mark.parametrize("task", ["classification", "reasoning", "tool_call", "json_extraction"])
def test_raising_thresholds_never_flips_keep_closed_to_migrate(task):
    rng = random.Random(task)
    flips_seen = 0
    for _ in range(60):
        table = {(tier, split): rng.choice(FRACTIONS) for tier in TIER_ORDER for split in ("dev", "holdout")}
        verdicts = [verdict(table, task, t, max_iterations=1) for t in GRID]  # iterations change nothing here
        for low, high in pairwise(verdicts):
            assert not (low.decision == "KEEP_CLOSED" and high.decision == "MIGRATE"), (table, low, high)
            if low.decision == high.decision == "MIGRATE":  # a stricter bar never picks a cheaper tier
                assert TIER_ORDER.index(high.tier) >= TIER_ORDER.index(low.tier)
            flips_seen += low.decision != high.decision
    assert flips_seen  # the grid does move decisions, so the assertion above is exercised


def test_exactly_at_the_threshold_passes_and_just_above_does_not():
    table = {(Tier.LIGHTNING, "dev"): 1.0, (Tier.LIGHTNING, "holdout"): 0.8}  # 12/15 holdout cases right
    assert sum(c.split == "holdout" for c in CASES) == 15
    assert verdict(table, "classification", 0.8).decision == "MIGRATE"
    assert verdict(table, "classification", 0.81).decision == "KEEP_CLOSED"


def test_known_limitation_iteration_is_path_dependent():
    """Not monotone when an iteration changes outputs (see the module docstring). At 0.5, v0 passes dev (60%)
    and fails holdout (0%), so the loop moves on and never makes v1; at 0.9, v0 fails dev, v1 is made and
    passes."""
    L = Tier.LIGHTNING
    table = {(L, "dev", 0): 0.6, (L, "holdout", 0): 0.0, (L, "dev", 1): 1.0, (L, "holdout", 1): 1.0}
    assert verdict(table, "classification", 0.5).decision == "KEEP_CLOSED"
    assert verdict(table, "classification", 0.9).decision == "MIGRATE"
