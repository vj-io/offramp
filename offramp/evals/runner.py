"""Eval runner (PRD §6.5): run one site's cases on one tier, score them, measure latency/tokens/cost.

Every TierResult number shown anywhere comes from here, so the rules are strict:
- every case of the requested split that has something to compare against is run and scored, failed calls
  included (scored as failures, never dropped);
- eval-target calls use role "eval_target" (the only Ultra role outside the 40-call agent cap) and send
  `reasoning` per AdapterSpec.reasoning, so enable_thinking is always explicit (Global Constraint 25);
- cost is tokens x get_tier(tier) prices (an OFFRAMP_MODEL_* override keeps the tier price);
- a per-run cap (BudgetExceeded, from an eval call or the judge) ends the tier with a BudgetStop result that
  the routing loop turns into KEEP_CLOSED("budget ..."); it never escapes as an exception.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import openai

from offramp.agents.judge import LOW_SCORE, case_passes, judge
from offramp.evals.errors import MissingReference, ProviderError
from offramp.evals.scorers import score_case
from offramp.evals.scorers.aggregate import CANDIDATE_ERROR, DEFAULT_THRESHOLDS, Thresholds, site_pass
from offramp.schemas import (
    Adapter,
    AdapterSpec,
    BaselineResult,
    CallSite,
    CandidateResult,
    CaseScore,
    EvalCase,
    RunConfig,
    Split,
    Tier,
    TierMetrics,
    TierResult,
)
from offramp.tf.budget import EVAL_TARGET_ROLE, BudgetExceeded
from offramp.tf.client import LLM
from offramp.tf.tiers import get_tier

MAX_CONCURRENCY = 8

BuildRequest = Callable[[AdapterSpec, EvalCase, Tier], dict[str, Any]]


@dataclass(frozen=True)
class BudgetStop:
    """run_tier ended early because a per-run cap was reached (Global Constraint 8)."""

    site_id: str
    tier: Tier
    split: Split
    reason: str  # "budget: <cap message>", ready for Verdict(decision="KEEP_CLOSED", reason=...)


def _default_build_request() -> BuildRequest:
    from offramp.agents.translate import build_request  # Task 6

    return build_request


def run_tier(
    site: CallSite,
    cases: Sequence[EvalCase],
    baselines: Mapping[str, BaselineResult] | Sequence[BaselineResult],
    adapter: Adapter,
    tier: Tier,
    client: LLM,
    split: Split,
    thresholds: Thresholds | RunConfig = DEFAULT_THRESHOLDS,
    *,
    build_request: BuildRequest | None = None,
) -> TierResult | BudgetStop:
    """Evaluate `adapter` on `tier` over the `split` cases of `site`.

    Deterministic sites run the cases that have a reference (a case without one has nothing to be
    compared with; it is left out before any call, so this cannot favour a pass). free_text sites run every
    case; references with text make the judge pairwise. CacheMiss (replay) and errors from build_request
    propagate: they are setup errors, not model behaviour.
    """
    contract = site.contract
    if contract is None:
        raise ValueError(f"site {site.id} has no contract: label it before evaluating")
    tier = Tier(tier)
    refs = baselines if isinstance(baselines, Mapping) else {b.case_id: b for b in baselines}
    free_text = contract.task_type == "free_text"
    selected = [c for c in cases if c.split == split and (free_text or c.id in refs)]

    build = build_request or _default_build_request()
    requests = [
        # build_request decides everything else; role and tier are not its call
        {
            "reasoning": adapter.spec.reasoning,
            **build(adapter.spec, c, tier),
            "role": EVAL_TARGET_ROLE,
            "tier": tier,
        }
        for c in selected
    ]
    try:
        results = _run_calls(client, requests, selected, tier, adapter.version)
        if free_text:
            scores, reference_mean = _judge_scores(client, selected, results, refs)
        else:
            scores = [
                score_case(c, refs[c.id].output, r.output, contract)
                for c, r in zip(selected, results, strict=True)
            ]
            reference_mean = None
    except BudgetExceeded as e:
        return BudgetStop(site_id=site.id, tier=tier, split=split, reason=f"budget: {e}")
    scores = [  # keep why a call failed: TierResult is the only record of it
        s.model_copy(update={"detail": f"candidate call failed: {r.error}"}) if r.error else s
        for s, r in zip(scores, results, strict=True)
    ]

    passed, score = site_pass(contract.task_type, scores, thresholds, reference_mean)
    return TierResult(
        site_id=site.id,
        tier=tier,
        adapter_version=adapter.version,
        split=split,
        passed=passed,
        score=score,
        metrics=_metrics(results),
        failing_case_ids=[s.case_id for s in scores if not s.passed],
        case_scores=scores,
    )


def _run_calls(
    client: LLM, requests: list[dict[str, Any]], cases: list[EvalCase], tier: Tier, version: int
) -> list[CandidateResult]:
    info = get_tier(tier)

    def call(req: dict[str, Any], case: EvalCase) -> CandidateResult:
        try:
            r = client.chat(**req)
        except openai.APIError as e:  # after the SDK's own retries: a failed candidate, scored as one
            return CandidateResult(
                case_id=case.id,
                tier=tier,
                adapter_version=version,
                output=None,
                error=f"{type(e).__name__}: {e}"[:500],
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
            )
        return CandidateResult(
            case_id=case.id,
            tier=tier,
            adapter_version=version,
            output=r.output,
            latency_ms=r.latency_ms,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            cost_usd=(r.input_tokens * info.input_usd_per_m + r.output_tokens * info.output_usd_per_m)
            / 1_000_000,
            cached=r.cached,
        )

    pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)
    try:
        futures = [pool.submit(call, req, case) for req, case in zip(requests, cases, strict=True)]
        return [f.result() for f in futures]
    finally:
        pool.shutdown(cancel_futures=True)  # on BudgetExceeded/CacheMiss: start no further calls


def _judge_scores(
    client: LLM, cases: list[EvalCase], results: list[CandidateResult], refs: Mapping[str, BaselineResult]
) -> tuple[list[CaseScore], float | None]:
    ok = [(c, r) for c, r in zip(cases, results, strict=True) if r.output is not None]
    judged = judge(
        client,
        [c for c, _ in ok],
        {c.id: r.output for c, r in ok},
        {cid: b.output for cid, b in refs.items()},
    )
    by_id = {s.case_id: s for s in judged.scores}
    scores = []
    for case in cases:
        s = by_id.get(case.id)
        if s is None:  # the call failed: nothing to judge
            scores.append(
                CaseScore(
                    case_id=case.id,
                    passed=False,
                    score=1.0,
                    detail="candidate call failed",
                    error_category=CANDIDATE_ERROR,
                )
            )
            continue
        passed = case_passes(s)
        scores.append(
            CaseScore(
                case_id=case.id,
                passed=passed,
                score=s.candidate,
                detail=s.detail,
                error_category=s.error_category or (None if passed else LOW_SCORE),
            )
        )
    return scores, judged.reference_mean


def _nearest_rank(sorted_values: list[int], pct: float) -> int:
    return sorted_values[max(0, math.ceil(pct / 100 * len(sorted_values)) - 1)] if sorted_values else 0


def _metrics(results: list[CandidateResult]) -> TierMetrics:
    """Over the calls that returned (a failed call measured nothing)."""
    ok = [r for r in results if r.output is not None]
    latencies = sorted(r.latency_ms for r in ok)
    mean_cost = math.fsum(r.cost_usd for r in ok) / len(ok) if ok else 0.0
    return TierMetrics(
        p50_ms=_nearest_rank(latencies, 50),
        p95_ms=_nearest_rank(latencies, 95),
        tokens_in=sum(r.input_tokens for r in ok),
        tokens_out=sum(r.output_tokens for r in ok),
        cost_per_1k_calls_usd=1000 * mean_cost,
    )
