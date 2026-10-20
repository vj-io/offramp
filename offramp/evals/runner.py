"""Eval runner (PRD §6.5): run one site's cases on one tier, score them, measure latency/tokens/cost.

Every TierResult number shown anywhere comes from here, so the rules are strict:
- every case of the requested split is run and scored; per-request model failures (timeout, 500, 400/422)
  are failed cases, never dropped, unless half the calls or more timed out or failed 5xx (an outage:
  ProviderError). A deterministic case without a reference raises MissingReference, and an
  account/credit/config error raises ProviderError: neither is ever turned into a measured result;
- eval-target calls use role "eval_target" (the only Ultra role outside the 40-call agent cap) and send
  `reasoning` per AdapterSpec.reasoning, so enable_thinking is always explicit (Global Constraint 25);
- cost is tokens x get_tier(tier) prices (an OFFRAMP_MODEL_* override keeps the tier price);
- a per-run cap (BudgetExceeded, from an eval call or the judge) ends the tier with a BudgetStop result that
  the routing loop turns into KEEP_CLOSED("budget ..."); it never escapes as an exception.
"""

from __future__ import annotations

import contextvars
import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import openai

from offramp.agents.judge import LOW_SCORE, case_passes, judge
from offramp.evals.errors import (
    MissingReference,
    ProviderError,
    is_case_failure,
    is_timeout_or_5xx,
    provider_error,
)
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
OUTAGE_FRACTION = 0.5  # this share of a run's calls timing out or failing 5xx is an outage, not a measurement

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
    return_candidates: bool = False,
) -> TierResult | BudgetStop | tuple[TierResult | BudgetStop, list[CandidateResult]]:
    """Evaluate `adapter` on `tier` over the `split` cases of `site`.

    Every case of the split is run and scored. Deterministic sites need a reference for each of them, else
    MissingReference is raised before any call (filter out audit-dropped cases first); free_text sites run
    without one, and references with text make the judge pairwise. ProviderError, CacheMiss (replay) and
    errors from build_request propagate: they are run or setup errors, not model behaviour.

    return_candidates=True returns (result, the CandidateResult of every case in case order) so the
    Iterator can show failing outputs; ([] with a BudgetStop).
    """
    contract = site.contract
    if contract is None:
        raise ValueError(f"site {site.id} has no contract: label it before evaluating")
    tier = Tier(tier)
    refs = baselines if isinstance(baselines, Mapping) else {b.case_id: b for b in baselines}
    free_text = contract.task_type == "free_text"
    selected = [c for c in cases if c.split == split]
    missing = [] if free_text else [c.id for c in selected if c.id not in refs]
    if missing:
        raise MissingReference(missing)

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
        stop = BudgetStop(site_id=site.id, tier=tier, split=split, reason=f"budget: {e}")
        return (stop, []) if return_candidates else stop
    scores = [  # keep why a call failed: TierResult is the only record of it
        s.model_copy(update={"detail": f"candidate call failed: {r.error}"}) if r.error else s
        for s, r in zip(scores, results, strict=True)
    ]

    passed, score = site_pass(contract.task_type, scores, thresholds, reference_mean, split=split)
    result = TierResult(
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
    return (result, results) if return_candidates else result


def _run_calls(
    client: LLM, requests: list[dict[str, Any]], cases: list[EvalCase], tier: Tier, version: int
) -> list[CandidateResult]:
    info = get_tier(tier)
    outage_like: list[str] = []  # case ids that timed out or failed 5xx (list.append is thread-safe)

    def call(req: dict[str, Any], case: EvalCase) -> CandidateResult:
        started = time.perf_counter()
        try:
            r = client.chat(**req)
        except openai.APIError as e:  # after the SDK's own retries
            if not is_case_failure(e):
                raise provider_error(e) from e
            if is_timeout_or_5xx(e):
                outage_like.append(case.id)
            return CandidateResult(
                case_id=case.id,
                tier=tier,
                adapter_version=version,
                output=None,
                error=f"{type(e).__name__}: {e}"[:500],
                latency_ms=round((time.perf_counter() - started) * 1000),  # time until failure
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
        # each task in a copy of this context: a LangSmith run_project (a contextvar) reaches the call (GC23)
        futures = [
            pool.submit(contextvars.copy_context().run, call, req, case)
            for req, case in zip(requests, cases, strict=True)
        ]
        results = [f.result() for f in futures]
    finally:
        pool.shutdown(cancel_futures=True)  # on BudgetExceeded/ProviderError/CacheMiss: no further calls
    if results and len(outage_like) / len(results) >= OUTAGE_FRACTION:
        raise ProviderError(
            f"{len(outage_like)}/{len(results)} calls timed out or failed with a 5xx: "
            "a provider outage, not a measurement"
        )
    return results


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
                reference_score=s.reference,
            )
        )
    return scores, judged.reference_mean


def _nearest_rank(sorted_values: list[int], pct: float) -> int:
    return sorted_values[max(0, math.ceil(pct / 100 * len(sorted_values)) - 1)] if sorted_values else 0


def _metrics(results: list[CandidateResult]) -> TierMetrics:
    """p50/p95 over every call: a returned call at its recorded latency, a failed one at its time until
    failure (a timeout is the tail). Tokens and $/1k calls over returned calls: a failed one reports no usage.
    """
    ok = [r for r in results if r.output is not None]
    latencies = sorted(r.latency_ms for r in results)
    mean_cost = math.fsum(r.cost_usd for r in ok) / len(ok) if ok else 0.0
    return TierMetrics(
        p50_ms=_nearest_rank(latencies, 50),
        p95_ms=_nearest_rank(latencies, 95),
        tokens_in=sum(r.input_tokens for r in ok),
        tokens_out=sum(r.output_tokens for r in ok),
        cost_per_1k_calls_usd=1000 * mean_cost,
    )
