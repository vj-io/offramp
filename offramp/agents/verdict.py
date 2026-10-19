"""Verdict Agent (PRD §6.7): the rule-based decision, its numbers, and one Super sentence for humans.

Rule: MIGRATE(tier) when the tier's adapter passed dev and then holdout; otherwise KEEP_CLOSED with the best
tier tried and its holdout parity. Every number is measured (Global Constraint 9) or labelled an estimate
(GC21):
- parity_pct: holdout parity vs reference (GC19): the site score x 100 for deterministic scorers, the share of
  passing cases for free text (whose score is a judge mean). A failing parity is floored, so it never displays
  at the threshold ("90.0% < 90%");
- cost_after_per_1k: the holdout run's measured $/1k calls. cost_before_per_1k: the source model's published
  price (Tavily lookup, Task 17; pricing.yaml when that module is absent) x the mean tokens of the same Token
  Factory calls, labelled "est." with its source and date;
- latency_delta_ms: None, the source model's latency was never measured.
Super writes the reason sentence. It is used only if every number in it appears in the facts it was given
and, for KEEP_CLOSED, it quotes the measured shortfall verbatim; otherwise, or when the call fails, the rule
sentence stands. A sentence is never worth a run error or an invented number.
"""

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import openai
from pydantic import BaseModel, Field

from offramp import pricing
from offramp.evals.scorers import score_case
from offramp.evals.scorers.aggregate import (
    CANDIDATE_ERROR,
    JUDGE_FLOOR,
    JUDGE_MARGIN,
    Thresholds,
    _at_least,
    site_pass,
)
from offramp.evals.scorers.json_schema import INVALID_CATEGORIES
from offramp.schemas import (
    BaselineResult,
    CallSite,
    CandidateResult,
    Decision,
    EvalCase,
    RunConfig,
    TaskType,
    Tier,
    TierResult,
    Verdict,
)
from offramp.tf.budget import BudgetExceeded
from offramp.tf.client import LLM
from offramp.tf.tiers import get_tier

VERDICT_ROLE = "verdict"
VERDICT_MAX_TOKENS = 300  # thinking off; one sentence in a JSON object
MAX_REASON = 400
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_NOT_VALID = INVALID_CATEGORIES | {CANDIDATE_ERROR}  # as aggregate.site_pass counts schema validity

VERDICT_SYSTEM = """You write the one-sentence reason for one Offramp verdict. Offramp tested whether NVIDIA \
Nemotron on Nebius Token Factory can take over one LLM call site of a Python repo, measured as agreement \
with reference outputs on held-out cases. MIGRATE: the named tier reached parity on the held-out cases, \
cheaper tiers did not. KEEP_CLOSED: no tier did, so the call stays on its current model.

Write one sentence of at most 40 words for an engineer reviewing the pull request: what the call site does \
and why the verdict follows. Use only the facts given. Every number you write must appear in the facts \
exactly as written: never compute new numbers, prices or savings. For KEEP_CLOSED, include the "result" text \
verbatim. Name tiers as Lightning, Super or Ultra.

Reply with one JSON object: {"reason": "<the sentence>"}."""


class VerdictReason(BaseModel):
    reason: str = Field(max_length=MAX_REASON)


@dataclass(frozen=True)
class Evidence:
    """What routing measured for the verdict's tier: MIGRATE's chosen tier, KEEP_CLOSED's best one."""

    tier: Tier
    dev: TierResult  # the dev run of the adapter evaluated on holdout
    holdout: TierResult  # the holdout run of that adapter
    calls: Sequence[CandidateResult]  # the holdout run's calls: token means for the cost estimate
    iterations: int  # adapter iterations run on this tier
    changes: list[str] = field(default_factory=list)  # their change logs, in order


@dataclass(frozen=True)
class SourcePrice:
    """The fields of offramp.integrations.tavily.SourceModelInfo (Task 17) the verdict reads."""

    model: str
    input_usd_per_m: float
    output_usd_per_m: float
    price_source_url: str
    fetched_at: datetime
    deprecation: str | None = None
    deprecation_url: str | None = None
    via: str = "pricing.yaml"


PriceLookup = Callable[[str], Any]  # model id -> SourceModelInfo-like; KeyError when no price is known


def pricing_yaml_price(model: str) -> SourcePrice:
    e = pricing.entry(model)  # KeyError when unknown
    return SourcePrice(
        model=model,
        input_usd_per_m=e["input_usd_per_m"],
        output_usd_per_m=e["output_usd_per_m"],
        price_source_url=e["source"],
        fetched_at=datetime.fromisoformat(str(e["as_of"])),
    )


def lookup_price(model: str) -> Any:
    """GC21: the Tavily runtime lookup (Task 17), or pricing.yaml where that module is not installed."""
    try:
        from offramp.integrations.tavily import lookup_source_model
    except ImportError:
        return pricing_yaml_price(model)
    return lookup_source_model(model)


def stopped(site: CallSite, decision: Decision, reason: str) -> Verdict:
    """A verdict without a measured comparison: budget stops (KEEP_CLOSED) and MANUAL_REVIEW."""
    return Verdict(site_id=site.id, decision=decision, reason=reason)


def closed_reference_parity(
    site: CallSite,
    cases: Sequence[EvalCase],
    references: Mapping[str, BaselineResult],
    closed: Mapping[str, BaselineResult] | None,
    thresholds: Thresholds | RunConfig,
) -> tuple[str, float] | None:
    """GC27(c): (closed model, its holdout parity %) on the same references, scored with the same scorer.

    Only holdout cases whose reference is a real reference count: a closed output standing in for a missing
    reference (free text, GC27a) would be scored against itself. None for free text and without closed rows.
    """
    if not closed or site.contract is None or site.contract.task_type == "free_text":
        return None
    scored, models = [], set()
    for case in cases:
        ref, out = references.get(case.id), closed.get(case.id)
        if case.split == "holdout" and ref is not None and out is not None and ref.source == "reference":
            scored.append(score_case(case, ref.output, out.output, site.contract))
            models.add(out.model)
    if not scored:
        return None
    _, score = site_pass(site.contract.task_type, scored, thresholds)
    return ", ".join(sorted(models)), round(100 * score, 1)


def parity_pct(result: TierResult, task_type: TaskType) -> float:
    """Parity vs reference in percent, one decimal; floored when the run failed."""
    if task_type == "free_text":  # score is a judge mean: parity is the share of passing cases
        rows = result.case_scores
        share = sum(c.passed for c in rows) / (len(rows) or 1)
    else:
        share = result.score
    return round(100 * share, 1) if result.passed else _floor(100 * share)


def _floor(pct: float) -> float:
    return math.floor(pct * 10 + 1e-6) / 10  # slack: 28.999999999999996 is 29.0


def _phrase(result: TierResult, task_type: TaskType, thresholds: Thresholds | RunConfig) -> str:
    """One run's measured outcome, in the words of the reason ("parity vs reference 71.0% < 90%")."""
    pct = parity_pct(result, task_type)
    if task_type == "free_text":
        mean, parity = result.score, f"(parity vs reference {pct:.1f}%)"
        if result.passed:
            return f"judge mean {mean:.2f} ≥ {JUDGE_FLOOR:.1f} {parity}"
        if not _at_least(mean, JUDGE_FLOOR):
            return f"judge mean {mean:.2f} < {JUDGE_FLOOR:.1f} {parity}"
        return f"judge mean {mean:.2f} more than {JUDGE_MARGIN:g} below the references' judge mean {parity}"
    bar = thresholds.reasoning_threshold if task_type == "reasoning" else thresholds.threshold
    head, t = f"parity vs reference {pct:.1f}%", f"{round(bar * 100, 4):g}%"
    if result.passed:
        return f"{head} ≥ {t}"
    if not _at_least(result.score, bar):
        return f"{head} < {t}"
    rows = result.case_scores  # json_extraction failing with mean field match at or above the bar
    invalid = sum(c.error_category in _NOT_VALID for c in rows)
    if invalid:
        return f"{head} but {invalid}/{len(rows)} outputs not schema-valid (100% required)"
    passing = _floor(100 * sum(c.passed for c in rows) / len(rows))
    return f"{head} but only {passing:.1f}% of cases pass (< {t})"


def _usd(x: float) -> str:
    return f"${x:.2f}" if round(x, 2) == x else f"${x:g}"


def _cost_before(
    model: str | None, evidence: Evidence, lookup: PriceLookup
) -> tuple[float | None, str, str | None]:
    """(cost_before_per_1k, cost_before_note, sunset_risk), GC21."""
    if model is None:
        return None, "source model unknown: cost before not estimated", None
    try:
        info = lookup(model)
    except KeyError:
        return None, f"no published price found for {model}: cost before not estimated", None
    sunset = None
    if info.deprecation:
        sunset = f"{info.deprecation} ({info.deprecation_url})" if info.deprecation_url else info.deprecation
    ok = [c for c in evidence.calls if c.output is not None]  # a failed call reports no usage
    if not ok:
        return None, "no Token Factory call returned output: cost before not estimated", sunset
    mean_in = math.fsum(c.input_tokens for c in ok) / len(ok)
    mean_out = math.fsum(c.output_tokens for c in ok) / len(ok)
    cost = 1000 * (mean_in * info.input_usd_per_m + mean_out * info.output_usd_per_m) / 1_000_000
    via = getattr(info, "via", None)
    source = (
        f"{info.price_source_url.split('://', 1)[-1]}, {f'{via} ' if via else ''}{info.fetched_at:%Y-%m-%d}"
    )
    prices = f"{_usd(info.input_usd_per_m)}/{_usd(info.output_usd_per_m)} per 1M"
    note = (
        f"est.: {info.model} {prices} ({source}) x mean tokens measured on {len(ok)} Token Factory calls "
        f"({evidence.tier.value}, holdout)"
    )
    return cost, note, sunset


def _iterations(n: int) -> str:
    return f"{n} adapter iteration{'' if n == 1 else 's'}"


def _accept(sentence: str, facts: str, required: str | None) -> bool:
    sentence = sentence.strip()
    if not sentence or "\n" in sentence or (required and required not in sentence):
        return False
    return set(_NUMBER.findall(sentence)) <= set(_NUMBER.findall(facts))


def _reason(llm: LLM, facts: dict[str, Any], required: str | None, fallback: str) -> str:
    text = json.dumps(facts, indent=1, ensure_ascii=False)
    try:
        result = llm.chat(
            role=VERDICT_ROLE,
            tier=Tier.SUPER,
            messages=[{"role": "system", "content": VERDICT_SYSTEM}, {"role": "user", "content": text}],
            max_tokens=VERDICT_MAX_TOKENS,
            temperature=0.0,
            response_model=VerdictReason,
            reasoning="off",
        )
    except (BudgetExceeded, openai.APIError):  # CacheMiss (a replay without its cassette) propagates
        return fallback
    parsed: VerdictReason | None = result.parsed
    return parsed.reason.strip() if parsed and _accept(parsed.reason, text, required) else fallback


def decide(
    site: CallSite,
    evidence: Evidence,
    llm: LLM,
    thresholds: Thresholds | RunConfig,
    *,
    closed_parity: tuple[str, float] | None = None,
    price_lookup: PriceLookup | None = None,
) -> Verdict:
    """MIGRATE or KEEP_CLOSED for a site with a measured holdout run (see the module docstring)."""
    task = site.contract.task_type
    tier, dev, holdout, n = Tier(evidence.tier), evidence.dev, evidence.holdout, evidence.iterations
    migrate = dev.passed and holdout.passed
    name = tier.value.capitalize()
    if migrate:
        result, required = f"{_phrase(holdout, task, thresholds)} on holdout", None
        fallback = f"{name}: {result} after {_iterations(n)}."
    else:
        if holdout.passed:  # dev never passed: the holdout pass is not a qualification
            result = f"dev {_phrase(dev, task, thresholds)} (holdout {parity_pct(holdout, task):.1f}%)"
        else:
            result = _phrase(holdout, task, thresholds)
        required, fallback = result, f"{result} on {name} after {_iterations(n)}."
    cost_before, note, sunset = _cost_before(site.source_model, evidence, price_lookup or lookup_price)
    facts = {
        "site": {
            "id": site.id,
            "file": site.file,
            "function": site.function,
            "source_model": site.source_model,
            "task_type": task,
        },
        "decision": "MIGRATE" if migrate else "KEEP_CLOSED",
        "tier": tier.value,
        "result": result,
        "iterations": n,
        "adapter_changes": evidence.changes,
        "holdout_failures_by_category": Counter(
            c.error_category for c in holdout.case_scores if not c.passed
        ),
        "closed_reference": None
        if closed_parity is None
        else {"model": closed_parity[0], "parity_pct": closed_parity[1]},
    }
    return Verdict(
        site_id=site.id,
        decision="MIGRATE" if migrate else "KEEP_CLOSED",
        tier=tier if migrate else None,
        model=get_tier(tier).model_id if migrate else None,
        parity_pct=parity_pct(holdout, task),
        best_tier=None if migrate else tier,
        best_score=None if migrate else holdout.score,
        iterations=n,
        latency_delta_ms=None,
        cost_before_per_1k=cost_before,
        cost_before_note=note,
        cost_after_per_1k=holdout.metrics.cost_per_1k_calls_usd if migrate else None,
        sunset_risk=sunset,
        closed_reference_model=closed_parity[0] if closed_parity else None,
        closed_reference_parity_pct=closed_parity[1] if closed_parity else None,
        reason=_reason(llm, facts, required, fallback),
    )
