"""Verdict Agent (PRD §6.7): the rule-based decision, its numbers, and one Super sentence for humans.

Rule: MIGRATE(tier) when the tier's adapter passed dev and then holdout; otherwise KEEP_CLOSED with the best
tier tried and its holdout parity. Every number is measured (Global Constraint 9) or labelled an estimate
(GC21):
- parity_pct: holdout parity vs reference (GC19): the site score x 100 for deterministic scorers, the share of
  passing cases for free text (whose score is a judge mean). A failing parity is floored, so it never displays
  at the threshold ("90.0% < 90%"); so is a failing judge mean;
- cost_after_per_1k: the holdout run's measured $/1k calls. cost_before_per_1k: the source model's published
  price (Task 17's Tavily lookup, which falls back to pricing.yaml and says why) x the mean tokens of the same
  Token Factory calls; cost_before_note is Task 17's "est." note plus how many calls were averaged;
- latency_delta_ms: None, the source model's latency was never measured;
- holdout_cases / holdout_passed / parity_lower_pct: the holdout run's case count, passing cases and the
  Wilson 95% lower bound of that pass rate; free text adds judge_mean and judge_min (its lowest case score);
- price_evidence and retires_on: the source-model lookup's evidence and the retirement date of its notice.
Verdict.reason = Super's qualitative sentence, then a code-built clause that carries every number (parity,
threshold, tier, iterations, cost before (est.) and after, the closed-reference row). Super gets no numbers
and may write none: a sentence with a digit or a number word, or none at all (the call failed), leaves the
clause alone as the reason. A sentence is never worth a run error or a misattributed number.
"""

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import openai
from pydantic import BaseModel, Field

from offramp.evals.scorers import score_case
from offramp.evals.scorers.aggregate import (
    _NOT_VALID,
    CANDIDATE_ERROR,
    JUDGE_CASE_FLOOR,
    JUDGE_FLOOR,
    JUDGE_MARGIN,
    NOT_JSON,
    WEAK_REFERENCE,
    Thresholds,
    _at_least,
    lower_pct,
    site_pass,
)
from offramp.integrations.tavily import SourceModelInfo, _from_yaml, lookup_source_model
from offramp.schemas import (
    TIER_ORDER,
    BaselineResult,
    CallSite,
    CandidateResult,
    Decision,
    EvalCase,
    Gate,
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
# Super may write no numbers: digits are checked one by one (str.isdigit), number words here. Left out: "one"
# (mostly a pronoun) and "half" ("half-up rounding"). ponytail: extend if Super starts spelling numbers out.
_NUMBER_WORDS = re.compile(
    r"%|\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|billion|dozen|percent)\b",
    re.I,
)

VERDICT_SYSTEM = """You write one qualitative sentence explaining one Offramp verdict. Offramp tested \
whether NVIDIA Nemotron on Nebius Token Factory can take over one LLM call site of a Python repo, measured \
as agreement with reference outputs on held-out cases. MIGRATE: the named tier reached parity on the \
held-out cases, cheaper tiers did not. KEEP_CLOSED: no tier did, so the call stays on its current model.

Write one sentence of at most 30 words for an engineer reviewing the pull request: what the call site does \
and why the verdict follows, in words only. Write no numbers at all: no digits, no number words, no \
percentages, and no model names or versions (say "the current model"). Offramp appends the measured numbers \
after your sentence. Name the tier as given in the facts.

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


PriceLookup = Callable[[str], SourceModelInfo]  # model id -> price; KeyError when no price is known


def pricing_yaml_price(model: str) -> SourceModelInfo:
    """An offline lookup (replays, tests): pricing.yaml through Task 17's own fallback, which says so."""
    return _from_yaml(model, datetime.now(UTC), "offline run, Tavily not asked", None)


def stopped(site: CallSite, decision: Decision, reason: str, gate: Gate | None = None) -> Verdict:
    """A verdict without a measured comparison: budget stops (KEEP_CLOSED, gate "budget"), MANUAL_REVIEW."""
    return Verdict(site_id=site.id, decision=decision, reason=reason, failed_gate=gate)


def _holdout_fields(holdout: TierResult, task: TaskType) -> dict[str, Any]:
    """The figures every verdict with a measured holdout carries, from that holdout run."""
    rows, free_text = holdout.case_scores, task == "free_text"
    return {
        "parity_pct": parity_pct(holdout, task),
        "holdout_cases": len(rows) or None,
        "holdout_passed": sum(c.passed for c in rows) if rows else None,
        "parity_lower_pct": lower_pct(rows),
        "judge_mean": _judge(holdout, holdout.score) if free_text else None,
        "judge_min": _judge(holdout, min(c.score for c in rows)) if free_text and rows else None,
    }


def budget_stopped(
    site: CallSite,
    reason: str,
    best: tuple[Tier, TierResult, int] | None,
    thresholds: Thresholds | RunConfig,
    holdout: TierResult | None = None,
) -> Verdict:
    """KEEP_CLOSED for a per-run cap that stopped routing before any verdict. `best`: the best adapter
    measured so far (its tier, its dev result, its iterations); `holdout`: its holdout run, when one was
    measured (it failed, or there would be a verdict). The reason says what was measured on which split, and
    the verdict carries the holdout's figures only when there is one."""
    if best is None:
        return stopped(site, "KEEP_CLOSED", reason, "budget")
    tier, dev, n = best
    task = site.contract.task_type
    measured = f"{'passed' if dev.passed else 'failed'} dev: {_phrase(dev, task, thresholds)[0]}"
    if holdout is None:
        measured += "; holdout not measured"
    else:
        result = "passed" if holdout.passed else "failed"
        measured += f"; {result} the holdout: {_phrase(holdout, task, thresholds)[0]}"
    return Verdict(
        site_id=site.id,
        decision="KEEP_CLOSED",
        best_tier=tier,
        best_score=(holdout or dev).score,
        best_score_split="holdout" if holdout else "dev",
        iterations=n,
        failed_gate="budget",
        reason=f"{reason}; best reached: {tier.value.capitalize()} after {_iterations(n)}, {measured}",
        **(_holdout_fields(holdout, task) if holdout else {}),
    )


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
    return round(100 * share, 1) if result.passed else _floor(100 * share, 1)


def _floor(x: float, places: int) -> float:
    """Round down, so a failing value never displays at its bar; slack: 28.999999999999996 is 29.0."""
    return math.floor(x * 10**places + 1e-6) / 10**places


def _phrase(
    result: TierResult, task_type: TaskType, thresholds: Thresholds | RunConfig
) -> tuple[str, Gate | None]:
    """One run's measured outcome, in the words of the reason ("parity vs reference 71.0% < 90%"), and the
    gate it failed (Verdict.failed_gate; None when it passed). Each branch is the site rule's own
    condition."""
    pct = parity_pct(result, task_type)
    if task_type == "free_text":
        mean = result.score if result.passed else _floor(result.score, 2)
        parity = f"(parity vs reference {pct:.1f}%)"
        if result.passed:
            return f"judge mean {mean:.2f} ≥ {JUDGE_FLOOR:.1f} {parity}", None
        if not _at_least(result.score, JUDGE_FLOOR):
            return f"judge mean {mean:.2f} < {JUDGE_FLOOR:.1f} {parity}", "judge_mean"
        low = min(result.case_scores, key=lambda c: c.score, default=None)
        if result.split == "holdout" and low is not None and not _at_least(low.score, JUDGE_CASE_FLOOR):
            low_text = f"scored {low.score:.2f} < {JUDGE_CASE_FLOOR:.1f}"
            if low.error_category == CANDIDATE_ERROR:  # scored 1.0 by the runner, never judged
                low_text = "had no answer (its call failed)"
            if low.error_category == NOT_JSON:  # likewise: json.loads in the calling code would reject it
                low_text = "was not JSON, which the calling code parses"
            head = f"judge mean {mean:.2f} ≥ {JUDGE_FLOOR:.1f}, but holdout case {low.case_id}"
            return f"{head} {low_text} {parity}", "case_floor"
        text = f"judge mean {mean:.2f} more than {JUDGE_MARGIN:g} below the references' judge mean {parity}"
        return text, "reference_margin"
    bar = thresholds.reasoning_threshold if task_type == "reasoning" else thresholds.threshold
    head, t = f"parity vs reference {pct:.1f}%", f"{round(bar * 100, 4):g}%"
    if result.passed:
        return f"{head} ≥ {t}", None
    if not _at_least(result.score, bar):
        return f"{head} < {t}", "threshold"
    rows = result.case_scores  # json_extraction failing with mean field match at or above the bar
    invalid = sum(c.error_category in _NOT_VALID for c in rows)
    if invalid:
        return f"{head} but {invalid}/{len(rows)} outputs not schema-valid (100% required)", "schema"
    passing = _floor(100 * sum(c.passed for c in rows) / len(rows), 1)
    return f"{head} but only {passing:.1f}% of cases pass (< {t})", "pass_share"


def _cost_before(
    model: str | None, evidence: Evidence, lookup: PriceLookup
) -> tuple[float | None, str, SourceModelInfo | None]:
    """(cost_before_per_1k, cost_before_note, the price lookup's answer), GC21: Task 17's note, plus the
    calls averaged."""
    if model is None:
        return None, "source model unknown: cost before not estimated", None
    try:
        info = lookup(model)
    except KeyError:
        return None, f"no published price found for {model}: cost before not estimated", None
    ok = [c for c in evidence.calls if c.output is not None]  # a failed call reports no usage
    if not ok:
        return None, "no Token Factory call returned output: cost before not estimated", info
    mean_in = math.fsum(c.input_tokens for c in ok) / len(ok)
    mean_out = math.fsum(c.output_tokens for c in ok) / len(ok)
    cost = 1000 * (mean_in * info.input_usd_per_m + mean_out * info.output_usd_per_m) / 1_000_000
    note = f"{info.cost_before_note()} (mean of {len(ok)} {evidence.tier.value} holdout calls)"
    return cost, note, info


def _judge(result: TierResult, x: float) -> float:
    """A judge score as the verdict states it: two decimals, rounded down when the run failed."""
    return round(x, 2) if result.passed else _floor(x, 2)


def _money(x: float) -> str:
    return f"${x:.2f}" if x >= 1 else f"${x:.3g}"


def _iterations(n: int) -> str:
    return f"{n} adapter iteration{'' if n == 1 else 's'}"


def _accept(sentence: str) -> bool:
    """One line, no digit, no number word: every number belongs to the code-built clause."""
    return (
        bool(sentence)
        and "\n" not in sentence
        and not any(ch.isdigit() for ch in sentence)
        and not (_NUMBER_WORDS.search(sentence))
    )


def _sentence(llm: LLM, facts: dict[str, Any]) -> str | None:
    """Super's qualitative sentence, or None when it failed the checks or the call failed."""
    try:
        result = llm.chat(
            role=VERDICT_ROLE,
            tier=Tier.SUPER,
            messages=[
                {"role": "system", "content": VERDICT_SYSTEM},
                {"role": "user", "content": json.dumps(facts, indent=1, ensure_ascii=False)},
            ],
            max_tokens=VERDICT_MAX_TOKENS,
            temperature=0.0,
            response_model=VerdictReason,
            reasoning="off",
        )
    except (BudgetExceeded, openai.APIError):  # CacheMiss (a replay without its cassette) propagates
        return None
    sentence = result.parsed.reason.strip() if result.parsed else ""
    if not _accept(sentence):
        return None
    return sentence if sentence.endswith((".", "!", "?")) else f"{sentence}."


def _clause(
    migrate: bool,
    name: str,
    result: str,
    n: int,
    cost_before: float | None,
    cost_after: float | None,
    closed_parity: tuple[str, float] | None,
) -> str:
    """The code-built part of the reason: every number the verdict states."""
    parts = [f"{name}: {result} on holdout" if migrate else f"{result} on {name}"]
    parts[0] += f" after {_iterations(n)}"
    if migrate and cost_before is not None:
        parts.append(f"cost per 1k calls {_money(cost_before)} before (est.), {_money(cost_after)} after")
    elif migrate:
        parts.append(f"cost per 1k calls {_money(cost_after)} after (before not estimated)")
    elif cost_before is not None:
        parts.append(f"cost per 1k calls {_money(cost_before)} before (est.)")
    if closed_parity:
        parts.append(f"closed reference {closed_parity[0]} {closed_parity[1]:.1f}% on the same references")
    return "; ".join(parts) + "."


def decide(
    site: CallSite,
    evidence: Evidence,
    llm: LLM,
    thresholds: Thresholds | RunConfig,
    *,
    closed_parity: tuple[str, float] | None = None,
    price_lookup: PriceLookup | None = None,
    untried: str | None = None,
) -> Verdict:
    """MIGRATE or KEEP_CLOSED for a site with a measured holdout run (see the module docstring). `untried`:
    for a MIGRATE secured before a budget stop, which cheaper tiers were not fully tried and why."""
    task = site.contract.task_type
    tier, dev, holdout, n = Tier(evidence.tier), evidence.dev, evidence.holdout, evidence.iterations
    migrate = dev.passed and holdout.passed
    name = tier.value.capitalize()
    if migrate:
        result, gate = _phrase(holdout, task, thresholds)
    elif holdout.passed:  # dev never passed: the holdout pass is not a qualification
        result, gate = (
            f"dev {_phrase(dev, task, thresholds)[0]} (holdout {parity_pct(holdout, task):.1f}%)",
            "dev",
        )
    else:
        result, gate = _phrase(holdout, task, thresholds)
    cost_before, note, info = _cost_before(site.source_model, evidence, price_lookup or lookup_source_model)
    cost_after = holdout.metrics.cost_per_1k_calls_usd if migrate else None
    facts = {  # qualitative only: Super has no number to misattribute
        "call_site": {"file": site.file, "function": site.function, "task_type": task},
        "decision": "MIGRATE" if migrate else "KEEP_CLOSED",
        "tier": name,
        "outcome": "reached parity with the references on the held-out cases; "
        + (
            "it is the cheapest tier"
            if tier == TIER_ORDER[0]
            else "cheaper tiers were not fully tried before the run's budget ran out"
            if untried
            else "cheaper tiers did not"
        )
        if migrate
        else "no tier reached parity with the references; this tier came closest",
        "adapter_changes": evidence.changes,
        "holdout_failure_kinds": sorted(
            {c.error_category or "failed" for c in holdout.case_scores if not c.passed}
        ),
        "closed_reference_compared": closed_parity is not None,
    }
    clause = _clause(migrate, name, result, n, cost_before, cost_after, closed_parity)
    if untried:
        clause = f"{clause[:-1]}; {untried}."
    sentence = _sentence(llm, facts)
    return Verdict(
        site_id=site.id,
        decision="MIGRATE" if migrate else "KEEP_CLOSED",
        tier=tier if migrate else None,
        model=get_tier(tier).model_id if migrate else None,
        best_tier=None if migrate else tier,
        best_score=None if migrate else holdout.score,
        best_score_split=None if migrate else "holdout",
        iterations=n,
        latency_delta_ms=None,
        cost_before_per_1k=cost_before,
        cost_before_note=note,
        cost_after_per_1k=cost_after,
        sunset_risk=info.sunset_risk() if info else None,
        closed_reference_model=closed_parity[0] if closed_parity else None,
        closed_reference_parity_pct=closed_parity[1] if closed_parity else None,
        weak_references={
            c.case_id: c.reference_score
            for c in holdout.case_scores
            if c.reference_score is not None and c.reference_score < WEAK_REFERENCE
        },
        retires_on=info.retires_on() if info else None,
        price_evidence=info.evidence() if info else None,
        failed_gate=gate,
        reason=f"{sentence} {clause}" if sentence else clause,
        **_holdout_fields(holdout, task),
    )
