"""Site-level pass rule (PRD §6.5, Global Constraint 11).

Takes per-case scores for one (site, tier, split) and decides whether the site passes. json_extraction
cases carry their field match (0..1) in `CaseScore.score`; free_text cases carry judge rubric scores
(1-5, computed elsewhere) in it.

Never passes on zero cases. Callers must hand in a CaseScore for *every* case, errored ones
included (see offramp.evals.scorers.score_case); a dropped case is invisible here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from offramp.evals.scorers.json_schema import INVALID_CATEGORIES
from offramp.schemas import CaseScore, RunConfig, TaskType

CANDIDATE_ERROR = "candidate_error"  # the candidate call itself failed: no output to validate
JUDGE_MARGIN = 0.3  # free_text with a reference mean: judge mean >= reference mean - 0.3 ...
JUDGE_FLOOR = 4.0  # ... and >= 4.0 on the 1-5 rubric ...
JUDGE_CASE_FLOOR = 3.0  # ... and, on holdout, no case below 3.0 (a mean can hide a plainly wrong answer)
WEAK_REFERENCE = 3.0  # a free-text reference the judge scores below this is flagged: shown, never dropped
RUBRIC_MIN, RUBRIC_MAX = 1.0, 5.0

_NOT_VALID = INVALID_CATEGORIES | {CANDIDATE_ERROR}
_EPS = 1e-9  # float slack so a mean that is mathematically on the boundary (4.5 - 0.3 == 4.2) passes


@dataclass(frozen=True)
class Thresholds:
    """Same field names and defaults as RunConfig, so a RunConfig works wherever Thresholds does."""

    threshold: float = 0.95  # json_extraction, classification, tool_call
    reasoning_threshold: float = 0.90  # reasoning


DEFAULT_THRESHOLDS = Thresholds()


def _at_least(value: float, bar: float) -> bool:
    return value >= bar or math.isclose(value, bar, rel_tol=0.0, abs_tol=_EPS)


def wilson_lower(passed: int, n: int, z: float = 1.96) -> float:
    """Wilson score 95% lower bound of a pass rate of `passed` out of `n` cases (0..1): how low the true
    rate could plausibly be. 9/9 passing is >= 0.700, 3/3 only >= 0.438. ValueError on n < 1."""
    if n < 1 or not 0 <= passed <= n:
        raise ValueError(f"no pass rate for {passed} of {n} cases")
    p, z2 = passed / n, z * z
    centre = p + z2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - spread) / (1 + z2 / n))


def lower_pct(case_scores: Sequence[CaseScore]) -> float | None:
    """wilson_lower of the share of passing cases, in percent rounded down to 0.1 so it never overstates the
    bound; None for no cases."""
    if not case_scores:
        return None
    return math.floor(1000 * wilson_lower(sum(c.passed for c in case_scores), len(case_scores)) + 1e-9) / 10


def site_pass(
    task_type: TaskType,
    case_scores: Sequence[CaseScore],
    thresholds: Thresholds | RunConfig = DEFAULT_THRESHOLDS,
    baseline_judge_mean: float | None = None,
    *,
    split: str | None = None,
) -> tuple[bool, float]:
    """(passed, score).

    classification / tool_call / reasoning: pass when the fraction of passing cases >= the threshold
    (reasoning_threshold for reasoning); score is that fraction.
    json_extraction: pass when every case is schema-valid AND the fraction of passing cases >= threshold
    AND the mean field match >= threshold; score is the mean field match.
    free_text: pass when the judge mean is >= 4.0 and, when `baseline_judge_mean` (the references' mean on
    the same rubric) is given, also >= baseline_judge_mean - 0.3; on the holdout split (`split="holdout"`,
    the verdict's) no case may score below 3.0 either. Dev keeps the mean rule: it only decides whether
    holdout runs. Score is the judge mean. Judge scores and the reference mean outside 1..5 raise ValueError:
    they cannot be rubric scores, so they never count.
    """
    if not case_scores:
        return False, 0.0
    if task_type == "free_text":
        return _free_text(case_scores, baseline_judge_mean, holdout=split == "holdout")
    fraction = sum(c.passed for c in case_scores) / len(case_scores)
    bar = thresholds.reasoning_threshold if task_type == "reasoning" else thresholds.threshold
    ok = _at_least(fraction, bar)
    if task_type == "json_extraction":
        mean_match = math.fsum(c.score for c in case_scores) / len(case_scores)
        all_valid = all(c.error_category not in _NOT_VALID for c in case_scores)
        ok = ok and all_valid and _at_least(mean_match, bar)
        return ok, mean_match
    return ok, fraction


def _free_text(
    case_scores: Sequence[CaseScore], baseline_judge_mean: float | None, holdout: bool
) -> tuple[bool, float]:
    for c in case_scores:
        if not RUBRIC_MIN <= c.score <= RUBRIC_MAX:  # also false for NaN
            raise ValueError(f"judge score {c.score!r} for case {c.case_id} is outside the 1-5 rubric")
    if baseline_judge_mean is not None and not RUBRIC_MIN <= baseline_judge_mean <= RUBRIC_MAX:
        raise ValueError(f"baseline_judge_mean {baseline_judge_mean!r} is outside the 1-5 rubric")
    mean = math.fsum(c.score for c in case_scores) / len(case_scores)
    ok = _at_least(mean, JUDGE_FLOOR)
    if holdout:
        ok = ok and all(_at_least(c.score, JUDGE_CASE_FLOOR) for c in case_scores)
    if baseline_judge_mean is not None:
        ok = ok and _at_least(mean, baseline_judge_mean - JUDGE_MARGIN)
    return ok, mean
