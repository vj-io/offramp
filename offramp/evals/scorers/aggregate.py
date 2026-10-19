"""Site-level pass rule (PRD §6.5, Global Constraint 11).

Takes per-case scores for one (site, tier, split) and decides whether the site passes. free_text
cases carry judge rubric scores (1-5, computed elsewhere) in `CaseScore.score`.

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
JUDGE_MARGIN = 0.3  # free_text: judge mean >= baseline mean - 0.3 ...
JUDGE_FLOOR = 4.0  # ... and >= 4.0 on the 1-5 rubric

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


def site_pass(
    task_type: TaskType,
    case_scores: Sequence[CaseScore],
    thresholds: Thresholds | RunConfig = DEFAULT_THRESHOLDS,
    baseline_judge_mean: float | None = None,
) -> tuple[bool, float]:
    """(passed, score): score is the passing fraction 0..1, or the judge mean 1..5 for free_text."""
    if not case_scores:
        return False, 0.0
    if task_type == "free_text":
        if baseline_judge_mean is None:
            raise ValueError("free_text needs baseline_judge_mean (the judge's mean for the baseline)")
        mean = math.fsum(c.score for c in case_scores) / len(case_scores)
        ok = _at_least(mean, baseline_judge_mean - JUDGE_MARGIN) and _at_least(mean, JUDGE_FLOOR)
        return ok, mean
    fraction = sum(c.passed for c in case_scores) / len(case_scores)
    bar = thresholds.reasoning_threshold if task_type == "reasoning" else thresholds.threshold
    ok = _at_least(fraction, bar)
    if task_type == "json_extraction":
        ok = ok and all(c.error_category not in _NOT_VALID for c in case_scores)
    return ok, fraction
