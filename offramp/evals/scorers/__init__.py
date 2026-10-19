"""Deterministic scorers (PRD §6.5). `score_case` routes a case to the scorer for its contract."""

from __future__ import annotations

from offramp.evals.scorers import aggregate, json_schema, label, reasoning, tool_call
from offramp.schemas import CaseScore, Contract, EvalCase, ModelOutput

_SCORERS = {
    "classification": label.score,
    "json_extraction": json_schema.score,
    "tool_call": tool_call.score,
    "reasoning": reasoning.score,
}

__all__ = ["aggregate", "json_schema", "label", "reasoning", "score_case", "tool_call"]


def score_case(
    case: EvalCase, baseline: ModelOutput, candidate: ModelOutput | None, contract: Contract
) -> CaseScore:
    """Score one case. `candidate=None` (the call errored) is a scored failure, never a dropped case."""
    scorer = _SCORERS.get(contract.task_type)
    if scorer is None:
        raise ValueError(f"{contract.task_type} cases are scored by the judge, not a deterministic scorer")
    if candidate is None:
        return CaseScore(
            case_id=case.id,
            passed=False,
            score=0.0,
            detail="candidate call failed",
            error_category=aggregate.CANDIDATE_ERROR,
        )
    return scorer(case, baseline, candidate, contract)
