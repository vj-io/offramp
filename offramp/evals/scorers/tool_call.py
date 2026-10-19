"""tool_call scorer (PRD §6.5): same tool calls, in order, with semantically equal arguments.

Names compare exactly (they are identifiers). Arguments compare after normalisation: key order is
irrelevant, strings are trimmed and casefolded, and "5" / 5 / 5.0 are one value. Lists stay ordered
and extra or missing arguments are a miss. The text channel is ignored, except when neither side calls a
tool: then the candidate must have answered with some text, so a blank or truncated output is not scored
as a correct abstention.
"""

from __future__ import annotations

from offramp.evals.scorers.json_schema import normalise
from offramp.schemas import CaseScore, Contract, EvalCase, ModelOutput

EMPTY_OUTPUT = "empty_output"  # neither side calls a tool and the candidate returned no text
NO_TOOL_CALL = "no_tool_call"  # baseline called a tool, candidate called none
WRONG_TOOL = "wrong_tool"  # different tool names, or a different number of calls
WRONG_ARGS = "wrong_args"

_ABSENT = object()


def _fail(case: EvalCase, category: str, detail: str) -> CaseScore:
    return CaseScore(case_id=case.id, passed=False, score=0.0, detail=detail, error_category=category)


def score(case: EvalCase, baseline: ModelOutput, candidate: ModelOutput, contract: Contract) -> CaseScore:
    want, got = baseline.tool_calls, candidate.tool_calls
    want_names, got_names = [c.name for c in want], [c.name for c in got]
    if not want and not got and not (candidate.text or "").strip():
        return _fail(case, EMPTY_OUTPUT, "neither side called a tool and the candidate returned no text")
    if want and not got:
        return _fail(case, NO_TOOL_CALL, f"expected {want_names}, candidate called no tool")
    if want_names != got_names:
        return _fail(case, WRONG_TOOL, f"expected {want_names}, got {got_names}")
    for index, (w, g) in enumerate(zip(want, got, strict=True)):
        want_args = normalise(w.arguments, numeric_strings=True)
        got_args = normalise(g.arguments, numeric_strings=True)
        if want_args != got_args:
            keys = sorted(
                k for k in want_args.keys() | got_args.keys()
                if want_args.get(k, _ABSENT) != got_args.get(k, _ABSENT)
            )
            return _fail(case, WRONG_ARGS, f"call {index} {w.name}: arguments differ at {', '.join(keys)}")
    return CaseScore(case_id=case.id, passed=True, score=1.0)
