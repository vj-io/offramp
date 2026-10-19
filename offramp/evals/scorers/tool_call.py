"""tool_call scorer (PRD §6.5): same tool calls, in order, with semantically equal arguments.

Names compare exactly (they are identifiers). Arguments compare after normalisation: key order is
irrelevant, strings are trimmed and casefolded, and "5" / 5 / 5.0 are one value. Lists stay ordered
and extra or missing arguments are a miss. The text channel is ignored, except when neither side calls a
tool: then the candidate must have answered with some text, so a blank or truncated output is not scored
as a correct abstention.

Free-text arguments: an EvalCase tag `free_text_arg:<arg_name>` (one per argument) marks that argument
as prose whose wording is not behaviour (an escalation `reason`, say). Where the reference call has the
argument, the candidate must supply it as a non-blank string, and the values are not compared. Every
other argument, and the tool name, keeps the exact normalised comparison. The excluded arguments are
named in `CaseScore.detail` so a report can disclose them. No tag: nothing changes.
"""

from __future__ import annotations

from offramp.evals.scorers.json_schema import normalise
from offramp.schemas import CaseScore, Contract, EvalCase, ModelOutput

EMPTY_OUTPUT = "empty_output"  # neither side calls a tool and the candidate returned no text
NO_TOOL_CALL = "no_tool_call"  # baseline called a tool, candidate called none
WRONG_TOOL = "wrong_tool"  # different tool names, or a different number of calls
WRONG_ARGS = "wrong_args"

FREE_TEXT_TAG = "free_text_arg:"

_ABSENT = object()


def _fail(case: EvalCase, category: str, detail: str) -> CaseScore:
    return CaseScore(case_id=case.id, passed=False, score=0.0, detail=detail, error_category=category)


def _free_text_args(case: EvalCase) -> set[str]:
    names = (tag.removeprefix(FREE_TEXT_TAG).strip() for tag in case.tags if tag.startswith(FREE_TEXT_TAG))
    return {name for name in names if name}


def _disclose(detail: str, excluded: set[str]) -> str:
    if not excluded:
        return detail
    note = f"free-text arguments not value-compared: {', '.join(sorted(excluded))}"
    return "; ".join(filter(None, (detail, note)))


def score(case: EvalCase, baseline: ModelOutput, candidate: ModelOutput, contract: Contract) -> CaseScore:
    want, got = baseline.tool_calls, candidate.tool_calls
    want_names, got_names = [c.name for c in want], [c.name for c in got]
    if not want and not got and not (candidate.text or "").strip():
        return _fail(case, EMPTY_OUTPUT, "neither side called a tool and the candidate returned no text")
    if want and not got:
        return _fail(case, NO_TOOL_CALL, f"expected {want_names}, candidate called no tool")
    if want_names != got_names:
        return _fail(case, WRONG_TOOL, f"expected {want_names}, got {got_names}")
    free = _free_text_args(case)
    skips = [free & w.arguments.keys() for w in want]  # only where the reference call has the argument
    excluded = {f"{w.name}.{name}" for w, skip in zip(want, skips, strict=True) for name in skip}
    for index, (w, g, skip) in enumerate(zip(want, got, skips, strict=True)):
        for name in sorted(skip):
            value = g.arguments.get(name)
            if not isinstance(value, str) or not value.strip():
                problem = "is missing" if name not in g.arguments else "is blank or not a string"
                detail = f"call {index} {w.name}: free-text argument {name!r} {problem}"
                return _fail(case, WRONG_ARGS, _disclose(detail, excluded))
        want_args = normalise({k: v for k, v in w.arguments.items() if k not in skip}, numeric_strings=True)
        got_args = normalise({k: v for k, v in g.arguments.items() if k not in skip}, numeric_strings=True)
        if want_args != got_args:
            keys = sorted(
                k for k in want_args.keys() | got_args.keys()
                if want_args.get(k, _ABSENT) != got_args.get(k, _ABSENT)
            )
            detail = f"call {index} {w.name}: arguments differ at {', '.join(keys)}"
            return _fail(case, WRONG_ARGS, _disclose(detail, excluded))
    return CaseScore(case_id=case.id, passed=True, score=1.0, detail=_disclose("", excluded))
