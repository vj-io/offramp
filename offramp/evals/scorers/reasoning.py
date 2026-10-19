"""reasoning scorer (PRD §6.5): the final answers must match, nothing else is compared.

Final answer = the last `ANSWER:` line, else the last number in the text. Two numeric answers match
within 1e-6 relative (no absolute slack, so 0 only matches 0); anything else must match exactly
(case-sensitive, after trimming emphasis markers and a trailing period).
"""

from __future__ import annotations

import math
import re

from offramp.schemas import CaseScore, Contract, EvalCase, ModelOutput

NO_ANSWER = "no_answer"  # candidate has no extractable final answer
BASELINE_UNUSABLE = "baseline_unusable"
WRONG_ANSWER = "wrong_answer"

REL_TOL = 1e-6

_ANSWER_LINE = re.compile(r"^[ \t>*_#-]*answer[ \t*_]*:(.*)$", re.IGNORECASE)
_NUMBER = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?|\.[0-9]+(?:[eE][-+]?[0-9]+)?"
_NUMBER_ONLY = re.compile(rf"[-+]?(?:{_NUMBER})")
_LAST_NUMBER = re.compile(rf"(?<![\w.])[-+]?(?:{_NUMBER})")  # "2024-05-01" is not 2024 and -05 and -01
_EMPHASIS = " \t*_`\"'"


def _clean(answer: str) -> str:
    return answer.strip(_EMPHASIS)


def _final_answer(text: str) -> str | None:
    text = text.replace("\u2212", "-")  # unicode minus
    for line in reversed(text.splitlines()):
        found = _ANSWER_LINE.match(line)
        if found and (answer := _clean(found.group(1))):
            return answer
    numbers = _LAST_NUMBER.findall(text)
    return numbers[-1] if numbers else None


def _typed(answer: str) -> tuple[str, float | str]:
    """("num", value) when the answer is one finite number, else ("str", text)."""
    candidate = answer.removeprefix("$").removesuffix(".")
    if _NUMBER_ONLY.fullmatch(candidate):
        value = float(candidate.replace(",", ""))
        if math.isfinite(value):
            return "num", value
    return "str", answer.removesuffix(".")


def _same(a: tuple[str, float | str], b: tuple[str, float | str]) -> bool:
    if a[0] == b[0] == "num":
        return math.isclose(a[1], b[1], rel_tol=REL_TOL, abs_tol=0.0)
    return a == b  # same-kind strings compare exactly; a number never equals a string


def _fail(case: EvalCase, category: str, detail: str) -> CaseScore:
    return CaseScore(case_id=case.id, passed=False, score=0.0, detail=detail, error_category=category)


def score(case: EvalCase, baseline: ModelOutput, candidate: ModelOutput, contract: Contract) -> CaseScore:
    got = _final_answer(candidate.text) if candidate.text else None
    if got is None:
        return _fail(case, NO_ANSWER, "candidate has no ANSWER line and no number")
    want = _final_answer(baseline.text) if baseline.text else None
    if want is None:
        return _fail(case, BASELINE_UNUSABLE, "baseline has no ANSWER line and no number")
    if _same(_typed(want), _typed(got)):
        return CaseScore(case_id=case.id, passed=True, score=1.0)
    return _fail(case, WRONG_ANSWER, f"expected {want!r}, got {got!r}")
