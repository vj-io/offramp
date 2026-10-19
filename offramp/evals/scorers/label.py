"""classification scorer (PRD §6.5): the candidate's label must equal the baseline's label.

Both sides go through the same normalisation, so only formatting differences are forgiven. There is
no substring or fuzzy matching: "not spam" is not "spam", and prose around a label is a miss.
"""

from __future__ import annotations

import unicodedata

from offramp.schemas import CaseScore, Contract, EvalCase, ModelOutput

EMPTY_OUTPUT = "empty_output"
BASELINE_UNUSABLE = "baseline_unusable"
INVALID_LABEL = "invalid_label"  # candidate is outside contract.labels (and not what the baseline said)
WRONG_LABEL = "wrong_label"

_WRAP = "\"'`*_"  # quotes and markdown emphasis wrapped around a label


def normalise_label(text: str) -> str:
    """Strip, drop wrapping quotes/markdown, casefold, drop trailing punctuation.

    Never returns "" for non-blank input: a label that is only punctuation ("?") is kept as is.
    """
    base = text.strip()
    folded = (base.strip(_WRAP).strip() or base).casefold()
    core = folded
    while core and unicodedata.category(core[-1]).startswith("P"):
        core = core[:-1]
    return core.rstrip() or folded


def _fail(case: EvalCase, category: str, detail: str) -> CaseScore:
    return CaseScore(case_id=case.id, passed=False, score=0.0, detail=detail, error_category=category)


def score(case: EvalCase, baseline: ModelOutput, candidate: ModelOutput, contract: Contract) -> CaseScore:
    if candidate.text is None or not candidate.text.strip():
        return _fail(case, EMPTY_OUTPUT, "candidate returned no label")
    if baseline.text is None or not baseline.text.strip():
        return _fail(case, BASELINE_UNUSABLE, "baseline has no label to compare against")
    labels = {normalise_label(name): name for name in contract.labels}
    got, want = normalise_label(candidate.text), normalise_label(baseline.text)
    if got == want:
        return CaseScore(case_id=case.id, passed=True, score=1.0)
    detail = f"expected {labels.get(want, want)!r}, got {labels.get(got, got)!r}"
    return _fail(case, INVALID_LABEL if labels and got not in labels else WRONG_LABEL, detail)
