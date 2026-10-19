"""Judge Agent (PRD §6.5, Global Constraint 20): Nemotron Ultra scores free-text answers 1-5.

Rubric (absolute, against the evidence = the case's full input: instructions, KB snippets or thread):
groundedness, completeness, format; an answer's score is the mean of the three. With a reference text for a
case (human or labelled closed-model reference) the judge sees both answers blind as A and B and is asked
twice, A/B then B/A; scores are averaged over the two orderings, and the swap gives position consistency.

Batched: one Ultra call (role "judge") scores every case of one batch per ordering, so a (site, tier, split)
costs 2 Ultra agent calls when references exist, 1 when not (the 40-call cap, Global Constraint 8).

Scores are always inside 1..5: anything the judge did not score validly (blank answer, invalid reply, case
left out) counts as 1.0 for the candidate. Never 0, never clamped.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NamedTuple

import openai
from pydantic import BaseModel

from offramp.evals.scorers.aggregate import JUDGE_FLOOR, JUDGE_MARGIN, RUBRIC_MIN, _at_least
from offramp.schemas import CaseScore, EvalCase, ModelOutput, Tier
from offramp.tf.client import LLM

JUDGE_ROLE = "judge"
UNJUDGED = RUBRIC_MIN  # 1.0
EMPTY_OUTPUT = "empty_output"  # no answer to judge (blank text, or no output at all)
NOT_SCORED = "unjudged"  # the judge returned no valid score for the case
LOW_SCORE = "low_judge_score"

MAX_TOKENS_BASE, MAX_TOKENS_PER_CASE = 400, 150  # thinking off: JSON only, ~90 tokens per scored case

Score = Literal[1, 2, 3, 4, 5]


class Rubric(BaseModel):
    groundedness: Score
    completeness: Score
    format: Score

    def mean(self) -> float:
        return (self.groundedness + self.completeness + self.format) / 3


class SingleItem(BaseModel):
    case: int
    note: str
    a: Rubric


class PairItem(BaseModel):
    case: int
    note: str
    a: Rubric
    b: Rubric


class SingleBatch(BaseModel):
    scores: list[SingleItem]


class PairBatch(BaseModel):
    scores: list[PairItem]


class JudgeScore(BaseModel):
    case_id: str
    candidate: float  # 1..5
    reference: float | None = None  # 1..5 when a reference was judged in at least one ordering
    consistent: bool | None = None  # pairwise and judged in both orderings: same preference after the swap
    detail: str = ""
    error_category: str | None = None  # EMPTY_OUTPUT / NOT_SCORED; None when judged


class JudgeResult(BaseModel):
    scores: list[JudgeScore]
    reference_mean: float | None = None  # mean reference score, None without judged references
    consistency: float | None = None  # fraction of swapped pairs that kept their preference


class Agreement(NamedTuple):
    n: int  # cases scored by both the judge and a deterministic scorer
    agreement: float  # fraction where the judge's per-case verdict equals the deterministic one
    consistency: float | None  # position-swap consistency on those cases


SYSTEM = """You grade answers written by an AI assistant.

For every case you get the evidence: the full input the assistant received (its instructions, any \
knowledge-base snippets or conversation thread, and the request). Then you get {answers}, written for \
that input.

Score each answer on three criteria, each an integer from 1 to 5:
- groundedness: every factual claim is supported by the evidence. 5 = fully supported; 3 = one unsupported \
or doubtful claim; 1 = invents facts or contradicts the evidence.
- completeness: it does what the request asks, with the details from the evidence the reader needs. \
5 = complete; 3 = misses something that matters; 1 = does not address the request.
- format: it follows the format, length, tone and language the instructions ask for. 5 = exactly; \
3 = noticeable deviations; 1 = ignores them.

Grade each answer on its own merits against the evidence. Position and length are not quality. Identical \
answers get identical scores.
Return one entry per case: "case" is the case number; write "note" before the scores, one short sentence \
naming the main problem or "no issues"; then {keys}."""

_ANSWERS = {False: "one answer, A", True: "two answers, A and B"}
_KEYS = {False: 'the scores of answer A in "a"', True: 'the scores of answer A in "a" and of answer B in "b"'}


def _blank(out: ModelOutput | None) -> bool:
    return out is None or not (out.text or "").strip()


def _evidence(case: EvalCase) -> str:
    parts = []
    for m in case.input_messages:
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        parts.append(f"[{m.get('role', 'user')}]\n{text}")
    return "\n\n".join(parts)


def _prompt(items: list[tuple[EvalCase, str, str | None]]) -> str:
    blocks = []
    for n, (case, a, b) in enumerate(items, 1):
        block = f"Case {n}\n<evidence>\n{_evidence(case)}\n</evidence>\n<answer_a>\n{a}\n</answer_a>"
        if b is not None:
            block += f"\n<answer_b>\n{b}\n</answer_b>"
        blocks.append(block)
    return "\n\n".join(blocks)


def _ask(client: LLM, items: list[tuple[EvalCase, str, str | None]], paired: bool) -> dict[int, Any]:
    """One batched Ultra call: {1-based case number: item}; {} if the call failed or the reply is invalid."""
    try:
        result = client.chat(
            role=JUDGE_ROLE,
            tier=Tier.ULTRA,
            messages=[
                {"role": "system", "content": SYSTEM.format(answers=_ANSWERS[paired], keys=_KEYS[paired])},
                {"role": "user", "content": _prompt(items)},
            ],
            max_tokens=MAX_TOKENS_BASE + MAX_TOKENS_PER_CASE * len(items),
            temperature=0.0,
            response_model=PairBatch if paired else SingleBatch,
            reasoning="off",
        )
    except openai.APIError:  # the judge call failed after the SDK's retries: nothing judged
        return {}
    if result.parsed is None:
        return {}
    by_case: dict[int, Any] = {}
    for item in result.parsed.scores:
        if 1 <= item.case <= len(items):
            by_case.setdefault(item.case, item)  # first entry wins
    return by_case


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def judge(
    client: LLM,
    cases: Sequence[EvalCase],
    candidates: Mapping[str, ModelOutput | None],
    references: Mapping[str, ModelOutput | None] | None = None,
) -> JudgeResult:
    """Score each case's candidate (and its reference, when the reference has text). Case order is kept."""
    references = references or {}
    scores: dict[str, JudgeScore] = {}
    pairs: list[tuple[EvalCase, str, str]] = []
    singles: list[tuple[EvalCase, str, None]] = []
    for case in cases:
        cand, ref = candidates.get(case.id), references.get(case.id)
        if _blank(cand):
            scores[case.id] = JudgeScore(
                case_id=case.id, candidate=UNJUDGED, detail="no answer to judge", error_category=EMPTY_OUTPUT
            )
        elif _blank(ref):
            singles.append((case, cand.text, None))
        else:
            pairs.append((case, cand.text, ref.text))

    if pairs:
        ab = _ask(client, pairs, paired=True)  # candidate is A
        ba = _ask(client, [(c, r, a) for c, a, r in pairs], paired=True)  # reference is A
        for n, (case, _, _) in enumerate(pairs, 1):
            cand_scores, ref_scores, prefs, notes = [], [], [], []
            for got, cand_key in ((ab.get(n), "a"), (ba.get(n), "b")):
                if got is None:
                    cand_scores.append(UNJUDGED)
                    continue
                c = getattr(got, cand_key).mean()
                r = (got.b if cand_key == "a" else got.a).mean()
                cand_scores.append(c)
                ref_scores.append(r)
                prefs.append(_sign(c - r))
                notes.append(
                    f"candidate is {cand_key.upper()}: {got.note.strip()}" if got.note.strip() else ""
                )
            candidate = math.fsum(cand_scores) / 2
            reference = math.fsum(ref_scores) / len(ref_scores) if ref_scores else None
            scores[case.id] = JudgeScore(
                case_id=case.id,
                candidate=candidate,
                reference=reference,
                consistent=prefs[0] == prefs[1] if len(prefs) == 2 else None,
                detail=_detail(candidate, reference, notes),
                error_category=NOT_SCORED if not notes else None,
            )

    if singles:
        got_by_case = _ask(client, singles, paired=False)
        for n, (case, _, _) in enumerate(singles, 1):
            got = got_by_case.get(n)
            if got is None:
                scores[case.id] = JudgeScore(
                    case_id=case.id,
                    candidate=UNJUDGED,
                    detail="judge returned no score",
                    error_category=NOT_SCORED,
                )
            else:
                scores[case.id] = JudgeScore(
                    case_id=case.id, candidate=got.a.mean(), detail=_detail(got.a.mean(), None, [got.note])
                )

    ordered = [scores[c.id] for c in cases]
    refs = [s.reference for s in ordered if s.reference is not None]
    swaps = [s.consistent for s in ordered if s.consistent is not None]
    return JudgeResult(
        scores=ordered,
        reference_mean=math.fsum(refs) / len(refs) if refs else None,
        consistency=sum(swaps) / len(swaps) if swaps else None,
    )


def _detail(candidate: float, reference: float | None, notes: list[str]) -> str:
    text = f"judge {candidate:.2f}" + (f" vs reference {reference:.2f}" if reference is not None else "")
    note = next((n.strip() for n in notes if n.strip()), "")
    return f"{text}; {note}" if note else text


def case_passes(score: JudgeScore) -> bool:
    """The site rule applied to one case: >= 4.0, and >= reference - 0.3 when the reference was judged."""
    ok = _at_least(score.candidate, JUDGE_FLOOR)
    if score.reference is not None:
        ok = ok and _at_least(score.candidate, score.reference - JUDGE_MARGIN)
    return ok


def judge_agreement(deterministic: Sequence[CaseScore], judged: JudgeResult) -> Agreement:
    """§12.3 sanity check on a site that also has deterministic scores (target: >= 85% agreement, >= 90%
    position-swap consistency). Compares per-case verdicts on the cases both scored."""
    det = {c.case_id: c.passed for c in deterministic}
    shared = [s for s in judged.scores if s.case_id in det]
    if not shared:
        raise ValueError("no case was scored by both the judge and a deterministic scorer")
    agree = sum(case_passes(s) == det[s.case_id] for s in shared)
    swaps = [s.consistent for s in shared if s.consistent is not None]
    return Agreement(
        n=len(shared),
        agreement=agree / len(shared),
        consistency=sum(swaps) / len(swaps) if swaps else None,
    )
