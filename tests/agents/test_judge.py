"""Judge Agent (PRD §6.5, GC20): Ultra scores free text 1-5 against the evidence, batched, position-swapped.

Two kinds of test:
- FakeLLM tests pin the mechanics: batching, position swap, score mapping, the 1.0 floor for anything the
  judge did not score, and the guards that keep every score inside the 1-5 rubric.
- Cassette tests replay real Nemotron Ultra responses recorded once with `pytest --live` into
  tests/fixtures/cassettes/judge/. The cases they judge are SYNTHETIC TEST INPUTS written for this test
  (made-up KB snippets, questions and answers), not product data.
"""

from pathlib import Path

import httpx
import openai
import pytest

from offramp.agents.judge import (
    JUDGE_ROLE,
    Agreement,
    JudgeResult,
    JudgeScore,
    case_passes,
    judge,
    judge_agreement,
)
from offramp.evals.errors import ProviderError
from offramp.schemas import CaseScore, EvalCase, ModelOutput, Tier
from offramp.tf import Budget, BudgetExceeded, FakeLLM, TFClient

CASSETTES = Path(__file__).parents[1] / "fixtures" / "cassettes" / "judge"


def _case(n: int, user: str = "question") -> EvalCase:
    return EvalCase(
        id=f"cs_04:{n:03d}",
        site_id="cs_04",
        input_messages=[
            {"role": "system", "content": "Answer from the KB snippets only."},
            {"role": "user", "content": f"{user} #{n}"},
        ],
        expected_kind="text",
        split="dev",
    )


def _r(g: int, c: int, f: int) -> dict:
    return {"groundedness": g, "completeness": c, "format": f}


def _pair(n: int, a: dict, b: dict, note: str = "ok") -> dict:
    return {"case": n, "note": note, "a": a, "b": b}


def _single(n: int, a: dict, note: str = "ok") -> dict:
    return {"case": n, "note": note, "a": a}


def _user_prompt(call: dict) -> str:
    return call["messages"][-1]["content"]


# ------------------------------------------------------------------ pairwise, position swap


def test_pairwise_judges_both_orderings_and_averages():
    cases = [_case(1), _case(2)]
    cands = {c.id: ModelOutput(text=f"candidate answer {i}") for i, c in enumerate(cases, 1)}
    refs = {c.id: ModelOutput(text=f"reference answer {i}") for i, c in enumerate(cases, 1)}
    fake = FakeLLM().add(
        JUDGE_ROLE,
        # A/B: candidate is A
        {"scores": [_pair(1, _r(5, 5, 5), _r(4, 4, 4)), _pair(2, _r(3, 3, 3), _r(5, 5, 5))]},
        # B/A: reference is A
        {"scores": [_pair(1, _r(5, 5, 5), _r(4, 4, 4)), _pair(2, _r(3, 3, 3), _r(4, 4, 4))]},
    )
    result = judge(fake, cases, cands, refs)

    assert len(fake.calls) == 2
    ab, ba = (_user_prompt(c) for c in fake.calls)
    assert ab.index("candidate answer 1") < ab.index("reference answer 1")  # candidate in position A
    assert ba.index("reference answer 1") < ba.index("candidate answer 1")  # swapped
    s1, s2 = result.scores
    # case 1: A/B cand 5 ref 4; B/A ref (A) 5, cand (B) 4 -> cand 4.5, ref 4.5; preference flipped
    assert (s1.case_id, s1.candidate, s1.reference, s1.consistent) == ("cs_04:001", 4.5, 4.5, False)
    # case 2: A/B cand 3 ref 5; B/A ref (A) 3, cand (B) 4 -> cand 3.5, ref 4.0; preference flipped too
    assert (s2.candidate, s2.reference, s2.consistent) == (3.5, 4.0, False)
    assert result.reference_mean == pytest.approx(4.25)
    assert result.consistency == 0.0


def test_pairwise_consistency_when_preference_survives_swap():
    cases = [_case(1), _case(2)]
    cands = {c.id: ModelOutput(text="cand") for c in cases}
    refs = {c.id: ModelOutput(text="ref") for c in cases}
    fake = FakeLLM().add(
        JUDGE_ROLE,
        {"scores": [_pair(1, _r(2, 2, 2), _r(5, 5, 5)), _pair(2, _r(5, 5, 5), _r(5, 5, 5))]},
        {"scores": [_pair(1, _r(5, 5, 5), _r(1, 2, 3)), _pair(2, _r(4, 4, 4), _r(4, 4, 4))]},
    )
    result = judge(fake, cases, cands, refs)
    assert [s.consistent for s in result.scores] == [True, True]  # ref wins twice; tie twice
    assert result.consistency == 1.0
    assert result.scores[0].candidate == pytest.approx(2.0)
    assert result.scores[0].reference == pytest.approx(5.0)


def test_criteria_mean_is_the_score():
    fake = FakeLLM().add(JUDGE_ROLE, {"scores": [_single(1, _r(5, 4, 2))]})
    result = judge(fake, [_case(1)], {"cs_04:001": ModelOutput(text="x")})
    assert result.scores[0].candidate == pytest.approx(11 / 3)


# ------------------------------------------------------------------ absolute rubric (no reference)


def test_absolute_mode_without_reference_is_one_call_and_no_answer_b():
    cases = [_case(1), _case(2)]
    cands = {c.id: ModelOutput(text=f"answer {c.id}") for c in cases}
    fake = FakeLLM().add(JUDGE_ROLE, {"scores": [_single(1, _r(5, 5, 5)), _single(2, _r(4, 3, 4))]})
    result = judge(fake, cases, cands, references={c.id: ModelOutput(text=None) for c in cases})

    assert len(fake.calls) == 1
    assert "answer_b" not in _user_prompt(fake.calls[0])
    assert [s.candidate for s in result.scores] == [5.0, pytest.approx(11 / 3)]
    assert all(s.reference is None and s.consistent is None for s in result.scores)
    assert result.reference_mean is None and result.consistency is None


def test_mixed_batch_pairs_cases_with_a_reference_only():
    cases = [_case(1), _case(2)]
    cands = {c.id: ModelOutput(text="cand") for c in cases}
    refs = {"cs_04:001": ModelOutput(text="reference text")}
    fake = FakeLLM().add(
        JUDGE_ROLE,
        {"scores": [_pair(1, _r(5, 5, 5), _r(5, 5, 5))]},
        {"scores": [_pair(1, _r(5, 5, 5), _r(5, 5, 5))]},
        {"scores": [_single(1, _r(2, 2, 2))]},
    )
    result = judge(fake, cases, cands, refs)
    assert len(fake.calls) == 3
    assert "question #1" not in _user_prompt(fake.calls[2])  # the absolute call holds only case 2
    assert "question #2" in _user_prompt(fake.calls[2])
    assert [s.reference for s in result.scores] == [5.0, None]
    assert result.scores[1].candidate == 2.0


# ------------------------------------------------------------------ evidence and request shape


def test_prompt_carries_evidence_and_call_is_ultra_judge_thinking_off():
    case = _case(1, user="Refunds within 30 days? KB: annual plans refundable for 30 days")
    fake = FakeLLM().add(JUDGE_ROLE, {"scores": [_single(1, _r(5, 5, 5))]})
    judge(fake, [case], {case.id: ModelOutput(text="Yes, within 30 days.")})
    call = fake.calls[0]
    assert call["role"] == "judge" and call["tier"] == Tier.ULTRA
    assert call["reasoning"] == "off" and call["temperature"] == 0.0
    assert call["response_model"] is not None
    prompt = _user_prompt(call)
    assert "Answer from the KB snippets only." in prompt  # system message of the case = evidence
    assert "annual plans refundable for 30 days" in prompt
    assert "Yes, within 30 days." in prompt
    assert "groundedness" in call["messages"][0]["content"]


def _all_scored(n: int) -> dict:
    return {"scores": [_single(i, _r(5, 5, 5)) for i in range(1, n + 1)]}


def test_max_tokens_grows_with_batch():
    small, big = FakeLLM(), FakeLLM()
    small.add(JUDGE_ROLE, _all_scored(1))
    big.add(JUDGE_ROLE, _all_scored(20))
    judge(small, [_case(1)], {"cs_04:001": ModelOutput(text="a")})
    cases = [_case(i) for i in range(1, 21)]
    judge(big, cases, {c.id: ModelOutput(text="a") for c in cases})
    assert len(big.calls) == 1  # a batch far below the prompt limit is one call
    assert big.calls[0]["max_tokens"] > small.calls[0]["max_tokens"] + 19 * 60


def test_batch_over_the_prompt_limit_is_chunked(monkeypatch):
    import offramp.agents.judge as judge_module

    cases = [_case(i, user="x" * 400) for i in range(1, 6)]  # ~100+ tokens of evidence each
    monkeypatch.setattr(judge_module, "MAX_PROMPT_TOKENS", 300)  # stands in for 80k
    fake = FakeLLM().add(JUDGE_ROLE, *[_all_scored(2), _all_scored(2), {"scores": [_single(1, _r(2, 2, 2))]}])
    result = judge(fake, cases, {c.id: ModelOutput(text=f"answer {c.id}") for c in cases})

    assert len(fake.calls) == 3  # 2 + 2 + 1 cases
    last = _user_prompt(fake.calls[2])
    assert last.startswith("Case 1\n") and " #5\n" in last and " #4\n" not in last
    assert [s.candidate for s in result.scores] == [5.0, 5.0, 5.0, 5.0, 2.0]  # numbering restarts per chunk


def test_default_prompt_limit_is_80k_tokens():
    import offramp.agents.judge as judge_module

    assert judge_module.MAX_PROMPT_TOKENS == 80_000


# ------------------------------------------------------------------ never outside 1..5, never 0


def test_blank_or_missing_candidate_scores_one_and_is_not_sent():
    cases = [_case(1), _case(2), _case(3)]
    cands = {"cs_04:001": ModelOutput(text="   "), "cs_04:002": None, "cs_04:003": ModelOutput(text="real")}
    fake = FakeLLM().add(JUDGE_ROLE, {"scores": [_single(1, _r(5, 5, 5))]})
    result = judge(fake, cases, cands)
    assert len(fake.calls) == 1
    assert "question #1" not in _user_prompt(fake.calls[0])
    assert "question #3" in _user_prompt(fake.calls[0])
    assert [s.candidate for s in result.scores] == [1.0, 1.0, 5.0]
    assert result.scores[0].error_category == "empty_output"


def test_nothing_to_judge_makes_no_call():
    fake = FakeLLM()  # no canned reply: a call would raise
    result = judge(fake, [_case(1)], {"cs_04:001": None})
    assert fake.calls == [] and result.scores[0].candidate == 1.0


def test_case_the_judge_left_out_twice_is_a_judge_failure_not_a_score():
    cases = [_case(1), _case(2)]
    fake = FakeLLM().add(JUDGE_ROLE, {"scores": [_single(2, _r(5, 5, 5))]}, {"scores": []})
    with pytest.raises(ProviderError, match="cs_04:001"):
        judge(fake, cases, {c.id: ModelOutput(text="x") for c in cases})
    assert len(fake.calls) == 2


def test_a_case_the_judge_skipped_is_asked_again_alone():
    """Live: an Ultra batch skipped one Japanese case; the run must not fail on a judge's omission."""
    cases = [_case(1), _case(2), _case(3)]
    fake = FakeLLM().add(
        JUDGE_ROLE,
        {"scores": [_single(1, _r(5, 5, 5)), _single(3, _r(2, 2, 2))]},
        {"scores": [_single(1, _r(4, 4, 4))]},  # the re-ask numbers its one case 1
    )
    result = judge(fake, cases, {c.id: ModelOutput(text="x") for c in cases})
    assert len(fake.calls) == 2
    again = _user_prompt(fake.calls[1])  # only the skipped case (case 2) is asked again
    assert "question #2" in again and "question #1" not in again and "question #3" not in again
    assert [s.candidate for s in result.scores] == [5.0, 4.0, 2.0]


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        {"scores": [_single(1, _r(7, 5, 5))]},  # 7 is off the rubric: invalid, never clamped
        {"scores": [_single(1, _r(0, 5, 5))]},  # 0 too
        {"scores": [{"case": 1, "note": "x", "a": {"groundedness": 5}}]},  # criteria missing
        openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid")),  # call timed out
        openai.InternalServerError(  # 5xx after the SDK's retries
            "Error code: 502",
            response=httpx.Response(502, request=httpx.Request("POST", "https://x")),
            body=None,
        ),
        openai.InternalServerError(
            "Error code: 500",
            response=httpx.Response(500, request=httpx.Request("POST", "https://x")),
            body=None,
        ),
    ],
)
def test_judge_side_failure_raises_never_scores(reply):
    # the judge is not the model under test: its timeouts, outages, bad replies say nothing of the candidate
    fake = FakeLLM().add(JUDGE_ROLE, reply)
    with pytest.raises(ProviderError, match="judge"):
        judge(fake, [_case(1)], {"cs_04:001": ModelOutput(text="x")})


def test_pairwise_ordering_the_judge_skipped_raises():
    case = _case(1)
    fake = FakeLLM().add(
        JUDGE_ROLE,
        {"scores": [_pair(1, _r(5, 5, 5), _r(5, 5, 5))]},
        {"scores": []},  # B/A ordering never scored the case
        {"scores": []},  # nor when asked again about it alone
    )
    with pytest.raises(ProviderError, match="cs_04:001"):
        judge(fake, [case], {case.id: ModelOutput(text="c")}, {case.id: ModelOutput(text="r")})


def test_unknown_and_duplicate_case_numbers_ignored():
    cases = [_case(1)]
    fake = FakeLLM().add(
        JUDGE_ROLE, {"scores": [_single(9, _r(5, 5, 5)), _single(1, _r(2, 2, 2)), _single(1, _r(5, 5, 5))]}
    )
    result = judge(fake, cases, {"cs_04:001": ModelOutput(text="x")})
    assert result.scores[0].candidate == 2.0  # first entry for case 1 wins; case 9 does not exist


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (401, "invalid api key"),
        (402, "insufficient balance"),
        (404, "model not found"),
        (429, "quota exceeded"),
    ],
)
def test_provider_error_is_raised_not_scored(status, body):
    exc = openai.APIStatusError(
        body,
        response=httpx.Response(status, request=httpx.Request("POST", "https://x"), text=body),
        body=None,
    )
    fake = FakeLLM().add(JUDGE_ROLE, exc)
    with pytest.raises(ProviderError):
        judge(fake, [_case(1)], {"cs_04:001": ModelOutput(text="x")})


def test_budget_exceeded_propagates():
    fake = FakeLLM().add(JUDGE_ROLE, BudgetExceeded("Ultra agent call cap reached (40)"))
    with pytest.raises(BudgetExceeded):
        judge(fake, [_case(1)], {"cs_04:001": ModelOutput(text="x")})


# ------------------------------------------------------------------ per-case pass and judge_agreement


@pytest.mark.parametrize(
    ("cand", "ref", "passes"),
    [
        (4.0, None, True),
        (3.99, None, False),
        (4.5, 4.8, True),  # 4.5 >= 4.8 - 0.3 (float boundary)
        (4.4, 4.8, False),
        (4.0, 5.0, False),  # floor met, margin not
        (8 / 3 + 4 / 3, None, True),  # 4.0 reached through float arithmetic
    ],
)
def test_case_passes_mirrors_site_rule(cand, ref, passes):
    assert case_passes(JudgeScore(case_id="x", candidate=cand, reference=ref)) is passes


def test_judge_agreement_counts_matching_verdicts_on_shared_cases():
    det = [
        CaseScore(case_id="a", passed=True, score=1.0),
        CaseScore(case_id="b", passed=False, score=0.0),
        CaseScore(case_id="c", passed=True, score=1.0),
        CaseScore(case_id="d", passed=False, score=0.0),
        CaseScore(case_id="not-judged", passed=True, score=1.0),
    ]
    judged = JudgeResult(
        scores=[
            JudgeScore(case_id="a", candidate=5.0, reference=5.0, consistent=True),  # pass == pass
            JudgeScore(case_id="b", candidate=2.0, reference=5.0, consistent=True),  # fail == fail
            JudgeScore(case_id="c", candidate=3.0, reference=5.0, consistent=False),  # fail != pass
            JudgeScore(case_id="d", candidate=5.0, reference=5.0, consistent=True),  # pass != fail
        ],
        reference_mean=5.0,
        consistency=0.75,
    )
    assert judge_agreement(det, judged) == Agreement(n=4, agreement=0.5, consistency=0.75)


def test_judge_agreement_without_overlap_raises():
    with pytest.raises(ValueError):
        judge_agreement([CaseScore(case_id="a", passed=True, score=1.0)], JudgeResult(scores=[]))


# ------------------------------------------------------------------ real Ultra responses (cassettes)
# SYNTHETIC TEST INPUTS: KB snippets, questions, candidate and reference answers below were written for
# this test. Candidates 1, 2 and 4 are faithful; candidate 3 invents a phone line and a 7-day window the
# KB does not contain (groundedness must catch it); candidate 5 ignores a two-sentence limit.

_SYSTEM = (
    "You are HelpHub's support assistant. Answer the customer using only the knowledge-base snippets "
    "provided. If the snippets do not cover the question, say so. Answer in at most two sentences."
)
_FREE_TEXT = [
    (
        "KB [refunds-1]: Annual plans can be refunded in full within 30 days of purchase. Monthly plans are "
        "not refundable.\nCustomer: I bought the annual plan 10 days ago. Can I get my money back?",
        "Yes. Annual plans are fully refundable within 30 days of purchase, and you are at day 10, so you "
        "qualify for a full refund.",
        "Yes, you can: annual plans get a full refund within 30 days of purchase, and you bought yours 10 "
        "days ago.",
    ),
    (
        "KB [seats-2]: Admins can add seats under Settings > Billing > Seats. Each added seat is billed "
        "pro rata for the rest of the billing period.\nCustomer: How do I add two more seats for my team?",
        "Go to Settings > Billing > Seats and add two seats; each one is billed pro rata for the rest of "
        "your current billing period.",
        "As an admin, open Settings > Billing > Seats and add the two seats. They are charged pro rata for "
        "the remainder of the billing period.",
    ),
    (
        "KB [password-3]: Password reset links expire 24 hours after they are sent. Request a new link from "
        "the sign-in page if yours expired.\nCustomer: My reset link from yesterday doesn't work anymore.",
        "Reset links stay valid for 7 days, so yours should still work; if not, call our 24/7 support "
        "phone line and an agent will reset it for you.",
        "Reset links expire 24 hours after they are sent, so yesterday's link has likely expired. Request a "
        "new one from the sign-in page.",
    ),
    (
        "KB [export-4]: Workspace owners can export all tickets as CSV from Settings > Data > Export. "
        "Exports over 50,000 tickets are emailed as a download link.\nCustomer: Can I export our tickets?",
        "Yes. A workspace owner can export all tickets as CSV from Settings > Data > Export; exports over "
        "50,000 tickets arrive by email as a download link.",
        "Yes, workspace owners can export every ticket as a CSV under Settings > Data > Export. Large "
        "exports (over 50,000 tickets) are emailed as a download link.",
    ),
    (
        "KB [sso-5]: SAML single sign-on is available on the Enterprise plan. It is configured by an admin "
        "under Settings > Security > SSO.\nCustomer: Do you support SSO?",
        "Great question! Here is everything you need to know about single sign-on:\n\n"
        "- **Plan:** SAML SSO is part of the Enterprise plan.\n"
        "- **Who sets it up:** an admin.\n"
        "- **Where:** Settings > Security > SSO.\n"
        "- **Next steps:** check your plan, then ask your admin.\n\n"
        "Let me know if there is anything else I can help you with today!",
        "Yes, SAML single sign-on is available on the Enterprise plan. An admin configures it under "
        "Settings > Security > SSO.",
    ),
]


def _free_text_inputs():
    cases, cands, refs = [], {}, {}
    for i, (user, cand, ref) in enumerate(_FREE_TEXT, 1):
        case = EvalCase(
            id=f"test_ft:{i:03d}",
            site_id="test_ft",
            input_messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            expected_kind="text",
            split="dev",
        )
        cases.append(case)
        cands[case.id], refs[case.id] = ModelOutput(text=cand), ModelOutput(text=ref)
    return cases, cands, refs


def _run_free_text(client):
    cases, cands, refs = _free_text_inputs()
    return judge(client, cases, cands, refs), judge(client, cases, cands)


@pytest.mark.live
def test_record_free_text_cassettes():
    budget = Budget()
    _run_free_text(TFClient(cache_dir=CASSETTES, live=True, budget=budget))
    assert budget.ultra_agent_calls <= 3


def test_cassette_pairwise_and_absolute_rank_the_faulty_answers_low():
    pairwise, absolute = _run_free_text(TFClient(cache_dir=CASSETTES, live=False))
    for result in (pairwise, absolute):
        assert all(1.0 <= s.candidate <= 5.0 for s in result.scores)
        by_id = {s.case_id: s for s in result.scores}
        faithful = [by_id[f"test_ft:{i:03d}"].candidate for i in (1, 2, 4)]
        assert min(faithful) >= 4.0  # faithful answers clear the floor
        assert by_id["test_ft:003"].candidate < 4.0  # invented facts fail groundedness
        assert by_id["test_ft:005"].candidate < min(faithful)  # broken format costs points
    assert pairwise.reference_mean is not None and pairwise.reference_mean >= 4.0
    assert pairwise.consistency is not None and pairwise.consistency >= 0.8
