"""§12.3 judge sanity ("evals aren't fake"): on a site that also has deterministic scores, the Ultra judge's
per-case verdicts must agree with the deterministic scorer on >= 85% of cases, and >= 90% of its pairwise
verdicts must survive the A/B -> B/A position swap.

Replays real Nemotron Ultra responses recorded once with `pytest --live` into tests/fixtures/cassettes/judge/.
SYNTHETIC TEST INPUTS: the tickets, reference labels and candidate labels below were written for this test.
Candidates 1-12 match the reference; 13-18 are plainly wrong labels; 19 has the right label wrapped in prose
(the label scorer fails it, the judge's format criterion should too); 20 is the right label with a trailing
period (the label scorer forgives it).
"""

from pathlib import Path

import pytest

from offramp.agents.judge import judge, judge_agreement
from offramp.evals.scorers import score_case
from offramp.schemas import Contract, EvalCase, ModelOutput
from offramp.tf import Budget, TFClient

CASSETTES = Path(__file__).parents[1] / "fixtures" / "cassettes" / "judge"
CONTRACT = Contract(
    task_type="classification", output="label", labels=["billing", "bug", "account", "feature_request"]
)
SYSTEM = (
    "Classify the support ticket into exactly one of: billing, bug, account, feature_request. "
    "Reply with the label only."
)
TICKETS = [  # (ticket, reference label, candidate output)
    ("I was charged twice for invoice 991 this month.", "billing", "billing"),
    (
        "The export button does nothing when I click it in Firefox; the console shows a TypeError.",
        "bug",
        "bug",
    ),
    ("I can't log in - it says my account is locked after too many attempts.", "account", "account"),
    ("It would be great if you supported dark mode in the dashboard.", "feature_request", "feature_request"),
    ("Please update the VAT number on our invoices to DE811234567.", "billing", "billing"),
    ("Since yesterday's update the mobile app crashes as soon as I open a ticket.", "bug", "bug"),
    ("How do I change the email address on my profile? I no longer use the old one.", "account", "account"),
    (
        "Could you add a Slack integration so new tickets get posted to a channel?",
        "feature_request",
        "feature_request",
    ),
    ("My card was declined but the money still left my bank account.", "billing", "billing"),
    ("Search returns no results for any query containing an apostrophe, e.g. O'Brien.", "bug", "bug"),
    ("Please delete my account and all my personal data.", "account", "account"),
    ("We'd love an API endpoint to bulk-close tickets.", "feature_request", "feature_request"),
    (
        "Our renewal went through at the old price even though we downgraded last week; please refund the "
        "difference.",
        "billing",
        "feature_request",
    ),
    ("Attachments larger than 2 MB fail to upload with a 500 error.", "bug", "billing"),
    ("I need to transfer ownership of the workspace to my colleague before I leave.", "account", "bug"),
    ("Please let us schedule reports to be emailed every Monday.", "feature_request", "account"),
    ("The date picker shows February 30 as a selectable day.", "bug", "feature_request"),
    ("Why was I billed for 12 seats when we only have 9 users?", "billing", "account"),
    (
        "I keep getting the two-factor code by SMS even though I switched to the authenticator app.",
        "account",
        "The ticket is about the customer's sign-in settings, so the label is account.",
    ),
    (
        "Your pricing page says annual plans get 20% off but I was charged the full price.",
        "billing",
        "Billing.",
    ),
]


def _inputs():
    cases, cands, refs = [], {}, {}
    for i, (ticket, ref, cand) in enumerate(TICKETS, 1):
        case = EvalCase(
            id=f"test_cls:{i:03d}",
            site_id="test_cls",
            input_messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": ticket}],
            expected_kind="label",
            split="dev",
        )
        cases.append(case)
        cands[case.id], refs[case.id] = ModelOutput(text=cand), ModelOutput(text=ref)
    return cases, cands, refs


def _agreement(client):
    cases, cands, refs = _inputs()
    deterministic = [score_case(c, refs[c.id], cands[c.id], CONTRACT) for c in cases]
    judged = judge(client, cases, cands, refs)
    return deterministic, judged, judge_agreement(deterministic, judged)


@pytest.mark.live
def test_record_overlap_cassettes():
    budget = Budget()
    _agreement(TFClient(cache_dir=CASSETTES, live=True, budget=budget))
    assert budget.ultra_agent_calls <= 2


def test_overlap_set_is_what_the_docstring_says():
    cases, cands, refs = _inputs()
    passed = [score_case(c, refs[c.id], cands[c.id], CONTRACT).passed for c in cases]
    assert passed == [True] * 12 + [False] * 7 + [True]


def test_judge_agrees_with_deterministic_scores_and_survives_position_swap():
    _, judged, agreement = _agreement(TFClient(cache_dir=CASSETTES, live=False))
    assert agreement.n == len(TICKETS)
    assert agreement.agreement >= 0.85
    assert agreement.consistency is not None and agreement.consistency >= 0.90
