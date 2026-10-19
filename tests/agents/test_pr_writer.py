"""PR Writer (PRD §6.9): Super writes prose only; numbers, table and <details> come from the run inputs."""

import json
from pathlib import Path

import pytest

from offramp.agents.pr_writer import PRContext, PRProse, SiteNote, render_pr, write_pr
from offramp.schemas import Tier
from offramp.tf import BudgetExceeded, FakeLLM, TFClient

FIXTURES = Path(__file__).parents[1] / "fixtures"
CASSETTES = FIXTURES / "cassettes" / "pr_writer"


def ctx() -> PRContext:
    data = json.loads((FIXTURES / "pr" / "demo_run.json").read_text())["context"]
    return PRContext.model_validate(data)


def test_prose_comes_from_super_with_thinking_off():
    prose = PRProse(
        summary="Five call sites move to Nemotron; the refund planner stays.",
        sites=[SiteNote(site_id="cs_02", note="Tool calls are read from tool_calls now.")],
    )
    llm = FakeLLM().add("pr_writer", prose)

    draft = write_pr(ctx(), llm)

    (call,) = llm.calls
    assert (call["role"], call["tier"], call["temperature"], call["reasoning"]) == (
        "pr_writer",
        Tier.SUPER,
        0.0,
        "off",
    )
    assert call["response_model"] is PRProse
    assert draft == render_pr(ctx(), prose)
    assert "Five call sites move to Nemotron" in draft.body


def test_prose_that_states_a_number_is_dropped():
    keep = "The cs_02 call to claude-sonnet-4-5 becomes a Nemotron 3 Super call; `next_action` is unchanged."
    prose = PRProse(
        summary="This cuts the LLM bill by 99%.",
        sites=[
            SiteNote(site_id="cs_02", note=keep),
            SiteNote(site_id="cs_03", note="Parity 96.4% on holdout."),
            SiteNote(site_id="cs_04", note="Answers are 3x faster."),
        ],
    )
    llm = FakeLLM().add("pr_writer", prose)

    body = write_pr(ctx(), llm).body

    assert "99%" not in body and "3x faster" not in body and "Parity 96.4% on holdout." not in body
    assert keep in body  # identifiers and tier names with digits are not numbers


def test_prose_for_an_unknown_site_is_ignored():
    llm = FakeLLM().add("pr_writer", PRProse(summary="ok", sites=[SiteNote(site_id="cs_99", note="ghost")]))

    assert "ghost" not in write_pr(ctx(), llm).body


@pytest.mark.parametrize("failure", [BudgetExceeded("output cap"), "not json"])
def test_no_prose_still_renders_the_evidence(failure):
    llm = FakeLLM().add("pr_writer", failure)

    assert write_pr(ctx(), llm) == render_pr(ctx())


def run_cassette(live: bool):
    return write_pr(ctx(), TFClient(cache_dir=CASSETTES, live=live))


@pytest.mark.live
def test_record_pr_prose():
    print(run_cassette(live=True).body)


def test_pr_prose_replays_from_cassette():
    draft = run_cassette(live=False)

    assert draft.title == render_pr(ctx()).title
    summary = draft.body.split("## Summary\n\n")[1].split("\n\n")[0]
    assert summary and summary != render_pr(ctx()).body.split("## Summary\n\n")[1].split("\n\n")[0]


def test_a_decision_label_echoed_at_the_start_of_a_note_is_dropped():
    llm = FakeLLM().add(
        "pr_writer", PRProse(sites=[SiteNote(site_id="cs_02", note="MIGRATE - Reads tool_calls.")])
    )

    body = write_pr(ctx(), llm).body

    assert "Nemotron 3 Super. Reads tool_calls." in body and "MIGRATE -" not in body
