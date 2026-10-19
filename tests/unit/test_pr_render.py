"""PR body rendering (PRD §8 + Amendment A): deterministic, every number from the run inputs.

The fixture tests/fixtures/pr/demo_run.json carries made-up numbers (labelled there). Regenerate the snapshot
after an intended template change with UPDATE_SNAPSHOTS=1 and review the diff.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from offramp.agents.pr_writer import (
    SOURCE_OUTPUTS_NOT_RECORDED,
    PRContext,
    PRProse,
    SiteNote,
    reference_summary,
    render_pr,
)
from offramp.schemas import BaselineResult, EvalCase, ModelOutput, ToolCall

FIXTURES = Path(__file__).parents[1] / "fixtures" / "pr"
SNAPSHOT = FIXTURES / "expected_body.md"
PROSE = PRProse(
    summary="This PR moves five HelpHub call sites to NVIDIA Nemotron on Nebius Token Factory. "
    "The refund planner stays on its original model because it did not reach parity.",
    sites=[
        SiteNote(site_id="cs_02", note="Anthropic tool_use blocks are now read from OpenAI tool_calls."),
        SiteNote(site_id="cs_05", note="Try distilling the refund arithmetic before the next attempt."),
    ],
)


def ctx(**update) -> PRContext:
    data = json.loads((FIXTURES / "demo_run.json").read_text())["context"]
    return PRContext.model_validate(data).model_copy(update=update)


def verdict(c: PRContext, site_id: str):
    return next(v for v in c.verdicts if v.site_id == site_id)


def test_body_matches_snapshot():
    body = render_pr(
        ctx(
            site_files={"cs_02": ["helphub/ops/agent_step.py", "tests/fakes.py", "tests/test_agent_step.py"]},
            sandbox="docker",
        ),
        PROSE,
    ).body
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(body)
    assert body == SNAPSHOT.read_text()


def test_title_counts_migrated_sites_and_estimates_the_saving_from_verdicts():
    c = ctx()
    migrated = [v for v in c.verdicts if v.decision == "MIGRATE"]
    before = sum(v.cost_before_per_1k for v in migrated)
    after = sum(v.cost_after_per_1k for v in migrated)

    title = render_pr(c).title

    assert title == (
        f"Offramp: migrate {len(migrated)}/{len(c.verdicts)} LLM call sites to Nemotron on Nebius Token "
        "Factory "
        f"(est. −{round(100 * (1 - after / before))}% cost)"
    )


def test_evidence_table_uses_the_amendment_a_columns():
    body = render_pr(ctx()).body

    assert (
        "| Site | File | Task | Before | After | Parity vs reference (holdout) | p50 latency (after) "
        "| $/1k calls before (est.) → after |"
    ) in body


def test_keep_closed_row_shows_parity_below_threshold_and_no_savings():
    body = render_pr(ctx()).body

    row = next(line for line in body.splitlines() if line.startswith("| cs_05 "))
    assert "**NOT MIGRATED**" in row
    assert "71.4% < 90%" in row
    assert row.endswith("| — | — |")


def test_migrated_row_numbers_come_from_the_verdict_and_holdout_tier_result():
    c = ctx()
    v = verdict(c, "cs_02")
    tr = next(t for t in c.tier_results if t.site_id == "cs_02" and t.split == "holdout")

    row = next(line for line in render_pr(c).body.splitlines() if line.startswith("| cs_02 "))

    assert f"{v.parity_pct:.1f}%" in row
    assert f"{tr.metrics.p50_ms / 1000:.1f} s" in row
    assert "$3.00 (est.) → $0.26" in row
    assert f"`{v.model}`" in row


def test_patch_failure_downgrade_is_shown_as_manual_review_and_not_counted():
    c = ctx()
    c.verdicts = [
        v.model_copy(update={"decision": "MANUAL_REVIEW", "reason": "patch failed the repo's tests"})
        if v.site_id == "cs_03"
        else v
        for v in c.verdicts
    ]

    draft = render_pr(c)

    assert draft.title.startswith("Offramp: migrate 4/6 ")
    row = next(line for line in draft.body.splitlines() if line.startswith("| cs_03 "))
    assert "**MANUAL REVIEW**" in row and row.endswith("| — | — |")
    assert "patch failed the repo's tests" in draft.body


def test_parity_section_says_source_outputs_were_never_recorded():
    section = render_pr(ctx()).body.split("## How parity was measured")[1]

    assert SOURCE_OUTPUTS_NOT_RECORDED == (
        "The source models' own outputs were never recorded; parity is agreement with references."
    )
    assert SOURCE_OUTPUTS_NOT_RECORDED in section


def test_closed_references_name_the_model_exactly_as_stored():
    c = ctx()
    model = "claude-sonnet-5.5 (via Claude Code; not the repo's source model)"
    assert c.references["cs_04"] == {model: 20}

    section = render_pr(c).body.split("## How parity was measured")[1]

    line = next(ln for ln in section.splitlines() if "were produced by" in ln)
    assert f"`{model}`" in line and "cs_04" in line and "cs_06" in line


def test_without_closed_references_no_closed_model_is_named():
    c = ctx()
    c.references = {s: {"reference:seed": 20} for s in c.references}

    section = render_pr(c).body.split("## How parity was measured")[1]

    assert SOURCE_OUTPUTS_NOT_RECORDED in section
    assert "Reference outputs for" not in section  # the auditor may still be named in the audit column


def test_price_sources_dates_and_langsmith_links_are_listed():
    body = render_pr(ctx()).body.split("## How parity was measured")[1]

    assert "https://nebius.com/token-factory/prices" in body and "2026-10-19" in body
    assert "developers.openai.com/api/docs/pricing" in body  # cost_before_note
    assert "https://smith.langchain.com/public/fixture-project/r" in body


def test_prose_without_a_note_falls_back_to_recorded_text():
    body = render_pr(ctx()).body  # no prose at all

    assert "## Summary" in body
    assert verdict(ctx(), "cs_05").reason in body


def test_reference_summary_counts_cases_and_reference_methods():
    def case(cid, split, tags=()):
        return EvalCase(
            id=cid,
            site_id=cid.split(":")[0],
            input_messages=[],
            expected_kind="label",
            split=split,
            tags=tags,
        )

    def ref(cid, model, source="reference", output=None):
        return BaselineResult(
            case_id=cid,
            source=source,
            model=model,
            output=output or ModelOutput(text="x"),
            recorded_at=datetime(2026, 10, 19, tzinfo=UTC),
        )

    escalate = ModelOutput(
        tool_calls=[ToolCall(name="escalate_to_human", arguments={"reason": "r", "priority": "low"})]
    )
    cases = [
        case("cs_01:000", "dev"),
        case("cs_01:001", "holdout"),
        case("cs_02:000", "dev", ["free_text_arg:reason", "locale:de"]),
        case("cs_02:001", "holdout", ["free_text_arg:note"]),  # no reference: the bare argument name
    ]
    refs = [
        ref("cs_01:000", "reference:seed"),
        ref("cs_01:001", "reference:human"),
        ref("cs_02:000", "claude-sonnet-5.5", source="closed_model", output=escalate),
    ]

    counts, methods, free = reference_summary(cases, refs)

    assert free == {"cs_02": ["escalate_to_human.reason", "note"]}

    assert counts == {"cs_01": {"dev": 1, "holdout": 1}, "cs_02": {"dev": 1, "holdout": 1}}
    assert methods == {
        "cs_01": {"reference:seed": 1, "reference:human": 1},
        "cs_02": {"claude-sonnet-5.5": 1},
    }


def test_free_text_tool_arguments_are_disclosed_per_site():
    section = render_pr(ctx()).body.split("## How parity was measured")[1]

    assert (
        "- cs_02: free-text argument escalate_to_human.reason checked for presence, not wording." in section
    )


def test_each_site_lists_the_test_files_its_patch_edited():
    c = ctx(site_files={"cs_02": ["helphub/cli.py", "helphub/ops/agent_step.py", "tests/fakes.py",
                                  "tests/test_cli.py"]})  # fmt: skip

    line = next(ln for ln in render_pr(c).body.splitlines() if ln.startswith("- **cs_02**"))

    assert "Files: `helphub/cli.py`, `helphub/ops/agent_step.py`." in line
    tests = "`tests/fakes.py`, `tests/test_cli.py`."
    assert f"Tests updated (they pinned the old provider's request or fakes): {tests}" in line


@pytest.mark.parametrize(
    ("path", "says"),
    [
        ("helphub/offramp_client.py", "inside the `helphub` package, so it ships with it"),
        (
            "offramp/client.py",
            "at the repository root, because the migrated call sites are not inside one package",
        ),
    ],
)
def test_the_body_says_where_the_client_module_lives_and_why(path, says):
    body = render_pr(ctx(client_path=path)).body

    line = next(ln for ln in body.splitlines() if ln.startswith(f"- `{path}` added"))
    assert says in line


def test_pass_bar_names_the_pairwise_rule_for_free_text_sites_with_text_references():
    section = render_pr(ctx()).body.split("## How parity was measured")[1]

    assert (
        "and at least the references' judge mean − 0.3 where references hold text (cs_04, cs_06)" in section
    )


def test_the_audit_column_names_the_auditor():
    row = next(ln for ln in render_pr(ctx()).body.splitlines() if ln.startswith("| cs_01 | reference:seed"))

    assert "26 / 25 / 1 / 0 by claude-sonnet-5.5 (via Claude Code)" in row
