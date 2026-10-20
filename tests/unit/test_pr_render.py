"""PR body rendering (PRD §8 + Amendment A): deterministic, every number from the run inputs.

The fixture tests/fixtures/pr/demo_run.json carries made-up numbers (labelled there). Regenerate the snapshot
after an intended template change with UPDATE_SNAPSHOTS=1 and review the diff.
"""

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from offramp.agents.pr_writer import (
    SOURCE_OUTPUTS_NOT_RECORDED,
    PRContext,
    PRProse,
    SiteNote,
    manual_reason,
    reference_summary,
    render_pr,
)
from offramp.schemas import BaselineResult, EvalCase, ModelOutput, PatchFailure, ToolCall

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
        f"(est. −{round(100 * (1 - after / before))}% cost on those {len(migrated)})"
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
    assert "71.4% (5/7, ≥35%) < 90%" in row
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

    assert (
        "- After: Nemotron 3 Super `nvidia/nemotron-3-super-120b-a12b` $0.30 / $0.90 "
        "(Token Factory model API, 2026-10-21)"
    ) in body
    assert "unverified" not in body
    assert "developers.openai.com/api/docs/pricing" in body  # cost_before_note
    assert "https://smith.langchain.com/public/fixture-project/r" in body


def test_an_after_price_not_read_in_the_run_is_pricing_yamls_and_says_so():
    c = ctx()
    c.verdicts = [v.model_copy(update={"after_price": None}) for v in c.verdicts]
    body = render_pr(c).body
    assert (
        "- After: Nemotron 3.5 Lightning `nvidia/Nemotron-3_5-Lightning` $0.06 / $0.24 (pricing.yaml as of "
        "2026-10-21; not read from the Token Factory model API in this run)"
    ) in body


def test_each_before_price_shows_how_it_was_found_and_the_quote():
    body = render_pr(ctx()).body
    line = next(ln for ln in body.splitlines() if ln.startswith("- Before, cs_02:"))
    assert line.endswith(". Price by Tavily search; the page prints `Input $3 / MTok Output $15 / MTok`.")
    line = next(ln for ln in body.splitlines() if ln.startswith("- Before, cs_04:"))
    assert line.endswith(". Price by pricing.yaml fallback.")
    line = next(ln for ln in body.splitlines() if ln.startswith("- Before, cs_01:"))
    assert "Price by" not in line  # a verdict recorded without evidence says nothing it does not know


def test_a_long_quote_is_cut_and_backticks_cannot_break_out_of_it():
    c = ctx()
    v = verdict(c, "cs_02")
    long = "Input `$3` " + "x " * 200
    v.price_evidence = v.price_evidence.model_copy(update={"quote": long})
    line = next(ln for ln in render_pr(c).body.splitlines() if ln.startswith("- Before, cs_02:"))
    assert "the page prints `Input '$3' x x" in line and line.endswith("…`.") and line.count("`") == 2


def evidence_table(body: str) -> str:
    return body.split("## Evidence")[1].split("## What changed")[0]


def test_sites_whose_source_model_retires_lead_the_summary_and_the_evidence_table():
    body = render_pr(ctx(), PROSE).body
    summary = body.split("## Evidence")[0]
    assert summary.startswith("## Summary\n\n**Claude Sonnet 4.5 retires on 30 Nov 2026: cs_02, cs_03.**\n\n")
    rows = [ln.split(" | ")[0] for ln in evidence_table(body).splitlines() if ln.startswith("| cs_")]
    assert rows == ["| cs_02", "| cs_03", "| cs_01", "| cs_04", "| cs_05", "| cs_06"]
    assert "| claude-sonnet-4-5 (retires 30 Nov 2026) |" in body


def test_the_soonest_retirement_comes_first_and_no_date_means_no_lead():
    c = ctx()
    verdict(c, "cs_06").retires_on = date(2026, 11, 1)
    body = render_pr(c).body
    assert (
        "**claude-haiku-4-5 retires on 1 Nov 2026: cs_06.**\n**Claude Sonnet 4.5 retires on 30 Nov 2026"
        in body
    )
    rows = [ln.split(" | ")[0] for ln in evidence_table(body).splitlines() if ln.startswith("| cs_")]
    assert rows[:3] == ["| cs_06", "| cs_02", "| cs_03"]
    c.verdicts = [v.model_copy(update={"retires_on": None}) for v in c.verdicts]
    assert "retires on" not in render_pr(c).body.split("## Evidence")[0]


def kept_free_text_row(**judge) -> str:
    """cs_06 as a kept free-text site: 6 of 8 holdout cases passing, with the given judge scores."""
    c = ctx()
    kept = dict(decision="KEEP_CLOSED", tier=None, model=None, best_tier="lightning", parity_pct=75.0,
                holdout_cases=8, holdout_passed=6, parity_lower_pct=40.9)  # fmt: skip
    c.verdicts = [v.model_copy(update=kept | judge) if v.site_id == "cs_06" else v for v in c.verdicts]
    return next(ln for ln in render_pr(c).body.splitlines() if ln.startswith("| cs_06 "))


def test_a_kept_free_text_site_is_compared_with_the_judges_bars_not_a_share():
    """Review T19 M4 and N2: each judge figure against its own bar, and only the gate the verdict records."""
    row = kept_free_text_row(judge_mean=3.95, judge_min=2.66, failed_gate="judge_mean")
    assert "| 75.0% (6/8, ≥40%), judge mean 3.95/5 < 4.0, lowest case 2.66/5 < 3.0 |" in row
    assert "95%" not in row
    row = kept_free_text_row(judge_mean=4.2, judge_min=2.66, failed_gate="case_floor")
    assert "| 75.0% (6/8, ≥40%), judge mean 4.2/5, lowest case 2.66/5 < 3.0 |" in row
    row = kept_free_text_row(judge_mean=4.2, judge_min=3.5, failed_gate="reference_margin")
    assert "lowest case 3.5/5, more than 0.3 below the references' judge mean |" in row


@pytest.mark.parametrize(
    "gate,cell",
    [
        ("dev", "75.0% (6/8, ≥40%), judge mean 4.5/5, lowest case 4/5; holdout passed, dev did not"),
        (None, "75.0% (6/8, ≥40%), judge mean 4.5/5, lowest case 4/5"),  # recorded before the gate: no cause
    ],
)
def test_a_kept_free_text_row_never_states_a_cause_the_verdict_does_not_record(gate, cell):
    row = kept_free_text_row(judge_mean=4.5, judge_min=4.0, failed_gate=gate)
    assert f"| {cell} |" in row and "references' judge mean" not in row


@pytest.mark.parametrize(
    "site_id,gate,cell",
    [
        ("cs_05", "threshold", "71.4% (5/7, ≥35%) < 90%"),
        ("cs_05", "dev", "71.4% (5/7, ≥35%); holdout passed, dev did not"),
        ("cs_03", "schema", "96.4% (8/8 pass, ≥67%), not every output schema-valid"),
        ("cs_03", "pass_share", "96.4% (8/8 pass, ≥67%), too few cases pass whole"),
    ],
)
def test_a_kept_row_says_the_bar_its_verdict_missed(site_id, gate, cell):
    c = ctx()
    kept = {"decision": "KEEP_CLOSED", "tier": None, "model": None, "failed_gate": gate}
    c.verdicts = [v.model_copy(update=kept) if v.site_id == site_id else v for v in c.verdicts]
    row = next(ln for ln in render_pr(c).body.splitlines() if ln.startswith(f"| {site_id} "))
    assert f"| **NOT MIGRATED** | {cell} |" in row


def test_a_budget_stopped_site_says_so_and_claims_no_parity():
    """Review T19 R2-m3: not "below 95%"."""
    c = ctx()
    stop = {
        "decision": "KEEP_CLOSED",
        "tier": None,
        "model": None,
        "parity_pct": None,
        "failed_gate": "budget",
    }
    c.verdicts = [v.model_copy(update=stop) if v.site_id == "cs_01" else v for v in c.verdicts]
    row = next(ln for ln in render_pr(c).body.splitlines() if ln.startswith("| cs_01 "))
    assert "| **NOT MIGRATED** | not measured: the run's budget stopped it |" in row and "95%" not in row


@pytest.mark.parametrize("site_id", ["cs_05", "cs_06"])
def test_a_kept_verdict_recorded_without_its_gate_states_figures_and_no_comparison(site_id):
    """Review T19 R2-m3: a verdict from before failed_gate (a legacy "dev never passed" row read "100.0% <
    95%")."""
    c = ctx()
    legacy = {"decision": "KEEP_CLOSED", "tier": None, "model": None, "failed_gate": None}
    c.verdicts = [v.model_copy(update=legacy) if v.site_id == site_id else v for v in c.verdicts]
    row = next(ln for ln in render_pr(c).body.splitlines() if ln.startswith(f"| {site_id} "))
    cell = row.split(" | ")[5]
    assert "<" not in cell and "below" not in cell and "dev" not in cell and "%" in cell


def test_a_retirement_date_already_past_says_retired():
    body = render_pr(ctx(today=date(2026, 12, 1))).body
    assert "**Claude Sonnet 4.5 retired on 30 Nov 2026: cs_02, cs_03.**" in body
    assert "| claude-sonnet-4-5 (retired 30 Nov 2026) |" in body


def test_the_bracket_sentence_says_what_json_extraction_counts():
    body = render_pr(ctx()).body
    assert "for JSON extraction, whose figure is the mean field match, the cases that pass whole" in body
    assert "with 95% confidence" in body


def test_parity_shows_its_case_count_and_wilson_bound_and_free_text_its_floor():
    body = render_pr(ctx()).body
    assert "| 97.1% (34/35, ≥85%) (after 1 adapter iteration) |" in body
    assert "| 100.0% (6/6, ≥60%), judge mean 4.3/5, lowest case 3.67/5 (after 1 adapter iteration) |" in body
    assert "Wilson 95% lower bound" in body and "costs cents on Nemotron 3.5 Lightning" in body
    c = ctx()
    c.verdicts = [v.model_copy(update={"holdout_cases": None, "judge_mean": None}) for v in c.verdicts]
    assert "| 97.1% (after 1 adapter iteration) |" in render_pr(c).body  # recorded before the bounds


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


def test_the_merged_check_a_combined_repair_and_a_dropped_site_are_reported():
    c = ctx(alone={"cs_02": True, "cs_03": True, "cs_04": True}, combined=["cs_02", "cs_04"],
            combined_repair=["tests/test_cli.py"])  # fmt: skip
    dropped = verdict(c, "cs_03").model_copy(
        update={"decision": "MANUAL_REVIEW", "reason": "green alone, but…"}
    )
    c.verdicts = [dropped if v.site_id == "cs_03" else v for v in c.verdicts]

    body = render_pr(c, PROSE).body

    assert (
        "- The patches green on their own (cs_02, cs_03, cs_04) were then checked merged: cs_02, cs_04 pass "
        "this repo's checks together, after one combined repair by Nemotron 3 Ultra to `tests/test_cli.py`; "
        "cs_03 dropped, see below." in body
    )
    assert "; 1 needs manual review." in body
    c.verdicts = [
        v.model_copy(update={"decision": "MANUAL_REVIEW"}) if v.site_id == "cs_04" else v for v in c.verdicts
    ]
    c.combined = ["cs_02"]
    plural = render_pr(c, PROSE).body
    assert "; 2 need manual review." in plural and "cs_02 pass this repo's checks together" in plural
    c.combined = ["cs_04", "cs_03", "cs_02"]  # in site order, not sorted: still all of them (M5)
    assert "(cs_02, cs_03, cs_04) were then checked merged: all pass" in render_pr(c, PROSE).body


# ---------------------------------------------------------------- Task 18: code-built site lines, the bar

CS03_RAW = (  # a live cs_03 reason's shape (run 5): the repair broke the format, the try before failed
    "the reply holds no complete edit block; last failure: repo checks failed on the patch (docker): "
    "FAILED tests/test_extract.py::test_request_contract - pydantic_core._pydantic... | "
    "FAILED tests/test_extract.py::test_every_fixture_email_is_sent_and_parsed[…] (13 cases) | "
    "E       pydantic_core._pydantic_core.ValidationError: 1 validation error for ChatCompletion | "
    "E       choices.0.message.content | E         Input should be a valid string [type=string_type, "
    "input_value=TextBlock(citations=None,...damaged\"}', type='text'), input_type=TextBlock]"
)


def manual(c: PRContext, site_id: str, reason: str, failure: PatchFailure | None = None) -> PRContext:
    c.verdicts = [
        v.model_copy(update={"decision": "MANUAL_REVIEW", "reason": reason, "patch_failure": failure})
        if v.site_id == site_id
        else v
        for v in c.verdicts
    ]
    return c


def test_a_manual_review_site_reads_as_one_plain_sentence_with_the_raw_failure_folded():
    guess = PRProse(sites=[SiteNote(site_id="cs_03", note="A reviewer must confirm the German example.")])
    failure = PatchFailure(kind="no_edit_block", last="checks_failed", tries=4, model_edit=True)

    body = render_pr(manual(ctx(), "cs_03", CS03_RAW, failure), guess).body

    line = next(ln for ln in body.splitlines() if ln.startswith("- **cs_03**"))
    assert line == (  # the outer cause first, then the last checked try; nothing read from the check output
        "- **cs_03** `helphub/orders/extract.py` needs manual review: The model's last reply had no complete "
        "edit, so nothing was applied; with the model's test edits applied, this repo's checks still failed."
    )
    assert "A reviewer must confirm" not in body  # no model prose on a manual-review site
    folded = body.split(line)[1].split("</details>")[0]
    assert "<details><summary>Raw failure</summary>" in folded
    assert "\n  FAILED tests/test_extract.py::test_every_fixture_email" in folded  # a check output per line
    assert CS03_RAW.split(" | ")[0] in folded


EDITED, ALONE = "with the model's test edits applied,", "with Offramp's changes alone,"
GREEN = "Its patch passes this repo's checks on its own, but"
NEVER = "and Offramp never ships a patch that deletes a test."
SHIPS = "code that ships changed (`app/x.py`), and only Offramp's code may write it."
F, M = PatchFailure, True  # M: the last checked try applied the model's edits


@pytest.mark.parametrize(
    ("failure", "sentence"),
    [
        (F(kind="search_miss", files=["tests/t.py"], model_edit=M),
         "The model's edit to `tests/t.py` quotes lines matching no single place there, so none applied."),
        (F(kind="deletes_tests", files=["tests/t.py"], tests=["tests/t.py::test_o3"], model_edit=M),
         f"The model's edit deleted the test `tests/t.py::test_o3`, {NEVER}"),
        (F(kind="locked_file", files=["app/x.py"], model_edit=M),
         "The model tried to edit `app/x.py`, which ships exactly as Offramp's code wrote it."),
        (F(kind="not_editable", files=["../x.py"], model_edit=M),
         "The model's edit to `../x.py` is not allowed for this site."),
        (F(kind="does_not_parse", files=["tests/t.py"], model_edit=M),
         "The model's edit left `tests/t.py` unparseable."),
        (F(kind="checks_failed", model_edit=M), f"{EDITED.capitalize()} this repo's checks still failed."),
        (F(kind="backstop", files=["app/x.py"], model_edit=M), f"{EDITED.capitalize()} {SHIPS}"),
        (F(kind="probe_mismatch", model_edit=M),
         f"{EDITED.capitalize()} the call does not send the request that passed the evaluation."),
        (F(kind="probe_mismatch"),
         "Offramp's own rewrite does not send the request that passed the evaluation."),
        (F(kind="merge_conflict", files=["tests/t.py"]),
         f"{GREEN} it conflicts with another migrated site's patch in `tests/t.py`."),
        (F(kind="not_together"), f"{GREEN} not merged with the other migrated sites' patches."),
        (F(kind="merged_drift"), f"{GREEN} on the merged tree its code or its evaluated request changed."),
        (F(kind="repo_red"),
         "This repo's own checks fail before any migration patch, so no patch could be checked."),
        (F(kind="budget_stop", last="checks_failed", tries=0),
         f"The run's model budget ran out before the model replied at this site's patch; {ALONE} this repo's "
         "checks fail."),
        (F(kind="budget_stop", last="deletes_tests", tries=2, model_edit=M, tests=["t.py::test_a"]),
         "The run's model budget ran out after 2 tries at this site's patch; the model's edit deleted the "
         f"test `t.py::test_a`, {NEVER}"),
        (F(kind="no_edit_block", last="checks_failed", tries=4),
         f"The model's last reply had no complete edit, so nothing was applied; {ALONE} this repo's checks "
         "fail."),
        (F(kind="no_change", last="backstop", tries=2, model_edit=M, files=["app/x.py"]),
         f"The model's last repair repeated an edit it had already tried; {EDITED} {SHIPS}"),
    ],
    ids=["search-miss", "deletes-tests", "locked", "not-editable", "parse", "checks", "backstop", "probe",
         "probe-offramp-alone", "merge-conflict", "not-together", "merged-drift", "repo-red", "budget-alone",
         "budget-after-tries", "no-edit-block", "no-change"],
)  # fmt: skip
def test_each_recorded_failure_kind_reads_as_one_plain_sentence(failure, sentence):
    v = verdict(manual(ctx(), "cs_03", "raw | output", failure), "cs_03")

    assert manual_reason(v) == (sentence, "raw | output")


def test_a_verdict_without_a_recorded_patch_failure_shows_its_reason_as_it_is():
    reason = "patch not composed: Offramp could not rewrite this call; a model-written call is not shipped"
    assert manual_reason(verdict(manual(ctx(), "cs_03", reason), "cs_03")) == (reason, None)

    body = render_pr(manual(ctx(), "cs_03", reason)).body

    assert f"needs manual review: {reason}" in body and "Raw failure" not in body


def test_a_migrated_site_line_is_built_from_the_measurements_not_model_prose():
    claim = PRProse(sites=[SiteNote(site_id="cs_02", note="Super matched reference behavior perfectly.")])
    c = ctx()
    tr = next(t for t in c.tier_results if t.site_id == "cs_01" and t.split == "holdout")

    body = render_pr(c, claim).body

    assert "matched reference behavior perfectly" not in body
    line = next(ln for ln in body.splitlines() if ln.startswith("- **cs_01**"))
    assert (
        f"→ Nemotron 3.5 Lightning. `triage_ticket` now sends the request that passed parity on the holdout "
        f"split: parity vs reference {verdict(c, 'cs_01').parity_pct:.1f}% over {len(tr.case_scores)} case"
    ) in line


def test_the_pass_bar_names_the_per_case_floor_for_free_text():
    section = render_pr(ctx()).body.split("## How parity was measured")[1]

    assert (
        "free text: judge mean of at least 4.0 on a 1–5 rubric, no holdout case below 3.0, and at least"
        in section
    )


def test_a_reference_the_judge_scored_below_3_is_flagged_and_kept():
    c = ctx()
    c.verdicts = [
        v.model_copy(update={"weak_references": {"cs_06:020": 1.0, "cs_06:011": 2.5}}) if v.site_id == "cs_06"
        else v
        for v in c.verdicts
    ]  # fmt: skip

    section = render_pr(c).body.split("## How parity was measured")[1]

    assert (
        "- cs_06: reference judged weak: cs_06:011, 2.5/5; cs_06:020, 1/5 (below 3; kept in the evaluation)."
        in section
    )
