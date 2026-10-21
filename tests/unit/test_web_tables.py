"""web/lib/pr-writer-tables.json mirrors the PR writer's tables, so the report says what the PR says."""

import json
from pathlib import Path
from typing import get_args

from offramp.agents import pr_writer
from offramp.evals.scorers.aggregate import JUDGE_CASE_FLOOR, JUDGE_FLOOR
from offramp.schemas import Gate, PatchFailureKind

WEB_TABLES = Path(__file__).resolve().parents[2] / "web" / "lib" / "pr-writer-tables.json"
WEB_HANDLES_ITSELF = {"budget", "threshold", "judge_mean", "case_floor"}  # in format.ts keptText


def test_the_web_tables_equal_the_pr_writers():
    web = json.loads(WEB_TABLES.read_text(encoding="utf-8"))
    assert web["cause"] == pr_writer._CAUSE
    assert web["alone"] == pr_writer._ALONE
    assert web["ended"] == pr_writer._ENDED
    assert web["missed"] == pr_writer._MISSED
    assert web["price_methods"] == pr_writer.PRICE_METHODS
    assert (web["judge_floor"], web["judge_case_floor"]) == (JUDGE_FLOOR, JUDGE_CASE_FLOOR)


def test_every_recorded_kind_and_gate_has_words_on_the_web():
    web = json.loads(WEB_TABLES.read_text(encoding="utf-8"))
    assert set(get_args(PatchFailureKind)) == set(web["cause"]) | set(web["ended"])
    assert set(get_args(Gate)) == set(web["missed"]) | WEB_HANDLES_ITSELF
