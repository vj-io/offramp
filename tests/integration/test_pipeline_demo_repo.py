"""The whole pipeline on the demo repo (offramp-demo-app), cache-only on committed cassettes (PLAN Task 12):
six verdicts as recorded (RECORDED), the PR body as a snapshot. No mocks: every agent, the eval matrix, the
routing loop, the Docker sandbox and the PR writer run for real; Token Factory is the recorded cassettes.

Inputs (committed, except the demo checkout):
- the demo repo at DEMO_COMMIT, cloned from a local offramp-demo-app checkout (OFFRAMP_DEMO_REPO_DIR, or a
  sibling directory of this repo); skipped without one, and without Docker (the patch stage's sandbox);
- tests/fixtures/pipeline/demo_evals/: the eval dataset `offramp prepare` wrote for that commit, overlaid on
  the clone's evals/. A test fixture, smaller than the real one (target 12 cases per site, 2 adversarial;
  see DATASET_COMMAND); its seed references are NOT audited and some are known to be wrong (see RECORDED).
  Task 14 prepares, audits and commits the real demo dataset;
- tests/fixtures/cassettes/pipeline/: every Token Factory response the run used, recorded by
  test_record_demo_run (`uv run --env-file .env pytest --live -m live -k record_demo_run`, about $0.50 of
  Token Factory credit), which also writes the PR body snapshot tests/fixtures/pipeline/expected_pr_body.md.
Prices come from pricing.yaml: no Tavily in tests. No LangSmith, no GitHub (open_pr=False), no Serverless Job.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from offramp import pipeline
from offramp.agents.verdict import pricing_yaml_price
from offramp.evals.scorers import aggregate
from offramp.sandbox import docker_available
from offramp.schemas import RunConfig, Stage
from offramp.tf import Budget, TFClient

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
DATASET = FIXTURES / "pipeline" / "demo_evals"
SNAPSHOT = FIXTURES / "pipeline" / "expected_pr_body.md"
CASSETTES = FIXTURES / "cassettes" / "pipeline"
DEMO_COMMIT = "8d76134"  # offramp-demo-app: six call sites + the refund oracle
DEMO_ORIGIN = "https://github.com/vj-io/offramp-demo-app.git"  # the PR names this repo, as a real run does
RUN_ID = "test-demo-run"
# Recorded live 2026-10-20 on the fixture dataset ($0.47 of Token Factory credit). cs_01-03 reached no parity
# on any tier; cs_04-06 reached it on Lightning. Patch stage re-recorded with the Task 10c fix-round-4 patcher
# (code that ships is Offramp's code only: each migrated module and its callers rewritten by code; Super may
# edit only tests and fakes, Ultra repairs): cs_04's patch (one Ultra repair) and cs_06's (Super alone) pass
# the demo repo's own ruff + pytest alone and together and ship; Super's cs_05 patch deleted a test of
# tests/test_call_sites.py and Ultra repeated it, so cs_05 needs a manual review. The PR writer's prose was
# not re-recorded in that round (its live calls went to the patch stage), so the snapshot has no prose.
# The fixture's seed references are NOT audited, and some are known to be wrong: cs_01:009 (a pricing and
# charge question labelled `general`; every tier answers `billing`, which is right) and both cs_03 holdout
# references (cs_03:005 takes the email's Date header as order_date, cs_03:009 leaves customer_email null
# though the From header states it). They decide cs_01's and cs_03's KEEP_CLOSED. That is why the demo's own
# dataset comes from `offramp prepare` plus the GC18 blind audit (Task 14). This test pins the pipeline on
# recorded responses; the verdicts it asserts are what was measured on this flawed dataset, not evidence about
# the models.
RECORDED = {
    "cs_01": "KEEP_CLOSED",
    "cs_02": "KEEP_CLOSED",
    "cs_03": "KEEP_CLOSED",
    "cs_04": "MIGRATE",
    "cs_05": "MANUAL_REVIEW",
    "cs_06": "MIGRATE",
}
DATASET_COMMAND = (
    "scan_repo + label_contracts + case_gen.prepare_repo(repo, scan, TFClient(live=True), target=12, "
    "adversarial=2) on a clone at DEMO_COMMIT; evals/ minus reference/ copied to DATASET"
)
UNSET = (  # nothing in a test run may reach these services, whatever the developer's environment holds
    "LANGSMITH_API_KEY",
    "LANGSMITH_API_KEY_FALLBACK",
    "LANGSMITH_TRACING",
    "TAVILY_API_KEY",
    "TAVILY_API_KEY_FALLBACK",
    "NEBIUS_PROJECT_ID",
    "NEBIUS_JOB_SECRET_ID",
    "OFFRAMP_EVAL_IMAGE",
    "GITHUB_TOKEN",
)


def demo_checkout() -> Path | None:
    env = os.environ.get("OFFRAMP_DEMO_REPO_DIR")
    candidates = [Path(env)] if env else [p / "offramp-demo-app" for p in ROOT.parents]
    return next((p for p in candidates if (p / "helphub").is_dir()), None)


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def run_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live: bool) -> pipeline.RunLog:
    if not docker_available():
        pytest.skip("the patch stage's sandbox needs Docker (and the patcher cassettes name its output)")
    source = demo_checkout()
    if source is None:
        pytest.skip("offramp-demo-app checkout not found (set OFFRAMP_DEMO_REPO_DIR)")
    if not live and not any(CASSETTES.glob("*.json")):
        pytest.skip("pipeline cassettes not recorded: pytest --live -m live -k record_demo_run")
    for name in UNSET:
        monkeypatch.delenv(name, raising=False)
    # target environments (Pydantic schemas, the refund oracle) are built once and reused across runs
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(ROOT / "var" / "test-pipeline-demo"))

    repo = tmp_path / "offramp-demo-app"
    subprocess.run(["git", "clone", "-q", str(source), str(repo)], check=True)
    try:
        git(repo, "-c", "advice.detachedHead=false", "checkout", "-q", DEMO_COMMIT)
    except subprocess.CalledProcessError:
        pytest.skip(f"the demo checkout does not have commit {DEMO_COMMIT}")
    git(repo, "remote", "set-url", "origin", DEMO_ORIGIN)
    shutil.copytree(DATASET, repo / "evals", dirs_exist_ok=True)

    config = RunConfig(repo_url=str(repo), open_pr=False, live=live)
    log = pipeline.new_run(config, tmp_path / "runs", run_id=RUN_ID)
    client = TFClient(cache_dir=CASSETTES, live=live, budget=Budget())
    stage = pipeline.Pipeline(log, client, cache_dir=CASSETTES, price_lookup=pricing_yaml_price).execute()
    assert stage == Stage.DONE, log.events[-1].payload
    return log


def verdicts(log: pipeline.RunLog) -> dict[str, dict]:
    return {v.site_id: v.model_dump() for v in pipeline.snapshot(log.run_id, log.events).verdicts}


@pytest.mark.live
def test_record_demo_run(tmp_path, monkeypatch):
    log = run_demo(tmp_path, monkeypatch, live=True)
    SNAPSHOT.write_text(pipeline.report(log.dir, log.run_id, log.events)["pr_body_markdown"])
    print({k: (v["decision"], v["reason"]) for k, v in verdicts(log).items()})


def test_demo_run_on_recorded_cassettes(tmp_path, monkeypatch, unrecorded_patcher_calls_stop):
    """Task 18 changed two things these cassettes predate; both go away with the next live re-record:
    - the patcher's repair turn after a deleted test: cs_05's Ultra repair is not recorded for it, so the
      replay stops cs_05 there (see the fixture), with its recorded first try's failure kept;
    - the free-text per-case floor: under it cs_04's Lightning dev run (judge mean 4.24) fails on a case
      below 3.0 and the pipeline iterates, which was never recorded. This replay keeps its recorded rule."""
    monkeypatch.setattr(aggregate, "JUDGE_CASE_FLOOR", aggregate.RUBRIC_MIN)
    log = run_demo(tmp_path, monkeypatch, live=False)
    snap = pipeline.snapshot(log.run_id, log.events)

    # What the recorded run measured on this fixture dataset (see RECORDED); every decision is asserted
    found = verdicts(log)
    assert {k: v["decision"] for k, v in found.items()} == RECORDED  # six sites; emb_01 is never routed
    for site_id in ("cs_01", "cs_02", "cs_03"):  # no tier reached parity: kept, with the best tier's number
        v = found[site_id]
        assert v["best_tier"] == "lightning" and v["parity_pct"] < 95
        assert f"parity vs reference {v['parity_pct']:.1f}% < 95%" in v["reason"]
    # cs_05, the reasoning canary, reached parity on Lightning here (12 of 12 oracle answers) but its patch
    # did not pass, and the site was re-announced for review; cs_06's patch passed
    routed = {
        sid: [(e.payload["decision"], e.payload["tier"]) for e in log.events
              if e.type == "verdict" and e.call_site_id == sid]
        for sid in ("cs_05", "cs_06")
    }  # fmt: skip
    assert routed == {
        "cs_05": [("MIGRATE", "lightning"), ("MANUAL_REVIEW", "lightning")],
        "cs_06": [("MIGRATE", "lightning")],
    }
    assert found["cs_05"]["reason"] == (
        "patch not composed: no recording of this patcher call; last failure: patch rejected (patch rules): "
        "patch removes tests from tests/test_call_sites.py: "
        "['test_openai_reasoning_site_follows_the_o3_parameter_rules']"
    )
    # cs_04's and cs_06's patches passed the demo repo's own ruff + pytest, alone and together
    patch = json.loads((log.dir / "patch.json").read_text())
    assert (patch["alone"], patch["combined"]) == (
        {"cs_04": True, "cs_05": False, "cs_06": True},
        ["cs_04", "cs_06"],
    )
    assert {"helphub/qa/rag_answer.py", "helphub/summaries/summarize.py"} <= set(patch["files"])
    assert "helphub/refunds/plan.py" not in patch["files"]
    assert snap.totals.sites == 6 and snap.totals.cases == sum(c.dev + c.holdout for c in snap.case_summaries)
    # cache-only: every response came from the cassettes, nothing was spent
    assert (snap.totals.tokens_in, snap.totals.tokens_out, snap.totals.tf_spend_usd) == (0, 0, 0.0)
    assert any("ran locally (nebius job: not configured)" in e.payload.get("message", "") for e in log.events)
    assert pipeline.report(log.dir, log.run_id, log.events)["free_text_arg_exemptions"] == {
        "cs_02": ["escalate_to_human.priority", "escalate_to_human.reason"]
    }

    body = pipeline.report(log.dir, log.run_id, log.events)["pr_body_markdown"]
    if os.environ.get("UPDATE_SNAPSHOTS"):  # after an intended template change, offline; review the diff
        SNAPSHOT.write_text(body)
    assert body == SNAPSHOT.read_text()
    for site_id, v in found.items():  # every verdict is in the PR, with the number it was decided on
        assert site_id in body
        if v["parity_pct"] is not None:
            assert f"{v['parity_pct']:.1f}%" in body
