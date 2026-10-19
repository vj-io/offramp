"""The whole pipeline on the demo repo (offramp-demo-app), cache-only on committed cassettes (PLAN Task 12):
six verdicts as recorded (RECORDED), the PR body as a snapshot. No mocks: every agent, the eval matrix, the
routing loop, the Docker sandbox and the PR writer run for real; Token Factory is the recorded cassettes.

Inputs (committed, except the demo checkout):
- the demo repo at DEMO_COMMIT, cloned from a local offramp-demo-app checkout (OFFRAMP_DEMO_REPO_DIR, or a
  sibling directory of this repo); skipped without one, and without Docker (the patch stage's sandbox);
- tests/fixtures/pipeline/demo_evals/: the eval dataset `offramp prepare` wrote for that commit, overlaid on
  the clone's evals/. A test fixture, smaller than the real one (target 12 cases per site, 2 adversarial;
  see DATASET_COMMAND), and its seed references are not blind-audited; Task 14 prepares, audits and commits
  the real demo dataset;
- tests/fixtures/cassettes/pipeline/: every Token Factory response the run used, recorded by
  test_record_demo_run (`uv run --env-file .env pytest --live -m live -k record_demo_run`, about $0.50 of
  Token Factory credit), which also writes the PR body snapshot tests/fixtures/pipeline/expected_pr_body.md.
Prices come from pricing.yaml: no Tavily in tests. No LangSmith, no GitHub (open_pr=False), no Serverless Job.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from offramp import pipeline
from offramp.agents.verdict import pricing_yaml_price
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
# Recorded live 2026-10-20 on the fixture dataset ($0.47 of Token Factory credit): no site migrated. cs_01-03
# reached no parity on any tier; cs_04-06 reached it on Lightning, and each patch then failed (no edit blocks
# in the patcher's reply, or the repo's ruff), so they need a manual review.
RECORDED = {
    "cs_01": "KEEP_CLOSED",
    "cs_02": "KEEP_CLOSED",
    "cs_03": "KEEP_CLOSED",
    "cs_04": "MANUAL_REVIEW",
    "cs_05": "MANUAL_REVIEW",
    "cs_06": "MANUAL_REVIEW",
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


def test_demo_run_on_recorded_cassettes(tmp_path, monkeypatch):
    log = run_demo(tmp_path, monkeypatch, live=False)
    snap = pipeline.snapshot(log.run_id, log.events)

    # What the recorded run measured on this fixture dataset (see RECORDED); every decision is asserted
    found = verdicts(log)
    assert {k: v["decision"] for k, v in found.items()} == RECORDED  # six sites; emb_01 is never routed
    for site_id in ("cs_01", "cs_02", "cs_03"):  # no tier reached parity: kept, with the best tier's number
        v = found[site_id]
        assert v["best_tier"] == "lightning" and v["parity_pct"] < 95
        assert f"parity vs reference {v['parity_pct']:.1f}% < 95%" in v["reason"]
    # cs_05, the reasoning canary, reached parity on Lightning here (12 of 12 oracle answers), so routing
    # said MIGRATE; the patch then failed the demo repo's own ruff, and the site was re-announced for review
    routed = [e.payload for e in log.events if e.type == "verdict" and e.call_site_id == "cs_05"]
    assert [(v["decision"], v["tier"]) for v in routed] == [
        ("MIGRATE", "lightning"),
        ("MANUAL_REVIEW", "lightning"),
    ]
    assert routed[-1]["reason"].startswith("repo checks failed on the patch (docker)")
    assert snap.totals.sites == 6 and snap.totals.cases == sum(c.dev + c.holdout for c in snap.case_summaries)
    # cache-only: every response came from the cassettes, nothing was spent
    assert (snap.totals.tokens_in, snap.totals.tokens_out, snap.totals.tf_spend_usd) == (0, 0, 0.0)
    assert any("ran locally (nebius job: not configured)" in e.payload.get("message", "") for e in log.events)
    assert pipeline.report(log.dir, log.run_id, log.events)["free_text_arg_exemptions"] == {
        "cs_02": ["escalate_to_human.priority", "escalate_to_human.reason"]
    }

    body = pipeline.report(log.dir, log.run_id, log.events)["pr_body_markdown"]
    assert body == SNAPSHOT.read_text()
    for site_id, v in found.items():  # every verdict is in the PR, with the number it was decided on
        assert site_id in body
        if v["parity_pct"] is not None:
            assert f"{v['parity_pct']:.1f}%" in body
