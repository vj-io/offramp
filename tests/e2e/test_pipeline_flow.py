"""The run state machine end to end on a tiny local repo (Global Constraint 12): stages, events, artifacts,
SQLite rows, failure as a run error, resume after the last finished step.

Agents answer through FakeLLM; the routing loop is the real one (Task 9). Stages owned by other tasks are
injected: the eval matrix (Task 11, whose own tests cover the job path), the patcher and the PR writer
(Task 10). The demo-repo run on recorded cassettes is tests/integration/test_pipeline_demo_repo.py.
"""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel
from sqlmodel import Session, select

from offramp import db, pipeline
from offramp.agents.case_gen import split_for
from offramp.agents.scanner import ContractLabel, scan_repo
from offramp.evals import baselines
from offramp.evals.errors import ProviderError
from offramp.evals.serverless_job.matrix import Matrix, MatrixResults
from offramp.schemas import (
    Adapter,
    BaselineResult,
    CaseScore,
    Contract,
    EvalCase,
    ModelOutput,
    RunConfig,
    Stage,
    Tier,
    TierMetrics,
    TierResult,
    Verdict,
)
from offramp.tf import BudgetExceeded, FakeLLM

APP = """from openai import OpenAI

client = OpenAI()


def triage(ticket: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Classify this ticket as billing, bug or other:\\n{ticket}"}],
        max_tokens=5,
        temperature=0,
    )
    return resp.choices[0].message.content


def embed(text: str) -> list[float]:
    return client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
"""
CONTRACT = Contract(task_type="classification", output="label", labels=["billing", "bug", "other"])
CLOSED = baselines.CLAUDE_CODE_REFERENCE  # GC27: the only closed model this build records
AUDITOR = f"blind:{CLOSED}"
STAGES_DONE = ["QUEUED", "CLONED", "SCANNED", "BASELINED", "EVALUATING", "ITERATING", "VERDICTS", "PATCHED"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No LangSmith or Tavily from the developer's environment: these runs never leave the machine."""
    for name in ("LANGSMITH_API_KEY", "LANGSMITH_API_KEY_FALLBACK", "LANGSMITH_TRACING", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return out.stdout


def make_repo(
    tmp_path: Path,
    *,
    prepared: bool = True,
    function: str = "triage",
    references: bool = True,
    closed: bool = False,
    auditor: str = AUDITOR,
) -> Path:
    """A git repo with one chat call (cs_01) and one embedding call; `prepared` adds evals/ the way
    `offramp prepare` leaves it (sites.json, cases, references blind-audited by `auditor`, audit.json,
    manifest). `closed` adds GC27 closed-model answers for every case plus one dropped case (cs_01:099)."""
    repo = tmp_path / "tiny"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "triage.py").write_text(APP)
    if prepared:
        evals = repo / "evals"
        site = scan_repo(repo).sites[0]
        baselines.save_sites(evals, [site.model_copy(update={"contract": CONTRACT, "function": function})])
        now = datetime.now(UTC)
        cases, refs = [], []
        for i in range(20):
            case_id = f"cs_01:{i:03d}"
            text = f"I was charged twice, order {i}"
            cases.append(
                EvalCase(
                    id=case_id,
                    site_id="cs_01",
                    variables={"ticket": text},
                    input_messages=[
                        {"role": "user", "content": f"Classify this ticket as billing, bug or other:\n{text}"}
                    ],
                    expected_kind="label",
                    tags=["length:short"],
                    split=split_for(case_id),
                )
            )
            refs.append(
                BaselineResult(
                    case_id=case_id,
                    model="reference:seed",
                    output=ModelOutput(text="billing"),
                    verified_by=[auditor],
                    recorded_at=now,
                )
            )
        baselines.save_cases(evals, "cs_01", cases)
        if references:
            baselines.save_references(evals, "cs_01", refs)
        if closed:
            rows = [
                r.model_copy(update={"source": "closed_model", "model": CLOSED, "verified_by": []})
                for r in refs
            ]
            dropped = rows[0].model_copy(update={"case_id": "cs_01:099"})  # its case was dropped by the audit
            baselines.save_closed(evals, "cs_01", [*rows, dropped])
        (evals / "audit.json").write_text(
            '{"cs_01": {"checked": 20, "agreed": 20, "dropped": 0, "human_verified": 0, "unchecked": 0, '
            f'"auditors": ["{AUDITOR}"]}}}}\n'
        )
        baselines.write_manifest(evals)
    git(repo, "init", "-q")
    git(repo, "remote", "add", "origin", "https://github.com/acme/tiny.git")
    git(repo, "add", "-A")
    git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init")
    return repo


def split_counts(repo: Path) -> tuple[int, int]:
    cases = baselines.load_cases(repo / "evals", "cs_01")
    return sum(c.split == "dev" for c in cases), sum(c.split == "holdout" for c in cases)


def tier_result(site_id: str, tier: Tier, split: str = "dev", passed: bool = True) -> TierResult:
    return TierResult(
        site_id=site_id,
        tier=tier,
        adapter_version=0,
        split=split,
        passed=passed,
        score=1.0 if passed else 0.5,
        metrics=TierMetrics(p50_ms=100, p95_ms=200, tokens_in=40, tokens_out=2, cost_per_1k_calls_usd=0.01),
        case_scores=[CaseScore(case_id=f"{site_id}:000", passed=passed, score=1.0)],
    )


class Recorder:
    """The injected stages, recording what they were given."""

    def __init__(self, matrix_entries: list[dict] | None = None):
        self.matrices: list[Matrix] = []
        self.patches: list[dict] = []
        self.prs: list[dict] = []
        self.matrix_entries = matrix_entries

    def eval_matrix(self, m: Matrix) -> MatrixResults:
        self.matrices.append(m)
        rows = self.matrix_entries
        if rows is None:
            rows = [
                {
                    "kind": "tier_result",
                    "site_id": s.id,
                    "tier": t.value,
                    "split": "dev",
                    "result": tier_result(s.id, t).model_dump(mode="json"),
                }
                for s in m.sites
                for t in m.tiers
            ]
        return MatrixResults(results=rows, ran_on="local", fallback_reason="nebius job: not configured")

    def patch(self, repo, sites, verdicts, adapters, llm, work_dir, *, config):
        self.patches.append({"repo": repo, "verdicts": verdicts, "adapters": adapters, "work_dir": work_dir})
        files = {"app/triage.py": "# patched\n"} if any(v.decision == "MIGRATE" for v in verdicts) else {}
        return FakePatch(files=files, verdicts=verdicts, adapters=adapters)

    def pr(self, **kw):
        self.prs.append(kw)
        url = "https://github.com/acme/tiny/pull/7" if kw["open_pr"] else None
        return {"title": "Offramp: migrate 1/1 call sites", "body": "## Evidence\n...", "url": url}


class FakePatch(BaseModel):  # the shape of Task 10's PatchResult that the pipeline reads
    files: dict[str, str]
    verdicts: list[Verdict]
    adapters: dict[str, Adapter]
    site_files: dict[str, list[str]] = {}
    warnings: list[str] = []


def llm_for_full_run(n_dev: int, n_holdout: int) -> FakeLLM:
    return FakeLLM(
        {
            "scanner": [ContractLabel(contract=CONTRACT, confidence=0.9)],
            "adapter": [{"reasoning": "off", "output_mode": "text"}],
            "eval_target": ["billing"] * (n_dev + n_holdout),
            "verdict": [{"reason": "The ticket triage call returns the same queue label as the references."}],
        }
    )


def start(tmp_path: Path, repo: Path, **config) -> tuple[pipeline.RunLog, object]:
    engine = db.init_db(db.engine(tmp_path / "offramp.db"))
    run_config = RunConfig(repo_url=str(repo), live=False, **config)
    return pipeline.new_run(run_config, tmp_path / "data", engine=engine), engine


def types(log: pipeline.RunLog) -> list[str]:
    return [e.payload["stage"] if e.type == "stage" else e.type for e in log.events]


def stages(log: pipeline.RunLog) -> list[str]:
    return [e.payload["stage"] for e in log.events if e.type == "stage"]


def test_full_run_streams_every_stage_and_persists_every_artifact(tmp_path):
    repo = make_repo(tmp_path)
    n_dev, n_holdout = split_counts(repo)
    assert n_dev and n_holdout
    log, engine = start(tmp_path, repo)
    llm, stub = llm_for_full_run(n_dev, n_holdout), Recorder()

    stage = pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr).execute()

    assert stage == Stage.DONE
    assert stages(log) == [*STAGES_DONE, "PR_OPENED", "DONE"]
    assert [e.seq for e in log.events] == list(range(len(log.events)))
    assert log.events[-1].type == "done"
    # cs_01 is the only site; the embedding call is mentioned, never evaluated
    assert [e.payload["id"] for e in log.events if e.type == "site"] == ["cs_01"]
    assert any(e.type == "log" and "emb_01" in e.payload["message"] for e in log.events)
    summary = next(e.payload for e in log.events if e.type == "case_summary")
    assert (summary["dev"], summary["holdout"], summary["references"]) == (n_dev, n_holdout, 20)
    assert summary["reference_methods"] == {"reference:seed": 20}
    assert summary["audit"]["agreed"] == 20 and summary["audit"]["auditor"].startswith("claude-sonnet-5.5")
    # the eval matrix: v0 on every tier's dev cases, with the caps left for this run; its cells are streamed
    (matrix,) = stub.matrices
    assert matrix.splits == ["dev"] and matrix.tiers == RunConfig(repo_url="x").tiers and not matrix.live
    assert set(matrix.budget) == {"max_in", "max_out", "max_ultra_agent_calls"}
    assert any("ran locally (nebius job: not configured)" in e.payload.get("message", "") for e in log.events)
    streamed = [e.payload for e in log.events if e.type == "tier_result"]
    assert all("case_scores" not in p for p in streamed)
    assert {(p["tier"], p["split"]) for p in streamed} >= {("lightning", "dev"), ("lightning", "holdout")}
    # the routing loop (Task 9) on the fakes: Lightning reaches parity on dev, then holdout
    (verdict,) = [e.payload for e in log.events if e.type == "verdict"]
    assert (
        verdict["decision"] == "MIGRATE" and verdict["tier"] == "lightning" and verdict["parity_pct"] == 100
    )
    assert "pricing.yaml" in verdict["cost_before_note"]  # a cache-only run never asks Tavily
    calls = [c["role"] for c in llm.calls]
    assert calls.count("eval_target") == n_dev + n_holdout and calls.count("verdict") == 1
    assert types(log)[-4:] == ["pr", "PR_OPENED", "DONE", "done"]
    (pr,) = stub.prs
    assert pr["open_pr"] and pr["repo"] == "acme/tiny" and pr["patch"]["files"]

    run = log.dir
    for name in ("clone.json", "scan.json", "cases.json", "adapters.json", "matrix.json", "verdicts.json"):
        assert (run / name).exists(), name
    for name in ("patch.json", "pr.json", "report.json", "events.jsonl", "evals/results/cs_01.json"):
        assert (run / name).exists(), name
    result = pipeline.json.loads((run / "evals" / "results" / "cs_01.json").read_text())
    assert result["verdict"]["decision"] == "MIGRATE" and result["tier_results"][0]["case_scores"]
    assert pipeline.read_events(log.file) == log.events
    report = pipeline.json.loads((run / "report.json").read_text())
    assert report["pr_body_markdown"] == "## Evidence\n..." and report["free_text_arg_exemptions"] == {}

    snap = pipeline.snapshot(log.run_id, log.events)
    assert snap.stage == Stage.DONE and snap.config.repo_url == str(repo)
    assert snap.totals.migrated == 1 and snap.totals.sites == 1 and snap.totals.cases == 20
    # one result per (site, tier, version, split), latest emission last: the route re-ran Lightning dev
    assert [(r.tier.value, r.split) for r in snap.tier_results] == [
        ("super", "dev"),
        ("ultra", "dev"),
        ("lightning", "dev"),
        ("lightning", "holdout"),
    ]
    # before: gpt-4o-mini's pricing.yaml price x the tokens measured on Token Factory (FakeLLM reports none)
    assert snap.totals.est_cost_before_per_1k == verdict["cost_before_per_1k"] == 0.0
    assert snap.totals.est_cost_after_per_1k == verdict["cost_after_per_1k"]
    assert snap.artifacts == {"report_url": f"/api/runs/{log.run_id}/report", "pr_url": pr_url(log)}
    with Session(engine) as s:
        assert s.get(db.Run, log.run_id).stage == "DONE"
        assert [v.decision for v in s.exec(select(db.Verdict))] == ["MIGRATE"]
        assert s.exec(select(db.TierResult)).first() is not None
    assert git(repo, "status", "--porcelain") == ""  # a local checkout is read, never written


def pr_url(log: pipeline.RunLog) -> str:
    return next(e.payload["url"] for e in log.events if e.type == "pr")


def test_a_failed_run_resumes_after_its_last_finished_step(tmp_path):
    repo = make_repo(tmp_path)
    n_dev, n_holdout = split_counts(repo)
    log, engine = start(tmp_path, repo)
    stub = Recorder()

    def outage(*args, **kwargs):
        raise ProviderError("Token Factory returned 503 twice", status=503)

    llm = llm_for_full_run(n_dev, n_holdout)
    failed = pipeline.Pipeline(
        log, llm, eval_matrix=stub.eval_matrix, route=outage, patch=stub.patch, pr=stub.pr
    )
    assert failed.execute() == Stage.FAILED
    assert types(log)[-2:] == ["FAILED", "error"]
    assert log.events[-1].payload == {
        "stage": "ITERATING",
        "reason": "ProviderError: Token Factory returned 503 twice",
    }
    assert not any(e.type == "tier_result" and e.payload["split"] == "holdout" for e in log.events)
    assert (log.dir / "report.json").exists() and not (log.dir / "verdicts.json").exists()
    before = len(log.events)

    # a new process: the log is read back from disk. Scan and adapter replies are used up, so a re-run of a
    # finished step would fail the test (FakeLLM raises); only the routing loop and the verdict may call.
    again = pipeline.RunLog(log.dir, log.run_id, engine=engine)
    resumed = pipeline.Pipeline(again, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr)
    assert resumed.execute() == Stage.DONE
    new = again.events[before:]
    assert new[0].type == "log" and "resuming" in new[0].payload["message"]
    assert [e.payload["stage"] for e in new if e.type == "stage"] == [
        "ITERATING",
        "VERDICTS",
        "PATCHED",
        "PR_OPENED",
        "DONE",
    ]
    assert len(stub.matrices) == 1  # the eval matrix finished before the failure: not run again
    assert [e.seq for e in again.events] == list(range(len(again.events)))
    assert pipeline.snapshot(again.run_id, again.events).stage == Stage.DONE

    assert pipeline.Pipeline(again, FakeLLM(), eval_matrix=stub.eval_matrix).execute() == Stage.DONE  # no-op
    assert len(again.events) == len(new) + before


def test_a_cache_miss_in_the_eval_matrix_is_a_run_error(tmp_path):
    repo = make_repo(tmp_path)
    log, _ = start(tmp_path, repo)
    miss = {
        "kind": "error",
        "site_id": "cs_01",
        "tier": "super",
        "split": "dev",
        "error_type": "CacheMiss",
        "message": "not in the cache: 4f1c...",
    }
    stub = Recorder(matrix_entries=[miss])
    llm = FakeLLM(
        {
            "scanner": [ContractLabel(contract=CONTRACT, confidence=0.9)],
            "adapter": [{"reasoning": "off", "output_mode": "text"}],
        }
    )

    assert pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix).execute() == Stage.FAILED
    error = log.events[-1].payload
    assert error["stage"] == "EVALUATING" and error["reason"].startswith("CacheMiss: cs_01 super dev")
    assert not (log.dir / "matrix.json").exists()  # a resume runs the matrix again


def test_a_budget_stop_keeps_the_site_closed_without_a_comparison(tmp_path):
    repo = make_repo(tmp_path)
    log, _ = start(tmp_path, repo)
    stub = Recorder()
    llm = FakeLLM(
        {
            "scanner": [ContractLabel(contract=CONTRACT, confidence=0.9)],
            "adapter": [BudgetExceeded("Ultra agent call cap reached (40)")],
        }
    )

    def route(*args, **kwargs):
        raise AssertionError("a site without an adapter is never routed")

    assert (
        pipeline.Pipeline(
            log, llm, eval_matrix=stub.eval_matrix, route=route, patch=stub.patch, pr=stub.pr
        ).execute()
        == Stage.DONE
    )
    (verdict,) = [e.payload for e in log.events if e.type == "verdict"]
    assert verdict["decision"] == "KEEP_CLOSED" and verdict["parity_pct"] is None
    assert verdict["reason"] == "budget: Ultra agent call cap reached (40)"
    assert stub.matrices == []  # nothing to evaluate
    assert pipeline.snapshot(log.run_id, log.events).totals.kept_closed == 1


def test_an_unprepared_repo_ends_in_manual_review_without_a_pr(tmp_path):
    repo = make_repo(tmp_path, prepared=False)
    log, _ = start(tmp_path, repo)
    stub = Recorder()
    llm = FakeLLM({"scanner": [ContractLabel(contract=CONTRACT, confidence=0.9)]})

    assert (
        pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr).execute()
        == Stage.DONE
    )
    messages = [e.payload["message"] for e in log.events if e.type == "log"]
    assert any("no evals/ in the repo: run `offramp prepare" in m for m in messages)
    (verdict,) = [e.payload for e in log.events if e.type == "verdict"]
    assert verdict["decision"] == "MANUAL_REVIEW" and verdict["reason"].startswith("no dev cases")
    assert stub.patches == [] and stub.prs[0]["open_pr"] is False
    assert "PR_OPENED" not in stages(log) and "pull request not opened: no file changed" in messages
    assert [c["role"] for c in llm.calls] == ["scanner"]  # no adapter call for a site with no cases


def test_cases_prepared_for_another_call_are_not_used(tmp_path):
    repo = make_repo(tmp_path, function="old_triage")  # sites.json says cs_01 was another function
    log, _ = start(tmp_path, repo)
    stub = Recorder()
    llm = FakeLLM({"scanner": [ContractLabel(contract=CONTRACT, confidence=0.9)]})

    assert (
        pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr).execute()
        == Stage.DONE
    )
    messages = [e.payload["message"] for e in log.events if e.type == "log"]
    assert any("cs_01: evals/ was prepared for app/triage.py:old_triage" in m for m in messages)
    assert not any(e.type == "case_summary" for e in log.events)
    (verdict,) = [e.payload for e in log.events if e.type == "verdict"]
    assert verdict["decision"] == "MANUAL_REVIEW" and "prepared for a different call" in verdict["reason"]


def test_no_pr_renders_the_body_and_opens_nothing(tmp_path):
    repo = make_repo(tmp_path)
    n_dev, n_holdout = split_counts(repo)
    log, _ = start(tmp_path, repo, open_pr=False)
    stub = Recorder()

    assert (
        pipeline.Pipeline(
            log,
            llm_for_full_run(n_dev, n_holdout),
            eval_matrix=stub.eval_matrix,
            patch=stub.patch,
            pr=stub.pr,
        ).execute()
        == Stage.DONE
    )
    assert stub.prs[0]["open_pr"] is False
    assert not any(e.type == "pr" for e in log.events) and "PR_OPENED" not in stages(log)
    assert any("not opened (--no-pr)" in e.payload.get("message", "") for e in log.events)
    assert pipeline.report(log.dir, log.run_id, log.events)["pr_body_markdown"] == "## Evidence\n..."


def test_a_patch_that_fails_reannounces_the_site_as_manual_review(tmp_path):
    repo = make_repo(tmp_path)
    n_dev, n_holdout = split_counts(repo)
    log, engine = start(tmp_path, repo)
    stub = Recorder()

    def failing_patch(repo, sites, verdicts, adapters, llm, work_dir, *, config):
        down = [
            v.model_copy(update={"decision": "MANUAL_REVIEW", "reason": "repo tests failed"})
            for v in verdicts
        ]
        return FakePatch(files={}, verdicts=down, adapters=adapters, warnings=["dropped a symlink"])

    llm = llm_for_full_run(n_dev, n_holdout)
    assert (
        pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=failing_patch, pr=stub.pr).execute()
        == Stage.DONE
    )
    verdicts = [e.payload["decision"] for e in log.events if e.type == "verdict"]
    assert verdicts == ["MIGRATE", "MANUAL_REVIEW"]
    snap = pipeline.snapshot(log.run_id, log.events)
    assert snap.verdicts[0].decision == "MANUAL_REVIEW" and snap.totals.migrated == 0
    assert any(e.payload.get("message") == "patch: dropped a symlink" for e in log.events if e.type == "log")
    with Session(engine) as s:
        assert [v.decision for v in s.exec(select(db.Verdict))] == ["MANUAL_REVIEW"]


@pytest.mark.parametrize("live", [True, False])
def test_price_lookup_is_tavily_only_for_live_runs(live):
    lookup = pipeline.default_price_lookup(RunConfig(repo_url="x", live=live), FakeLLM())
    assert (lookup is pipeline.pricing_yaml_price) is not live


def test_totals_cost_kept_sites_on_their_source_model():
    def verdict(site_id, decision, before, after=None):
        return Verdict(site_id=site_id, decision=decision, cost_before_per_1k=before, cost_after_per_1k=after,
                       reason="r")  # fmt: skip

    verdicts = [
        verdict("cs_01", "MIGRATE", 1.0, 0.2),
        verdict("cs_02", "KEEP_CLOSED", 3.0),
        verdict("cs_03", "MANUAL_REVIEW", None),
        verdict("cs_04", "MANUAL_REVIEW", 5.0),  # a downgraded MIGRATE keeps its estimate: still on its model
    ]
    t = pipeline.totals([], [], verdicts)
    assert (t.migrated, t.kept_closed, t.manual_review) == (1, 1, 2)
    assert t.est_cost_before_per_1k == pytest.approx(9.0)
    assert t.est_cost_after_per_1k == pytest.approx(8.2)


def test_the_pr_writer_renders_the_body_from_the_run_and_opens_the_pr(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    n_dev, n_holdout = split_counts(repo)
    log, _ = start(tmp_path, repo)
    stub, opened = Recorder(), {}

    def open_pr(repo, *, run_id, files, title, body, **kw):  # GitHub itself is Task 10's test (MockTransport)
        opened.update(repo=repo, run_id=run_id, files=files, title=title, body=body)
        return "https://github.com/acme/tiny/pull/9"

    monkeypatch.setattr(pipeline.github, "open_pr", open_pr)
    prose = {
        "summary": "Moves ticket triage to Nemotron.",
        "sites": [{"site_id": "cs_01", "note": "Same labels."}],
    }
    llm = llm_for_full_run(n_dev, n_holdout).add("pr_writer", prose)

    assert pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch).execute() == Stage.DONE
    body = pipeline.report(log.dir, log.run_id, log.events)["pr_body_markdown"]
    assert opened["repo"] == "acme/tiny" and opened["run_id"] == log.run_id
    assert opened["files"] == {"app/triage.py": "# patched\n"} and opened["body"] == body
    assert "Moves ticket triage to Nemotron." in body and "## How parity was measured" in body
    assert "100.0%" in body  # the holdout parity the verdict was decided on
    assert pr_url(log) == "https://github.com/acme/tiny/pull/9"
    assert [e.payload["title"] for e in log.events if e.type == "pr"] == [opened["title"]]


def test_a_pr_for_a_repo_that_is_not_on_github_is_a_run_error(tmp_path):
    repo = make_repo(tmp_path)
    git(repo, "remote", "remove", "origin")
    n_dev, n_holdout = split_counts(repo)
    log, _ = start(tmp_path, repo)
    stub = Recorder()

    run = pipeline.Pipeline(
        log, llm_for_full_run(n_dev, n_holdout), eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr
    )
    assert run.execute() == Stage.FAILED
    assert log.events[-1].payload["reason"].endswith("is not a GitHub repository")
    assert stub.prs[0]["open_pr"] is False  # the body was still rendered; resuming after a fix opens it


def test_secrets_never_reach_events_the_event_file_or_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-live-secret-123")
    monkeypatch.setenv("LANGSMITH_API_KEY_FALLBACK", "lsv2-fallback-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret_456789")
    engine = db.init_db(db.engine(tmp_path / "offramp.db"))
    config = RunConfig(repo_url="https://bob:ghp_secret_456789@github.com/acme/tiny")
    log = pipeline.new_run(config, tmp_path / "data", engine=engine)
    log.emit(
        "log", {"level": "error", "message": "Tavily 401 for key tvly-live-secret-123 (lsv2-fallback-secret)"}
    )
    log.emit("stage", {"stage": "FAILED", "detail": "clone https://x:y@example.com/r failed"})

    stored = log.file.read_text() + pipeline.json.dumps([e.model_dump(mode="json") for e in log.events])
    with Session(engine) as s:
        stored += s.get(db.Run, log.run_id).repo_url
    for secret in ("tvly-live-secret-123", "lsv2-fallback-secret", "ghp_secret_456789", "x:y@"):
        assert secret not in stored
    assert "https://***@github.com/acme/tiny" in stored and "for key ***" in stored


def test_the_run_budget_survives_a_resume(tmp_path):
    """GC8 caps are per run, not per attempt: a resumed run starts from what it had already spent."""
    repo = make_repo(tmp_path)
    n_dev, n_holdout = split_counts(repo)
    log, engine = start(tmp_path, repo)
    stub = Recorder()
    spent = {"tokens_in": 1200, "tokens_out": 80, "ultra_agent_calls": 2, "tf_spend_usd": 0.25}

    def matrix_that_spent(m):
        return stub.eval_matrix(m).model_copy(update={"budget": spent})  # usage of its own client

    def outage(*args, **kwargs):
        raise ProviderError("Token Factory returned 503 twice", status=503)

    llm = llm_for_full_run(n_dev, n_holdout)
    first = pipeline.Pipeline(
        log, llm, eval_matrix=matrix_that_spent, route=outage, patch=stub.patch, pr=stub.pr
    )
    assert first.execute() == Stage.FAILED

    again = pipeline.RunLog(log.dir, log.run_id, engine=engine)
    resumed = pipeline.Pipeline(again, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr)
    assert resumed.budget.snapshot()["ultra_agent_calls"] == 2  # not a fresh 0 of 40
    assert resumed.execute() == Stage.DONE
    totals = pipeline.snapshot(again.run_id, again.events).totals
    assert (totals.tokens_in, totals.tokens_out, totals.tf_spend_usd) == (1200, 80, 0.25)


def run_full(tmp_path, repo):
    n_dev, n_holdout = split_counts(repo)
    log, _ = start(tmp_path, repo)
    stub = Recorder()
    llm = llm_for_full_run(n_dev, n_holdout)
    assert (
        pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr).execute()
        == Stage.DONE
    )
    return log, stub


def test_closed_answers_that_are_the_references_are_counted_and_named(tmp_path):
    """GC27a: a site without reference rows (free text in the demo) is compared against the closed model's
    answers, so its case summary counts them and names their method; the dropped case's answer is not one."""
    log, stub = run_full(tmp_path, make_repo(tmp_path, references=False, closed=True))
    summary = next(e.payload for e in log.events if e.type == "case_summary")
    assert summary["references"] == 20
    assert summary["reference_methods"] == {f"closed:{CLOSED}": 20}
    assert stub.prs[0]["cases"]["references"] == {
        "cs_01": {CLOSED: 20}
    }  # what the PR lists, without cs_01:099
    (verdict,) = [e.payload for e in log.events if e.type == "verdict"]
    assert verdict["decision"] == "MIGRATE" and verdict["closed_reference_parity_pct"] is None


def test_no_closed_comparison_against_references_the_closed_model_audited(tmp_path):
    """GC27c is circular when the closed model also blind-audited the references: only its agreeing answers
    were kept, so it scores 100% by construction. The verdict names it with no parity and no clause."""
    log, _ = run_full(tmp_path, make_repo(tmp_path, closed=True))
    summary = next(e.payload for e in log.events if e.type == "case_summary")
    assert summary["reference_methods"] == {"reference:seed": 20}  # every case has its own reference
    (verdict,) = [e.payload for e in log.events if e.type == "verdict"]
    assert verdict["closed_reference_model"] == CLOSED and verdict["closed_reference_parity_pct"] is None
    assert "closed reference" not in verdict["reason"]
    notes = [e.payload["message"] for e in log.events if e.type == "log" and e.call_site_id == "cs_01"]
    assert any("not compared" in m and "by construction" in m for m in notes)


def test_a_closed_model_that_did_not_audit_the_references_is_compared(tmp_path):
    other = "blind:nvidia/Nemotron-3-Ultra-550b-a55b"
    log, _ = run_full(tmp_path, make_repo(tmp_path, closed=True, auditor=other))
    (verdict,) = [e.payload for e in log.events if e.type == "verdict"]
    assert verdict["closed_reference_model"] == CLOSED and verdict["closed_reference_parity_pct"] == 100.0
    assert f"closed reference {CLOSED} 100.0% on the same references" in verdict["reason"]
