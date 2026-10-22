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
from offramp.evals.route import SiteRoute
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
from offramp.tf import prices as tf_prices

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
    contract: Contract = CONTRACT,
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
        baselines.save_sites(evals, [site.model_copy(update={"contract": contract, "function": function})])
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


def matrix_cell(m: Matrix, site_id: str, tier: Tier, passed: bool = True) -> TierResult:
    dev = [c for c in m.cases if c.site_id == site_id and c.split == "dev"]
    rows = [CaseScore(case_id=c.id, passed=passed, score=float(passed)) for c in dev]
    return tier_result(site_id, tier, passed=passed).model_copy(
        update={"case_scores": rows, "failing_case_ids": [r.case_id for r in rows if not passed]}
    )


class Recorder:
    """The injected stages, recording what they were given."""

    def __init__(self, matrix_entries: list[dict] | None = None, passed: bool = True):
        self.matrices: list[Matrix] = []
        self.patches: list[dict] = []
        self.prs: list[dict] = []
        self.matrix_entries, self.passed = matrix_entries, passed

    def eval_matrix(self, m: Matrix) -> MatrixResults:
        self.matrices.append(m)
        rows = self.matrix_entries
        if rows is None:  # every dev case passing (or failing), scored per case as run_tier scores them
            rows = [
                {
                    "kind": "tier_result",
                    "site_id": s.id,
                    "tier": t.value,
                    "split": "dev",
                    "result": matrix_cell(m, s.id, t, self.passed).model_dump(mode="json"),
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


def test_a_migrated_verdict_carries_its_tiers_token_factory_price(tmp_path, monkeypatch):
    """Cache-only: never the model API; the newest cached read when there is one, else pricing.yaml."""
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    cached = tmp_path / "data" / "tf-prices" / "2026-10-21.json"
    cached.parent.mkdir(parents=True)
    cached.write_text(pipeline.json.dumps({"prices": {"nvidia/Nemotron-3_5-Lightning": [0.06, 0.24]}}))
    repo = make_repo(tmp_path)
    log, _ = start(tmp_path, repo)
    llm, stub = llm_for_full_run(*split_counts(repo)), Recorder()

    stage = pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr).execute()

    assert stage == Stage.DONE
    (verdict,) = [e.payload for e in log.events if e.type == "verdict"]
    assert verdict["after_price"] == {
        "model": "nvidia/Nemotron-3_5-Lightning",
        "input_usd_per_m": 0.06,
        "output_usd_per_m": 0.24,
        "via": "token_factory_api",
        "source": "Token Factory model API, 2026-10-21 (cached; cache-only run)",
    }
    saved = pipeline.json.loads((log.dir / "verdicts.json").read_text())
    assert saved[0]["after_price"] == verdict["after_price"]  # a resumed run keeps it


def test_only_the_migrated_tiers_are_priced(tmp_path):
    repo = make_repo(tmp_path)
    log, _ = start(tmp_path, repo)
    llm, stub, asked = llm_for_full_run(*split_counts(repo)), Recorder(), []

    def after_prices(tiers):
        asked.append(tiers)
        return {t: tf_prices.from_yaml(t, "test") for t in tiers}

    pipeline.Pipeline(
        log, llm, after_prices=after_prices, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr
    ).execute()

    assert asked == [[Tier.LIGHTNING]]
    (verdict,) = [e.payload for e in log.events if e.type == "verdict"]
    assert verdict["after_price"]["source"] == "pricing.yaml as of 2026-10-21; test"


def pr_url(log: pipeline.RunLog) -> str:
    return next(e.payload["url"] for e in log.events if e.type == "pr")


def test_the_routing_loop_gets_the_cells_the_eval_matrix_measured(tmp_path):
    """Their scores are reused, never judged again (run 20261022-023459043-590031 re-judged them), and the dev
    cells tell the loop which tier's holdout secures a verdict first."""
    repo = make_repo(tmp_path)
    log, _ = start(tmp_path, repo)
    llm, stub, seen = llm_for_full_run(*split_counts(repo)), Recorder(), []

    def route(site, cases, refs, adapter, client, config, **kw):
        seen.append(kw["prior"])
        return SiteRoute(Verdict(site_id=site.id, decision="MANUAL_REVIEW", reason="stub"), None, [])

    pipeline.Pipeline(
        log, llm, eval_matrix=stub.eval_matrix, route=route, patch=stub.patch, pr=stub.pr
    ).execute()

    (prior,) = seen
    (m,) = stub.matrices
    assert prior == {(t, 0, "dev"): matrix_cell(m, "cs_01", t) for t in RunConfig(repo_url="x").tiers}


def test_every_site_secures_its_verdict_before_any_site_is_routed(tmp_path):
    """Routing goes site by site, so one site's iterations could spend the Ultra cap another site's holdout
    needed (run 20261022-023459043-590031: cs_01 iterated first, cs_04 never ran its holdout). The holdouts
    of tiers that passed dev above a failing one run first; routing gets them with the matrix's cells."""
    repo = make_repo(tmp_path)
    n_dev, n_holdout = split_counts(repo)
    log, _ = start(tmp_path, repo)
    llm, seen = llm_for_full_run(n_dev, n_holdout), []

    class Matrix(Recorder):  # Lightning failed dev, Super and Ultra passed
        def eval_matrix(self, m):
            self.matrix_entries = [
                {"kind": "tier_result", "site_id": "cs_01", "tier": t.value, "split": "dev",
                 "result": matrix_cell(m, "cs_01", t, passed=t != Tier.LIGHTNING).model_dump(mode="json")}
                for t in m.tiers
            ]  # fmt: skip
            return super().eval_matrix(m)

    def route(site, cases, refs, adapter, client, config, **kw):
        seen.append(kw["prior"])
        return SiteRoute(Verdict(site_id=site.id, decision="MANUAL_REVIEW", reason="stub"), None, [])

    stub = Matrix()
    pipeline.Pipeline(
        log, llm, eval_matrix=stub.eval_matrix, route=route, patch=stub.patch, pr=stub.pr
    ).execute()

    (prior,) = seen
    holdout = prior[Tier.SUPER, 0, "holdout"]
    assert holdout.passed and holdout.cases == n_holdout and (Tier.ULTRA, 0, "holdout") not in prior
    assert [c["role"] for c in llm.calls].count("eval_target") == n_holdout


def test_a_prepared_site_gets_how_its_code_parses_the_reply_from_the_code_as_it_is_now(tmp_path):
    """Contract.reply_json is read from the code, not labelled: a run takes it from its own scan, also for a
    site whose contract comes from evals/sites.json (prepared before the code, or Offramp, changed)."""
    repo = make_repo(tmp_path)
    app = repo / "app" / "triage.py"
    app.write_text(
        "import json\n\n"
        + APP.replace(
            "return resp.choices[0].message.content", "return json.loads(resp.choices[0].message.content)"
        )
    )
    git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qam", "json reply")
    log, _ = start(tmp_path, repo)
    llm, stub = llm_for_full_run(*split_counts(repo)), Recorder()

    assert (
        pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr).execute()
        == Stage.DONE
    )

    (site,) = [e.payload for e in log.events if e.type == "site"]
    assert site["contract"]["task_type"] == "classification" and site["contract"]["reply_json"] == "exact"
    adapter = pipeline.json.loads((log.dir / "adapters.json").read_text())["adapters"]["cs_01"]
    assert "Reply with the JSON only" in adapter["spec"]["user_suffix"]


def duplicate_inputs(repo: Path, *, across_splits: bool) -> None:
    """Give one case the input of another, as `offramp prepare` did before it dropped repeats (run
    20261022-023459043-590031): across dev and holdout, or within one split."""
    cases = baselines.load_cases(repo / "evals", "cs_01")
    first = cases[0]
    other = next(c for c in cases[1:] if (c.split != first.split) == across_splits)
    rows = [
        c.model_copy(update={"input_messages": first.input_messages}) if c.id == other.id else c
        for c in cases
    ]
    (repo / "evals" / "cases" / "cs_01.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in rows))
    baselines.write_manifest(repo / "evals")
    git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qam", "duplicate input")


def test_a_run_on_a_dataset_with_an_input_in_both_splits_refuses_to_start(tmp_path):
    """Review M4: the Iterator would tune on inputs the holdout verdict then reads; a warning let it run."""
    repo = make_repo(tmp_path)
    duplicate_inputs(repo, across_splits=True)
    log, _ = start(tmp_path, repo)
    llm = FakeLLM()  # nothing may be spent on it: any call fails the test

    assert pipeline.Pipeline(log, llm, eval_matrix=Recorder().eval_matrix).execute() == Stage.FAILED
    reason = log.events[-1].payload["reason"]
    assert reason.startswith(
        "ValueError: re-run `offramp prepare`: 1 duplicate input spans dev and holdout (cs_01:"
    )
    assert llm.calls == [] and log.events[-1].payload["stage"] == "CLONED"


def test_inputs_repeated_within_one_split_are_a_warning_with_their_count(tmp_path):
    repo = make_repo(tmp_path)
    duplicate_inputs(repo, across_splits=False)
    log, _ = start(tmp_path, repo)
    llm, stub = llm_for_full_run(*split_counts(repo)), Recorder()

    assert (
        pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr).execute()
        == Stage.DONE
    )
    warnings = [e.payload["message"] for e in log.events if e.type == "log" and e.payload["level"] == "warn"]
    assert any(m.startswith("1 duplicate input repeats within one split (cs_01: cs_01:") for m in warnings)


def test_a_refused_adapter_agent_reply_is_on_the_sites_record_for_the_pr(tmp_path):
    """Task 29 A: the Adapter Agent's v0 failed the sanity gate, so the translation ran; the PR is told."""
    repo = make_repo(tmp_path)
    log, _ = start(tmp_path, repo)
    llm, stub = llm_for_full_run(*split_counts(repo)), Recorder()
    llm.queues["adapter"][0] = {
        "reasoning": "off",
        "output_mode": "text",
        "user_suffix": "Labels: " + "billing " * 20,
    }

    assert (
        pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr).execute()
        == Stage.DONE
    )

    result = pipeline.json.loads((log.dir / "evals" / "results" / "cs_01.json").read_text())
    assert result["refused"] == [
        "Adapter v0 refused before evaluation (the translation ran instead): user_suffix repeats 'billing' "
        "20 times in a row"
    ]
    (pr,) = stub.prs
    assert pr["routes"]["cs_01"]["refused"] == result["refused"]


class FakeGitHub:
    """GitHub's REST API for acme/tiny, as github.open_pr calls it; records every request."""

    def __init__(self):
        self.requests: list = []

    def __call__(self, request):
        import httpx

        self.requests.append(request)
        repo = "/repos/acme/tiny"
        routes = {
            ("GET", repo): {"default_branch": "main"},
            ("GET", f"{repo}/git/ref/heads/main"): {"object": {"sha": "base"}},
            ("GET", f"{repo}/git/commits/{request.url.path.rsplit('/', 1)[-1]}"): {"tree": {"sha": "tree"}},
            ("POST", f"{repo}/git/trees"): {"sha": "new-tree"},
            ("POST", f"{repo}/git/commits"): {"sha": "new-commit"},
            ("POST", f"{repo}/git/refs"): {},
            ("POST", f"{repo}/pulls"): {"html_url": "https://github.com/acme/tiny/pull/9"},
        }
        body = routes.get((request.method, request.url.path))
        return httpx.Response(404 if body is None else 201, json=body or {"message": "Not Found"})

    def opener(self):
        import functools

        import httpx

        from offramp import github

        return functools.partial(github.open_pr, token="t", transport=httpx.MockTransport(self))


def no_pr_run(tmp_path, *, migrate: bool = True) -> pipeline.RunLog:
    """A finished `--no-pr` run as `offramp run --no-pr` leaves it: pr.json holds the rendered title, body."""
    repo = make_repo(tmp_path)
    log, _ = start(tmp_path, repo, open_pr=False, max_iterations=0)
    n_dev, n_holdout = split_counts(repo)
    llm, stub = llm_for_full_run(n_dev, n_holdout), Recorder(passed=migrate)
    if not migrate:
        llm.queues["eval_target"] = type(llm.queues["eval_target"])(["bug"] * (3 * n_dev + n_holdout))
    assert (
        pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr).execute()
        == Stage.DONE
    )
    return log


def test_offramp_pr_opens_a_finished_no_pr_runs_pr_from_its_own_artifacts(tmp_path):
    """Task 29 B: a capped run is never continued to open its PR; its recorded patch and body are opened as
    they are, on the branch the PR stage uses, and its event log gains the PR, so export-replay shows it."""
    log = no_pr_run(tmp_path)
    before = len(log.events)
    gh = FakeGitHub()

    url = pipeline.open_pr_from_run(log.run_id, tmp_path / "data", opener=gh.opener())

    assert url == "https://github.com/acme/tiny/pull/9"
    ref = next(r for r in gh.requests if r.url.path.endswith("/git/refs"))
    assert pipeline.json.loads(ref.content)["ref"] == f"refs/heads/offramp/{log.run_id}"
    commit = pipeline.json.loads(next(r for r in gh.requests if r.url.path.endswith("/git/commits")).content)
    assert commit["parents"] == [pipeline.json.loads((log.dir / "clone.json").read_text())["commit_sha"]]
    pulls = pipeline.json.loads(next(r for r in gh.requests if r.url.path.endswith("/pulls")).content)
    pr = pipeline.json.loads((log.dir / "pr.json").read_text())
    assert (pulls["title"], pulls["body"], pr["url"]) == (pr["title"], pr["body"], url)
    again = pipeline.RunLog(log.dir, log.run_id)
    added = [e.payload.get("stage", e.type) for e in again.events[before:]]
    assert added == ["pr", "PR_OPENED", "DONE", "done"] and again.stage == Stage.DONE
    snap = pipeline.snapshot(again.run_id, again.events)
    assert snap.artifacts["pr_url"] == url and snap.stage == Stage.DONE
    with pytest.raises(ValueError, match="already opened its PR"):
        pipeline.open_pr_from_run(log.run_id, tmp_path / "data", opener=gh.opener())


def test_offramp_pr_holds_a_lock_so_a_second_call_cannot_reach_github(tmp_path):
    """Review M4: two concurrent `offramp pr` calls both reached GitHub; one crashed on a shared tmp file."""
    log = no_pr_run(tmp_path)
    gh = FakeGitHub()
    (log.dir / "pr.lock").write_text("held by another `offramp pr`")
    with pytest.raises(ValueError, match="a PR is already being opened for this run"):
        pipeline.open_pr_from_run(log.run_id, tmp_path / "data", opener=gh.opener())
    assert gh.requests == []
    (log.dir / "pr.lock").unlink()
    pipeline.open_pr_from_run(log.run_id, tmp_path / "data", opener=gh.opener())
    assert not (log.dir / "pr.lock").exists()  # released when done
    report = pipeline.json.loads((log.dir / "report.json").read_text())  # review M6 (mutant B7)
    assert report["snapshot"]["artifacts"]["pr_url"] == "https://github.com/acme/tiny/pull/9"


def test_offramp_pr_refuses_a_run_that_migrated_nothing_or_has_not_finished(tmp_path):
    log = no_pr_run(tmp_path, migrate=False)
    with pytest.raises(ValueError, match="migrated nothing"):
        pipeline.open_pr_from_run(log.run_id, tmp_path / "data", opener=FakeGitHub().opener())
    unfinished = pipeline.new_run(RunConfig(repo_url="x", open_pr=False), tmp_path / "data")
    with pytest.raises(ValueError, match="has not finished"):
        pipeline.open_pr_from_run(unfinished.run_id, tmp_path / "data", opener=FakeGitHub().opener())


def test_a_prepared_site_gets_its_calls_timeout_from_the_code_as_it_is_now(tmp_path):
    """Task 29 C: Contract.timeout_s is read from the code, also for a contract from evals/sites.json."""
    repo = make_repo(tmp_path)
    (repo / "app" / "triage.py").write_text(APP.replace("client = OpenAI()", "client = OpenAI(timeout=30)"))
    git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qam", "timeout")
    log, _ = start(tmp_path, repo)
    llm, stub = llm_for_full_run(*split_counts(repo)), Recorder()

    assert (
        pipeline.Pipeline(log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, pr=stub.pr).execute()
        == Stage.DONE
    )

    (site,) = [e.payload for e in log.events if e.type == "site"]
    assert site["contract"]["task_type"] == "classification" and site["contract"]["timeout_s"] == 30.0
    assert (
        "timeout_s" not in pipeline.json.loads(llm.calls[1]["messages"][1]["content"])["contract"]
    )  # adapter


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


def test_a_matrix_cell_the_judge_could_not_score_fails_the_run_resumably(tmp_path):
    """Task 29 D: run 20261022-065443547-22ed36 left cs_05's Lightning dev cell empty after a judge reply off
    the rubric schema, with a warning only. A cell is never left silently empty: the run fails, and a resume
    runs the matrix again (cached cells are free)."""
    repo = make_repo(tmp_path)
    log, _ = start(tmp_path, repo)
    cell = {"kind": "error", "site_id": "cs_01", "tier": "lightning", "split": "dev"}
    judge = cell | {"error_type": "ProviderError", "message": "judge: reply does not match the rubric schema"}
    llm = FakeLLM(
        {
            "scanner": [ContractLabel(contract=CONTRACT, confidence=0.9)],
            "adapter": [{"reasoning": "off", "output_mode": "text"}],
        }
    )

    assert (
        pipeline.Pipeline(log, llm, eval_matrix=Recorder(matrix_entries=[judge]).eval_matrix).execute()
        == Stage.FAILED
    )
    error = log.events[-1].payload
    assert error["stage"] == "EVALUATING"
    assert (
        error["reason"] == "ProviderError: cs_01 lightning dev: judge: reply does not match the rubric schema"
    )
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
    assert verdict["failed_gate"] == "budget"
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
    # the per-site results file carries the decision the run ended on, and the eval stage's beside it
    result = pipeline.json.loads((log.dir / "evals" / "results" / "cs_01.json").read_text())
    assert (
        result["verdict"]["decision"] == "MANUAL_REVIEW"
        and result["verdict"]["reason"] == "repo tests failed"
    )
    assert result["eval_verdict"]["decision"] == "MIGRATE" and result["tier_results"]


def test_results_files_written_from_a_finished_runs_artifacts_are_unchanged_by_a_rewrite(tmp_path):
    routes = {
        "cs_01": {
            "site": {"id": "cs_01"},
            "verdict": {"site_id": "cs_01", "decision": "MIGRATE"},
            "adapter": None,
            "tier_results": [{"tier": "lightning"}],
        },
        "cs_02": {
            "site": {"id": "cs_02"},
            "verdict": {"site_id": "cs_02", "decision": "KEEP_CLOSED"},
            "adapter": None,
            "tier_results": [],
        },
    }
    final = [
        {"site_id": "cs_01", "decision": "MANUAL_REVIEW"},
        {"site_id": "cs_02", "decision": "KEEP_CLOSED"},
    ]

    pipeline.write_results(tmp_path, routes, final)
    first = {p.name: p.read_text() for p in (tmp_path / "evals" / "results").iterdir()}
    again = {k: pipeline.json.loads(v) for k, v in first.items()}
    pipeline.write_results(tmp_path, {k.removesuffix(".json"): v for k, v in again.items()}, final)

    assert {p.name: p.read_text() for p in (tmp_path / "evals" / "results").iterdir()} == first  # idempotent
    cs_01 = again["cs_01.json"]
    assert list(cs_01) == ["site", "verdict", "eval_verdict", "adapter", "tier_results"]
    assert (cs_01["verdict"]["decision"], cs_01["eval_verdict"]["decision"]) == ("MANUAL_REVIEW", "MIGRATE")
    assert again["cs_02.json"]["verdict"] == again["cs_02.json"]["eval_verdict"]


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


@pytest.mark.parametrize("public", [False, True])
def test_langsmith_links_are_private_unless_the_run_opts_in_to_public_ones(tmp_path, monkeypatch, public):
    """A public share link exposes the target's prompts and inputs to anyone with it: only on
    OFFRAMP_LANGSMITH_PUBLIC=1. Otherwise the PR and report carry the dataset's private URL."""
    if public:
        monkeypatch.setenv("OFFRAMP_LANGSMITH_PUBLIC", "1")
    else:
        monkeypatch.delenv("OFFRAMP_LANGSMITH_PUBLIC", raising=False)
    calls = []

    def link(kind):
        def get(site, *, run_id):
            calls.append((kind, site.id, run_id))
            return f"https://smith.test/{kind}/{site.id}"

        return get

    monkeypatch.setattr(pipeline.langsmith, "share_link", link("public"))
    monkeypatch.setattr(pipeline.langsmith, "dataset_url", link("private"))
    repo = make_repo(tmp_path)
    log, _ = start(tmp_path, repo, open_pr=False)
    llm = llm_for_full_run(*split_counts(repo)).add("pr_writer", {"summary": "", "sites": []})

    assert (
        pipeline.Pipeline(log, llm, eval_matrix=Recorder().eval_matrix, patch=Recorder().patch).execute()
        == Stage.DONE
    )

    body = pipeline.report(log.dir, log.run_id, log.events)["pr_body_markdown"]
    kind = "public" if public else "private"
    assert calls == [(kind, "cs_01", log.run_id)]  # never a share link without the opt-in
    label = "public, read-only" if public else "private: members of the LangSmith workspace only"
    assert f"LangSmith ({label}): [cs_01](https://smith.test/{kind}/cs_01)" in body


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


def test_secrets_never_leave_in_the_pr_body_or_the_report(tmp_path, monkeypatch):
    """Verdict reasons carry sandbox output tails into the PR body: it is scrubbed before GitHub gets it, and
    pr.json and the report with it."""
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "not-a-real-secret-value")
    repo = make_repo(tmp_path)
    log, _ = start(tmp_path, repo)
    opened = {}

    def leaky_patch(repo, sites, verdicts, adapters, llm, work_dir, *, config):
        why = "repo checks failed: env SOME_SERVICE_TOKEN=not-a-real-secret-value"
        down = [v.model_copy(update={"decision": "MANUAL_REVIEW", "reason": why}) for v in verdicts]
        return FakePatch(files={"app/triage.py": "# patched\n"}, verdicts=down, adapters=adapters)

    def open_pr(repo, *, run_id, files, title, body, **kw):
        opened.update(title=title, body=body)
        return "https://github.com/acme/tiny/pull/9"

    monkeypatch.setattr(pipeline.github, "open_pr", open_pr)
    llm = llm_for_full_run(*split_counts(repo)).add("pr_writer", {"summary": "", "sites": []})
    stage = pipeline.Pipeline(log, llm, eval_matrix=Recorder().eval_matrix, patch=leaky_patch).execute()

    assert stage == Stage.DONE and "SOME_SERVICE_TOKEN=***" in opened["body"]
    left = [opened["body"], (log.dir / "pr.json").read_text(), (log.dir / "report.json").read_text()]
    left.append(pipeline.json.dumps(pipeline.report(log.dir, log.run_id, log.events)))
    assert not [text for text in left if "not-a-real-secret-value" in text]
    # a pr.json written before bodies were scrubbed: the report scrubs it on the way out
    (log.dir / "pr.json").write_text(
        pipeline.json.dumps({"body": "SOME_SERVICE_TOKEN=not-a-real-secret-value"})
    )
    assert pipeline.report(log.dir, log.run_id, log.events)["pr_body_markdown"] == "SOME_SERVICE_TOKEN=***"


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
    """GC27a: a site without reference rows (the demo's free-text sites; here a classification site, since
    baselines_for does not depend on the task) is compared against the closed model's answers, so its case
    summary counts them under the closed model's label; the dropped case's answer is not one of them."""
    log, stub = run_full(tmp_path, make_repo(tmp_path, references=False, closed=True))
    summary = next(e.payload for e in log.events if e.type == "case_summary")
    assert summary["references"] == 20
    assert summary["reference_methods"] == {CLOSED: 20}  # the label the PR's reference table uses too
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


# ---- the PR body the real PR writer renders from this data (pr_writer itself is Task 10's)

FREE_TEXT = Contract(task_type="free_text", output="text")


def pr_body(tmp_path: Path, repo: Path, llm: FakeLLM, route=None, passed: bool = True) -> str:
    """Run the pipeline with the real PR writer (no GitHub: open_pr=False) and return the body it rendered.
    `passed`: whether the eval matrix found every tier passing dev (the routing loop reuses its cells)."""
    log, _ = start(tmp_path, repo, open_pr=False, max_iterations=0)
    stub = Recorder(passed=passed)
    llm.add("pr_writer", {"summary": "", "sites": []})
    run = pipeline.Pipeline(
        log, llm, eval_matrix=stub.eval_matrix, patch=stub.patch, **({"route": route} if route else {})
    )
    assert run.execute() == Stage.DONE
    return pipeline.report(log.dir, log.run_id, log.events)["pr_body_markdown"]


def keep_closed_llm(repo: Path) -> FakeLLM:
    """Every tier answers the wrong label: KEEP_CLOSED, whose reason the PR prints in full."""
    n_dev, n_holdout = split_counts(repo)
    return FakeLLM(
        {
            "scanner": [ContractLabel(contract=CONTRACT, confidence=0.9)],
            "adapter": [{"reasoning": "off", "output_mode": "text"}],
            "eval_target": ["bug"] * (3 * n_dev + n_holdout),  # dev on 3 tiers (no iterations), holdout once
            "verdict": [{"reason": "The triage call picks a different queue than the references."}],
        }
    )


@pytest.mark.parametrize("auditor", [AUDITOR, "blind:nvidia/Nemotron-3-Ultra-550b-a55b"])
def test_the_pr_body_lists_only_the_rows_compared_and_no_circular_closed_row(tmp_path, auditor):
    repo = make_repo(tmp_path, closed=True, auditor=auditor)  # closed answers incl. dropped case cs_01:099
    body = pr_body(tmp_path, repo, keep_closed_llm(repo), passed=False)

    assert "stays on gpt-4o-mini: " in body  # a kept site: the PR prints its whole reason
    assert "| cs_01 | reference:seed × 20 |" in body  # what the cases were compared against, nothing else
    assert f"{CLOSED} ×" not in body and "Reference outputs for" not in body  # not the dropped case's answer
    clause = f"closed reference {CLOSED} 100.0% on the same references"
    if auditor == AUDITOR:  # the closed model kept only the references it agreed with: no comparison row
        assert "closed reference" not in body
    else:
        assert clause in body


RUNTIME_PROMPT = """

def ask(messages: list[dict]) -> str:
    return client.chat.completions.create(model="gpt-4o-mini", messages=messages).choices[0].message.content
"""


def test_a_call_whose_prompt_is_built_at_runtime_says_why_it_has_no_cases_and_borrows_no_label(tmp_path):
    """idea-reality-mcp's cs_03 (run 20261022-023459043-590031): prepare skipped it, its prompt being built at
    runtime, yet the run told the user to run `offramp prepare`, and the PR table showed the run's own label
    for it (json_extraction) beside the labels the other sites were prepared with (free_text)."""
    repo = make_repo(tmp_path)  # evals/ prepared for cs_01 only, as prepare leaves a repo with such a call
    with (repo / "app" / "triage.py").open("a") as f:
        f.write(RUNTIME_PROMPT)
    git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qam", "runtime prompt")
    llm = llm_for_full_run(*split_counts(repo))
    llm.add(
        "scanner", ContractLabel(contract=Contract(task_type="json_extraction", output="json"), confidence=1)
    )

    body = pr_body(tmp_path, repo, llm)

    assert "| cs_02 | `app/triage.py` | unknown | gpt-4o-mini | **MANUAL REVIEW** |" in body
    (line,) = [ln for ln in body.splitlines() if ln.startswith("- **cs_02**")]
    assert line.endswith(
        "needs manual review: Offramp can't render this call's prompt statically, so it has no eval cases; "
        "runtime capture isn't supported yet"
    )
    assert "| cs_01 | `app/triage.py` | classification |" in body  # the prepared site keeps its label


def test_the_pr_body_names_the_closed_references_of_a_free_text_site(tmp_path):
    repo = make_repo(tmp_path, references=False, closed=True, contract=FREE_TEXT)

    def route(site, cases, references, adapter, client, config, **kw):  # judging free text is Task 8's
        verdict = Verdict(
            site_id=site.id, decision="MIGRATE", tier="lightning", parity_pct=100.0, reason="ok"
        )
        return SiteRoute(verdict, adapter, [tier_result(site.id, Tier.LIGHTNING, "holdout")])

    llm = FakeLLM(
        {
            "scanner": [ContractLabel(contract=FREE_TEXT, confidence=0.9)],
            "adapter": [{"reasoning": "off", "output_mode": "text"}],
        }
    )
    body = pr_body(tmp_path, repo, llm, route=route)

    assert f"Reference outputs for cs_01 were produced by `{CLOSED}`." in body
    assert f"| cs_01 | {CLOSED} × 20 |" in body  # 20 cases, not the dropped case's 21st answer
    assert "where references hold text (cs_01)" in body  # so the pairwise rule applies (GC20, GC27a)
