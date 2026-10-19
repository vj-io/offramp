import json

from sqlmodel import Session, select

from offramp import db
from offramp.schemas import Verdict as VerdictSchema


def test_init_db_is_idempotent_and_creates_the_prd_tables(tmp_path):
    eng = db.init_db(db.engine(tmp_path / "sub" / "offramp.db"))
    db.init_db(eng)
    names = set(db.SQLModel.metadata.tables)
    expected = {"run", "callsite", "evalcase", "baseline", "adapter", "tierresult", "verdict", "artifact"}
    assert expected <= names
    assert (tmp_path / "sub" / "offramp.db").exists()


def test_round_trip_in_a_temp_sqlite(tmp_path):
    eng = db.init_db(db.engine(tmp_path / "offramp.db"))
    totals = {"sites": 2, "migrated": 1}
    verdict = VerdictSchema(
        site_id="cs_01", decision="MIGRATE", tier="super", model="nvidia/nemotron-3-super-120b-a12b",
        parity_pct=97.8, iterations=1, cost_before_per_1k=3.1, cost_after_per_1k=0.48, reason="ok",
    )
    with Session(eng) as s:
        s.add(db.Run(id="r1", repo_url="https://github.com/x/y", totals_json=json.dumps(totals)))
        s.add(
            db.CallSite(
                run_id="r1", site_id="cs_01", file="app/a.py", line_start=3, line_end=9, sdk="openai",
                source_model="gpt-4o-mini", contract_json='{"task_type": "classification"}',
            )
        )
        s.add(db.EvalCase(run_id="r1", case_id="cs_01:000", site_id="cs_01", split="dev", input_json="[]"))
        s.add(
            db.Baseline(
                run_id="r1", case_id="cs_01:000", model="gpt-4o-mini", output_json="{}", latency_ms=300,
                input_tokens=10, output_tokens=2, cost_usd=0.0001,
            )
        )
        s.add(db.Adapter(run_id="r1", site_id="cs_01", version=0, patch="--- a\n+++ b\n"))
        s.add(db.TierResult(run_id="r1", site_id="cs_01", adapter_version=0, tier="lightning", split="dev",
                            passed=False, score=0.8, p50_ms=200, p95_ms=400, tokens_in=1000, tokens_out=100,
                            cost_per_1k_calls_usd=0.01, failing_case_ids_json='["cs_01:002"]'))
        s.add(db.Verdict(run_id="r1", **verdict.model_dump()))
        s.add(db.Artifact(run_id="r1", kind="pr_url", path="https://github.com/x/y/pull/1"))
        s.commit()

    with Session(db.engine(tmp_path / "offramp.db")) as s:  # fresh engine: it really hit the file
        run = s.get(db.Run, "r1")
        assert run.stage == "QUEUED" and run.created_at is not None
        assert json.loads(run.totals_json) == totals
        site = s.exec(select(db.CallSite).where(db.CallSite.run_id == "r1")).one()
        assert (site.site_id, site.line_start, json.loads(site.contract_json)["task_type"]) == (
            "cs_01", 3, "classification")
        tr = s.exec(select(db.TierResult)).one()
        assert tr.passed is False and json.loads(tr.failing_case_ids_json) == ["cs_01:002"]
        v = s.exec(select(db.Verdict)).one()
        assert VerdictSchema(**v.model_dump(exclude={"id", "run_id"})) == verdict
        assert s.exec(select(db.Artifact)).one().kind == "pr_url"


def test_site_ids_may_repeat_across_runs(tmp_path):
    eng = db.init_db(db.engine(tmp_path / "offramp.db"))
    with Session(eng) as s:
        for run_id in ("r1", "r2"):
            s.add(db.Run(id=run_id, repo_url="u"))
            site = dict(site_id="cs_01", file="a.py", line_start=1, line_end=2, sdk="openai")
            s.add(db.CallSite(run_id=run_id, **site))
        s.commit()
        assert len(s.exec(select(db.CallSite).where(db.CallSite.site_id == "cs_01")).all()) == 2


def test_default_engine_path_follows_offramp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    db.init_db()
    assert (tmp_path / "data" / "offramp.db").exists()
