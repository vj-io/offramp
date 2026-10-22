"""The API contract the dashboard depends on (Global Constraint 13) and replay runs (Global Constraint 14).

Runs here never call Token Factory: replays play a recording written by the test (a stand-in for a real run's
events, timestamps 10 ms apart), and queued runs execute an injected stand-in for the pipeline.
"""

import json
import threading
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from offramp import db, pipeline
from offramp.api import app as api
from offramp.api.app import RateLimit, create_app, replay_delays
from offramp.schemas import RunConfig, RunEvent, Stage

RECORDED_ID = "20261019-120000-abc123"
CONFIG = {"repo_url": "https://github.com/acme/tiny", "threshold": 0.9, "live": False}


def recorded_events(gap_ms: int = 10) -> list[RunEvent]:
    t0 = datetime(2026, 10, 19, 12, tzinfo=UTC)
    site = {
        "id": "cs_01",
        "file": "app/triage.py",
        "lines": [7, 12],
        "sdk": "openai",
        "call": "client.chat.completions.create",
        "prompt_source": "fstring",
    }
    result = {
        "site_id": "cs_01",
        "tier": "lightning",
        "adapter_version": 0,
        "split": "holdout",
        "passed": True,
        "score": 1.0,
        "metrics": {
            "p50_ms": 100,
            "p95_ms": 200,
            "tokens_in": 40,
            "tokens_out": 2,
            "cost_per_1k_calls_usd": 0.01,
        },
    }
    verdict = {
        "site_id": "cs_01",
        "decision": "MIGRATE",
        "tier": "lightning",
        "parity_pct": 100.0,
        "reason": "ok",
    }
    summary = {"site_id": "cs_01", "dev": 14, "holdout": 6, "references": 20}
    totals = {"sites": 1, "migrated": 1, "cases": 20, "tokens_in": 900}
    rows = [
        ("stage", {"stage": "QUEUED", "config": CONFIG}, None),
        ("stage", {"stage": "CLONED"}, None),
        ("stage", {"stage": "SCANNED"}, None),
        ("site", site, "cs_01"),
        ("stage", {"stage": "BASELINED"}, None),
        ("case_summary", summary, "cs_01"),
        ("stage", {"stage": "EVALUATING"}, None),
        ("tier_result", result, "cs_01"),
        ("stage", {"stage": "VERDICTS"}, None),
        ("verdict", verdict, "cs_01"),
        ("pr", {"url": "https://github.com/acme/tiny/pull/7", "title": "Offramp"}, None),
        ("stage", {"stage": "DONE"}, None),
        ("done", {"totals": totals}, None),
    ]
    stage, out = Stage.QUEUED, []
    for i, (kind, payload, site_id) in enumerate(rows):
        stage = Stage(payload["stage"]) if kind == "stage" else stage
        out.append(
            RunEvent(
                run_id=RECORDED_ID,
                seq=i,
                ts=t0 + timedelta(milliseconds=gap_ms * i),
                stage=stage,
                type=kind,
                call_site_id=site_id,
                payload=payload,
            )
        )
    return out


def write_jsonl(path, events):
    path.write_text("".join(e.model_dump_json() + "\n" for e in events))
    return path


def sse(text: str) -> list[dict]:
    """Parse a text/event-stream body into [{"id", "event", "data"}]."""
    out = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if fields:
            out.append({**fields, "data": json.loads(fields["data"])})
    return out


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    monkeypatch.delenv("OFFRAMP_RATE_LIMIT_PER_MIN", raising=False)
    monkeypatch.delenv("OFFRAMP_WEB_ORIGIN", raising=False)
    monkeypatch.delenv("OFFRAMP_ALLOW_LIVE_RUNS", raising=False)
    clients = []

    def make(**kw):
        kw.setdefault("replay_file", tmp_path / "missing.jsonl")
        client = TestClient(create_app(tmp_path / "data", **kw))
        client.__enter__()  # runs the lifespan: starts the worker thread
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def test_health_says_whether_token_factory_is_configured(make_client, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_API_KEY_FALLBACK", raising=False)
    client = make_client()
    assert client.get("/api/health").json() == {"ok": True, "tf_configured": False}
    monkeypatch.setenv("NEBIUS_API_KEY_FALLBACK", "k")
    assert client.get("/api/health").json() == {"ok": True, "tf_configured": True}


def test_replay_without_a_recording_says_there_is_none_yet(make_client):
    r = make_client().post("/api/runs?replay=demo")
    assert r.status_code == 404 and r.json()["detail"].startswith("no recorded run yet")


@pytest.mark.parametrize("body", ["fixture-flag", "fixture-id"])
def test_fixture_data_never_plays_as_the_demo(make_client, tmp_path, body):
    events = recorded_events()
    path = tmp_path / "demo.json"
    if body == "fixture-flag":
        path.write_text(json.dumps({"fixture": True, "events": [e.model_dump(mode="json") for e in events]}))
    else:
        write_jsonl(path, [e.model_copy(update={"run_id": "fixture-dev"}) for e in events])
    r = make_client(replay_file=path).post("/api/runs?replay=demo")
    assert r.status_code == 404 and "no recorded run yet" in r.json()["detail"]


def test_a_replay_streams_the_recorded_run_then_serves_its_snapshot_and_report(make_client, tmp_path):
    client = make_client(replay_file=write_jsonl(tmp_path / "demo.jsonl", recorded_events()))
    r = client.post("/api/runs?replay=demo")
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert run_id.startswith("replay-")

    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        events = sse(stream.read().decode())
    recorded = recorded_events()
    assert [int(e["id"]) for e in events] == [e.seq for e in recorded]
    assert [e["event"] for e in events] == [e.type for e in recorded]
    assert all(e["data"]["run_id"] == run_id for e in events)  # re-stamped; payloads as recorded
    assert [e["data"]["payload"] for e in events] == [e.payload for e in recorded]
    assert events[-1]["event"] == "done"  # the stream ended after it

    snap = client.get(f"/api/runs/{run_id}").json()
    assert snap["run_id"] == run_id and snap["stage"] == "DONE"
    assert snap["config"]["threshold"] == 0.9 and snap["config"]["repo_url"] == CONFIG["repo_url"]
    assert [s["id"] for s in snap["sites"]] == ["cs_01"]
    assert snap["verdicts"][0]["decision"] == "MIGRATE"
    assert snap["case_summaries"][0]["holdout"] == 6
    assert snap["tier_results"][0]["split"] == "holdout"
    assert snap["totals"]["migrated"] == 1 and snap["totals"]["tokens_in"] == 900  # the done event's totals
    assert snap["artifacts"] == {
        "report_url": f"/api/runs/{run_id}/report",
        "pr_url": "https://github.com/acme/tiny/pull/7",
    }
    report = client.get(f"/api/runs/{run_id}/report")
    assert "attachment" in report.headers["content-disposition"]
    assert report.json()["snapshot"] == snap and report.json()["pr_body_markdown"] is None
    with Session(client.app.state.engine) as s:
        row = s.get(db.Run, run_id)
        assert row.stage == "DONE" and row.repo_url == CONFIG["repo_url"]


def finish(client, run_id):
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        stream.read()


def test_a_replay_carries_the_recorded_runs_pr_body(make_client, tmp_path):
    replay = write_jsonl(tmp_path / "demo-events.jsonl", recorded_events())
    api.pr_body_file(replay).write_text("## Summary\n\nrecorded body\n")
    client = make_client(replay_file=replay)
    run_id = client.post("/api/runs?replay=demo").json()["run_id"]
    finish(client, run_id)
    assert (
        client.get(f"/api/runs/{run_id}/report").json()["pr_body_markdown"] == "## Summary\n\nrecorded body\n"
    )


def test_only_the_newest_replay_runs_are_kept(make_client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "REPLAY_KEEP", 2)
    client = make_client(replay_file=write_jsonl(tmp_path / "demo.jsonl", recorded_events()))
    ids = []
    for _ in range(3):
        ids.append(client.post("/api/runs?replay=demo").json()["run_id"])
        finish(client, ids[-1])
    assert ids == sorted(ids)
    runs = tmp_path / "data" / "runs"
    assert sorted(p.name for p in runs.iterdir()) == ids[1:]  # the oldest went: directory, memory, rows
    assert client.get(f"/api/runs/{ids[0]}").status_code == 404
    with Session(client.app.state.engine) as s:
        assert s.get(db.Run, ids[0]) is None and s.get(db.Run, ids[2]) is not None


def test_the_event_stream_resumes_after_last_event_id(make_client, tmp_path):
    client = make_client(replay_file=write_jsonl(tmp_path / "demo.jsonl", recorded_events()))
    run_id = client.post("/api/runs?replay=demo").json()["run_id"]
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        stream.read()  # the replay has finished
    with client.stream("GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "9"}) as stream:
        ids = [int(e["id"]) for e in sse(stream.read().decode())]
    assert ids == [10, 11, 12]


def test_replay_timing_is_four_times_faster_with_gaps_capped_first():
    t0 = datetime(2026, 10, 19, tzinfo=UTC)
    at = [0, 1, 11, 11.2, 11.1]  # seconds; the last one is out of order
    events = [
        RunEvent(run_id="r", seq=i, ts=t0 + timedelta(seconds=s), stage=Stage.QUEUED, type="log")
        for i, s in enumerate(at)
    ]
    assert replay_delays(events) == pytest.approx([0, 0.25, 0.375, 0.05, 0])
    assert replay_delays([]) == []


def test_rate_limit_answers_429_past_the_limit(make_client, monkeypatch, tmp_path):
    monkeypatch.setenv("OFFRAMP_RATE_LIMIT_PER_MIN", "2")
    client = make_client(replay_file=write_jsonl(tmp_path / "demo.jsonl", recorded_events()))
    assert [client.post("/api/runs?replay=demo").status_code for _ in range(3)] == [200, 200, 429]
    assert client.get("/api/health").status_code == 200  # only POST /api/runs is limited


def test_rate_limit_window_slides():
    limit = RateLimit(2)
    assert [limit.allow("a", now=t) for t in (0, 1, 2)] == [True, True, False]
    assert limit.allow("b", now=2)  # per address
    assert limit.allow("a", now=60.5) and not limit.allow("a", now=60.6)


def test_cors_allows_only_the_web_origin(make_client, monkeypatch):
    monkeypatch.setenv("OFFRAMP_WEB_ORIGIN", "https://offramp.example, http://localhost:3000")
    client = make_client()
    ask = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    ok = client.options("/api/runs", headers={"Origin": "https://offramp.example", **ask})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "https://offramp.example"
    other = client.options("/api/runs", headers={"Origin": "https://elsewhere.example", **ask})
    assert "access-control-allow-origin" not in other.headers
    get = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert get.headers["access-control-allow-origin"] == "http://localhost:3000"


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"repo_url": "/etc"}, "https://"),
        ({"repo_url": "file:///etc/passwd"}, "https://"),
        ({"repo_url": "https://user:ghp_secret@github.com/acme/tiny"}, "https://"),  # no credentials in URLs
        ({"threshold": 2}, "thresholds"),
        ({"max_iterations": 9}, "max_iterations"),
        ({"tiers": []}, "tier"),
    ],
)
def test_bad_configs_are_refused(make_client, change, problem):
    r = make_client().post("/api/runs", json={**CONFIG, **change})
    assert r.status_code == 422 and problem in r.json()["detail"]


@pytest.mark.parametrize("live", [True, False])
def test_runs_are_refused_unless_the_server_opts_in(make_client, monkeypatch, tmp_path, live):
    """Any run, cache-only too, clones the repo it is given, runs its code in the sandbox and can open a PR
    with the server's GitHub token: the API starts one only when the operator says so
    (OFFRAMP_ALLOW_LIVE_RUNS=1). Replays stay open."""
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    started = []
    client = make_client(execute=started.append)
    r = client.post("/api/runs", json={**CONFIG, "live": live})
    assert r.status_code == 403 and "OFFRAMP_ALLOW_LIVE_RUNS=1" in r.json()["detail"]
    monkeypatch.setenv("OFFRAMP_ALLOW_LIVE_RUNS", "true")  # only the exact value 1 opens it
    assert client.post("/api/runs", json={**CONFIG, "live": live}).status_code == 403
    assert started == [] and not client.app.state.logs and not list((tmp_path / "data" / "runs").glob("*"))
    monkeypatch.setenv("OFFRAMP_ALLOW_LIVE_RUNS", "1")
    assert client.post("/api/runs", json={**CONFIG, "live": live}).status_code == 200


def test_a_live_run_needs_token_factory(make_client, monkeypatch):
    monkeypatch.setenv("OFFRAMP_ALLOW_LIVE_RUNS", "1")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_API_KEY_FALLBACK", raising=False)
    r = make_client().post("/api/runs", json={**CONFIG, "live": True})
    assert r.status_code == 503 and "replay" in r.json()["detail"]
    assert make_client().post("/api/runs").status_code == 422  # no body, no replay


def test_a_queued_run_executes_on_the_worker_and_streams_live(make_client, monkeypatch):
    monkeypatch.setenv("OFFRAMP_ALLOW_LIVE_RUNS", "1")
    seen = {}
    release = threading.Event()

    def execute(log: pipeline.RunLog) -> Stage:
        seen["thread"] = threading.current_thread().name
        seen["config"] = log.config
        log.emit("stage", {"stage": "CLONED", "commit_sha": "abc1234"})
        release.wait(5)  # the client is already streaming: these arrive live
        log.emit("log", {"level": "info", "message": "working"})
        log.emit("stage", {"stage": "DONE"})
        log.emit("done", {"totals": {"sites": 0}})
        return Stage.DONE

    client = make_client(execute=execute)
    r = client.post("/api/runs", json=CONFIG)  # live=False: replays the cache, needs no key
    run_id = r.json()["run_id"]
    threading.Timer(0.5, release.set).start()
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        events = sse(stream.read().decode())
    assert [e["event"] for e in events] == ["stage", "stage", "log", "stage", "done"]
    assert events[0]["data"]["payload"]["config"]["repo_url"] == CONFIG["repo_url"]
    assert seen == {"thread": "offramp-runs", "config": RunConfig(**CONFIG)}
    with Session(client.app.state.engine) as s:
        row = s.get(db.Run, run_id)
        assert (row.stage, row.commit_sha) == ("DONE", "abc1234")
    assert (client.app.state.logs[run_id].dir / "events.jsonl").exists()


def test_a_crash_outside_the_pipeline_still_ends_the_stream_with_an_error(make_client, monkeypatch):
    monkeypatch.setenv("OFFRAMP_ALLOW_LIVE_RUNS", "1")

    def execute(log):
        raise RuntimeError("worker fell over")

    client = make_client(execute=execute)
    run_id = client.post("/api/runs", json=CONFIG).json()["run_id"]
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        events = sse(stream.read().decode())
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["payload"] == {"stage": "QUEUED", "reason": "RuntimeError: worker fell over"}
    assert client.get(f"/api/runs/{run_id}").json()["stage"] == "FAILED"


def test_runs_from_an_earlier_process_are_served_from_disk(make_client, tmp_path):
    root = tmp_path / "data"
    log = pipeline.RunLog(root / "runs" / RECORDED_ID, RECORDED_ID)
    for e in recorded_events()[:7]:  # it stopped mid-run (the process died): not active here
        log.add(e)
    client = make_client()
    assert client.get(f"/api/runs/{RECORDED_ID}").json()["stage"] == "EVALUATING"
    with client.stream("GET", f"/api/runs/{RECORDED_ID}/events") as stream:
        assert len(sse(stream.read().decode())) == 7  # ends: nothing in this process will add more


def test_events_another_process_appends_to_a_run_are_served(make_client, tmp_path):
    """Review M3 (task 29): `offramp pr` appends the PR's events to a finished run in another process; an
    API that had already loaded the run serves them too."""
    root = tmp_path / "data"
    log = pipeline.RunLog(root / "runs" / RECORDED_ID, RECORDED_ID)
    for e in recorded_events():
        if e.type != "pr":  # a --no-pr run
            log.add(e)
    client = make_client()
    assert "pr_url" not in client.get(f"/api/runs/{RECORDED_ID}").json()["artifacts"]
    other = pipeline.RunLog(root / "runs" / RECORDED_ID, RECORDED_ID)  # the `offramp pr` process
    pipeline.announce_pr(other, "https://github.com/acme/tiny/pull/9", "T")
    assert client.get(f"/api/runs/{RECORDED_ID}").json()["artifacts"]["pr_url"].endswith("/pull/9")
    with client.stream("GET", f"/api/runs/{RECORDED_ID}/events") as stream:
        assert [e["event"] for e in sse(stream.read().decode())][-2:] == ["pr", "stage"]


@pytest.mark.parametrize("run_id", ["nope", "..%2F..%2Fetc", "a" * 70])
def test_unknown_runs_are_404(make_client, run_id):
    client = make_client()
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.get(f"/api/runs/{run_id}/events").status_code == 404
    assert client.get(f"/api/runs/{run_id}/report").status_code == 404
