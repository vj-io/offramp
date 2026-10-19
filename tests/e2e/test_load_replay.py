"""The load-test script (tests/e2e/load_replay.py) against an in-process API: 5 concurrent replay runs of a
test recording (60 events, 100 ms apart, so each replay streams for about 1.5 s), API p95 under 500 ms."""

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

from offramp.schemas import RunEvent, Stage

spec = importlib.util.spec_from_file_location("load_replay", Path(__file__).with_name("load_replay.py"))
load_replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_replay)


def recording(path: Path, n: int = 60) -> Path:
    t0 = datetime(2026, 10, 19, tzinfo=UTC)
    config = {"repo_url": "https://github.com/acme/tiny"}
    rows = [("stage", {"stage": "QUEUED", "config": config})]
    rows += [("log", {"level": "info", "message": f"step {i}"}) for i in range(n - 3)]
    rows += [("stage", {"stage": "DONE"}), ("done", {"totals": {"sites": 0}})]
    events = [
        RunEvent(
            run_id="20261019-000000-load01",
            seq=i,
            ts=t0 + timedelta(milliseconds=100 * i),
            stage=Stage(p["stage"]) if kind == "stage" else Stage.QUEUED,
            type=kind,
            payload=p,
        )
        for i, (kind, p) in enumerate(rows)
    ]
    path.write_text("".join(e.model_dump_json() + "\n" for e in events))
    return path


def test_five_concurrent_replays_keep_api_p95_under_500_ms(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OFFRAMP_RATE_LIMIT_PER_MIN", raising=False)
    assert load_replay.main(["--replay", str(recording(tmp_path / "demo.jsonl")), "--runs", "5"]) == 0
    out = capsys.readouterr().out
    assert "5/5 streams ended with `done`" in out


def test_without_a_recording_it_says_so(tmp_path, capsys):
    assert load_replay.main(["--replay", str(tmp_path / "none.jsonl")]) == 2
    assert "no recorded run yet" in capsys.readouterr().err
