"""`offramp` CLI: scan, export-replay and the run/resume argument checks (the pipeline itself is covered by
test_pipeline_flow.py; `serve` by the load test)."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from offramp import cli, pipeline
from offramp.api.app import load_recording
from offramp.schemas import RunConfig, RunEvent, Stage

APP = """from openai import OpenAI

client = OpenAI()


def triage(ticket):
    return client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": ticket}])
"""


def test_scan_prints_the_call_sites_without_model_calls(tmp_path, capsys):
    (tmp_path / "app.py").write_text(APP)
    assert cli.main(["scan", str(tmp_path)]) == 0
    scan = json.loads(capsys.readouterr().out)
    assert [(s["id"], s["file"], s["source_model"]) for s in scan["sites"]] == [
        ("cs_01", "app.py", "gpt-4o-mini")
    ]


def finished_run(root, run_id="20261019-120000-abc123", done=True):
    log = pipeline.new_run(RunConfig(repo_url="https://github.com/acme/tiny"), root, run_id=run_id)
    log.emit("stage", {"stage": "DONE"})
    if done:
        log.emit("done", {"totals": {}})
    # recorded gaps: 2 s and 0.4 s; at 4x speed with the 1.5 s cap the replay plays in 0.375 + 0.1 s
    t0 = datetime(2026, 10, 19, tzinfo=UTC)
    for event, gap in zip(log.events, (0, 2, 2.4), strict=False):
        event.ts = t0 + timedelta(seconds=gap)
    log.file.write_text("".join(e.model_dump_json() + "\n" for e in log.events))
    return log


def test_export_replay_writes_both_replay_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    log = finished_run(tmp_path / "data")
    api, web = tmp_path / "api" / "demo-events.jsonl", tmp_path / "web" / "demo-events.json"

    assert cli.main(["export-replay", log.run_id, "--api-out", str(api), "--web-out", str(web)]) == 0
    assert load_recording(api) == log.events  # what POST /api/runs?replay=demo plays
    assert [RunEvent.model_validate(e) for e in json.loads(web.read_text())] == log.events  # a JSON array
    assert "3 events" in capsys.readouterr().out


def test_export_replay_refuses_an_unfinished_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    log = finished_run(tmp_path / "data", done=False)
    api = tmp_path / "out.jsonl"
    assert cli.main(["export-replay", log.run_id, "--api-out", str(api), "--web-out", str(api)]) == 1
    assert "has not finished" in capsys.readouterr().out and not api.exists()


@pytest.mark.parametrize("argv", [["run"], ["run", "--resume", "20261019-000000-nope00"]])
def test_run_needs_a_repo_or_a_known_run(tmp_path, monkeypatch, argv):
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path))
    assert cli.main(argv) == 2


def test_resume_continues_the_stored_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path))
    log = finished_run(tmp_path)  # finished: resuming it is a no-op that reports DONE
    assert cli.main(["run", "--resume", log.run_id]) == 0
    assert f"run {log.run_id}: {Stage.DONE.value}" in capsys.readouterr().out
