"""`offramp` CLI: scan, export-replay and the run/resume argument checks (the pipeline itself is covered by
test_pipeline_flow.py; `serve` by the load test)."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from offramp import cli, pipeline
from offramp.api.app import load_recording, pr_body_file
from offramp.schemas import RunConfig, RunEvent, Stage

APP = """from openai import OpenAI

client = OpenAI()


def triage(ticket):
    return client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": ticket}])
"""


def test_scan_prints_the_call_sites_without_model_calls(tmp_path, capsys):
    (tmp_path / "app.py").write_text(APP)
    assert cli.main(["scan", str(tmp_path), "--json"]) == 0
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
    recording = json.loads(web.read_text())
    assert [RunEvent.model_validate(e) for e in recording["events"]] == log.events
    assert recording["diffs"] == {}  # this run has no patch.json
    assert "3 events" in capsys.readouterr().out
    assert not pr_body_file(api).exists()  # this run rendered no PR body

    (log.dir / "pr.json").write_text(json.dumps({"title": "t", "body": "## Summary\n", "url": None}))
    assert cli.main(["export-replay", log.run_id, "--api-out", str(api), "--web-out", str(web)]) == 0
    assert pr_body_file(api).read_text() == "## Summary\n"  # the API's replay serves it in its report


def test_export_replay_adds_each_migrated_sites_diff_scrubbed(tmp_path, monkeypatch):
    """The report's "What changed" panel: the diff each migrated site ships, from the run's own patch.json.
    A kept site's comment diff and a manual-review site never reach the web; secrets are scrubbed."""
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NEBIUS_API_KEY", "tf-secret-0123456789")
    log = finished_run(tmp_path / "data")
    shipped = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-x = 'tf-secret-0123456789'\n+x = 1\n"
    patch = {
        "files": {"app.py": "x = 1\n"},
        "diffs": {"cs_01": shipped, "cs_02": "--- a/kept.py\n+++ b/kept.py\n", "cs_03": "--- a/x.py\n"},
        "merged_diff": "--- a/cli.py\n+++ b/cli.py\n@@ -1 +0,0 @@\n-from anthropic import Anthropic\n",
        "verdicts": [
            {"site_id": "cs_01", "decision": "MIGRATE"},
            {"site_id": "cs_02", "decision": "KEEP_CLOSED"},
            {"site_id": "cs_03", "decision": "MANUAL_REVIEW"},
        ],
    }
    (log.dir / "patch.json").write_text(json.dumps(patch))
    api, web = tmp_path / "api.jsonl", tmp_path / "web.json"

    assert cli.main(["export-replay", log.run_id, "--api-out", str(api), "--web-out", str(web)]) == 0
    assert json.loads(web.read_text())["diffs"] == {
        "cs_01": shipped.replace("tf-secret-0123456789", "***"),
        "merged": patch["merged_diff"],  # what only merging the sites changed (Task 26 review M4)
    }


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


def test_run_refuses_credentials_in_the_repo_url(tmp_path, monkeypatch, capsys):
    """The run log masks credentials, so a credentialed URL would be cloned as https://***@host."""
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path))
    assert cli.main(["run", "https://user:tok@github.com/o/r", "--cache-only"]) == 2
    assert "GITHUB_TOKEN" in capsys.readouterr().err and not (tmp_path / "runs").exists()


@pytest.mark.parametrize(("argv", "live"), [([], True), (["--cache-only"], False)])
def test_resume_honours_cache_only(tmp_path, monkeypatch, argv, live):
    """`--resume <id> --cache-only` on a run started live replays the cache: the Token Factory client and the
    pipeline's own live switches (Tavily price lookup, the eval matrix) are all off."""
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path))
    log = pipeline.new_run(RunConfig(repo_url="https://github.com/acme/tiny", live=True), tmp_path)
    seen = {}

    class Resumed(pipeline.Pipeline):
        def execute(self):
            seen.update(client=self.client.live, config=self.config.live)
            return Stage.DONE

    monkeypatch.setattr(pipeline, "Pipeline", Resumed)

    assert cli.main(["run", "--resume", log.run_id, *argv]) == 0
    assert seen == {"client": live, "config": live}


def test_resume_continues_the_stored_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path))
    log = finished_run(tmp_path)  # finished: resuming it is a no-op that reports DONE
    assert cli.main(["run", "--resume", log.run_id]) == 0
    assert f"run {log.run_id}: {Stage.DONE.value}" in capsys.readouterr().out
