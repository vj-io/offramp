"""Serverless eval job (PRD §6.5, §12.4): the local worker and the Nebius job path give the same results.

Nothing here is mocked inside the code under test:
- Token Factory is a stdlib HTTP server (`FakeTF`) the client reaches through NEBIUS_BASE_URL, so the job's
  subprocess talks to it as the container would talk to the real endpoint;
- the `nebius` CLI is a script on PATH (`FAKE_NEBIUS`) that does what the real one does for the verbs
  submit.py uses: `ai job create` runs the container entrypoint (`python -m
  offramp.evals.serverless_job.main`) with the --env pairs and --inject-file parts it was given, and
  `get`/`logs`/`cancel` read the job it recorded. What it prints is only as faithful as
  docs.nebius.com/cli/reference/ai/job/*; whether a real job accepts these flags needs credentials.
Latencies are measured per live call, so a live run and a job run differ in p50/p95. The job path is
therefore compared with a local run over the same cache (replay), which is how a pipeline re-run behaves.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from offramp.evals.serverless_job import submit, transport
from offramp.evals.serverless_job.matrix import Matrix, MatrixResults, dumps
from offramp.evals.serverless_job.submit import JobError, JobUnavailable, redact, run_matrix_job
from offramp.schemas import (
    TIER_ORDER,
    Adapter,
    AdapterSpec,
    BaselineResult,
    CallSite,
    Contract,
    EvalCase,
    ModelOutput,
    Tier,
    TierResult,
)
from offramp.tf import get_tier

REPO = Path(__file__).resolve().parents[2]
SECRET = "nbk-SECRET-0123456789"
LIGHTNING = get_tier(Tier.LIGHTNING).model_id

FAKE_NEBIUS = """#!{python}
import json, os, shlex, shutil, subprocess, sys
from pathlib import Path

state, argv = Path(os.environ["FAKE_NEBIUS_STATE"]), sys.argv[1:]
with (state / "calls.jsonl").open("a") as f:
    f.write(json.dumps(argv) + "\\n")
PREFIX = "2026-10-19T12:00:00Z stdout F "  # the real log format is undocumented: be tolerant of a prefix


def flag_values(name):
    return [argv[i + 1] for i, a in enumerate(argv) if a == name]


verb, job = argv[2], (argv[3] if len(argv) > 3 and not argv[3].startswith("--") else "job-1")
if verb == "create":
    if os.environ.get("FAKE_NEBIUS_CREATE_FAILS"):
        sys.exit("error: invalid request: " + " ".join(argv))  # a CLI that echoes its arguments
    root = state / "jobs" / "job-1"
    mount = root / "fs"
    env = {{k: v for k, v in os.environ.items() if not k.startswith(("NEBIUS_", "OFFRAMP_", "FAKE_"))}}
    env["PYTHONPATH"] = os.environ["FAKE_NEBIUS_REPO"]
    for pair in flag_values("--env"):
        k, _, v = pair.partition("=")
        env[k] = v
    for pair in flag_values("--inject-file"):
        local, _, container = pair.partition(":")
        target = mount / container.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, target)
    args = flag_values("--args")[0].replace("/offramp/input", str(mount / "offramp" / "input"))
    command = flag_values("--container-command")[0]
    command = sys.executable if command == "python" else command
    done = subprocess.run([command, *shlex.split(args)], env=env, cwd=os.environ["FAKE_NEBIUS_REPO"],
                          capture_output=True, text=True)
    logs = [PREFIX + line for line in (done.stdout + done.stderr).splitlines()]
    drop = os.environ.get("FAKE_NEBIUS_DROP_CHUNK")
    if drop:  # logs that lost a line: the chunk sent as "<drop>/<n>"
        logs = [line for line in logs if " %s/" % drop not in line]
    (root / "logs.txt").write_text("\\n".join(logs) + "\\n")
    final = os.environ.get("FAKE_NEBIUS_FINAL_STATE") or ("COMPLETED" if done.returncode == 0 else "FAILED")
    (root / "state").write_text(final)
    print(json.dumps({{"id": "operation-1", "resource_id": "job-1"}}))
elif verb == "get":
    root = state / "jobs" / job
    polls = state / "polls"
    n = int(polls.read_text()) + 1 if polls.exists() else 1
    polls.write_text(str(n))
    running = n <= int(os.environ.get("FAKE_NEBIUS_RUNNING_POLLS", "0"))
    status = {{"state": "RUNNING" if running else (root / "state").read_text(),
               "state_details": {{"message": "scripted state"}}}}
    print(json.dumps({{"metadata": {{"id": job}}, "status": status}}))
elif verb == "logs":
    print((state / "jobs" / job / "logs.txt").read_text(), end="")
elif verb == "cancel":
    (state / "cancelled").write_text(job)
else:
    sys.exit("fake nebius: unsupported " + " ".join(argv))
"""


class FakeTF:
    """Token Factory over real HTTP. Lightning always answers "bug"; the others answer the case's label."""

    def __init__(self, reject_models=()):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"model": body["model"], "auth": self.headers.get("Authorization")})
                if body["model"] in reject_models:
                    status, payload = 401, {"error": {"message": "invalid api key", "type": "auth"}}
                else:
                    text = (
                        "bug" if body["model"] == LIGHTNING else body["messages"][-1]["content"].split(":")[0]
                    )
                    status, payload = (
                        200,
                        {
                            "id": "chatcmpl-1",
                            "object": "chat.completion",
                            "created": 1,
                            "model": body["model"],
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": text},
                                }
                            ],
                            "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
                        },
                    )
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    for name in ("NEBIUS_API_KEY_FALLBACK", *(f"OFFRAMP_MODEL_{t.name}" for t in TIER_ORDER)):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NEBIUS_API_KEY", SECRET)
    monkeypatch.delenv("NEBIUS_PROJECT_ID", raising=False)
    monkeypatch.delenv("OFFRAMP_EVAL_IMAGE", raising=False)


@pytest.fixture
def tf(monkeypatch):
    server = FakeTF()
    monkeypatch.setenv("NEBIUS_BASE_URL", server.url)
    yield server
    server.stop()


class FakeCli:
    """What the fake `nebius` recorded: the argv of every call, and the job it ran."""

    def __init__(self, root: Path):
        self.root = root

    def calls(self) -> list[list[str]]:
        return [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]

    @property
    def logs(self) -> str:
        return (self.root / "jobs" / "job-1" / "logs.txt").read_text()


@pytest.fixture
def nebius(tmp_path, monkeypatch):
    """The fake CLI on PATH, with the project and image the submit path needs."""
    bin_dir, state = tmp_path / "bin", tmp_path / "nebius-state"
    bin_dir.mkdir()
    state.mkdir()
    script = bin_dir / "nebius"
    script.write_text(FAKE_NEBIUS.format(python=sys.executable))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_NEBIUS_STATE", str(state))
    monkeypatch.setenv("FAKE_NEBIUS_REPO", str(REPO))
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "project-e00test")
    monkeypatch.setenv("OFFRAMP_EVAL_IMAGE", "cr.eu-north1.nebius.cloud/e00test/offramp-eval:test")
    return FakeCli(state)


SITE_LABELS = {"cs_01": ["billing", "bug"], "cs_02": ["spam", "ham"]}


def make_matrix(**overrides) -> Matrix:
    sites, cases, refs, adapters = [], [], [], {}
    for site_id, labels in SITE_LABELS.items():
        sites.append(
            CallSite(
                id=site_id,
                file="app/x.py",
                lines=[1, 9],
                sdk="openai",
                call="client.chat.completions.create",
                prompt_source="literal",
                contract=Contract(task_type="classification", output="label", labels=labels),
            )
        )
        adapters[site_id] = {
            t: Adapter(site_id=site_id, version=1, spec=AdapterSpec(max_tokens=16)) for t in TIER_ORDER
        }
        for n in range(6):
            label = labels[n % 2]
            case_id = f"{site_id}:{n:03d}"
            cases.append(
                EvalCase(
                    id=case_id,
                    site_id=site_id,
                    expected_kind="label",
                    split="dev" if n < 3 else "holdout",
                    input_messages=[
                        {"role": "system", "content": "Classify the ticket."},
                        {"role": "user", "content": f"{label}: message {n}"},
                    ],
                )
            )
            refs.append(
                BaselineResult(
                    case_id=case_id,
                    model="reference:seed",
                    output=ModelOutput(text=label),
                    recorded_at=datetime(2026, 10, 19, tzinfo=UTC),
                )
            )
    return Matrix(sites=sites, adapters=adapters, cases=cases, references=refs, **overrides)


def local(matrix, cache_dir, **kw) -> MatrixResults:
    return run_matrix_job(matrix, mode="local", cache_dir=cache_dir, **kw)


def job(matrix, cache_dir, **kw) -> MatrixResults:
    return run_matrix_job(matrix, mode="job", cache_dir=cache_dir, poll_s=0.05, **kw)


def cache_keys(directory: Path) -> set[str]:
    return {p.stem for p in directory.glob("*.json")}


# ------------------------------------------------------------------------------------------ the matrix


def test_local_run_covers_every_cell_sorted(tf, tmp_path):
    results = local(make_matrix(), tmp_path / "cache").results
    cells = [(e["site_id"], e["tier"], e["split"]) for e in results]
    assert len(cells) == 2 * 3 * 2
    assert cells == sorted(cells, key=lambda c: (c[0], TIER_ORDER.index(Tier(c[1])), c[2]))
    assert {e["kind"] for e in results} == {"tier_result"}
    by_tier = {e["tier"]: TierResult.model_validate(e["result"]).passed for e in results}
    assert by_tier == {"lightning": False, "super": True, "ultra": True}
    assert len(tf.requests) == 2 * 3 * 6  # each case once per tier, both splits


def test_matrix_needs_an_adapter_for_every_site_and_tier():
    matrix = make_matrix()
    del matrix.adapters["cs_02"][Tier.ULTRA]
    with pytest.raises(ValidationError, match=r"cs_02/ultra"):
        Matrix.model_validate_json(matrix.model_dump_json())


# ------------------------------------------------------------------------- local vs Serverless Job path


def test_job_results_equal_local_byte_for_byte(tf, nebius, tmp_path, monkeypatch):
    matrix = make_matrix()
    local_run = local(matrix, tmp_path / "local-cache")
    shutil.copytree(tmp_path / "local-cache", tmp_path / "job-cache")
    served = len(tf.requests)
    monkeypatch.setenv(
        "NEBIUS_BASE_URL", "http://127.0.0.1:9/v1"
    )  # nothing listens: any cache miss is an error

    job_run = job(matrix, tmp_path / "job-cache")

    assert dumps(job_run.results) == dumps(local_run.results)
    assert dumps(sorted(job_run.results, key=json.dumps)) == dumps(sorted(local_run.results, key=json.dumps))
    assert {e["kind"] for e in job_run.results} == {"tier_result"}
    assert job_run.cache == {}  # everything replayed from the shipped cache
    assert len(tf.requests) == served
    verbs = [c[2] for c in nebius.calls()]
    assert verbs[0] == "create" and "get" in verbs and "logs" in verbs


def test_job_writes_new_cache_entries_back_like_a_local_run(tf, nebius, tmp_path):
    matrix = make_matrix()
    local(matrix, tmp_path / "local-cache")
    local_keys, served = cache_keys(tmp_path / "local-cache"), len(tf.requests)

    job_run = job(matrix, tmp_path / "job-cache")  # empty cache: the job calls Token Factory live

    assert len(tf.requests) == 2 * served
    assert cache_keys(tmp_path / "job-cache") == local_keys == set(job_run.cache)
    replay = local(matrix.model_copy(update={"live": False}), tmp_path / "job-cache")
    assert dumps(replay.results) == dumps(job_run.results)  # the filled cache replays to the same numbers
    assert len(tf.requests) == 2 * served


def test_api_key_is_passed_to_the_job_but_never_logged(tf, nebius, tmp_path, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    job(make_matrix(), tmp_path / "cache")

    create = next(c for c in nebius.calls() if c[2] == "create")
    assert f"NEBIUS_API_KEY={SECRET}" in create  # the job's calls need it, as a --env value
    assert {a for a in create if a.startswith("NEBIUS_BASE_URL=")}  # tier/endpoint config travels with it
    seen = capsys.readouterr()
    assert SECRET not in caplog.text + seen.out + seen.err
    assert "NEBIUS_API_KEY=***" in caplog.text  # the command line is logged, redacted
    assert SECRET not in nebius.logs  # nor does it reach the job's logs


def test_replay_job_gets_no_key(tf, nebius, tmp_path):
    matrix = make_matrix()
    local(matrix, tmp_path / "cache")
    job(matrix.model_copy(update={"live": False}), tmp_path / "cache")
    create = next(c for c in nebius.calls() if c[2] == "create")
    assert not [a for a in create if "NEBIUS_API_KEY" in a]


def test_a_failing_cli_never_leaks_the_key_in_the_error(tf, nebius, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_NEBIUS_CREATE_FAILS", "1")
    with pytest.raises(JobError) as e:
        job(make_matrix(), tmp_path / "cache")
    assert "invalid request" in str(e.value) and SECRET not in str(e.value)


def test_redact_hides_secret_env_values_only():
    argv = [
        "ai",
        "job",
        "create",
        "--env",
        f"NEBIUS_API_KEY={SECRET}",
        "--env",
        "OFFRAMP_MODEL_SUPER=m/x",
        "--registry-password",
        "hunter2",
    ]
    shown = redact(argv)
    assert SECRET not in shown and "hunter2" not in shown
    assert "NEBIUS_API_KEY=***" in shown and "OFFRAMP_MODEL_SUPER=m/x" in shown


def test_job_is_polled_until_it_finishes(tf, nebius, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_NEBIUS_RUNNING_POLLS", "2")
    job(make_matrix(), tmp_path / "cache")
    assert [c[2] for c in nebius.calls()].count("get") == 3


def test_results_larger_than_one_log_line_come_back_in_chunks(tf, nebius, tmp_path, monkeypatch):
    monkeypatch.setattr(submit, "CHUNK_CHARS", 300)
    matrix = make_matrix()
    run = job(matrix, tmp_path / "cache")
    assert nebius.logs.count(transport.RESULTS_TAG) > 3
    assert run.results and {e["kind"] for e in run.results} == {"tier_result"}


def test_truncated_logs_are_an_error_not_partial_results(tf, nebius, tmp_path, monkeypatch):
    monkeypatch.setattr(submit, "CHUNK_CHARS", 300)
    monkeypatch.setenv("FAKE_NEBIUS_DROP_CHUNK", "2")
    with pytest.raises(JobError, match="missing"):
        job(make_matrix(), tmp_path / "cache")


def test_chunk_encoding_roundtrips_with_noise_and_any_order():
    out = MatrixResults(
        results=[{"k": i, "text": hashlib.sha256(bytes([i])).hexdigest() * 2} for i in range(40)],
        cache={"a" * 64: {"v": 1}},
        budget={"tokens_in": 3},
    )
    lines = transport.encode_results(out, chunk_chars=120)
    assert len(lines) > 5 and all(len(line) < 200 for line in lines)
    noisy = ["boot", *(f"2026-10-19T12:00:00Z {line}" for line in reversed(lines)), "exit 0"]
    assert transport.decode_results("\n".join(noisy)) == out
    one = transport.encode_results(out, chunk_chars=10**9)
    assert len(one) == 1 and len(one[0].split()) == 2  # a single line: "OFFRAMP_RESULTS <base64-gzip-json>"
    assert transport.decode_results(one[0]) == out
    with pytest.raises(JobError, match="no OFFRAMP_RESULTS"):
        transport.decode_results("nothing here")


# ---------------------------------------------------------------------------------------- availability


def test_auto_runs_locally_without_the_cli(tf, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # no `nebius` anywhere
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "project-e00test")
    monkeypatch.setenv("OFFRAMP_EVAL_IMAGE", "img:1")
    matrix = make_matrix()
    expected = local(matrix, tmp_path / "a").results
    auto = run_matrix_job(matrix, cache_dir=tmp_path / "a")
    assert dumps(auto.results) == dumps(expected)
    with pytest.raises(JobUnavailable, match="nebius"):
        run_matrix_job(matrix, mode="job", cache_dir=tmp_path / "a")


@pytest.mark.parametrize("missing", ["NEBIUS_PROJECT_ID", "OFFRAMP_EVAL_IMAGE"])
def test_job_needs_project_and_image(tf, nebius, tmp_path, monkeypatch, missing):
    monkeypatch.delenv(missing)
    matrix = make_matrix()
    with pytest.raises(JobUnavailable, match=missing):
        run_matrix_job(matrix, mode="job", cache_dir=tmp_path / "cache")
    assert not (nebius.root / "calls.jsonl").exists()
    assert run_matrix_job(matrix, cache_dir=tmp_path / "cache").results  # auto: local


def test_failed_job_raises_in_job_mode_and_falls_back_in_auto(tf, nebius, tmp_path, monkeypatch):
    matrix = make_matrix()
    expected = local(matrix, tmp_path / "local").results
    shutil.copytree(tmp_path / "local", tmp_path / "cache")
    monkeypatch.setenv("FAKE_NEBIUS_FINAL_STATE", "FAILED")
    with pytest.raises(JobError, match="FAILED"):
        job(matrix, tmp_path / "cache")
    auto = run_matrix_job(matrix, cache_dir=tmp_path / "cache", poll_s=0.05)
    assert dumps(auto.results) == dumps(expected)


def test_too_big_an_input_is_unavailable_not_truncated(monkeypatch):
    monkeypatch.setattr(transport, "PART_BYTES", 10)
    monkeypatch.setattr(transport, "MAX_PARTS", 2)
    with pytest.raises(JobUnavailable, match="too large"):
        transport.split_parts(b"x" * 100)


# ------------------------------------------------------------------- failures are entries, not crashes


def test_run_errors_are_structured_entries_equal_on_both_paths(nebius, tmp_path, monkeypatch):
    matrix = make_matrix()
    # cs_01: no references -> MissingReference; cs_02: an unresolvable response_format -> UnresolvedCallSite
    matrix = matrix.model_copy(
        update={"references": [r for r in matrix.references if r.case_id[:5] != "cs_01"]}
    )
    bad = matrix.adapters["cs_02"][Tier.SUPER].spec.model_copy(update={"response_format": {"type": "<expr>"}})
    matrix.adapters["cs_02"][Tier.SUPER] = Adapter(site_id="cs_02", version=1, spec=bad)
    server = FakeTF(reject_models={get_tier(Tier.ULTRA).model_id})  # cs_02 on ultra: a provider error
    monkeypatch.setenv("NEBIUS_BASE_URL", server.url)
    try:
        run = local(matrix, tmp_path / "local")
        shutil.copytree(tmp_path / "local", tmp_path / "cache")
        via_job = job(matrix, tmp_path / "cache")
    finally:
        server.stop()
    assert dumps(via_job.results) == dumps(run.results)
    cells = {(e["site_id"], e["tier"], e["split"]): e for e in run.results}
    kinds = {cell: e.get("error_type", e["kind"]) for cell, e in cells.items()}
    expected = {}
    for split in ("dev", "holdout"):
        expected |= {("cs_01", t.value, split): "MissingReference" for t in TIER_ORDER}
        expected |= {
            ("cs_02", "lightning", split): "tier_result",
            ("cs_02", "super", split): "UnresolvedCallSite",
            ("cs_02", "ultra", split): "ProviderError",
        }
    assert kinds == expected
    assert cells[("cs_01", "super", "dev")]["case_ids"] == ["cs_01:000", "cs_01:001", "cs_01:002"]
    assert cells[("cs_02", "ultra", "dev")]["status"] == 401
    assert SECRET not in dumps(run.results)


def test_a_per_run_cap_is_a_budget_stop_entry(tf, tmp_path):
    out = local(make_matrix(budget={"max_out": 1}), tmp_path / "cache", workers=1).results
    stops = [e for e in out if e["kind"] == "budget_stop"]
    assert len(stops) == len(out) and all(e["reason"].startswith("budget: ") for e in stops)


# ------------------------------------------------------------------------------------ the entrypoint


def run_entrypoint(*args) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "NEBIUS_API_KEY"}  # replay needs no key
    return subprocess.run(
        [sys.executable, "-m", "offramp.evals.serverless_job.main", *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )


def test_entrypoint_writes_the_results_file(tf, tmp_path):
    matrix = make_matrix()
    (tmp_path / "matrix.json").write_text(matrix.model_dump_json())
    expected = local(matrix, tmp_path / "cache")

    done = run_entrypoint(
        "--input",
        str(tmp_path / "matrix.json"),
        "--output",
        str(tmp_path / "results.json"),
        "--cache-dir",
        str(tmp_path / "cache"),
    )

    assert done.returncode == 0, done.stderr
    assert (tmp_path / "results.json").read_text() == dumps(expected.results, indent=2) + "\n"
    assert transport.RESULTS_TAG not in done.stdout  # stdout results only on request


def test_entrypoint_needs_somewhere_to_put_the_results(tmp_path):
    (tmp_path / "matrix.json").write_text(make_matrix().model_dump_json())
    done = run_entrypoint("--input", str(tmp_path / "matrix.json"))
    assert done.returncode != 0 and "--output" in done.stderr


# ------------------------------------------------------------------------------------- the image


def test_dockerfile_bakes_in_no_secret_and_runs_the_entrypoint():
    text = (REPO / "Dockerfile.eval").read_text()
    assert "offramp.evals.serverless_job.main" in text
    assert not [
        line for line in text.splitlines() if "KEY" in line.upper() and not line.lstrip().startswith("#")
    ]
