"""Serverless eval job (PRD §6.5, §12.4): the local worker and the Nebius job path give the same results.

Built and tested against a fake `nebius` CLI; never run on Nebius AI Cloud (it needs separate billing, and
Token Factory credit cannot fund it). Nothing is mocked inside the code under test:
- Token Factory is a stdlib HTTP server (`FakeTF`) the client reaches through NEBIUS_BASE_URL, so the job's
  subprocess talks to it as the container would talk to the real endpoint;
- the `nebius` CLI is a script on PATH (`FAKE_NEBIUS`) that does what the real one does for the verbs
  submit.py uses: `ai job create` runs the container entrypoint (`python -m
  offramp.evals.serverless_job.main`) with the --env, --env-secret (resolved from a fake MysteryBox) and
  --inject-file parts it was given, and `get`/`logs`/`cancel` read the job it recorded. What it prints is only
  as faithful as docs.nebius.com/cli/reference/ai/job/* and `nebius ai job create --help`.
Latencies are measured per live call, so a live run and a job run differ in p50/p95. The job path is
therefore compared with a local run over the same cache (replay), which is how a pipeline re-run behaves.
"""

import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from offramp.evals.serverless_job import submit, transport
from offramp.evals.serverless_job.matrix import (
    Matrix,
    MatrixResults,
    absorb_usage,
    dumps,
    needed_cache_keys,
    remaining_caps,
)
from offramp.evals.serverless_job.submit import (
    NOT_CONFIGURED,
    JobError,
    JobUnavailable,
    redact,
    run_matrix_job,
)
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
from offramp.tf import Budget, BudgetExceeded, Cache, get_tier

REPO = Path(__file__).resolve().parents[2]
SECRET = "nbk-SECRET-0123456789"
FALLBACK_SECRET = "nbk-FALLBACK-9876543210"
SELECTOR = "mbsec-e00test"
LIGHTNING = get_tier(Tier.LIGHTNING).model_id
CAPS = remaining_caps(Budget())  # a run that has spent nothing

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
        sys.exit(os.environ["FAKE_NEBIUS_CREATE_FAILS"])  # stderr text of the test's choosing
    root = state / "jobs" / "job-1"
    mount = root / "fs"
    env = {{k: v for k, v in os.environ.items() if not k.startswith(("NEBIUS_", "OFFRAMP_", "FAKE_"))}}
    env["PYTHONPATH"] = os.environ["FAKE_NEBIUS_REPO"]
    for pair in flag_values("--env"):
        k, _, v = pair.partition("=")
        env[k] = v
    mysterybox = json.loads((state / "secrets.json").read_text())  # selector -> payload
    for pair in flag_values("--env-secret"):  # KEY=SELECTOR: the payload key must be named KEY
        name, _, selector = pair.partition("=")
        if name not in mysterybox.get(selector, {{}}):
            sys.exit("error: secret %s has no payload key %s" % (selector, name))
        env[name] = mysterybox[selector][name]
    for pair in flag_values("--inject-file"):
        local, _, container = pair.partition(":")
        target = mount / container.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, target)
    logs, returncode = [], 0
    if not os.environ.get("FAKE_NEBIUS_SKIP_RUN"):
        args = flag_values("--args")[0].replace("/offramp/input", str(mount / "offramp" / "input"))
        command = flag_values("--container-command")[0]
        command = sys.executable if command == "python" else command
        done = subprocess.run([command, *shlex.split(args)], env=env, cwd=os.environ["FAKE_NEBIUS_REPO"],
                              capture_output=True, text=True)
        logs, returncode = (done.stdout + done.stderr).splitlines(), done.returncode
    logs += os.environ.get("FAKE_NEBIUS_EXTRA_LOG", "").splitlines()
    logs = [PREFIX + line for line in logs]
    drop = os.environ.get("FAKE_NEBIUS_DROP_CHUNK")
    if drop:  # logs that lost a line: the chunk sent as "<drop>/<n>"
        logs = [line for line in logs if " %s/" % drop not in line]
    (root / "logs.txt").write_text("\\n".join(logs) + "\\n")
    (root / "returncode").write_text(str(returncode))
    print(json.dumps({{"id": "operation-1", "resource_id": "job-1"}}))
elif verb == "get":
    polls = state / "polls"
    n = int(polls.read_text()) + 1 if polls.exists() else 1
    polls.write_text(str(n))
    failures = int(os.environ.get("FAKE_NEBIUS_GET_FAILS", "0"))
    if n <= failures:
        sys.exit("error: service unavailable")  # a transient failure
    default = "COMPLETED" if (state / "jobs" / job / "returncode").read_text() == "0" else "FAILED"
    states = os.environ.get("FAKE_NEBIUS_STATES", default).split(",")
    i = min(n - failures - 1, len(states) - 1)
    status = {{"state": states[i], "state_details": {{"message": "scripted state"}}}}
    if "RUNNING" in states[: i + 1] or states[i] == "COMPLETED":
        status["started_at"] = "2026-10-19T12:00:00Z"
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
    for name in (
        "NEBIUS_API_KEY_FALLBACK",
        "NEBIUS_JOB_SECRET_ID",
        "NEBIUS_JOB_SECRET_ID_FALLBACK",
        *(f"OFFRAMP_MODEL_{t.name}" for t in TIER_ORDER),
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NEBIUS_API_KEY", SECRET)
    monkeypatch.delenv("NEBIUS_PROJECT_ID", raising=False)
    monkeypatch.delenv("OFFRAMP_EVAL_IMAGE", raising=False)
    monkeypatch.setattr(
        submit, "NEBIUS_CONFIG", tmp_path / "no-such-nebius-config.yaml"
    )  # not this machine's


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
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def verbs(self) -> list[str]:
        return [c[2] for c in self.calls()]

    @property
    def cancelled(self) -> bool:
        return (self.root / "cancelled").exists()

    @property
    def logs(self) -> str:
        return (self.root / "jobs" / "job-1" / "logs.txt").read_text()


@pytest.fixture
def nebius(tmp_path, monkeypatch):
    """The fake CLI on PATH, a profile, the project, image and MysteryBox secret the job path needs."""
    bin_dir, state = tmp_path / "bin", tmp_path / "nebius-state"
    bin_dir.mkdir()
    state.mkdir()
    script = bin_dir / "nebius"
    script.write_text(FAKE_NEBIUS.format(python=sys.executable))
    script.chmod(0o755)
    (state / "secrets.json").write_text(
        json.dumps({SELECTOR: {"NEBIUS_API_KEY": SECRET, "NEBIUS_API_KEY_FALLBACK": FALLBACK_SECRET}})
    )
    config = tmp_path / "nebius-config.yaml"
    config.write_text("default: offramp\nprofiles:\n  offramp:\n    endpoint: api.example.invalid\n")
    monkeypatch.setattr(submit, "NEBIUS_CONFIG", config)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_NEBIUS_STATE", str(state))
    monkeypatch.setenv("FAKE_NEBIUS_REPO", str(REPO))
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "project-e00test")
    monkeypatch.setenv("NEBIUS_JOB_SECRET_ID", SELECTOR)
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
    fields = {"budget": dict(CAPS), **overrides}
    return Matrix(sites=sites, adapters=adapters, cases=cases, references=refs, **fields)


def local(matrix, cache_dir, **kw) -> MatrixResults:
    return run_matrix_job(matrix, mode="local", cache_dir=cache_dir, **kw)


def job(matrix, cache_dir, **kw) -> MatrixResults:
    return run_matrix_job(matrix, mode="job", cache_dir=cache_dir, poll_s=0.05, **kw)


def auto(matrix, cache_dir, **kw) -> MatrixResults:
    return run_matrix_job(matrix, cache_dir=cache_dir, poll_s=0.05, **kw)


def cache_keys(directory: Path) -> set[str]:
    return {p.stem for p in directory.glob("*.json")}


def straddling(prefix_len: int) -> str:
    """Text in which SECRET starts so that a cut at `prefix_len` lands in the middle of it."""
    return "e" * (prefix_len - len(SECRET) // 2) + SECRET


def partial_key_in(text: str) -> bool:
    """Any run of 6 or more characters of SECRET at either end of it, or all of it."""
    return any(SECRET[:n] in text for n in range(6, len(SECRET) + 1)) or any(
        SECRET[i:] in text for i in range(len(SECRET) - 5)
    )


# ------------------------------------------------------------------------------------------ the matrix


def test_local_run_covers_every_cell_sorted(tf, tmp_path):
    run = local(make_matrix(), tmp_path / "cache")
    results = run.results
    cells = [(e["site_id"], e["tier"], e["split"]) for e in results]
    assert len(cells) == 2 * 3 * 2
    assert cells == sorted(cells, key=lambda c: (c[0], TIER_ORDER.index(Tier(c[1])), c[2]))
    assert {e["kind"] for e in results} == {"tier_result"}
    by_tier = {e["tier"]: TierResult.model_validate(e["result"]).passed for e in results}
    assert by_tier == {"lightning": False, "super": True, "ultra": True}
    assert len(tf.requests) == 2 * 3 * 6  # each case once per tier, both splits
    assert (run.ran_on, run.job_id, run.fallback_reason) == ("local", None, None)


def test_matrix_needs_an_adapter_for_every_site_and_tier():
    matrix = make_matrix()
    del matrix.adapters["cs_02"][Tier.ULTRA]
    with pytest.raises(ValidationError, match=r"cs_02/ultra"):
        Matrix.model_validate_json(matrix.model_dump_json())


@pytest.mark.parametrize(
    "budget",
    [
        {},  # not a fresh full cap
        {"max_in": 1, "max_out": 1},
        {"max_in": 1, "max_out": 1, "max_ultra": 1},  # a typo
        {**CAPS, "max_out": -1},
    ],
)
def test_the_budget_must_be_explicit_and_valid(budget):
    with pytest.raises(ValidationError, match="budget must give"):
        make_matrix(budget=budget)


# ------------------------------------------------------------------------- local vs Serverless Job path


def test_job_results_equal_local_byte_for_byte(tf, nebius, tmp_path, monkeypatch):
    matrix = make_matrix()
    local_run = local(matrix, tmp_path / "local-cache")
    shutil.copytree(tmp_path / "local-cache", tmp_path / "job-cache")
    for n in range(3):  # entries of other runs in the same cache directory
        (tmp_path / "job-cache" / f"{n:064x}.json").write_text("{}")
    served = len(tf.requests)
    monkeypatch.setenv(
        "NEBIUS_BASE_URL", "http://127.0.0.1:9/v1"
    )  # nothing listens: a cache miss is an error

    job_run = job(matrix, tmp_path / "job-cache")

    assert dumps(job_run.results) == dumps(local_run.results)
    assert dumps(sorted(job_run.results, key=json.dumps)) == dumps(sorted(local_run.results, key=json.dumps))
    assert {e["kind"] for e in job_run.results} == {"tier_result"}
    assert job_run.cache == {}  # everything replayed from the shipped cache
    assert len(tf.requests) == served
    assert (job_run.ran_on, job_run.job_id, job_run.fallback_reason) == ("job", "job-1", None)
    assert nebius.verbs()[0] == "create" and {"get", "logs"} <= set(nebius.verbs())
    # MatrixResults.budget is the usage of the run alone: the live run spent, its replay did not
    assert (local_run.budget["tokens_in"], local_run.budget["tokens_out"]) == (12 * 36, 2 * 36)
    assert job_run.budget["tokens_in"] == job_run.budget["tokens_out"] == job_run.budget["tf_spend_usd"] == 0


def test_job_writes_new_cache_entries_back_like_a_local_run(tf, nebius, tmp_path):
    matrix = make_matrix()
    local_run = local(matrix, tmp_path / "local-cache")
    local_keys, served = cache_keys(tmp_path / "local-cache"), len(tf.requests)

    job_run = job(matrix, tmp_path / "job-cache")  # empty cache: the job calls Token Factory live

    assert len(tf.requests) == 2 * served
    assert {r["auth"] for r in tf.requests[served:]} == {f"Bearer {SECRET}"}  # the secret reached the job
    assert cache_keys(tmp_path / "job-cache") == local_keys == set(job_run.cache)
    replay = local(matrix.model_copy(update={"live": False}), tmp_path / "job-cache")
    assert dumps(replay.results) == dumps(job_run.results)  # the filled cache replays to the same numbers
    assert len(tf.requests) == 2 * served
    for name in ("tokens_in", "tokens_out", "ultra_agent_calls"):  # the same usage on both paths
        assert job_run.budget[name] == local_run.budget[name]
    assert job_run.budget["tf_spend_usd"] == pytest.approx(local_run.budget["tf_spend_usd"])


def test_only_the_cache_entries_the_matrix_replays_are_shipped(tf, tmp_path):
    matrix = make_matrix()
    local(matrix, tmp_path / "cache")
    wanted = cache_keys(tmp_path / "cache")
    for n in range(3):
        Cache(tmp_path / "cache").put(f"{n:064x}", {"other": "run"})

    keys = needed_cache_keys(matrix, tmp_path / "cache")

    assert keys == wanted and len(wanted) == 2 * 3 * 6
    tar = tarfile.open(
        fileobj=io.BytesIO(transport.pack_input(matrix, tmp_path / "cache", keys)), mode="r:gz"
    )
    assert sorted(tar.getnames()) == sorted(["matrix.json", *(f"cache/{k}.json" for k in wanted)])
    assert needed_cache_keys(matrix, tmp_path / "empty") == set()  # nothing cached: nothing to ship


# --------------------------------------------------------------------------------------- key handling


def test_no_key_value_reaches_an_argv_a_log_or_a_result(tf, nebius, tmp_path, caplog, capsys, monkeypatch):
    monkeypatch.setenv("NEBIUS_JOB_SECRET_ID_FALLBACK", SELECTOR)  # the same secret holds the fallback key
    monkeypatch.setenv("NEBIUS_API_KEY_FALLBACK", FALLBACK_SECRET)
    caplog.set_level(logging.DEBUG)

    run = job(make_matrix(), tmp_path / "cache")

    calls = nebius.calls()
    assert {c[2] for c in calls} >= {"create", "get", "logs"}
    for argv in calls:  # every argv the submitter built
        assert SECRET not in " ".join(argv) and FALLBACK_SECRET not in " ".join(argv)
    create = next(c for c in calls if c[2] == "create")
    assert [create[i + 1] for i, a in enumerate(create) if a == "--env-secret"] == [
        f"NEBIUS_API_KEY={SELECTOR}",
        f"NEBIUS_API_KEY_FALLBACK={SELECTOR}",
    ]  # the key is named, never passed
    plain = [create[i + 1] for i, a in enumerate(create) if a == "--env"]
    assert plain and not [v for v in plain if "KEY" in v.upper()]
    seen = capsys.readouterr()
    everything = caplog.text + seen.out + seen.err + nebius.logs
    assert SECRET not in everything and FALLBACK_SECRET not in everything
    payload = transport.decode_results(nebius.logs)  # the gzip+base64 payload: results, cache entries, budget
    assert SECRET not in payload.model_dump_json() and SECRET not in run.model_dump_json()


def test_a_replay_job_is_passed_no_secret(tf, nebius, tmp_path):
    matrix = make_matrix()
    local(matrix, tmp_path / "cache")
    job(matrix.model_copy(update={"live": False}), tmp_path / "cache")
    create = next(c for c in nebius.calls() if c[2] == "create")
    assert "--env-secret" not in create and not [a for a in create if "NEBIUS_API_KEY" in a]


def test_the_key_is_scrubbed_before_an_error_is_cut_short(tf, nebius, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_NEBIUS_CREATE_FAILS", straddling(500) + "tail")  # the cut lands inside the key
    with pytest.raises(JobError, match="exited 1") as e:
        job(make_matrix(), tmp_path / "cache")
    assert not partial_key_in(str(e.value))


def test_redact_hides_secret_env_values_only():
    argv = [
        "ai",
        "job",
        "create",
        "--env",
        f"NEBIUS_API_KEY={SECRET}",
        "--env",
        "OFFRAMP_MODEL_SUPER=m/x",
        "--env-secret",
        f"NEBIUS_API_KEY={SELECTOR}",
        "--registry-password",
        "hunter2",
    ]
    shown = redact(argv)
    assert SECRET not in shown and "hunter2" not in shown
    assert "NEBIUS_API_KEY=***" in shown and "OFFRAMP_MODEL_SUPER=m/x" in shown
    assert f"NEBIUS_API_KEY={SELECTOR}" in shown  # a selector names a secret and is not one


# ------------------------------------------------------------------------------ configuration, fallback


def test_the_job_path_is_not_configured_without_a_profile_or_a_secret_id(tf, nebius, tmp_path, monkeypatch):
    """The default (`auto`) path of every run without Nebius AI Cloud: local, quickly, no CLI call at all."""

    def no_cli(*args, **kw):
        raise AssertionError(f"the nebius CLI was called: {args}")

    monkeypatch.setattr(submit.subprocess, "run", no_cli)
    matrix = make_matrix()
    expected = local(matrix, tmp_path / "cache").results

    monkeypatch.delenv("NEBIUS_JOB_SECRET_ID")  # no secret
    no_secret = auto(matrix, tmp_path / "cache")
    monkeypatch.setenv("NEBIUS_JOB_SECRET_ID", SELECTOR)
    monkeypatch.setattr(submit, "NEBIUS_CONFIG", tmp_path / "missing.yaml")  # no profile
    no_profile = auto(matrix, tmp_path / "cache")
    (tmp_path / "bad.yaml").write_text(
        "default: other\nprofiles:\n  offramp: {}\n"
    )  # default is not a profile
    monkeypatch.setattr(submit, "NEBIUS_CONFIG", tmp_path / "bad.yaml")
    bad_default = auto(matrix, tmp_path / "cache")

    for run in (no_secret, no_profile, bad_default):
        assert dumps(run.results) == dumps(expected)
        assert (run.ran_on, run.job_id, run.fallback_reason) == ("local", None, "nebius job: not configured")
        assert run.fallback_reason == NOT_CONFIGURED
    assert nebius.calls() == []
    with pytest.raises(JobUnavailable, match="profile"):
        run_matrix_job(matrix, mode="job", cache_dir=tmp_path / "cache")


@pytest.mark.parametrize(
    "missing", ["NEBIUS_PROJECT_ID", "NEBIUS_JOB_SECRET_ID", "OFFRAMP_EVAL_IMAGE", "the nebius CLI"]
)
def test_job_mode_refuses_with_what_is_missing(tf, nebius, tmp_path, monkeypatch, missing):
    if missing == "the nebius CLI":
        monkeypatch.setenv("PATH", str(tmp_path))
    else:
        monkeypatch.delenv(missing)
    matrix = make_matrix()
    with pytest.raises(JobUnavailable, match=missing):
        run_matrix_job(matrix, mode="job", cache_dir=tmp_path / "cache")
    assert nebius.calls() == []
    run = run_matrix_job(matrix, cache_dir=tmp_path / "cache")  # auto
    assert run.ran_on == "local" and run.fallback_reason == NOT_CONFIGURED and run.results


def test_a_create_that_fails_falls_back_in_auto_and_says_so(tf, nebius, tmp_path, monkeypatch):
    matrix = make_matrix()
    expected = local(matrix, tmp_path / "local").results
    shutil.copytree(tmp_path / "local", tmp_path / "cache")
    monkeypatch.setenv("FAKE_NEBIUS_CREATE_FAILS", "error: quota exceeded")
    with pytest.raises(JobError, match="quota exceeded") as e:
        job(matrix, tmp_path / "cache")
    assert not e.value.ran and e.value.job_id is None

    run = auto(matrix, tmp_path / "cache")

    assert dumps(run.results) == dumps(expected)
    assert run.ran_on == "local" and run.job_id is None
    assert run.fallback_reason.startswith("Serverless Job path failed before the job ran")
    assert "quota exceeded" in run.fallback_reason


def test_too_big_an_input_is_unavailable_not_truncated(monkeypatch):
    monkeypatch.setattr(transport, "PART_BYTES", 10)
    monkeypatch.setattr(transport, "MAX_PARTS", 2)
    with pytest.raises(JobUnavailable, match="too large"):
        transport.split_parts(b"x" * 100)


# --------------------------------------------------------------------------------------------- lifecycle


def test_job_is_polled_until_it_finishes_and_a_blip_is_retried(tf, nebius, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_NEBIUS_STATES", "PROVISIONING,RUNNING,COMPLETED")
    monkeypatch.setenv("FAKE_NEBIUS_GET_FAILS", "2")  # the first two status reads fail
    run = job(make_matrix(), tmp_path / "cache")
    assert run.results and nebius.verbs().count("get") == 2 + 3
    assert not nebius.cancelled


def test_status_reads_that_keep_failing_cancel_the_job(tf, nebius, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_NEBIUS_SKIP_RUN", "1")
    monkeypatch.setenv("FAKE_NEBIUS_GET_FAILS", "99")
    for run in (job, auto):  # the job may be running and spending: not re-run locally either
        with pytest.raises(JobError, match="after 3 tries") as e:
            run(make_matrix(), tmp_path / "cache")
        assert e.value.ran and e.value.job_id == "job-1"
    assert nebius.cancelled
    assert not tf.requests


def test_an_unrecognised_job_state_fails_fast_instead_of_polling_for_hours(tf, nebius, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_NEBIUS_SKIP_RUN", "1")
    monkeypatch.setenv("FAKE_NEBIUS_STATES", "SOMETHING_NEW")
    with pytest.raises(JobError, match="no recognised status.state") as e:
        job(make_matrix(), tmp_path / "cache")
    assert e.value.ran and nebius.verbs().count("get") == submit.UNKNOWN_POLLS
    assert nebius.cancelled


def test_a_failed_job_error_carries_its_scrubbed_logs_and_is_not_re_run(tf, nebius, tmp_path, monkeypatch):
    # the last 1500 characters of the logs begin inside the key
    tail = "z" * (1500 - len(SECRET) // 2 - len("\nValueError: boom") - 30 - 5) + "\nValueError: boom"
    monkeypatch.setenv("FAKE_NEBIUS_SKIP_RUN", "1")
    monkeypatch.setenv("FAKE_NEBIUS_STATES", "RUNNING,FAILED")
    monkeypatch.setenv("FAKE_NEBIUS_EXTRA_LOG", f"Traceback\n{SECRET}{tail}\n{transport.RESULTS_TAG} junk")
    for run in (job, auto):
        with pytest.raises(JobError, match="ended FAILED") as e:
            run(make_matrix(), tmp_path / "cache")
        assert e.value.ran and e.value.job_id == "job-1"
        assert "ValueError: boom" in str(e.value) and transport.RESULTS_TAG not in str(e.value)
        assert not partial_key_in(str(e.value))
    assert nebius.cancelled
    assert not tf.requests  # auto did not run the matrix on this machine after the job had run


def test_a_job_that_never_started_falls_back_in_auto_and_is_cancelled(tf, nebius, tmp_path, monkeypatch):
    matrix = make_matrix()
    expected = local(matrix, tmp_path / "local").results
    shutil.copytree(tmp_path / "local", tmp_path / "cache")
    monkeypatch.setenv("FAKE_NEBIUS_SKIP_RUN", "1")
    monkeypatch.setenv("FAKE_NEBIUS_STATES", "PROVISIONING,ERROR")
    with pytest.raises(JobError, match="ended ERROR") as e:
        job(matrix, tmp_path / "cache")
    assert not e.value.ran and e.value.job_id == "job-1"

    run = auto(matrix, tmp_path / "cache")

    assert dumps(run.results) == dumps(expected)
    assert run.ran_on == "local" and run.job_id is None
    assert "before the job ran" in run.fallback_reason and "job job-1 ended ERROR" in run.fallback_reason
    assert nebius.cancelled


def test_lost_logs_cancel_the_job_and_are_not_re_run_locally(tf, nebius, tmp_path, monkeypatch):
    monkeypatch.setattr(submit, "CHUNK_CHARS", 300)
    monkeypatch.setenv("FAKE_NEBIUS_DROP_CHUNK", "2")
    with pytest.raises(JobError, match="missing") as e:
        job(make_matrix(), tmp_path / "cache")
    assert e.value.ran and e.value.job_id == "job-1" and nebius.cancelled
    served = len(tf.requests)
    assert served == 2 * 3 * 6  # the job spent: Token Factory saw its calls
    with pytest.raises(JobError, match="missing"):
        auto(make_matrix(), tmp_path / "cache")  # a second job; never the local worker
    assert len(tf.requests) == 2 * served


def test_job_cache_entries_under_a_bad_key_are_refused(tmp_path):
    cache = Cache(tmp_path / "cache")
    good = "a" * 64
    with pytest.raises(JobError, match="invalid keys"):
        submit._merge_cache(cache, {good: {"v": 1}, "../../escaped": {"v": 2}})
    assert cache_keys(tmp_path / "cache") == set()
    assert not (tmp_path / "escaped.json").exists()
    submit._merge_cache(cache, {good: {"v": 1}})
    assert cache.get(good) == {"v": 1}


# ------------------------------------------------------------------------------------ transport


def test_results_larger_than_one_log_line_come_back_in_chunks(tf, nebius, tmp_path, monkeypatch):
    monkeypatch.setattr(submit, "CHUNK_CHARS", 300)
    matrix = make_matrix()
    run = job(matrix, tmp_path / "cache")
    assert nebius.logs.count(transport.RESULTS_TAG) > 3
    assert run.results and {e["kind"] for e in run.results} == {"tier_result"}


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


# ------------------------------------------------------------------------------------------- budget


def test_remaining_caps_and_absorbing_a_runs_usage_back():
    budget = Budget(max_in=1000, max_out=100, max_ultra_agent_calls=5)
    budget.charge("eval_target", Tier.SUPER, 400, 30, 0.01)
    budget.charge("judge", Tier.ULTRA, 10, 5, 0.02)  # one Ultra agent call
    assert remaining_caps(budget) == {"max_in": 590, "max_out": 65, "max_ultra_agent_calls": 4}

    used = {"tokens_in": 100, "tokens_out": 10, "ultra_agent_calls": 2, "tf_spend_usd": 0.5}
    absorb_usage(budget, MatrixResults(results=[], budget=used))
    snapshot = budget.snapshot()
    assert (snapshot["tokens_in"], snapshot["tokens_out"], snapshot["ultra_agent_calls"]) == (510, 45, 3)
    assert snapshot["tf_spend_usd"] == pytest.approx(0.53)

    absorb_usage(budget, MatrixResults(results=[]))  # a run that reports no usage changes nothing
    assert budget.snapshot() == snapshot

    over = {"tokens_in": 1, "tokens_out": 100, "ultra_agent_calls": 3, "tf_spend_usd": 0.1}
    with pytest.raises(BudgetExceeded):  # the spend is real: all of it is recorded, then the cap is reported
        absorb_usage(budget, MatrixResults(results=[], budget=over))
    snapshot = budget.snapshot()
    assert (snapshot["tokens_in"], snapshot["tokens_out"], snapshot["ultra_agent_calls"]) == (511, 145, 6)


def test_a_per_run_cap_is_a_budget_stop_entry(tf, tmp_path):
    matrix = make_matrix(budget={**CAPS, "max_out": 1})
    out = local(matrix, tmp_path / "cache", workers=1).results
    stops = [e for e in out if e["kind"] == "budget_stop"]
    assert len(stops) == len(out) and all(e["reason"].startswith("budget: ") for e in stops)


def test_the_job_honours_the_caps_left_and_reports_its_usage(tf, nebius, tmp_path):
    caller = Budget()
    caller.charge("eval_target", Tier.SUPER, 1_000_000, 299_999, 0.0)  # a run that has spent almost its cap
    matrix = make_matrix(budget=remaining_caps(caller))
    assert matrix.budget["max_out"] == 1

    run = job(matrix, tmp_path / "cache")

    assert {e["kind"] for e in run.results} == {"budget_stop"}  # the job stopped at the cap it was given
    assert run.budget["tokens_out"] >= 2 and run.ran_on == "job"
    before = caller.snapshot()
    with pytest.raises(BudgetExceeded):  # what the job spent goes back on the caller's budget: over its cap
        absorb_usage(caller, run)
    assert caller.snapshot()["tokens_out"] == before["tokens_out"] + run.budget["tokens_out"]


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
    (tmp_path / "matrix.json").write_text(matrix.to_json())
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
    (tmp_path / "matrix.json").write_text(make_matrix().to_json())
    done = run_entrypoint("--input", str(tmp_path / "matrix.json"))
    assert done.returncode != 0 and "--output" in done.stderr


# ------------------------------------------------------------------------------------- the image


def test_dockerfile_bakes_in_no_secret_and_runs_the_entrypoint():
    text = (REPO / "Dockerfile.eval").read_text()
    assert "offramp.evals.serverless_job.main" in text
    assert not [
        line for line in text.splitlines() if "KEY" in line.upper() and not line.lstrip().startswith("#")
    ]
