"""Run the eval matrix as a Nebius Serverless Job or locally: same entries either way (PRD §6.5, §12.4).

`run_matrix_job(matrix, mode="auto")` submits with the `nebius` CLI when the job path is configured: a
default profile in ~/.nebius/config.yaml, the CLI on PATH, NEBIUS_PROJECT_ID, NEBIUS_JOB_SECRET_ID and
OFFRAMP_EVAL_IMAGE (an image built from Dockerfile.eval, pushed where the project can pull it). Otherwise
`auto` runs the same `run_matrix` in-process with `fallback_reason == NOT_CONFIGURED`, without a single CLI
call (the checks read the environment and the config file only); `mode="job"` raises JobUnavailable naming
what is missing.

The key never travels on a command line or in the job spec. A live job gets it as
`--env-secret NEBIUS_API_KEY=<NEBIUS_JOB_SECRET_ID>`, a MysteryBox secret whose payload has a key named
NEBIUS_API_KEY (the variable name must equal the payload key; the CLI has no other syntax). The same goes
for NEBIUS_API_KEY_FALLBACK with NEBIUS_JOB_SECRET_ID_FALLBACK, which may name the same secret. The job's
service account must be allowed to read the secret. A replay (`live=False`) job is passed no key.

The matrix and the cache entries it replays go in as `--inject-file` parts. Results and the cache entries
the job wrote come back on stdout (transport.py) and are merged into the local cache, so a job run fills
the same cache a local run would. `MatrixResults.ran_on` / `job_id` / `fallback_reason` say where the
cells ran.

A job that may have spent Token Factory credit is never silently re-run: `auto` falls back to local only
when the job could not be created or never started. After any failure past `create` the job is cancelled.

Verbs and flags: docs.nebius.com/cli/reference/ai/job/{create,get,logs,cancel}, `nebius ai job create
--help`. Checked against those: the flags below, `create --async --format json` printing the Operation with
`resource_id`, `--env-secret KEY=SECRET_SELECTOR`, and the job state names (nebius/api ai/v1/job.proto).
Not checked: this path has only run against a fake `nebius` (Nebius AI Cloud needs separate billing, and
Token Factory credit cannot fund it). So unknown: that a real job accepts these flags as used (--args split
on spaces, several --inject-file), the CPU platform/preset names, the `get --format json` field names, and
the log line format (parsing tolerates a prefix).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml

from offramp.evals.serverless_job.matrix import (
    DEFAULT_WORKERS,
    Matrix,
    MatrixResults,
    budget_usage,
    dumps,
    make_client,
    needed_cache_keys,
    run_matrix,
    scrub,
)
from offramp.evals.serverless_job.transport import (
    CHUNK_CHARS,
    PART_PREFIX,
    RESULTS_TAG,
    JobError,
    JobUnavailable,
    decode_results,
    pack_input,
    split_parts,
)
from offramp.tf import LLM, Cache, TFClient

__all__ = ["JobError", "JobUnavailable", "redact", "run_local", "run_matrix_job"]

log = logging.getLogger(__name__)

NOT_CONFIGURED = "nebius job: not configured"  # MatrixResults.fallback_reason when auto never tried a job
NEBIUS_CONFIG = Path.home() / ".nebius" / "config.yaml"  # where the CLI keeps its profiles
INPUT_DIR = "/offramp/input"  # where --inject-file puts the parts inside the container
JOB_PLATFORM, JOB_PRESET = "cpu-d3", "4vcpu-16gb"  # the work is network-bound: a CPU VM, not the GPU default
# Config that must match the submitter or the job's cache keys and prices differ from a local run's.
FORWARDED_ENV = ("NEBIUS_BASE_URL", "OFFRAMP_MODEL_LIGHTNING", "OFFRAMP_MODEL_SUPER", "OFFRAMP_MODEL_ULTRA")
IN_PROGRESS = {"PROVISIONING", "STARTING", "IMAGE_PULLING", "RUNNING"}  # JobStatus.State
ENDED_BADLY = {"FAILED", "CANCELLED", "CANCELLING", "DELETING", "ERROR"}  # COMPLETED is the good end
GET_ATTEMPTS = 3  # tries per status read, with backoff
UNKNOWN_POLLS = 3  # polls without a recognised status.state before giving up
LOG_ATTEMPTS = 5  # the job's logs can lag its COMPLETED state
LOG_TAIL_CHARS = 1500  # of a failed job's logs, in the error
CLI_TIMEOUT_S = 300
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.I)
CACHE_KEY = re.compile(r"[0-9a-f]{64}")


def redact(argv: Sequence[str]) -> str:
    """A command line safe to print: secret `--env` values and registry credentials replaced by ***.
    (The key is never on a command line; this is the second line of defence.)"""
    shown, previous = [], ""
    for arg in argv:
        name = arg.partition("=")[0]
        if previous == "--env" and SECRET_NAME.search(name):
            arg = f"{name}=***"
        elif previous in ("--registry-username", "--registry-password"):
            arg = "***"
        shown.append(arg)
        previous = arg
    return shlex.join(shown)


def _head(text: str, n: int) -> str:
    return scrub(text).strip()[:n]  # scrub first: a cut can fall inside a key and leave its prefix


def _tail(text: str, n: int) -> str:
    return scrub(text).strip()[-n:]


def _cli(*args: str) -> str:
    cmd = [shutil.which("nebius") or "nebius", *args]
    log.info("nebius %s", redact(args))
    verb = " ".join(args[:3])
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=CLI_TIMEOUT_S)
    except subprocess.TimeoutExpired:  # `from None` throughout: the exception carries the argv
        raise JobError(
            f"nebius {verb} timed out after {CLI_TIMEOUT_S} s; its outcome is unknown, "
            "look in `nebius ai job list`",
            ran=True,
        ) from None
    except OSError as e:
        raise JobError(f"nebius {verb}: {type(e).__name__}") from None
    if done.returncode != 0:  # the full command is in the log, redacted; the error only needs the verb
        raise JobError(f"nebius {verb} exited {done.returncode}: {_head(done.stderr, 500)}")
    return done.stdout


def run_local(
    matrix: Matrix,
    *,
    cache_dir: str | Path | None = None,
    client: LLM | None = None,
    workers: int = DEFAULT_WORKERS,
) -> MatrixResults:
    """The in-process worker. `MatrixResults.budget` is this run's usage alone. With your own `client`, that
    client's Budget has already been charged by the run: do not absorb_usage the results into it again."""
    client = client or make_client(matrix, cache_dir)
    before = budget_usage(client)
    results = run_matrix(matrix, client, workers=workers)
    after = budget_usage(client)
    return MatrixResults(results=results, budget={k: after[k] - before[k] for k in after}, ran_on="local")


def _profile_configured() -> bool:
    """The CLI has a default profile. Read from its config file, not by running it."""
    try:
        config = yaml.safe_load(NEBIUS_CONFIG.read_text())
        return config["default"] in config["profiles"]
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return False


def _unavailable(project_id: str | None, image: str | None, secret_id: str | None) -> str | None:
    """What is missing for the job path, or None. No CLI call: this is the fast path of every local run."""
    if not _profile_configured():
        return f"no default nebius profile in {NEBIUS_CONFIG}"
    if not shutil.which("nebius"):
        return "the nebius CLI is not on PATH"
    if not project_id:
        return "NEBIUS_PROJECT_ID is not set"
    if not secret_id:
        return (
            "NEBIUS_JOB_SECRET_ID is not set: a live job gets the Token Factory key only as a MysteryBox "
            "secret (--env-secret), never on the command line"
        )
    if not image:
        return "OFFRAMP_EVAL_IMAGE is not set"
    return None


def _create(
    parts: list[bytes],
    *,
    name: str,
    image: str,
    project_id: str,
    platform: str,
    preset: str,
    timeout_h: int,
    secrets: dict[str, str],
) -> str:
    env = {k: os.environ[k] for k in FORWARDED_ENV if os.environ.get(k)}
    with tempfile.TemporaryDirectory() as scratch:
        inject = []
        for i, part in enumerate(parts):
            path = Path(scratch) / f"{PART_PREFIX}{i:03d}"
            path.write_bytes(part)
            inject += ["--inject-file", f"{path}:{INPUT_DIR}/{path.name}"]
        entry = f"-m offramp.evals.serverless_job.main --bundle-dir {INPUT_DIR} --emit-stdout"
        flags = {
            "--name": name,
            "--image": image,
            "--parent-id": project_id,
            "--container-command": "python",
            "--args": f"{entry} --chunk-chars {CHUNK_CHARS}",
            "--platform": platform,
            "--preset": preset,
            "--disk-size": "20Gi",  # the documented minimum for a CPU-only job
            "--timeout": f"{timeout_h}h",
            "--format": "json",
        }
        # a private registry's MysteryBox secret; the project's own Container Registry needs none
        if registry_secret := os.environ.get("OFFRAMP_EVAL_REGISTRY_SECRET"):
            flags["--registry-secret"] = registry_secret
        args = ["ai", "job", "create", "--async", *(arg for pair in flags.items() for arg in pair)]
        for key, value in env.items():
            args += ["--env", f"{key}={value}"]
        for variable, selector in secrets.items():  # the selector names a secret; it is not the key
            args += ["--env-secret", f"{variable}={selector}"]
        out = _cli(*args, *inject)
    try:
        return str(json.loads(out)["resource_id"])
    except (ValueError, KeyError, TypeError):
        raise JobError(f"`nebius ai job create` printed no resource_id: {_head(out, 300)!r}") from None


def _cancel(job_id: str) -> None:
    """Best effort, and harmless on a job that has ended: stops a billing VM nobody is waiting for."""
    try:
        _cli("ai", "job", "cancel", job_id)
    except JobError as e:
        log.warning("could not cancel job %s: %s", job_id, e)


def _status(job_id: str, poll_s: float) -> dict[str, Any]:
    """`status` of `nebius ai job get`, retrying a failed read: one blip must not orphan a running job."""
    for attempt in range(GET_ATTEMPTS):
        try:
            return json.loads(_cli("ai", "job", "get", job_id, "--format", "json")).get("status") or {}
        except (JobError, ValueError, AttributeError) as e:
            if attempt == GET_ATTEMPTS - 1:
                why = e if isinstance(e, JobError) else "unreadable output"
                raise JobError(
                    f"could not read job {job_id} after {GET_ATTEMPTS} tries: {why}", ran=True
                ) from None
            time.sleep(poll_s * 2**attempt)
    raise AssertionError("unreachable")


def _failure_logs(job_id: str) -> str:
    """What the container printed before it failed (scrubbed, results lines dropped), '' if none."""
    try:
        logs = _cli("ai", "job", "logs", job_id, "--since", "24h")
    except JobError:
        return ""
    return _tail("\n".join(ln for ln in logs.splitlines() if RESULTS_TAG not in ln), LOG_TAIL_CHARS)


def _wait(job_id: str, *, timeout_h: int, poll_s: float) -> None:
    deadline = time.monotonic() + timeout_h * 3600 + 120  # the job's own --timeout should end it first
    ran, unknown = False, 0
    while True:
        status = _status(job_id, poll_s)
        state = status.get("state") if isinstance(status.get("state"), str) else None
        ran = ran or state == "RUNNING"
        if state == "COMPLETED":
            return
        if state in ENDED_BADLY:
            logs = _failure_logs(job_id)
            ran = ran or bool(status.get("started_at")) or bool(logs)  # a container that printed did run
            detail = _head(str((status.get("state_details") or {}).get("message", "")), 500)
            raise JobError(f"job {job_id} ended {state}: {detail}\n{logs}".rstrip(), ran=ran)
        unknown = 0 if state in IN_PROGRESS else unknown + 1
        if unknown >= UNKNOWN_POLLS:
            raise JobError(
                f"job {job_id}: `nebius ai job get` shows no recognised status.state ({state!r}); "
                "the job's output format is not what submit.py expects",
                ran=True,
            )
        if time.monotonic() > deadline:
            raise JobError(f"job {job_id} still {state} after {timeout_h}h", ran=True)
        time.sleep(poll_s)


def _results(job_id: str, *, timeout_h: int, poll_s: float) -> MatrixResults:
    for attempt in range(1, LOG_ATTEMPTS + 1):
        try:
            return decode_results(_cli("ai", "job", "logs", job_id, "--since", f"{timeout_h + 1}h"))
        except JobError as e:
            if attempt == LOG_ATTEMPTS:
                raise JobError(str(e), ran=True) from None  # the job ran: its results are what is lost
            time.sleep(poll_s)
    raise AssertionError("unreachable")


def _merge_cache(cache: Cache, entries: dict[str, dict[str, Any]]) -> None:
    """Add what the job wrote to the local cache. The keys come from the job's logs: check them first."""
    if bad := [key for key in entries if not CACHE_KEY.fullmatch(key)]:
        raise JobError(
            f"job returned cache entries under invalid keys ({bad[0][:80]!r}): none written", ran=True
        )
    for key, entry in entries.items():
        if cache.get(key) is None:
            cache.put(key, entry)


def _run_job(
    matrix: Matrix,
    cache_dir: str | Path | None,
    *,
    workers: int,
    run_id: str | None,
    image: str,
    project_id: str,
    platform: str,
    preset: str,
    timeout_h: int,
    poll_s: float,
    secrets: dict[str, str],
) -> MatrixResults:
    cache = TFClient(cache_dir=cache_dir).cache  # the same directory a local run would read and fill
    keys = needed_cache_keys(matrix, cache.dir, workers)  # ship what the matrix replays, not the whole cache
    parts = split_parts(pack_input(matrix, cache.dir, keys))
    slug = re.sub(r"[^a-z0-9-]+", "-", (run_id or uuid.uuid4().hex[:8]).lower()).strip("-")[:40] or "run"
    job_id = _create(
        parts,
        name=f"offramp-eval-{slug}",
        image=image,
        project_id=project_id,
        platform=platform,
        preset=preset,
        timeout_h=timeout_h,
        secrets=secrets,
    )
    log.info("eval matrix submitted as Nebius job %s", job_id)
    try:
        _wait(job_id, timeout_h=timeout_h, poll_s=poll_s)
        out = _results(job_id, timeout_h=timeout_h, poll_s=poll_s)
        _merge_cache(cache, out.cache)
    except JobError as e:
        _cancel(job_id)
        e.job_id = job_id
        raise
    except KeyboardInterrupt:
        _cancel(job_id)
        raise
    return out.model_copy(update={"ran_on": "job", "job_id": job_id})


def run_matrix_job(
    matrix: Matrix,
    *,
    mode: Literal["auto", "job", "local"] = "auto",
    cache_dir: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
    run_id: str | None = None,
    image: str | None = None,
    project_id: str | None = None,
    secret_id: str | None = None,
    fallback_secret_id: str | None = None,
    platform: str = JOB_PLATFORM,
    preset: str = JOB_PRESET,
    timeout_h: int = 2,
    poll_s: float = 10.0,
) -> MatrixResults:
    """Run `matrix` and return its results; `.ran_on`, `.job_id` and `.fallback_reason` say where.

    The run builds its own client from `matrix.budget` (Matrix(budget=remaining_caps(budget))), so
    `.budget` is its usage alone on either path: `absorb_usage(budget, results)` adds it to your Budget.
    `mode="job"` raises JobUnavailable (cannot try) or JobError; `auto` falls back to local only when the
    job could not be created or never started, and raises when the job had run (re-running would spend again).
    """
    image = image or os.environ.get("OFFRAMP_EVAL_IMAGE")
    project_id = project_id or os.environ.get("NEBIUS_PROJECT_ID")
    secret_id = secret_id or os.environ.get("NEBIUS_JOB_SECRET_ID")
    fallback_secret_id = fallback_secret_id or os.environ.get("NEBIUS_JOB_SECRET_ID_FALLBACK")

    def local(reason: str | None = None) -> MatrixResults:
        return run_local(matrix, cache_dir=cache_dir, workers=workers).model_copy(
            update={"fallback_reason": reason}
        )

    if mode == "local":
        return local()
    if reason := _unavailable(project_id, image, secret_id):
        if mode == "job":
            raise JobUnavailable(reason)
        log.info("eval matrix runs locally (%s): %s", NOT_CONFIGURED, reason)
        return local(NOT_CONFIGURED)
    assert image and project_id
    secrets = {}
    if matrix.live:
        secrets["NEBIUS_API_KEY"] = secret_id or ""  # (set: checked above)
        if fallback_secret_id:
            secrets["NEBIUS_API_KEY_FALLBACK"] = fallback_secret_id
    try:
        return _run_job(
            matrix,
            cache_dir,
            workers=workers,
            run_id=run_id,
            image=image,
            project_id=project_id,
            platform=platform,
            preset=preset,
            timeout_h=timeout_h,
            poll_s=poll_s,
            secrets=secrets,
        )
    except JobError as e:
        if mode == "job" or e.ran:
            if mode == "auto":
                log.error(
                    "Serverless Job %s failed after it had run; not re-running locally: %s", e.job_id, e
                )
            raise
        reason = f"Serverless Job path failed before the job ran: {e}"
        log.warning("%s; running the matrix locally", reason)
        return local(reason)


def main(argv: list[str] | None = None) -> int:
    """Smoke test: `python -m offramp.evals.serverless_job.submit --input m.json --output results.json`."""
    ap = argparse.ArgumentParser(
        prog="python -m offramp.evals.serverless_job.submit", description=main.__doc__
    )
    ap.add_argument("--input", type=Path, required=True, help="matrix JSON (Matrix.to_json)")
    ap.add_argument("--output", type=Path, required=True, help="results list JSON")
    ap.add_argument("--mode", choices=["auto", "job", "local"], default="auto")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--run-id")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = run_matrix_job(
        Matrix.from_json(args.input.read_text()), mode=args.mode, cache_dir=args.cache_dir, run_id=args.run_id
    )
    args.output.write_text(dumps(out.results, indent=2) + "\n")
    log.info("ran on %s (job %s) %s", out.ran_on, out.job_id, out.fallback_reason or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
