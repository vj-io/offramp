"""Run the eval matrix as a Nebius Serverless Job or locally: same entries either way (PRD §6.5, §12.4).

`run_matrix_job(matrix, mode="auto")` submits with the `nebius` CLI when it is on PATH and NEBIUS_PROJECT_ID
and OFFRAMP_EVAL_IMAGE (an image built from Dockerfile.eval, pushed where the project can pull it) are set;
otherwise, or when the job fails, it runs the same `run_matrix` in-process (`mode="job"` raises instead).
The job gets the matrix and the Token Factory cache as `--inject-file` parts and the provider key as `--env`
(from this process's environment, only when the matrix is live, never in a log). It hands back its results
and the cache entries it wrote on stdout (transport.py); those are merged into the local cache, so a job run
fills the same cache a local run would.

Verbs and flags: docs.nebius.com/cli/reference/ai/job/{create,get,logs,cancel}. Checked against those docs:
the flags below, `create --async --format json` printing the Operation with `resource_id`, and the job state
names (nebius/api ai/v1/job.proto). Not checked, no credentials yet: that a real job accepts these flags as
used (--args split on spaces, several --inject-file), the CPU platform/preset names, and the log line format
(parsing tolerates a prefix).
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
from typing import Literal

from offramp.evals.serverless_job.matrix import (
    DEFAULT_WORKERS,
    SECRET_ENV,
    Matrix,
    MatrixResults,
    budget_snapshot,
    dumps,
    make_client,
    run_matrix,
    scrub,
)
from offramp.evals.serverless_job.transport import (
    CHUNK_CHARS,
    PART_PREFIX,
    JobError,
    JobUnavailable,
    decode_results,
    pack_input,
    split_parts,
)
from offramp.tf import LLM, TFClient

__all__ = ["JobError", "JobUnavailable", "redact", "run_local", "run_matrix_job"]

log = logging.getLogger(__name__)

INPUT_DIR = "/offramp/input"  # where --inject-file puts the parts inside the container
JOB_PLATFORM, JOB_PRESET = "cpu-d3", "4vcpu-16gb"  # the work is network-bound: a CPU VM, not the GPU default
# Config that must match the submitter or the job's cache keys and prices differ from a local run's.
FORWARDED_ENV = ("NEBIUS_BASE_URL", "OFFRAMP_MODEL_LIGHTNING", "OFFRAMP_MODEL_SUPER", "OFFRAMP_MODEL_ULTRA")
ENDED_BADLY = {"FAILED", "CANCELLED", "CANCELLING", "DELETING", "ERROR"}  # JobStatus.State; COMPLETED is good
LOG_ATTEMPTS = 5  # the job's logs can lag its COMPLETED state
CLI_TIMEOUT_S = 300
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.I)


def redact(argv: Sequence[str]) -> str:
    """A command line safe to print: secret `--env` values and registry credentials replaced by ***."""
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


def _cli(*args: str) -> str:
    cmd = [shutil.which("nebius") or "nebius", *args]
    log.info("nebius %s", redact(args))
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=CLI_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        # `from None`, not `from e`: the exception carries the argv, key included
        raise JobError(f"nebius {' '.join(args[:3])}: {type(e).__name__}") from None
    if done.returncode != 0:  # the full command is in the log, redacted; the error only needs the verb
        raise JobError(
            scrub(f"nebius {' '.join(args[:3])} exited {done.returncode}: {done.stderr.strip()[:500]}")
        )
    return done.stdout


def run_local(
    matrix: Matrix,
    *,
    cache_dir: str | Path | None = None,
    client: LLM | None = None,
    workers: int = DEFAULT_WORKERS,
) -> MatrixResults:
    """The in-process worker. `client` shares the pipeline's budget; else one is built from the matrix."""
    client = client or make_client(matrix, cache_dir)
    return MatrixResults(results=run_matrix(matrix, client, workers=workers), budget=budget_snapshot(client))


def _unavailable(matrix: Matrix, project_id: str | None, image: str | None) -> str | None:
    if not shutil.which("nebius"):
        return "the nebius CLI is not on PATH"
    if not project_id:
        return "NEBIUS_PROJECT_ID is not set"
    if not image:
        return "OFFRAMP_EVAL_IMAGE is not set"
    if matrix.live and not os.environ.get("NEBIUS_API_KEY"):
        return "NEBIUS_API_KEY is not set (the job's Token Factory calls need it)"
    return None


def _create(
    matrix: Matrix,
    parts: list[bytes],
    *,
    name: str,
    image: str,
    project_id: str,
    platform: str,
    preset: str,
    timeout_h: int,
) -> str:
    env = {k: os.environ[k] for k in FORWARDED_ENV if os.environ.get(k)}
    if matrix.live:  # a replay needs no key: the job does not get one
        env |= {k: os.environ[k] for k in SECRET_ENV if os.environ.get(k)}
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
        out = _cli(*args, *inject)
    try:
        return str(json.loads(out)["resource_id"])
    except (ValueError, KeyError, TypeError):
        raise JobError(f"`nebius ai job create` printed no resource_id: {scrub(out)[:300]!r}") from None


def _cancel(job_id: str) -> None:
    try:
        _cli("ai", "job", "cancel", job_id)
    except JobError as e:  # nothing more to do about it, and the job's own --timeout still ends it
        log.warning("could not cancel job %s: %s", job_id, e)


def _wait(job_id: str, *, timeout_h: int, poll_s: float) -> None:
    deadline = time.monotonic() + timeout_h * 3600 + 120  # the job's own --timeout should end it first
    try:
        while True:
            try:
                status = json.loads(_cli("ai", "job", "get", job_id, "--format", "json")).get("status") or {}
            except (ValueError, AttributeError):
                raise JobError(f"unreadable output from `nebius ai job get {job_id}`") from None
            state = status.get("state")
            if state == "COMPLETED":
                return
            if state in ENDED_BADLY:
                detail = (status.get("state_details") or {}).get("message", "")
                raise JobError(f"job {job_id} ended {state}: {detail}")
            if time.monotonic() > deadline:
                _cancel(job_id)  # stop paying for a job nobody waits for
                raise JobError(f"job {job_id} still {state} after {timeout_h}h: cancelled")
            time.sleep(poll_s)
    except KeyboardInterrupt:
        _cancel(job_id)
        raise


def _results(job_id: str, *, timeout_h: int, poll_s: float) -> MatrixResults:
    for attempt in range(1, LOG_ATTEMPTS + 1):
        try:
            return decode_results(_cli("ai", "job", "logs", job_id, "--since", f"{timeout_h + 1}h"))
        except JobError:
            if attempt == LOG_ATTEMPTS:
                raise
            time.sleep(poll_s)


def _run_job(
    matrix: Matrix,
    cache_dir: str | Path | None,
    *,
    run_id: str | None,
    image: str,
    project_id: str,
    platform: str,
    preset: str,
    timeout_h: int,
    poll_s: float,
) -> MatrixResults:
    cache = TFClient(cache_dir=cache_dir).cache  # the same directory a local run would read and fill
    parts = split_parts(pack_input(matrix, cache.dir))
    slug = re.sub(r"[^a-z0-9-]+", "-", (run_id or uuid.uuid4().hex[:8]).lower()).strip("-")[:40] or "run"
    job_id = _create(
        matrix,
        parts,
        name=f"offramp-eval-{slug}",
        image=image,
        project_id=project_id,
        platform=platform,
        preset=preset,
        timeout_h=timeout_h,
    )
    log.info("eval matrix submitted as Nebius job %s", job_id)
    _wait(job_id, timeout_h=timeout_h, poll_s=poll_s)
    out = _results(job_id, timeout_h=timeout_h, poll_s=poll_s)
    for key, entry in out.cache.items():
        if cache.get(key) is None:
            cache.put(key, entry)
    return out


def run_matrix_job(
    matrix: Matrix,
    *,
    mode: Literal["auto", "job", "local"] = "auto",
    cache_dir: str | Path | None = None,
    client: LLM | None = None,
    workers: int = DEFAULT_WORKERS,
    run_id: str | None = None,
    image: str | None = None,
    project_id: str | None = None,
    platform: str = JOB_PLATFORM,
    preset: str = JOB_PRESET,
    timeout_h: int = 2,
    poll_s: float = 10.0,
) -> MatrixResults:
    """`client` and `workers` only apply to a local run (a job builds its own client from the matrix).

    A job that is cancelled after `timeout_h` may already have spent Token Factory credit that a fallback run
    pays again; the cache entries it wrote are lost with it.
    """
    image = image or os.environ.get("OFFRAMP_EVAL_IMAGE")
    project_id = project_id or os.environ.get("NEBIUS_PROJECT_ID")
    if mode == "local":
        return run_local(matrix, cache_dir=cache_dir, client=client, workers=workers)
    if reason := _unavailable(matrix, project_id, image):
        if mode == "job":
            raise JobUnavailable(reason)
        log.info("eval matrix runs locally: %s", reason)
        return run_local(matrix, cache_dir=cache_dir, client=client, workers=workers)
    assert image and project_id
    try:
        return _run_job(
            matrix,
            cache_dir,
            run_id=run_id,
            image=image,
            project_id=project_id,
            platform=platform,
            preset=preset,
            timeout_h=timeout_h,
            poll_s=poll_s,
        )
    except JobError as e:
        if mode == "job":
            raise
        log.warning("Serverless Job path failed (%s); running the matrix locally", e)
        return run_local(matrix, cache_dir=cache_dir, client=client, workers=workers)


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
