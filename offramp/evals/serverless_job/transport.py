"""How a matrix gets into a job and its results get out. Only the submitter and the entrypoint use this.

In:  one tar.gz holding `matrix.json` and `cache/<key>.json` (the Token Factory cache entries the matrix
     replays), cut into `part-NNN` files that `nebius ai job create --inject-file` places in the container
     (64 KiB each).
Out: stdout lines `OFFRAMP_RESULTS <urlsafe-base64 of gzip of JSON>`: one line when it fits, else
     `OFFRAMP_RESULTS <i>/<n> <chunk>` lines, because job logs are the only channel and a long line can
     be split or lost. The JSON is a `MatrixResults`: the entries, the cache entries the job wrote, its
     budget.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import re
import tarfile
import zlib
from collections.abc import Iterable
from pathlib import Path

from offramp.evals.serverless_job.matrix import Matrix, MatrixResults

RESULTS_TAG = "OFFRAMP_RESULTS"
CHUNK_CHARS = 8_000  # below the 16 KiB at which container runtimes split a log line
PART_BYTES = 48 * 1024  # --inject-file allows 64 KiB per file
MAX_PARTS = 32  # ponytail: ~1.5 MB of compressed matrix + cache; a volume is the way past it (needs a bucket)
PART_PREFIX = "part-"

_CHUNK = re.compile(rf"{RESULTS_TAG} (?:(\d+)/(\d+) )?([A-Za-z0-9_=-]*)\s*$")


class JobError(RuntimeError):
    """The Serverless Job path did not produce results (submit failed, job failed, results lost).

    `ran` is True when the job may have run and spent Token Factory credit; a caller must not then re-run
    the matrix elsewhere on its own. `job_id` is set once the job exists."""

    def __init__(self, message: str, *, ran: bool = False, job_id: str | None = None):
        super().__init__(message)
        self.ran = ran
        self.job_id = job_id


class JobUnavailable(JobError):
    """The Serverless Job path cannot be tried at all: no CLI, no project, no image, input too large."""


def pack_input(matrix: Matrix, cache_dir: Path, cache_keys: Iterable[str]) -> bytes:
    """tar.gz of the matrix and the cache entries `cache_keys` (not the whole cache directory)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:

        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add("matrix.json", matrix.to_json().encode())
        for key in sorted(cache_keys):
            add(f"cache/{key}.json", (cache_dir / f"{key}.json").read_bytes())
    return buf.getvalue()


def split_parts(blob: bytes) -> list[bytes]:
    parts = [blob[i : i + PART_BYTES] for i in range(0, len(blob), PART_BYTES)]
    if len(parts) > MAX_PARTS:
        raise JobUnavailable(
            f"job input is too large to inject ({len(blob)} bytes > {MAX_PARTS} x {PART_BYTES}): "
            "point cache_dir at a smaller cache or run locally"
        )
    return parts


def unpack_input(parts_dir: Path, dest: Path) -> tuple[Matrix, Path]:
    """The matrix and the cache directory (created if the input held no cache) inside `dest`."""
    blob = b"".join(p.read_bytes() for p in sorted(parts_dir.glob(f"{PART_PREFIX}*")))
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    (dest / "cache").mkdir(exist_ok=True)
    return Matrix.from_json((dest / "matrix.json").read_text()), dest / "cache"


def encode_results(out: MatrixResults, chunk_chars: int = CHUNK_CHARS) -> list[str]:
    text = json.dumps(out.model_dump(mode="json"), ensure_ascii=False)
    blob = base64.urlsafe_b64encode(gzip.compress(text.encode(), mtime=0)).decode()
    chunks = [blob[i : i + chunk_chars] for i in range(0, len(blob), chunk_chars)]
    if len(chunks) == 1:
        return [f"{RESULTS_TAG} {chunks[0]}"]
    return [f"{RESULTS_TAG} {i}/{len(chunks)} {chunk}" for i, chunk in enumerate(chunks, 1)]


def decode_results(logs: str) -> MatrixResults:
    """Reassemble the `OFFRAMP_RESULTS` lines found anywhere in `logs` (any order, any line prefix)."""
    whole, chunks, total = None, {}, 0
    for line in logs.splitlines():
        if (found := _CHUNK.search(line)) is None:
            continue
        i, n, data = found.groups()
        if i is None:
            whole = data
        else:
            chunks[int(i)], total = data, int(n)
    if whole is None and not chunks:
        raise JobError(f"no {RESULTS_TAG} line in the job logs (truncated, or the job printed none)")
    if whole is None:
        if missing := sorted(set(range(1, total + 1)) - chunks.keys()):
            raise JobError(f"job logs are missing {RESULTS_TAG} chunk(s) {missing} of {total}")
        whole = "".join(chunks[i] for i in range(1, total + 1))
    try:
        return MatrixResults.model_validate(json.loads(gzip.decompress(base64.urlsafe_b64decode(whole))))
    except (ValueError, OSError, EOFError, zlib.error) as e:  # bad base64, gzip or JSON, wrong shape
        raise JobError(f"{RESULTS_TAG} payload is corrupt: {type(e).__name__}: {e}") from e
