"""Job entrypoint: `python -m offramp.evals.serverless_job.main --input matrix.json --output results.json`.

The same command runs in the Nebius Serverless Job container (`--bundle-dir` is where `--inject-file` put the
packed matrix + cache, `--emit-stdout` is how the results leave) and on a laptop (`--input`, `--cache-dir`).
Provider keys come from the environment (NEBIUS_API_KEY, ...), never from the matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from offramp.evals.serverless_job.matrix import (
    DEFAULT_WORKERS,
    Matrix,
    MatrixResults,
    budget_snapshot,
    dumps,
    make_client,
    run_matrix,
)
from offramp.evals.serverless_job.transport import CHUNK_CHARS, encode_results, unpack_input


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m offramp.evals.serverless_job.main", description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="matrix JSON (Matrix.to_json)")
    source.add_argument("--bundle-dir", type=Path, help="directory of part-NNN files: packed matrix + cache")
    ap.add_argument("--output", type=Path, help="write the results list (sorted entries, JSON) to this file")
    ap.add_argument(
        "--emit-stdout", action="store_true", help="print the OFFRAMP_RESULTS line(s): the job's way out"
    )
    ap.add_argument(
        "--cache-dir", type=Path, help="Token Factory cache (default: as TFClient; with --input only)"
    )
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    args = ap.parse_args(argv)
    if not (args.output or args.emit_stdout):
        ap.error("nowhere to put the results: give --output and/or --emit-stdout")
    if args.bundle_dir and args.cache_dir:
        ap.error("--cache-dir is the bundle's own cache with --bundle-dir")

    with tempfile.TemporaryDirectory() as scratch:
        if args.bundle_dir:
            matrix, cache_dir = unpack_input(args.bundle_dir, Path(scratch))
        else:
            matrix, cache_dir = Matrix.from_json(args.input.read_text()), args.cache_dir
        client = make_client(matrix, cache_dir)
        before = {p.name for p in client.cache.dir.glob("*.json")}
        results = run_matrix(matrix, client, workers=args.workers)
        written = {  # what this run added to the cache: the submitter merges it into its own
            p.stem: json.loads(p.read_text()) for p in client.cache.dir.glob("*.json") if p.name not in before
        }
    out = MatrixResults(results=results, cache=written, budget=budget_snapshot(client))
    if args.output:
        args.output.write_text(dumps(results, indent=2) + "\n")
    if args.emit_stdout:
        for line in encode_results(out, args.chunk_chars):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
