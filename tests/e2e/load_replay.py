"""Load test (PLAN Task 12): concurrent replay runs; the API's p95 latency must stay under 500 ms.

    uv run python tests/e2e/load_replay.py [--url http://127.0.0.1:8000] [--replay FILE] [--runs 5]

Without --url the API runs in-process (uvicorn on a free port, a temporary OFFRAMP_DATA_DIR) and replays
--replay FILE, default the recorded demo run (offramp/api/replay/demo-events.jsonl). With --url it loads a
running API, which replays its own recording. Each run: POST /api/runs?replay=demo, follow the event stream to
its `done` event while another client polls the run's snapshot and /api/health, then fetch the report.
Latency is measured on every request except the stream, which is checked for completeness instead.
Exit status 0 when every stream ended with `done` and p95 < 500 ms.
"""

from __future__ import annotations

import argparse
import socket
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

P95_LIMIT_S = 0.5


def _timed(client: httpx.Client, method: str, path: str, timings: list[float]) -> httpx.Response:
    start = time.perf_counter()
    r = client.request(method, path)
    timings.append(time.perf_counter() - start)
    r.raise_for_status()
    return r


def one_run(base: str, timings: list[float]) -> list[str]:
    """The event types one replay run streamed, in order."""
    with httpx.Client(base_url=base, timeout=30) as client, httpx.Client(base_url=base, timeout=30) as poller:
        run_id = _timed(client, "POST", "/api/runs?replay=demo", timings).json()["run_id"]
        stop = threading.Event()

        def poll() -> None:
            while not stop.is_set():
                _timed(poller, "GET", f"/api/runs/{run_id}", timings)
                _timed(poller, "GET", "/api/health", timings)
                stop.wait(0.05)

        polling = threading.Thread(target=poll)
        polling.start()
        types = []
        try:
            with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
                types = [line[7:] for line in stream.iter_lines() if line.startswith("event: ")]
        finally:
            stop.set()
            polling.join()
        _timed(client, "GET", f"/api/runs/{run_id}/report", timings)
        return types


def load(base: str, runs: int) -> tuple[list[float], list[list[str]], float]:
    timings: list[float] = []
    streams: list[list[str]] = [[] for _ in range(runs)]

    def go(i: int) -> None:
        streams[i] = one_run(base, timings)

    start = time.perf_counter()
    threads = [threading.Thread(target=go, args=(i,)) for i in range(runs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return timings, streams, time.perf_counter() - start


def serve_in_process(replay: Path, root: Path):
    """The API on a free local port; returns (base url, server) — set server.should_exit to stop it."""
    import uvicorn

    from offramp.api.app import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(root, replay_file=replay), host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the API did not start in 15 s")
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", server


def main(argv: list[str] | None = None) -> int:
    from offramp.api.app import REPLAY_FILE

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", help="a running API (default: start one in-process)")
    parser.add_argument("--replay", type=Path, default=REPLAY_FILE, help="recording for the in-process API")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args(argv)

    server = None
    with tempfile.TemporaryDirectory() as root:
        if args.url:
            base = args.url.rstrip("/")
        elif not args.replay.exists():
            print(f"no recorded run yet at {args.replay}: pass --replay FILE or --url", file=sys.stderr)
            return 2
        else:
            base, server = serve_in_process(args.replay, Path(root))
        try:
            timings, streams, wall = load(base, args.runs)
        finally:
            if server is not None:
                server.should_exit = True
    p50 = statistics.median(timings)
    p95 = statistics.quantiles(timings, n=20, method="inclusive")[18]
    complete = sum(bool(s) and s[-1] == "done" for s in streams)
    print(f"{args.runs} concurrent replays in {wall:.1f} s: {complete}/{args.runs} streams ended with `done`")
    print(f"{len(timings)} requests: p50 {p50 * 1000:.0f} ms, p95 {p95 * 1000:.0f} ms (limit 500 ms)")
    return 0 if complete == args.runs and p95 < P95_LIMIT_S else 1


if __name__ == "__main__":
    sys.exit(main())
