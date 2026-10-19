"""Offramp's HTTP API (Global Constraint 13): the web dashboard's only contract with the backend.

    POST /api/runs                  body RunConfig -> {"run_id"}; ?replay=demo replays the recorded demo run
    GET  /api/runs/{run_id}         RunSnapshot, folded from the run's events
    GET  /api/runs/{run_id}/events  text/event-stream: `id: <seq>`, `event: <type>`, `data: <RunEvent json>`;
                                    honours Last-Event-ID; ends after `done` or `error`
    GET  /api/runs/{run_id}/report  JSON download {snapshot, pr_body_markdown, free_text_arg_exemptions}
    GET  /api/health                {"ok": true, "tf_configured": bool}

Runs wait on a queue that one background thread works through, so the per-run caps stay per run and two
runs never race for the same credit. A replay (Global Constraint 14) never calls Token Factory: a thread
appends the recorded events at 4x speed, each recorded gap capped at 1.5 s first. POST /api/runs is rate
limited per IP (OFFRAMP_RATE_LIMIT_PER_MIN, default 30); CORS allows OFFRAMP_WEB_ORIGIN (comma-separated,
default http://localhost:3000). A live run imports the target repo's code on this host, so the API starts one
only when OFFRAMP_ALLOW_LIVE_RUNS=1 (403 otherwise); replays and cache-only runs need no opt-in.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import shutil
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from offramp import db, pipeline
from offramp.schemas import RunConfig, RunEvent

REPLAY_FILE = Path(__file__).parent / "replay" / "demo-events.jsonl"
REPLAY_SPEED, REPLAY_GAP_CAP_S = 4.0, 1.5
NO_RECORDING = (
    "no recorded run yet: the demo replay is produced by a real run (`offramp export-replay <run_id>`)"
)
LIVE_OPT_IN = "OFFRAMP_ALLOW_LIVE_RUNS"
LIVE_REFUSED = (
    "live runs are off on this server: a live run imports the target repo's code here and spends Token "
    "Factory credit. Watch the replay (?replay=demo), start a cache-only run (live=false), or set "
    "OFFRAMP_ALLOW_LIVE_RUNS=1 on a server you trust with the repos it is given."
)
DEFAULT_ORIGIN = "http://localhost:3000"
POLL_S = 0.2  # how often an event stream looks for new events
KEEPALIVE_S = 15.0  # an SSE comment after this long without events, so proxies keep the stream open
REPLAY_KEEP = 20  # replay runs kept (memory, disk, SQLite); the oldest finished ones go first
HTTPS_URL = re.compile(r"https://[^\s/@]+/\S+")  # no user:password@ part: credentials never enter a run


def tf_configured() -> bool:
    return bool(os.environ.get("NEBIUS_API_KEY") or os.environ.get("NEBIUS_API_KEY_FALLBACK"))


def replay_delays(events: Sequence[RunEvent]) -> list[float]:
    """Seconds to wait before each recorded event: the recorded gap, capped at 1.5 s, at 4x speed (GC14)."""
    if not events:
        return []
    gaps = [(b.ts - a.ts).total_seconds() for a, b in zip(events, events[1:], strict=False)]
    return [0.0] + [min(max(g, 0.0), REPLAY_GAP_CAP_S) / REPLAY_SPEED for g in gaps]


def pr_body_file(replay_file: Path) -> Path:
    """Where `offramp export-replay` puts the recorded run's PR body, next to its events."""
    return replay_file.with_suffix(".pr.md")


def pr_body(replay_file: Path) -> str | None:
    path = pr_body_file(replay_file)
    return path.read_text(encoding="utf-8") if path.exists() else None


def load_recording(path: Path) -> list[RunEvent] | None:
    """The recorded demo run, or None when there is none yet: no file, no events, or fixture data."""
    if not path.exists():
        return None
    try:
        flagged = json.loads(path.read_text(encoding="utf-8")).get("fixture") is True
    except (ValueError, AttributeError):  # JSON Lines, or a bare array
        flagged = False
    events = pipeline.read_events(path)
    if flagged or not events or any(e.run_id.startswith("fixture-") for e in events):
        return None
    return events


def config_problem(config: RunConfig) -> str | None:
    """Why the API refuses this config, if it does. Only https repos: a run clones whatever it is given."""
    if not HTTPS_URL.fullmatch(config.repo_url):
        return "repo_url must be an https:// URL of a git repository, without credentials in it"
    if not (0.5 <= config.threshold <= 1 and 0.5 <= config.reasoning_threshold <= 1):
        return "thresholds must be between 0.5 and 1"
    if not 0 <= config.max_iterations <= 3:
        return "max_iterations must be between 0 and 3"
    if not config.tiers:
        return "pick at least one tier"
    return None


class RateLimit:
    """At most `per_minute` hits per key in any 60 s window. ponytail: in-process memory, one API process."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self.hits[key]
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True


def _sse(e: RunEvent) -> str:
    return f"id: {e.seq}\nevent: {e.type}\ndata: {e.model_dump_json()}\n\n"


def create_app(
    root: Path | None = None,
    *,
    replay_file: Path = REPLAY_FILE,
    execute: Callable[[pipeline.RunLog], object] | None = None,
) -> FastAPI:
    """The API over OFFRAMP_DATA_DIR (`root`). `execute(log)` runs a queued run (default: the pipeline)."""
    root = Path(root or pipeline.data_dir())
    engine = db.init_db(db.engine(root / "offramp.db"))
    execute = execute or pipeline.execute_run
    logs: dict[str, pipeline.RunLog] = {}
    active: set[str] = set()  # queued, running or replaying in this process
    jobs: queue.Queue[str | None] = queue.Queue()
    limit = RateLimit(int(os.environ.get("OFFRAMP_RATE_LIMIT_PER_MIN") or 30))

    def work() -> None:
        while (run_id := jobs.get()) is not None:
            log = logs[run_id]
            try:
                execute(log)
            except Exception as e:  # the pipeline reports its own failures; this is the last line
                log.emit("error", {"stage": log.stage.value, "reason": f"{type(e).__name__}: {e}"[:500]})
            finally:
                active.discard(run_id)

    def play(log: pipeline.RunLog, events: list[RunEvent]) -> None:
        try:
            for delay, e in zip(replay_delays(events), events, strict=True):
                time.sleep(delay)
                log.add(e)
        finally:
            active.discard(log.run_id)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        worker = threading.Thread(target=work, name="offramp-runs", daemon=True)
        worker.start()
        yield
        jobs.put(None)

    app = FastAPI(title="Offramp API", lifespan=lifespan)
    origins = os.environ.get("OFFRAMP_WEB_ORIGIN") or DEFAULT_ORIGIN
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip().rstrip("/") for o in origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "last-event-id"],
    )

    def evict_replays() -> None:
        """Keep the newest REPLAY_KEEP - 1 finished replay runs before adding one: replays are free to start,
        so their logs, directories and rows must not pile up."""
        runs = root / "runs"
        ids = {p.name for p in runs.glob("replay-*")} | {k for k in logs if k.startswith("replay-")}
        done = sorted(i for i in ids if i not in active)  # ids start with a timestamp: oldest first
        for run_id in done[: max(0, len(ids) - REPLAY_KEEP + 1)]:
            logs.pop(run_id, None)
            shutil.rmtree(runs / run_id, ignore_errors=True)
            pipeline.forget(engine, run_id)

    def run_log(run_id: str) -> pipeline.RunLog:
        log = logs.get(run_id)
        if log is None and pipeline.RUN_ID.fullmatch(run_id):
            candidate = pipeline.RunLog(root / "runs" / run_id, run_id)
            if candidate.events:
                log = logs[run_id] = candidate
        if log is None:
            raise HTTPException(404, f"no run {run_id}")
        return log

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True, "tf_configured": tf_configured()}

    @app.post("/api/runs")
    def start_run(
        request: Request, config: Annotated[RunConfig | None, Body()] = None, replay: str | None = None
    ) -> dict[str, str]:
        if not limit.allow(request.client.host if request.client else "unknown"):
            too_many = f"at most {limit.per_minute} new runs per minute per address"
            raise HTTPException(429, too_many, headers={"Retry-After": "60"})
        if replay is not None:
            if replay != "demo":
                raise HTTPException(404, "the only replay is ?replay=demo")
            events = load_recording(replay_file)
            if events is None:
                raise HTTPException(404, NO_RECORDING)
            evict_replays()
            run_id = pipeline.new_run_id("replay-")
            log = logs[run_id] = pipeline.RunLog(root / "runs" / run_id, run_id, engine=engine)
            active.add(run_id)
            if (body := pr_body(replay_file)) is not None:  # the recorded run's PR body, for the report
                log.dir.mkdir(parents=True, exist_ok=True)
                (log.dir / "pr.json").write_text(json.dumps({"title": None, "body": body, "url": None}))
            threading.Thread(target=play, args=(log, events), name=f"replay-{run_id}", daemon=True).start()
            return {"run_id": run_id}
        if config is None:
            raise HTTPException(422, "the body must be a RunConfig")
        if problem := config_problem(config):
            raise HTTPException(422, problem)
        if config.live and os.environ.get(LIVE_OPT_IN) != "1":
            raise HTTPException(403, LIVE_REFUSED)
        if config.live and not tf_configured():
            raise HTTPException(503, "Token Factory is not configured on this server; watch the replay")
        log = pipeline.new_run(config, root, engine=engine)
        logs[log.run_id] = log
        active.add(log.run_id)
        jobs.put(log.run_id)
        return {"run_id": log.run_id}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return pipeline.snapshot(run_id, list(run_log(run_id).events)).model_dump(mode="json")

    @app.get("/api/runs/{run_id}/report")
    def get_report(run_id: str) -> JSONResponse:
        log = run_log(run_id)
        return JSONResponse(
            pipeline.report(log.dir, run_id, list(log.events)),
            headers={"Content-Disposition": f'attachment; filename="offramp-{run_id}.json"'},
        )

    @app.get("/api/runs/{run_id}/events")
    async def get_events(
        run_id: str, request: Request, last_event_id: str | None = Header(None)
    ) -> StreamingResponse:
        log = run_log(run_id)
        after = int(last_event_id) if last_event_id and last_event_id.strip().isdigit() else -1

        async def stream():
            sent, quiet = 0, 0.0  # events of log.events already looked at; seconds since the last write
            while True:
                batch = log.events[sent:]
                sent += len(batch)
                for e in batch:
                    if e.seq > after:
                        yield _sse(e)
                caught_up = sent == len(log.events)
                ended = log.events and log.events[-1].type in pipeline.TERMINAL
                if caught_up and (ended or run_id not in active):
                    return
                if batch:
                    quiet = 0.0
                elif quiet >= KEEPALIVE_S:
                    quiet = 0.0
                    yield ": keepalive\n\n"
                if await request.is_disconnected():
                    return
                await asyncio.sleep(POLL_S)
                quiet += POLL_S

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.state.logs, app.state.active, app.state.engine = logs, active, engine
    return app
