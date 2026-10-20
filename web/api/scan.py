"""POST /api/scan {"github_url": "https://github.com/<owner>/<repo>[/tree/<branch>]"}: the `offramp scan` table.

A Vercel Python function. It runs Offramp's own scanner (offramp/github_scan.py) from web/pyvendor, a byte-for-byte
copy of offramp/ written by `make vendor-scan` and checked by tests/unit/test_scan_vendor.py. Static only: it reads
the repo's .py files from GitHub's tarball, never imports or runs them, makes no Token Factory call and needs no key.

Each scan runs in a forked child with a hard bound: its address space may grow by MEMORY_LIMIT (RLIMIT_AS, enforced
on Linux, which is what Vercel runs) and it is killed at KILL_AFTER_S. A child that hits either gets a JSON 413.
"""

import json
import os
import resource
import select
import shutil
import signal
import sys
import tempfile
import time
import traceback
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pyvendor"))

from offramp.github_scan import TIME_LIMIT_S, ScanError, parse_github_url, scan_github  # noqa: E402

MEMORY_LIMIT = 1 << 30  # bytes the scan child may add to the address space it starts with
KILL_AFTER_S = TIME_LIMIT_S + 1.5  # the scan's own deadline is TIME_LIMIT_S; this catches what it cannot tick
CLI = "uvx --from git+https://github.com/vj-io/offramp offramp scan ."
TOO_BIG = {
    "error": "too_large_or_complex",
    "message": f"This repo is too large or complex for the free hosted scan; run `{CLI}` locally.",
}
INTERNAL = {"error": "internal", "message": "The scan failed on our side. Try again, or run the scan locally."}

RATE_LIMIT, WINDOW_S = 6, 60.0  # scans per client IP per window
MAX_BODY = 4096
# ponytail: best-effort limit kept in this instance's memory. Vercel runs several instances and recycles them, so a
# client can get past it; move to the Vercel Firewall's rate limiting or a shared store if abuse shows up.
# Refused requests count too, so bad input cannot be used to probe without limit.
_hits: dict[str, deque[float]] = defaultdict(deque)
MAX_TRACKED_IPS = 10_000


def limited(ip: str, now: float) -> bool:
    if len(_hits) > MAX_TRACKED_IPS:  # forget idle addresses, so the table cannot grow for the instance's life
        for stale in [k for k, v in _hits.items() if not v or now - v[-1] > WINDOW_S]:
            del _hits[stale]
    hits = _hits[ip]
    while hits and now - hits[0] > WINDOW_S:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        return True
    hits.append(now)
    return False


class handler(BaseHTTPRequestHandler):  # noqa: N801 - the name Vercel's Python runtime loads
    def do_POST(self):  # noqa: N802
        ip = self.headers.get("x-real-ip") or (self.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if limited(ip or self.client_address[0], time.monotonic()):
            return self.send(429, "rate_limited", f"At most {RATE_LIMIT} scans a minute from one address.")
        try:
            length = int(self.headers.get("content-length") or 0)
            if not 0 < length <= MAX_BODY:
                raise ValueError
            url = json.loads(self.rfile.read(length)).get("github_url")
        except (ValueError, AttributeError):
            return self.send(400, "bad_request", 'Send JSON: {"github_url": "https://github.com/<owner>/<repo>"}.')
        try:
            parse_github_url(url)  # refuse a bad URL before forking
        except ScanError as exc:
            return self.send(exc.status, exc.code, exc.message)
        status, body = scan_bounded(url)
        self.send(status, body=body)

    def send(self, status: int, error: str = "", message: str = "", body: dict | None = None):
        data = json.dumps(body if body is not None else {"error": error, "message": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def scan_bounded(url: str) -> tuple[int, dict]:
    """(status, JSON body) of scan_github(url) run in a child process under MEMORY_LIMIT and KILL_AFTER_S."""
    work = tempfile.mkdtemp(prefix="offramp-req-")  # the child's temp files, removed even if it is killed
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # the child: never returns
        code = 1
        try:
            os.close(read)
            tempfile.tempdir = work
            _limit_memory()
            try:
                out = [200, scan_github(url)]
            except ScanError as exc:
                out = [exc.status, {"error": exc.code, "message": exc.message}]
            except MemoryError:
                out = [413, TOO_BIG]
            except Exception:  # the traceback goes to the function log, never to the client
                traceback.print_exc()
                out = [500, INTERNAL]
            with os.fdopen(write, "wb") as pipe:
                pipe.write(json.dumps(out).encode())
            code = 0
        finally:
            os._exit(code)  # no atexit handlers, no buffers of the parent's flushed twice
    os.close(write)
    chunks, finished, kill_at = [], False, time.monotonic() + KILL_AFTER_S
    try:
        while (left := kill_at - time.monotonic()) > 0 and select.select([read], [], [], left)[0]:
            if not (chunk := os.read(read, 1 << 16)):
                finished = True
                break
            chunks.append(chunk)
    finally:
        os.close(read)
        if not finished:
            os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        shutil.rmtree(work, ignore_errors=True)
    try:
        status, body = json.loads(b"".join(chunks))
    except ValueError:  # killed at the deadline, or died on the memory limit before it could answer
        return 413, TOO_BIG
    return status, body


def _limit_memory() -> None:
    """RLIMIT_AS = the address space mapped now + MEMORY_LIMIT. Linux enforces it; without /proc (macOS) no limit."""
    try:
        with open("/proc/self/statm") as statm:
            mapped = int(statm.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        return
    resource.setrlimit(resource.RLIMIT_AS, (mapped + MEMORY_LIMIT, mapped + MEMORY_LIMIT))
