"""POST /api/scan {"github_url": "https://github.com/<owner>/<repo>[/tree/<branch>]"}: the `offramp scan` table.

A Vercel Python function. It runs Offramp's own scanner (offramp/github_scan.py) from web/pyvendor, a byte-for-byte
copy of offramp/ written by `make vendor-scan` and checked by tests/unit/test_scan_vendor.py. Static only: it reads
the repo's .py files from GitHub's tarball, never imports or runs them, makes no Token Factory call and needs no key.

Each scan runs in a forked child with a hard bound: its address space may grow by MEMORY_LIMIT (RLIMIT_AS, enforced
on Linux, which is what Vercel runs), it is killed at KILL_AFTER_S or once its answer passes MAX_ANSWER_BYTES, and a
child that hits any of these gets a JSON 413. One scan runs per instance at a time; another request gets a 503.
"""

import errno
import json
import os
import resource
import select
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pyvendor"))

from offramp.github_scan import TIME_LIMIT_S, ScanError, parse_github_url, scan_github  # noqa: E402

MEMORY_LIMIT = 1 << 30  # bytes the scan child may add to the address space it starts with
MAX_ANSWER_BYTES = 4_000_000  # the child's JSON answer; Vercel refuses responses over 4.5 MB
KILL_AFTER_S = TIME_LIMIT_S + 1.5  # the scan's own deadline is TIME_LIMIT_S; this catches what it cannot tick
CLI = "uvx --from git+https://github.com/vj-io/offramp offramp scan ."
TOO_BIG = {
    "error": "too_large_or_complex",
    "message": f"This repo is too large or complex for the free hosted scan; run `{CLI}` locally.",
}
INTERNAL = {"error": "internal", "message": "The scan failed on our side. Try again, or run the scan locally."}
BUSY = {"error": "busy", "message": "The hosted scan is busy with another repo; try again in a few seconds."}
_one_scan = threading.Lock()

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
    """(status, JSON body) of scan_github(url) run in a child process under MEMORY_LIMIT, KILL_AFTER_S and
    MAX_ANSWER_BYTES. One at a time per instance: each child may use MEMORY_LIMIT on its own."""
    if not _one_scan.acquire(blocking=False):
        return 503, BUSY
    work = read = write = None
    pid, started = 0, time.monotonic()
    try:
        work = tempfile.mkdtemp(prefix="offramp-req-")  # the child's temp files, removed even if it is killed
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:
            _child(url, work, read, write)  # never returns
        os.close(write)
        write = None
        answer, why = _read_answer(read, started + KILL_AFTER_S)
        if why:
            os.kill(pid, signal.SIGKILL)
        code = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
        pid = 0
        if answer is None:  # killed at the deadline or the byte cap, out of memory, or crashed
            why = why or ("killed by signal %d" % -code if code < 0 else f"exited {code} without an answer")
            print(f"scan child: {why} after {time.monotonic() - started:.1f}s ({url})", file=sys.stderr)
            return 413, TOO_BIG
        return answer
    except Exception:  # mkdtemp, pipe or fork failed in this process: a JSON 500, never a dropped connection
        traceback.print_exc()
        return 500, INTERNAL
    finally:
        for fd in (read, write):
            if fd is not None:
                os.close(fd)
        if pid:  # an error after the fork: never leave the child running or unreaped
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        if work:
            shutil.rmtree(work, ignore_errors=True)
        _one_scan.release()


def _read_answer(read: int, kill_at: float) -> tuple[tuple[int, dict] | None, str | None]:
    """The child's (status, body), or None; plus why the child must be killed (deadline, byte cap), if so."""
    chunks, size = [], 0
    while True:
        left = kill_at - time.monotonic()
        if left <= 0 or not select.select([read], [], [], left)[0]:
            return None, f"killed at the {KILL_AFTER_S:.1f}s deadline"
        chunk = os.read(read, 1 << 16)
        if not chunk:
            break
        if (size := size + len(chunk)) > MAX_ANSWER_BYTES:
            return None, f"killed: its answer passed {MAX_ANSWER_BYTES:,} bytes"
        chunks.append(chunk)
    try:
        status, body = json.loads(b"".join(chunks))
    except ValueError:  # died before or while writing
        return None, None
    return (status, body), None


def _child(url: str, work: str, read: int, write: int) -> None:
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
            print("scan child: out of memory", file=sys.stderr)
            out = [413, TOO_BIG]
        except Exception as exc:  # the traceback goes to the function log, never to the client
            if isinstance(exc, OSError) and exc.errno == errno.ENOMEM:  # an mmap refused by RLIMIT_AS
                print("scan child: out of memory (ENOMEM)", file=sys.stderr)
                out = [413, TOO_BIG]
            else:
                traceback.print_exc()
                out = [500, INTERNAL]
        with os.fdopen(write, "wb") as pipe:
            pipe.write(json.dumps(out).encode())
        code = 0
    finally:
        os._exit(code)  # no atexit handlers, no buffers of the parent's flushed twice


def _limit_memory() -> None:
    """RLIMIT_AS = the address space mapped now + MEMORY_LIMIT. Linux enforces it; without /proc (macOS) no limit."""
    try:
        with open("/proc/self/statm") as statm:
            mapped = int(statm.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        return
    resource.setrlimit(resource.RLIMIT_AS, (mapped + MEMORY_LIMIT, mapped + MEMORY_LIMIT))
