"""POST /api/scan {"github_url": "https://github.com/<owner>/<repo>[/tree/<branch>]"}: the `offramp scan` table.

A Vercel Python function. It runs Offramp's own scanner (offramp/github_scan.py) from web/pyvendor, a byte-for-byte
copy of offramp/ written by `make vendor-scan` and checked by tests/unit/test_scan_vendor.py. Static only: it reads
the repo's .py files from GitHub's tarball, never imports or runs them, makes no Token Factory call and needs no key.
"""

import json
import sys
import time
import traceback
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pyvendor"))

from offramp.github_scan import ScanError, scan_github  # noqa: E402

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
            body = scan_github(url)
        except ScanError as exc:
            return self.send(exc.status, exc.code, exc.message)
        except Exception:  # the traceback goes to the function log; the client never sees it or a server path
            traceback.print_exc()
            return self.send(500, "internal", "The scan failed on our side. Try again, or run the scan locally.")
        self.send(200, body=body)

    def send(self, status: int, error: str = "", message: str = "", body: dict | None = None):
        data = json.dumps(body if body is not None else {"error": error, "message": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
