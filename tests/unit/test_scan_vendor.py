"""/api/scan (web/api/scan.py) runs Offramp's own scanner from web/pyvendor: that copy must equal offramp/,
and the function must import it from there and nowhere else."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
VENDOR = ROOT / "web" / "pyvendor"
FUNCTION = ROOT / "web" / "api" / "scan.py"


def test_vendored_scanner_is_a_byte_for_byte_copy_of_offramp():
    vendored = sorted(p for p in VENDOR.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert vendored, "web/pyvendor is empty: run `make vendor-scan`"
    stale = [
        p.relative_to(VENDOR).as_posix()
        for p in vendored
        if p.read_bytes() != (ROOT / p.relative_to(VENDOR)).read_bytes()
    ]
    assert not stale, f"web/pyvendor differs from offramp/ in {stale}: run `make vendor-scan`"


# Runs in a fresh interpreter with the venv's editable offramp install still on the path, so a module missing
# from web/pyvendor fails the import (the vendored package shadows the install) instead of loading offramp/.
PROBE = """
import importlib.util, io, json, os, resource, sys, tarfile, threading, time, urllib.request
from http.server import HTTPServer

spec = importlib.util.spec_from_file_location("scan_function", sys.argv[1])
fn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fn)
import offramp, offramp.github_scan as gs
loaded = sorted({m.__file__ for name, m in sys.modules.items() if name.startswith("offramp") and m.__file__})

buf = io.BytesIO()
sha = {"comment": "abc123"}
with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT, pax_headers=sha) as tar:
    body = b'from openai import OpenAI\\nOpenAI().chat.completions.create(model="gpt-4o", messages=[])\\n'
    info = tarfile.TarInfo("r-abc/app.py")
    info.size = len(body)
    tar.addfile(info, io.BytesIO(body))

class Res(io.BytesIO):
    headers = {}

served = [buf.getvalue()]
gs.urlopen = lambda req, timeout: Res(served[0])
server = HTTPServer(("127.0.0.1", 0), fn.handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

def post(data, ip="1.2.3.4"):
    req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/scan", data=data,
                                 headers={"content-type": "application/json", "x-real-ip": ip})
    try:
        with urllib.request.urlopen(req) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())

out = {
    "loaded": loaded,
    "ok": post(json.dumps({"github_url": "https://github.com/a/b"}).encode()),
    "bad_url": post(json.dumps({"github_url": "https://gitlab.com/a/b"}).encode()),
    "bad_json": post(b"not json"),
    "not_object": post(b"[1]"),
}
out["burst"] = [post(b"{}", ip="5.6.7.8")[0] for _ in range(fn.RATE_LIMIT + 1)]


def crash(url):
    raise RuntimeError("boom in /srv/secret/path")


real_scan, fn.scan_github = fn.scan_github, crash
url = json.dumps({"github_url": "https://github.com/a/b"}).encode()
out["crash"] = post(url, ip="9.9.9.9")

# Hard bound, time: a scan that never returns is killed at KILL_AFTER_S and answered 413.
fn.scan_github, fn.KILL_AFTER_S = (lambda u: time.sleep(60)), 0.5
start = time.monotonic()
out["hang"] = [*post(url, ip="9.9.9.10"), time.monotonic() - start]
fn.scan_github, fn.KILL_AFTER_S = real_scan, 21.5

# The re-review's file: one 99,000-character prompt reused by 9,000 call sites (891 KB, a few KB gzipped).
p = "P = " + repr("x" * 99_000) + "\\nfrom openai import OpenAI\\nc = OpenAI()\\n\\n\\ndef f(t):\\n"
calls = '    c.chat.completions.create(model="m", messages=[{"role": "user", "content": P + t}])\\n' * 9000
many = io.BytesIO()
with tarfile.open(fileobj=many, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
    body = (p + calls).encode()
    info = tarfile.TarInfo("r-abc/amp.py")
    info.size = len(body)
    tar.addfile(info, io.BytesIO(body))
served[0] = many.getvalue()
status, body = post(url, ip="9.9.9.11")
scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
out["amplifier"] = [status, len(body.get("rows", [])), body.get("warnings", []), body.get("error"),
                    resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale]
if os.path.exists("/proc/self/statm"):  # Linux enforces RLIMIT_AS: a tight limit makes the same file a 413
    fn.MEMORY_LIMIT = 64 << 20
    out["amplifier_tight"] = post(url, ip="9.9.9.12")
print(json.dumps(out))
"""


def test_function_serves_the_vendored_scanner(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-I", "-c", PROBE, str(FUNCTION)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["loaded"] and all(f.startswith(str(VENDOR)) for f in out["loaded"]), out["loaded"]
    status, body = out["ok"]
    assert status == 200 and body["commit"] == "abc123"
    [row] = body["rows"]
    assert (row["file"], row["model"], row["price_ratio"]["lightning"]) == (
        "app.py",
        "gpt-4o",
        {"input": 41.67, "output": 41.67},
    )
    assert out["bad_url"][0] == 400 and out["bad_url"][1]["error"] == "bad_url"
    assert out["bad_json"][0] == 400 and out["not_object"][0] == 400
    assert out["burst"][-1] == 429 and 429 not in out["burst"][:-1]  # per-IP limit, other IPs unaffected
    status, body = out["crash"]  # any other failure: JSON 500, no traceback or server path in the response
    assert status == 500 and body["error"] == "internal" and "secret" not in json.dumps(body)

    status, body, took = out["hang"]  # the child is killed at KILL_AFTER_S (0.5 s here), not left running
    assert (status, body["error"]) == (413, "too_large_or_complex") and took < 30
    assert "uvx --from git+https://github.com/vj-io/offramp offramp scan ." in body["message"]

    # The amplifier: answered under the limits (sites capped at 2,000) with bounded memory, or refused as 413.
    status, rows, warnings, error, child_rss = out["amplifier"]
    assert (status, rows) == (200, 2000) or (status, error) == (413, "too_large_or_complex")
    if status == 200:
        assert warnings == ["truncated at 2,000 call sites: the rest of the repository was not scanned"]
    assert child_rss < 1.2 * 2**30  # the re-review measured 1.06 GB for this file before the site cap
    if "amplifier_tight" in out:  # Linux only: RLIMIT_AS is enforced there, and on Vercel
        status, body = out["amplifier_tight"]
        assert (status, body["error"]) == (413, "too_large_or_complex")
