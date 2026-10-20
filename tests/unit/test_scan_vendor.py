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
import importlib.util, io, json, sys, tarfile, threading, urllib.request
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

gs.urlopen = lambda req, timeout: Res(buf.getvalue())
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
print(json.dumps(out))
"""


def test_function_serves_the_vendored_scanner(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-I", "-c", PROBE, str(FUNCTION)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["loaded"] and all(f.startswith(str(VENDOR)) for f in out["loaded"]), out["loaded"]
    status, body = out["ok"]
    assert status == 200 and body["commit"] == "abc123"
    assert [(r["file"], r["model"], r["price_ratio"]) for r in body["rows"]] == [
        ("app.py", "gpt-4o", {"input": 41.7, "output": 41.7})
    ]
    assert out["bad_url"][0] == 400 and out["bad_url"][1]["error"] == "bad_url"
    assert out["bad_json"][0] == 400 and out["not_object"][0] == 400
    assert out["burst"][-1] == 429 and 429 not in out["burst"][:-1]  # per-IP limit, other IPs unaffected
