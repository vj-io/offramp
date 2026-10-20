"""/api/scan's core: GitHub URL validation, the tarball caps, the symlink drop, never running target code."""

import io
import sys
import tarfile
import time
from pathlib import Path
from urllib.error import HTTPError

import pytest

from offramp import cli, github_scan
from offramp.github_scan import ScanError, extract_python, parse_github_url, scan_github

SHA = "0123456789abcdef0123456789abcdef01234567"
CALL = (
    "from openai import OpenAI\n\nclient = OpenAI()\n\n\ndef triage(t):\n"
    "    return client.chat.completions.create(\n"
    '        model="gpt-4o-mini", messages=[{"role": "user", "content": t}]\n'
    "    )\n"
)


def poisoned(marker: Path) -> str:
    """A module that leaves a marker file and raises the moment anything imports or runs it."""
    return f"open({str(marker)!r}, 'w').close()\nraise SystemExit('target code ran')\n\n" + CALL


def tarball(members: dict[str, str | tuple[str, str]], sha: str | None = SHA) -> bytes:
    """A GitHub-style archive: everything under one top directory, the commit SHA in the pax global header.
    A tuple value is (link type, target): ("sym", ...) or ("hard", ...)."""
    buf = io.BytesIO()
    headers = {"comment": sha} if sha else {}
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT, pax_headers=headers) as tar:
        for name, body in members.items():
            info = tarfile.TarInfo(name if name.startswith(("/", "..")) else f"repo-{SHA[:7]}/{name}")
            if isinstance(body, tuple):
                info.type = tarfile.SYMTYPE if body[0] == "sym" else tarfile.LNKTYPE
                info.linkname = body[1]
                tar.addfile(info)
            else:
                data = body.encode()
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, length: int | None = None):
        super().__init__(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}


def serve(monkeypatch, body: bytes, length: int | None = None, seen: list | None = None):
    def urlopen(req, timeout):
        if seen is not None:
            seen.append(req.full_url)
        return FakeResponse(body, length)

    monkeypatch.setattr(github_scan, "urlopen", urlopen)


# ---------------------------------------------------------------- URL validation


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/vj-io/offramp-demo-app", ("vj-io", "offramp-demo-app", "HEAD")),
        ("  https://github.com/vj-io/offramp-demo-app/  ", ("vj-io", "offramp-demo-app", "HEAD")),
        ("https://github.com/openai/openai-python.git", ("openai", "openai-python", "HEAD")),
        ("https://github.com/a/b/tree/main", ("a", "b", "main")),
        ("https://github.com/a/b/tree/feature/x-1.2/", ("a", "b", "feature/x-1.2")),
        ("https://github.com/acme/.github", ("acme", ".github", "HEAD")),
    ],
)
def test_accepts_github_repo_urls_with_an_optional_branch(url, expected):
    assert parse_github_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/a/b",  # not https
        "github.com/a/b",
        "https://gitlab.com/a/b",
        "https://www.github.com/a/b",
        "https://github.com.evil.example/a/b",
        "https://user:pw@github.com/a/b",
        "https://github.com:8443/a/b",
        "https://github.com/a",
        "https://github.com/a/b?tab=readme",
        "https://github.com/a/b#readme",
        "https://github.com/a/b/blob/main/app.py",
        "https://github.com/a/b/tree/",
        "https://github.com/a/b/tree/../../x",
        "https://github.com/a/b/tree/-x",
        "https://github.com/a/..",
        "https://github.com/-a/b",
        "https://github.com/a/b c",
        "https://github.com/a/b\n/tree/main",
        "file:///etc/passwd",
        "",
    ],
)
def test_refuses_everything_else(url):
    with pytest.raises(ScanError) as exc:
        parse_github_url(url)
    assert exc.value.status == 400


def test_downloads_the_codeload_tarball_of_the_branch(monkeypatch):
    seen = []
    serve(monkeypatch, tarball({"app.py": CALL}), seen=seen)
    out = scan_github("https://github.com/a/b/tree/dev")
    assert seen == ["https://codeload.github.com/a/b/tar.gz/dev"]
    assert (out["repo"], out["ref"], out["commit"], out["python_files"]) == (
        "https://github.com/a/b",
        "dev",
        SHA,
        1,
    )
    assert [(r["id"], r["file"], r["model"]) for r in out["rows"]] == [("cs_01", "app.py", "gpt-4o-mini")]
    assert out["ratio_label"] == "list-price ratio, not measured parity"


# ---------------------------------------------------------------- caps


def test_refuses_a_download_over_the_size_cap(monkeypatch):
    serve(monkeypatch, b"x" * 2048)
    with pytest.raises(ScanError) as exc:
        github_scan.download("a", "b", "HEAD", time.monotonic() + 5, max_bytes=1024)
    assert (exc.value.status, exc.value.code) == (413, "too_large")


def test_refuses_a_declared_length_over_the_cap_before_reading(monkeypatch):
    serve(monkeypatch, b"", length=github_scan.MAX_DOWNLOAD + 1)
    with pytest.raises(ScanError) as exc:
        github_scan.download("a", "b", "HEAD", time.monotonic() + 5)
    assert exc.value.code == "too_large"


def test_missing_or_private_repo_is_a_404(monkeypatch):
    def urlopen(req, timeout):
        raise HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(github_scan, "urlopen", urlopen)
    with pytest.raises(ScanError) as exc:
        scan_github("https://github.com/a/missing")
    assert (exc.value.status, exc.value.code) == (404, "not_found")


def test_refuses_more_python_files_than_the_cap(tmp_path):
    data = tarball({f"m{i}.py": "x = 1\n" for i in range(4)})
    with pytest.raises(ScanError) as exc:
        extract_python(data, tmp_path, time.monotonic() + 5, max_files=3)
    assert (exc.value.status, exc.value.code) == (413, "too_many_files")


def test_refuses_an_archive_that_unpacks_past_the_cap(tmp_path):
    data = tarball({"big.bin": "0" * 5000, "app.py": CALL})  # a zip bomb's shape: small download, big unpack
    with pytest.raises(ScanError) as exc:
        extract_python(data, tmp_path, time.monotonic() + 5, max_unpacked=4000)
    assert exc.value.code == "too_large"


def test_stops_at_the_time_limit(tmp_path):
    with pytest.raises(ScanError) as exc:
        extract_python(tarball({"app.py": CALL}), tmp_path, time.monotonic() - 1)
    assert (exc.value.status, exc.value.code) == (504, "timeout")


def test_scan_stops_at_the_time_limit(monkeypatch):
    serve(monkeypatch, tarball({"app.py": CALL}))

    def slow_extract(data, dest, deadline):
        (dest / "app.py").write_text(CALL)
        time.sleep(0.05)
        return 1, SHA

    monkeypatch.setattr(github_scan, "extract_python", slow_extract)
    with pytest.raises(ScanError) as exc:
        scan_github("https://github.com/a/b", time_limit=0.01)
    assert exc.value.code == "timeout"


def test_not_a_tarball_is_a_bad_archive(tmp_path):
    with pytest.raises(ScanError) as exc:
        extract_python(b"<html>rate limited</html>", tmp_path, time.monotonic() + 5)
    assert (exc.value.status, exc.value.code) == (502, "bad_archive")


# ---------------------------------------------------------------- symlinks and paths


def test_keeps_only_regular_python_files_and_drops_links(tmp_path):
    outside = tmp_path / "secret.py"
    outside.write_text(CALL)
    data = tarball(
        {
            "app.py": CALL,
            "README.md": "# hi",
            "linked.py": ("sym", str(outside)),
            "relative.py": ("sym", "../../secret.py"),
            "hard.py": ("hard", f"repo-{SHA[:7]}/app.py"),
            "../escape.py": CALL,
            "/abs.py": CALL,
        }
    )
    dest = tmp_path / "out"
    dest.mkdir()
    files, commit = extract_python(data, dest, time.monotonic() + 5)
    assert commit == SHA
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*")) == ["app.py"]
    assert files == 1
    assert not any(p.is_symlink() for p in dest.rglob("*"))
    assert not (tmp_path / "escape.py").exists()


def test_archive_without_a_commit_header_still_scans(tmp_path):
    files, commit = extract_python(tarball({"app.py": CALL}, sha=None), tmp_path, time.monotonic() + 5)
    assert (files, commit) == (1, None)


# ---------------------------------------------------------------- never runs the scanned code


def test_hosted_scan_never_imports_the_target_code(monkeypatch, tmp_path):
    marker = tmp_path / "ran"
    serve(monkeypatch, tarball({"triage.py": poisoned(marker), "pkg/__init__.py": poisoned(marker)}))
    out = scan_github("https://github.com/a/b")
    assert [r["file"] for r in out["rows"]] == ["pkg/__init__.py", "triage.py"]  # found by reading the files
    assert not marker.exists()
    assert "triage" not in sys.modules and "pkg" not in sys.modules


def test_cli_scan_never_imports_the_target_code(tmp_path, capsys):
    marker = tmp_path / "ran"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "triage.py").write_text(poisoned(marker))
    assert cli.main(["scan", str(repo)]) == 0
    assert "triage.py:" in capsys.readouterr().out
    assert not marker.exists()
