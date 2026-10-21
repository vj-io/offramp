"""Scan a public GitHub repo without cloning or running it: what /api/scan and scripts/ecosystem_scan.py do.

Downloads the codeload tarball, unpacks only its regular .py files (no symlinks, no hard links, nothing
outside the destination) and runs the static scanner over them. The code is read, never imported or executed.
"""

from __future__ import annotations

import errno
import io
import re
import tarfile
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from offramp.agents.scanner import scan_repo
from offramp.scan_table import table

CODELOAD = "https://codeload.github.com/{owner}/{repo}/tar.gz/{ref}"
MAX_DOWNLOAD = 25 * 1024 * 1024  # bytes of tar.gz
MAX_PY_FILES = 5000
MAX_UNPACKED = 250 * 1024 * 1024  # declared size of every member: bounds the decompression work (zip bombs)
TIME_LIMIT_S = 20.0
MAX_ROWS, MAX_WARNINGS = 2000, 200  # keep the JSON far under Vercel's 4.5 MB response limit
GITHUB_URL = re.compile(
    r"https://github\.com/(?P<owner>[A-Za-z0-9][A-Za-z0-9-]{0,38})/(?P<repo>[A-Za-z0-9._-]{1,100}?)(?:\.git)?"
    r"(?:/tree/(?P<ref>[A-Za-z0-9._/-]{1,200}?))?/?"
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # a 3xx surfaces as HTTPError: codeload answers directly
        return None


_OPENER = build_opener(_NoRedirect)


def urlopen(req: Request, timeout: float):
    return _OPENER.open(req, timeout=timeout)


class ScanError(Exception):
    """A refusal or failure with the HTTP status /api/scan answers with."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def parse_github_url(url: str) -> tuple[str, str, str]:
    """(owner, repo, ref) of https://github.com/<owner>/<repo>[/tree/<branch>]; "HEAD": default branch."""
    if isinstance(url, str) and (url := url.strip())[:19].lower() == "https://github.com/":
        url = "https://github.com/" + url[19:]  # scheme and host are case-insensitive
    m = GITHUB_URL.fullmatch(url) if isinstance(url, str) else None
    ref = m and m["ref"]
    if not m or m["repo"] in (".", "..") or (ref and (".." in ref or "//" in ref or ref[0] in "-./")):
        raise ScanError(
            400, "bad_url", "Give a public GitHub repository URL: https://github.com/<owner>/<repo>."
        )
    return m["owner"], m["repo"], ref or "HEAD"


def download(owner: str, repo: str, ref: str, deadline: float, max_bytes: int = MAX_DOWNLOAD) -> bytes:
    too_large = ScanError(413, "too_large", f"The repository archive is over {max_bytes // 2**20} MB.")
    url = CODELOAD.format(owner=owner, repo=repo, ref=ref)
    chunks, size = [], 0
    try:
        timeout = min(10.0, max(1.0, deadline - time.monotonic()))
        with urlopen(Request(url, headers={"User-Agent": "offramp-scan"}), timeout=timeout) as res:
            if int(res.headers.get("Content-Length") or 0) > max_bytes:
                raise too_large
            while chunk := res.read1(1 << 16):  # what has arrived, so a trickle cannot outlast the deadline
                size += len(chunk)
                if size > max_bytes:
                    raise too_large
                if time.monotonic() > deadline:
                    raise ScanError(504, "timeout", "Downloading the repository took too long.")
                chunks.append(chunk)
    except HTTPError as exc:
        if exc.code == 404:
            raise ScanError(404, "not_found", "No such public repository or branch on GitHub.") from None
        raise ScanError(502, "github_error", f"GitHub answered HTTP {exc.code}.") from None
    except TimeoutError:
        raise ScanError(504, "timeout", "GitHub did not answer in time.") from None
    except OSError as exc:
        if exc.errno == errno.ENOMEM:
            raise MemoryError from exc
        raise ScanError(502, "github_error", f"Could not reach GitHub ({type(exc).__name__}).") from None
    return b"".join(chunks)


def extract_python(
    data: bytes,
    dest: Path,
    deadline: float,
    max_files: int = MAX_PY_FILES,
    max_unpacked: int = MAX_UNPACKED,
) -> tuple[int, str | None, list[str]]:
    """Unpack the regular .py files of a GitHub tarball into `dest`, without its top directory.

    Returns (.py files written, commit SHA from the archive's pax header or None, members skipped)."""
    files = unpacked = 0
    skipped: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for m in tar:
                if time.monotonic() > deadline:
                    raise ScanError(504, "timeout", "Unpacking the repository took too long.")
                unpacked += m.size
                if unpacked > max_unpacked:
                    raise ScanError(
                        413, "too_large", f"The repository unpacks to over {max_unpacked // 2**20} MB."
                    )
                top, _, rel = m.name.partition("/")
                # isfile() is False for symlinks, hard links, devices and directories: all dropped.
                if not (m.isfile() and rel.endswith(".py")) or top in ("", ".", ".."):
                    continue
                files += 1
                if files > max_files:
                    raise ScanError(
                        413, "too_many_files", f"The repository has over {max_files} Python files."
                    )
                try:
                    tar.extract(m.replace(name=rel), dest, filter="data")
                except tarfile.FilterError:  # an absolute or ../ path: never written
                    files -= 1
                    skipped.append(f"skipped {rel}: a path outside the repository")
                except OSError as exc:  # a name too long for the filesystem, a file/directory clash
                    if exc.errno == errno.ENOMEM:
                        raise MemoryError from exc
                    files -= 1
                    skipped.append(f"skipped {rel}: could not be written ({type(exc).__name__})")
            commit = tar.pax_headers.get("comment")  # git archive puts the commit SHA here
    except (tarfile.TarError, EOFError, zlib.error, OSError) as exc:
        if isinstance(exc, OSError) and exc.errno == errno.ENOMEM:  # out of memory, not a bad archive
            raise MemoryError from exc
        why = f"GitHub sent an archive that could not be read ({type(exc).__name__})."
        raise ScanError(502, "bad_archive", why) from None
    return files, commit, skipped


def scan_github(url: str, time_limit: float = TIME_LIMIT_S) -> dict[str, Any]:
    """The `offramp scan` table of a public GitHub repo, as JSON. Raises ScanError."""
    deadline = time.monotonic() + time_limit
    owner, repo, ref = parse_github_url(url)
    data = download(owner, repo, ref, deadline)
    with tempfile.TemporaryDirectory(prefix="offramp-scan-") as tmp:
        files, commit, skipped = extract_python(data, Path(tmp), deadline)
        try:
            result = scan_repo(tmp, deadline=deadline)
            result.warnings[:0] = skipped
            out = table(result, deadline)
        except TimeoutError:
            raise ScanError(504, "timeout", "Scanning the repository took too long.") from None
    rows, warnings = out["rows"], out["warnings"]
    return (
        {
            "repo": f"https://github.com/{owner}/{repo}",
            "ref": ref,
            "commit": commit,
            "python_files": files,
        }
        | out
        | {
            "rows": rows[:MAX_ROWS],
            "rows_total": len(rows),
            "warnings": warnings[:MAX_WARNINGS],
            "warnings_total": len(warnings),
        }
    )
