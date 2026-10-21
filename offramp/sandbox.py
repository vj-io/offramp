"""Run a target repo's own `ruff check` + `pytest` on a patched copy (PRD §6.8, §10).

Docker: the repo's dependencies, then the project itself (as its CI's `uv sync` installs it, editable at
/repo), are installed into a throwaway image while the network is still on (cached per build-context hash; the
context takes only a git checkout's tracked regular files, or a plain directory's under an explicit opt-in);
the checks then run with no network, 2 CPUs, 2 GB, 256 pids, no
capabilities and no privilege escalation, the copy mounted read-only over /repo (so the editable install runs
the mounted code), under a 10-minute cap. It gets no host environment. ruff runs only when the repo configures
it. A Python version whose image does not build falls back to the newest that does, and a project that does
not install falls back to its dependencies alone; each fallback is recorded in `notes`.

Without Docker, mode "local" runs the checks as a `uv run` subprocess, only on machines that opt in with
OFFRAMP_ALLOW_LOCAL_SANDBOX=1 (developer laptops). It is not isolated, but its environment is minimal: no
provider keys or tokens. Otherwise mode is "none" and nothing runs: the caller reports "no sandbox".

The same sandbox runs the target code the case generator needs (`python`: Pydantic schema imports, oracles).
"""

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TIMEOUT_S = 600
BUILD_TIMEOUT_S = 900
OUTPUT_TAIL = 4000
DEP_FILES = ("pyproject.toml", "uv.lock", "requirements.txt", ".python-version")
LOCAL_OPT_IN = "OFFRAMP_ALLOW_LOCAL_SANDBOX"
# A directory that is not a git checkout goes into the sandbox image whole (minus the denylist) only under
# this opt-in: its untracked files (.envrc, .netrc, cloud credentials) would otherwise persist in an image
# built with the network on.
UNTRACKED_OPT_IN = "OFFRAMP_ALLOW_UNTRACKED_SANDBOX"
NO_SANDBOX = "no sandbox (Docker unavailable)"
NO_LINTER = "this repo configures no ruff linter; Offramp ran its tests only"
NOT_INSTALLED = (
    "this repo's own package did not install in the sandbox image, so Offramp ran its checks with its "
    "dependencies alone"
)
NOT_COPIED = (".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox",
              "node_modules")  # fmt: skip
PYTHONS = ("3.14", "3.13", "3.12", "3.11", "3.10")  # tried newest first when the asked version does not build
DOCKER_LIMITS = ["--network", "none", "--cpus", "2", "--memory", "2g", "--pids-limit", "256",
                 "--cap-drop", "ALL", "--security-opt", "no-new-privileges"]  # fmt: skip
Mode = Literal["docker", "local", "none"]
PYTHON_VERSION = re.compile(r"\d+\.\d+(\.\d+)?")  # .python-version goes into the Dockerfile's FROM line
CHECKS = (
    ["ruff", "check", "--no-cache", "--output-format", "concise", "."],
    ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
)

# Two stages: the dependencies (cached apart from the code), then the project itself installed editable at
# /repo, which the checks mount the patched copy over. UV_PYTHON: uv takes the image's Python whatever
# .python-version asks for, so a fallback image can build.
DOCKERFILE = """\
FROM ghcr.io/astral-sh/uv:python{python}-bookworm-slim AS deps
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PYTHON={python}
WORKDIR /repo
COPY deps/ ./
RUN if [ -f pyproject.toml ]; then \\
      uv sync --no-install-project $( [ -f uv.lock ] && echo --frozen ); \\
    else \\
      uv venv /opt/venv && ( [ ! -f requirements.txt ] \\
        || uv pip install --python /opt/venv/bin/python -r requirements.txt ); \\
    fi \\
 && uv pip install --python /opt/venv/bin/python pytest ruff
{openai}ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv PYTHONDONTWRITEBYTECODE=1 HOME=/tmp

FROM deps AS project
COPY repo/ ./
RUN if [ -f pyproject.toml ]; then \\
      uv sync --inexact $( [ -f uv.lock ] && echo --frozen ); \\
    elif [ -f setup.py ]; then \\
      uv pip install --python /opt/venv/bin/python -e .; \\
    fi
"""


# When the dependencies installed a replaced SDK, the migration's own dependency edit declares openai beside
# it, so the repo's CI installs openai after the merge: the image does too (a locked `uv sync` would not read
# the edit). An SDK the image does not install leaves openai out, as CI would.
OPENAI_LAYER = """RUN if ( {sdks} ) && ! /opt/venv/bin/python -c "import openai" 2>/dev/null; then \\
      uv pip install --python /opt/venv/bin/python "{requirement}"; \\
    fi
"""
# the floor Offramp's own pyproject declares; the patcher declares the same
OPENAI_REQUIREMENT = "openai>=1.50"


def tracked_files(repo: Path) -> list[str] | None:
    """The tracked files of a git checkout whose top is `repo` (`git ls-files`); None for anything else,
    including a folder inside another checkout. git gets a minimal environment (no GIT_* variable, no key) and
    no fsmonitor hook."""
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}
    git = ["git", "-c", "core.fsmonitor=false", "-C", str(repo)]
    try:
        top = subprocess.run(
            [*git, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=30, env=env
        )
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != Path(repo).resolve():
            return None
        out = subprocess.run([*git, "ls-files", "-z"], capture_output=True, timeout=120, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (
        sorted(p for p in out.stdout.decode(errors="surrogateescape").split("\0") if p)
        if out.returncode == 0
        else None
    )


def _shippable(repo: Path, path: str) -> bool:
    """A regular file (no link, pipe or device, nothing under a link), outside the folders a copy leaves out,
    and no .env file or folder."""
    parts = Path(path).parts
    if set(NOT_COPIED) & set(parts[:-1]) or any(p == ".env" or p.startswith(".env.") for p in parts):
        return False
    try:
        return stat.S_ISREG(os.lstat(repo / path).st_mode) and not any(
            (repo / Path(*parts[:k])).is_symlink() for k in range(1, len(parts))
        )
    except OSError:
        return False


def context_files(repo: Path, files: list[str] | None = None) -> list[str]:
    """What the sandbox image may take from the repo: `files` when given (a copy made from tracked files), a
    git checkout's tracked files, or, only under UNTRACKED_OPT_IN, every file of a plain directory; always
    regular files only, never a .env file or a folder a copy leaves out. RuntimeError otherwise."""
    repo = Path(repo)
    if files is None:
        files = tracked_files(repo)
    if files is None:
        if os.environ.get(UNTRACKED_OPT_IN) != "1":
            why = f"set {UNTRACKED_OPT_IN}=1 to let a plain directory in, minus .env files"
            raise RuntimeError(
                f"{repo.name} is not a git checkout: the sandbox image takes only a git repository's tracked "
                f"files ({why}); no sandbox image"
            )
        files = [  # os.walk never follows a linked folder
            (Path(d) / f).relative_to(repo).as_posix() for d, _, names in os.walk(repo) for f in names
        ]
    return sorted(p for p in files if _shippable(repo, p))


def configures_ruff(repo: Path) -> bool:
    """Whether the repo lints with ruff: `[tool.ruff]` in pyproject.toml, a ruff.toml or .ruff.toml, or ruff
    named in its CI workflows or pre-commit config. A linked file is never read."""
    repo = Path(repo)
    if any(
        (repo / name).is_file() and not (repo / name).is_symlink() for name in ("ruff.toml", ".ruff.toml")
    ):
        return True
    pyproject = repo / "pyproject.toml"
    try:
        if not pyproject.is_symlink() and "ruff" in tomllib.loads(pyproject.read_text()).get("tool", {}):
            return True
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        pass
    ci = [repo / ".pre-commit-config.yaml", *(repo / ".github" / "workflows").glob("*.y*ml")]
    return any(
        f.is_file() and not f.is_symlink() and re.search(r"\bruff\b", f.read_text(errors="replace"))
        for f in ci
    )


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except subprocess.TimeoutExpired:
        return False


@dataclass
class CheckResult:
    ok: bool
    mode: Mode
    output: str  # tail of the failing (or last) command's output

    @property
    def isolated(self) -> bool:
        return self.mode == "docker"


class Sandbox:
    def __init__(self, repo: Path, timeout_s: int = TIMEOUT_S, docker: bool | None = None):
        self.repo = Path(repo).resolve()
        self.timeout_s = timeout_s
        if docker_available() if docker is None else docker:
            self.mode: Mode = "docker"
        else:
            self.mode = "local" if os.environ.get(LOCAL_OPT_IN) == "1" else "none"
        self._image: str | None = None
        self.ruff = configures_ruff(self.repo)
        self.notes: list[str] = [] if self.ruff else [NO_LINTER]  # what the PR states about these checks
        self.files: list[str] | None = None  # set by whoever copied the repo from its tracked files
        self.sdks: set[str] = set()  # SDKs a migration replaces: see OPENAI_LAYER

    def check(self) -> CheckResult:
        """ruff (when the repo configures it), then pytest; the first failure ends the check. One 10-minute
        cap covers both."""
        if self.mode == "none":
            return CheckResult(False, "none", NO_SANDBOX)
        deadline = self._deadline()
        text = ""
        for cmd in CHECKS if self.ruff else CHECKS[1:]:
            code, text = self._run(cmd, deadline)
            if code != 0:
                return CheckResult(
                    False, self.mode, f"$ {' '.join(cmd)} (exit {code})\n{text}"[-OUTPUT_TAIL:]
                )
        return CheckResult(True, self.mode, text[-OUTPUT_TAIL:])

    def probe(self, script: str) -> tuple[int, str]:
        """Run a Python script (sent on stdin) against the copy, under the same isolation as the checks."""
        if self.mode == "none":
            return 1, NO_SANDBOX
        return self._run(["python", "-"], self._deadline(), stdin=script)

    def python(self, code: str, *args: str, stdin: str | None = None) -> tuple[int, str, str]:
        """`python -B -c code *args` in the repo, isolated as the checks are: (exit code, stdout, stderr)."""
        if self.mode == "none":
            return 1, "", NO_SANDBOX
        return self._exec(["python", "-B", "-c", code, *args], self._deadline(), stdin)

    def formatted(self) -> bool:
        """Whether the repo already keeps `ruff format` clean (then patched files are formatted too); never
        for a repo that does not configure ruff."""
        return (
            self.ruff
            and self._run(["ruff", "format", "--no-cache", "--check", "."], self._deadline())[0] == 0
        )

    def tidy(self, paths: list[str], format_code: bool = False) -> None:
        """Mechanical clean-up of patched files with the repo's own ruff: import order and unused imports
        (I001, F401), plus `ruff format` when the repo is formatted.

        Files go through ruff on stdin/stdout and are written here: the mount stays read-only, and writes made
        through a Docker Desktop bind mount were seen truncated on the host. A repo that does not configure
        ruff gets no ruff edits."""
        if not self.ruff:
            return
        deadline = self._deadline()
        for path in paths:
            text = (self.repo / path).read_text()
            steps = [
                ["ruff", "check", "--no-cache", "--fix", "--exit-zero", "--quiet", "--select", "I001,F401"]
            ]
            steps += [["ruff", "format", "--no-cache", "--quiet"]] if format_code else []
            for cmd in steps:
                code, out = self._run(
                    [*cmd, "--stdin-filename", path, "-"], deadline, stdin=text, stderr=False
                )
                if code == 0 and out.strip():
                    text = out
            (self.repo / path).write_text(text)

    def _local_env(self) -> dict[str, str]:
        """Only what uv and pytest need: no *_KEY / *_TOKEN / *_FALLBACK, HOME and TMPDIR in a scratch dir."""
        scratch = self.repo.parent / f"{self.repo.name}-home"
        scratch.mkdir(exist_ok=True)
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(scratch),
            "TMPDIR": str(scratch),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or str(Path.home() / ".cache" / "uv"),
            "UV_PYTHON_INSTALL_DIR": os.environ.get("UV_PYTHON_INSTALL_DIR")
            or str(Path.home() / ".local" / "share" / "uv" / "python"),
        }

    def _deadline(self) -> float:
        return time.monotonic() + self.timeout_s

    def _run(
        self, cmd: list[str], deadline: float, stdin: str | None = None, stderr: bool = True
    ) -> tuple[int, str]:
        code, out, err = self._exec(cmd, deadline, stdin)
        return code, out + (err if stderr else "")

    def _exec(self, cmd: list[str], deadline: float, stdin: str | None = None) -> tuple[int, str, str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 124, "", f"sandbox time cap of {self.timeout_s}s reached"
        name = f"offramp-sbx-{uuid.uuid4().hex[:12]}"
        if self.mode == "docker":
            try:
                image = self._ensure_image()
            except (RuntimeError, subprocess.TimeoutExpired) as e:  # no image, no check: report, never crash
                return 1, "", str(e)
            user = ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
            full = ["docker", "run", "--rm", "-i", "--name", name, *DOCKER_LIMITS, *user,
                    "-v", f"{self.repo}:/repo:ro", "-w", "/repo", image, *cmd]  # fmt: skip
            env = None  # the docker CLI's own environment; `docker run` passes none of it into the container
        elif self.mode == "local":
            frozen = ["--frozen"] if (self.repo / "uv.lock").exists() else []
            full = ["uv", "run", "--directory", str(self.repo), *frozen, *cmd]
            env = self._local_env()
        else:
            return 1, "", NO_SANDBOX
        try:
            p = subprocess.run(
                full, input=stdin or "", capture_output=True, text=True, timeout=remaining, env=env
            )
        except subprocess.TimeoutExpired:
            if self.mode == "docker":
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            return 124, "", f"sandbox time cap of {self.timeout_s}s reached running {' '.join(cmd)[:200]}"
        return p.returncode, p.stdout, p.stderr

    def _ensure_image(self) -> str:
        """Image with the repo's dependencies and the project installed (network on at build time only),
        cached by the hash of its build context. The Python .python-version asks for (3.12 without one) first;
        when its image does not build, the newest of PYTHONS that does, with a note."""
        if self._image:
            return self._image
        # never follow a link: the repo may be a raw clone, where runs/<id>/repo/requirements.txt ->
        # ../../../.env would copy a host file into the image (and a linked .python-version into a message)
        linked = [f for f in DEP_FILES if (self.repo / f).is_symlink()]
        if linked:
            raise RuntimeError(
                f"{', '.join(linked)} is a symlink: the sandbox image takes dependency files only as regular "
                "files in the repo; no sandbox image"
            )
        present = [f for f in DEP_FILES if (self.repo / f).is_file()]
        version = (
            (self.repo / ".python-version").read_text().strip() if ".python-version" in present else "3.12"
        )
        if not PYTHON_VERSION.fullmatch(version):  # target-repo text: refuse anything but a version number
            raise RuntimeError(
                f".python-version {version[:40]!r} is not a version such as 3.12: no sandbox image"
            )
        asked = ".".join(version.split(".")[:2])
        files = context_files(self.repo, self.files)  # what the image may take: refused before any build
        present = [f for f in present if f in files]  # an untracked dependency file is not the repo's
        deadline, failures = time.monotonic() + BUILD_TIMEOUT_S, []
        for python in [asked, *(p for p in PYTHONS if p != asked)]:
            try:
                self._image = self._build(python, present, files, deadline)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                failures.append((python, str(e)))
                if time.monotonic() < deadline:
                    continue
                break
            if python != asked:
                why = ".python-version asks for" if ".python-version" in present else "the default is"
                self.notes.append(f"{why} Python {asked}, whose sandbox image did not build; Offramp ran the "
                                  f"checks on Python {python}, the newest that built")  # fmt: skip
            return self._image
        tried = ", ".join(python for python, _ in failures[1:])
        raise RuntimeError(failures[0][1] + (f"\n(no image built on Python {tried} either)" if tried else ""))

    def _build(self, python: str, present: list[str], files: list[str], deadline: float) -> str:
        """Build (or reuse) the image for one Python: the dependencies stage first; when only the project's
        own install fails, that stage alone, with a note. RuntimeError when the dependencies do not build."""
        sdks = " || ".join(f'/opt/venv/bin/python -c "import {m}" 2>/dev/null' for m in sorted(self.sdks))
        layer = OPENAI_LAYER.format(sdks=sdks, requirement=OPENAI_REQUIREMENT) if sdks else ""
        with tempfile.TemporaryDirectory(prefix="offramp-image-") as ctx:  # never beside a local checkout
            (Path(ctx) / "deps").mkdir()
            for f in present:
                shutil.copy2(self.repo / f, Path(ctx) / "deps" / f)
            for f in files:  # tracked regular files only
                (Path(ctx) / "repo" / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.repo / f, Path(ctx) / "repo" / f)
            (Path(ctx) / "repo").mkdir(exist_ok=True)
            (Path(ctx) / "Dockerfile").write_text(DOCKERFILE.format(python=python, openai=layer))
            digest = hashlib.sha256()
            for f in sorted(p for p in Path(ctx).rglob("*") if p.is_file()):
                digest.update(f.relative_to(ctx).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
            tag = f"offramp-sandbox:{digest.hexdigest()[:16]}"
            if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
                return tag
            build = ["docker", "build", "-q"]
            p = subprocess.run([*build, "--target", "deps", "-t", f"{tag}-deps", ctx], capture_output=True,
                               text=True, timeout=max(deadline - time.monotonic(), 1))  # fmt: skip
            if p.returncode != 0:
                raise RuntimeError(f"sandbox image build failed:\n{(p.stdout + p.stderr)[-OUTPUT_TAIL:]}")
            try:
                p = subprocess.run([*build, "-t", tag, ctx], capture_output=True, text=True,
                                   timeout=max(deadline - time.monotonic(), 1))  # fmt: skip
            except subprocess.TimeoutExpired:  # the dependencies built: they stand without the project
                p = subprocess.CompletedProcess([], 124)
        if p.returncode != 0:
            self.notes.append(NOT_INSTALLED)
            return f"{tag}-deps"
        return tag
