"""Run a target repo's own `ruff check` + `pytest` on a patched copy (PRD §6.8, §10).

Docker: the repo's dependencies are installed into a throwaway image while the network is still on (cached per
dependency-file hash); the checks then run with no network, 2 CPUs, 2 GB, 256 pids, no capabilities and no
privilege escalation, the copy mounted read-only, under a 10-minute cap. It gets no host environment.

Without Docker, mode "local" runs the checks as a `uv run` subprocess, only on machines that opt in with
OFFRAMP_ALLOW_LOCAL_SANDBOX=1 (developer laptops). It is not isolated, but its environment is minimal: no
provider keys or tokens. Otherwise mode is "none" and nothing runs: the caller reports "no sandbox".

The same sandbox runs the target code the case generator needs (`python`: Pydantic schema imports, oracles).
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TIMEOUT_S = 600
BUILD_TIMEOUT_S = 900
OUTPUT_TAIL = 4000
DEP_FILES = ("pyproject.toml", "uv.lock", "requirements.txt", ".python-version")
LOCAL_OPT_IN = "OFFRAMP_ALLOW_LOCAL_SANDBOX"
NO_SANDBOX = "no sandbox (Docker unavailable)"
DOCKER_LIMITS = ["--network", "none", "--cpus", "2", "--memory", "2g", "--pids-limit", "256",
                 "--cap-drop", "ALL", "--security-opt", "no-new-privileges"]  # fmt: skip
Mode = Literal["docker", "local", "none"]
CHECKS = (
    ["ruff", "check", "--no-cache", "--output-format", "concise", "."],
    ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
)

DOCKERFILE = """\
FROM ghcr.io/astral-sh/uv:python{python}-bookworm-slim
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /deps
COPY . .
RUN if [ -f pyproject.toml ]; then \\
      uv sync --no-install-project $( [ -f uv.lock ] && echo --frozen ); \\
    else \\
      uv venv /opt/venv && ( [ ! -f requirements.txt ] \\
        || uv pip install --python /opt/venv/bin/python -r requirements.txt ); \\
    fi \\
 && uv pip install --python /opt/venv/bin/python pytest ruff
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
WORKDIR /repo
"""


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

    def check(self) -> CheckResult:
        """ruff, then pytest; the first failure ends the check. One 10-minute cap covers both."""
        if self.mode == "none":
            return CheckResult(False, "none", NO_SANDBOX)
        deadline = self._deadline()
        text = ""
        for cmd in CHECKS:
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
        """Whether the repo already keeps `ruff format` clean (then patched files are formatted too)."""
        return self._run(["ruff", "format", "--no-cache", "--check", "."], self._deadline())[0] == 0

    def tidy(self, paths: list[str], format_code: bool = False) -> None:
        """Mechanical clean-up of patched files with the repo's own ruff: import order and unused imports
        (I001, F401), plus `ruff format` when the repo is formatted.

        Files go through ruff on stdin/stdout and are written here: the mount stays read-only, and writes made
        through a Docker Desktop bind mount were seen truncated on the host."""
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
        """Image with the repo's dependencies (network on at build time only), cached by dependency hash."""
        if self._image:
            return self._image
        present = [f for f in DEP_FILES if (self.repo / f).is_file()]
        version = (
            (self.repo / ".python-version").read_text().strip() if ".python-version" in present else "3.12"
        )
        dockerfile = DOCKERFILE.format(python=".".join(version.split(".")[:2]))
        digest = hashlib.sha256(dockerfile.encode())
        for f in present:
            digest.update(f.encode() + b"\0" + (self.repo / f).read_bytes())
        tag = f"offramp-sandbox:{digest.hexdigest()[:16]}"
        if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode != 0:
            with tempfile.TemporaryDirectory(prefix="offramp-image-") as ctx:  # never beside a local checkout
                for f in present:
                    shutil.copy2(self.repo / f, Path(ctx) / f)
                (Path(ctx) / "Dockerfile").write_text(dockerfile)
                p = subprocess.run(
                    ["docker", "build", "-q", "-t", tag, ctx],
                    capture_output=True,
                    text=True,
                    timeout=BUILD_TIMEOUT_S,
                )
            if p.returncode != 0:
                raise RuntimeError(f"sandbox image build failed:\n{(p.stdout + p.stderr)[-OUTPUT_TAIL:]}")
        self._image = tag
        return tag
