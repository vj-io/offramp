"""Run a target repo's own `ruff check` + `pytest` on a patched copy (PRD §6.8, §10).

Docker (preferred): the repo's dependencies are installed into a throwaway image while the network is still on
(cached per dependency-file hash); the checks then run with `--network none --cpus 2`, the copy mounted
read-only, under a 10-minute cap. Without Docker the checks run as a local `uv run` subprocess: mode "local",
which is not isolated and is reported as such.
"""

import hashlib
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TIMEOUT_S = 600
BUILD_TIMEOUT_S = 900
OUTPUT_TAIL = 4000
DEP_FILES = ("pyproject.toml", "uv.lock", "requirements.txt", ".python-version")
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
    mode: Literal["docker", "local"]
    output: str  # tail of the failing (or last) command's output

    @property
    def isolated(self) -> bool:
        return self.mode == "docker"


class Sandbox:
    def __init__(self, repo: Path, timeout_s: int = TIMEOUT_S, docker: bool | None = None):
        self.repo = Path(repo).resolve()
        self.timeout_s = timeout_s
        self.mode: Literal["docker", "local"] = (
            "docker" if (docker_available() if docker is None else docker) else "local"
        )
        self._image: str | None = None

    def check(self) -> CheckResult:
        """ruff, then pytest; the first failure ends the check. One 10-minute cap covers both."""
        deadline = self._deadline()
        text = ""
        for cmd in CHECKS:
            code, text = self._run(cmd, deadline)
            if code != 0:
                return CheckResult(
                    False, self.mode, f"$ {' '.join(cmd)} (exit {code})\n{text}"[-OUTPUT_TAIL:]
                )
        return CheckResult(True, self.mode, text[-OUTPUT_TAIL:])

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

    def _deadline(self) -> float:
        return time.monotonic() + self.timeout_s

    def _run(
        self, cmd: list[str], deadline: float, stdin: str | None = None, stderr: bool = True
    ) -> tuple[int, str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 124, f"sandbox time cap of {self.timeout_s}s reached"
        name = f"offramp-sbx-{uuid.uuid4().hex[:12]}"
        if self.mode == "docker":
            try:
                image = self._ensure_image()
            except (RuntimeError, subprocess.TimeoutExpired) as e:  # no image, no check: report, never crash
                return 1, str(e)
            user = ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
            full = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--cpus", "2", *user,
                    "-v", f"{self.repo}:/repo:ro", "-w", "/repo", image, *cmd]  # fmt: skip
        else:
            frozen = ["--frozen"] if (self.repo / "uv.lock").exists() else []
            full = ["uv", "run", "--directory", str(self.repo), *frozen, *cmd]
        try:
            p = subprocess.run(full, input=stdin or "", capture_output=True, text=True, timeout=remaining)
        except subprocess.TimeoutExpired:
            if self.mode == "docker":
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            return 124, f"sandbox time cap of {self.timeout_s}s reached running {' '.join(cmd)}"
        return p.returncode, p.stdout + (p.stderr if stderr else "")

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
            ctx = self.repo.parent / f"{self.repo.name}-image"
            shutil.rmtree(ctx, ignore_errors=True)
            ctx.mkdir(parents=True)
            for f in present:
                shutil.copy2(self.repo / f, ctx / f)
            (ctx / "Dockerfile").write_text(dockerfile)
            p = subprocess.run(
                ["docker", "build", "-q", "-t", tag, str(ctx)],
                capture_output=True,
                text=True,
                timeout=BUILD_TIMEOUT_S,
            )
            if p.returncode != 0:
                raise RuntimeError(f"sandbox image build failed:\n{(p.stdout + p.stderr)[-OUTPUT_TAIL:]}")
        self._image = tag
        return tag
