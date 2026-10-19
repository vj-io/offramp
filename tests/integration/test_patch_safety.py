"""Patch safety (PRD §12.4): a patch that breaks a repo test is downgraded to MANUAL_REVIEW and never reaches
the PR, judged by the target repo's own ruff + pytest in the real sandbox (Docker, network off; local
subprocess fallback when Docker is unavailable)."""

import shutil
import sys
from pathlib import Path

import pytest

from offramp.agents.patcher import Edit, compose_patches
from offramp.sandbox import Sandbox, docker_available
from offramp.schemas import RunConfig

sys.path.insert(0, str(Path(__file__).parents[1] / "agents"))
from test_patcher import (  # noqa: E402  (shared tiny-repo fixtures)
    BUILD,
    TINY,
    migrate,
    patcher_llm,
    tiny_adapters,
    tiny_sites,
    tiny_verdicts,
    work_dir,
)

pytestmark = pytest.mark.skipif(not (docker_available() or shutil.which("uv")), reason="no docker and no uv")


def test_a_patch_that_breaks_a_repo_test_is_downgraded_and_dropped():
    breaking = migrate(
        "tinyapp/shout.py",
        extra=[Edit(path="tinyapp/shout.py", search=".strip().upper()", replace=".strip()")],
    )
    llm = patcher_llm(migrate("tinyapp/greet.py"), breaking, breaking)  # try + repair round

    result = compose_patches(
        TINY,
        tiny_sites(),
        tiny_verdicts(),
        tiny_adapters(),
        llm,
        work_dir("safety"),
        config=RunConfig(repo_url="x"),
        build_request=BUILD,
    )

    decisions = {v.site_id: v.decision for v in result.verdicts}
    assert decisions == {"cs_01": "MIGRATE", "cs_02": "KEEP_CLOSED", "cs_03": "MANUAL_REVIEW"}
    shout = next(v for v in result.verdicts if v.site_id == "cs_03")
    assert "test_shout_upper_cases_the_reply" in shout.reason
    assert "tinyapp/shout.py" not in result.files and "cs_03" not in result.diffs
    assert result.adapters["cs_03"].patch is None
    assert result.adapters["cs_01"].patch and "tinyapp/greet.py" in result.files
    assert result.sandbox_mode == ("docker" if docker_available() else "local")


def test_the_sandbox_has_no_network_and_reads_the_repo_read_only():
    if not docker_available():
        pytest.skip("network isolation needs docker")
    repo = work_dir("isolation") / "repo"
    shutil.copytree(TINY, repo)
    (repo / "tests" / "test_isolation.py").write_text(
        "import socket\n\nimport pytest\n\n\n"
        "def test_no_network():\n"
        "    with pytest.raises(OSError):\n"
        "        socket.create_connection(('pypi.org', 443), timeout=3)\n\n\n"
        "def test_repo_is_read_only():\n"
        "    with pytest.raises(OSError):\n"
        "        open('written.txt', 'w')\n"
    )

    result = Sandbox(repo).check()

    assert result.ok, result.output
    assert result.mode == "docker" and result.isolated
    assert not (repo / "written.txt").exists()


def test_tidy_fixes_imports_and_layout_through_stdin_not_the_mount():
    if not docker_available():
        pytest.skip("needs docker")
    repo = work_dir("tidy") / "repo"
    shutil.copytree(TINY, repo)
    path = repo / "tinyapp" / "greet.py"
    words = "a reasonably long literal that pushes this line past the limit"
    long_call = f'    return str({{"role": "user", "content": "{words}"}})\n'
    path.write_text("import sys\nimport json\n\n\ndef f():\n" + long_call)

    Sandbox(repo).tidy(["tinyapp/greet.py"], format_code=True)

    text = path.read_text()
    assert "import sys" not in text and "import json" not in text  # F401 removed both
    assert f'"content": "{words}",' in text and text.rstrip().endswith(")")  # reformatted, not cut short
    assert max(map(len, text.splitlines())) <= 100
