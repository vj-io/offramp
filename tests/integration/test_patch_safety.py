"""Patch safety (PRD §12.4): a patch that breaks a repo test is downgraded to MANUAL_REVIEW and never reaches
the PR, judged by the target repo's own ruff + pytest in the real sandbox (Docker, network off; local
subprocess fallback when Docker is unavailable)."""

import shutil
import sys
from pathlib import Path

import pytest

from offramp.agents.patcher import Edit, PatchEdits, compose_patches
from offramp.sandbox import NO_LINTER, Sandbox, docker_available
from offramp.schemas import RunConfig, Tier
from offramp.tf import get_tier

sys.path.insert(0, str(Path(__file__).parents[1] / "agents"))
from test_patcher import (  # noqa: E402  (shared tiny-repo fixtures)
    BREAK_SHOUT,
    BUILD,
    CLIENT_MODULE,
    GREET,
    LOCAL_IMPORT,
    RETARGETED,
    TESTS,
    TINY,
    async_sites,
    async_tiny,
    migrate,
    patcher_llm,
    tiny_adapters,
    tiny_sites,
    tiny_verdicts,
    work_dir,
    write_case,
)

pytestmark = pytest.mark.skipif(not docker_available(), reason="needs docker")
CANARY = "canary-secret-123"
PINS = """

def test_greet_request():
    client = FakeClient("x")
    greet(client, "y")
    assert client.calls[0]["model"] == "gpt-4o-mini"  # greet


def test_shout_request():
    client = FakeClient("x")
    shout(client, "y")
    assert client.calls[0]["model"] == "gpt-4o-mini"  # shout
"""


def pinned_tiny(name: str) -> Path:
    """The tiny repo plus tests pinning each site's old model: Offramp's own changes alone fail them."""
    repo = work_dir(name) / "src"
    shutil.copytree(TINY, repo, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    (repo / TESTS).write_text((repo / TESTS).read_text() + PINS)
    return repo


def repin(site: str) -> Edit:
    old = f'== "gpt-4o-mini"  # {site}'
    return Edit(path=TESTS, search=old, replace=f'== "{get_tier(Tier.LIGHTNING).model_id}"  # {site}')


def test_a_patch_that_breaks_a_repo_test_is_downgraded_and_dropped():
    breaking = PatchEdits(edits=[repin("shout"), BREAK_SHOUT])  # the test now expects the wrong reply
    llm = patcher_llm(PatchEdits(edits=[repin("greet")]), breaking, breaking)  # try + repair round

    result = compose_patches(
        pinned_tiny("safety-src"),
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
    assert result.sandbox_mode == "docker"


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


ENV_TEST = f"""import os


def test_no_provider_keys_reach_the_sandbox():
    names = {{k for k in os.environ if k.endswith(("_KEY", "_TOKEN", "_FALLBACK"))}}
    leaked = sorted(names - {{"GPG_KEY"}})  # GPG_KEY: python image's public signing key id
    assert not leaked, leaked
    assert not any("{CANARY}" in v for v in os.environ.values())
"""


@pytest.mark.parametrize("docker", [True, False], ids=["docker", "local"])
def test_the_sandbox_environment_carries_no_keys(monkeypatch, docker):
    if not docker and not shutil.which("uv"):
        pytest.skip("local mode needs uv")
    for name in ("NEBIUS_API_KEY", "GITHUB_TOKEN", "NEBIUS_API_KEY_FALLBACK"):
        monkeypatch.setenv(name, CANARY)
    monkeypatch.setenv("OFFRAMP_ALLOW_LOCAL_SANDBOX", "1")
    repo = work_dir(f"env-{docker}") / "repo"
    shutil.copytree(TINY, repo)
    (repo / "tests" / "test_env.py").write_text(ENV_TEST)

    result = Sandbox(repo, docker=docker).check()

    assert result.mode == ("docker" if docker else "local")
    assert result.ok, result.output


def test_a_symlink_to_a_host_file_never_reaches_the_sandbox_or_a_reason():
    src = work_dir("symlink-attack") / "src"
    shutil.copytree(TINY, src)
    secret = src.parent / "environ"  # stands in for /proc/self/environ on a Linux host
    secret.write_text(f"NEBIUS_API_KEY={CANARY}")
    (src / "tinyapp" / "data.txt").symlink_to(secret)
    (src / "tests" / "test_leak.py").write_text(
        "from pathlib import Path\n\n\n"
        "def test_leak():\n"
        "    assert False, (Path(__file__).parents[1] / 'tinyapp' / 'data.txt').read_text()\n"
    )
    llm = patcher_llm(migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))

    result = compose_patches(src, tiny_sites(), tiny_verdicts(), tiny_adapters(), llm, src.parent / "work",
                             config=RunConfig(repo_url="x"), build_request=BUILD)  # fmt: skip

    assert result.warnings == ["dropped symlink tinyapp/data.txt: it points outside the repository"]
    assert all(CANARY not in v.reason for v in result.verdicts)
    assert "the untouched repo" in next(v for v in result.verdicts if v.site_id == "cs_01").reason


def test_a_patch_with_an_over_long_string_is_wrapped_and_passes_ruff():
    """The model writes a long string on one line of a test; ruff E501 would fail it. The wrap keeps its
    value."""
    long = "Ship it, then ship it again, and keep shipping it until the release notes write themselves!"
    old = 'assert motto(FakeClient("Ship it")) == "Ship it"'
    content = Edit(path=TESTS, search=old, replace=f'assert motto(FakeClient("{long}")) == "{long}"')
    llm = patcher_llm(PatchEdits(edits=[repin("greet"), content]), PatchEdits(edits=[repin("shout")]))

    src, config = pinned_tiny("e501-src"), RunConfig(repo_url="x")
    result = compose_patches(src, tiny_sites(), tiny_verdicts(), tiny_adapters(), llm, work_dir("e501"),
                             config=config, build_request=BUILD)  # fmt: skip

    assert result.alone == {"cs_01": True, "cs_03": True} and result.combined == ["cs_01", "cs_03"]
    tests = result.files[TESTS]
    assert max(map(len, tests.splitlines())) <= 100 and "noqa" not in tests and long not in tests


def test_the_probe_runs_the_migrated_function_in_docker_and_compares_its_request():
    """The shipped call, run in the sandbox (network off) with a recorder for its client, sends exactly what
    build_request evaluated for one of the site's own cases."""
    src = work_dir("probe-src") / "src"
    shutil.copytree(TINY, src, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    write_case(src, "cs_01", {"name": "Ana"}, "You write one-line greetings.", "Greet Ana.")

    work, config = work_dir("probe"), RunConfig(repo_url="x")
    result = compose_patches(src, tiny_sites(), tiny_verdicts(), tiny_adapters(), patcher_llm(), work,
                             config=config, build_request=BUILD)  # fmt: skip

    assert result.combined == ["cs_01", "cs_03"] and result.sandbox_mode == "docker"
    assert (
        result.verification["cs_01"]
        == "probe: run in the sandbox with case cs_01:001, it sends the evaluated request"
    )


SRC_LAYOUT = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n\n'
    '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n',
    "src/pkg/__init__.py": "import os  # unused: no linter is configured, only tests run\n\nVALUE = 1\n",
    "tests/test_pkg.py": "from pkg import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n",
}


def test_a_src_layout_project_is_installed_and_its_checks_run_the_mounted_code():
    """The image installs the project as its CI would (`uv sync`), so `import pkg` works from src/ with the
    network off; installed editable at /repo, the checks run the copy mounted there, never the snapshot the
    image was built from. No ruff config: the unused import fails nothing, and the note says so."""
    repo = work_dir("src-layout") / "repo"
    for path, text in SRC_LAYOUT.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    box = Sandbox(repo)

    first = box.check()
    (repo / "src" / "pkg" / "__init__.py").write_text("VALUE = 2\n")
    (repo / "tests" / "test_pkg.py").write_text(SRC_LAYOUT["tests/test_pkg.py"].replace("== 1", "== 2"))
    second = box.check()

    assert first.ok, first.output
    assert second.ok, second.output
    assert box.notes == [NO_LINTER]


def test_an_async_site_building_its_client_inside_migrates_with_its_sys_modules_fake_retargeted():
    """The real app's shape, end to end in Docker: Offramp's code rewrites the async call and its client;
    the repo's test faked `anthropic` in sys.modules, and the model's test edit fakes Offramp's client module
    the same way. The openai package is not installed here, so the probe records through a stand-in module."""
    retarget = Edit(path="tests/test_greet.py", search="", replace=RETARGETED, whole=True)
    llm = patcher_llm(PatchEdits(edits=[retarget]))

    src, work, config = async_tiny("async-docker-src"), work_dir("async-docker"), RunConfig(repo_url="x")
    result = compose_patches(src, async_sites(), tiny_verdicts(), tiny_adapters(), llm, work, config=config,
                             build_request=BUILD)  # fmt: skip

    assert result.sandbox_mode == "docker" and result.combined == ["cs_01", "cs_03"], result.verdicts
    shipped = result.files[GREET]
    assert LOCAL_IMPORT in shipped and "await client.chat.completions.create(" in shipped
    assert "anthropic" not in shipped.lower()
    assert f'sys.modules["{CLIENT_MODULE}"]' in result.files["tinyapp/offramp_client.py"]
    assert result.verification["cs_01"] == (
        "probe: run in the sandbox with case cs_01:001, it sends the evaluated request"
    )
