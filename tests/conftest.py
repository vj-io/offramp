import pytest

from offramp import sandbox


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", help="re-record cassettes against real Token Factory")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="needs --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def fixture_repos_enter_the_sandbox_whole(monkeypatch):
    """The fixture repos are plain directories (copies under var/), not git checkouts: the sandbox image takes
    them whole (minus .env files) under its explicit opt-in. Tests of the default, a git checkout's tracked
    files only, unset it."""
    monkeypatch.setenv(sandbox.UNTRACKED_OPT_IN, "1")


@pytest.fixture
def target_code_here(monkeypatch):
    """Target-repo code (Pydantic schema imports, oracles) runs on this machine under the
    OFFRAMP_ALLOW_LOCAL_SANDBOX opt-in, never in Docker: for tests of what that code produces, on the trusted
    fixture repos. Where it runs otherwise is tests/e2e/test_target_isolation.py's subject."""
    monkeypatch.setattr(sandbox, "docker_available", lambda: False)
    monkeypatch.setenv(sandbox.LOCAL_OPT_IN, "1")
