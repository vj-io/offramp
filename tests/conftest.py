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


@pytest.fixture
def target_code_here(monkeypatch):
    """Target-repo code (Pydantic schema imports, oracles) runs on this machine under the
    OFFRAMP_ALLOW_LOCAL_SANDBOX opt-in, never in Docker: for tests of what that code produces, on the trusted
    fixture repos. Where it runs otherwise is tests/e2e/test_target_isolation.py's subject."""
    monkeypatch.setattr(sandbox, "docker_available", lambda: False)
    monkeypatch.setenv(sandbox.LOCAL_OPT_IN, "1")
