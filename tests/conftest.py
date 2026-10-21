import pytest

from offramp import sandbox
from offramp.tf import BudgetExceeded, CacheMiss, TFClient


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


# the demo cassettes' first repair turns that no live re-record has caught up with yet: site -> what marks it
UNRECORDED_REPAIRS = {
    "cs_05": "Never delete a test",  # Task 18: the repair after a deleted test names it
    "cs_04": "## Files now",  # Task 24: the repair after a red check shows the files as checked
}


@pytest.fixture
def unrecorded_patcher_calls_stop(monkeypatch):
    """A cache-only replay in which one of UNRECORDED_REPAIRS (a site's first repair turn, changed since the
    demo cassettes were recorded) stops its site the way a spent budget does (BudgetExceeded: MANUAL_REVIEW,
    the last failure kept) instead of failing the whole replay, until the next live re-record. Any other
    unrecorded call, another site's or a later turn's among them, still raises CacheMiss."""
    chat = TFClient.chat

    def replay(self, **kw):
        try:
            return chat(self, **kw)
        except CacheMiss:
            m = kw["messages"]
            site = m[1]["content"].split("\n", 1)[0].removeprefix("## Call site ") if len(m) == 4 else None
            if (
                self.live
                or kw.get("role") != "patcher"
                or UNRECORDED_REPAIRS.get(site, "\0") not in m[-1]["content"]
            ):
                raise
            raise BudgetExceeded("no recording of this patcher call") from None

    monkeypatch.setattr(TFClient, "chat", replay)
