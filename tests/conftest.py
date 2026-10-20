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


@pytest.fixture
def target_code_here(monkeypatch):
    """Target-repo code (Pydantic schema imports, oracles) runs on this machine under the
    OFFRAMP_ALLOW_LOCAL_SANDBOX opt-in, never in Docker: for tests of what that code produces, on the trusted
    fixture repos. Where it runs otherwise is tests/e2e/test_target_isolation.py's subject."""
    monkeypatch.setattr(sandbox, "docker_available", lambda: False)
    monkeypatch.setenv(sandbox.LOCAL_OPT_IN, "1")


@pytest.fixture
def unrecorded_patcher_calls_stop(monkeypatch):
    """A cache-only replay in which an unrecorded patcher repair turn naming deleted tests (new in Task 18)
    stops its site the way a spent budget does (BudgetExceeded: MANUAL_REVIEW, the last failure kept) instead
    of failing the whole replay: the demo cassettes hold no reply to cs_05's new repair prompt until the next
    live re-record. Any other unrecorded call still raises CacheMiss."""
    chat = TFClient.chat

    def replay(self, **kw):
        try:
            return chat(self, **kw)
        except CacheMiss:
            if (
                self.live
                or kw.get("role") != "patcher"
                or "Never delete a test" not in kw["messages"][-1]["content"]
            ):
                raise
            raise BudgetExceeded("no recording of this patcher call") from None

    monkeypatch.setattr(TFClient, "chat", replay)
