from types import SimpleNamespace

from tinyapp.greet import greet
from tinyapp.motto import motto
from tinyapp.shout import shout


class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.reply, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_greet_strips_the_reply():
    assert greet(FakeClient("  Hi Ana! "), "Ana") == "Hi Ana!"


def test_shout_upper_cases_the_reply():
    assert shout(FakeClient("big news"), "x") == "BIG NEWS"


def test_motto_returns_the_reply():
    assert motto(FakeClient("Ship it")) == "Ship it"
