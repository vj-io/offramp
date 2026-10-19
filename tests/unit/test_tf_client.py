"""TFClient against httpx.MockTransport: no network, no key."""

import json
import sys
import time
import types

import httpx
import openai
import pytest
from pydantic import BaseModel

from offramp.schemas import ModelOutput, Tier
from offramp.tf import Budget, BudgetExceeded, CacheMiss, ChatResult, FakeLLM, TFClient
from offramp.tf import tiers as tiers_module
from offramp.tf.client import strip_think

MSGS = [{"role": "user", "content": "Classify: my card was charged twice"}]
SUPER_ID = "nvidia/nemotron-3-super-120b-a12b"
WEATHER_ARGS = '{"city": "Berlin", "days": 3}'


class Label(BaseModel):
    label: str
    confidence: float = 1.0


class Server:
    """Scripted Token Factory: each reply is a chat.completion body (dict), an HTTP status (int) or an
    exception to raise (transport failure)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"message": "scripted failure"}})
        return httpx.Response(200, json=reply)


def completion(content=None, *, tool_calls=None, usage=(10, 5), **message_extra):
    message = {"role": "assistant", "content": content, **message_extra}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
    }


def make_client(tmp_path, server, **kw):
    return TFClient(
        api_key="test-key",
        base_url="https://tf.test/v1",
        cache_dir=tmp_path / "cache",
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
        **kw,
    )


def chat(client, **kw):
    kw = {"role": "scanner", "tier": Tier.SUPER, "messages": MSGS, "max_tokens": 64} | kw
    return client.chat(**kw)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Nothing in the developer's shell (model overrides, keys, tracing) may change these tests."""
    ambient = [f"OFFRAMP_MODEL_{t.name}" for t in Tier]
    ambient += ["NEBIUS_API_KEY", "NEBIUS_BASE_URL", "LANGSMITH_TRACING", "OFFRAMP_DATA_DIR"]
    for name in ambient:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)  # the SDK backs off with time.sleep


@pytest.fixture
def no_json_schema(monkeypatch):
    for tier in Tier:
        monkeypatch.setitem(tiers_module.SUPPORTS_JSON_SCHEMA, tier, False)


# ------------------------------------------------------------------ request shape, json_schema path


def test_json_schema_path_sends_response_format_and_parses_into_the_model(tmp_path):
    server = Server(completion('{"label": "billing", "confidence": 0.9}', usage=(120, 8)))
    r = chat(make_client(tmp_path, server), response_model=Label)

    assert r.parsed == Label(label="billing", confidence=0.9)
    assert (r.json_path, r.repaired, r.cached) == ("json_schema", False, False)
    assert r.output == ModelOutput(text='{"label": "billing", "confidence": 0.9}')
    assert (r.model, r.tier) == (SUPER_ID, Tier.SUPER)
    assert (r.input_tokens, r.output_tokens) == (120, 8)
    assert r.cost_usd == pytest.approx((120 * 0.30 + 8 * 0.90) / 1e6)
    assert r.latency_ms >= 0

    (body,) = server.requests
    assert body["model"] == SUPER_ID
    assert (body["temperature"], body["max_tokens"], body["messages"]) == (0.0, 64, MSGS)
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Label"
    assert rf["json_schema"]["schema"]["properties"]["label"] == {"title": "Label", "type": "string"}
    assert "tools" not in body and "tool_choice" not in body


SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


@pytest.mark.parametrize(
    "given,expected_rf",
    [
        (SCHEMA, {"type": "json_schema", "json_schema": {"name": "response", "schema": SCHEMA}}),
        (
            {"name": "num", "schema": SCHEMA},
            {"type": "json_schema", "json_schema": {"name": "num", "schema": SCHEMA}},
        ),
        (
            {"type": "json_schema", "json_schema": {"name": "num", "schema": SCHEMA, "strict": True}},
            {"type": "json_schema", "json_schema": {"name": "num", "schema": SCHEMA, "strict": True}},
        ),
    ],
    ids=["bare-schema", "inner-payload", "full-response-format"],
)
def test_json_schema_param_accepts_bare_inner_and_full_forms(tmp_path, given, expected_rf):
    server = Server(completion('{"n": 3}'), completion('{"n": "three"}'))
    client = make_client(tmp_path, server)
    ok = chat(client, json_schema=given)
    assert server.requests[0]["response_format"] == expected_rf
    assert ok.parsed == {"n": 3} and ok.json_path == "json_schema"
    invalid = chat(client, json_schema=given, max_tokens=65)  # distinct request: not served from cache
    assert invalid.parsed is None and invalid.repaired is False  # json_schema path never repairs
    assert len(server.requests) == 2


def test_no_schema_means_no_response_format(tmp_path):
    server = Server(completion("hello"))
    r = chat(make_client(tmp_path, server))
    assert "response_format" not in server.requests[0]
    assert (r.json_path, r.parsed, r.output.text) == ("none", None, "hello")


def test_tools_and_tool_choice_pass_through(tmp_path):
    fn = {"name": "f", "parameters": {"type": "object", "properties": {}}}
    tools = [{"type": "function", "function": fn}]
    server = Server(completion("ok"))
    chat(make_client(tmp_path, server), tools=tools, tool_choice="auto")
    assert server.requests[0]["tools"] == tools and server.requests[0]["tool_choice"] == "auto"


def test_env_model_override_reaches_the_request(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFRAMP_MODEL_SUPER", "nvidia/override-1")
    server = Server(completion("ok"))
    r = chat(make_client(tmp_path, server))
    assert server.requests[0]["model"] == "nvidia/override-1" == r.model


# ------------------------------------------------------------------ reasoning control


def test_reasoning_is_off_by_default_and_toggled_by_level(tmp_path):
    server = Server(completion("a"), completion("b"))
    client = make_client(tmp_path, server)
    chat(client)
    chat(client, reasoning="high")
    assert server.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert server.requests[1]["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize("level,expected", [("off", False), ("low", True), ("medium", True), ("high", True)])
@pytest.mark.parametrize(
    "extra_body",
    [None, {}, {"top_k": 1}, {"chat_template_kwargs": {}}, {"chat_template_kwargs": {"low_effort": True}}],
    ids=["none", "empty", "other-key", "empty-kwargs", "other-kwarg"],
)
def test_enable_thinking_is_always_sent_explicitly(tmp_path, level, expected, extra_body):
    """Nemotron thinks by default (docs/plan/tf-probe.md), so the flag must be on every request (GC25)."""
    server = Server(completion("a"))
    chat(make_client(tmp_path, server), reasoning=level, extra_body=extra_body)
    body = server.requests[0]
    assert body["chat_template_kwargs"]["enable_thinking"] is expected
    for key, value in (extra_body or {}).items():  # the caller's own extras are kept, merged in
        if key == "chat_template_kwargs":
            assert value.items() <= body[key].items()
        else:
            assert body[key] == value


def test_reasoning_level_is_authoritative_over_a_caller_enable_thinking(tmp_path):
    server = Server(completion("a"), completion("b"))
    client = make_client(tmp_path, server)
    chat(client, reasoning="off", extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    chat(client, reasoning="high", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    assert server.requests[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert server.requests[1]["chat_template_kwargs"]["enable_thinking"] is True


def test_caller_extra_body_is_deep_merged_with_reasoning_params(tmp_path):
    server = Server(completion("a"))
    chat(make_client(tmp_path, server), extra_body={"chat_template_kwargs": {"low_effort": True}, "top_k": 1})
    body = server.requests[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False, "low_effort": True}
    assert body["top_k"] == 1


def test_reasoning_level_is_part_of_the_cache_key(tmp_path):
    server = Server(completion("a"), completion("b"))
    client = make_client(tmp_path, server)
    assert chat(client).output.text == "a"
    assert chat(client, reasoning="high").output.text == "b"
    assert len(server.requests) == 2


# ------------------------------------------------------------------ json_object + repair path


def test_json_object_path_repairs_a_malformed_first_reply(tmp_path, no_json_schema):
    server = Server(
        completion("Sure! {label: billing", usage=(100, 7)),
        completion('{"label": "billing"}', usage=(130, 6)),
    )
    r = chat(make_client(tmp_path, server), response_model=Label)

    assert r.parsed == Label(label="billing")
    assert (r.json_path, r.repaired) == ("json_object", True)
    assert r.output.text == '{"label": "billing"}'  # the corrected reply is the output
    assert (r.input_tokens, r.output_tokens) == (230, 13)  # both calls counted
    assert len(server.requests) == 2

    first, second = server.requests
    assert first["response_format"] == second["response_format"] == {"type": "json_object"}
    assert second["messages"][: len(MSGS)] == MSGS
    assert second["messages"][-2] == {"role": "assistant", "content": "Sure! {label: billing"}
    ask = second["messages"][-1]
    assert ask["role"] == "user"
    assert "Validation error" in ask["content"] and "label" in ask["content"]  # error + schema sent back


def test_json_object_path_valid_first_reply_does_not_repair(tmp_path, no_json_schema):
    server = Server(completion('{"label": "ok"}'))
    r = chat(make_client(tmp_path, server), response_model=Label)
    assert (r.parsed, r.json_path, r.repaired) == (Label(label="ok"), "json_object", False)
    assert len(server.requests) == 1


def test_repair_is_attempted_exactly_once(tmp_path, no_json_schema):
    server = Server(completion("nope"), completion("still nope"), completion('{"label": "never reached"}'))
    r = chat(make_client(tmp_path, server), response_model=Label)
    assert r.parsed is None and r.repaired is True
    assert r.output.text == "still nope"  # the eval runner scores this as invalid JSON
    assert len(server.requests) == 2


def test_json_object_path_with_a_plain_json_schema_validates_with_jsonschema(tmp_path, no_json_schema):
    server = Server(completion('{"n": "three"}'), completion('{"n": 3}'))
    r = chat(make_client(tmp_path, server), json_schema=SCHEMA)
    assert (r.parsed, r.repaired, r.json_path) == ({"n": 3}, True, "json_object")
    assert "'three' is not of type 'integer'" in server.requests[1]["messages"][-1]["content"]


def test_cached_repair_round_trip_costs_no_requests(tmp_path, no_json_schema):
    server = Server(completion("nope"), completion('{"label": "x"}'))
    client = make_client(tmp_path, server)
    first = chat(client, response_model=Label)
    again = chat(client, response_model=Label)
    assert (first.cached, again.cached) == (False, True)
    assert again.repaired is True and again.parsed == first.parsed
    assert len(server.requests) == 2


# ------------------------------------------------------------------ AdapterSpec.response_format forms

OBJECT_MODE = {"type": "json_object"}
BAD_SCHEMA = {"type": "bogus"}  # not a valid JSON Schema (an Ultra-written adapter could produce this)


def test_json_object_response_format_is_passed_through_and_validated_parse_only(tmp_path):
    server = Server(completion('{"anything": [1, 2]}'))
    r = chat(make_client(tmp_path, server), json_schema=OBJECT_MODE)
    assert server.requests[0]["response_format"] == {"type": "json_object"}
    assert (r.parsed, r.json_path, r.repaired) == ({"anything": [1, 2]}, "json_object", False)


def test_json_object_reply_that_is_not_an_object_gets_one_repair(tmp_path):
    server = Server(completion("[1, 2]"), completion('{"a": 1}'))
    r = chat(make_client(tmp_path, server), json_schema=OBJECT_MODE)
    assert (r.parsed, r.repaired, len(server.requests)) == ({"a": 1}, True, 2)
    assert "JSON object" in server.requests[1]["messages"][-1]["content"]


def test_json_object_case_replays_from_the_cache_without_crashing(tmp_path, monkeypatch):
    chat(make_client(tmp_path, Server(completion('{"a": 1}'))), json_schema=OBJECT_MODE)
    replay = chat(TFClient(cache_dir=tmp_path / "cache", live=False), json_schema=OBJECT_MODE)
    assert (replay.cached, replay.parsed, replay.json_path) == (True, {"a": 1}, "json_object")


def test_text_response_format_means_no_structured_output(tmp_path):
    server = Server(completion("just prose"))
    r = chat(make_client(tmp_path, server), json_schema={"type": "text"})
    assert "response_format" not in server.requests[0]
    assert (r.json_path, r.parsed, r.output.text) == ("none", None, "just prose")


def test_json_mode_fallback_forces_the_json_object_path_even_when_the_tier_supports_json_schema(tmp_path):
    server = Server(completion("oops {"), completion('{"n": 3}'))
    r = chat(make_client(tmp_path, server), json_schema=SCHEMA, json_mode_fallback=True)
    assert [b["response_format"] for b in server.requests] == [OBJECT_MODE, OBJECT_MODE]
    assert (r.json_path, r.repaired, r.parsed) == ("json_object", True, {"n": 3})
    repair = server.requests[1]["messages"][-1]["content"]
    assert json.dumps(SCHEMA) in repair  # the repair carries the schema


def test_json_mode_fallback_validates_against_the_response_model(tmp_path):
    server = Server(completion('{"label": 7}'), completion('{"label": "seven"}'))
    r = chat(make_client(tmp_path, server), response_model=Label, json_mode_fallback=True)
    assert (r.json_path, r.repaired, r.parsed) == ("json_object", True, Label(label="seven"))


def test_json_mode_fallback_with_nothing_to_fall_back_from_changes_nothing(tmp_path):
    """No response_format and no response_model: there is no structured contract, so no JSON mode either."""
    server = Server(completion('{"a": 1}'))
    r = chat(make_client(tmp_path, server), json_mode_fallback=True)
    assert "response_format" not in server.requests[0]
    assert (r.json_path, r.parsed) == ("none", None)


def test_an_invalid_json_schema_gives_parsed_none_not_an_exception_on_the_json_schema_path(tmp_path):
    server = Server(completion('{"a": 1}'))
    client = make_client(tmp_path, server)
    r = chat(client, json_schema=BAD_SCHEMA)
    assert (r.parsed, r.repaired, len(server.requests)) == (None, False, 1)
    assert r.output.text == '{"a": 1}'
    replay = chat(TFClient(cache_dir=tmp_path / "cache", live=False), json_schema=BAD_SCHEMA)
    assert (replay.cached, replay.parsed) == (True, None)  # a paid, cached call never crashes on replay


def test_an_invalid_json_schema_on_the_json_object_path_does_not_waste_a_repair_call(tmp_path):
    server = Server(completion('{"a": 1}'), completion('{"a": 2}'))
    r = chat(make_client(tmp_path, server), json_schema=BAD_SCHEMA, json_mode_fallback=True)
    assert (r.parsed, r.repaired, len(server.requests)) == (None, False, 1)


# ------------------------------------------------------------------ output extraction


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("plain", "plain"),
        ("<think>hmm</think>\n\nanswer", "answer"),
        ("<think>a\nmultiline\nthought</think>answer", "answer"),
        ("<think>one</think>x<think>two</think>y", "xy"),
        ("long reasoning...\n</think>\nanswer", "answer"),  # template pre-opened <think>
        ("<think>truncated mid-thought", ""),
        ("  keeps surrounding whitespace  ", "  keeps surrounding whitespace  "),
    ],
)
def test_strip_think(raw, expected):
    assert strip_think(raw) == expected


def test_output_text_is_content_only_reasoning_content_is_ignored(tmp_path):
    server = Server(
        completion("<think>private</think>The answer is 42.", reasoning_content="deep chain of thought")
    )
    r = chat(make_client(tmp_path, server))
    assert r.output.text == "The answer is 42."
    assert "chain of thought" not in r.model_dump_json()


def test_think_blocks_are_stripped_before_json_validation(tmp_path):
    server = Server(completion('<think>{"label": "wrong"}</think>{"label": "right"}'))
    r = chat(make_client(tmp_path, server), response_model=Label)
    assert r.parsed == Label(label="right")


def test_tool_calls_are_parsed_into_model_output(tmp_path):
    calls = [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": WEATHER_ARGS}},
        {"id": "c2", "type": "function", "function": {"name": "ping", "arguments": ""}},
    ]
    server = Server(completion(None, tool_calls=calls))
    r = chat(make_client(tmp_path, server))
    assert r.output.text is None
    assert [(c.name, c.arguments) for c in r.output.tool_calls] == [
        ("get_weather", {"city": "Berlin", "days": 3}),
        ("ping", {}),
    ]


def test_malformed_tool_arguments_are_kept_and_flagged_not_crashed_on(tmp_path):
    calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{city: Berlin"}}]
    r = chat(make_client(tmp_path, Server(completion("calling", tool_calls=calls))))
    assert r.output.text == "calling"
    assert r.output.tool_calls[0].arguments == {"__invalid_arguments__": "{city: Berlin"}


def test_empty_choices_yield_empty_text_not_an_exception(tmp_path):
    body = completion("x")
    body["choices"] = []
    r = chat(make_client(tmp_path, Server(body)))
    assert r.output == ModelOutput(text="", tool_calls=[])


# ------------------------------------------------------------------ cache = cassette


def test_second_identical_call_is_served_from_cache_without_http(tmp_path):
    server = Server(completion("first", usage=(10, 5)))
    client = make_client(tmp_path, server)
    a = chat(client)
    b = chat(client)
    assert (a.cached, b.cached) == (False, True)
    assert b.output == a.output and b.latency_ms == a.latency_ms  # recorded latency is replayed
    assert (b.input_tokens, b.output_tokens, b.cost_usd) == (a.input_tokens, a.output_tokens, a.cost_usd)
    assert len(server.requests) == 1


def test_cache_entry_is_a_complete_cassette(tmp_path):
    chat(make_client(tmp_path, Server(completion("hi", usage=(7, 3)))), max_tokens=11)
    (path,) = (tmp_path / "cache").glob("*.json")
    entry = json.loads(path.read_text())
    assert entry["request"]["model"] == SUPER_ID
    assert entry["request"]["messages"] == MSGS and entry["request"]["params"]["max_tokens"] == 11
    assert entry["response"]["choices"][0]["message"]["content"] == "hi"
    assert entry["usage"] == {"prompt_tokens": 7, "completion_tokens": 3}
    assert isinstance(entry["latency_ms"], int) and entry["created_at"]
    assert "synthetic" not in entry


def test_cache_only_mode_raises_cache_miss_and_never_touches_the_network(tmp_path, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    client = TFClient(cache_dir=tmp_path / "empty", live=False)  # no key, no transport
    with pytest.raises(CacheMiss, match="role='scanner'"):
        chat(client)
    assert not (tmp_path / "empty").exists()


def test_cache_only_mode_replays_what_a_live_run_recorded(tmp_path, monkeypatch):
    server = Server(completion('{"label": "billing"}', usage=(50, 4)))
    live = chat(make_client(tmp_path, server), response_model=Label)

    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    replay = chat(TFClient(cache_dir=tmp_path / "cache", live=False), response_model=Label)
    assert replay.cached is True and replay.parsed == live.parsed
    assert (replay.input_tokens, replay.output_tokens) == (50, 4)


def test_synthetic_cassette_is_replayed_like_a_recorded_one(tmp_path):
    """A hand-authored cassette (Global Constraint 7): same key derivation, "synthetic": true."""
    from offramp.tf.cache import Cache, cache_key

    thinking_off = {"chat_template_kwargs": {"enable_thinking": False}}
    params = {"max_tokens": 64, "temperature": 0.0, "extra_body": thinking_off}
    Cache(tmp_path / "cassette").put(
        cache_key(SUPER_ID, MSGS, params),
        {
            "request": {"model": SUPER_ID, "messages": MSGS, "params": params},
            "response": completion("canned"),
            "latency_ms": 321,
            "usage": {"prompt_tokens": 9, "completion_tokens": 2},
            "created_at": "2026-10-19T00:00:00+00:00",
            "synthetic": True,
        },
    )
    r = chat(TFClient(cache_dir=tmp_path / "cassette", live=False))
    assert (r.output.text, r.cached, r.latency_ms, r.input_tokens) == ("canned", True, 321, 9)


# ------------------------------------------------------------------ budget


def test_budget_is_charged_only_for_uncached_calls(tmp_path):
    budget = Budget()
    server = Server(completion("a", usage=(1000, 100)))
    client = make_client(tmp_path, server, budget=budget)
    chat(client)
    chat(client)
    chat(client)
    snap = budget.snapshot()
    assert (snap["tokens_in"], snap["tokens_out"]) == (1000, 100)
    assert snap["tf_spend_usd"] == pytest.approx((1000 * 0.30 + 100 * 0.90) / 1e6)


def test_repair_retry_is_charged_too(tmp_path, no_json_schema):
    budget = Budget()
    server = Server(completion("bad", usage=(10, 1)), completion('{"label": "x"}', usage=(20, 2)))
    chat(make_client(tmp_path, server, budget=budget), response_model=Label)
    assert (budget.tokens_in, budget.tokens_out) == (30, 3)


def test_ultra_agent_cap_refuses_the_next_call_before_any_http(tmp_path):
    budget = Budget(max_ultra_agent_calls=2)
    server = Server(*[completion(f"r{i}") for i in range(5)])
    client = make_client(tmp_path, server, budget=budget)
    for i in range(2):
        chat(client, role="adapter", tier=Tier.ULTRA, messages=[{"role": "user", "content": str(i)}])
    with pytest.raises(BudgetExceeded, match="Ultra agent call cap"):
        chat(client, role="judge", tier=Tier.ULTRA, messages=[{"role": "user", "content": "3"}])
    assert len(server.requests) == 2  # the refused call never reached the server


def test_ultra_eval_target_calls_count_toward_tokens_not_the_agent_cap(tmp_path):
    budget = Budget(max_ultra_agent_calls=1)
    server = Server(*[completion("x", usage=(10, 1)) for _ in range(4)])
    client = make_client(tmp_path, server, budget=budget)
    for i in range(4):
        chat(client, role="eval_target", tier=Tier.ULTRA, messages=[{"role": "user", "content": str(i)}])
    snap = budget.snapshot()
    assert (snap["ultra_agent_calls"], snap["tokens_in"]) == (0, 40)


def test_token_cap_trips_but_the_paid_response_is_cached_and_replays_free(tmp_path):
    budget = Budget(max_in=15)
    server = Server(completion("one", usage=(10, 1)), completion("two", usage=(10, 1)))
    client = make_client(tmp_path, server, budget=budget)
    chat(client, messages=[{"role": "user", "content": "1"}])
    two = [{"role": "user", "content": "2"}]
    with pytest.raises(BudgetExceeded, match="token cap exceeded"):
        chat(client, messages=two)  # crosses 15 input tokens
    with pytest.raises(BudgetExceeded):  # budget is now closed to new calls...
        chat(client, messages=[{"role": "user", "content": "3"}])
    assert chat(client, messages=two).output.text == "two"  # ...but the cached call is free
    assert len(server.requests) == 2 and budget.tokens_in == 20


# ------------------------------------------------------------------ transport behaviour


def test_retries_timeouts_then_succeeds(tmp_path):
    server = Server(httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slower"), completion("made it"))
    assert chat(make_client(tmp_path, server)).output.text == "made it"
    assert len(server.requests) == 3


def test_retries_429_then_succeeds(tmp_path):
    server = Server(429, 429, completion("finally"))
    assert chat(make_client(tmp_path, server)).output.text == "finally"
    assert len(server.requests) == 3


def test_retries_5xx_and_gives_up_after_three_retries(tmp_path):
    server = Server(500, 502, 503, 500, completion("never"))
    with pytest.raises(openai.InternalServerError):
        chat(make_client(tmp_path, server))
    assert len(server.requests) == 4  # 1 attempt + 3 retries


def test_failed_calls_are_not_cached_or_charged(tmp_path):
    budget = Budget()
    server = Server(500, 500, 500, 500)
    client = make_client(tmp_path, server, budget=budget)
    with pytest.raises(openai.InternalServerError):
        chat(client)
    assert budget.snapshot()["tokens_in"] == 0
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").glob("*.json"))


def test_non_retryable_4xx_fails_immediately(tmp_path):
    server = Server(400, completion("never"))
    with pytest.raises(openai.BadRequestError):
        chat(make_client(tmp_path, server))
    assert len(server.requests) == 1


def test_live_client_without_a_key_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NEBIUS_API_KEY"):
        chat(TFClient(cache_dir=tmp_path, live=True))


def test_base_url_and_key_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://tf.test/v1")
    server = Server(completion("ok"))
    client = TFClient(cache_dir=tmp_path, http_client=httpx.Client(transport=httpx.MockTransport(server)))
    assert chat(client).output.text == "ok"


def test_default_base_url_is_token_factory(monkeypatch):
    monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)
    assert TFClient(api_key="k").base_url == "https://api.tokenfactory.nebius.com/v1"


# ------------------------------------------------------------------ optional LangSmith tracing


@pytest.fixture
def fake_langsmith(monkeypatch):
    wrappers = types.ModuleType("langsmith.wrappers")
    wrappers.wrap_openai = lambda sdk: ("wrapped", sdk)
    monkeypatch.setitem(sys.modules, "langsmith", types.ModuleType("langsmith"))
    monkeypatch.setitem(sys.modules, "langsmith.wrappers", wrappers)


def test_langsmith_wraps_the_openai_client_only_when_tracing_is_on(tmp_path, monkeypatch, fake_langsmith):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    wrapped = TFClient(api_key="k", cache_dir=tmp_path)._openai()
    assert wrapped[0] == "wrapped" and isinstance(wrapped[1], openai.OpenAI)

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert isinstance(TFClient(api_key="k", cache_dir=tmp_path)._openai(), openai.OpenAI)


def test_tracing_without_langsmith_installed_is_silently_untraced(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setitem(sys.modules, "langsmith", None)  # makes `import langsmith...` raise ImportError
    assert isinstance(TFClient(api_key="k", cache_dir=tmp_path)._openai(), openai.OpenAI)


# ------------------------------------------------------------------ FakeLLM


def test_fake_llm_serves_queued_replies_by_role_and_records_calls():
    fake = FakeLLM({"scanner": ["a", Label(label="x")]}).add("judge", {"k": 1}, ModelOutput(text="m"))
    r1 = fake.chat(role="scanner", tier=Tier.SUPER, messages=MSGS, max_tokens=5)
    r2 = fake.chat(role="scanner", tier=Tier.SUPER, messages=MSGS, max_tokens=5, response_model=Label)
    r3 = fake.chat(role="judge", tier=Tier.ULTRA, messages=MSGS, max_tokens=9, json_schema={"type": "object"})
    r4 = fake.chat(role="judge", tier=Tier.ULTRA, messages=MSGS, max_tokens=9)

    assert (r1.output.text, r1.tier, r1.parsed) == ("a", Tier.SUPER, None)
    assert r2.parsed == Label(label="x") and r2.output.text == '{"label":"x","confidence":1.0}'
    assert (r3.parsed, r3.output.text) == ({"k": 1}, '{"k": 1}')
    assert r4.output.text == "m"
    assert [(c["role"], c["tier"], c["max_tokens"]) for c in fake.calls] == [
        ("scanner", Tier.SUPER, 5),
        ("scanner", Tier.SUPER, 5),
        ("judge", Tier.ULTRA, 9),
        ("judge", Tier.ULTRA, 9),
    ]
    assert fake.calls[2]["json_schema"] == {"type": "object"}


def test_fake_llm_validates_string_replies_against_response_model_like_the_real_client():
    fake = FakeLLM({"scanner": ['{"label": "ok"}', "not json"]})
    ok = fake.chat(role="scanner", tier=Tier.SUPER, messages=MSGS, max_tokens=5, response_model=Label)
    bad = fake.chat(role="scanner", tier=Tier.SUPER, messages=MSGS, max_tokens=5, response_model=Label)
    assert ok.parsed == Label(label="ok") and bad.parsed is None


def test_fake_llm_passes_chat_results_through_raises_exceptions_and_errors_when_empty():
    canned = ChatResult(output=ModelOutput(text="x"), model="m", tier=Tier.SUPER, cached=True)
    fake = FakeLLM({"scanner": [canned, BudgetExceeded("cap")]})
    assert fake.chat(role="scanner", tier=Tier.SUPER, messages=MSGS, max_tokens=5) is canned
    with pytest.raises(BudgetExceeded):
        fake.chat(role="scanner", tier=Tier.SUPER, messages=MSGS, max_tokens=5)
    with pytest.raises(AssertionError, match="role 'scanner'"):
        fake.chat(role="scanner", tier=Tier.SUPER, messages=MSGS, max_tokens=5)
    assert len(fake.calls) == 3




# ------------------------------------------------------------------ FakeLLM `parsed` matches the real client

_PARITY = [
    ("dict-reply+response_model", {"label": "x", "confidence": 0.5}, {"response_model": Label}),
    ("json-text+response_model", '{"label": "x"}', {"response_model": Label}),
    ("model-reply+response_model", Label(label="x"), {"response_model": Label}),
    ("json-text+json_schema", '{"n": 3}', {"json_schema": SCHEMA}),
    ("dict-reply+json_schema", {"n": 3}, {"json_schema": SCHEMA}),
    ("json-text+json_object", '{"a": [1]}', {"json_schema": OBJECT_MODE}),
    ("json-text+json_mode+model", '{"label": "y"}', {"response_model": Label, "json_mode_fallback": True}),
    ("invalid-vs-model", '{"nope": 1}', {"response_model": Label}),
    ("invalid-vs-schema", '{"n": "x"}', {"json_schema": SCHEMA}),
    ("dict-reply+no-contract", {"a": 1}, {}),
    ("plain-text+no-contract", "hello", {}),
    ("model-reply+no-contract", Label(label="x"), {}),
]


@pytest.mark.parametrize("reply,kw", [p[1:] for p in _PARITY], ids=[p[0] for p in _PARITY])
def test_fake_and_real_client_return_the_same_parsed_value_type_and_json_path(tmp_path, reply, kw):
    if isinstance(reply, str):
        wire = reply
    elif isinstance(reply, BaseModel):
        wire = reply.model_dump_json()
    else:
        wire = json.dumps(reply)
    real = chat(make_client(tmp_path, Server(completion(wire))), **kw)
    fake = FakeLLM({"scanner": [reply]}).chat(
        role="scanner", tier=Tier.SUPER, messages=MSGS, max_tokens=64, **kw
    )
    assert type(fake.parsed) is type(real.parsed)
    assert fake.parsed == real.parsed
    assert fake.json_path == real.json_path
    assert fake.output.text == real.output.text


def test_fake_llm_records_json_mode_fallback_and_other_chat_kwargs():
    fake = FakeLLM({"scanner": ['{"a": 1}']})
    fake.chat(role="scanner", tier=Tier.SUPER, messages=MSGS, max_tokens=5, json_mode_fallback=True)
    assert fake.calls[0]["json_mode_fallback"] is True
