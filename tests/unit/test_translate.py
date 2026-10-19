"""Golden tests for the deterministic call-site translation (PRD §6.4) and the eval request builder."""

import copy
import inspect
import json
from pathlib import Path

import httpx
import pytest

from offramp.agents.translate import (
    DEFAULT_MAX_TOKENS,
    REASONING_ROOM,
    UnresolvedCallSite,
    build_messages,
    build_request,
    max_tokens_for,
    openai_tool,
    openai_tool_choice,
    site_schema,
    spec_v0,
    strict_json_schema,
    system_message,
    translate,
)
from offramp.schemas import AdapterSpec, CallSite, EvalCase, Tier
from offramp.tf import FakeLLM, TFClient

DEMO = json.loads((Path(__file__).parents[1] / "fixtures/cassettes/adapter/demo_sites.json").read_text())
SITES = {s["id"]: CallSite(**s) for s in DEMO["sites"]}
ORDER_SCHEMA = DEMO["schemas"]["OrderFields"]
SCHEMAS = {"OrderFields": ORDER_SCHEMA}


def site(**over) -> CallSite:
    base = dict(
        id="cs_99",
        file="app.py",
        lines=[1, 9],
        sdk="openai",
        call="client.chat.completions.create",
        source_model="gpt-4o-mini",
        prompt_source="literal",
        params={},
    )
    return CallSite(**(base | over))


def case(messages) -> EvalCase:
    return EvalCase(
        id="cs_99:001", site_id="cs_99", input_messages=messages, expected_kind="text", split="dev"
    )


# ---------------------------------------------------------------- tools and tool_choice


def test_anthropic_tools_become_openai_function_tools():
    tools = SITES["cs_02"].params["tools"]
    assert [openai_tool(t) for t in tools] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Fetch an order's status, items, total and charges from the order system.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string", "description": "Order id, e.g. HH-48213"}},
                    "required": ["order_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "issue_refund",
                "description": "Refund part or all of an order to the original payment method.",
                "parameters": tools[1]["input_schema"],
            },
        },
        {
            "type": "function",
            "function": {
                "name": "escalate_to_human",
                "description": "Hand the ticket to a human support agent.",
                "parameters": tools[2]["input_schema"],
            },
        },
    ]


def test_tool_translation_edge_cases():
    no_desc = {
        "name": "ping",
        "input_schema": {"type": "object", "properties": {}},
        "cache_control": {"type": "ephemeral"},
    }
    assert openai_tool(no_desc) == {
        "type": "function",
        "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}},
    }
    already = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    assert openai_tool(already) == already
    assert openai_tool({"type": "web_search_20250305", "name": "web_search"}) is None  # Anthropic server tool


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ({"type": "auto"}, "auto"),
        ({"type": "any"}, "required"),
        ({"type": "none"}, "none"),
        (
            {"type": "tool", "name": "lookup_order"},
            {"type": "function", "function": {"name": "lookup_order"}},
        ),
        ({"type": "any", "disable_parallel_tool_use": True}, "required"),
        ("auto", "auto"),  # OpenAI values pass through
        ("required", "required"),
        ({"type": "function", "function": {"name": "f"}}, {"type": "function", "function": {"name": "f"}}),
        ({"type": "function", "name": "f"}, {"type": "function", "function": {"name": "f"}}),  # Responses API
        (None, None),
    ],
)
def test_tool_choice(source, expected):
    assert openai_tool_choice(source) == expected


# ---------------------------------------------------------------- system and schemas


def test_system_message():
    assert system_message("Be terse.") == {"role": "system", "content": "Be terse."}
    blocks = [
        {"type": "text", "text": "Rule one.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Rule two."},
    ]
    assert system_message(blocks) == {"role": "system", "content": "Rule one.\nRule two."}


def test_pydantic_schema_becomes_strict_json_schema():
    nested = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}, "note": {"type": "string", "default": ""}},
        "required": ["item"],
        "$defs": {"Item": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": []}},
    }
    original = copy.deepcopy(nested)
    assert strict_json_schema(nested, "Order") == {
        "type": "json_schema",
        "json_schema": {
            "name": "Order",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"item": {"$ref": "#/$defs/Item"}, "note": {"type": "string", "default": ""}},
                "required": ["item", "note"],
                "additionalProperties": False,
                "$defs": {
                    "Item": {
                        "type": "object",
                        "properties": {"sku": {"type": "string"}},
                        "required": ["sku"],
                        "additionalProperties": False,
                    }
                },
            },
        },
    }
    assert nested == original  # input untouched


def test_site_schema_sources():
    assert site_schema(SITES["cs_03"], SCHEMAS) == ("OrderFields", ORDER_SCHEMA)  # marker in the prompt
    assert site_schema(SITES["cs_03"], None) is None  # marker, nothing to resolve it with
    assert site_schema(SITES["cs_01"], SCHEMAS) is None
    inline = {"type": "object", "properties": {"a": {"type": "string"}}}
    rf = {"type": "json_schema", "json_schema": {"name": "thing", "schema": inline}}
    assert site_schema(site(params={"response_format": rf}), None) == ("thing", inline)
    labelled = site(contract={"task_type": "json_extraction", "output": "json_schema", "json_schema": inline})
    assert site_schema(labelled, None) == ("response", inline)


# ---------------------------------------------------------------- spec_v0 goldens: the demo repo's six sites


def test_spec_v0_classification_openai():
    assert spec_v0(SITES["cs_01"]) == AdapterSpec(max_tokens=16, temperature=0.0)


def test_spec_v0_anthropic_tools():
    s = SITES["cs_02"]
    assert spec_v0(s) == AdapterSpec(
        tools=[openai_tool(t) for t in s.params["tools"]],
        tool_choice="required",
        max_tokens=512,
        temperature=0.0,
    )


def test_spec_v0_anthropic_extraction_leaves_output_format_to_the_prompt():
    assert spec_v0(SITES["cs_03"], SCHEMAS) == AdapterSpec(max_tokens=1024)


def test_spec_v0_free_text_keeps_source_temperature():
    assert spec_v0(SITES["cs_04"]) == AdapterSpec(max_tokens=400, temperature=0.2)
    assert spec_v0(SITES["cs_06"]) == AdapterSpec(max_tokens=400)


def test_spec_v0_reasoning_model():
    # o3 reasons by default (effort medium), and max_completion_tokens already counts reasoning tokens
    assert spec_v0(SITES["cs_05"]) == AdapterSpec(max_tokens=16000, reasoning="medium")


def test_translate_notes_and_risks_for_demo_sites_are_clean():
    for sid in ("cs_01", "cs_02", "cs_03", "cs_04", "cs_05", "cs_06"):
        assert translate(SITES[sid], SCHEMAS).risk_flags == [], sid
    notes = translate(SITES["cs_02"]).notes
    assert any("3 Anthropic tools" in n for n in notes)
    assert any("'required'" in n for n in notes)
    assert any("max_completion_tokens" in n for n in translate(SITES["cs_05"]).notes)


# ---------------------------------------------------------------- other source shapes


def test_reasoning_effort_and_max_token_aliases():
    assert (
        spec_v0(
            site(source_model="o4-mini", params={"reasoning_effort": "high", "max_output_tokens": 900})
        ).reasoning
        == "high"
    )
    assert spec_v0(site(source_model="o3", params={"reasoning_effort": "minimal"})).reasoning == "low"
    assert spec_v0(site(params={"max_output_tokens": 900})).max_tokens == 900
    assert spec_v0(site(source_model="gpt-4o")).reasoning == "off"


def test_openai_response_format_passes_through():
    rf = {"type": "json_schema", "json_schema": {"name": "t", "schema": {"type": "object"}, "strict": True}}
    assert spec_v0(site(params={"response_format": rf})).response_format == rf
    assert spec_v0(site(params={"response_format": {"type": "json_object"}})).response_format == {
        "type": "json_object"
    }


def test_responses_api_text_format_becomes_response_format():
    fmt = {"type": "json_schema", "name": "t", "schema": {"type": "object"}, "strict": True}
    assert spec_v0(
        site(call="client.responses.create", params={"text": {"format": fmt}})
    ).response_format == {
        "type": "json_schema",
        "json_schema": {"name": "t", "schema": {"type": "object"}, "strict": True},
    }


def test_responses_api_tools_tool_choice_and_reasoning():
    params = {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}
    tool = {
        "type": "function",
        "name": "lookup_order",
        "description": "Fetch an order.",
        "parameters": params,
        "strict": True,
    }
    chat_tool = {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Fetch an order.",
            "parameters": params,
            "strict": True,
        },
    }
    assert openai_tool(tool) == chat_tool
    assert openai_tool({"type": "web_search_preview"}) is None  # Responses built-in tool
    s = site(
        call="client.responses.create",
        source_model="gpt-5",
        params={
            "tools": [tool],
            "tool_choice": {"type": "function", "name": "lookup_order"},
            "reasoning": {"effort": "high"},
            "max_output_tokens": 800,
        },
    )
    t = translate(s)
    assert t.spec == AdapterSpec(
        tools=[chat_tool],
        tool_choice={"type": "function", "function": {"name": "lookup_order"}},
        reasoning="high",
        max_tokens=800,
    )
    assert t.risk_flags == []
    assert spec_v0(site(params={"reasoning": {"effort": "minimal"}})).reasoning == "low"
    assert spec_v0(site(params={"reasoning": {"summary": "auto"}})).reasoning == "off"


def test_pydantic_response_format_resolved_or_flagged():
    s = site(call="client.beta.chat.completions.parse", params={"response_format": "<pydantic:OrderFields>"})
    assert spec_v0(s, SCHEMAS).response_format == strict_json_schema(ORDER_SCHEMA, "OrderFields")
    unresolved = translate(s)
    assert (
        unresolved.spec.response_format["json_schema"]["schema"] == "<pydantic:OrderFields>"
    )  # left for caller
    assert unresolved.risk_flags == [
        "unresolved_schema: OrderFields (pass its model_json_schema() in schemas)"
    ]
    nested = site(
        params={
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "o", "schema": "<pydantic:OrderFields>"},
            }
        }
    )
    assert spec_v0(nested, SCHEMAS).response_format == {
        "type": "json_schema",
        "json_schema": {"name": "o", "schema": ORDER_SCHEMA},
    }


def test_unresolved_params_streaming_and_server_tools_are_flagged():
    s = site(
        sdk="anthropic",
        params={
            "max_tokens": "<expr>",
            "temperature": "<expr>",
            "stream": True,
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"name": "f", "input_schema": {"type": "object"}},
            ],
        },
    )
    t = translate(s)
    assert t.spec.max_tokens is None and t.spec.temperature == 0.0
    assert t.spec.tools == [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    assert t.risk_flags == [
        "unresolved_param: max_tokens",
        "unresolved_param: temperature",
        "unsupported_tool: web_search (server-side tool, no Token Factory equivalent)",
        "streaming: the source streams; Offramp evaluates the non-streaming call",
    ]
    assert "NEEDS_RUNTIME_CAPTURE" not in s.flags  # max_tokens/temperature are re-chosen, not translated
    build_request(t.spec, case(MSGS), Tier.SUPER)


TOOL = {"name": "f", "input_schema": {"type": "object"}}


@pytest.mark.parametrize(
    ("params", "key"),
    [
        ({"tools": "<expr>"}, "tools"),
        ({"tools": [TOOL, "<expr>"]}, "tools"),
        ({"tools": [{"name": "f", "input_schema": {"type": "object", "properties": "<expr>"}}]}, "tools"),
        ({"tools": [TOOL], "tool_choice": "<expr>"}, "tool_choice"),
        ({"tools": [TOOL], "tool_choice": {"type": "tool", "name": "<expr>"}}, "tool_choice"),
        ({"response_format": "<expr>"}, "response_format"),
        (
            {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": "<expr>"}}},
            "response_format",
        ),
        ({"text": "<expr>"}, "text"),
        ({"text": {"format": "<expr>"}}, "text"),
    ],
)
def test_unresolved_call_shaping_param_blocks_evaluation(params, key):
    """tools / tool_choice / response_format define the call's contract: unknown at any depth, the call that
    ships is unknown, so it is never evaluated with a guess in its place."""
    s = site(sdk="anthropic", params=params)
    t = translate(s)
    assert f"unresolved_param: {key}" in t.risk_flags
    assert any(f.startswith("unevaluable: ") and f.endswith("needs runtime capture") for f in t.risk_flags)
    assert s.flags == ["NEEDS_RUNTIME_CAPTURE"]
    translate(s)
    assert s.flags == ["NEEDS_RUNTIME_CAPTURE"]  # marked once
    with pytest.raises(UnresolvedCallSite, match="<expr>"):
        build_request(t.spec, case(MSGS), Tier.SUPER)


def test_unresolved_schema_in_tools_is_flagged_and_blocks_evaluation():
    s = site(sdk="anthropic", params={"tools": [{"name": "f", "input_schema": "<pydantic:Args>"}]})
    t = translate(s)
    assert t.risk_flags == ["unresolved_schema: Args (pass its model_json_schema() in schemas)"]
    with pytest.raises(UnresolvedCallSite, match="Args"):
        build_request(t.spec, case(MSGS), Tier.SUPER)
    resolved = spec_v0(s, {"Args": {"type": "object"}})
    assert resolved.tools == [
        {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    ]


def test_disable_parallel_tool_use_maps_to_parallel_tool_calls():
    s = site(
        sdk="anthropic",
        params={
            "tools": [{"name": "f", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        },
    )
    spec = spec_v0(s)
    assert spec.tool_choice == "auto" and spec.extra_body == {"parallel_tool_calls": False}


# ---------------------------------------------------------------- build_request


MSGS = [{"role": "system", "content": "Classify."}, {"role": "user", "content": "Ticket: refund please"}]


def test_build_messages_applies_spec_without_mutating_the_case():
    shots = [
        {"role": "user", "content": "Ticket: where is my parcel"},
        {"role": "assistant", "content": "shipping"},
    ]
    spec = AdapterSpec(
        system_prompt="Route tickets.", few_shot=shots, user_prefix="> ", user_suffix="\n\nLabel only."
    )
    original = copy.deepcopy(MSGS)
    assert build_messages(spec, MSGS) == [
        {"role": "system", "content": "Route tickets."},
        *shots,
        {"role": "user", "content": "> Ticket: refund please\n\nLabel only."},
    ]
    assert MSGS == original
    no_system = [{"role": "user", "content": "hi"}]
    assert build_messages(AdapterSpec(system_prompt="S"), no_system) == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"},
    ]
    assert build_messages(AdapterSpec(), MSGS) == MSGS


def test_max_tokens_leaves_room_for_reasoning():
    assert max_tokens_for(AdapterSpec()) == DEFAULT_MAX_TOKENS
    assert max_tokens_for(AdapterSpec(max_tokens=16)) == 16
    assert max_tokens_for(AdapterSpec(max_tokens=16, reasoning="low")) == 16 + REASONING_ROOM
    assert max_tokens_for(AdapterSpec(reasoning="high")) >= 2000  # GC25


def test_build_request_matches_tfclient_chat():
    spec = spec_v0(SITES["cs_02"]).model_copy(update={"user_suffix": "\n\nOne tool."})
    req = build_request(spec, case(MSGS), "super")
    assert req == {
        "role": "eval_target",
        "tier": Tier.SUPER,
        "messages": build_messages(spec, MSGS),
        "max_tokens": 512,
        "temperature": 0.0,
        "json_schema": None,
        "json_mode_fallback": False,
        "tools": spec.tools,
        "tool_choice": "required",
        "reasoning": "off",
        "extra_body": {},
    }
    assert set(req) <= set(inspect.signature(TFClient.chat).parameters)
    FakeLLM().add("eval_target", "ok").chat(**req)  # the LLM protocol accepts it as is


def test_lightning_gets_auto_instead_of_required_tool_choice():
    """Lightning loops tool calls under tool_choice="required" (recorded: tests/agents/test_adapter.py)."""
    spec = spec_v0(SITES["cs_02"])
    assert build_request(spec, case(MSGS), Tier.LIGHTNING)["tool_choice"] == "auto"
    assert build_request(spec, case(MSGS), Tier.SUPER)["tool_choice"] == "required"
    assert build_request(spec, case(MSGS), Tier.ULTRA)["tool_choice"] == "required"
    named = {"type": "function", "function": {"name": "lookup_order"}}
    forced = spec.model_copy(update={"tool_choice": named})
    assert build_request(forced, case(MSGS), Tier.LIGHTNING)["tool_choice"] == forced.tool_choice


def test_build_request_refuses_unresolved_values():
    bad = site(params={"response_format": "<pydantic:OrderFields>"})
    with pytest.raises(UnresolvedCallSite, match="OrderFields"):
        build_request(spec_v0(bad), case(MSGS), Tier.SUPER)
    with pytest.raises(UnresolvedCallSite, match="<expr>"):
        build_request(
            AdapterSpec(tools=[{"type": "function", "function": {"name": "f", "parameters": "<expr>"}}]),
            case(MSGS),
            Tier.SUPER,
        )


def test_build_request_wire_format(tmp_path):
    """What reaches Token Factory: OpenAI tools, tool_choice, strict schema and thinking switch."""
    bodies = []

    def server(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        message = {"role": "assistant", "content": '{"order_id": null}'}
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = TFClient(
        api_key="k",
        base_url="https://tf.test/v1",
        cache_dir=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
    )
    tools_spec = spec_v0(SITES["cs_02"])
    client.chat(**build_request(tools_spec, case(MSGS), Tier.SUPER))
    json_spec = AdapterSpec(
        response_format=strict_json_schema(ORDER_SCHEMA, "OrderFields"), reasoning="medium"
    )
    client.chat(**build_request(json_spec, case(MSGS), Tier.SUPER))
    tools_body, json_body = bodies
    assert tools_body["tools"][0] == {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": SITES["cs_02"].params["tools"][0]["description"],
            "parameters": SITES["cs_02"].params["tools"][0]["input_schema"],
        },
    }
    assert tools_body["tool_choice"] == "required"
    assert tools_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert json_body["response_format"]["json_schema"]["strict"] is True
    assert json_body["chat_template_kwargs"] == {"enable_thinking": True}
    assert json_body["max_tokens"] == 1024 + REASONING_ROOM
