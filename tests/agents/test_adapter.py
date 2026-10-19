"""Adapter Agent contract: FakeLLM unit tests, then cassette replay (recorded live from Token Factory)."""

import json
from pathlib import Path

import pytest

from offramp.agents.adapter import (
    ADAPTER_MAX_TOKENS,
    INVALID_OUTPUT,
    LIGHTNING_LEAK,
    LIGHTNING_TOOL_CHOICE,
    AdapterDecision,
    adapt,
)
from offramp.agents.translate import (
    REASONING_ROOM,
    UnresolvedCallSite,
    build_request,
    max_tokens_for,
    spec_v0,
    strict_json_schema,
    translate,
)
from offramp.schemas import CallSite, EvalCase, ModelOutput, Tier
from offramp.tf import Budget, BudgetExceeded, ChatResult, FakeLLM, TFClient
from offramp.tf.tiers import get_tier

CASSETTES = Path(__file__).parents[1] / "fixtures/cassettes/adapter"
REPAIR = CASSETTES / "repair"  # one synthetic (hand-corrupted) cassette + the live repair reply
DEMO = json.loads((CASSETTES / "demo_sites.json").read_text())
SITES = {s["id"]: CallSite(**s) for s in DEMO["sites"]}
SCHEMAS = DEMO["schemas"]
ORDER_SCHEMA = SCHEMAS["OrderFields"]
RECORDED = [
    "cs_01",
    "cs_02",
    "cs_03",
    "cs_05",
    "cs_06",
]  # cs_04 is free text like cs_06: budget kept for repair

# Eval-target smoke cases, rendered from the demo templates the way the case generator renders them.
EVAL_VARS = {
    "cs_02": {
        "ticket_text": "Hi, order HH-48213 arrived with a cracked screen. I paid 89.90 EUR. "
        "Can I get my money back?",
        "steps": "(none yet)",
    },
    "cs_03": {
        "email": "Hallo, ich bin Jana Weber (jana.weber@example.de). Meine Bestellung HH-50731 vom "
        "3. September 2026 über 129,50 € kam beschädigt an. Bitte um Hilfe."
    },
}


def decision(**over) -> dict:
    return {
        "reasoning": "off",
        "output_mode": "text",
        "user_suffix": "",
        "notes": [],
        "risk_flags": [],
    } | over


def render(site: CallSite, variables: dict[str, str]) -> EvalCase:
    msgs = []
    for role, text in (("system", site.template.system), ("user", site.template.user)):
        if text is not None:
            text = text.format(**variables)
            for name, schema in SCHEMAS.items():
                text = text.replace(f"<pydantic:{name}>", json.dumps(schema))
            msgs.append({"role": role, "content": text})
    return EvalCase(
        id=f"{site.id}:smoke",
        site_id=site.id,
        variables=variables,
        input_messages=msgs,
        expected_kind="text",
        split="dev",
    )


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Model overrides would change cassette keys; tracing would wrap the SDK."""
    for name in [f"OFFRAMP_MODEL_{t.name}" for t in Tier] + ["LANGSMITH_TRACING", "OFFRAMP_DATA_DIR"]:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- FakeLLM: the call and the guards


def test_one_structured_ultra_call_with_the_site_facts():
    llm = FakeLLM().add("adapter", decision())
    adapter = adapt(SITES["cs_01"], llm, SCHEMAS)
    [call] = llm.calls
    assert call["role"] == "adapter" and call["tier"] == Tier.ULTRA
    assert call["temperature"] == 0.0 and call["reasoning"] == "off"  # GC6, GC25
    assert call["response_model"] is AdapterDecision and call["max_tokens"] == ADAPTER_MAX_TOKENS
    system, user = call["messages"]
    assert system["role"] == "system" and "Adapter Agent" in system["content"]
    facts = json.loads(user["content"])
    assert facts["site"]["id"] == "cs_01" and facts["site"]["source_model"] == "gpt-4o-mini"
    assert facts["contract"]["labels"][0] == "billing"
    assert facts["spec_v0"] == spec_v0(SITES["cs_01"]).model_dump(mode="json")
    assert facts["output_schema_known"] is False
    assert adapter.site_id == "cs_01" and adapter.version == 0 and adapter.tier is None
    assert adapter.spec == spec_v0(SITES["cs_01"])
    assert adapter.notes == translate(SITES["cs_01"]).notes and adapter.risk_flags == []


def test_reasoning_and_format_instruction_are_applied():
    llm = FakeLLM().add(
        "adapter", decision(reasoning="low", user_suffix="  Reply with the queue name only. ")
    )
    spec = adapt(SITES["cs_01"], llm).spec
    assert spec.reasoning == "low" and spec.user_suffix == "\n\nReply with the queue name only."
    assert max_tokens_for(spec) == 16 + REASONING_ROOM


def test_json_schema_mode_builds_a_strict_schema_from_the_resolved_pydantic_model():
    spec = adapt(SITES["cs_03"], FakeLLM().add("adapter", decision(output_mode="json_schema")), SCHEMAS).spec
    assert spec.response_format == strict_json_schema(ORDER_SCHEMA, "OrderFields")
    assert spec.json_mode_fallback is False


def test_json_object_mode_validates_against_the_schema_afterwards():
    spec = adapt(SITES["cs_03"], FakeLLM().add("adapter", decision(output_mode="json_object")), SCHEMAS).spec
    assert spec.response_format == strict_json_schema(ORDER_SCHEMA, "OrderFields")
    assert spec.json_mode_fallback is True  # TFClient: json_object + schema validation + 1 repair


def test_json_mode_without_a_known_schema_is_flagged():
    adapter = adapt(SITES["cs_03"], FakeLLM().add("adapter", decision(output_mode="json_schema")))
    assert adapter.spec.response_format == {"type": "json_object"} and adapter.spec.json_mode_fallback
    assert any(f.startswith("no_output_schema:") for f in adapter.risk_flags)


@pytest.mark.parametrize("sid", ["cs_01", "cs_02", "cs_06"])
def test_json_modes_are_ignored_where_the_code_does_not_parse_json(sid):
    adapter = adapt(SITES[sid], FakeLLM().add("adapter", decision(output_mode="json_schema")), SCHEMAS)
    assert adapter.spec.response_format is None and adapter.spec.json_mode_fallback is False
    assert any(n.startswith("output_mode json_schema: ignored") for n in adapter.notes)


def test_tool_site_keeps_translated_tools():
    adapter = adapt(SITES["cs_02"], FakeLLM().add("adapter", decision()))
    assert adapter.spec.tools == spec_v0(SITES["cs_02"]).tools and adapter.spec.tool_choice == "required"
    assert LIGHTNING_TOOL_CHOICE in adapter.risk_flags  # Lightning is sent "auto" (translate.build_request)
    assert LIGHTNING_TOOL_CHOICE not in adapt(SITES["cs_01"], FakeLLM().add("adapter", decision())).risk_flags


def test_reasoning_heavy_contract_keeps_thinking_on():
    adapter = adapt(SITES["cs_05"], FakeLLM().add("adapter", decision(reasoning="off")))
    assert adapter.spec.reasoning == "medium"
    assert "reasoning: kept on (contract is reasoning_heavy)" in adapter.notes
    assert LIGHTNING_LEAK in adapter.risk_flags  # GC25: Lightning leaks reasoning into content
    assert max_tokens_for(adapter.spec) == 16000 + REASONING_ROOM


def test_agent_notes_and_flags_follow_the_translation_without_duplicates():
    llm = FakeLLM().add(
        "adapter",
        decision(reasoning="high", notes=["n1"], risk_flags=["tone: may be terser", LIGHTNING_LEAK]),
    )
    adapter = adapt(SITES["cs_06"], llm)
    assert adapter.notes == [*translate(SITES["cs_06"]).notes, "n1"]
    assert adapter.risk_flags == [LIGHTNING_LEAK, "tone: may be terser"]


def test_malformed_reply_is_repaired_exactly_once():
    llm = FakeLLM().add(
        "adapter", '{"reasoning": "off", "output_mode": ', decision(user_suffix="Label only.")
    )
    adapter = adapt(SITES["cs_01"], llm)
    first, second = llm.calls
    assert second["messages"][:2] == first["messages"]
    assert second["messages"][2] == {"role": "assistant", "content": '{"reasoning": "off", "output_mode": '}
    assert "not valid" in second["messages"][3]["content"]
    assert {k: v for k, v in second.items() if k != "messages"} == {
        k: v for k, v in first.items() if k != "messages"
    }
    assert adapter.spec.user_suffix == "\n\nLabel only." and INVALID_OUTPUT not in adapter.risk_flags
    assert any("repaired" in n for n in adapter.notes)


def test_still_malformed_after_one_repair_falls_back_to_the_translation():
    llm = FakeLLM().add("adapter", "nope", '{"reasoning": "sometimes"}')
    adapter = adapt(SITES["cs_05"], llm)
    assert len(llm.calls) == 2
    assert adapter.spec == spec_v0(SITES["cs_05"])  # o3 -> reasoning medium survives the fallback
    assert INVALID_OUTPUT in adapter.risk_flags and LIGHTNING_LEAK in adapter.risk_flags


def test_no_second_repair_when_the_client_already_repaired():
    bad = ChatResult(
        output=ModelOutput(text="nope"),
        parsed=None,
        model="m",
        tier=Tier.ULTRA,
        json_path="json_object",
        repaired=True,
    )
    llm = FakeLLM().add("adapter", bad)
    assert INVALID_OUTPUT in adapt(SITES["cs_01"], llm).risk_flags
    assert len(llm.calls) == 1


def test_errors_propagate():
    with pytest.raises(BudgetExceeded):
        adapt(SITES["cs_01"], FakeLLM().add("adapter", BudgetExceeded("Ultra agent call cap reached (40)")))
    with pytest.raises(ValueError, match="contract"):
        adapt(SITES["cs_01"].model_copy(update={"contract": None}), FakeLLM())


def test_unevaluable_site_spends_no_ultra_call():
    """tool_choice unresolved: the call cannot be evaluated, so Ultra is not asked to adapt it."""
    site = SITES["cs_02"].model_copy(deep=True)
    site.params["tool_choice"] = "<expr>"
    llm = FakeLLM()
    adapter = adapt(site, llm)
    assert llm.calls == [] and site.flags == ["NEEDS_RUNTIME_CAPTURE"]
    assert any(f.startswith("unevaluable: tool_choice") for f in adapter.risk_flags)
    assert any(n.startswith("adapter agent: not called") for n in adapter.notes)
    with pytest.raises(UnresolvedCallSite):
        build_request(adapter.spec, render(site, {v: "x" for v in site.template.variables}), Tier.SUPER)


# ---------------------------------------------------------------- cassettes recorded live from Token Factory


def replay(directory: Path = CASSETTES) -> TFClient:
    return TFClient(cache_dir=directory, live=False)


@pytest.mark.parametrize("sid", RECORDED)
def test_adapter_on_recorded_cassettes(sid):
    adapter = adapt(SITES[sid], replay(), SCHEMAS)
    assert adapter.site_id == sid and adapter.version == 0
    assert INVALID_OUTPUT not in adapter.risk_flags
    assert adapter.spec.temperature == spec_v0(SITES[sid]).temperature
    build_request(
        adapter.spec, render(SITES[sid], {v: "x" for v in SITES[sid].template.variables}), "lightning"
    )
    if sid == "cs_02":
        assert adapter.spec.tools == spec_v0(SITES[sid]).tools and adapter.spec.response_format is None
    if sid == "cs_05":
        assert adapter.spec.reasoning != "off" and LIGHTNING_LEAK in adapter.risk_flags
    # What Ultra chose in the recorded replies (re-recording may change these pins)
    reasoning, response_format = {
        "cs_01": ("off", None),
        "cs_02": ("off", None),
        "cs_03": ("off", strict_json_schema(ORDER_SCHEMA, "OrderFields")),
        "cs_05": ("high", None),
        "cs_06": ("off", None),
    }[sid]
    assert (adapter.spec.reasoning, adapter.spec.response_format) == (reasoning, response_format)
    assert adapter.spec.user_suffix == "" and len(adapter.notes) > len(translate(SITES[sid], SCHEMAS).notes)


def test_malformed_cassette_is_repaired():
    """repair/ holds a hand-corrupted copy of cs_01's recorded reply (synthetic) and the live repair reply."""
    synthetic = [p for p in REPAIR.glob("*.json") if json.loads(p.read_text()).get("synthetic")]
    assert len(synthetic) == 1 and len(list(REPAIR.glob("*.json"))) == 2
    adapter = adapt(SITES["cs_01"], replay(REPAIR), SCHEMAS)
    assert INVALID_OUTPUT not in adapter.risk_flags and any("repaired" in n for n in adapter.notes)


@pytest.mark.parametrize("sid", sorted(EVAL_VARS))
def test_adapted_requests_run_on_token_factory(sid):
    """The spec's request, sent as an eval target on Lightning, is accepted and answered (recorded live)."""
    client = replay()
    spec = adapt(SITES[sid], client, SCHEMAS).spec
    result = client.chat(**build_request(spec, render(SITES[sid], EVAL_VARS[sid]), Tier.LIGHTNING))
    if sid == "cs_02":
        [call] = result.output.tool_calls
        assert call.name == "lookup_order" and call.arguments == {"order_id": "HH-48213"}
    else:
        record = json.loads(result.output.text)
        assert record["order_id"] == "HH-50731" and record["issue"] == "damaged"


def required_tool_choice_requests() -> dict[str, dict]:
    """cs_02 sent with tool_choice="required": the evidence behind translate.REQUIRED_TOOL_CHOICE_OK."""
    spec, case = spec_v0(SITES["cs_02"]), render(SITES["cs_02"], EVAL_VARS["cs_02"])
    lightning = build_request(spec, case, Tier.LIGHTNING) | {"tool_choice": "required"}
    return {
        "lightning": lightning,
        "lightning_no_parallel": lightning | {"extra_body": {"parallel_tool_calls": False}},
        "super": build_request(spec, case, Tier.SUPER),
    }


def test_required_tool_choice_on_token_factory():
    client = replay()
    results = {k: client.chat(**req) for k, req in required_tool_choice_requests().items()}
    for looped in (results["lightning"], results["lightning_no_parallel"]):
        assert len(looped.output.tool_calls) == 10 and looped.output_tokens == 512  # ran into max_tokens
    assert [(c.name, c.arguments) for c in results["super"].output.tool_calls] == [
        ("lookup_order", {"order_id": "HH-48213"})
    ]


@pytest.mark.live
def test_record_cassettes():
    """pytest --live: records every cassette above (6 Ultra, 4 Lightning, 1 Super call; hits are free)."""
    client = TFClient(cache_dir=CASSETTES, live=True, budget=Budget(max_ultra_agent_calls=len(RECORDED)))
    adapters = {sid: adapt(SITES[sid], client, SCHEMAS) for sid in RECORDED}
    for sid, variables in EVAL_VARS.items():
        client.chat(**build_request(adapters[sid].spec, render(SITES[sid], variables), Tier.LIGHTNING))
    for request in required_tool_choice_requests().values():
        client.chat(**request)

    ultra = get_tier(Tier.ULTRA).model_id
    [source] = [
        p
        for p in CASSETTES.glob("*.json")
        if p.name != "demo_sites.json"
        and (e := json.loads(p.read_text()))["request"]["model"] == ultra
        and '"id": "cs_01"' in e["request"]["messages"][1]["content"]
    ]
    entry = json.loads(source.read_text())
    entry["response"]["choices"][0]["message"]["content"] = '{"reasoning": "off", "output_mode": '
    entry["synthetic"] = True  # hand-corrupted: exercises the one repair retry
    REPAIR.mkdir(exist_ok=True)
    (REPAIR / source.name).write_text(json.dumps(entry, indent=2, ensure_ascii=False))
    adapt(
        SITES["cs_01"], TFClient(cache_dir=REPAIR, live=True, budget=Budget(max_ultra_agent_calls=1)), SCHEMAS
    )
