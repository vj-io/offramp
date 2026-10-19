"""Deterministic call-site translation to Token Factory (PRD §6.4) and the eval request builder.

`translate(site)` maps the closed-model call's own parameters onto an `AdapterSpec` (spec v0, before the
Adapter Agent): Anthropic tools -> OpenAI function tools, Anthropic tool_choice -> OpenAI, max_tokens
aliases, response_format passthrough, Pydantic schema -> strict json_schema, the source model's reasoning.
`build_request(spec, case, tier)` turns a spec plus one eval case into the exact `TFClient.chat` kwargs. The
patch composer implements the same spec, so the request evaluated is the request that ships.
"""

import copy
import re
from typing import Any, NamedTuple

from offramp.schemas import AdapterSpec, CallSite, EvalCase, Tier
from offramp.tf.budget import EVAL_TARGET_ROLE

DEFAULT_MAX_TOKENS = 1024  # answer budget when the source sets none (OpenAI's own default is unbounded)
# Added on top of the answer budget when thinking is on: reasoning tokens count toward max_tokens (GC25).
REASONING_ROOM = 2000

EXPR = "<expr>"  # scanner: value not resolvable statically
MARKER = re.compile(r"^<pydantic:(\w+)>$")  # scanner: X.model_json_schema() of a Pydantic model X
ANY_MARKER = re.compile(r"<pydantic:(\w+)>")
MAX_TOKEN_PARAMS = ("max_tokens", "max_completion_tokens", "max_output_tokens")
# Request params the translation reads; an "<expr>" anywhere inside one is a risk flag.
TRANSLATED = (
    "tools",
    "tool_choice",
    "response_format",
    "text",
    *MAX_TOKEN_PARAMS,
    "temperature",
    "reasoning_effort",
    "stream",
    "system",
    "instructions",
)
ANTHROPIC_TOOL_CHOICE = {"auto": "auto", "any": "required", "none": "none"}  # plus {"type": "tool", "name"}
REASONING_MODEL = re.compile(r"^o\d")  # OpenAI o-series: reasons by default, effort "medium"
# Per-tier capability, like tiers.SUPPORTS_JSON_SCHEMA. Recorded 2026-10-19 (tests/agents/test_adapter.py):
# with tool_choice="required" Lightning emits tool call after tool call until max_tokens, with or without
# parallel_tool_calls=false; with "auto" it makes the one call. Super makes one call either way. Tiers marked
# False are sent "auto" instead.
REQUIRED_TOOL_CHOICE_OK = {Tier.LIGHTNING: False, Tier.SUPER: True, Tier.ULTRA: True}


class Translation(NamedTuple):
    spec: AdapterSpec
    notes: list[str]
    risk_flags: list[str]


def openai_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    """One tool definition as an OpenAI function tool; None for an Anthropic server tool (web_search...)."""
    if tool.get("type") == "function":
        return tool
    if "input_schema" not in tool:
        return None
    # cache_control has no OpenAI equivalent: dropped
    fn = {"name": tool["name"], "parameters": tool["input_schema"]}
    if tool.get("description"):
        fn["description"] = tool["description"]
    return {"type": "function", "function": fn}


def openai_tool_choice(choice: Any) -> Any:
    """Anthropic {"type": auto|any|none|tool} -> OpenAI; OpenAI values pass through unchanged."""
    if isinstance(choice, dict) and choice.get("type") == "tool":
        return {"type": "function", "function": {"name": choice["name"]}}
    if isinstance(choice, dict) and choice.get("type") in ANTHROPIC_TOOL_CHOICE:
        return ANTHROPIC_TOOL_CHOICE[choice["type"]]
    return choice


def system_message(system: str | list[dict[str, Any]]) -> dict[str, str]:
    """Anthropic `system=` (a string or text blocks, joined like the scanner joins them) -> system message."""
    if not isinstance(system, str):
        system = "\n".join(b["text"] for b in system if b.get("type") == "text")
    return {"role": "system", "content": system}


def _strict(node: Any) -> Any:
    if isinstance(node, list):
        return [_strict(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {k: _strict(v) for k, v in node.items()}
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    return out


def strict_json_schema(schema: dict[str, Any], name: str) -> dict[str, Any]:
    """A Pydantic model_json_schema() as an OpenAI strict json_schema response_format.

    Strict mode wants every property required and no extra keys on every object; Pydantic `T | None` fields
    already allow null, which is how strict mode expresses an optional value.
    """
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": _strict(schema)}}


def _resolve(value: Any, schemas: dict[str, dict[str, Any]]) -> Any:
    """Replace "<pydantic:X>" leaves with schemas[X] where known; unknown markers stay for the caller."""
    if isinstance(value, dict):
        return {k: _resolve(v, schemas) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, schemas) for v in value]
    if isinstance(value, str) and (m := MARKER.match(value)) and m[1] in schemas:
        return schemas[m[1]]
    return value


def _leaves(value: Any):
    if isinstance(value, dict):
        for v in value.values():
            yield from _leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from _leaves(v)
    else:
        yield value


def _markers(value: Any) -> list[str]:
    return [m[1] for v in _leaves(value) if isinstance(v, str) and (m := MARKER.match(v))]


def site_schema(
    site: CallSite, schemas: dict[str, dict[str, Any]] | None
) -> tuple[str, dict[str, Any]] | None:
    """(name, JSON schema) the call's output must satisfy, if known: its response_format, the labelled
    contract, or a "<pydantic:X>" schema embedded in its prompt (cs_03 style), resolved through `schemas`."""
    rf = translate(site, schemas).spec.response_format or {}
    inner = rf.get("json_schema") or {}
    if isinstance(inner.get("schema"), dict):
        return inner.get("name", "response"), inner["schema"]
    if site.contract and site.contract.json_schema:
        return "response", site.contract.json_schema
    prompt = f"{site.template.system or ''}\n{site.template.user}" if site.template else ""
    for name in ANY_MARKER.findall(prompt):
        if name in (schemas or {}):
            return name, schemas[name]
    return None


def _response_format(params: dict[str, Any], schemas: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    rf, text = params.get("response_format"), params.get("text")
    if rf is None and isinstance(text, dict) and isinstance(text.get("format"), dict):  # Responses API
        fmt = dict(text["format"])
        kind = fmt.pop("type", None)
        rf = {"type": "json_schema", "json_schema": fmt} if kind == "json_schema" else {"type": kind}
    if isinstance(rf, str) and (m := MARKER.match(rf)):  # .parse(response_format=PydanticModel)
        name = m[1]
        return (
            strict_json_schema(schemas[name], name)
            if name in schemas
            else {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": rf}}
        )
    return _resolve(rf, schemas) if isinstance(rf, dict) else None


def translate(site: CallSite, schemas: dict[str, dict[str, Any]] | None = None) -> Translation:
    """Spec v0 for one call site, with what was mapped (notes) and what could not be (risk flags)."""
    p, schemas = site.params, schemas or {}
    notes: list[str] = []
    flags = [f"unresolved_param: {k}" for k in TRANSLATED if k in p and EXPR in _leaves(p[k])]

    tools = None
    if isinstance(p.get("tools"), list):
        converted = [(t, openai_tool(_resolve(t, schemas))) for t in p["tools"]]
        tools = [o for _, o in converted if o is not None]
        flags += [
            f"unsupported_tool: {t.get('name', t.get('type'))} (Anthropic server tool, no Token Factory "
            "equivalent)"
            for t, o in converted
            if o is None
        ]
        anthropic = sum("input_schema" in t for t, _ in converted)
        if anthropic:
            notes.append(f"tools: {anthropic} Anthropic tools -> OpenAI function tools")

    extra_body: dict[str, Any] = {}
    source_choice = p.get("tool_choice")
    tool_choice = openai_tool_choice(source_choice) if source_choice != EXPR else None
    if tool_choice != source_choice and tool_choice is not None:
        notes.append(f"tool_choice: {source_choice} -> {tool_choice!r}")
    if isinstance(source_choice, dict) and source_choice.get("disable_parallel_tool_use"):
        extra_body["parallel_tool_calls"] = False

    max_tokens = None
    for key in MAX_TOKEN_PARAMS:
        if isinstance(p.get(key), int):
            max_tokens = p[key]
            if key != "max_tokens":
                notes.append(f"max_tokens: {key}={max_tokens} -> max_tokens={max_tokens}")
            break
    if max_tokens is None:
        notes.append(f"max_tokens: not set in source -> {DEFAULT_MAX_TOKENS}")

    t = p.get("temperature")
    temperature = float(t) if isinstance(t, int | float) and not isinstance(t, bool) else 0.0
    if "temperature" not in p:
        notes.append("temperature: not set in source -> 0 (deterministic eval)")

    effort = p.get("reasoning_effort")
    if effort in ("low", "medium", "high"):
        reasoning = effort
    elif effort == "minimal":
        reasoning = "low"
    elif REASONING_MODEL.match(site.source_model or ""):
        reasoning = "medium"
        notes.append(f"reasoning: {site.source_model} is a reasoning model -> Nemotron thinking on")
    else:
        reasoning = "off"

    if site.sdk == "anthropic" and "system" in p:
        notes.append("system: Anthropic system= -> leading system message (rendered into each case)")
    if p.get("stream") is True:
        flags.append("streaming: the source streams; Offramp evaluates the non-streaming call")

    response_format = _response_format(p, schemas)
    flags += [
        f"unresolved_schema: {n} (pass its model_json_schema() in schemas)" for n in _markers(response_format)
    ]

    spec = AdapterSpec(
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        reasoning=reasoning,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )
    return Translation(spec, notes, flags)


def spec_v0(site: CallSite, schemas: dict[str, dict[str, Any]] | None = None) -> AdapterSpec:
    return translate(site, schemas).spec


def max_tokens_for(spec: AdapterSpec) -> int:
    """max_tokens actually sent: the answer budget, plus REASONING_ROOM when thinking is on (GC25)."""
    budget = spec.max_tokens or DEFAULT_MAX_TOKENS
    return budget + REASONING_ROOM if spec.reasoning != "off" else budget


def build_messages(spec: AdapterSpec, input_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the spec to a case's rendered messages: system_prompt replaces/sets the system message,
    few_shot goes right after it, user_prefix/user_suffix wrap the last user message (the case's input)."""
    msgs = copy.deepcopy(input_messages)
    head = [msgs.pop(0)] if msgs and msgs[0].get("role") == "system" else []
    if spec.system_prompt is not None:
        head = [system_message(spec.system_prompt)]
    msgs = [*head, *copy.deepcopy(spec.few_shot), *msgs]
    if spec.user_prefix or spec.user_suffix:
        last = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
        if last is None or not isinstance(last.get("content"), str):
            raise ValueError("user_prefix/user_suffix need a final user message with string content")
        last["content"] = spec.user_prefix + last["content"] + spec.user_suffix
    return msgs


def build_request(spec: AdapterSpec, case: EvalCase, tier: Tier | str) -> dict[str, Any]:
    """`TFClient.chat(**build_request(...))` runs one eval case exactly as the migrated code would call it on
    `tier`. The patch composer takes the call's parameters from here too (tier quirks included)."""
    unresolved = _markers([spec.response_format, spec.tools])
    if unresolved:
        raise ValueError(f"spec still holds <pydantic:{unresolved[0]}>: resolve the schema before evaluating")
    if EXPR in _leaves([spec.response_format, spec.tools, spec.tool_choice]):
        raise ValueError(f"spec holds an unresolved {EXPR} value: it cannot be sent")
    tier = Tier(tier)
    tool_choice = spec.tool_choice
    if tool_choice == "required" and not REQUIRED_TOOL_CHOICE_OK[tier]:
        tool_choice = "auto"
    return {
        "role": EVAL_TARGET_ROLE,
        "tier": tier,
        "messages": build_messages(spec, case.input_messages),
        "max_tokens": max_tokens_for(spec),
        "temperature": spec.temperature,
        "json_schema": spec.response_format,
        "json_mode_fallback": spec.json_mode_fallback,
        "tools": spec.tools,
        "tool_choice": tool_choice,
        "reasoning": spec.reasoning,
        "extra_body": dict(spec.extra_body),
    }
