"""Token Factory client: the one way anything in Offramp calls Nemotron (Global Constraint 3).

OpenAI SDK pointed at Nebius, a content-addressed cassette cache in front (Global Constraint 7), per-run
budget (Global Constraint 8) and the JSON-contract handling of Global Constraint 6.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import jsonschema
from openai import APIStatusError, OpenAI
from pydantic import BaseModel

from offramp.integrations import keys
from offramp.schemas import ModelOutput, Tier, ToolCall
from offramp.tf.budget import Budget
from offramp.tf.cache import Cache, cache_key
from offramp.tf.tiers import TierInfo, get_tier, reasoning_params

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
MAX_RETRIES = 3  # SDK-level exponential backoff on 429 / 5xx / connection errors / timeouts
TIMEOUT_S = 120.0

JsonPath = Literal["json_schema", "json_object", "none"]


class CacheMiss(Exception):
    """live=False and the request is not in the cache."""


class ChatResult(BaseModel):
    """cost_usd is tokens x published price (so cache hits report the cost the call represents);
    actual spend is whatever the Budget was charged. repaired is True iff a repair retry was made."""

    output: ModelOutput
    parsed: Any | None = None  # response_model instance, or the decoded JSON for json_schema; None if invalid
    model: str
    tier: Tier
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    json_path: JsonPath = "none"
    repaired: bool = False
    cache_key: str | None = None  # the cached response's key: TFClient.forget(key) drops a reply found bad


class LLM(Protocol):
    def chat(
        self,
        *,
        role: str,
        tier: Tier,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        response_model: type[BaseModel] | None = None,
        json_schema: dict[str, Any] | None = None,
        json_mode_fallback: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        reasoning: str = "off",
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult: ...


def strip_think(text: str) -> str:
    """Final answer only: drop <think>...</think> (also a lone </think>, or a truncated unclosed <think>)."""
    if "think>" not in text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


BAD_SCHEMA = "invalid JSON schema: "


@dataclass(frozen=True)
class StructuredPlan:
    """What structured output a call asks for. `schema` None means parse-only (any JSON object)."""

    response_format: dict[str, Any] | None
    schema: dict[str, Any] | None
    json_path: JsonPath


def plan_structured(
    json_schema: dict[str, Any] | None,
    response_model: type[BaseModel] | None,
    json_mode_fallback: bool,
    supports_json_schema: bool,
) -> StructuredPlan:
    """Pick response_format, validation schema and json_path for one call.

    json_schema may be an OpenAI response_format ({"type": "json_schema", ...}, {"type": "json_object"} or
    {"type": "text"}: AdapterSpec.response_format passes straight through), its inner {"name", "schema"}
    payload, or a bare JSON Schema; with none given the schema is response_model's. The json_object path is
    used for {"type": "json_object"}, for json_mode_fallback=True (AdapterSpec.json_mode_fallback) and for
    tiers without json_schema support.
    """
    if json_schema is not None and json_schema.get("type") == "text":
        json_schema = None
    if json_schema is None and response_model is None:
        return StructuredPlan(None, None, "none")
    full: dict[str, Any] | None = None
    if json_schema is not None and json_schema.get("type") == "json_object":
        schema = response_model.model_json_schema() if response_model else None
    else:
        js = json_schema if json_schema is not None else response_model.model_json_schema()
        if js.get("type") == "json_schema":
            full, schema = js, js["json_schema"]["schema"]
        elif isinstance(js.get("schema"), dict):
            full, schema = {"type": "json_schema", "json_schema": js}, js["schema"]
        else:
            name = response_model.__name__ if response_model else "response"
            full, schema = {"type": "json_schema", "json_schema": {"name": name, "schema": js}}, js
    if full is None or json_mode_fallback or not supports_json_schema:
        return StructuredPlan({"type": "json_object"}, schema, "json_object")
    return StructuredPlan(full, schema, "json_schema")


def validate_structured(
    text: str, schema: dict[str, Any] | None, response_model: type[BaseModel] | None
) -> tuple[Any, str | None]:
    """(parsed, None) when valid, else (None, error message). Never raises on bad output or a bad schema."""
    try:
        if response_model is not None:
            return response_model.model_validate_json(text), None
        obj = json.loads(text)
        if schema is None:
            if not isinstance(obj, dict):
                raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
        else:
            jsonschema.validate(obj, schema)
        return obj, None
    except jsonschema.SchemaError as e:  # the schema itself is broken: no reply can satisfy it
        return None, f"{BAD_SCHEMA}{e.message}"
    except (ValueError, jsonschema.ValidationError) as e:  # JSONDecodeError and pydantic's are ValueErrors
        return None, str(e)[:1500]


def _repair_messages(messages: list[dict[str, Any]], bad: str, error: str, schema: dict[str, Any] | None):
    shown = f"\n\nJSON schema:\n{json.dumps(schema)}" if schema is not None else ""
    ask = (
        "Your previous reply is not valid.\n"
        f"Validation error: {error}{shown}\n\n"
        "Reply again with only the corrected JSON object, no commentary."
    )
    return [*messages, {"role": "assistant", "content": bad}, {"role": "user", "content": ask}]


def _tool_args(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        value = None
    return value if isinstance(value, dict) else {"__invalid_arguments__": raw}  # keep it, fail comparisons


@dataclass
class _Call:
    output: ModelOutput
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cached: bool

    @classmethod
    def from_entry(cls, entry: dict[str, Any], cached: bool) -> "_Call":
        msg = ((entry["response"].get("choices") or [{}])[0]).get("message") or {}
        tool_calls = [
            ToolCall(name=tc["function"]["name"], arguments=_tool_args(tc["function"].get("arguments")))
            for tc in msg.get("tool_calls") or []
        ]
        text = strip_think(msg.get("content") or "")  # reasoning_content is deliberately never read
        usage = entry.get("usage") or {}
        return cls(
            output=ModelOutput(text=None if tool_calls and not text else text, tool_calls=tool_calls),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=entry.get("latency_ms", 0),
            cached=cached,
        )


class TFClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path | None = None,
        live: bool = True,
        budget: Budget | None = None,
        http_client: httpx.Client | None = None,
    ):
        self.api_key = api_key or os.environ.get("NEBIUS_API_KEY")
        self.base_url = base_url or os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL
        self.cache = Cache(cache_dir or Path(os.environ.get("OFFRAMP_DATA_DIR", "./var")) / "tf-cache")
        self.live = live
        self.budget = budget
        self._http_client = http_client  # tests inject httpx.MockTransport here
        self._conn: tuple[OpenAI, str] | None = None  # (SDK client, key it carries); never logged

    def _connection(self) -> tuple[OpenAI, str]:
        """The SDK client and the key it carries, as one pair: a thread that has sent a request with this
        client must know which key that was even if another thread has since moved on to the fallback."""
        conn = self._conn
        if conn is None:  # built lazily so live=False (cassette replay) never needs a key
            key = keys.current("nebius", self.api_key)
            if not key:
                raise RuntimeError("NEBIUS_API_KEY is not set (live=False replays the cache without a key)")
            sdk = OpenAI(
                api_key=key,
                base_url=self.base_url,
                max_retries=MAX_RETRIES,
                timeout=TIMEOUT_S,
                http_client=self._http_client,
            )
            if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
                try:
                    from langsmith.wrappers import wrap_openai
                except ImportError:  # optional extra not installed: run untraced
                    pass
                else:
                    sdk = wrap_openai(sdk)
            self._conn = conn = sdk, key
        return conn  # the pair held here, not a re-read of the attribute another thread may have reset

    def _openai(self) -> OpenAI:
        return self._connection()[0]

    def _create(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Any:
        """chat.completions.create; a rejected / out-of-credit primary key moves the process to the
        NEBIUS_API_KEY_FALLBACK and retries the same request once (Global Constraint 28)."""
        sdk, key = self._connection()  # the key THIS request carries, whatever other threads do meanwhile
        try:
            return sdk.chat.completions.create(model=model, messages=messages, **params)
        except APIStatusError as e:
            if not keys.switch_to_fallback("nebius", key, *keys.status_and_body(e)):
                raise
            self._conn = None  # rebuilt on the fallback key
            sdk, _ = self._connection()
            return sdk.chat.completions.create(model=model, messages=messages, **params)

    def forget(self, key: str) -> None:
        """Drop a cached response its caller found unusable (a judge reply off the rubric), so the next run
        sends the request again instead of replaying it forever."""
        self.cache.forget(key)

    def _complete(
        self, role: str, info: TierInfo, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> _Call:
        key = cache_key(info.model_id, messages, params)
        entry = self.cache.get(key)
        if entry is not None:
            return _Call.from_entry(entry, cached=True)
        if not self.live:
            raise CacheMiss(
                f"no cached response for role={role!r} model={info.model_id} key={key} in {self.cache.dir}"
            )
        if self.budget:
            self.budget.check(role, info.tier)
        started = time.perf_counter()
        resp = self._create(info.model_id, messages, params)
        latency_ms = round((time.perf_counter() - started) * 1000)
        raw = resp.model_dump(mode="json", exclude_none=True)
        usage = raw.get("usage") or {}
        tin, tout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        entry = {
            "request": {"model": info.model_id, "messages": messages, "params": params},
            "response": raw,
            "latency_ms": latency_ms,
            "usage": {"prompt_tokens": tin, "completion_tokens": tout},
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.cache.put(key, entry)  # cached before charging: a call that trips a cap is never paid twice
        if self.budget:
            cost = (tin * info.input_usd_per_m + tout * info.output_usd_per_m) / 1_000_000
            self.budget.charge(role, info.tier, tin, tout, cost)
        return _Call.from_entry(entry, cached=False)

    def chat(
        self,
        *,
        role: str,
        tier: Tier,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        response_model: type[BaseModel] | None = None,
        json_schema: dict[str, Any] | None = None,
        json_mode_fallback: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        reasoning: str = "off",
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        info = get_tier(tier)
        plan = plan_structured(json_schema, response_model, json_mode_fallback, info.supports_json_schema)
        params: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if plan.response_format is not None:
            params["response_format"] = plan.response_format
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        # Nemotron thinks by default (docs/plan/tf-probe.md): enable_thinking is ALWAYS sent, and `reasoning`
        # (not a caller's extra_body) decides its value.
        params["extra_body"] = _merge(extra_body or {}, reasoning_params(info.tier, reasoning)["extra_body"])

        call = self._complete(role, info, messages, params)
        parsed, repaired = None, False
        if plan.json_path != "none":
            parsed, error = validate_structured(call.output.text or "", plan.schema, response_model)
            if error and plan.json_path == "json_object" and not error.startswith(BAD_SCHEMA):
                repaired = True  # exactly one repair retry (Global Constraint 6)
                retry_messages = _repair_messages(messages, call.output.text or "", error, plan.schema)
                retry = self._complete(role, info, retry_messages, params)
                parsed, _ = validate_structured(retry.output.text or "", plan.schema, response_model)
                call = _Call(
                    output=retry.output,
                    input_tokens=call.input_tokens + retry.input_tokens,
                    output_tokens=call.output_tokens + retry.output_tokens,
                    latency_ms=call.latency_ms + retry.latency_ms,
                    cached=call.cached and retry.cached,
                )
        return ChatResult(
            output=call.output,
            parsed=parsed,
            model=info.model_id,
            tier=info.tier,
            latency_ms=call.latency_ms,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            cost_usd=(call.input_tokens * info.input_usd_per_m + call.output_tokens * info.output_usd_per_m)
            / 1_000_000,
            cached=call.cached,
            json_path=plan.json_path,
            repaired=repaired,
            cache_key=None if repaired else cache_key(info.model_id, messages, params),
        )
