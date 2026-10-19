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
from openai import OpenAI
from pydantic import BaseModel

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


def _response_format(
    json_schema: dict[str, Any] | None, response_model: type[BaseModel] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(OpenAI response_format, bare JSON Schema). json_schema may be a full response_format, its inner
    {"name", "schema"} payload, or a bare JSON Schema; with none given, the schema is response_model's."""
    js = json_schema if json_schema is not None else response_model.model_json_schema()
    if js.get("type") == "json_schema":
        return js, js["json_schema"]["schema"]
    if isinstance(js.get("schema"), dict):
        return {"type": "json_schema", "json_schema": js}, js["schema"]
    name = response_model.__name__ if response_model else "response"
    return {"type": "json_schema", "json_schema": {"name": name, "schema": js}}, js


def _validate(
    text: str, schema: dict[str, Any], response_model: type[BaseModel] | None
) -> tuple[Any, str | None]:
    """(parsed, None) when valid, else (None, error message)."""
    try:
        if response_model is not None:
            return response_model.model_validate_json(text), None
        obj = json.loads(text)
        jsonschema.validate(obj, schema)
        return obj, None
    except (ValueError, jsonschema.ValidationError) as e:  # JSONDecodeError and pydantic's are ValueErrors
        return None, str(e)[:1500]


def _repair_messages(messages: list[dict[str, Any]], bad: str, error: str, schema: dict[str, Any]):
    ask = (
        "Your previous reply is not valid against the required JSON schema.\n"
        f"Validation error: {error}\n\nJSON schema:\n{json.dumps(schema)}\n\n"
        "Reply again with only the corrected JSON, no commentary."
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
        self._sdk: OpenAI | None = None

    def _openai(self) -> OpenAI:
        if self._sdk is None:  # built lazily so live=False (cassette replay) never needs a key
            if not self.api_key:
                raise RuntimeError("NEBIUS_API_KEY is not set (live=False replays the cache without a key)")
            sdk = OpenAI(
                api_key=self.api_key,
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
            self._sdk = sdk
        return self._sdk

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
        resp = self._openai().chat.completions.create(model=info.model_id, messages=messages, **params)
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        reasoning: str = "off",
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        info = get_tier(tier)
        params: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        json_path: JsonPath = "none"
        schema: dict[str, Any] = {}
        if response_model is not None or json_schema is not None:
            response_format, schema = _response_format(json_schema, response_model)
            json_path = "json_schema" if info.supports_json_schema else "json_object"
            params["response_format"] = (
                response_format if json_path == "json_schema" else {"type": "json_object"}
            )
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        params["extra_body"] = _merge(reasoning_params(info.tier, reasoning)["extra_body"], extra_body or {})

        call = self._complete(role, info, messages, params)
        parsed, repaired = None, False
        if json_path != "none":
            parsed, error = _validate(call.output.text or "", schema, response_model)
            if error and json_path == "json_object":  # exactly one repair retry (Global Constraint 6)
                repaired = True
                retry_messages = _repair_messages(messages, call.output.text or "", error, schema)
                retry = self._complete(role, info, retry_messages, params)
                parsed, _ = _validate(retry.output.text or "", schema, response_model)
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
            json_path=json_path,
            repaired=repaired,
        )
