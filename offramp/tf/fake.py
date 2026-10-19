"""FakeLLM: canned-response stand-in for the LLM protocol, for other tasks' unit tests."""

import json
from collections import deque
from typing import Any

from pydantic import BaseModel

from offramp.schemas import ModelOutput, Tier
from offramp.tf.client import ChatResult, plan_structured, validate_structured
from offramp.tf.tiers import get_tier


class FakeLLM:
    """Queue of canned replies per role; every chat() call is recorded in `calls` (the kwargs passed).

    A reply is a ChatResult (returned as is), ModelOutput, str (the text), pydantic instance or dict/list
    (serialised to JSON text), or an Exception instance (raised, e.g. BudgetExceeded).

    `parsed` and `json_path` come from the same code TFClient uses: with response_model or json_schema the
    reply text is validated (a model instance, or the decoded JSON; None if invalid, with no repair retry);
    with neither, `parsed` is None, as with the real client.
    """

    def __init__(self, replies: dict[str, list[Any]] | None = None):
        self.queues: dict[str, deque[Any]] = {role: deque(items) for role, items in (replies or {}).items()}
        self.calls: list[dict[str, Any]] = []

    def add(self, role: str, *replies: Any) -> "FakeLLM":
        self.queues.setdefault(role, deque()).extend(replies)
        return self

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
        self.calls.append(
            dict(role=role, tier=tier, messages=messages, max_tokens=max_tokens, temperature=temperature,
                 response_model=response_model, json_schema=json_schema,
                 json_mode_fallback=json_mode_fallback, tools=tools, tool_choice=tool_choice,
                 reasoning=reasoning, extra_body=extra_body)
        )
        queue = self.queues.get(role)
        if not queue:
            raise AssertionError(f"FakeLLM: no canned reply left for role {role!r}")
        reply = queue.popleft()
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, ChatResult):
            return reply
        if isinstance(reply, ModelOutput):
            output = reply
        elif isinstance(reply, str):
            output = ModelOutput(text=reply)
        elif isinstance(reply, BaseModel):
            output = ModelOutput(text=reply.model_dump_json())
        else:
            output = ModelOutput(text=json.dumps(reply))
        info = get_tier(tier)
        plan = plan_structured(json_schema, response_model, json_mode_fallback, info.supports_json_schema)
        parsed = None
        if plan.json_path != "none":
            parsed, _ = validate_structured(output.text or "", plan.schema, response_model)
        return ChatResult(
            output=output, parsed=parsed, model=info.model_id, tier=info.tier, json_path=plan.json_path
        )
