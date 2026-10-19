"""FakeLLM: canned-response stand-in for the LLM protocol, for other tasks' unit tests."""

import json
from collections import deque
from typing import Any

from pydantic import BaseModel, ValidationError

from offramp.schemas import ModelOutput, Tier
from offramp.tf.client import ChatResult


class FakeLLM:
    """Queue of canned replies per role; every chat() call is recorded in `calls` (the kwargs passed).

    A reply is a ChatResult (returned as is), ModelOutput, str (the text), pydantic instance or dict/list
    (serialised to JSON text, also set as `parsed`), or an Exception instance (raised, e.g. BudgetExceeded).
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        reasoning: str = "off",
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls.append(
            dict(role=role, tier=tier, messages=messages, max_tokens=max_tokens, temperature=temperature,
                 response_model=response_model, json_schema=json_schema, tools=tools, tool_choice=tool_choice,
                 reasoning=reasoning, extra_body=extra_body)
        )
        queue = self.queues.get(role)
        if not queue:
            raise AssertionError(f"FakeLLM: no canned reply left for role {role!r}")
        reply = queue.popleft()
        if isinstance(reply, BaseException):
            raise reply
        return self._as_result(reply, tier, response_model)

    @staticmethod
    def _as_result(reply: Any, tier: Tier, response_model: type[BaseModel] | None) -> ChatResult:
        if isinstance(reply, ChatResult):
            return reply
        parsed = None
        if isinstance(reply, ModelOutput):
            output = reply
        elif isinstance(reply, str):
            output = ModelOutput(text=reply)
        elif isinstance(reply, BaseModel):
            output, parsed = ModelOutput(text=reply.model_dump_json()), reply
        else:
            output, parsed = ModelOutput(text=json.dumps(reply)), reply
        if parsed is None and response_model is not None and output.text:
            try:
                parsed = response_model.model_validate_json(output.text)
            except ValidationError:
                pass  # like the real client: invalid JSON leaves parsed None
        return ChatResult(output=output, parsed=parsed, model="fake", tier=tier)
