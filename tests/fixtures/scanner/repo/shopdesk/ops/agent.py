"""Pick the next ops tool (fixture: anthropic tools=, client on self, pydantic tool schema)."""

from anthropic import Anthropic
from pydantic import BaseModel


class RefundArgs(BaseModel):
    order_id: str
    amount: float


AGENT_SYSTEM = "You are a support operations agent. Call exactly one tool."
TOOLS = [
    {
        "name": "lookup_order",
        "description": "Fetch an order by id.",
        "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {"name": "issue_refund", "description": "Refund part or all of an order.", "input_schema": RefundArgs.model_json_schema()},
]


class OpsAgent:
    def __init__(self, client: Anthropic | None = None) -> None:
        self.client = client or Anthropic()

    def next_step(self, ticket: str, history: str):
        prompt = f"Ticket:\n{ticket}\n\nHistory so far:\n{history}"
        resp = self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=AGENT_SYSTEM,
            tools=TOOLS,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content
