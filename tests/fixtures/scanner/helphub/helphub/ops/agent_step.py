"""Pick the next tool for HelpHub's order-support agent."""

from typing import Any

from anthropic import Anthropic
from pydantic import BaseModel

from helphub.models import ToolResult

AGENT_SYSTEM_PROMPT = """You are HelpHub's order-support agent. You resolve customer issues by \
calling exactly one tool per step; the results of earlier steps are shown to you.

- lookup_order: use when you need order details (status, items, total, charges) that you do not \
have yet. Only use an order id that appears in the customer's message.
- issue_refund: use only after the order has been looked up and the customer is eligible: \
the item arrived damaged or missing, or the customer was charged twice. Refund no more than the \
amount paid, and never more than 200.00 in one step.
- escalate_to_human: use for legal threats, fraud or chargeback claims, refunds above 200.00, \
abusive messages, requests without any order id, and anything the other tools cannot resolve."""

TOOLS = [
    {
        "name": "lookup_order",
        "description": "Fetch an order's status, items, total and charges from the order system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order id, e.g. HH-48213"},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Refund part or all of an order to the original payment method.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order id, e.g. HH-48213"},
                "amount": {"type": "number", "description": "Refund amount, at most 200.00"},
                "currency": {"type": "string", "description": "ISO 4217 code of the order"},
                "reason": {
                    "type": "string",
                    "enum": ["damaged", "missing_item", "duplicate_charge", "other"],
                },
            },
            "required": ["order_id", "amount", "currency", "reason"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the ticket to a human support agent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "One sentence for the human agent"},
                "priority": {"type": "string", "enum": ["low", "normal", "urgent"]},
            },
            "required": ["reason", "priority"],
        },
    },
]
TOOL_NAMES = {tool["name"] for tool in TOOLS}


class AgentAction(BaseModel):
    tool: str
    args: dict[str, Any]


def next_action(client: Anthropic, ticket_text: str, history: list[ToolResult]) -> AgentAction:
    steps = "\n".join(f"- {step.tool} returned: {step.output}" for step in history) or "(none yet)"
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=AGENT_SYSTEM_PROMPT,
        tools=TOOLS,
        tool_choice={"type": "any"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Customer message:\n{ticket_text}\n\n"
                    f"Tool results so far:\n{steps}\n\n"
                    "Choose the next tool."
                ),
            }
        ],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name in TOOL_NAMES:
            return AgentAction(tool=block.name, args=dict(block.input))
    return AgentAction(
        tool="escalate_to_human",
        args={"reason": "The agent did not return a valid tool call.", "priority": "normal"},
    )
