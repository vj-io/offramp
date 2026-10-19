"""Pick the next tool for the order agent."""

AGENT_PROMPT = """You are the shop's order agent. Call exactly one tool per step.
- lookup_order: when you need an order's details and the customer gave an order id.
- escalate_to_human: for threats, fraud, abuse, or when there is no order id."""

TOOLS = [
    {
        "name": "lookup_order",
        "description": "Fetch an order's status and items.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order id, e.g. SH-1234"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the ticket to a human agent.",
        "input_schema": {
            "type": "object",
            "properties": {"priority": {"type": "string", "enum": ["low", "normal", "urgent"]}},
            "required": ["priority"],
        },
    },
]


def next_tool(client, ticket_text: str, history: list[dict]):
    steps = "\n".join(f"- {s['tool']} returned: {s['output']}" for s in history) or "(none yet)"
    return client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        system=AGENT_PROMPT,
        tools=TOOLS,
        tool_choice={"type": "any"},
        messages=[
            {
                "role": "user",
                "content": f"Customer message:\n{ticket_text}\n\nTool results so far:\n{steps}\n\nChoose the next tool.",
            }
        ],
    )
