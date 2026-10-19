"""Extract order fields from an email (fixture: anthropic, pydantic JSON schema in the prompt, .format template)."""

import json

import anthropic
from pydantic import BaseModel


class OrderFields(BaseModel):
    order_id: str
    customer_email: str
    items: list[str]


ORDER_SCHEMA = OrderFields.model_json_schema()
EXTRACT_SYSTEM = f"Extract the order fields from the email. Reply with JSON matching this schema:\n{json.dumps(ORDER_SCHEMA, indent=2)}"
EXTRACT_USER = 'Email:\n{email}\n\nReturn JSON only, e.g. {{"order_id": "A-1"}}.'

client = anthropic.Anthropic()


def extract_order(email: str) -> OrderFields:
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=800,
        system=EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": EXTRACT_USER.format(email=email)}],
    )
    return OrderFields.model_validate_json(resp.content[0].text)
