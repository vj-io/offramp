"""Pull order fields out of an email."""

import json
from typing import Literal

from pydantic import BaseModel, Field


class OrderInfo(BaseModel):
    order_id: str | None = Field(description="Order id such as SH-1234, null if not stated")
    total: float | None = Field(description="Order total as a plain number, null if not stated")
    currency: str | None = Field(description="ISO 4217 currency code, e.g. EUR")
    issue: Literal["damaged", "late", "other"]


EXTRACT_PROMPT = """Extract the order from the email as one JSON object matching the schema.
Use null for anything the email does not state. Reply with the JSON object only."""


def extract(client, email: str):
    return client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system=EXTRACT_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Schema:\n{json.dumps(OrderInfo.model_json_schema())}\n\nEmail:\n{email}",
            }
        ],
    )
