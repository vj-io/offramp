"""Pull structured order and customer fields out of a customer email."""

import json
from typing import Literal

from anthropic import Anthropic
from pydantic import BaseModel, Field, ValidationError


class OrderFields(BaseModel):
    order_id: str | None = Field(description="Order id such as HH-48213, null if not stated")
    customer_name: str | None = Field(description="Full name of the customer as written")
    customer_email: str | None = Field(description="Customer's email address")
    order_date: str | None = Field(description="Order date as YYYY-MM-DD, null if not stated")
    total_amount: float | None = Field(description="Order total as a plain number, no symbol")
    currency: str | None = Field(description="ISO 4217 currency code, e.g. EUR")
    issue: Literal[
        "damaged", "missing_item", "wrong_item", "late_delivery", "billing_error", "other"
    ]


ORDER_SCHEMA = OrderFields.model_json_schema()

EXTRACT_SYSTEM_PROMPT = """You extract structured order data from customer support emails.
Reply with a single JSON object that validates against the JSON schema given in the user message.
Use null for any field the email does not state; never guess.
Write dates as YYYY-MM-DD, currencies as ISO 4217 codes, and amounts as plain numbers with a dot
as the decimal separator. Keep names exactly as written. The email may be in any language.
Do not wrap the JSON in markdown and do not add any commentary."""


class ExtractionError(ValueError):
    """The model's reply was not a JSON object matching `OrderFields`."""


def parse_order_fields(text: str) -> OrderFields:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ExtractionError(f"no JSON object in model reply: {text[:80]!r}")
    try:
        return OrderFields.model_validate_json(text[start : end + 1])
    except ValidationError as exc:
        raise ExtractionError(str(exc)) from exc


def extract_order(client: Anthropic, email: str) -> OrderFields:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=EXTRACT_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"JSON schema:\n{json.dumps(ORDER_SCHEMA)}\n\nEmail:\n{email}",
            }
        ],
    )
    reply = "".join(block.text for block in response.content if block.type == "text")
    return parse_order_fields(reply)
