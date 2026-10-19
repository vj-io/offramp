"""Send SMS notifications (fixture negative: openai imported but unused; a non-LLM messages.create)."""

import openai  # noqa: F401  (kept for a future feature)
from openai import OpenAI  # noqa: F401


class SmsNotifier:
    def __init__(self, twilio_client) -> None:
        self.twilio = twilio_client

    def notify(self, phone: str, body: str) -> None:
        self.twilio.messages.create(to=phone, from_="+15550100", body=body)
