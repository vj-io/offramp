"""Draft a customer reply (fixture: openai Responses API, instructions + f-string input, client on self)."""

from openai import OpenAI

REPLY_INSTRUCTIONS = "Write a short, friendly reply to the customer. Never promise refunds."


class ReplyDrafter:
    def __init__(self) -> None:
        self.client = OpenAI()

    def draft(self, customer_name: str, message: str) -> str:
        resp = self.client.responses.create(
            model="gpt-4.1-mini",
            instructions=REPLY_INSTRUCTIONS,
            input=f"Customer {customer_name} wrote:\n{message}",
            max_output_tokens=400,
        )
        return resp.output_text
