"""Summarise a ticket thread for handoff (fixture: anthropic haiku, system and content as text blocks)."""

from anthropic import Anthropic

SUMMARY_SYSTEM = "Summarise the ticket thread for the next agent in at most three bullet points."
client = Anthropic()


def summarize(thread: str) -> str:
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        system=[{"type": "text", "text": SUMMARY_SYSTEM}],
        messages=[{"role": "user", "content": [{"type": "text", "text": thread}]}],
    )
    return resp.content[0].text
