"""Summarise a ticket thread so the next agent can pick it up cold."""

from anthropic import Anthropic

from helphub.models import ThreadMessage

SUMMARY_SYSTEM_PROMPT = """You write handoff notes for HelpHub support agents taking over a ticket.
Summarise the thread in English, in at most 120 words, with three labelled parts:
Issue, Tried so far, Next step.
Quote order ids and amounts exactly as written. Skip greetings and pleasantries.
Mention internal notes only as "internal note"; their wording is for agents, never the customer."""


def summarize_thread(client: Anthropic, messages: list[ThreadMessage]) -> str:
    transcript = "\n".join(f"[{m.sent_at}] {m.role} ({m.author}): {m.text}" for m in messages)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        system=SUMMARY_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Ticket thread, {len(messages)} messages:\n\n{transcript}",
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()
