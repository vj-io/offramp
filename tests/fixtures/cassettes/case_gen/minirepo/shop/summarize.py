"""Summarise a ticket thread for handoff."""

SUMMARY_PROMPT = """Write a handoff note for the next support agent in at most 60 words:
Issue, Tried so far, Next step."""


def summarize(client, messages: list[dict]):
    transcript = "\n".join(f"{m['role']}: {m['text']}" for m in messages)
    return client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        system=SUMMARY_PROMPT,
        messages=[{"role": "user", "content": f"Thread, {len(messages)} messages:\n\n{transcript}"}],
    )
