"""Stream a reply token by token (fixture negative: stream=True needs manual review)."""

import openai

client = openai.OpenAI()
STREAM_SYSTEM = "You are a concise support assistant."


def stream_reply(question: str):
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": STREAM_SYSTEM},
            {"role": "user", "content": question},
        ],
        stream=True,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""
