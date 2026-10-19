"""Route a support ticket to a queue (fixture: openai client, labels joined into a constant, f-string user prompt)."""

from openai import OpenAI

QUEUES = ["billing", "shipping", "returns", "technical", "account", "other"]
MODEL = "gpt-4o-mini"
CLASSIFY_SYSTEM = f"Classify the support ticket into exactly one queue: {', '.join(QUEUES)}. Reply with the queue name only."

_client = OpenAI()


def classify_ticket(subject: str, body: str, client: OpenAI | None = None) -> str:
    client = client or _client
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": f"Subject: {subject}\n\n{body}"},
        ],
        temperature=0,
        max_tokens=5,
    )
    return resp.choices[0].message.content.strip()
