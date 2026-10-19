"""Route a ticket to a queue."""

TRIAGE_PROMPT = """Classify the customer's ticket into exactly one queue:
- billing: charges, invoices, refunds of charges
- shipping: delivery, tracking, delays
- other: anything else
Reply with the queue name only: billing, shipping or other."""


def triage(client, subject: str, body: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=8,
        messages=[
            {"role": "system", "content": TRIAGE_PROMPT},
            {"role": "user", "content": f"Subject: {subject}\n\n{body}"},
        ],
    )
    return response.choices[0].message.content
