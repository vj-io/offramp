"""Route an incoming ticket to one of HelpHub's support queues."""

from openai import OpenAI

from helphub.models import Queue

TRIAGE_SYSTEM_PROMPT = """You are the ticket router for HelpHub, a customer support platform.
Classify the customer's ticket into exactly one queue:

- billing: charges, invoices, payment methods, subscription pricing
- shipping: delivery status, tracking, delays, lost parcels, address changes
- returns_refunds: returns, exchanges, damaged or wrong items, refund requests
- technical_support: bugs, errors, integrations, API and webhook problems
- account_access: login, password reset, two-factor, locked or compromised accounts
- general: anything else, including feedback, partnerships and sales questions

Reply with the queue name only, exactly one of: billing, shipping, returns_refunds, \
technical_support, account_access, general. No punctuation, no explanation."""


def parse_queue(text: str | None) -> Queue:
    """Map the model's reply to a queue; anything unrecognised goes to `general`."""
    label = (text or "").strip().lower()
    for queue in Queue:
        if queue.value in label:
            return queue
    return Queue.GENERAL


def triage_ticket(client: OpenAI, subject: str, body: str) -> Queue:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=16,
        messages=[
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Subject: {subject}\n\n{body}"},
        ],
    )
    return parse_queue(response.choices[0].message.content)
