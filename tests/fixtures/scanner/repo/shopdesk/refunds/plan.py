"""Plan a refund with prorations (fixture: module-level openai call, o3, max_completion_tokens, .format prompt)."""

import os

import openai

REFUND_MODEL = os.getenv("REFUND_MODEL", "o3")
PLAN_PROMPT = """Compute the refund for this subscription change.
Plan: {plan}
Days used this cycle: {days_used}
Show your working, then finish with a final line `ANSWER: <number>`."""


def plan_refund(plan: str, days_used: int) -> str:
    resp = openai.chat.completions.create(
        model=REFUND_MODEL,
        messages=[{"role": "user", "content": PLAN_PROMPT.format(plan=plan, days_used=days_used)}],
        max_completion_tokens=4000,
        reasoning_effort="high",
    )
    return resp.choices[0].message.content
