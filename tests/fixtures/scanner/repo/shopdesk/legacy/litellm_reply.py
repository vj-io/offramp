"""Legacy reply paths through litellm (fixture: litellm.completion and a from-imported completion)."""

import litellm
from litellm import completion as llm_completion

TONE = "polite"


def polite_reply(message: str) -> str:
    resp = litellm.completion(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": f"Rewrite this reply in a {TONE} tone:\n{message}"}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def translate(message: str, language: str) -> str:
    resp = llm_completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Translate to " + language + ":\n" + message}],
    )
    return resp.choices[0].message.content
