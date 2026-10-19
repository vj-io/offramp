"""Write a one-line greeting."""

GREET_SYSTEM_PROMPT = "You write one-line greetings."


def greet(client, name: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=32,
        messages=[
            {"role": "system", "content": GREET_SYSTEM_PROMPT},
            {"role": "user", "content": f"Greet {name}."},
        ],
    )
    return (response.choices[0].message.content or "").strip()
