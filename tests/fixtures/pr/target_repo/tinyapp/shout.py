"""Turn a line of text into a headline."""

SHOUT_SYSTEM_PROMPT = "You rewrite text as a short headline."


def shout(client, text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=32,
        messages=[
            {"role": "system", "content": SHOUT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Headline: {text}"},
        ],
    )
    return (response.choices[0].message.content or "").strip().upper()
