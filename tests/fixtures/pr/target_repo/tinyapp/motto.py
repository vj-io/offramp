"""Write the company motto."""


def motto(client) -> str:
    response = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": "Write a five-word company motto."}],
    )
    return response.choices[0].message.content or ""
