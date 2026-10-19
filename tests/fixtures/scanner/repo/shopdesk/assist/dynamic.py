"""Continue a chat (fixture negative: messages built at runtime, so the prompt is dynamic)."""

from openai import OpenAI

client = OpenAI()


def continue_chat(history: list[tuple[str, str]], message: str) -> str:
    messages = [{"role": "system", "content": "You are a helpful support agent."}]
    for role, text in history:
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": message})
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return resp.choices[0].message.content
