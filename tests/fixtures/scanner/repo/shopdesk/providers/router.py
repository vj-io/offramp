"""Provider adapters (fixture: one variable name bound to two SDKs in different functions and classes)."""

from anthropic import Anthropic
from openai import OpenAI


def ask_openai(question: str) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": question}])
    return resp.choices[0].message.content


def ask_claude(question: str) -> str:
    client = Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5", max_tokens=200, messages=[{"role": "user", "content": question}]
    )
    return resp.content[0].text


class OpenAIBackend:
    def __init__(self) -> None:
        self.client = OpenAI()

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content


class ClaudeBackend:
    def __init__(self) -> None:
        self.client = Anthropic()

    def complete(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model="claude-sonnet-4-5", max_tokens=300, messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text
