"""Timeouts a call site's code sets, as Offramp reads them statically (task 29 C). The first is the shape of
idea-reality-mcp's cs_01 (api/main.py at 83281b4): an AsyncAnthropic client with timeout=30.0 built in the
function; the last is a demo-like site with no timeout at all."""

import anthropic
import httpx
from openai import OpenAI

WAIT_S = 12
module_client = OpenAI(timeout=20)
slow_client = OpenAI(timeout=httpx.Timeout(45.0))
shared = OpenAI()


async def extract_keywords(idea_text: str) -> str:  # idea-reality's cs_01 shape
    client = anthropic.AsyncAnthropic(api_key="x", timeout=30.0)
    message = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        system="Output ONLY a JSON array of strings.",
        messages=[{"role": "user", "content": idea_text}],
    )
    return message.content[0].text


def per_call(text: str) -> str:  # the request's own timeout wins over the client's
    resp = module_client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": text}], timeout=WAIT_S
    )
    return resp.choices[0].message.content


def module_level(text: str) -> str:
    resp = module_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": text}])
    return resp.choices[0].message.content


def httpx_timeout(text: str) -> str:
    resp = slow_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": text}])
    return resp.choices[0].message.content


def no_timeout(ticket: str) -> str:  # demo-like: the SDK default, nothing to compare with
    resp = shared.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": ticket}])
    return resp.choices[0].message.content
