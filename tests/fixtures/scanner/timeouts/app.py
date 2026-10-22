"""Timeouts a call site's code sets, as Offramp reads them statically (task 29 C). The first is the shape of
idea-reality-mcp's cs_01 (api/main.py at 83281b4): an AsyncAnthropic client with timeout=30.0 built in the
function; the last is a demo-like site with no timeout at all."""

import os

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


# Review I3 (ct_probe.py): timeouts the call does not use are never read (no gate rather than a false one).
SHORT = OpenAI(timeout=5)
REQUEST_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "120"))


def summarise_with_given_client(SHORT, text: str) -> str:  # a parameter shadows the module client
    resp = SHORT.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": text}])
    return resp.choices[0].message.content


def request_timeout_from_env(text: str) -> str:  # the request's own timeout is not a literal
    resp = SHORT.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": text}], timeout=REQUEST_TIMEOUT
    )
    return resp.choices[0].message.content


def request_timeout_none(text: str) -> str:  # timeout=None on the request: no timeout for this call
    resp = SHORT.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": text}], timeout=None)
    return resp.choices[0].message.content


def request_timeout_product(text: str) -> str:  # not a literal: unknown
    resp = SHORT.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": text}], timeout=60 * 5)
    return resp.choices[0].message.content


def read_timeout_only(text: str) -> str:  # httpx.Timeout with per-phase values: not one number
    c = anthropic.Anthropic(timeout=httpx.Timeout(connect=5.0, read=90.0, write=10.0, pool=5.0))
    msg = c.messages.create(model="claude-haiku-4-5", max_tokens=100, messages=[{"role": "user", "content": text}])
    return msg.content[0].text


def client_bound_twice(text: str, slow: bool) -> str:  # which client is used depends on the path
    c = anthropic.Anthropic(timeout=10.0)
    if slow:
        c = anthropic.Anthropic(timeout=600.0)
    msg = c.messages.create(model="claude-haiku-4-5", max_tokens=100, messages=[{"role": "user", "content": text}])
    return msg.content[0].text
