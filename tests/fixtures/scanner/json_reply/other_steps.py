"""Re-review 1 (R1-b, R1-c, R1-d): steps on the reply other than strips and the one fence removal the gate
replays exactly (idea-reality's guarded line filter). Each of these leaves the site to the judge (None),
except a plain strip in its own statement ("exact") and json.loads(..., strict=False) ("exact_lenient")."""

import json
import re

import anthropic
from openai import OpenAI

client = OpenAI()
claude = anthropic.Anthropic()


def single_quotes_to_double(q: str) -> dict:  # replace without a fence: the code accepts {'a': 1}
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    text = resp.choices[0].message.content
    return json.loads(text.replace("'", '"'))


def drops_comment_lines(q: str) -> dict:  # a line filter for another reason than a fence
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    lines = resp.choices[0].message.content.split("\n")
    lines = [ln for ln in lines if not ln.startswith("#")]
    return json.loads("\n".join(lines))


def lowered_then_stripped(q: str) -> dict:  # a transform two names back
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    raw = resp.choices[0].message.content.lower()
    text = raw.strip()
    return json.loads(text)


def strips_backticks(q: str) -> list:  # a fence removal the gate does not replay
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    text = resp.choices[0].message.content.strip().strip("`")
    return json.loads(text)


def re_sub_fence(q: str) -> list:  # likewise
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    text = re.sub(r"```(?:json)?", "", resp.choices[0].message.content)
    return json.loads(text)


def handler_falls_through(q: str) -> list:  # the handler sets a default and carries on
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    text = resp.choices[0].message.content
    try:
        items = json.loads(text)
    except ValueError:
        items = []
    return items


def removeprefix_only(q: str) -> list:  # half a fence removal: the trailing ``` stays
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    reply = resp.choices[0].message.content
    return json.loads(reply.strip().removeprefix("```json"))


def prefix_and_suffix(q: str) -> list:  # removeprefix("```json") + removesuffix("```"): misses "```\n"
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    reply = resp.choices[0].message.content
    return json.loads(reply.strip().removeprefix("```json").removesuffix("```"))


def unguarded_line_filter(q: str) -> list:  # drops fence lines anywhere, not only from a fenced reply
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    lines = resp.choices[0].message.content.strip().split("\n")
    lines = [ln for ln in lines if not ln.strip().startswith("```")]
    return json.loads("\n".join(lines))


def if_else_one_branch_transforms(q: str) -> list:  # one branch lower-cases the reply
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    if q.startswith("!"):
        text = resp.choices[0].message.content.lower()
    else:
        text = resp.choices[0].message.content
    return json.loads(text)


def optional_content_or_empty(q: str) -> list:  # the usual guard for OpenAI's Optional content
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    text = (resp.choices[0].message.content or "").strip()
    return json.loads(text)


def retry_loop_continue(q: str) -> list:  # a reply that is not JSON is retried
    for _ in range(3):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
        try:
            return json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("no JSON")


def anthropic_three_statements(q: str) -> list:  # text = resp.content[0].text; text = text.strip(); loads
    resp = claude.messages.create(model="claude-haiku-4-5", max_tokens=200, messages=[{"role": "user", "content": q}])
    text = resp.content[0].text
    text = text.strip()
    return json.loads(text)


def lenient_strict_false(q: str) -> dict:  # control characters (raw newlines) allowed inside strings
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    text = resp.choices[0].message.content
    return json.loads(text, strict=False)


def loads_with_a_hook(q: str) -> dict:  # another keyword: not replayed by the gate
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    return json.loads(resp.choices[0].message.content, parse_float=str)


def guarded_filter_without_strip(q: str) -> list:  # idea-reality's form, but `ln.startswith` unstripped
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [ln for ln in lines if not ln.startswith("```")]
        raw = "\n".join(lines).strip()
    return json.loads(raw)
