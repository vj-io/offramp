"""Code that calls json.loads near an LLM reply without parsing the reply text itself, or that accepts a reply
which is not JSON (task 28 review, I2): the JSON gate must leave these sites alone. Then three shapes that do
parse the reply: the call nested deeper than the parse, a strip in its own statement, and a ``` literal that
is only in the prompt (so nothing strips a fence)."""

import json
import re

from openai import OpenAI

client = OpenAI()
CANNED = {"yes": '{"ok": true}', "no": '{"ok": false}'}


def fallback_to_text(q: str) -> dict:  # json.loads in a try, the reply text used as the fallback
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    text = resp.choices[0].message.content
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"answer": text}


def regex_extracts_object(q: str) -> dict:  # the code pulls the {...} out of surrounding prose
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    reply = resp.choices[0].message.content
    m = re.search(r"\{.*\}", reply, re.S)
    return json.loads(m.group(0)) if m else {}


def slices_from_brace(q: str) -> list:  # reply[reply.find("["):]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    reply = resp.choices[0].message.content
    return json.loads(reply[reply.find("[") : reply.rfind("]") + 1])


def logs_the_response_object(q: str) -> str:  # json.loads of the response's own JSON, not the reply text
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    meta = json.loads(resp.model_dump_json())
    print(meta["usage"])
    return resp.choices[0].message.content


def wraps_then_loads(q: str) -> dict:  # json.loads of a document built from the reply: always parses
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    reply = resp.choices[0].message.content
    payload = json.dumps({"reply": reply})
    return json.loads(payload)


def reply_as_lookup_key(q: str) -> dict:  # the reply is a key; json.loads reads a fixed table entry
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    reply = resp.choices[0].message.content.strip().lower()
    return json.loads(CANNED.get(reply, "{}"))


class Summariser:
    def __init__(self) -> None:
        self.client = OpenAI()
        self.options_json = '{"max_words": 80}'

    def summarise(self, text: str) -> str:  # free text; json.loads only reads the object's own settings
        resp = self.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": text}])
        self.last_reply = resp.choices[0].message.content
        options = json.loads(self.options_json)
        return self.last_reply[: options["max_words"] * 8]


def call_in_try_parse_after(q: str) -> list:  # the call is nested deeper than the parse
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    except Exception:
        return []
    text = resp.choices[0].message.content
    return json.loads(text)


def strip_in_its_own_statement(q: str) -> list:
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": q}])
    raw = resp.choices[0].message.content
    raw = raw.strip()
    return json.loads(raw)


def fence_in_prompt_only(q: str) -> list:  # a ``` literal in the prompt; the reply is json.loads'd as is
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Answer as a JSON array, never inside ``` fences.\n" + q}],
    )
    return json.loads(resp.choices[0].message.content)
