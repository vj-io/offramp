# ruff: noqa: E501
"""One-off Token Factory capability probe; results in docs/plan/tf-probe.md."""
import json
import os
import time

from openai import OpenAI

c = OpenAI(api_key=os.environ["NEBIUS_API_KEY"], base_url=os.environ["NEBIUS_BASE_URL"], timeout=120)
M = {"lightning": "nvidia/Nemotron-3_5-Lightning", "super": "nvidia/nemotron-3-super-120b-a12b", "ultra": "nvidia/Nemotron-3-Ultra-550b-a55b"}
schema = {"name": "lbl", "strict": True, "schema": {"type": "object", "properties": {"label": {"type": "string", "enum": ["billing", "bug"]}}, "required": ["label"], "additionalProperties": False}}
tools = [{"type": "function", "function": {"name": "lookup_order", "description": "Look up order", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}]
msg = [{"role": "user", "content": "Ticket: 'I was charged twice for invoice 991.' Classify."}]
def run(tag, model, **kw):
    t = time.time()
    try:
        r = c.chat.completions.create(model=model, messages=kw.pop("messages", msg), max_tokens=kw.pop("max_tokens", 400), temperature=0, **kw)
        m = r.choices[0].message
        rc = getattr(m, "reasoning_content", None) or (m.model_extra or {}).get("reasoning_content")
        print(json.dumps({"tag": tag, "ms": int((time.time()-t)*1000), "content": (m.content or "")[:160], "think_tag": "<think>" in (m.content or ""), "reasoning_content": bool(rc), "tool_calls": [tc.function.name + tc.function.arguments for tc in (m.tool_calls or [])], "usage": [r.usage.prompt_tokens, r.usage.completion_tokens], "finish": r.choices[0].finish_reason}))
    except Exception as e:
        print(json.dumps({"tag": tag, "error": str(e)[:300]}))
for tier, model in M.items():
    run(f"{tier}/plain", model)
    run(f"{tier}/json_schema", model, response_format={"type": "json_schema", "json_schema": schema})
    run(f"{tier}/json_object", model, response_format={"type": "json_object"}, messages=[{"role": "user", "content": msg[0]["content"] + ' Reply JSON {"label": "billing"|"bug"}.'}])
    run(f"{tier}/tools", model, tools=tools, tool_choice="auto", messages=[{"role": "user", "content": "Where is my order A-1042?"}])
    run(f"{tier}/think_off", model, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    run(f"{tier}/think_on", model, extra_body={"chat_template_kwargs": {"enable_thinking": True}}, max_tokens=1500)
    run(f"{tier}/effort_low", model, reasoning_effort="low", max_tokens=1500)
