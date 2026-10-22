"""How calling code parses an LLM reply as JSON. The first three are the shapes of idea-reality-mcp's cs_01,
cs_02 and cs_04 (api/main.py, api/report.py at 83281b4), cut down to the call and what reads its reply."""

import json
import logging
from json import loads as parse_json

import anthropic
from openai import OpenAI

logger = logging.getLogger(__name__)
openai_client = OpenAI()


async def extract_keywords(idea_text: str) -> list[str] | None:  # cs_01: fences stripped, then json.loads
    try:
        client = anthropic.AsyncAnthropic(api_key="x", timeout=30.0)
        message = await client.messages.create(
            model="claude-haiku-4-5",
            temperature=0,
            max_tokens=200,
            system="Output ONLY a JSON array of strings. No explanation, no markdown.",
            messages=[{"role": "user", "content": idea_text}],
        )

        raw = message.content[0].text.strip()

        # Strip markdown code fences if present (e.g. ```json ... ```)
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        keywords = json.loads(raw)
        if not isinstance(keywords, list) or len(keywords) < 2:
            return None
        return [str(k).strip() for k in keywords][:8]
    except Exception:
        logger.exception("keyword extraction failed")
        return None


async def pivot_hints(idea_text: str, evidence_text: str) -> list[str] | None:  # cs_02: JSONDecodeError caught
    user_prompt = f"""Idea: {idea_text}

Evidence:
{evidence_text}"""
    try:
        client = anthropic.AsyncAnthropic(api_key="x", timeout=30.0)
        message = await client.messages.create(
            model="claude-haiku-4-5",
            temperature=0,
            max_tokens=500,
            system="Generate 3 pivot suggestions as a JSON array of strings.",
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = message.content[0].text.strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        hints = json.loads(raw)
        if not isinstance(hints, list) or len(hints) < 2:
            logger.warning("invalid list: %s", raw[:100])
            return None
        return [str(h).strip() for h in hints[:3]]
    except json.JSONDecodeError:
        logger.warning("non-JSON: %s", raw[:200])
        return None


async def search_angles(idea_text: str) -> list[str]:  # cs_04: no temperature, a fallback on any failure
    try:
        client = anthropic.AsyncAnthropic(api_key="x", timeout=30.0)
        message = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            system="Generate 3-5 search angles. Output ONLY a JSON array of strings. No explanation.",
            messages=[{"role": "user", "content": idea_text}],
        )

        raw = message.content[0].text.strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        angles = json.loads(raw)
        if not isinstance(angles, list) or len(angles) < 2:
            return [idea_text]
        return [str(a).strip() for a in angles][:5]
    except Exception:
        return [idea_text]


def tags_direct(text: str) -> list[str]:  # the reply straight into json.loads, no fence handling
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Tags for this text, as a JSON array:\n{text}"}],
    )
    return json.loads(resp.choices[0].message.content)


def tags_through_a_helper(text: str) -> list[str]:  # a helper defined in the function, an aliased loads
    def parse(reply: str) -> list[str]:
        return parse_json(reply)

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Tags for this text, as a JSON array:\n{text}"}],
    )
    content = resp.choices[0].message.content
    return parse(content)


def summary(text: str) -> str:  # free text: the reply is returned as is
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarise:\n{text}"}],
    )
    settings = json.loads('{"max": 3}')  # a json.loads that never sees the reply
    return resp.choices[0].message.content[: settings["max"] * 100]


def lookup_order(question: str) -> dict:  # tool-call arguments are not the reply text
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        tools=[{"type": "function", "function": {"name": "get_order", "parameters": {"type": "object"}}}],
    )
    call = resp.choices[0].message.tool_calls[0]
    return json.loads(call.function.arguments)
