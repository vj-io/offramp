"""Adapter Agent (PRD §6.4): Nemotron Ultra turns a call site's deterministic translation into Adapter v0.

translate.py maps everything mechanical (tools, tool_choice, system, max_tokens, response_format). The agent
decides what needs judgement: reasoning on or off, structured-output mode, and an output-format instruction.
Deterministic guards then keep its choices inside what the calling code can still parse.
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from offramp.agents.translate import (
    REASONING_ROOM,
    Translation,
    site_schema,
    strict_json_schema,
    translate,
    unresolved,
)
from offramp.schemas import Adapter, AdapterSpec, CallSite, Tier
from offramp.tf.client import LLM, _repair_messages, validate_structured
from offramp.tf.tiers import SUPPORTS_JSON_SCHEMA

ADAPTER_ROLE = "adapter"
ADAPTER_MAX_TOKENS = 1500  # thinking off; the decision is a few hundred tokens
LIGHTNING_LEAK = (
    "lightning_reasoning_leak: with thinking on, Lightning writes its reasoning into the reply text "
    "(docs/plan/tf-probe.md), so Lightning may fail this site on format"
)
LIGHTNING_TOOL_CHOICE = (
    "lightning_tool_choice_auto: Lightning loops tool calls under tool_choice=required, so it is sent 'auto' "
    "and may answer in text where the source forced a tool call"
)
INVALID_OUTPUT = (
    "adapter_agent_invalid_output: Adapter Agent reply invalid after one repair; spec is the translation"
)
JSON_ONLY = "Reply with the JSON only: no other text and no code fence."  # contract.reply_json sites
# Sanity of a model-written adapter (Adapter Agent and Iterator alike), checked before it is evaluated:
# A loop: a phrase of up to MAX_NGRAM words, or a unit of up to MAX_UNIT characters holding a letter or
# digit, repeated in a row REPEATS or more times beyond its first occurrence and beyond the source's run.
# A single character needs CHAR_REPEATS: ids, masks, blanks and hashes repeat one (100000000000, ____).
REPEATS = 10
CHAR_REPEATS = 40
MAX_NGRAM = 4
MAX_UNIT = 80
SCAN = 20_000  # characters of a text the character-level check reads
_UNIT_LOOP = re.compile(rf"(.{{1,{MAX_UNIT}}}?)\1{{{REPEATS - 1},}}", re.S)
GROWTH, GROWTH_PLUS = 2, 2000  # a field may reach 2x the source prompt (system + user template) + 2,000 chars
_NUMBERED = re.compile(r"(?m)^\s*\d{1,2}[.)]\s")  # a numbered rule starting a line (what the source counts)
_RULE_LINE = re.compile(r"(?m)^\s*(?:\d{1,2}[.)]|[-*•])\s")  # numbered or bulleted (what a rewrite counts)
_INLINE_RULE = re.compile(r"(?<![\w.])(\d{1,2})[.)](?=\s)")  # "7. " anywhere: rewrites put rules on one line
REFUSED = "adapter_agent_refused"

ADAPTER_SYSTEM = f"""You are Offramp's Adapter Agent. Offramp moves one LLM call site in a Python repo \
from a closed model (OpenAI or Anthropic SDK) to NVIDIA Nemotron on Nebius Token Factory, an \
OpenAI-compatible chat.completions API, and evaluates the migrated call before opening a pull \
request. The patched code sends exactly what the evaluation sends, and the code around the call does \
not change: the reply must stay parseable by it.

A deterministic translator has already mapped the call (spec_v0): Anthropic tools to OpenAI function \
tools, tool_choice, system= to a system message, max_tokens, response_format and temperature. You decide \
only what it cannot:

reasoning: "off", or "low"/"medium"/"high" for on (Token Factory has a single thinking switch, so the \
three levels behave the same). Nemotron thinks by default; Offramp always sends the switch explicitly \
and adds {REASONING_ROOM} tokens of room when thinking is on. Thinking costs latency and output tokens. \
Turn it on when the task needs multi-step reasoning (arithmetic, policies with several rules, \
multi-constraint logic) or the source model is itself a reasoning model. Keep it off for \
classification, extraction, tool choice, summaries and short answers. On the Lightning tier, thinking \
leaks reasoning prose into the reply text.

output_mode: "text" keeps the call's own output format. "json_schema" constrains the reply to the site's \
JSON schema with server-side guided decoding. "json_object" asks for any JSON object and validates it \
against the schema afterwards, with one repair retry. Choose a JSON mode only when the calling code \
parses the reply as a JSON object; never for labels, free text or tool calls. Without a known output \
schema, json_schema falls back to json_object.

user_suffix: an optional instruction appended to the end of the user message to pin the exact output \
format the calling code parses, for example "Reply with the queue name only." Leave it empty when the \
prompt already pins the format unambiguously. Never change the task, add facts or contradict the system \
prompt.

notes: up to 5 short facts for the reviewer about this migration. risk_flags: up to 5 concrete ways \
Nemotron's behaviour could differ from the source model at this call site, each "short_id: explanation".

Reply with one JSON object only."""


class AdapterDecision(BaseModel):
    """The Adapter Agent's reply: only the choices the deterministic translation cannot make."""

    reasoning: Literal["off", "low", "medium", "high"]
    output_mode: Literal["text", "json_schema", "json_object"]
    user_suffix: str = Field("", max_length=500)
    notes: list[str] = Field(default_factory=list, max_length=5)
    risk_flags: list[str] = Field(default_factory=list, max_length=5)


def _facts(site: CallSite, tr: Translation, schemas: dict[str, dict[str, Any]] | None) -> str:
    facts = {
        "site": site.model_dump(
            mode="json",
            include={
                "id",
                "file",
                "function",
                "sdk",
                "call",
                "source_model",
                "prompt_source",
                "template",
                "params",
                "flags",
            },
        ),
        "contract": site.contract.shown(),
        "spec_v0": tr.spec.model_dump(mode="json"),
        "translation_notes": tr.notes,
        "translation_risk_flags": tr.risk_flags,
        "output_schema_known": site_schema(site, schemas) is not None,
        "json_schema_supported": {tier.value: ok for tier, ok in SUPPORTS_JSON_SCHEMA.items()},
    }
    return json.dumps(facts, indent=1, ensure_ascii=False)


def _word_runs(text: str) -> dict[tuple[str, ...], int]:
    """Each phrase of up to MAX_NGRAM words (case-folded) repeated in a row, and its longest run."""
    words, runs = re.findall(r"\w+", text.casefold()), {}
    for n in range(1, MAX_NGRAM + 1):
        streak = 0
        for i in range(n, len(words) + 1):
            if i < len(words) and words[i] == words[i - n]:
                streak += 1
                continue
            if streak >= n:
                unit = tuple(words[i - streak - n : i - streak])
                runs[unit] = max(runs.get(unit, 0), streak // n + 1)
            streak = 0
    return runs


def _unit_run(text: str, unit: str) -> int:
    return max((len(m[0]) // len(unit) for m in re.finditer(f"(?:{re.escape(unit)})+", text)), default=0)


def _loop(text: str, source: str) -> str | None:
    """ "'utility' 752 times in a row" for a loop the model added (see REPEATS), else None. Word phrases first
    (case and punctuation aside), then character units, which catch scripts without spaces and sentences
    longer than MAX_NGRAM words. ponytail: the unit scan is one regex over the first SCAN characters (a later
    loop still trips the growth cap); a loop with small variations between repeats is not caught."""
    ours, theirs = _word_runs(text), _word_runs(source)
    for unit, run in ours.items():
        if run - max(theirs.get(unit, 0), 1) >= REPEATS:
            return f"{' '.join(unit)!r} {run} times in a row"
    for m in _UNIT_LOOP.finditer(text[:SCAN]):
        unit, run = m[1], len(m[0]) // len(m[1])
        need = CHAR_REPEATS if len(unit) == 1 else REPEATS
        if re.search(r"\w", unit) and run - max(_unit_run(source, unit), 1) >= need:
            return f"{unit!r} {run} times in a row"
    return None


def _rules(text: str) -> int:
    """How many rules a rewrite states: numbered or bulleted lines, or numbered markers on one line."""
    return max(len(_RULE_LINE.findall(text)), len(set(_INLINE_RULE.findall(text))))


def adapter_problem(site: CallSite, spec: AdapterSpec) -> str | None:
    """Why a model-written adapter must not be evaluated (nor shipped), or None: a loop the model added to
    one of its texts (_loop), a text grown past GROWTH x the source prompt + GROWTH_PLUS characters, or a
    rewritten system prompt stating fewer rules than the source's (numbers turned into bullets pass)."""
    t = site.template
    system = (t.system or "") if t else ""
    source = f"{system}\n{t.user if t else ''}"
    cap = GROWTH * len(source) + GROWTH_PLUS
    texts = [
        ("system_prompt", spec.system_prompt or ""),
        ("user_prefix", spec.user_prefix),
        ("user_suffix", spec.user_suffix),
        *(("few_shot", str(m.get("content") or "")) for m in spec.few_shot),  # one example at a time
    ]
    for name, text in texts:
        if loop := _loop(text, source):
            return f"{name} repeats {loop}"
    shots = "\n".join(str(m.get("content") or "") for m in spec.few_shot)
    for name, text in [*texts[:3], ("few_shot", shots)]:
        if len(text) > cap:
            return (
                f"{name} is {len(text):,} characters, over the cap of {cap:,} (2x the source prompt + 2,000)"
            )
    # ponytail: the source's numbered rules only. Its bulleted lines are often a list of labels or options
    # (the demo's cs_01 queues), and the demo's recorded rewrite folds them into a sentence: counting them
    # too would refuse a recorded adapter and change recorded requests, which waits for a ruling.
    had = len(_NUMBERED.findall(system))
    if spec.system_prompt is not None and (have := _rules(spec.system_prompt)) < had:
        return f"system_prompt has {have} rules, the source's system prompt {had}"
    return None


def _apply(
    site: CallSite, tr: Translation, decision: AdapterDecision, schemas: dict[str, dict[str, Any]] | None
) -> Adapter:
    """Decision -> Adapter v0, kept inside what the calling code can parse."""
    contract = site.contract
    spec = tr.spec.model_copy(deep=True)
    notes, flags = list(tr.notes), list(tr.risk_flags)

    spec.reasoning = decision.reasoning
    if contract.reasoning_heavy and spec.reasoning == "off":
        spec.reasoning = "medium"
        notes.append("reasoning: kept on (contract is reasoning_heavy)")
    if decision.user_suffix.strip():
        spec.user_suffix = "\n\n" + decision.user_suffix.strip()
    if contract.reply_json and not spec.tools and "json" not in spec.user_suffix.lower():
        spec.user_suffix = (
            f"{spec.user_suffix} {JSON_ONLY}" if spec.user_suffix.strip() else f"\n\n{JSON_ONLY}"
        )
        notes.append("user_suffix: asks for JSON alone (the calling code runs json.loads on the reply)")

    mode = decision.output_mode
    json_reply = contract.task_type == "json_extraction" or contract.output in ("json", "json_schema")
    if mode != "text" and (spec.tools or not (json_reply or spec.response_format)):
        why = "the call uses tools" if spec.tools else "the code does not parse a JSON object"
        notes.append(f"output_mode {mode}: ignored, {why}")
    elif mode != "text":
        current = spec.response_format or {}
        found = site_schema(site, schemas)
        if current.get("type") == "json_schema":
            schema_format = current
        else:
            schema_format = strict_json_schema(found[1], found[0]) if found else None
        if schema_format is None:
            flags.append(
                f"no_output_schema: {mode} requested but no schema is known; JSON is parsed, not validated"
            )
        spec.response_format = schema_format or {"type": "json_object"}
        spec.json_mode_fallback = mode == "json_object" or schema_format is None

    if spec.reasoning != "off":
        flags.append(LIGHTNING_LEAK)
    if spec.tool_choice == "required":
        flags.append(LIGHTNING_TOOL_CHOICE)
    notes += decision.notes
    flags += decision.risk_flags
    return Adapter(site_id=site.id, version=0, spec=spec, notes=notes, risk_flags=list(dict.fromkeys(flags)))


def adapt(site: CallSite, llm: LLM, schemas: dict[str, dict[str, Any]] | None = None) -> Adapter:
    """Adapter v0 for one labelled call site. `schemas` maps Pydantic class names to model_json_schema().

    One Ultra call (role "adapter"); an invalid reply gets exactly one repair retry, and if that is invalid
    too the deterministic translation is used as v0 with a risk flag. A site whose call cannot be evaluated
    (translate.unresolved) gets the translation without a call. BudgetExceeded and CacheMiss propagate.
    """
    if site.contract is None:
        raise ValueError(f"{site.id}: no contract; label the site (scanner.label_contracts) first")
    tr = translate(site, schemas)
    if problems := unresolved(tr.spec):  # build_request would refuse every case: no Ultra call for it
        adapter = _apply(site, tr, AdapterDecision(reasoning=tr.spec.reasoning, output_mode="text"), schemas)
        adapter.notes.append(
            f"adapter agent: not called, the call cannot be evaluated ({'; '.join(problems)})"
        )
        return adapter
    messages = [
        {"role": "system", "content": ADAPTER_SYSTEM},
        {"role": "user", "content": _facts(site, tr, schemas)},
    ]
    kw = dict(
        role=ADAPTER_ROLE,
        tier=Tier.ULTRA,
        max_tokens=ADAPTER_MAX_TOKENS,
        temperature=0.0,
        response_model=AdapterDecision,
        reasoning="off",
    )
    result = llm.chat(messages=messages, **kw)
    decision, repaired = result.parsed, False
    if decision is None and not result.repaired:  # json_schema path: the client does not repair, we do once
        bad = result.output.text or ""
        _, error = validate_structured(bad, None, AdapterDecision)
        retry = llm.chat(
            messages=_repair_messages(messages, bad, error, AdapterDecision.model_json_schema()), **kw
        )
        decision, repaired = retry.parsed, True

    if decision is None:
        adapter = _apply(site, tr, AdapterDecision(reasoning=tr.spec.reasoning, output_mode="text"), schemas)
        adapter.risk_flags.append(INVALID_OUTPUT)
        return adapter
    adapter = _apply(site, tr, decision, schemas)
    if problem := adapter_problem(site, adapter.spec):  # never evaluated: the translation is v0 instead
        adapter = _apply(site, tr, AdapterDecision(reasoning=tr.spec.reasoning, output_mode="text"), schemas)
        adapter.risk_flags.append(f"{REFUSED}: {problem}")
        return adapter
    if repaired:
        adapter.notes.append(
            "adapter agent: reply repaired once (first reply was not valid JSON for the schema)"
        )
    return adapter
