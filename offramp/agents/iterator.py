"""Iterator Agent (PRD §6.6): Nemotron Ultra turns failing dev cases into the next Adapter version.

Holdout guard (Global Constraint 10), at the type level: the Iterator accepts only `DevCase` views, and a
DevCase cannot hold a holdout case id (the id's GC10 hash decides, not a field someone can set). Every
Iterator records the ids it was shown in `seen_case_ids`, which the leakage harness test checks.

The agent may change only the prompt side of the call (system prompt, user prefix/suffix, few-shot examples),
the reasoning switch, the output mode and the answer budget. Tools, tool_choice, temperature and the rest
stay as the Adapter Agent set them, and the Adapter Agent's own guards (adapter._apply) keep the output mode
inside what the calling code parses. Two more guards come from recorded replies: few-shot examples that copy
a shown case's input are dropped, and a JSON site's example replies must be valid for the site.
"""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from offramp.agents.adapter import LIGHTNING_LEAK, AdapterDecision, _apply
from offramp.agents.translate import Translation, site_schema
from offramp.schemas import (
    Adapter,
    BaselineResult,
    CallSite,
    CandidateResult,
    EvalCase,
    ModelOutput,
    Split,
    Tier,
    TierResult,
)
from offramp.tf.client import LLM, _repair_messages, validate_structured

ITERATOR_ROLE = "iterator"
ITERATOR_MAX_TOKENS = 4000  # thinking off; a rewritten system prompt plus a few examples
MAX_CASES = 10  # PRD §6.6: up to 10 failing dev cases per iteration
MAX_CHARS = 3000  # per message shown; ponytail: plain truncation, summarise long inputs if a site needs it
# An example whose input words (outside the prompt template) are at least this share of one shown case's input
# words is a copy. Recorded replies: copies 0.89-1.0; new examples 0.62 and 0.8 (a 5-word paraphrase).
# ponytail: word-set containment; a fuzzy matcher if paraphrased copies slip through
COPY_OVERLAP = 0.85

ITERATOR_SYSTEM = """You are Offramp's Iterator Agent. Offramp moves one LLM call site in a Python repo \
from a closed model to NVIDIA Nemotron on Nebius Token Factory and evaluates the migrated call against \
reference outputs before opening a pull request. The current adapter was evaluated on the tier named in the \
facts and failed some development cases. Propose the next adapter version: fix what the failing cases have \
in common without breaking the cases that already pass. The code around the call does not change, so the \
reply must stay exactly as parseable by it as before.

You may change only these fields:
- system_prompt: the complete replacement system message, or null to keep the current one. Keep every rule \
of the original; you may clarify, reorder or add instructions.
- user_prefix / user_suffix: text placed before / after the user message (the case input). Return the \
complete new values ("" for none).
- few_shot: example exchanges inserted after the system message, as user/assistant pairs. Each user message \
is shaped like a real input; each assistant message is exactly the reply format the code parses. Return the \
complete new list. Write new examples: never copy a failing case's input with its reference answer.
- reasoning: "off", or "low"/"medium"/"high" for thinking on (the three behave the same). Thinking helps \
multi-step arithmetic and multi-rule policies and costs latency and tokens; on the Lightning tier it leaks \
reasoning prose into the reply.
- output_mode: "text" keeps the current format. "json_schema" (guided decoding against the site's schema) or \
"json_object" (any JSON object, validated afterwards) only when the code parses a JSON object.
- max_tokens: the answer budget, or null to keep it; room for thinking is added automatically.
Tools, tool_choice, temperature and every other parameter stay as they are.

Each failing case shows its input messages before the adapter is applied, the reference output it must \
match, the candidate's output (or the call error), and the scorer's error category and detail.

diagnosis: what the failures have in common, in 1-3 sentences, decided before the changes.
change_log: 1-5 short entries, one per change, in plain words (e.g. "Added a worked example of the proration \
rule").

Reply with one JSON object only."""


def split_for(case_id: str) -> Split:
    """Global Constraint 10: deterministic 70/30 dev/holdout split by hash of the case id."""
    return "holdout" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100 < 30 else "dev"


class DevCase(BaseModel):
    """A failing dev case as the Iterator sees it. A holdout case id cannot be put in one."""

    model_config = ConfigDict(frozen=True)

    id: str
    input_messages: list[dict[str, Any]]
    reference: ModelOutput | None = None
    candidate: ModelOutput | None = None
    error: str | None = None  # the candidate call's error, when it returned no output
    error_category: str | None = None
    detail: str = ""

    @field_validator("id")
    @classmethod
    def _dev_only(cls, case_id: str) -> str:
        if split_for(case_id) != "dev":
            raise ValueError(f"{case_id} is a holdout case: the Iterator never sees holdout (GC10)")
        return case_id


def failing_dev_cases(
    result: TierResult,
    cases: Sequence[EvalCase],
    references: Mapping[str, BaselineResult],
    candidates: Sequence[CandidateResult],
    limit: int = MAX_CASES,
) -> list[DevCase]:
    """Up to `limit` failing cases of a dev TierResult, taking turns over error categories so one dominant
    category does not hide the others."""
    if result.split != "dev":
        raise ValueError(f"only a dev TierResult feeds the Iterator, got split={result.split!r}")
    by_id = {c.id: c for c in cases}
    calls = {r.case_id: r for r in candidates}
    failing = [s for s in result.case_scores if not s.passed]
    turn: Counter[str | None] = Counter()
    order = []
    for i, s in enumerate(failing):
        order.append((turn[s.error_category], i))
        turn[s.error_category] += 1
    picked = [failing[i] for _, i in sorted(order)[:limit]]
    views = []
    for s in picked:
        ref, call = references.get(s.case_id), calls.get(s.case_id)
        views.append(
            DevCase(
                id=s.case_id,
                input_messages=by_id[s.case_id].input_messages,
                reference=ref.output if ref else None,
                candidate=call.output if call else None,
                error=call.error if call else None,
                error_category=s.error_category,
                detail=s.detail,
            )
        )
    return views


class FewShotMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class IteratorDecision(BaseModel):
    """The Iterator's reply: the next values of the fields it may change."""

    diagnosis: str = Field(max_length=1500)
    system_prompt: str | None = Field(None, max_length=8000)
    user_prefix: str = Field("", max_length=1000)
    user_suffix: str = Field("", max_length=1000)
    few_shot: list[FewShotMessage] = Field(default_factory=list, max_length=6)
    reasoning: Literal["off", "low", "medium", "high"]
    output_mode: Literal["text", "json_schema", "json_object"]
    max_tokens: int | None = Field(None, ge=16, le=16000)
    change_log: list[str] = Field(min_length=1, max_length=5)


def _cut(text: str | None) -> str | None:
    return text if text is None or len(text) <= MAX_CHARS else text[:MAX_CHARS] + " …[truncated]"


def _output(o: ModelOutput | None) -> dict[str, Any] | None:
    return None if o is None else {"text": _cut(o.text), "tool_calls": [t.model_dump() for t in o.tool_calls]}


def _shown(case: DevCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "input_messages": [{**m, "content": _cut(m.get("content"))} for m in case.input_messages],
        "reference": _output(case.reference),
        "candidate": _output(case.candidate),
        "error": case.error,
        "error_category": case.error_category,
        "detail": case.detail,
    }


def _facts(
    adapter: Adapter,
    site: CallSite,
    cases: Sequence[DevCase],
    tier: Tier,
    error_counts: Mapping[str, int],
    schemas: dict[str, dict[str, Any]] | None,
) -> str:
    spec = adapter.spec.model_dump(
        mode="json",
        include={
            "system_prompt",
            "user_prefix",
            "user_suffix",
            "few_shot",
            "response_format",
            "json_mode_fallback",
            "tools",
            "tool_choice",
            "reasoning",
            "max_tokens",
        },
    )
    facts = {
        "tier": Tier(tier).value,
        "site": site.model_dump(
            mode="json", include={"id", "file", "function", "sdk", "source_model", "template", "params"}
        ),
        "contract": site.contract.model_dump(mode="json"),
        "output_schema_known": site_schema(site, schemas) is not None,
        "current_adapter": {"version": adapter.version, "spec": spec, "last_change_log": adapter.change_log},
        "failing_dev_cases": {
            "count_by_error_category": dict(error_counts),
            "shown": [_shown(c) for c in cases],
        },
    }
    return json.dumps(facts, indent=1, ensure_ascii=False)


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _copies(example: str, inputs: list[set[str]], template: set[str]) -> bool:
    """True when the example's own words (the template's removed) mostly come from one shown case's input."""
    own = _words(example) - template
    return bool(own) and any(len(own & case) / len(own) >= COPY_OVERLAP for case in inputs)


def _last_user(case: DevCase) -> str:
    return next((str(m.get("content")) for m in reversed(case.input_messages) if m.get("role") == "user"), "")


def _system_rendered_per_case(site: CallSite) -> bool:
    t = site.template
    return bool(t and t.system and any("{" + v + "}" in t.system for v in t.variables))


class Iterator:
    """Callable Iterator Agent for one run. `seen_case_ids` lists every case id it was shown, in order."""

    def __init__(self, llm: LLM, schemas: dict[str, dict[str, Any]] | None = None):
        self.llm = llm
        self.schemas = schemas
        self.seen_case_ids: list[str] = []

    def __call__(
        self,
        adapter: Adapter,
        site: CallSite,
        failing: Sequence[DevCase],
        tier: Tier,
        error_counts: Mapping[str, int] | None = None,
    ) -> Adapter | None:
        """Adapter(version + 1, tier, change_log) from up to MAX_CASES failing dev cases, or None when Ultra's
        reply is still invalid after one repair. BudgetExceeded, CacheMiss and provider errors propagate."""
        if not all(isinstance(c, DevCase) for c in failing):
            raise TypeError(
                "the Iterator takes DevCase views only (offramp.agents.iterator.failing_dev_cases)"
            )
        shown = list(failing[:MAX_CASES])
        self.seen_case_ids.extend(c.id for c in shown)
        if error_counts is None:
            error_counts = Counter(c.error_category for c in shown)
        messages = [
            {"role": "system", "content": ITERATOR_SYSTEM},
            {"role": "user", "content": _facts(adapter, site, shown, tier, error_counts, self.schemas)},
        ]
        kw = dict(
            role=ITERATOR_ROLE,
            tier=Tier.ULTRA,
            max_tokens=ITERATOR_MAX_TOKENS,
            temperature=0.0,
            response_model=IteratorDecision,
            reasoning="off",
        )
        result = self.llm.chat(messages=messages, **kw)
        decision = result.parsed
        # json_schema path: the client does not repair, so the Iterator does, once
        if decision is None and not result.repaired:
            bad = result.output.text or ""
            _, error = validate_structured(bad, None, IteratorDecision)
            retry = self.llm.chat(
                messages=_repair_messages(messages, bad, error, IteratorDecision.model_json_schema()), **kw
            )
            decision = retry.parsed
        if decision is None:
            return None
        return self._next(adapter, site, decision, Tier(tier), shown)

    def _few_shot(
        self, site: CallSite, d: IteratorDecision, shown: list[DevCase]
    ) -> tuple[list | None, str | None]:
        """The examples to use (None keeps the current ones) and a note on what was refused.

        Recorded live: Ultra copied failing dev inputs into examples (memorising dev instead of fixing the
        cause: dev passes, holdout fails, the tier is abandoned) and wrote "{" as a JSON example reply."""
        roles = [m.role for m in d.few_shot]
        if roles != ["user", "assistant"] * (len(roles) // 2):
            return None, "iterator: few_shot ignored, examples must be user/assistant pairs"
        pairs = [d.few_shot[i : i + 2] for i in range(0, len(d.few_shot), 2)]
        contract = site.contract
        if contract.task_type == "json_extraction" or contract.output in ("json", "json_schema"):
            found = site_schema(site, self.schemas)
            if any(validate_structured(a.content, found and found[1], None)[1] for _, a in pairs):
                return (
                    None,
                    "iterator: few_shot ignored, an example reply is not a valid JSON reply for the site",
                )
        t = site.template
        template = _words(re.sub(r"\{\w+\}", " ", f"{t.system or ''} {t.user}")) if t else set()
        inputs = [_words(_last_user(c)) for c in shown]  # _copies drops template words on the example side
        kept = [p for p in pairs if not _copies(p[0].content, inputs, template)]
        note = None
        if len(kept) < len(pairs):
            dropped = len(pairs) - len(kept)
            note = f"iterator: {dropped} few-shot example(s) dropped, they copy a failing dev case's input"
        return [m.model_dump() for p in kept for m in p], note

    def _next(
        self, adapter: Adapter, site: CallSite, d: IteratorDecision, tier: Tier, shown: list[DevCase]
    ) -> Adapter:
        spec = adapter.spec.model_copy(deep=True)
        notes: list[str] = []
        if d.system_prompt is not None:
            if _system_rendered_per_case(site):
                notes.append(
                    "iterator: system_prompt ignored, the call's system message is rendered per case"
                )
            else:
                spec.system_prompt = d.system_prompt.strip()
        spec.user_prefix = d.user_prefix.strip() + "\n\n" if d.user_prefix.strip() else ""
        spec.user_suffix = "\n\n" + d.user_suffix.strip() if d.user_suffix.strip() else ""
        few_shot, note = self._few_shot(site, d, shown)
        if few_shot is not None:
            spec.few_shot = few_shot
        if note:
            notes.append(note)
        if d.max_tokens is not None:
            spec.max_tokens = d.max_tokens
        # The Adapter Agent's guards: reasoning_heavy keeps thinking on; JSON modes only where the code parses
        # JSON, built from the site's resolved schema; Lightning flags follow the new spec.
        guarded = _apply(
            site,
            Translation(spec, [], []),
            AdapterDecision(reasoning=d.reasoning, output_mode=d.output_mode),
            self.schemas,
        )
        kept_flags = [f for f in adapter.risk_flags if f != LIGHTNING_LEAK]
        return Adapter(
            site_id=site.id,
            version=adapter.version + 1,
            tier=tier,
            spec=guarded.spec,
            notes=[*adapter.notes, *notes, *(f"iterator: {n}" for n in guarded.notes)],
            risk_flags=list(dict.fromkeys([*kept_flags, *guarded.risk_flags])),
            change_log=d.change_log,
        )
