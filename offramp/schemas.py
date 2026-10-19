"""Shared contracts between Offramp components (PRD §6, §7).

Every agent input/output and every SSE payload is one of these models. Components
import from here; nobody redefines these shapes locally.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Tier(StrEnum):
    """Logical Nemotron tiers. offramp/tf/tiers.py maps each to a Token Factory model id."""

    LIGHTNING = "lightning"
    SUPER = "super"
    ULTRA = "ultra"


TIER_ORDER: list[Tier] = [Tier.LIGHTNING, Tier.SUPER, Tier.ULTRA]  # cheapest first

TaskType = Literal["classification", "json_extraction", "tool_call", "free_text", "reasoning"]
Decision = Literal["MIGRATE", "KEEP_CLOSED", "MANUAL_REVIEW"]
Split = Literal["dev", "holdout"]


class Stage(StrEnum):
    QUEUED = "QUEUED"
    CLONED = "CLONED"
    SCANNED = "SCANNED"
    BASELINED = "BASELINED"
    EVALUATING = "EVALUATING"
    ITERATING = "ITERATING"
    VERDICTS = "VERDICTS"
    PATCHED = "PATCHED"
    PR_OPENED = "PR_OPENED"
    DONE = "DONE"
    FAILED = "FAILED"


STAGE_ORDER: list[Stage] = list(Stage)[:-1]  # FAILED is terminal, reachable from any stage


# ---------------------------------------------------------------- scanner (§6.1)


class Contract(BaseModel):
    task_type: TaskType
    output: Literal["text", "label", "json_schema", "json", "tool_call"]
    tools: bool = False
    streaming: bool = False
    latency_budget_ms: int | None = None
    reasoning_heavy: bool = False
    labels: list[str] = Field(default_factory=list)  # classification label set, when known
    json_schema: dict[str, Any] | None = None  # expected output schema, when known


class PromptTemplate(BaseModel):
    """Prompt rendered as OpenAI-style messages. `{name}` placeholders come from f-strings."""

    system: str | None = None
    user: str  # e.g. "Classify this ticket:\n{ticket_text}"
    variables: list[str] = Field(default_factory=list)


class CallSite(BaseModel):
    id: str  # "cs_01", stable order: file path then line
    file: str  # repo-relative posix path
    lines: list[int]  # [start, end], 1-based inclusive
    kind: Literal["chat", "embedding"] = "chat"
    sdk: Literal["openai", "anthropic", "litellm", "langchain"]
    call: str  # e.g. "client.chat.completions.create"
    function: str | None = None  # enclosing function name
    source_model: str | None = None
    prompt_source: Literal["literal", "fstring", "variable", "dynamic"]
    prompt_ref: str | None = None  # module constant name holding the prompt, if any
    template: PromptTemplate | None = None  # None when prompt_source == "dynamic"
    params: dict[str, Any] = Field(default_factory=dict)  # tools, tool_choice, response_format, stream,
    # max_tokens, temperature, system — literal values resolved from the module; "<expr>" when unresolvable
    contract: Contract | None = None  # filled by Super labelling
    confidence: float = 1.0
    flags: list[str] = Field(default_factory=list)  # MANUAL_REVIEW, NEEDS_RUNTIME_CAPTURE, EMBEDDINGS


class ScanResult(BaseModel):
    repo: str
    commit_sha: str | None = None
    sites: list[CallSite]


# ---------------------------------------------------------------- cases & baselines (§6.2, §6.3)


class EvalCase(BaseModel):
    id: str  # f"{site_id}:{nnn}"
    site_id: str
    variables: dict[str, str] = Field(default_factory=dict)  # values for template.variables
    input_messages: list[dict[str, Any]]  # rendered OpenAI-format messages (system first, if any)
    expected_kind: Literal["label", "json", "tool_call", "text", "number"]
    tags: list[str] = Field(default_factory=list)  # length:long, locale:de, malformed, empty, unicode, ...
    split: Split  # deterministic: sha256(id) % 100 < 30 -> holdout
    source: Literal["repo_fixture", "synthetic", "adversarial"] = "synthetic"


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class ModelOutput(BaseModel):
    """Canonical output shape shared by closed baselines and Token Factory candidates."""

    text: str | None = None  # final text, reasoning/<think> stripped
    tool_calls: list[ToolCall] = Field(default_factory=list)


class BaselineResult(BaseModel):
    """Expected output for one case (PLAN Amendment A).

    source="reference": ground truth known by construction — model is "reference:seed" (expected output chosen
    first, input written around it), "reference:oracle" (deterministic reference code) or "reference:human"
    (written/verified by a human expert). Latency/tokens/cost are None: nothing was measured.
    source="closed_model": recorded from the original closed model (scripts/record_baselines.py, needs keys).
    """

    case_id: str
    source: Literal["reference", "closed_model"] = "reference"
    model: str
    output: ModelOutput
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    verified_by: list[str] = Field(default_factory=list)  # e.g. ["blind:nemotron-ultra", "human:tendem"]
    recorded_at: datetime


class CandidateResult(BaseModel):
    case_id: str
    tier: Tier
    adapter_version: int
    output: ModelOutput | None  # None on error
    error: str | None = None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False


# ---------------------------------------------------------------- adapter & iteration (§6.4, §6.6)


class AdapterSpec(BaseModel):
    """Executable description of how a call site runs on Token Factory.

    The eval runner applies this to each case's input_messages; the patch composer later turns
    the winning spec into code. Both read the same object, so what was evaluated is what ships.
    """

    system_prompt: str | None = None  # replaces/sets the system message when not None
    user_prefix: str = ""
    user_suffix: str = ""
    few_shot: list[dict[str, Any]] = Field(default_factory=list)  # messages inserted after system
    response_format: dict[str, Any] | None = None  # OpenAI response_format ({"type":"json_schema",...})
    json_mode_fallback: bool = False  # True -> json_object + Pydantic/jsonschema validation + 1 repair
    tools: list[dict[str, Any]] | None = None  # OpenAI function-calling tool schemas
    tool_choice: Any | None = None
    reasoning: Literal["off", "low", "medium", "high"] = "off"
    max_tokens: int | None = None
    temperature: float = 0.0
    extra_body: dict[str, Any] = Field(default_factory=dict)


class Adapter(BaseModel):
    site_id: str
    version: int  # 0 = Adapter Agent output, n = Iterator loop n
    tier: Tier | None = None  # tier the iteration targeted; None for v0
    spec: AdapterSpec
    patch: str | None = None  # unified diff; set by patch composer for MIGRATE sites
    notes: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    change_log: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- scoring & verdicts (§6.5, §6.7)


class CaseScore(BaseModel):
    case_id: str
    passed: bool
    score: float  # 0..1 for deterministic checks; judge rubric mean 1..5 for free_text
    detail: str = ""
    error_category: str | None = None  # e.g. invalid_json, missing_key, wrong_label, wrong_tool


class TierMetrics(BaseModel):
    p50_ms: int
    p95_ms: int
    tokens_in: int
    tokens_out: int
    cost_per_1k_calls_usd: float


class TierResult(BaseModel):
    site_id: str
    tier: Tier
    adapter_version: int
    split: Split
    passed: bool
    score: float  # parity fraction 0..1 (free_text: judge mean 1..5)
    metrics: TierMetrics
    failing_case_ids: list[str] = Field(default_factory=list)
    case_scores: list[CaseScore] = Field(default_factory=list)


class Verdict(BaseModel):
    site_id: str
    decision: Decision
    tier: Tier | None = None  # set when MIGRATE
    model: str | None = None  # Token Factory model id when MIGRATE
    parity_pct: float | None = None  # holdout parity (0..100) of the chosen/best tier
    best_tier: Tier | None = None  # KEEP_CLOSED: best tier tried
    best_score: float | None = None
    iterations: int = 0  # adapter iterations used for the chosen/best tier
    latency_delta_ms: int | None = None  # p50 after - before; None when "before" was never measured
    cost_before_per_1k: float | None = None
    cost_before_note: str | None = None  # e.g. "est.: gpt-4o-mini $0.15/$0.60 per 1M (openai.com/api/pricing,
    # fetched via Tavily 2026-10-19) x tokens measured on the Token Factory call"
    cost_after_per_1k: float | None = None
    sunset_risk: str | None = None  # deprecation notice for the source model found via Tavily, with URL
    reason: str


# ---------------------------------------------------------------- run + SSE events (§5.2, §9)


EventType = Literal[
    "stage",  # payload: {"stage": Stage, "detail"?: str}
    "site",  # payload: CallSite
    "case_summary",  # payload: {"site_id", "dev": int, "holdout": int, "baselines": int}
    "tier_result",  # payload: TierResult (case_scores omitted)
    "iteration",  # payload: Adapter (spec + change_log), call_site_id set
    "verdict",  # payload: Verdict
    "pr",  # payload: {"url": str, "title": str}
    "log",  # payload: {"level": "info"|"warn"|"error", "message": str}
    "error",  # payload: {"stage": Stage, "reason": str}
    "done",  # payload: {"totals": RunTotals}
]


class RunEvent(BaseModel):
    run_id: str
    seq: int  # monotonically increasing per run; SSE `id:` field
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stage: Stage
    type: EventType
    call_site_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class RunTotals(BaseModel):
    sites: int = 0
    migrated: int = 0
    kept_closed: int = 0
    manual_review: int = 0
    cases: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    ultra_agent_calls: int = 0
    tf_spend_usd: float = 0.0
    est_cost_before_per_1k: float = 0.0
    est_cost_after_per_1k: float = 0.0


class RunConfig(BaseModel):
    repo_url: str
    threshold: float = 0.95
    reasoning_threshold: float = 0.90
    tiers: list[Tier] = Field(default_factory=lambda: list(TIER_ORDER))
    max_iterations: int = 3
    open_pr: bool = True
    live: bool = True  # False -> cache-only (replay); any cache miss is an error
