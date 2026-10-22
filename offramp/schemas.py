"""Shared contracts between Offramp components (PRD §6, §7).

Every agent input/output and every SSE payload is one of these models. Components
import from here; nobody redefines these shapes locally.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


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
    # Read from the code, never asked of the labeller (so not in its schema): the function around the call
    # runs json.loads on the reply as it is ("exact") or after stripping a ``` fence ("fence_stripped"),
    # with strict=False when the value ends in "_lenient".
    reply_json: SkipJsonSchema[
        Literal["exact", "fence_stripped", "exact_lenient", "fence_stripped_lenient"] | None
    ] = None
    # Read from the code too: the timeout (seconds) the call or its client sets. A tier whose holdout median
    # latency exceeds it fails for the site (verdict.too_slow): the migrated call keeps that timeout.
    timeout_s: SkipJsonSchema[float | None] = None

    def shown(self) -> dict[str, Any]:
        """The contract as agent prompts show it: reply_json only when set, never timeout_s, so a site's
        prompt is the one it always got unless it has reply_json (its recorded cassettes replay)."""
        return self.model_dump(
            mode="json", exclude={"timeout_s"} | (set() if self.reply_json else {"reply_json"})
        )


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
    warnings: list[str] = Field(default_factory=list)  # files skipped (syntax error, recursion limit), etc.


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
    reference_score: float | None = None  # free_text judged pairwise: the judge's 1..5 score of the reference


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
    # how many cases the run scored (failing_case_ids failed) and the Wilson 95% lower bound (0..100) of the
    # share that passed: what the grid shows next to each score. None in results recorded before Task 19
    cases: int | None = None
    parity_lower_pct: float | None = None


PatchFailureKind = Literal[
    "budget_stop",  # the run's model budget stopped the site's repairs
    "provider_error",  # Token Factory refused a model request (too large, another 4xx): this site only
    "internal_check",  # Offramp's own check stopped the repairs before a request (nothing was sent)
    "no_edit_block",  # the model's last reply held no complete edit block, so nothing was applied
    "no_change",  # a repair repeated an edit already tried
    "search_miss",  # patch rules: a SEARCH that does not match exactly one place
    "deletes_tests",  # patch rules: the patch deletes tests
    "skips_tests",  # patch rules: the patch makes a test skip or xfail
    "locked_file",  # patch rules: an edit to code that ships
    "not_editable",  # patch rules: a file this site's patch may not change (or not whole)
    "does_not_parse",  # patch rules: an edited file that does not parse
    "checks_failed",  # the repo's ruff + pytest failed on the patched tree
    "backstop",  # code that ships changed on disk
    "probe_mismatch",  # the patched call does not send the evaluated request
    "merge_conflict",  # green alone; its patch conflicts with another site's
    "not_together",  # green alone; the merged tree fails the repo's checks
    "merged_drift",  # green alone; on the merged tree its code or its request changed
    "repo_red",  # the repo's checks fail before any patch
]


class PatchFailure(BaseModel):
    """Why the patch stage sent a MIGRATE site to manual review, recorded where it happened: the PR's plain
    sentence is built from this, never from the reason's text (which holds the target repo's check output)."""

    kind: PatchFailureKind  # the outermost, decisive failure
    last: PatchFailureKind | None = None  # budget_stop / provider_error / no_edit_block / no_change: the last
    # checked try's
    tries: int = 0  # the model's replies for this site
    model_edit: bool = False  # the last checked try was the model's edits (else Offramp's changes alone)
    files: list[str] = Field(default_factory=list)  # the files that failure names (Offramp's own record)
    tests: list[str] = Field(default_factory=list)  # deletes_tests: "path::name"


class PriceEvidence(BaseModel):
    """Where the source model's price behind cost_before came from: the report's Tavily evidence panel."""

    # Tavily Search; Tavily Extract printing pricing.yaml's own price (it never replaces it); pricing.yaml
    method: Literal["search", "extract (confirms table)", "yaml fallback"]
    query: str | None = None  # the Tavily search query; None when Tavily was not asked
    url: str  # the page the price was read from (pricing.yaml's own source on a fallback)
    quote: str | None = None  # that page's text printing the price, checked on the page; None on fallback
    date: str  # YYYY-MM-DD: the day Tavily returned the page, or pricing.yaml's as_of
    note: str | None = None  # on a fallback, why ("Tavily page <url> shows $X/$Y; checked-in table $A/$B")


class AfterPrice(BaseModel):
    """A Nemotron tier's price on Token Factory (USD per 1M tokens) and where it was read."""

    model: str
    input_usd_per_m: float
    output_usd_per_m: float
    via: Literal["token_factory_api", "pricing.yaml"]
    source: str  # "Token Factory model API, 2026-10-21" or "pricing.yaml as of 2026-10-21; <why>"


# Why a KEEP_CLOSED verdict keeps the site: "budget" when the run's budget stopped it before a verdict (its
# holdout figures are set only if the best adapter's holdout was measured, and failed);
# "dev" when dev never passed (holdout may have); else the bar its holdout missed: the parity threshold, every
# output schema-valid, enough JSON cases passing whole, the judge mean of 4.0, no holdout case below 3.0, or a
# judge mean within 0.3 of the references'; or "latency": its holdout median exceeds the call's own timeout.
Gate = Literal[
    "budget",
    "dev",
    "threshold",
    "schema",
    "pass_share",
    "judge_mean",
    "case_floor",
    "reference_margin",
    "latency",
]


class Verdict(BaseModel):
    site_id: str
    decision: Decision
    tier: Tier | None = None  # set when MIGRATE
    model: str | None = None  # Token Factory model id when MIGRATE
    parity_pct: float | None = None  # holdout parity (0..100) of the chosen/best tier
    best_tier: Tier | None = None  # KEEP_CLOSED: best tier tried
    best_score: float | None = None  # that tier's score on best_score_split (a judge mean for free text)
    best_score_split: Split | None = (
        None  # "holdout" when measured there; "dev" when a budget stop came first
    )
    iterations: int = 0  # adapter iterations used for the chosen/best tier
    latency_delta_ms: int | None = None  # p50 after - before; None when "before" was never measured
    cost_before_per_1k: float | None = None
    cost_before_note: str | None = None  # e.g. "est.: gpt-4o-mini $0.15/$0.60 per 1M (openai.com/api/pricing,
    # fetched via Tavily 2026-10-19) x tokens measured on the Token Factory call"
    cost_after_per_1k: float | None = None
    sunset_risk: str | None = None  # deprecation notice for the source model found via Tavily, with URL
    # GC27(c): e.g. "claude-sonnet-5.5 (via Claude Code; not the repo's source model)"
    closed_reference_model: str | None = None
    # that model's holdout parity on the same references. Model named, parity None: that model blind-audited
    # these references (only those it agreed with were kept), so a comparison would be circular; none is made
    closed_reference_parity_pct: float | None = None
    # free text: holdout case -> the judge's score of its reference, for references scored below 3 (reference
    # QA: shown in the PR and the report, never excluded from the evaluation)
    weak_references: dict[str, float] = Field(default_factory=dict)
    patch_failure: PatchFailure | None = None  # MANUAL_REVIEW from the patch stage: why, structured
    # the holdout cases behind parity_pct: how many, how many passed, and the Wilson 95% lower bound (0..100)
    # of that pass rate, which says how sure the figure is ("100.0% (3/3, ≥43%)")
    holdout_cases: int | None = None
    holdout_passed: int | None = None
    parity_lower_pct: float | None = None
    # free text: the judge's mean and lowest holdout case score (1..5), rounded down when the run failed
    judge_mean: float | None = None
    judge_min: float | None = None
    retires_on: date | None = None  # the source model's retirement date, from the notice in sunset_risk
    price_evidence: PriceEvidence | None = None  # where cost_before's source-model price came from
    after_price: AfterPrice | None = None  # MIGRATE: the tier's Token Factory price and where it was read
    failed_gate: Gate | None = None  # KEEP_CLOSED: why (None when passing, or recorded before Task 19)
    reason: str


# ---------------------------------------------------------------- run + SSE events (§5.2, §9)


EventType = Literal[
    "stage",  # payload: {"stage": Stage, "detail"?: str}
    "site",  # payload: CallSite
    "case_summary",  # payload: CaseSummary
    "tier_result",  # payload: TierResult (case_scores omitted)
    "iteration",  # payload: Adapter (spec + change_log), call_site_id set
    "verdict",  # payload: Verdict
    "pr",  # payload: {"url": str, "title": str}
    "log",  # payload: {"level": "info"|"warn"|"error", "message": str}
    "error",  # payload: {"stage": Stage, "reason": str}
    "done",  # payload: {"totals": RunTotals}
]


class ReferenceAudit(BaseModel):
    """GC18: blind check of seed references by an independent model; human audit when available."""

    checked: int = 0
    agreed: int = 0
    dropped: int = 0
    human_verified: int = 0
    auditor: str | None = None  # e.g. "claude-sonnet-5.5 (via Claude Code)"


class CaseSummary(BaseModel):
    site_id: str
    dev: int
    holdout: int
    references: int
    reference_methods: dict[str, int] = Field(default_factory=dict)  # {"reference:seed": 20, ...}
    audit: ReferenceAudit = Field(default_factory=ReferenceAudit)


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
