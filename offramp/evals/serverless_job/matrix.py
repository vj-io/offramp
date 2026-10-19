"""The eval matrix (PRD §6.5): one input, one output, in a Nebius Serverless Job or locally.

`Matrix` is `sites x tiers x splits` over the given cases. Each cell is one `run_tier` call and its outcome
is one entry of the output list, so a cell that cannot be measured never costs the others:

  {"kind": "tier_result", "site_id", "tier", "split", "result": <TierResult as JSON>}
  {"kind": "budget_stop", "site_id", "tier", "split", "reason": "budget: ..."}      # a BudgetStop
  {"kind": "error", "site_id", "tier", "split", "error_type": "ProviderError" | "MissingReference" |
       "UnresolvedCallSite" | "CacheMiss" | ..., "message",
       ["status" (ProviderError)], ["case_ids" (MissingReference)]}

Entries are sorted by (site, tier order, split), so the same matrix over the same cache gives byte-identical
JSON whichever path ran it. Which cell trips a per-run cap, if one does, depends on call timing (workers=1
pins the cell order, not the calls inside a cell); everything else is deterministic.

Budget (Global Constraint 8): `Matrix.budget` is required and holds the caps LEFT for this run
(`remaining_caps(budget)`); an empty one is an error, never a fresh full cap. `MatrixResults.budget` is always
the usage of this run alone (a delta); `absorb_usage(budget, results)` adds it to the caller's Budget.

Tracing (Global Constraint 23): a job does not trace to LangSmith (no key, no tracing env, no extra in the
image), so nothing may present a job-run cell as traced.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from offramp.evals.errors import MissingReference, ProviderError
from offramp.evals.runner import BudgetStop, BuildRequest, run_tier
from offramp.evals.scorers.aggregate import DEFAULT_THRESHOLDS, Thresholds
from offramp.schemas import (
    TIER_ORDER,
    Adapter,
    BaselineResult,
    CallSite,
    EvalCase,
    ModelOutput,
    Split,
    Tier,
)
from offramp.tf import LLM, Budget, BudgetExceeded, TFClient
from offramp.tf.budget import EVAL_TARGET_ROLE
from offramp.tf.cache import cache_key
from offramp.tf.client import _Call

DEFAULT_WORKERS = 3  # cells in flight (one per tier); each runs up to runner.MAX_CONCURRENCY calls itself
BUDGET_CAPS = ("max_in", "max_out", "max_ultra_agent_calls")  # the Budget constructor's arguments
USAGE = ("tokens_in", "tokens_out", "ultra_agent_calls", "tf_spend_usd")  # what MatrixResults.budget holds
SECRET_ENV = ("NEBIUS_API_KEY", "NEBIUS_API_KEY_FALLBACK")  # values that must never appear in any output


class Matrix(BaseModel):
    sites: list[CallSite]
    adapters: dict[str, dict[Tier, Adapter]]  # site id -> tier -> the adapter evaluated on that tier
    cases: list[EvalCase]
    references: list[BaselineResult] = Field(default_factory=list)
    tiers: list[Tier] = Field(default_factory=lambda: list(TIER_ORDER))
    splits: list[Split] = Field(default_factory=lambda: ["dev", "holdout"])  # no cases in a split: skipped
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    live: bool = True  # False: replay the cache only (RunConfig.live); a miss is an error entry
    budget: dict[str, int]  # the caps left for this run: Budget(**budget). Required; see remaining_caps()

    @model_validator(mode="after")
    def _every_cell_has_an_adapter(self) -> Matrix:
        have = {s.id: self.adapters.get(s.id, {}) for s in self.sites}
        missing = [f"{s.id}/{t.value}" for s in self.sites for t in self.tiers if t not in have[s.id]]
        if missing:
            raise ValueError(f"no adapter for: {', '.join(missing)}")
        return self

    @model_validator(mode="after")
    def _budget_is_explicit(self) -> Matrix:
        if set(self.budget) != set(BUDGET_CAPS) or any(v < 0 for v in self.budget.values()):
            raise ValueError(
                f"budget must give {', '.join(BUDGET_CAPS)} as non-negative ints, the caps left for this "
                f"run (remaining_caps(budget)); an empty budget is not a fresh full cap; got {self.budget}"
            )
        return self

    # stdlib json both ways: a float survives the trip exactly, so both paths score the same numbers
    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Matrix:
        return cls.model_validate(json.loads(text))


class MatrixResults(BaseModel):
    results: list[dict[str, Any]]  # the sorted entries above
    cache: dict[str, dict[str, Any]] = Field(default_factory=dict)  # key -> entry the job wrote (job path)
    budget: dict[str, Any] = Field(default_factory=dict)  # USAGE of this run alone, {} if the client had none
    ran_on: Literal["job", "local"] = "local"  # where the cells ran: report it, never let a fallback pass
    job_id: str | None = None  # the Nebius job, when one was created
    fallback_reason: str | None = None  # why mode="auto" ran locally instead of in a job


def dumps(results: Sequence[dict[str, Any]], indent: int | None = None) -> str:
    """The canonical JSON of a results list: what is compared, and what results.json holds."""
    separators = None if indent else (",", ":")
    return json.dumps(list(results), sort_keys=True, indent=indent, separators=separators, ensure_ascii=False)


def scrub(text: str) -> str:
    """`text` without the values of the provider keys (Global Constraint 15)."""
    for name in SECRET_ENV:
        if value := os.environ.get(name):
            text = text.replace(value, "***")
    return text


def make_client(matrix: Matrix, cache_dir: str | Path | None = None) -> TFClient:
    return TFClient(cache_dir=cache_dir, live=matrix.live, budget=Budget(**matrix.budget))


def budget_usage(client: LLM) -> dict[str, Any]:
    """What the client's Budget has spent so far ({} for a client without one)."""
    budget = getattr(client, "budget", None)
    if budget is None:
        return {}
    snapshot = budget.snapshot()
    return {name: snapshot[name] for name in USAGE}


def remaining_caps(budget: Budget) -> dict[str, int]:
    """`Matrix.budget` for a run that has already spent part of `budget`: the caps that are left."""
    s = budget.snapshot()
    return {
        "max_in": max(0, s["max_in"] - s["tokens_in"]),
        "max_out": max(0, s["max_out"] - s["tokens_out"]),
        "max_ultra_agent_calls": max(0, s["max_ultra_agent_calls"] - s["ultra_agent_calls"]),
    }


def absorb_usage(budget: Budget, results: MatrixResults) -> None:
    """Charge `results.budget` (the run's own usage) to the caller's Budget. All of it is recorded; if that
    pushes a cap over, BudgetExceeded is raised afterwards, as Budget.charge does."""
    usage = results.budget
    if not usage:
        return
    over: BudgetExceeded | None = None
    charges = [(EVAL_TARGET_ROLE, Tier.SUPER, usage["tokens_in"], usage["tokens_out"], usage["tf_spend_usd"])]
    charges += [("judge", Tier.ULTRA, 0, 0, 0.0)] * usage[
        "ultra_agent_calls"
    ]  # Ultra agent calls count by role
    for role, tier, tokens_in, tokens_out, cost in charges:
        try:
            budget.charge(role, tier, tokens_in, tokens_out, cost)
        except BudgetExceeded as e:
            over = over or e
    if over:
        raise over


def _entry(site: CallSite, tier: Tier, split: Split, kind: str, **fields: Any) -> dict[str, Any]:
    return {"kind": kind, "site_id": site.id, "tier": tier.value, "split": split, **fields}


def _error(site: CallSite, tier: Tier, split: Split, e: Exception) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if isinstance(e, ProviderError):
        extra["status"] = e.status
    if isinstance(e, MissingReference):
        extra["case_ids"] = e.case_ids
    message = scrub(str(e))
    return _entry(site, tier, split, "error", error_type=type(e).__name__, message=message, **extra)


def run_matrix(
    matrix: Matrix, client: LLM, *, workers: int = DEFAULT_WORKERS, build_request: BuildRequest | None = None
) -> list[dict[str, Any]]:
    cases = {s.id: [c for c in matrix.cases if c.site_id == s.id] for s in matrix.sites}
    refs = {r.case_id: r for r in matrix.references}
    cells = [
        (site, tier, split)
        for site in matrix.sites
        for tier in matrix.tiers
        for split in matrix.splits
        if any(c.split == split for c in cases[site.id])
    ]

    def run(cell: tuple[CallSite, Tier, Split]) -> dict[str, Any]:
        site, tier, split = cell
        site_refs = {c.id: refs[c.id] for c in cases[site.id] if c.id in refs}
        try:
            out = run_tier(
                site,
                cases[site.id],
                site_refs,
                matrix.adapters[site.id][tier],
                tier,
                client,
                split,
                matrix.thresholds,
                build_request=build_request,
            )
        except Exception as e:  # a cell that cannot be measured is an entry, not a dead job
            return _error(site, tier, split, e)
        if isinstance(out, BudgetStop):
            return _entry(site, tier, split, "budget_stop", reason=out.reason)
        return _entry(site, tier, split, "tier_result", result=out.model_dump(mode="json"))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        entries = list(pool.map(run, cells))
    entries.sort(key=lambda e: (e["site_id"], TIER_ORDER.index(Tier(e["tier"])), e["split"]))
    return entries


class _Probe(TFClient):
    """Replays the cache and notes which entries the matrix reads. A miss answers empty, so every cell
    is walked and no call is made."""

    def __init__(self, cache_dir: str | Path | None):
        super().__init__(cache_dir=cache_dir, live=False)
        self.keys: set[str] = set()

    def _complete(self, role, info, messages, params):  # type: ignore[no-untyped-def]
        key = cache_key(info.model_id, messages, params)
        if self.cache.get(key) is None:
            return _Call(
                output=ModelOutput(text=""), input_tokens=0, output_tokens=0, latency_ms=0, cached=True
            )
        self.keys.add(key)
        return super()._complete(role, info, messages, params)


def needed_cache_keys(
    matrix: Matrix, cache_dir: str | Path | None, workers: int = DEFAULT_WORKERS
) -> set[str]:
    """The cache entries `matrix` would replay: what a job needs of the cache, instead of all of it. Exact: a
    later call (a judge, a repair retry) is keyed on earlier replies, and those are replayed for real."""
    probe = _Probe(cache_dir)
    run_matrix(matrix, probe, workers=workers)
    return probe.keys
