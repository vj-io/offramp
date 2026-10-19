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
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

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
    Split,
    Tier,
)
from offramp.tf import LLM, Budget, TFClient

DEFAULT_WORKERS = (
    3  # cells in flight (one per tier); each cell runs up to runner.MAX_CONCURRENCY calls itself
)
SECRET_ENV = ("NEBIUS_API_KEY", "NEBIUS_API_KEY_FALLBACK")  # values that must never appear in any output


class Matrix(BaseModel):
    sites: list[CallSite]
    adapters: dict[str, dict[Tier, Adapter]]  # site id -> tier -> the adapter evaluated on that tier
    cases: list[EvalCase]
    references: list[BaselineResult] = Field(default_factory=list)
    tiers: list[Tier] = Field(default_factory=lambda: list(TIER_ORDER))
    splits: list[Split] = Field(
        default_factory=lambda: ["dev", "holdout"]
    )  # a split without cases is skipped
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    live: bool = True  # False: replay the cache only (RunConfig.live); a miss is an error entry
    budget: dict[str, int] = Field(default_factory=dict)  # Budget(**budget): the caps left for this job

    @model_validator(mode="after")
    def _every_cell_has_an_adapter(self) -> Matrix:
        have = {s.id: self.adapters.get(s.id, {}) for s in self.sites}
        missing = [f"{s.id}/{t.value}" for s in self.sites for t in self.tiers if t not in have[s.id]]
        if missing:
            raise ValueError(f"no adapter for: {', '.join(missing)}")
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
    budget: dict[str, Any] = Field(default_factory=dict)  # Budget.snapshot() of the run, {} if it had none


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


def budget_snapshot(client: LLM) -> dict[str, Any]:
    budget = getattr(client, "budget", None)
    return budget.snapshot() if budget is not None else {}


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
