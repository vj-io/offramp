"""Per-site routing loop (PRD §6.6, §6.7): the cheapest Nemotron tier that reaches parity, or an honest no.

For each tier in config.tiers (cheapest first), starting from the Adapter Agent's v0:
  dev; on a failure the Iterator writes the next adapter from failing dev cases only (at most
  config.max_iterations per tier) and dev runs again; once dev passes, and only then, holdout. A holdout
  pass is MIGRATE(tier). A holdout failure moves on to the next tier without iterating: holdout never steers
  the Iterator.
All tiers failing: KEEP_CLOSED with the best tier, i.e. the adapter with the highest dev score (the first one
wins a tie), measured on holdout once so the verdict's parity is a holdout number.

Run errors propagate and are never verdicts: ProviderError (outage, account), CacheMiss (replay without its
cassette). Verdicts without a comparison: a per-run cap (BudgetStop from the runner, BudgetExceeded from the
Iterator) is KEEP_CLOSED("budget: ..."); a site the loop cannot evaluate (no contract, flagged MANUAL_REVIEW,
no dev or no holdout cases, a case without a reference, a call with unresolved parameters) is MANUAL_REVIEW.

on_event(type, payload) gets JSON-ready RunEvent payloads: "tier_result" (TierResult without case_scores),
"iteration" (Adapter), "log" ({"level", "message"}) and "verdict" (Verdict).
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from offramp.agents.iterator import DevCase, Iterator, failing_dev_cases
from offramp.agents.translate import UnresolvedCallSite
from offramp.agents.verdict import Evidence, PriceLookup, closed_reference_parity, decide, stopped
from offramp.evals.errors import MissingReference
from offramp.evals.runner import BudgetStop, run_tier
from offramp.schemas import (
    Adapter,
    BaselineResult,
    CallSite,
    CandidateResult,
    EvalCase,
    RunConfig,
    Split,
    Tier,
    TierResult,
    Verdict,
)
from offramp.tf.budget import BudgetExceeded
from offramp.tf.client import LLM

IterateFn = Callable[[Adapter, CallSite, list[DevCase], Tier, Mapping[str, int]], Adapter | None]
EventFn = Callable[[str, dict[str, Any]], None]


@dataclass
class SiteRoute:
    verdict: Verdict
    adapter: (
        Adapter | None
    )  # MIGRATE: the adapter that passed (what the patch implements); KEEP_CLOSED: the best
    # one tried; None when nothing was compared (budget stop, MANUAL_REVIEW)
    results: list[TierResult] = field(default_factory=list)  # every run, in order, case_scores kept


class _BudgetStop(Exception):
    pass


def route_site(
    site: CallSite,
    cases: Sequence[EvalCase],
    references: Mapping[str, BaselineResult],
    adapter: Adapter,
    client: LLM,
    config: RunConfig,
    *,
    closed: Mapping[str, BaselineResult] | None = None,
    schemas: dict[str, dict[str, Any]] | None = None,
    iterator: IterateFn | None = None,
    runner: Callable[..., Any] = run_tier,
    price_lookup: PriceLookup | None = None,
    on_event: EventFn | None = None,
) -> SiteRoute:
    """Route one site from Adapter v0 (`adapter`) to its verdict. `references`: what candidates are compared
    against (references, or closed outputs where a site has none, as baselines.baselines_for returns them);
    `closed`: closed-model outputs for the verdict's comparison row (GC27c). `iterator` defaults to
    Iterator(client, schemas), `runner` to run_tier."""
    if not config.tiers:
        raise ValueError("config.tiers is empty")
    emit = on_event or (lambda event_type, payload: None)
    route = _route(
        site,
        cases,
        references,
        adapter,
        client,
        config,
        closed,
        iterator or Iterator(client, schemas),
        runner,
        price_lookup,
        emit,
    )
    emit("verdict", route.verdict.model_dump(mode="json"))
    return route


def _route(site, cases, references, adapter, client, config, closed, iterate, runner, price_lookup, emit):
    results: list[TierResult] = []
    unfit = _unfit(site, cases)
    if unfit:
        return SiteRoute(stopped(site, "MANUAL_REVIEW", unfit), None, results)

    def run(current: Adapter, tier: Tier, split: Split) -> tuple[TierResult, list[CandidateResult]]:
        out, calls = runner(
            site, cases, references, current, tier, client, split, config, return_candidates=True
        )
        if isinstance(out, BudgetStop):
            raise _BudgetStop(out.reason)
        results.append(out)
        emit("tier_result", out.model_dump(mode="json", exclude={"case_scores"}))
        return out, calls

    def verdict(tier, dev, holdout, calls, iterations, changes) -> Verdict:
        return decide(
            site,
            Evidence(tier, dev, holdout, calls, iterations, changes),
            client,
            config,
            closed_parity=closed_reference_parity(site, cases, references, closed, config),
            price_lookup=price_lookup,
        )

    # the adapter with the highest dev score so far: (score, tier, adapter, its dev result)
    best: tuple[float, Tier, Adapter, TierResult] | None = None
    holdouts: dict[tuple[Tier, int], tuple[TierResult, list[CandidateResult]]] = {}
    tried: dict[Tier, list[str]] = {}  # change logs of the iterations run per tier
    iterations: dict[Tier, int] = {}
    try:
        for tier in map(Tier, config.tiers):
            current, iterations[tier], tried[tier] = adapter, 0, []
            while True:
                dev, calls = run(current, tier, "dev")
                if best is None or dev.score > best[0]:
                    best = (dev.score, tier, current, dev)
                if dev.passed or iterations[tier] >= config.max_iterations:
                    break
                failed = [s for s in dev.case_scores if not s.passed]
                counts = Counter(s.error_category or "failed" for s in failed)
                new = iterate(current, site, failing_dev_cases(dev, cases, references, calls), tier, counts)
                if new is None:
                    emit(
                        "log",
                        {
                            "level": "warn",
                            "message": f"{site.id}: the Iterator returned no valid adapter "
                            f"on {tier.value} (reply invalid after one repair); moving on",
                        },
                    )
                    break
                emit("iteration", new.model_dump(mode="json"))
                current, iterations[tier] = new, iterations[tier] + 1
                tried[tier] += new.change_log
            if dev.passed:
                holdout, holdout_calls = holdouts[tier, current.version] = run(current, tier, "holdout")
                if holdout.passed:
                    v = verdict(tier, dev, holdout, holdout_calls, iterations[tier], tried[tier])
                    return SiteRoute(v, current, results)
        _, tier, best_adapter, dev = best
        holdout, holdout_calls = holdouts.get((tier, best_adapter.version)) or run(
            best_adapter, tier, "holdout"
        )
        v = verdict(tier, dev, holdout, holdout_calls, iterations[tier], tried[tier])
        return SiteRoute(v, best_adapter, results)
    except (_BudgetStop, BudgetExceeded) as e:
        reason = str(e) if isinstance(e, _BudgetStop) else f"budget: {e}"
        return SiteRoute(stopped(site, "KEEP_CLOSED", reason), None, results)
    except (MissingReference, UnresolvedCallSite) as e:
        return SiteRoute(stopped(site, "MANUAL_REVIEW", f"cannot evaluate: {e}"), None, results)


def _unfit(site: CallSite, cases: Sequence[EvalCase]) -> str | None:
    """Why the loop cannot evaluate this site at all, if it cannot."""
    if site.contract is None:
        return "no contract: the site was never labelled"
    if "MANUAL_REVIEW" in site.flags:
        return "scanner flagged MANUAL_REVIEW (streaming call or unlabelled site)"
    splits = {c.split for c in cases}
    return next((f"no {s} cases" for s in ("dev", "holdout") if s not in splits), None)
