"""Per-site routing loop (PRD §6.6, §6.7): the cheapest Nemotron tier that reaches parity, or an honest no.

For each tier in config.tiers, cheapest first and once each (TIER_ORDER), starting from the Adapter Agent's
v0: dev; on a failure the Iterator writes the next adapter from failing dev cases only (at most
  config.max_iterations per tier) and dev runs again; once dev passes, and only then, holdout. A holdout
  pass is MIGRATE(tier). A holdout failure moves on to the next tier without iterating: holdout never steers
  the Iterator.
All tiers failing: KEEP_CLOSED with the best tier, i.e. the adapter with the highest dev score (the first one
wins a tie), measured on holdout once so the verdict's parity is a holdout number; the verdict counts that
adapter's iterations and changes, not the later ones tried on its tier.

A ProviderError from a tier run (an outage, an account error, the judge leaving a case out) is retried once;
a second one propagates as a run error, never a score. CacheMiss (replay without its cassette) propagates.
Verdicts without a comparison: a per-run cap (BudgetStop from the runner, BudgetExceeded from the Iterator)
is KEEP_CLOSED("budget: ..."); a site the loop cannot evaluate (no contract, flagged MANUAL_REVIEW, no dev or
no holdout cases, a case without a reference, a call with unresolved parameters) is MANUAL_REVIEW.

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
from offramp.evals.errors import MissingReference, ProviderError
from offramp.evals.runner import BudgetStop, run_tier
from offramp.schemas import (
    TIER_ORDER,
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
    # MIGRATE: the adapter that passed (what the patch implements); KEEP_CLOSED: the best one tried; None when
    # nothing was compared (budget stop, MANUAL_REVIEW)
    adapter: Adapter | None
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
        args = (site, cases, references, current, tier, client, split, config)
        try:
            out, calls = runner(*args, return_candidates=True)
        except ProviderError as e:  # once more: an outage may have passed (a cached failure replays for free)
            emit(
                "log",
                {
                    "level": "warn",
                    "message": f"{site.id}: {tier.value} {split}: {e}"[:300] + "; retrying once",
                },
            )
            out, calls = runner(*args, return_candidates=True)  # a second ProviderError is a run error
        if isinstance(out, BudgetStop):
            raise _BudgetStop(out.reason)
        results.append(out)
        emit("tier_result", out.model_dump(mode="json", exclude={"case_scores"}))
        return out, calls

    history: dict[Tier, list[Adapter]] = {}  # per tier: v0, then each iteration's adapter in order
    holdouts: dict[
        tuple[Tier, int], tuple[TierResult, list[CandidateResult]]
    ] = {}  # by (tier, history index)

    def verdict(tier: Tier, i: int, dev: TierResult, holdout: TierResult, calls) -> Verdict:
        """The verdict for history[tier][i]: i iterations, and the change logs of exactly those."""
        changes = [entry for a in history[tier][1 : i + 1] for entry in a.change_log]
        return decide(
            site,
            Evidence(tier, dev, holdout, calls, i, changes),
            client,
            config,
            closed_parity=closed_reference_parity(site, cases, references, closed, config),
            price_lookup=price_lookup,
        )

    # the adapter with the highest dev score so far: (score, tier, its history index, its dev result)
    best: tuple[float, Tier, int, TierResult] | None = None
    try:
        for tier in sorted(set(map(Tier, config.tiers)), key=TIER_ORDER.index):
            history[tier] = [adapter]
            while True:
                i = len(history[tier]) - 1
                current = history[tier][i]
                dev, calls = run(current, tier, "dev")
                if best is None or dev.score > best[0]:
                    best = (dev.score, tier, i, dev)
                if dev.passed or i >= config.max_iterations:
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
                history[tier].append(new)
            if dev.passed:
                holdout, holdout_calls = holdouts[tier, i] = run(current, tier, "holdout")
                if holdout.passed:
                    return SiteRoute(verdict(tier, i, dev, holdout, holdout_calls), current, results)
        _, tier, i, dev = best
        holdout, holdout_calls = holdouts.get((tier, i)) or run(history[tier][i], tier, "holdout")
        return SiteRoute(verdict(tier, i, dev, holdout, holdout_calls), history[tier][i], results)
    except (_BudgetStop, BudgetExceeded) as e:
        reason = str(e) if isinstance(e, _BudgetStop) else f"budget: {e}"
        return SiteRoute(stopped(site, "KEEP_CLOSED", reason, "budget"), None, results)
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
