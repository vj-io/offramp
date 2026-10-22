"""Per-site routing loop (PRD §6.6, §6.7): the cheapest Nemotron tier that reaches parity, or an honest no.

For each tier in config.tiers, cheapest first and once each (TIER_ORDER), starting from the Adapter Agent's
v0: dev; on a failure the Iterator writes the next adapter from failing dev cases only (at most
  config.max_iterations per tier) and dev runs again; once dev passes, and only then, holdout. A holdout
  pass is MIGRATE(tier). A holdout failure moves on to the next tier without iterating: holdout never steers
  the Iterator.
All tiers failing: KEEP_CLOSED with the best tier, i.e. the adapter with the highest dev score (the first one
wins a tie), measured on holdout once so the verdict's parity is a holdout number; the verdict counts that
adapter's iterations and changes, not the later ones tried on its tier.

`prior` holds cells measured before routing (the eval matrix: v0 on every tier's dev cases). Their scores are
reused, never judged again. Verdict first under the Ultra cap: before the first Iterator call on a tier
cheaper than one whose v0 passed dev there, that tier's holdout runs (the cheapest such tier whose holdout
passes is a secured verdict); then the cheaper tiers are iterated with what is left. The pipeline runs the
same step for every site before routing any (secure_holdouts), so one site's iterations cannot spend what
another site's verdict needed. The result is the same cheapest passing tier as without the reordering; a
budget stop after a secured verdict keeps it, and its reason says which cheaper tiers were not fully tried.

A ProviderError from a tier run (an outage, an account error, the judge leaving a case out) is retried once;
a second one propagates as a run error, never a score. CacheMiss (replay without its cassette) propagates.
Verdicts without a comparison: a per-run cap (BudgetStop from the runner, BudgetExceeded from the Iterator)
is KEEP_CLOSED("budget: ...") with the best adapter measured so far (tier, dev score, iterations); a site the
loop cannot evaluate (no contract, flagged MANUAL_REVIEW, no dev or no holdout cases, a case without a
reference, a call with unresolved parameters) is MANUAL_REVIEW.

on_event(type, payload) gets JSON-ready RunEvent payloads: "tier_result" (TierResult without case_scores),
"iteration" (Adapter), "log" ({"level", "message"}) and "verdict" (Verdict).
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from offramp.agents.adapter import adapter_problem
from offramp.agents.iterator import DevCase, Iterator, failing_dev_cases
from offramp.agents.translate import UnresolvedCallSite
from offramp.agents.verdict import (
    Evidence,
    PriceLookup,
    budget_stopped,
    closed_reference_parity,
    decide,
    stopped,
)
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
    refused: list[str] = field(default_factory=list)  # model-written rewrites refused before evaluation


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
    prior: Mapping[tuple[Tier, int, Split], TierResult] | None = None,
) -> SiteRoute:
    """Route one site from Adapter v0 (`adapter`) to its verdict. `references`: what candidates are compared
    against (references, or closed outputs where a site has none, as baselines.baselines_for returns them);
    `closed`: closed-model outputs for the verdict's comparison row (GC27c). `iterator` defaults to
    Iterator(client, schemas), `runner` to run_tier. `prior`: this site's cells measured before routing, by
    (tier, adapter version, split): the eval matrix's."""
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
        prior or {},
        refused := [],
    )
    route.refused = refused
    emit("verdict", route.verdict.model_dump(mode="json"))
    return route


def _route(
    site,
    cases,
    references,
    adapter,
    client,
    config,
    closed,
    iterate,
    runner,
    price_lookup,
    emit,
    prior,
    refused,
):
    results: list[TierResult] = []
    unfit = _unfit(site, cases)
    if unfit:
        return SiteRoute(stopped(site, "MANUAL_REVIEW", unfit), None, results)

    def run(current: Adapter, tier: Tier, split: Split) -> tuple[TierResult, list[CandidateResult]]:
        args = (site, cases, references, current, tier, client, split, config)
        scored = prior.get((tier, current.version, split))
        kw = {"scored": scored} if scored is not None else {}  # measured already: reused, never re-judged
        try:
            out, calls = runner(*args, return_candidates=True, **kw)
        except ProviderError as e:  # once more: an outage may have passed (a cached failure replays for free)
            emit(
                "log",
                {
                    "level": "warn",
                    "message": f"{site.id}: {tier.value} {split}: {e}"[:300] + "; retrying once",
                },
            )
            out, calls = runner(*args, return_candidates=True, **kw)  # a second ProviderError is a run error
        if isinstance(out, BudgetStop):
            raise _BudgetStop(out.reason)
        results.append(out)
        emit("tier_result", out.model_dump(mode="json", exclude={"case_scores"}))
        return out, calls

    tiers = sorted(set(map(Tier, config.tiers)), key=TIER_ORDER.index)
    history: dict[Tier, list[Adapter]] = {}  # per tier: v0, then each iteration's adapter in order
    holdouts: dict[
        tuple[Tier, int], tuple[TierResult, list[CandidateResult]]
    ] = {}  # by (tier, history index)
    secured: tuple[Tier, TierResult] | None = None  # a v0 that passed dev (in `prior`) and then holdout
    finished: set[Tier] = set()  # tiers whose tries ended (dev passed and holdout ran, or no try left)

    def verdict(tier: Tier, i: int, dev: TierResult, untried: str | None = None) -> Verdict:
        """The verdict for history[tier][i]: i iterations, and the change logs of exactly those."""
        holdout, calls = holdouts[tier, i]
        changes = [entry for a in history[tier][1 : i + 1] for entry in a.change_log]
        return decide(
            site,
            Evidence(tier, dev, holdout, calls, i, changes),
            client,
            config,
            closed_parity=closed_reference_parity(site, cases, references, closed, config),
            price_lookup=price_lookup,
            untried=untried,
        )

    def secure(cheaper_than: Tier) -> None:
        """Verdict first: before the first Iterator call on `cheaper_than`, the holdout of each more expensive
        tier whose v0 passed dev in `prior`, cheapest first, until one passes. One holdout run is then all the
        verdict needs, so the Ultra cap can no longer take it away while cheaper tiers iterate."""
        nonlocal secured
        if secured is not None:
            return
        for tier in tiers[tiers.index(cheaper_than) + 1 :]:
            dev = prior.get((tier, adapter.version, "dev"))
            if dev is None or not dev.passed:
                continue
            history.setdefault(tier, [adapter])
            if (tier, 0) not in holdouts:
                holdouts[tier, 0] = run(adapter, tier, "holdout")
            if holdouts[tier, 0][0].passed:
                secured = (tier, dev)
                return
            finished.add(tier)  # its holdout failed: routing moves on from it without iterating

    # the adapter with the highest dev score so far: (score, tier, its history index, its dev result)
    best: tuple[float, Tier, int, TierResult] | None = None
    try:
        for tier in tiers:
            history[tier] = [adapter]
            tries = 0  # Iterator calls on this tier: an evaluated rewrite or a refused one each
            while True:
                i = len(history[tier]) - 1
                current = history[tier][i]
                dev, calls = run(current, tier, "dev")
                if best is None or dev.score > best[0]:
                    best = (dev.score, tier, i, dev)
                if dev.passed or tries >= config.max_iterations:
                    break
                secure(tier)
                failed = [s for s in dev.case_scores if not s.passed]
                counts = Counter(s.error_category or "failed" for s in failed)
                failing = failing_dev_cases(dev, cases, references, calls)
                why = None  # a rewrite that fails adapter_problem costs its try, never an evaluation
                while tries < config.max_iterations:
                    tries += 1
                    new = iterate(current, site, failing, tier, counts, **({"refused": why} if why else {}))
                    if new is None or not (why := adapter_problem(site, new.spec)):
                        break
                    refused.append(
                        f"{tier.value.capitalize()} v{new.version} refused before evaluation: {why}"
                    )
                    emit("log", {"level": "warn", "message": f"{site.id}: {refused[-1]}"})
                if new is not None and why:
                    break  # every try left was refused
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
                if (tier, i) not in holdouts:
                    holdouts[tier, i] = run(current, tier, "holdout")
                if holdouts[tier, i][0].passed:
                    return SiteRoute(verdict(tier, i, dev), current, results)
            finished.add(tier)
        _, tier, i, dev = best
        if (tier, i) not in holdouts:
            holdouts[tier, i] = run(history[tier][i], tier, "holdout")
        return SiteRoute(verdict(tier, i, dev), history[tier][i], results)
    except (_BudgetStop, BudgetExceeded) as e:
        reason = str(e) if isinstance(e, _BudgetStop) else f"budget: {e}"
        if secured is not None:  # the verdict stands; name the cheaper tiers the stop cut short, if any
            tier, dev = secured
            cut = [t.value.capitalize() for t in tiers[: tiers.index(tier)] if t not in finished]
            untried = f"{' and '.join(cut)} not fully tried ({reason})" if cut else None
            return SiteRoute(verdict(tier, 0, dev, untried), adapter, results)
        # the best adapter measured so far, the matrix's cells of tiers not reached yet included, with its
        # holdout if one was measured (by routing or by the securing pass before it)
        reached = [(b[1], b[3], b[2]) for b in [best] if b]
        reached += [(t, r, 0) for (t, v, split), r in prior.items() if (v, split) == (adapter.version, "dev")]
        top = max(reached, key=lambda b: b[1].score, default=None)  # the first one wins a tie
        holdout = None
        if top is not None:
            tier, _, i = top
            measured = holdouts.get((tier, i))
            holdout = measured[0] if measured else prior.get((tier, i, "holdout")) if i == 0 else None
        return SiteRoute(budget_stopped(site, reason, top, config, holdout), None, results)
    except (MissingReference, UnresolvedCallSite) as e:
        return SiteRoute(stopped(site, "MANUAL_REVIEW", f"cannot evaluate: {e}"), None, results)


def secure_holdouts(
    site: CallSite,
    cases: Sequence[EvalCase],
    references: Mapping[str, BaselineResult],
    adapter: Adapter,
    client: LLM,
    config: RunConfig,
    prior: Mapping[tuple[Tier, int, Split], TierResult],
    *,
    runner: Callable[..., Any] = run_tier,
    on_event: EventFn | None = None,
) -> dict[tuple[Tier, int, Split], TierResult]:
    """Verdict first across sites: routing goes site by site, so one site's iterations could spend the Ultra
    cap that another site's verdict needed. The pipeline calls this for every site before routing any: the
    holdout of each tier whose v0 passed dev in `prior` (the matrix's cells), cheapest first until one
    passes, wherever routing would iterate a cheaper tier first. Returns the holdout cells measured, which
    routing then reuses through `prior`. A per-run cap ends it quietly (routing stops on the cap as well);
    a run error propagates as in routing; a site routing cannot evaluate is left to routing to report."""
    tiers = sorted(set(map(Tier, config.tiers)), key=TIER_ORDER.index)
    passing = [t for t in tiers if (r := prior.get((t, adapter.version, "dev"))) is not None and r.passed]
    if not passing or passing[0] == tiers[0] or config.max_iterations == 0 or _unfit(site, cases):
        return {}
    emit = on_event or (lambda event_type, payload: None)
    out: dict[tuple[Tier, int, Split], TierResult] = {}
    for tier in passing:
        args = (site, cases, references, adapter, tier, client, "holdout", config)
        try:
            try:
                result, _ = runner(*args, return_candidates=True)
            except ProviderError:  # once more, as routing does
                result, _ = runner(*args, return_candidates=True)
        except (MissingReference, UnresolvedCallSite):
            return out
        if isinstance(result, BudgetStop):
            return out
        emit("tier_result", result.model_dump(mode="json", exclude={"case_scores"}))
        out[tier, adapter.version, "holdout"] = result
        if result.passed:
            break
    return out


def _unfit(site: CallSite, cases: Sequence[EvalCase]) -> str | None:
    """Why the loop cannot evaluate this site at all, if it cannot."""
    if site.contract is None:
        return "no contract: the site was never labelled"
    if "MANUAL_REVIEW" in site.flags:
        return "scanner flagged MANUAL_REVIEW (streaming call or unlabelled site)"
    splits = {c.split for c in cases}
    return next((f"no {s} cases" for s in ("dev", "holdout") if s not in splits), None)
