"""PR Writer (PRD §6.9, template §8, Amendment A).

The title, evidence table, every number and the <details> blocks are rendered deterministically from the run's
Verdict / TierResult / reference inputs (`render_pr`): each parity figure with its holdout pass count and
Wilson 95% lower bound, the sites whose source model retires first, each price with its source. Nemotron 3
Super (role `pr_writer`) writes prose only (`write_pr`): a short summary and a note per kept site. Prose that
contains a number is discarded, so a number in the PR can only come from a measurement. A migrated site's line
and a manual-review site's reason are built by code: the measurements, and the patch stage's failure as one
plain sentence with the raw failure folded.
"""

import json
import math
import re
from collections import Counter
from datetime import UTC, date, datetime

import openai
from pydantic import BaseModel, Field

from offramp.evals.errors import ProviderError
from offramp.evals.scorers.aggregate import JUDGE_CASE_FLOOR, JUDGE_FLOOR, JUDGE_MARGIN, WEAK_REFERENCE
from offramp.integrations.tavily import spelled
from offramp.schemas import (
    TIER_ORDER,
    Adapter,
    BaselineResult,
    CallSite,
    EvalCase,
    ReferenceAudit,
    RunConfig,
    Tier,
    TierResult,
    Verdict,
)
from offramp.tf.budget import BudgetExceeded
from offramp.tf.client import LLM, CacheMiss
from offramp.tf.prices import from_yaml
from offramp.tf.tiers import get_tier

FREE_TEXT_TAG = "free_text_arg:"  # offramp.evals.scorers.tool_call
ROLE = "pr_writer"
MAX_TOKENS = 1500
MAX_FAILING_SHOWN = 10
SOURCE_OUTPUTS_NOT_RECORDED = (
    "The source models' own outputs were never recorded; parity is agreement with references."
)
BOUNDS = (
    "In brackets after a parity figure: holdout cases passing / evaluated (for JSON extraction, whose figure "
    "is the mean field match, the cases that pass whole), and the Wilson 95% lower bound of that pass rate; "
    "with few holdout cases, even 100% only shows, with 95% confidence, that the rate is above that bound. "
    "More holdout cases narrow it, and evaluating them costs cents on Nemotron 3.5 Lightning."
)
PRICE_METHODS = {
    "search": "Tavily search",
    "extract (confirms table)": "Tavily Extract, confirming pricing.yaml",
    "yaml fallback": "pricing.yaml fallback",
}
QUOTE_CHARS = 200
TIER_NAMES = {
    Tier.LIGHTNING: "Nemotron 3.5 Lightning",
    Tier.SUPER: "Nemotron 3 Super",
    Tier.ULTRA: "Nemotron 3 Ultra",
}


class SiteNote(BaseModel):
    site_id: str
    note: str


class PRProse(BaseModel):
    summary: str = ""
    sites: list[SiteNote] = Field(default_factory=list)


class PRDraft(BaseModel):
    title: str
    body: str


class PRContext(BaseModel):
    """Everything the PR shows. The pipeline fills it from the run's artifacts and the PatchResult."""

    run_id: str
    repo: str  # "owner/name"
    config: RunConfig
    sites: list[CallSite]
    verdicts: list[Verdict]  # after the patch composer (failed patches are MANUAL_REVIEW)
    tier_results: list[TierResult] = Field(default_factory=list)
    adapters: dict[str, Adapter] = Field(default_factory=dict)  # site_id -> winning / best adapter
    case_counts: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )  # site_id -> {"dev": n, "holdout": n}
    references: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )  # site_id -> {BaselineResult.model: n}
    audit: dict[str, ReferenceAudit] = Field(default_factory=dict)  # evals/audit.json (GC18)
    free_text_args: dict[str, list[str]] = Field(default_factory=dict)  # site_id -> ["tool.arg", ...]
    langsmith_links: dict[str, str] = Field(default_factory=dict)  # "project" / site_id -> dataset URL
    langsmith_public: bool = False  # the links are public share links (OFFRAMP_LANGSMITH_PUBLIC=1)
    site_files: dict[str, list[str]] = Field(default_factory=dict)  # PatchResult.site_files
    sandbox: str | None = None  # PatchResult.sandbox_mode: "docker" | "local"
    linted: bool = True  # PatchResult.linted: the checks ran the repo's ruff too
    sandbox_notes: list[str] = Field(default_factory=list)  # PatchResult.sandbox_notes, stated as they are
    async_client: bool = False  # PatchResult.async_client: the client module holds token_factory_async_client
    client_packaged: bool = True  # PatchResult.client_packaged: the client module sits inside a package
    change_notes: list[str] = Field(default_factory=list)  # PatchResult.change_notes, one bullet each
    client_path: str | None = None  # PatchResult.client_path: where the Token Factory client module lives
    alone: dict[str, bool] = Field(default_factory=dict)  # PatchResult.alone: site -> green on its own
    combined: list[str] = Field(default_factory=list)  # PatchResult.combined: sites in the checked tree
    combined_repair: list[str] = Field(default_factory=list)  # PatchResult.combined_repair: files it changed
    refused: dict[str, list[str]] = Field(default_factory=dict)  # site_id -> rewrites refused unevaluated
    today: date = Field(default_factory=lambda: datetime.now(UTC).date())  # "retires" or "retired" on a date


def is_test_path(path: str) -> bool:
    """tests/..., test/..., test_*.py, *_test.py and conftest.py count as test code."""
    parts = path.split("/")
    name = parts[-1]
    return (
        any(p in ("tests", "test") for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def reference_summary(
    cases: list[EvalCase], baselines: list[BaselineResult]
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, list[str]]]:
    """(case_counts, references, free_text_args) for PRContext from the run's cases and references.

    free_text_args: tool-call arguments the scorer checks for presence, not wording (EvalCase tags
    `free_text_arg:<arg>`), named `<tool>.<arg>` after the reference calls that carry them.
    """
    counts: dict[str, dict[str, int]] = {}
    for c in cases:
        counts.setdefault(c.site_id, {"dev": 0, "holdout": 0})[c.split] += 1
    methods: dict[str, Counter] = {}
    for b in baselines:
        methods.setdefault(b.case_id.rsplit(":", 1)[0], Counter())[b.model] += 1
    refs = {b.case_id: b for b in baselines}
    free: dict[str, set[str]] = {}
    for c in cases:
        args = {t.removeprefix(FREE_TEXT_TAG).strip() for t in c.tags if t.startswith(FREE_TEXT_TAG)} - {""}
        calls = refs[c.id].output.tool_calls if c.id in refs else []
        for arg in args:
            named = {f"{tc.name}.{arg}" for tc in calls if arg in tc.arguments}
            free.setdefault(c.site_id, set()).update(named or {arg})
    return counts, {s: dict(m) for s, m in methods.items()}, {s: sorted(a) for s, a in free.items()}


# ---------------------------------------------------------------- formatting


def _usd(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.1 else f"${x:.2g}"


def _ms(ms: int) -> str:
    return f"{ms} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def _pct(p: float | None) -> str:
    return "—" if p is None else f"{p:.1f}%"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def threshold(site: CallSite | None, config: RunConfig) -> float:
    reasoning = site is not None and site.contract is not None and site.contract.task_type == "reasoning"
    return config.reasoning_threshold if reasoning else config.threshold


def _task(site: CallSite | None) -> str:
    return site.contract.task_type if site and site.contract else "unknown"


def _result(ctx: PRContext, v: Verdict) -> TierResult | None:
    """The TierResult behind a verdict: the chosen (else best) tier, holdout first, latest adapter."""
    tier = v.tier or v.best_tier
    rows = [t for t in ctx.tier_results if t.site_id == v.site_id and (tier is None or t.tier == tier)]
    return max(rows, key=lambda t: (t.split == "holdout", t.adapter_version), default=None)


def _iterations(n: int) -> str:
    return f" (after {n} adapter iteration{'s' if n != 1 else ''})" if n else ""


def _parity(v: Verdict, site: CallSite | None, failed: bool = False) -> str:
    """The parity figure, its holdout pass count and Wilson 95% lower bound ("100.0% (9/9, ≥70%)"); free text
    adds the judge's mean and lowest case, each against its own bar when the site `failed`. JSON extraction's
    figure is a field match, so its count says "pass"."""
    text = _pct(v.parity_pct)
    if v.holdout_cases and v.parity_lower_pct is not None:
        what = " pass" if _task(site) == "json_extraction" else ""
        text += f" ({v.holdout_passed}/{v.holdout_cases}{what}, ≥{math.floor(v.parity_lower_pct)}%)"
    bars = [(v.judge_mean, "judge mean", JUDGE_FLOOR), (v.judge_min, "lowest case", JUDGE_CASE_FLOOR)]
    for score, name, bar in bars:
        if score is not None:
            text += f", {name} {score:g}/5" + (f" < {bar:.1f}" if failed and score < bar else "")
    return text


_MISSED = {  # Verdict.failed_gate -> what the kept row says after its figures
    "dev": "; holdout passed, dev did not",
    "schema": ", not every output schema-valid",
    "pass_share": ", too few cases pass whole",
    "reference_margin": f", more than {JUDGE_MARGIN:g} below the references' judge mean",
    "latency": "; too slow for the call's own timeout",
}


def _kept(v: Verdict, site: CallSite | None, bar: str) -> str:
    """A kept site's parity cell, saying only why the verdict records it was kept (failed_gate): free text's
    judge figures are compared with their own bars, never with a share. A verdict recorded without the gate
    states its figures and no comparison; one never measured says so."""
    if v.failed_gate == "budget":  # holdout figures only when the best adapter's holdout ran (and failed)
        if v.parity_pct is None:
            return "not measured: the run's budget stopped it"
        figures = _parity(v, site, failed=_task(site) == "free_text")
        return f"{figures}: holdout failed; the run's budget stopped the rest"
    if v.parity_pct is None:
        return "not measured"
    free_text = _task(site) == "free_text"
    text = _parity(v, site, failed=free_text and v.failed_gate in ("judge_mean", "case_floor"))
    if v.failed_gate in _MISSED:
        return text + _MISSED[v.failed_gate]
    return f"{text} < {bar}" if v.failed_gate == "threshold" else text


def _urgency(v: Verdict) -> tuple:
    """Sites whose source model retires come first, the soonest first."""
    return (v.retires_on is None, v.retires_on or date.max, v.site_id)


def _retires(day: date, today: date) -> str:
    return f"{'retired' if day < today else 'retires'} on {day.day} {day:%b %Y}"


def _retiring(verdicts: list[Verdict], sites: dict[str, CallSite], today: date) -> list[str]:
    """The summary's lead: "Claude Sonnet 4.5 retires on 30 Nov 2026: cs_02, cs_03.", one line per date and
    model, the model named as its vendor's notice spells it; the dates come from the notices."""
    retiring: dict[tuple[date, str], list[str]] = {}
    for v in verdicts:
        if v.retires_on:
            model = sites[v.site_id].source_model if v.site_id in sites else None
            name = spelled(model, v.sunset_risk or "") if model else "The source model"
            retiring.setdefault((v.retires_on, name), []).append(v.site_id)
    return [
        f"**{name} {_retires(day, today)}: {', '.join(sids)}.**"
        for (day, name), sids in sorted(retiring.items())
    ]


def _quoted(text: str) -> str:
    """A page's text as an inline code span, at most QUOTE_CHARS long."""
    text = " ".join(text.replace("`", "'").split())
    return f"`{text[:QUOTE_CHARS]}…`" if len(text) > QUOTE_CHARS else f"`{text}`"


def _row(ctx: PRContext, v: Verdict, site: CallSite | None) -> str:
    before = (site.source_model if site else None) or "—"
    if v.retires_on:
        before += f" ({_retires(v.retires_on, ctx.today).replace(' on ', ' ')})"
    elif v.sunset_risk:
        before += " (sunset risk)"
    file = f"`{site.file}`" if site else "—"
    if v.decision == "MIGRATE":
        tr = _result(ctx, v)
        model = v.model or get_tier(v.tier).model_id
        parity = _parity(v, site) + _iterations(v.iterations)
        p50 = _ms(tr.metrics.p50_ms) if tr is not None and tr.split == "holdout" else "—"
        cost_before = f"{_usd(v.cost_before_per_1k)} (est.)" if v.cost_before_per_1k is not None else "—"
        cost_after = _usd(v.cost_after_per_1k) if v.cost_after_per_1k is not None else "—"
        cells = [TIER_NAMES[Tier(v.tier)] + f" (`{model}`)", parity, p50, f"{cost_before} → {cost_after}"]
    elif v.decision == "KEEP_CLOSED":
        cells = ["**NOT MIGRATED**", _kept(v, site, f"{threshold(site, ctx.config) * 100:g}%"), "—", "—"]
    else:
        cells = ["**MANUAL REVIEW**", _parity(v, site), "—", "—"]
    return "| " + " | ".join(_cell(c) for c in [v.site_id, file, _task(site), before, *cells]) + " |"


def _title(verdicts: list[Verdict]) -> str:
    migrated = [v for v in verdicts if v.decision == "MIGRATE"]
    title = (
        f"Offramp: migrate {len(migrated)}/{len(verdicts)} LLM call sites to Nemotron on Nebius Token Factory"
    )
    priced = [v for v in migrated if v.cost_before_per_1k and v.cost_after_per_1k is not None]
    if priced:
        change = round(
            100 * (1 - sum(v.cost_after_per_1k for v in priced) / sum(v.cost_before_per_1k for v in priced))
        )
        scope = f"cost on those {len(priced)}"  # the migrated sites only; the report headline covers all
        title += f" (est. −{change}% {scope})" if change >= 0 else f" (est. +{-change}% {scope})"
    return title


def _failing(ctx: PRContext, v: Verdict) -> list[str]:
    tr = _result(ctx, v)
    if tr is None or not tr.failing_case_ids:
        return []
    scores = {s.case_id: s for s in tr.case_scores}
    lines = [
        f"<details><summary>Failing cases: {v.site_id} ({TIER_NAMES[tr.tier]}, "
        f"adapter v{tr.adapter_version}, {tr.split} split)</summary>",
        "",
        "| Case | Error | Detail |",
        "|---|---|---|",
    ]
    for cid in tr.failing_case_ids[:MAX_FAILING_SHOWN]:
        s = scores.get(cid)
        lines.append(
            f"| {cid} | {_cell(s.error_category or '—') if s else '—'} | {_cell(s.detail) if s else '—'} |"
        )
    if len(tr.failing_case_ids) > MAX_FAILING_SHOWN:
        lines.append(f"\n… and {len(tr.failing_case_ids) - MAX_FAILING_SHOWN} more.")
    return [*lines, "", "</details>", ""]


def _migrated_line(ctx: PRContext, v: Verdict, site: CallSite | None) -> str:
    """What the call does now and what was measured on the holdout split; nothing else is claimed."""
    tr = _result(ctx, v)
    measured = ""
    if tr is not None and tr.split == "holdout" and tr.case_scores:
        n = f"{len(tr.case_scores)} case{'s' if len(tr.case_scores) != 1 else ''}"
        if _task(site) == "free_text":
            low = min(c.score for c in tr.case_scores)
            measured = f": judge mean {round(tr.score, 2):g}/5 over {n}, lowest case {round(low, 2):g}/5"
        else:
            measured = f": parity vs reference {_pct(v.parity_pct)} over {n}"
    call = f"`{site.function}`" if site and site.function else "The call"
    return f"{call} now sends the request that passed parity on the holdout split{measured}."


# PatchFailure.kind -> what failed, from the patch stage's own record. The reason's text (the target repo's
# check output among it) is never searched for a cause.
_EDITED = "with the model's test edits applied,"
_GREEN_ALONE = "its patch passes this repo's checks on its own, but"
_CAUSE = {
    "search_miss": "the model's edit to {files} quotes lines matching no single place there, so none applied",
    "deletes_tests": "the model's edit deleted {tests}, and Offramp never ships a patch that deletes a test",
    "skips_tests": "the model's edit made {tests} skip, and Offramp never ships a patch that skips a test",
    "locked_file": "the model tried to edit {files}, which ships exactly as Offramp's code wrote it",
    "not_editable": "the model's edit to {files} is not allowed for this site",
    "does_not_parse": "the model's edit left {files} unparseable",
    "checks_failed": f"{_EDITED} this repo's checks still failed",
    "backstop": f"{_EDITED} code that ships changed ({{files}}), and only Offramp's code may write it",
    "probe_mismatch": f"{_EDITED} the call does not send the request that passed the evaluation",
    "merge_conflict": f"{_GREEN_ALONE} it conflicts with another migrated site's patch in {{files}}",
    "not_together": f"{_GREEN_ALONE} not merged with the other migrated sites' patches",
    "merged_drift": f"{_GREEN_ALONE} on the merged tree its code or its evaluated request changed",
    "repo_red": "this repo's own checks fail before any migration patch, so no patch could be checked",
}
_ALONE = {  # the last check ran Offramp's changes alone: no edit of the model's was ever applied
    "checks_failed": "with Offramp's changes alone, this repo's checks fail",
    "backstop": "with Offramp's changes alone, a check run changed code that ships ({files})",
    "probe_mismatch": "Offramp's own rewrite does not send the request that passed the evaluation",
}
_ENDED = {  # how the repairs ended, said before what the last checked try showed
    "budget_stop": "The run's model budget ran out {tries} at this site's patch",
    "provider_error": "Token Factory refused a model request {tries} at this site's patch",
    "internal_check": "Offramp's own check stopped this site's patch {tries}, before a model request",
    "no_edit_block": "The model's last reply had no complete edit, so nothing was applied",
    "no_change": "The model's last repair repeated an edit it had already tried",
}


def manual_reason(v: Verdict) -> tuple[str, str | None]:
    """A manual-review verdict as one sentence a reviewer reads, plus the raw reason to fold under it:
    (sentence, raw). Built from the patch stage's structured record; without one, the reason shows as is."""
    f = v.patch_failure
    if f is None:
        return v.reason, None
    shown = f.last or f.kind
    cause = (_ALONE if not f.model_edit and shown in _ALONE else _CAUSE).get(shown, "")
    tests = f"the test{'s' if len(f.tests) != 1 else ''} " + ", ".join(f"`{t}`" for t in f.tests)
    cause = cause.format(files=", ".join(f"`{p}`" for p in f.files) or "a file", tests=tests)
    if f.kind in _ENDED:
        tries = (
            f"after {f.tries} {'try' if f.tries == 1 else 'tries'}" if f.tries else "before the model replied"
        )
        sentence = _ENDED[f.kind].format(tries=tries) + (f"; {cause}" if cause else "")
    else:
        sentence = cause[:1].upper() + cause[1:]
    return f"{sentence}.", v.reason


def _folded(raw: str) -> list[str]:
    """The raw failure in a collapsed block under its list item, one check output per line."""
    lines = raw.replace(" | ", "\n").splitlines()
    fence = "`" * max(3, 1 + max((len(t) for t in re.findall(r"`+", raw)), default=0))
    block = [f"{fence}text", *lines, fence, "", "</details>"]
    return [
        "",
        "  <details><summary>Raw failure</summary>",
        "",
        *(f"  {ln}" if ln else "" for ln in block),
        "",
    ]


def render_pr(ctx: PRContext, prose: PRProse | None = None) -> PRDraft:
    prose = prose or PRProse()
    notes = {n.site_id: n.note for n in prose.sites}
    sites = {s.id: s for s in ctx.sites}
    verdicts = sorted(ctx.verdicts, key=lambda v: v.site_id)
    migrated = [v for v in verdicts if v.decision == "MIGRATE"]
    kept = [v for v in verdicts if v.decision == "KEEP_CLOSED"]
    manual = [v for v in verdicts if v.decision == "MANUAL_REVIEW"]

    def changelog(sid: str) -> str:
        a = ctx.adapters.get(sid)
        text = f"Adapter changes: {'; '.join(a.change_log)}." if a and a.change_log else ""
        if sid in ctx.refused:  # model-written adapters the sanity gate stopped (adapter.adapter_problem)
            text += f" Rewrites never evaluated nor shipped: {'; '.join(ctx.refused[sid])}."
        return text.strip()

    def files(sid: str) -> str:
        paths = ctx.site_files.get(sid, [])
        code = [f"`{p}`" for p in paths if not is_test_path(p)]
        tests = [f"`{p}`" for p in paths if is_test_path(p)]
        out = [f"Files: {', '.join(code)}."] if code else []
        if tests:
            out.append(
                f"Tests updated (they pinned the old provider's request or fakes): {', '.join(tests)}."
            )
        return " ".join(out)

    out = ["## Summary", ""]
    if lead := _retiring(verdicts, sites, ctx.today):
        out += [*lead, ""]
    if prose.summary:
        out += [prose.summary, ""]
    counts = ctx.case_counts.values()
    cases = (
        f" ({sum(c.get('dev', 0) + c.get('holdout', 0) for c in counts)} cases, "
        f"{sum(c.get('holdout', 0) for c in counts)} in the holdout split)"
        if ctx.case_counts
        else ""
    )
    status = (
        f"{len(migrated)} migrated to the cheapest Nemotron tier that passed parity on the holdout split; "
    )
    status += f"{len(kept)} kept on the original model"
    status += f"; {len(manual)} {'needs' if len(manual) == 1 else 'need'} manual review." if manual else "."
    out += [f"Offramp evaluated {len(verdicts)} LLM call sites against references{cases}. {status}", ""]

    out += [
        "## Evidence",
        "",
        "| Site | File | Task | Before | After | Parity vs reference (holdout) | p50 latency (after) "
        "| $/1k calls before (est.) → after |",
        "|---|---|---|---|---|---|---|---|",
        *(_row(ctx, v, sites.get(v.site_id)) for v in sorted(verdicts, key=_urgency)),
        "",
        "## What changed",
        "",
    ]
    if migrated:
        client = ctx.client_path or "offramp_client.py"
        folder = client.rsplit("/", 2)[-2] if "/" in client else ""
        if client.endswith("/offramp_client.py") and ctx.client_packaged:
            where = f"inside the `{folder}` package, so it ships with it"
        elif client.endswith("/offramp_client.py"):
            where = f"beside the migrated modules in the `{folder}/` folder, which is not a package"
        else:
            where = "at the repository root, because the migrated call sites are not inside one package"
        asynchronous = (
            " For the async call sites, `token_factory_async_client()` (`openai.AsyncOpenAI`): each still "
            "`await`s its call, and a test fakes it the way it faked the old SDK client."
            if ctx.async_client
            else ""
        )
        out.append(
            f"- `{client}` added {where}: OpenAI SDK client for Nebius Token Factory "
            "(`token_factory_client()`, reads `NEBIUS_API_KEY` and optional `NEBIUS_BASE_URL`) and the "
            "Nemotron tier constants `LIGHTNING`, `SUPER`, `ULTRA`. Every migrated call sends "
            "`chat_template_kwargs.enable_thinking` explicitly, since Nemotron reasons by default."
            + asynchronous
        )
    out += [f"- {note[0].upper()}{note[1:]}." for note in ctx.change_notes] if migrated else []
    for v in migrated:
        site = sites.get(v.site_id)
        text = " ".join(
            t for t in [_migrated_line(ctx, v, site), changelog(v.site_id), files(v.site_id)] if t
        )
        out.append(
            f"- **{v.site_id}** `{site.file if site else '?'}` → {TIER_NAMES[Tier(v.tier)]}. {text}".rstrip()
        )
    if migrated:
        out.append(
            "- Offramp's code rewrote each migrated call (the evaluated request as written-out keyword "
            "arguments, the reply read the OpenAI way) and the callers that build its client, and that code "
            "ships as Offramp wrote it; Nemotron 3 Super edited only tests and their fakes (each site's test "
            "files are listed above) where they pinned the old provider, with repair rounds on "
            "Nemotron 3 Ultra (thinking off) when this repo's checks failed."
        )
    green = sorted(sid for sid, ok in ctx.alone.items() if ok)
    if len(green) > 1:
        together = (
            "all pass this repo's checks together" if set(ctx.combined) == set(green)
            else f"{', '.join(ctx.combined) or 'none of them'} pass this repo's checks together"
        )  # fmt: skip
        if ctx.combined_repair:
            files = ", ".join(f"`{p}`" for p in ctx.combined_repair)
            together += f", after one combined repair by Nemotron 3 Ultra to {files}"
        dropped = [sid for sid in green if sid not in ctx.combined]
        if dropped:
            together += f"; {', '.join(dropped)} dropped, see below"
        out.append(
            f"- The patches green on their own ({', '.join(green)}) were then checked merged: {together}."
        )
    checks = "`ruff check` and `pytest`" if ctx.linted else "`pytest`"
    if ctx.sandbox == "docker":
        out.append(
            f"- Each site's patch was checked with this repo's own {checks} in Docker "
            "(network off, 2 CPUs); a patch that failed them was dropped and its site marked for manual "
            "review."
        )
    elif ctx.sandbox:
        out.append(
            f"- Each site's patch was checked with this repo's own {checks} in a local "
            "subprocess (Docker was unavailable, so the run was not network-isolated); a patch that failed "
            "them was dropped and its site marked for manual review."
        )
    out += [f"- {note[0].upper()}{note[1:]}." for note in ctx.sandbox_notes if ctx.sandbox]
    if not migrated:
        out.append("- No call site was migrated.")

    out += ["", "## What Offramp did NOT change", ""]
    for v in kept:
        site = sites.get(v.site_id)
        extra = " ".join(t for t in [notes.get(v.site_id, ""), changelog(v.site_id)] if t)
        before = (site.source_model if site else None) or "its original model"
        out.append(
            f"- **{v.site_id}** `{site.file if site else '?'}` stays on {before}: {v.reason} {extra}".rstrip()
        )
    for v in manual:  # no model prose: the code-built reason, the raw failure folded under it
        site = sites.get(v.site_id)
        sentence, raw = manual_reason(v)
        out.append(f"- **{v.site_id}** `{site.file if site else '?'}` needs manual review: {sentence}")
        out += _folded(raw) if raw else []
    if not kept and not manual:
        out.append("- Nothing: every evaluated call site was migrated.")

    out += ["", "## How parity was measured", "", SOURCE_OUTPUTS_NOT_RECORDED]
    closed: dict[str, list[str]] = {}
    for sid in sorted(ctx.references):
        for model in ctx.references[sid]:
            if not model.startswith("reference:"):
                closed.setdefault(model, []).append(sid)
    for model, sids in closed.items():
        out.append(f"Reference outputs for {', '.join(sids)} were produced by `{model}`.")
    texts = [  # free-text sites whose references hold text: the pairwise rule applies (GC20, GC27)
        s.id
        for s in sorted(ctx.sites, key=lambda s: s.id)
        if _task(s) == "free_text"
        and any(m not in ("reference:seed", "reference:oracle") for m in ctx.references.get(s.id, {}))
    ]
    pairwise = (
        f", and at least the references' judge mean − {JUDGE_MARGIN:g} where references hold text "
        f"({', '.join(texts)})"
        if texts
        else ""
    )
    out += [
        "",
        "Each site's cases are split once into dev (adapter iteration) and holdout (verdict only). Pass bar: "
        f"{ctx.config.threshold * 100:g}% agreement with the references "
        f"({ctx.config.reasoning_threshold * 100:g}% for reasoning); free text: judge mean of at least "
        f"{JUDGE_FLOOR:.1f} on a 1–5 rubric, no holdout case below {JUDGE_CASE_FLOOR:.1f}{pairwise}. "
        + BOUNDS,
        *(
            f"- {sid}: free-text argument{'s' if len(args) > 1 else ''} {', '.join(args)} checked for "
            "presence, not wording."
            for sid, args in sorted(ctx.free_text_args.items())
        ),
        *(  # reference QA: shown, never excluded
            f"- {v.site_id}: reference judged weak: "
            + "; ".join(f"{cid}, {round(x, 2):g}/5" for cid, x in sorted(v.weak_references.items()))
            + f" (below {WEAK_REFERENCE:g}; kept in the evaluation)."
            for v in verdicts
            if v.weak_references
        ),
        "",
        "| Site | References | Cases (dev / holdout) "
        "| Reference audit (checked / agreed / dropped / human-verified, by) |",
        "|---|---|---|---|",
    ]
    for v in verdicts:
        refs = ", ".join(f"{m} × {n}" for m, n in ctx.references.get(v.site_id, {}).items()) or "—"
        c = ctx.case_counts.get(v.site_id)
        split = f"{c.get('dev', 0)} / {c.get('holdout', 0)}" if c else "—"
        a = ctx.audit.get(v.site_id)
        audit = f"{a.checked} / {a.agreed} / {a.dropped} / {a.human_verified}" if a else "—"
        if a and a.auditor:
            audit += f" by {a.auditor}"
        out.append("| " + " | ".join(_cell(x) for x in [v.site_id, refs, split, audit]) + " |")

    out += ["", "Prices (USD per 1M input / output tokens):"]
    for tier in TIER_ORDER:
        if v := next((v for v in migrated if v.tier == tier), None):
            p = v.after_price or from_yaml(tier, "not read from the Token Factory model API in this run")
            price = f"${p.input_usd_per_m:.2f} / ${p.output_usd_per_m:.2f}"
            out.append(f"- After: {TIER_NAMES[tier]} `{p.model}` {price} ({p.source})")
    for v in migrated:
        if v.cost_before_note:
            line = f"- Before, {v.site_id}: {v.cost_before_note}"
            if e := v.price_evidence:
                quote = f"; the page prints {_quoted(e.quote)}" if e.quote else ""
                line += f". Price by {PRICE_METHODS[e.method]}{quote}."
            out.append(line)
    sunset = [v for v in verdicts if v.sunset_risk]
    if sunset:
        out += ["", "Sunset risk (source-model deprecation notices):"]
        out += [f"- {v.site_id}: {v.sunset_risk}" for v in sunset]
    if ctx.langsmith_links:
        keys = sorted(ctx.langsmith_links, key=lambda k: (k != "project", k))
        scope = (
            "public, read-only"
            if ctx.langsmith_public
            else "private: members of the LangSmith workspace only"
        )
        out += ["", f"LangSmith ({scope}): " + " · ".join(f"[{k}]({ctx.langsmith_links[k]})" for k in keys)]
    out.append("")

    for v in verdicts:
        out += _failing(ctx, v)
    out.append(
        "Generated by Offramp · Nemotron 3 Ultra / Super / 3.5 Lightning via Nebius Token Factory · "
        f"run {ctx.run_id}"
    )
    return PRDraft(title=_title(verdicts), body="\n".join(out) + "\n")


# ---------------------------------------------------------------- prose (Nemotron 3 Super)

SYSTEM = """You write the prose of a GitHub pull request opened by Offramp. Offramp measured which LLM call \
sites of a repository NVIDIA Nemotron on Nebius Token Factory can take over, patched the ones that passed, \
and renders every number of the PR itself from the measurements.

Return JSON: {"summary": str, "sites": [{"site_id": str, "note": str}]}.
- summary: two or three plain sentences for the reviewer: what this PR changes and how the evidence was \
produced (agreement with references on a held-out split; the repo's own tests run on the patch).
- one note per site, one sentence, without repeating its decision label: for a migrated site, what changed \
in the call and why the function's behaviour is kept; for a kept site, why it stays and one concrete next \
step; for a manual-review site, what a reviewer must check.
- Never state a number, percentage, price, latency or count, in digits or in words ("one", "five", \
"a dozen", "twice"): such prose is discarded. Identifiers such as cs_02, file names or model ids are fine.
- Do not count sites: write "the refund planner stays on its model", not "one site stays".
- Parity is agreement with references (expected outputs built for the evaluation), never with the original \
model, whose outputs were not recorded: do not claim consistency or agreement with the original or prior \
model.
- Use only the facts given. No marketing language, no emoji."""

_LABEL = re.compile(
    r"^\s*(MIGRATE|KEEP_CLOSED|MANUAL_REVIEW)\s*[-–—:]\s*"
)  # the model echoing the prompt's labels
# Identifiers may hold digits: `code` that starts like a name, tier names, words like cs_02 or gpt-4o-mini.
_IDENTIFIER = re.compile(r"`[A-Za-z_][^`]*`|Nemotron 3(?:\.5)?|\b[A-Za-z_][\w.\-/]*\d[\w.\-/]*")
_NUMBER_WORD = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundreds?|"
    r"thousands?|millions?|billions?|dozens?|half|twice|thrice)\b",
    re.I,
)


def has_number(text: str) -> bool:
    """True if `text` states a number: digits outside identifiers, or a number word ("five", "a dozen")."""
    return bool(re.search(r"\d", _IDENTIFIER.sub("", text)) or _NUMBER_WORD.search(text))


def _facts(ctx: PRContext) -> str:
    sites = {s.id: s for s in ctx.sites}
    rows = []
    for v in sorted(ctx.verdicts, key=lambda v: v.site_id):
        s, a = sites.get(v.site_id), ctx.adapters.get(v.site_id)
        rows.append(
            {
                "site_id": v.site_id,
                "file": s.file if s else None,
                "function": s.function if s else None,
                "task": _task(s),
                "decision": v.decision,
                "before": s.source_model if s else None,
                "after": TIER_NAMES[Tier(v.tier)] if v.decision == "MIGRATE" else None,
                "reason": v.reason,
                "adapter_changes": a.change_log if a else [],
                "files_changed": ctx.site_files.get(v.site_id, []),
            }
        )
    return json.dumps({"repo": ctx.repo, "sites": rows}, indent=1, ensure_ascii=False)


def write_pr(ctx: PRContext, llm: LLM) -> PRDraft:
    """Super writes the prose; anything stating a number, or about an unknown site, is dropped."""
    try:
        res = llm.chat(
            role=ROLE,
            tier=Tier.SUPER,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": _facts(ctx)}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            response_model=PRProse,
            reasoning="off",
        )
    except (openai.APIError, BudgetExceeded, CacheMiss, ProviderError):  # prose is optional; evidence is not
        return render_pr(ctx)
    prose = res.parsed if isinstance(res.parsed, PRProse) else None
    if prose is None:
        return render_pr(ctx)
    known = {v.site_id for v in ctx.verdicts}
    return render_pr(
        ctx,
        PRProse(
            summary="" if has_number(prose.summary) else prose.summary.strip(),
            sites=[
                n.model_copy(update={"note": _LABEL.sub("", n.note).strip()})
                for n in prose.sites
                if n.site_id in known and not has_number(n.note)
            ],
        ),
    )
