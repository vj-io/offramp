"""The run state machine (PRD §5.2, Global Constraint 12): every agent over one repo, as one run.

    QUEUED -> CLONED -> SCANNED -> BASELINED -> EVALUATING -> ITERATING -> VERDICTS -> PATCHED -> PR_OPENED
    -> DONE, or FAILED from any stage (a `stage` event, then an `error` event).

Stage names are past tense: CLONED is emitted once the clone exists, SCANNED once the sites are labelled.
EVALUATING and ITERATING are emitted when that work starts, so the results it produces stream under them.

A run lives in OFFRAMP_DATA_DIR/runs/<run_id>/:
    events.jsonl            every RunEvent in seq order; the QUEUED event carries the RunConfig. The run's
                            source of truth: GET /api/runs/{id} is a fold of it (`snapshot`)
    repo/                   the clone (a local path is read in place and never written to)
    clone.json scan.json cases.json adapters.json matrix.json verdicts.json patch.json pr.json
                            one artifact per step, written when the step has finished
    evals/results/<site>.json  per site: verdict (the final decision, after the patch stage), eval_verdict
                            (the eval stage's), the adapter it ended on, every TierResult (case scores kept)
    report.json             {snapshot, pr_body_markdown, free_text_arg_exemptions}, written at the end
A step whose artifact exists is loaded instead of run, so `execute()` on an interrupted or failed run picks up
after its last finished step; routing resumes per site, and a PR is never opened twice.

Errors: ProviderError (after the route's one retry), CacheMiss (cache-only run without its cassette) and
anything unexpected end the run FAILED with the reason; nothing is ever turned into a score. A per-run cap
(BudgetExceeded) becomes KEEP_CLOSED("budget: ...") for the sites it stops.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine
from sqlmodel import Session, delete

from offramp import db, github
from offramp.agents import case_gen
from offramp.agents.adapter import REFUSED, adapt
from offramp.agents.patcher import compose_patches
from offramp.agents.pr_writer import PRContext, reference_summary, write_pr
from offramp.agents.scanner import label_contracts, scan_repo
from offramp.agents.verdict import pricing_yaml_price
from offramp.evals import baselines
from offramp.evals.route import route_site, secure_holdouts
from offramp.evals.runner import run_tier
from offramp.evals.scorers.aggregate import Thresholds
from offramp.evals.serverless_job.matrix import Matrix, MatrixResults, absorb_usage, remaining_caps
from offramp.integrations import langsmith
from offramp.integrations.tavily import lookup_source_model
from offramp.schemas import (
    TIER_ORDER,
    Adapter,
    AfterPrice,
    CallSite,
    CaseSummary,
    ReferenceAudit,
    RunConfig,
    RunEvent,
    RunTotals,
    ScanResult,
    Stage,
    Tier,
    TierResult,
    Verdict,
)
from offramp.tf import LLM, Budget, BudgetExceeded, CacheMiss, TFClient
from offramp.tf import prices as tf_prices

TERMINAL = ("done", "error")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
GIT_TIMEOUT_S = 300
GITHUB_REPO = re.compile(r"github\.com[:/]([\w.-]+/[\w.-]+?)(?:\.git)?/?$")
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.I)  # *_API_KEY(_FALLBACK), GITHUB_TOKEN, ...
URL_USERINFO = re.compile(r"(?<=://)[^/@\s]+@")
BUDGET_FILE = "budget.json"  # the run's Token Factory usage so far: caps hold across resumes
# the log line for a call site the run does not evaluate; the dashboard reads it back (web/lib/offramp-tables)
NOT_EVALUATED = "{id}: {kind} call in {file}, not evaluated (chat calls only)"


def data_dir() -> Path:
    return Path(os.environ.get("OFFRAMP_DATA_DIR", "./var"))


def new_run_id(prefix: str = "") -> str:
    """e.g. 20261020-101503123-a1b2c3: milliseconds included, so ids sort in creation order."""
    now = datetime.now(UTC)
    return f"{prefix}{now:%Y%m%d-%H%M%S}{now.microsecond // 1000:03d}-{secrets.token_hex(3)}"


def scrub(value: Any) -> Any:
    """`value` with the value of every secret-named environment variable (KEY, TOKEN, SECRET or PASSWORD in
    the name: every provider key and its fallback, GITHUB_TOKEN) and any user:password@ in a URL replaced by
    ***; dicts and lists are scrubbed throughout. Every event goes through this before it is stored."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if not isinstance(value, str):
        return value
    for name, secret in os.environ.items():
        if len(secret) >= 8 and SECRET_NAME.search(name):
            value = value.replace(secret, "***")
    return URL_USERINFO.sub("***@", value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)  # an artifact is either absent or complete: it marks its step finished


def write_results(directory: Path, routes: dict[str, dict[str, Any]], verdicts: list[dict[str, Any]]) -> None:
    """evals/results/<site>.json under `directory`, with the decision the run ended on: `verdict` is the final
    one (`verdicts`: after the patch stage, which moves a MIGRATE whose patch fails to MANUAL_REVIEW),
    `eval_verdict` the eval stage's. `routes`: the eval stage's results, or files this wrote (idempotent)."""
    final = {v["site_id"]: v for v in verdicts}
    for site_id, result in routes.items():
        evaluated = result.get("eval_verdict", result["verdict"])
        out = {"site": result["site"], "verdict": final.get(site_id, evaluated), "eval_verdict": evaluated}
        out |= {k: v for k, v in result.items() if k not in out}
        _write_json(directory / "evals" / "results" / f"{site_id}.json", out)


# ---------------------------------------------------------------- event log


def read_events(path: Path) -> list[RunEvent]:
    """RunEvents from an events.jsonl (one per line); a JSON array or {"events": [...]} is read too."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    try:
        body = json.loads(text)
    except json.JSONDecodeError:  # JSON Lines (more than one line)
        body = None
    if isinstance(body, list) or (isinstance(body, dict) and "events" in body):
        rows = body["events"] if isinstance(body, dict) else body
        return [RunEvent.model_validate(r) for r in rows]
    return [RunEvent.model_validate_json(line) for line in text.splitlines() if line.strip()]


class RunLog:
    """One run's events: appended to events.jsonl, kept in memory for the API, mirrored to SQLite."""

    def __init__(
        self,
        directory: Path,
        run_id: str,
        engine: Engine | None = None,
        on_event: Callable[[RunEvent], None] | None = None,
    ):
        self.run_id, self.dir = run_id, Path(directory)
        self.file = self.dir / "events.jsonl"
        self.events: list[RunEvent] = read_events(self.file)
        self.engine, self.on_event = engine, on_event
        self._lock = threading.Lock()

    @property
    def stage(self) -> Stage:
        for e in reversed(self.events):
            if e.type == "error":
                return Stage.FAILED
            if e.type == "stage":
                return Stage(e.payload["stage"])
        return Stage.QUEUED

    @property
    def config(self) -> RunConfig | None:
        return next(
            (RunConfig.model_validate(e.payload["config"]) for e in self.events if "config" in e.payload),
            None,
        )

    def emit(
        self, type: str, payload: dict[str, Any] | None = None, call_site_id: str | None = None
    ) -> RunEvent:
        payload = payload or {}
        with self._lock:
            if type == "stage":
                stage = Stage(payload["stage"])
            else:
                stage = Stage.FAILED if type == "error" else self.stage
            seq = self.events[-1].seq + 1 if self.events else 0
            event = RunEvent(
                run_id=self.run_id,
                seq=seq,
                stage=stage,
                type=type,
                call_site_id=call_site_id,
                payload=payload,
            )
            return self._add(event)

    def add(self, event: RunEvent) -> RunEvent:
        """Append a recorded event as is (seq, ts, payload), under this run's id: replays."""
        with self._lock:
            return self._add(event.model_copy(update={"run_id": self.run_id}))

    def _add(self, event: RunEvent) -> RunEvent:
        event = event.model_copy(update={"payload": scrub(event.payload)})  # file, memory and SQLite alike
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.file.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
        self.events.append(event)
        if self.engine is not None:
            record(self.engine, event)
        if self.on_event is not None:
            self.on_event(event)
        return event


_DB_LOCK = threading.Lock()  # ponytail: one writer at a time (all SQLite allows); a queue if it ever shows


def record(engine: Engine, e: RunEvent) -> None:
    """Mirror one event into the PRD §7 summary tables (offramp/db.py). events.jsonl stays the full record."""
    p = e.payload
    with _DB_LOCK, Session(engine) as s:
        run = s.get(db.Run, e.run_id) or db.Run(
            id=e.run_id, repo_url=(p.get("config") or {}).get("repo_url", "")
        )
        if e.type == "stage":
            run.stage = p["stage"]
            run.commit_sha = p.get("commit_sha") or run.commit_sha
        elif e.type == "error":
            run.stage = Stage.FAILED.value
        elif e.type == "done":
            run.totals_json = json.dumps(p["totals"])
        elif e.type == "site":
            c = CallSite.model_validate(p)
            s.exec(delete(db.CallSite).where(db.CallSite.run_id == e.run_id, db.CallSite.site_id == c.id))
            contract = c.contract.model_dump_json() if c.contract else None
            s.add(
                db.CallSite(
                    run_id=e.run_id,
                    site_id=c.id,
                    file=c.file,
                    line_start=c.lines[0],
                    line_end=c.lines[-1],
                    sdk=c.sdk,
                    source_model=c.source_model,
                    contract_json=contract,
                    prompt_ref=c.prompt_ref,
                )
            )
        elif e.type == "tier_result":
            r, t = TierResult.model_validate(p), db.TierResult
            key = (t.site_id == r.site_id, t.tier == r.tier.value, t.adapter_version == r.adapter_version)
            s.exec(delete(t).where(t.run_id == e.run_id, *key, t.split == r.split))
            row = r.model_dump(
                mode="json", include={"site_id", "tier", "adapter_version", "split", "passed", "score"}
            )
            ids = json.dumps(r.failing_case_ids)
            s.add(t(run_id=e.run_id, **row, **r.metrics.model_dump(), failing_case_ids_json=ids))
        elif e.type == "iteration":
            a = Adapter.model_validate(p)
            spec, notes = a.spec.model_dump_json(), json.dumps(a.notes)
            tier = a.tier.value if a.tier else None
            s.add(db.Adapter(run_id=e.run_id, site_id=a.site_id, version=a.version, tier=tier, spec_json=spec,
                             patch=a.patch, notes_json=notes))  # fmt: skip
        elif e.type == "verdict":
            v = Verdict.model_validate(p)
            s.exec(delete(db.Verdict).where(db.Verdict.run_id == e.run_id, db.Verdict.site_id == v.site_id))
            fields = set(db.Verdict.model_fields) - {"id", "run_id"}
            s.add(db.Verdict(run_id=e.run_id, **v.model_dump(mode="json", include=fields)))
        elif e.type == "pr":
            s.add(db.Artifact(run_id=e.run_id, kind="pr_url", path=p["url"]))
        s.add(run)
        s.commit()


def forget(engine: Engine, run_id: str) -> None:
    """Delete a run's summary rows (an evicted replay run)."""
    with _DB_LOCK, Session(engine) as s:
        s.exec(delete(db.Run).where(db.Run.id == run_id))
        for table in (
            db.CallSite,
            db.EvalCase,
            db.Baseline,
            db.Adapter,
            db.TierResult,
            db.Verdict,
            db.Artifact,
        ):
            s.exec(delete(table).where(table.run_id == run_id))
        s.commit()


# ---------------------------------------------------------------- snapshot (GET /api/runs/{id})


class RunSnapshot(BaseModel):
    """Global Constraint 13: the run as the dashboard shows it, folded from its events."""

    run_id: str
    config: RunConfig | None = None
    stage: Stage = Stage.QUEUED
    sites: list[CallSite] = Field(default_factory=list)
    tier_results: list[TierResult] = Field(default_factory=list)  # latest per (site, tier, version, split)
    verdicts: list[Verdict] = Field(default_factory=list)
    case_summaries: list[CaseSummary] = Field(default_factory=list)
    totals: RunTotals = Field(default_factory=RunTotals)
    artifacts: dict[str, str] = Field(default_factory=dict)  # pr_url, report_url


def totals(
    sites: Iterable[CallSite],
    summaries: Iterable[CaseSummary],
    verdicts: Iterable[Verdict],
    budget: Budget | None = None,
) -> RunTotals:
    """Run totals. Cost per 1k calls: before = the estimates summed over the sites that have one; after = the
    same sites, each MIGRATE site on its tier's measured cost and every other one on its source model."""
    verdicts = list(verdicts)
    priced = [v for v in verdicts if v.cost_before_per_1k is not None]
    priced = [v for v in priced if v.decision != "MIGRATE" or v.cost_after_per_1k is not None]
    count = Counter(v.decision for v in verdicts)
    spend = budget.snapshot() if budget else {}
    return RunTotals(
        sites=len(list(sites)),
        migrated=count["MIGRATE"],
        kept_closed=count["KEEP_CLOSED"],
        manual_review=count["MANUAL_REVIEW"],
        cases=sum(c.dev + c.holdout for c in summaries),
        tokens_in=spend.get("tokens_in", 0),
        tokens_out=spend.get("tokens_out", 0),
        ultra_agent_calls=spend.get("ultra_agent_calls", 0),
        tf_spend_usd=spend.get("tf_spend_usd", 0.0),
        est_cost_before_per_1k=sum(v.cost_before_per_1k for v in priced),
        est_cost_after_per_1k=sum(
            v.cost_after_per_1k if v.decision == "MIGRATE" else v.cost_before_per_1k for v in priced
        ),
    )


def snapshot(run_id: str, events: Iterable[RunEvent]) -> RunSnapshot:
    snap = RunSnapshot(run_id=run_id, artifacts={"report_url": f"/api/runs/{run_id}/report"})
    sites: dict[str, CallSite] = {}
    cases: dict[str, CaseSummary] = {}
    verdicts: dict[str, Verdict] = {}
    results: dict[tuple, TierResult] = {}
    done: RunTotals | None = None
    for e in events:
        p = e.payload
        if e.type == "stage":
            snap.stage = Stage(p["stage"])
            if "config" in p:
                snap.config = RunConfig.model_validate(p["config"])
        elif e.type == "error":
            snap.stage = Stage.FAILED
        elif e.type == "site":
            sites[p["id"]] = CallSite.model_validate(p)
        elif e.type == "case_summary":
            cases[p["site_id"]] = CaseSummary.model_validate(p)
        elif e.type == "tier_result":
            r = TierResult.model_validate(p)
            key = (r.site_id, r.tier, r.adapter_version, r.split)
            results.pop(key, None)  # latest emission last
            results[key] = r
        elif e.type == "verdict":
            verdicts[p["site_id"]] = Verdict.model_validate(p)
        elif e.type == "pr":
            snap.artifacts["pr_url"] = p["url"]
        elif e.type == "done":
            done = RunTotals.model_validate(p["totals"])
    snap.sites, snap.case_summaries = list(sites.values()), list(cases.values())
    snap.tier_results, snap.verdicts = list(results.values()), list(verdicts.values())
    snap.totals = done or totals(snap.sites, snap.case_summaries, snap.verdicts)
    return snap


def report(directory: Path, run_id: str, events: Iterable[RunEvent]) -> dict[str, Any]:
    """GET /api/runs/{id}/report and report.json: snapshot, PR body, free-text tool-argument exemptions;
    scrubbed like the events."""
    pr, cases = directory / "pr.json", directory / "cases.json"
    body = json.loads(pr.read_text())["body"] if pr.exists() else None
    free = json.loads(cases.read_text())["free_text_args"] if cases.exists() else {}
    snap = snapshot(run_id, events).model_dump(mode="json")
    return scrub({"snapshot": snap, "pr_body_markdown": body, "free_text_arg_exemptions": free})


# ---------------------------------------------------------------- the run


def new_run(config: RunConfig, root: Path | None = None, run_id: str | None = None, **log_kw: Any) -> RunLog:
    """Create a run's directory and its QUEUED event (which carries the config)."""
    run_id = run_id or new_run_id()
    log = RunLog((root or data_dir()) / "runs" / run_id, run_id, **log_kw)
    log.emit(
        "stage",
        {"stage": Stage.QUEUED.value, "detail": "Run accepted", "config": config.model_dump(mode="json")},
    )
    return log


def _git(path: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, timeout=GIT_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def github_repo(repo_url: str, path: Path) -> str | None:
    """ "owner/name" of the GitHub repo: from the URL, else from a local checkout's origin remote."""
    for url in (repo_url, _git(path, "remote", "get-url", "origin") or ""):
        if m := GITHUB_REPO.search(url.strip()):
            return m[1]
    return None


class Pipeline:
    """One run, executed (or resumed) by `execute()`. Every agent call goes through `client`; the stages that
    other tasks own are plain callables so they can be swapped out (`eval_matrix`, `route`, `patch`, `pr`)."""

    def __init__(
        self,
        log: RunLog,
        client: LLM,
        *,
        cache_dir: str | Path | None = None,
        price_lookup: Callable[[str], Any] | None = None,
        after_prices: Callable[[list[Tier]], dict[Tier, AfterPrice]] | None = None,
        eval_matrix: Callable[[Matrix], MatrixResults] | None = None,
        route: Callable[..., Any] | None = None,
        patch: Callable[..., Any] | None = None,
        pr: Callable[..., dict[str, Any]] | None = None,
        live: bool | None = None,
    ):
        """`live` overrides the stored config's (a live run resumed cache-only)."""
        config = log.config
        if config is None:
            raise ValueError(f"run {log.run_id} has no QUEUED event with its config")
        if live is not None:
            config = config.model_copy(update={"live": live})
        self.log, self.config, self.client, self.dir = log, config, client, log.dir
        self.budget: Budget = getattr(client, "budget", None) or Budget()
        self._restore_budget()
        self.cache_dir = cache_dir
        self.price_lookup = price_lookup or default_price_lookup(config, client)
        # the migrated tiers' Token Factory prices: the model API (cached per day), never called cache-only
        self.after_prices = after_prices or (lambda tiers: tf_prices.after_prices(tiers, live=config.live))
        self.eval_matrix = eval_matrix or self._run_matrix_job
        self.route, self.patch, self.pr = (
            route or route_site,
            patch or compose_patches,
            pr or write_and_open_pr,
        )

    # -------------------------------------------------------- driver

    def execute(self) -> Stage:
        """Run every step not finished yet; DONE or FAILED."""
        if self.log.stage == Stage.DONE:
            return Stage.DONE
        if len(self.log.events) > 1:
            self._log("info", f"resuming run {self.log.run_id} after {self.log.stage.value}")
        try:
            with langsmith.run_project(self.log.run_id):
                self._steps()
        except Exception as e:  # every failure is reported as the run's error, never as a score
            reason = scrub(f"{type(e).__name__}: {e}")[:500]
            at = self.log.stage
            self.log.emit("stage", {"stage": Stage.FAILED.value, "detail": reason})
            self.log.emit("error", {"stage": at.value, "reason": reason})
            self._save_budget()
            _write_json(self.dir / "report.json", report(self.dir, self.log.run_id, self.log.events))
            return Stage.FAILED
        return Stage.DONE

    def _steps(self) -> None:
        clone = self._artifact("clone.json", self._clone)
        repo = Path(clone["path"])
        self._check_dataset(repo / "evals")
        scan = self._artifact("scan.json", lambda: self._scan(repo))
        sites = [CallSite.model_validate(s) for s in scan["sites"]]
        chat = [s for s in sites if s.kind == "chat"]
        evals = repo / "evals"
        cases = self._artifact("cases.json", lambda: self._baseline(evals, chat, set(scan["stale"])))
        unfit = {s.id: why for s in chat if (why := self._unfit(s, scan["stale"], cases["summaries"]))}
        adapters = self._artifact("adapters.json", lambda: self._adapters(chat, unfit, scan["schemas"]))
        stopped = {k: Verdict.model_validate(v) for k, v in adapters["stopped"].items()}
        adapters = {k: Adapter.model_validate(v) for k, v in adapters["adapters"].items()}
        matrix = self._artifact("matrix.json", lambda: self._matrix(chat, adapters, evals))
        routes = self._routes(chat, adapters, stopped, evals, scan["schemas"], matrix)
        verdicts = self._artifact("verdicts.json", lambda: self._verdicts(chat, routes))
        patch = self._artifact("patch.json", lambda: self._patch(repo, chat, verdicts, routes))
        write_results(self.dir, routes, patch["verdicts"])
        self._artifact("pr.json", lambda: self._pr(repo, chat, patch, routes, evals, cases))
        self._finish(chat, cases, patch)

    def _check_dataset(self, evals: Path) -> None:
        """Before anything is spent: an input in both dev and holdout refuses the run (ValueError, the run's
        error); inputs repeated within one split are a warning with their count."""
        baselines.refuse_leaks(evals)
        _, within = baselines.dataset_duplicates(evals)
        if within:
            text = baselines.duplicates_text(within, "repeat within one split")
            self._log("warn", f"{text}: each counts more than once; re-run `offramp prepare` to drop them")

    def _artifact(self, name: str, make: Callable[[], Any]) -> Any:
        """The step's artifact: loaded when an earlier attempt finished the step, else made and saved."""
        path = self.dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        value = make()
        _write_json(path, value)
        self._save_budget()
        return value

    def _save_budget(self) -> None:
        _write_json(self.dir / BUDGET_FILE, self.budget.snapshot())

    def _restore_budget(self) -> None:
        """A resumed run keeps what it spent before: the per-run caps (GC8) count every attempt of the run.
        ponytail: saved after each finished step and site, so a crash (not a failure) forgets the partial
        step's spend; save per call if that ever matters."""
        path = self.dir / BUDGET_FILE
        if not path.exists():
            return
        saved, b = json.loads(path.read_text(encoding="utf-8")), self.budget
        b.tokens_in = max(b.tokens_in, saved["tokens_in"])
        b.tokens_out = max(b.tokens_out, saved["tokens_out"])
        b.ultra_agent_calls = max(b.ultra_agent_calls, saved["ultra_agent_calls"])
        b.spend_usd = max(b.spend_usd, saved["tf_spend_usd"])

    def _stage(self, stage: Stage, detail: str | None = None, **extra: Any) -> None:
        if self.log.stage != stage:
            self.log.emit("stage", {"stage": stage.value, **({"detail": detail} if detail else {}), **extra})

    def _log(self, level: str, message: str, site_id: str | None = None) -> None:
        self.log.emit("log", {"level": level, "message": message}, call_site_id=site_id)

    # -------------------------------------------------------- steps

    def _clone(self) -> dict[str, Any]:
        url = self.config.repo_url
        local = Path(url).expanduser()
        if not re.match(r"^[a-z][a-z0-9+.-]*://", url) and local.is_dir():
            path, how = local.resolve(), "Using the local checkout"
        else:
            path, how = self.dir / "repo", "Cloned"
            if not path.exists():
                env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
                done = subprocess.run(
                    ["git", "clone", "--depth", "1", "--quiet", "--", url, str(path)],
                    capture_output=True,
                    text=True,
                    timeout=GIT_TIMEOUT_S,
                    env=env,
                )
                if done.returncode != 0:
                    raise RuntimeError(f"git clone {url} failed: {scrub(done.stderr.strip())[-300:]}")
        sha = _git(path, "rev-parse", "HEAD")
        at = f" at {sha[:7]}" if sha else ""
        self._stage(Stage.CLONED, f"{how} {url}{at}", commit_sha=sha)
        return {"path": str(path), "commit_sha": sha}

    def _scan(self, repo: Path) -> dict[str, Any]:
        """Scan, label (Super), resolve Pydantic schemas, then swap in the prepared sites (evals/sites.json):
        those are what the cases were generated for (resolved templates, contracts)."""
        scan = scan_repo(repo)
        labelled = label_contracts(scan.sites, self.client, root=repo)
        schemas = {s.id: case_gen.resolved_schemas(s, repo) for s in labelled if s.kind == "chat"}
        prepared = {s.id: s for s in baselines.load_sites(repo / "evals")}
        sites, stale, notes = [], [], []
        for s in labelled:
            p = prepared.get(s.id)
            if s.kind != "chat":
                sites.append(s)
            elif p is None:  # a contract from this run's labelling, not the one the other sites were prepared
                sites.append(s.model_copy(update={"contract": None}) if prepared else s)  # with: unknown
            elif (p.file, p.function) != (s.file, s.function):
                stale.append(s.id)
                sites.append(s)
                notes.append(f"{s.id}: evals/ was prepared for {p.file}:{p.function}, the repo now has "
                             f"{s.file}:{s.function} at this id; run `offramp prepare` again")  # fmt: skip
            else:
                if s.contract and p.contract and s.contract.task_type != p.contract.task_type:
                    notes.append(f"{s.id}: labelled {s.contract.task_type} now, prepared as "
                                 f"{p.contract.task_type}; evaluating as prepared")  # fmt: skip
                # the code's current position, how it parses the reply and its timeout now (read from code)
                read = ("reply_json", "timeout_s")
                code = {k: getattr(s.contract, k) if s.contract else None for k in read}
                contract = p.contract.model_copy(update=code) if p.contract else None
                sites.append(p.model_copy(update={"lines": s.lines, "contract": contract}))
        chat = [s for s in sites if s.kind == "chat"]
        self._stage(Stage.SCANNED, f"{len(chat)} call sites found")
        for message in scan.warnings + notes:
            self._log("warn", message)
        for s in sites:
            if s.kind == "chat":
                self.log.emit("site", s.model_dump(mode="json"), call_site_id=s.id)
            else:
                self._log("info", NOT_EVALUATED.format(id=s.id, kind=s.kind, file=s.file), s.id)
        result = ScanResult(repo=str(repo), sites=sites, warnings=scan.warnings)
        return {**result.model_dump(mode="json"), "schemas": schemas, "stale": stale}

    def _baseline(self, evals: Path, chat: list[CallSite], stale: set[str]) -> dict[str, Any]:
        """Cases and references from the target's evals/ (written by `offramp prepare`), summed per site."""
        if not evals.is_dir():
            self._log("warn", f"no evals/ in the repo: run `offramp prepare {self.config.repo_url}` first")
        else:
            for problem in baselines.verify_manifest(evals):
                self._log("warn", f"evals/manifest.json: {problem}")
        audit_file = evals / "audit.json"
        audit = json.loads(audit_file.read_text()) if audit_file.exists() else {}
        summaries, all_cases, compared = {}, [], []
        for site in chat:
            cases = [] if site.id in stale else baselines.load_cases(evals, site.id)
            if not cases:
                continue
            ids = {c.id for c in cases}
            # what these cases are compared against: their references, else the closed model's answers (GC27a,
            # free text). Counted by BaselineResult.model, the label the PR's reference table uses too
            # ("reference:*" methods; anything else is a closed model). Rows of dropped cases are not counted.
            against = {k: r for k, r in baselines.baselines_for(evals, site.id).items() if k in ids}
            a = audit.get(site.id, {})
            auditors = [x.removeprefix("blind:") for x in a.get("auditors", [])]
            summaries[site.id] = CaseSummary(
                site_id=site.id,
                dev=sum(c.split == "dev" for c in cases),
                holdout=sum(c.split == "holdout" for c in cases),
                references=len(against),
                reference_methods=dict(Counter(r.model for r in against.values())),
                audit=ReferenceAudit(
                    **{k: a.get(k, 0) for k in ("checked", "agreed", "dropped", "human_verified")},
                    auditor=", ".join(auditors) or None,
                ),
            ).model_dump(mode="json")
            all_cases += cases
            compared += against.values()
            langsmith.upload_dataset(site, cases, list(against.values()), run_id=self.log.run_id)
        # for the PR (Task 10): per-site split counts, what the cases are compared against, and the tool
        # arguments checked for presence only (case tags free_text_arg:<name>), disclosed in the PR and report
        counts, methods, free_args = reference_summary(all_cases, compared)
        n = sum(s["dev"] + s["holdout"] for s in summaries.values())
        self._stage(Stage.BASELINED, f"{n} cases with references for {len(summaries)} call sites")
        for site_id, summary in summaries.items():
            self.log.emit("case_summary", summary, call_site_id=site_id)
        return {
            "summaries": summaries,
            "case_counts": counts,
            "references": methods,
            "free_text_args": free_args,
        }

    @staticmethod
    def _unfit(site: CallSite, stale: list[str], summaries: dict[str, Any]) -> str | None:
        """Why the site cannot be evaluated at all (a MANUAL_REVIEW verdict), if it cannot."""
        if site.id in stale:
            return "evals/ was prepared for a different call at this id: run `offramp prepare` again"
        summary = summaries.get(site.id) or {"dev": 0, "holdout": 0}
        missing = [s for s in ("dev", "holdout") if not summary[s]]
        if missing and (why := case_gen.runtime_only(site)):  # prepare again can't help: it skipped it
            return why
        # no contract and no flag: labelled by this run only, its label dropped (_scan); prepare labels it
        if site.contract is None and (not missing or "MANUAL_REVIEW" in site.flags):
            return "no contract: the site was never labelled"
        if "MANUAL_REVIEW" in site.flags:
            return "scanner flagged MANUAL_REVIEW (streaming call or unlabelled site)"
        return f"no {missing[0]} cases: run `offramp prepare` on the repo" if missing else None

    def _adapters(
        self, chat: list[CallSite], unfit: dict[str, str], schemas: dict[str, Any]
    ) -> dict[str, Any]:
        """Adapter v0 (Ultra) per evaluable site; a site without one gets its verdict here."""
        self._stage(Stage.EVALUATING)
        adapters, stopped = {}, {}
        for site in chat:
            if site.id in unfit:
                stopped[site.id] = Verdict(site_id=site.id, decision="MANUAL_REVIEW", reason=unfit[site.id])
                continue
            try:
                adapters[site.id] = adapt(site, self.client, schemas.get(site.id) or None)
            except BudgetExceeded as e:
                stopped[site.id] = Verdict(
                    site_id=site.id, decision="KEEP_CLOSED", reason=f"budget: {e}", failed_gate="budget"
                )
        return {
            "adapters": {k: a.model_dump(mode="json") for k, a in adapters.items()},
            "stopped": {k: v.model_dump(mode="json") for k, v in stopped.items()},
        }

    def _matrix(self, chat: list[CallSite], adapters: dict[str, Adapter], evals: Path) -> dict[str, Any]:
        """Adapter v0 on every tier's dev cases, as one eval matrix (a Nebius Serverless Job when one is
        configured, else locally; Task 11). Fills the cache the routing loop then replays for free."""
        self._stage(Stage.EVALUATING)
        sites = [s for s in chat if s.id in adapters]
        if not sites:
            return {}
        cases = [c for s in sites for c in baselines.load_cases(evals, s.id)]
        refs = [r for s in sites for r in baselines.baselines_for(evals, s.id).values()]
        matrix = Matrix(
            sites=sites,
            adapters={s.id: {t: adapters[s.id] for t in self.config.tiers} for s in sites},
            cases=cases,
            references=refs,
            tiers=self.config.tiers,
            splits=["dev"],
            thresholds=Thresholds(self.config.threshold, self.config.reasoning_threshold),
            live=self.config.live,
            budget=remaining_caps(self.budget),
        )
        out = self.eval_matrix(matrix)
        where = f"a Nebius Serverless Job ({out.job_id})" if out.ran_on == "job" else "locally"
        reason = f" ({out.fallback_reason})" if out.fallback_reason else ""
        self._log("info", f"eval matrix ({len(out.results)} cells) ran {where}{reason}")
        for entry in out.results:
            site_id, cell = entry["site_id"], f"{entry['site_id']} {entry['tier']} {entry['split']}"
            if entry["kind"] == "tier_result":
                result = {k: v for k, v in entry["result"].items() if k != "case_scores"}
                self.log.emit("tier_result", result, call_site_id=site_id)
            elif entry["kind"] == "error" and entry["error_type"] == "CacheMiss":
                raise CacheMiss(f"{cell}: {entry['message']}")
            else:
                why = entry.get("reason") or f"{entry['error_type']}: {entry['message']}"
                self._log("warn", f"{cell}: {why}"[:300], site_id)
        try:
            absorb_usage(self.budget, out)
        except BudgetExceeded as e:  # the routing loop's next call stops on it: KEEP_CLOSED("budget: ...")
            self._log("warn", f"per-run cap reached in the eval matrix: {e}")
        return out.model_dump(mode="json", exclude={"cache"})

    def _routes(
        self,
        chat: list[CallSite],
        adapters: dict[str, Adapter],
        stopped: dict[str, Verdict],
        evals: Path,
        schemas: dict[str, Any],
        matrix: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Route each site (Task 9) to its verdict, given the cells the eval matrix measured (reused, never
        judged again). evals/results/<site>.json is written per site, so a resumed run routes only the sites
        it had not finished (write_results adds the final verdict after the patch stage). Verdicts are
        announced together at VERDICTS."""
        cells: dict[str, dict[tuple[Tier, int, str], TierResult]] = {}
        for entry in matrix.get("results", []):
            if entry["kind"] == "tier_result":
                r = TierResult.model_validate(entry["result"])
                cells.setdefault(r.site_id, {})[r.tier, r.adapter_version, r.split] = r
        done = {s.id for s in chat if (self.dir / "evals" / "results" / f"{s.id}.json").exists()}
        for site in chat:  # verdict first: every site's securing holdout before any site iterates
            if site.id in done or site.id in stopped or site.id not in cells:
                continue
            self._stage(Stage.ITERATING)
            cells[site.id] |= secure_holdouts(
                site,
                baselines.load_cases(evals, site.id),
                baselines.baselines_for(evals, site.id),
                adapters[site.id],
                self.client,
                self.config,
                cells[site.id],
                runner=self._runner(site),
                on_event=lambda t, p, sid=site.id: self.log.emit(t, p, call_site_id=sid),
            )
        self._save_budget()
        out = {}
        for site in chat:
            path = self.dir / "evals" / "results" / f"{site.id}.json"
            if not path.exists():
                self._stage(Stage.ITERATING)
                if site.id in stopped:
                    result = {
                        "verdict": stopped[site.id].model_dump(mode="json"),
                        "adapter": None,
                        "tier_results": [],
                    }
                else:
                    result = self._route_site(
                        site, adapters[site.id], evals, schemas.get(site.id) or None, cells.get(site.id, {})
                    )
                _write_json(path, {"site": site.model_dump(mode="json"), **result})
                self._save_budget()
            out[site.id] = json.loads(path.read_text(encoding="utf-8"))
        return out

    def _route_site(
        self,
        site: CallSite,
        adapter: Adapter,
        evals: Path,
        schemas: dict | None,
        prior: dict[tuple[Tier, int, str], TierResult],
    ) -> dict[str, Any]:
        def on_event(event_type: str, payload: dict[str, Any]) -> None:
            if event_type != "verdict":  # announced at VERDICTS
                self.log.emit(event_type, payload, call_site_id=site.id)

        closed = baselines.load_closed(evals, site.id)
        circular = self._closed_auditors(evals, site, closed)
        if circular:
            self._log("info", f"{site.id}: closed reference {circular} not compared: it blind-audited these "
                      "references and only the ones it agreed with were kept, so it agrees with them by "
                      "construction", site.id)  # fmt: skip
        route = self.route(
            site,
            baselines.load_cases(evals, site.id),
            baselines.baselines_for(evals, site.id),
            adapter,
            self.client,
            self.config,
            closed=None if circular else closed,  # GC27c: no comparison row (nor reason clause) when circular
            schemas=schemas,
            runner=self._runner(site),
            price_lookup=self.price_lookup,
            on_event=on_event,
            prior=prior,
        )
        verdict = route.verdict
        if circular and route.adapter is not None:  # compared, but not with the closed model: say which one
            verdict = verdict.model_copy(update={"closed_reference_model": circular})
        v0 = [f.removeprefix(f"{REFUSED}: ") for f in adapter.risk_flags if f.startswith(f"{REFUSED}: ")]
        refused = [f"Adapter v0 refused before evaluation (the translation ran instead): {w}" for w in v0]
        return {
            "verdict": verdict.model_dump(mode="json"),
            "adapter": route.adapter.model_dump(mode="json") if route.adapter else None,
            "tier_results": [r.model_dump(mode="json") for r in route.results],
            "refused": refused + route.refused,
        }

    def _runner(self, site: CallSite) -> Callable[..., Any]:
        def runner(*args: Any, **kwargs: Any) -> Any:
            """run_tier, plus each measured (site, tier, split) as a LangSmith experiment (GC23; no-op without
            a key). Routing calls it with (site, cases, refs, adapter, tier, client, split, config)."""
            out = run_tier(*args, **kwargs)
            result, calls = out if isinstance(out, tuple) else (out, [])
            if isinstance(result, TierResult) and calls:
                cases = {c.id: c for c in args[1]}
                scores = {s.case_id: s for s in result.case_scores}
                rows = [langsmith.CaseResult(cases[c.case_id], c, scores[c.case_id]) for c in calls]
                langsmith.log_experiment(site, result.tier, result.split, rows, run_id=self.log.run_id)
            return out

        return runner

    @staticmethod
    def _closed_auditors(evals: Path, site: CallSite, closed: dict[str, Any]) -> str | None:
        """The closed model, if it also blind-audited the site's references (GC18, GC27b): the references are
        then the cases it agreed with, and its parity on them is 100% by construction, not evidence (GC27c).
        A verdict for such a site names the model with closed_reference_parity_pct None."""
        refs = baselines.load_references(evals, site.id).values()
        auditors = {v.removeprefix("blind:") for r in refs for v in r.verified_by if v.startswith("blind:")}
        return ", ".join(sorted({r.model for r in closed.values()} & auditors)) or None

    def _verdicts(self, chat: list[CallSite], routes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """Each site's verdict, a MIGRATE one with its tier's Token Factory price and where it was read."""
        self._stage(Stage.VERDICTS)
        verdicts, out = [routes[s.id]["verdict"] for s in chat], []
        tiers = {Tier(v["tier"]) for v in verdicts if v["decision"] == "MIGRATE"}
        after = self.after_prices([t for t in TIER_ORDER if t in tiers]) if tiers else {}
        for v in verdicts:
            if v["decision"] == "MIGRATE":
                v = v | {"after_price": after[Tier(v["tier"])].model_dump(mode="json")}
            self.log.emit("verdict", v, call_site_id=v["site_id"])
            out.append(v)
        return out

    def _patch(
        self,
        repo: Path,
        chat: list[CallSite],
        verdicts: list[dict[str, Any]],
        routes: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Patch Composer (Task 10) on a sandbox copy: a MIGRATE site whose patch fails is MANUAL_REVIEW."""
        before = [Verdict.model_validate(v) for v in verdicts]
        if not any(v.decision in ("MIGRATE", "KEEP_CLOSED") for v in before):
            self._stage(Stage.PATCHED, "nothing to patch: every site needs a manual review")
            return {"files": {}, "verdicts": verdicts, "adapters": {}, "site_files": {}}
        adapters = {k: Adapter.model_validate(r["adapter"]) for k, r in routes.items() if r["adapter"]}
        result = self.patch(repo, chat, before, adapters, self.client, self.dir / "patch", config=self.config)
        for message in result.warnings:
            self._log("warn", f"patch: {message}")
        after = {v.site_id: v for v in result.verdicts}
        migrated = sum(v.decision == "MIGRATE" for v in result.verdicts)
        kept = sum(v.decision == "KEEP_CLOSED" for v in result.verdicts)
        self._stage(Stage.PATCHED, f"Patched {migrated} call sites; {kept} kept with a comment")
        for v in before:
            if after[v.site_id] != v:  # downgraded: re-announce the site's verdict
                self.log.emit("verdict", after[v.site_id].model_dump(mode="json"), call_site_id=v.site_id)
        return result.model_dump(mode="json")

    def _pr(
        self,
        repo: Path,
        chat: list[CallSite],
        patch: dict[str, Any],
        routes: dict[str, dict[str, Any]],
        evals: Path,
        cases: dict[str, Any],
    ) -> dict[str, Any]:
        """PR Writer (Task 10): the body is always rendered (it is in the report); the PR is opened only with
        config.open_pr and something to commit."""
        gh = github_repo(self.config.repo_url, repo)
        draft = self.pr(
            run_id=self.log.run_id,
            repo=gh or self.config.repo_url,
            config=self.config,
            sites=chat,
            patch=patch,
            routes=routes,
            evals=evals,
            cases=cases,
            client=self.client,
            open_pr=self.config.open_pr and bool(patch["files"]) and gh is not None,
        )
        if draft.get("url"):
            announce_pr(self.log, draft["url"], draft["title"])
        elif not self.config.open_pr:
            self._log("info", "pull request not opened (--no-pr); its body is in the report")
        elif not patch["files"]:
            self._log("info", "pull request not opened: no file changed")
        else:
            raise RuntimeError(
                f"cannot open a pull request: {self.config.repo_url} is not a GitHub repository"
            )
        return draft

    def _finish(self, chat: list[CallSite], cases: dict[str, Any], patch: dict[str, Any]) -> None:
        summaries = [CaseSummary.model_validate(s) for s in cases["summaries"].values()]
        verdicts = [Verdict.model_validate(v) for v in patch["verdicts"]]
        run_totals = totals(chat, summaries, verdicts, self.budget)
        self._stage(Stage.DONE)
        self.log.emit("done", {"totals": run_totals.model_dump(mode="json")})
        _write_json(self.dir / "report.json", report(self.dir, self.log.run_id, self.log.events))

    # -------------------------------------------------------- defaults: the other tasks' stages

    def _run_matrix_job(self, matrix: Matrix) -> MatrixResults:
        from offramp.evals.serverless_job.submit import run_matrix_job

        return run_matrix_job(matrix, mode="auto", cache_dir=self.cache_dir, run_id=self.log.run_id)


def open_run_pr(
    repo: str,
    run_id: str,
    files: dict[str, str],
    title: str,
    body: str,
    opener: Callable[..., str] | None = None,
) -> str:
    """The PR stage's GitHub step, shared by a run and by `offramp pr`: one commit of `files` on branch
    offramp/<run_id> of `repo` ("owner/name"), then the PR; its URL. `opener` stands in for github.open_pr."""
    return (opener or github.open_pr)(repo, run_id=run_id, files=files, title=title, body=body)


def announce_pr(log: RunLog, url: str, title: str) -> None:
    """The PR stage's events, shared by a run and by `offramp pr`: the `pr` event, then PR_OPENED."""
    log.emit("pr", {"url": url, "title": title})
    log.emit("stage", {"stage": Stage.PR_OPENED.value})


def open_pr_from_run(
    run_id: str, root: Path | None = None, *, opener: Callable[..., str] | None = None, **log_kw: Any
) -> str:
    """`offramp pr <run_id>`: open the PR of a finished `--no-pr` run from its own artifacts (patch.json's
    files, pr.json's title and body, on the repo its config names), with no model call, so a capped run is
    never continued to get its PR. The run's log gains the PR stage's events, then DONE again, so its
    snapshot and export-replay show the PR. ValueError when the run has not finished, already opened its
    PR, migrated nothing, or is not on GitHub."""
    run_dir = (root or data_dir()) / "runs" / run_id
    log = RunLog(run_dir, run_id, **log_kw)  # engine, on_event: as `offramp run` mirrors and shows events
    if not log.events or log.stage != Stage.DONE or log.events[-1].type != "done":
        raise ValueError(
            f"run {run_id} has not finished (stage {log.stage.value}): only a DONE run's PR opens"
        )
    if opened := next((e.payload["url"] for e in log.events if e.type == "pr"), None):
        raise ValueError(f"run {run_id} already opened its PR: {opened}")
    patch = json.loads((run_dir / "patch.json").read_text(encoding="utf-8"))
    if not patch["files"] or not any(v["decision"] == "MIGRATE" for v in patch["verdicts"]):
        raise ValueError(f"run {run_id} migrated nothing: there is no PR to open")
    clone = json.loads((run_dir / "clone.json").read_text(encoding="utf-8"))
    repo = github_repo(log.config.repo_url, Path(clone["path"]))
    if repo is None:
        raise ValueError(f"run {run_id}: {log.config.repo_url} is not a GitHub repository")
    draft = json.loads((run_dir / "pr.json").read_text(encoding="utf-8"))
    url = open_run_pr(repo, run_id, patch["files"], draft["title"], draft["body"], opener)
    _write_json(run_dir / "pr.json", draft | {"url": url})
    done = log.events[-1].payload  # the run's totals, unchanged: nothing was spent
    announce_pr(log, url, draft["title"])
    log.emit("stage", {"stage": Stage.DONE.value})
    log.emit("done", done)
    _write_json(run_dir / "report.json", report(run_dir, run_id, log.events))
    return url


def write_and_open_pr(
    *,
    run_id: str,
    repo: str,
    config: RunConfig,
    sites: list[CallSite],
    patch: dict[str, Any],
    routes: dict[str, dict[str, Any]],
    cases: dict[str, Any],
    client: LLM,
    open_pr: bool,
    **_: Any,
) -> dict[str, Any]:
    """PR Writer (Task 10): Super's prose around the evidence rendered from the run's measurements, then the
    PR on GitHub when `open_pr` (branch offramp/<run_id>, the patched files in one commit)."""
    tier_results = [TierResult.model_validate(r) for s in sites for r in routes[s.id]["tier_results"]]
    adapters = {k: r["adapter"] for k, r in routes.items() if r["adapter"]} | patch.get("adapters", {})
    shared = [s for s in sites if s.id in cases["summaries"]]  # the sites whose dataset was uploaded
    public = langsmith.public_links()  # a share link publishes the target's prompts and inputs: opt-in only
    link = langsmith.share_link if public else langsmith.dataset_url
    links = {s.id: url for s in shared if (url := link(s, run_id=run_id))}
    ctx = PRContext(
        run_id=run_id,
        repo=repo,
        config=config,
        sites=sites,
        verdicts=patch["verdicts"],
        tier_results=tier_results,
        adapters=adapters,
        case_counts=cases["case_counts"],
        references=cases["references"],
        audit={k: v["audit"] for k, v in cases["summaries"].items()},
        free_text_args=cases["free_text_args"],
        langsmith_links=links,
        langsmith_public=public,
        site_files=patch.get("site_files", {}),
        sandbox=patch.get("sandbox_mode"),
        linted=patch.get("linted", True),
        sandbox_notes=patch.get("sandbox_notes", []),
        async_client=patch.get("async_client", False),
        client_packaged=patch.get("client_packaged", True),
        change_notes=patch.get("change_notes", []),
        client_path=patch.get("client_path"),
        alone=patch.get("alone", {}),
        combined=patch.get("combined", []),
        combined_repair=patch.get("combined_repair", []),
        refused={k: r["refused"] for k, r in routes.items() if r.get("refused")},
    )
    draft = write_pr(ctx, client)
    files, title, body = patch["files"], scrub(draft.title), scrub(draft.body)  # reasons carry sandbox output
    url = open_run_pr(repo, run_id, files, title, body) if open_pr else None
    return {"title": title, "body": body, "url": url}


def default_price_lookup(config: RunConfig, client: LLM) -> Callable[[str], Any]:
    """Source-model prices for the cost-before estimate (GC21): Tavily at runtime for live runs, with the
    extraction call on the run's own client; pricing.yaml for cache-only runs, which never call Tavily."""
    if not config.live:
        return pricing_yaml_price
    return lambda model: lookup_source_model(model, llm=client)


def execute_run(log: RunLog, *, cache_dir: str | Path | None = None, cache_only: bool = False) -> Stage:
    """Execute (or resume) the run in `log` with a Token Factory client built from its config. `cache_only`
    replays the response cache even when the run was started live (`offramp run --resume ID --cache-only`)."""
    if log.config is None:
        raise ValueError(f"run {log.run_id} has no QUEUED event with its config")
    live = log.config.live and not cache_only
    client = TFClient(cache_dir=cache_dir, live=live, budget=Budget())
    return Pipeline(log, client, cache_dir=cache_dir, live=live).execute()
