"""LangSmith helpers (Global Constraint 23): traces, datasets and experiments for the audit trail.

Thin and import-guarded: without LANGSMITH_API_KEY, or without the optional `langsmith` package, every helper
does nothing and returns None. With a key, a LangSmith failure is logged (never the key, never the SDK's
message) and also returns None: telemetry must not fail a run. Rejected / out-of-credit keys move to
LANGSMITH_API_KEY_FALLBACK once (Global Constraint 28).

What lands in LangSmith for a run `r1`: the traces of every Token Factory call in project `offramp-r1`
(`run_project`), one dataset `offramp-r1-<site>` per call site holding its cases and references
(`upload_dataset`), and one experiment `offramp-r1-<site>-<tier>-<split>` per evaluation (`log_experiment`).
"""

import contextlib
import logging
import os
import uuid
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

from offramp.integrations import keys
from offramp.schemas import BaselineResult, CallSite, CandidateResult, CaseScore, EvalCase, Split, Tier

log = logging.getLogger(__name__)


class CaseResult(NamedTuple):
    """One evaluated case: what was asked, what the tier answered, how it scored."""

    case: EvalCase
    candidate: CandidateResult
    score: CaseScore


def dataset_name(run_id: str, site_id: str) -> str:
    return f"offramp-{run_id}-{site_id}"


def experiment_name(run_id: str, site_id: str, tier: Tier | str, split: Split) -> str:
    return f"{dataset_name(run_id, site_id)}-{tier}-{split}"


def _langsmith() -> Any | None:
    try:
        import langsmith  # the optional SDK (absolute import: this module is offramp.integrations.langsmith)
    except ImportError:
        return None
    return langsmith


def _new_client(key: str) -> Any | None:
    sdk = _langsmith()
    # auto_batch_tracing off: each call is synchronous, so a rejected key raises here instead of being
    # swallowed by a background thread
    return sdk.Client(api_key=key, auto_batch_tracing=False) if sdk else None


def _primary() -> str | None:
    return os.environ.get("LANGSMITH_API_KEY")


def tracing_enabled() -> bool:
    """Key present, LANGSMITH_TRACING=true (the switch TFClient honours) and the SDK importable."""
    on = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    return bool(on and keys.current("langsmith", _primary()) and _langsmith())


@contextlib.contextmanager
def run_project(run_id: str) -> Iterator[str | None]:
    """`with run_project(run_id) as project:` sends every traced call made inside it (TFClient wraps its
    OpenAI client when LANGSMITH_TRACING=true) to LangSmith project `offramp-<run_id>`; yields that name, or
    None (and does nothing) when tracing is off.

    The context is a contextvar: calls made from a worker thread see it only if that thread runs inside
    `contextvars.copy_context().run(...)`.
    """
    sdk = _langsmith() if tracing_enabled() else None
    client = _new_client(keys.current("langsmith", _primary()) or "") if sdk else None
    if sdk is None or client is None:
        yield None
        return
    project = f"offramp-{run_id}"
    with sdk.tracing_context(project_name=project, client=client, enabled=True):
        yield project


def _call[T](what: str, fn: Callable[[Any], T]) -> T | None:
    """fn(client), or None when there is no key / SDK or the call failed; one retry on the fallback key."""
    for attempt in range(2):
        key = keys.current("langsmith", _primary())
        client = _new_client(key) if key else None
        if client is None:
            return None
        try:
            return fn(client)
        except Exception as e:  # telemetry must not fail a run; what is logged is the class and status only
            status, body = keys.status_and_body(e)
            if attempt == 0 and keys.switch_to_fallback("langsmith", key, status, body):
                continue
            log.warning(
                "langsmith: %s failed: %s%s", what, type(e).__name__, f" (HTTP {status})" if status else ""
            )
            return None
    return None


def _example_id(dataset: str, case_id: str) -> uuid.UUID:
    """Stable per (dataset, case): log_experiment links its runs to the examples without listing them."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset}/{case_id}")


def _example(dataset: str, case: EvalCase, ref: BaselineResult | None) -> dict[str, Any]:
    meta = {
        "case_id": case.id,
        "site_id": case.site_id,
        "tags": case.tags,
        "source": case.source,
        "expected_kind": case.expected_kind,
    }
    if ref:
        meta |= {"reference_model": ref.model, "reference_source": ref.source, "verified_by": ref.verified_by}
    return {
        "id": _example_id(dataset, case.id),
        "inputs": {"messages": case.input_messages},
        "outputs": {"reference": ref.output.model_dump(mode="json")} if ref else {},
        "metadata": meta,
        "split": case.split,
    }


def upload_dataset(
    site: CallSite, cases: Sequence[EvalCase], references: Sequence[BaselineResult], *, run_id: str
) -> str | None:
    """The site's cases and their references as dataset `offramp-<run_id>-<site.id>`; its URL.

    `run_id` is required: dataset names are workspace-wide, and two runs of the same site id must not share
    (or silently reuse) one. Calling again for the same run and site returns the existing dataset's URL.
    """
    if not cases:
        return None
    name = dataset_name(run_id, site.id)
    refs = {r.case_id: r for r in references}

    def go(client: Any) -> str | None:
        if client.has_dataset(dataset_name=name):
            return client.read_dataset(dataset_name=name).url
        dataset = client.create_dataset(
            name,
            description=f"Offramp {site.id} ({site.file}:{site.lines[0]}): {len(cases)} cases, run {run_id}",
            metadata={
                "run_id": run_id,
                "site_id": site.id,
                "file": site.file,
                "source_model": site.source_model,
            },
        )
        client.create_examples(
            dataset_id=dataset.id, examples=[_example(name, c, refs.get(c.id)) for c in cases]
        )
        return dataset.url

    return _call("upload_dataset", go)


def log_experiment(
    site: CallSite, tier: Tier, split: Split, case_results: Sequence[CaseResult], *, run_id: str
) -> str | None:
    """One (site, tier, split) evaluation as an experiment on the site's dataset (see upload_dataset): a run
    per case linked to its example, with `passed` and `score` feedback. Returns the experiment's URL.

    ponytail: one request per run and per feedback, in order; batch if a run ever logs thousands of cases.
    """
    if not case_results:
        return None
    dataset, name = dataset_name(run_id, site.id), experiment_name(run_id, site.id, tier, split)

    def go(client: Any) -> str | None:
        project = client.create_project(
            name,
            reference_dataset_id=client.read_dataset(dataset_name=dataset).id,
            metadata={
                "run_id": run_id,
                "site_id": site.id,
                "tier": str(tier),
                "split": split,
                "cases": len(case_results),
                "passed": sum(r.score.passed for r in case_results),
            },
            upsert=True,
        )
        for r in case_results:
            run = uuid.uuid4()
            end = datetime.now(UTC)
            c = r.candidate
            client.create_run(
                id=run,
                name=f"offramp-{tier}",
                run_type="llm",
                project_name=name,
                reference_example_id=_example_id(dataset, r.case.id),
                inputs={"messages": r.case.input_messages},
                outputs=c.output.model_dump(mode="json") if c.output else None,
                error=c.error,
                start_time=end - timedelta(milliseconds=c.latency_ms),
                end_time=end,
                tags=[str(tier), split],
                extra={
                    "metadata": {
                        "tier": str(tier),
                        "split": split,
                        "adapter_version": c.adapter_version,
                        "latency_ms": c.latency_ms,
                        "input_tokens": c.input_tokens,
                        "output_tokens": c.output_tokens,
                        "cost_usd": c.cost_usd,
                        "cached": c.cached,
                    }
                },
            )
            passed = 1.0 if r.score.passed else 0.0
            client.create_feedback(
                run, "passed", score=passed, comment=r.score.detail or None, session_id=project.id
            )
            client.create_feedback(run, "score", score=r.score.score, session_id=project.id)
        return project.url

    return _call("log_experiment", go)


def share_link(site: CallSite, *, run_id: str) -> str | None:
    """Public read-only link to the site's dataset (its experiments come along). Anyone with the link can open
    it: call this only for runs whose evidence is meant to be public (the report and PR)."""
    return _call(
        "share_link", lambda client: client.share_dataset(dataset_name=dataset_name(run_id, site.id))["url"]
    )
