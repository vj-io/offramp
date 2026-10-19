"""Record closed-model baselines from each call site's own source model (PRD G2/§6.3, PLAN GC22).

For teams that have the source model's keys (OPENAI_API_KEY / ANTHROPIC_API_KEY). The Offramp demo has none,
so it never runs this; its baselines are references (GC17). Outputs go to evals/closed/<site>.jsonl as
BaselineResult(source="closed_model") with measured latency, tokens and cost (pricing.yaml).

    uv run --extra baseline python scripts/record_baselines.py <repo> [--sites cs_01,cs_03] [--self-agreement]

--self-agreement records a second pass and scores it against the first with the site's own scorer: the
closed model's agreement with itself, the noise floor any candidate is compared under
(evals/closed/self_agreement.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from offramp import pricing
from offramp.evals import baselines as b
from offramp.evals.scorers import score_case
from offramp.schemas import BaselineResult, CallSite, EvalCase, ModelOutput, ToolCall

OPENAI_PARAMS = (
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "seed",
    "tools",
    "tool_choice",
    "response_format",
)
ANTHROPIC_PARAMS = ("max_tokens", "temperature", "top_p", "tools", "tool_choice")


def _literal_params(site: CallSite, keys: tuple[str, ...]) -> dict[str, Any]:
    """The site's resolved call parameters; scanner placeholders ("<expr>", "<pydantic:X>") are left out."""

    def unresolved(v: Any) -> bool:
        return isinstance(v, str) and v.startswith("<") and v.endswith(">")

    return {k: site.params[k] for k in keys if k in site.params and not unresolved(site.params[k])}


def _openai_output(resp: Any) -> tuple[ModelOutput, int, int]:
    msg = resp.choices[0].message
    calls = [
        ToolCall(name=c.function.name, arguments=json.loads(c.function.arguments or "{}"))
        for c in msg.tool_calls or []
    ]
    return (
        ModelOutput(text=msg.content, tool_calls=calls),
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


def _anthropic_output(resp: Any) -> tuple[ModelOutput, int, int]:
    text = "".join(block.text for block in resp.content if block.type == "text")
    calls = [
        ToolCall(name=block.name, arguments=dict(block.input))
        for block in resp.content
        if block.type == "tool_use"
    ]
    output = ModelOutput(text=text or None, tool_calls=calls)
    return output, resp.usage.input_tokens, resp.usage.output_tokens


def call_closed(site: CallSite, case: EvalCase, openai_client: Any = None, anthropic_client: Any = None):
    """Run one case on the site's source model through its own SDK; return a measured BaselineResult."""
    started = time.perf_counter()
    if site.sdk == "anthropic":
        system = [m["content"] for m in case.input_messages if m["role"] == "system"]
        kwargs = {
            "max_tokens": 1024,
            **_literal_params(site, ANTHROPIC_PARAMS),
        }  # Anthropic requires max_tokens
        if system:
            kwargs["system"] = system[0]
        resp = anthropic_client.messages.create(
            model=site.source_model,
            messages=[m for m in case.input_messages if m["role"] != "system"],
            **kwargs,
        )
        output, tin, tout = _anthropic_output(resp)
    elif site.sdk == "openai":
        resp = openai_client.chat.completions.create(
            model=site.source_model, messages=case.input_messages, **_literal_params(site, OPENAI_PARAMS)
        )
        output, tin, tout = _openai_output(resp)
    else:
        raise ValueError(f"{site.id}: recording {site.sdk} call sites is not supported")
    latency_ms = round((time.perf_counter() - started) * 1000)
    try:
        cost = pricing.cost_usd(site.source_model, tin, tout)
    except KeyError:
        cost = None  # unknown price: report nothing rather than a guess
    return BaselineResult(
        case_id=case.id,
        source="closed_model",
        model=site.source_model,
        output=output,
        latency_ms=latency_ms,
        input_tokens=tin,
        output_tokens=tout,
        cost_usd=cost,
        recorded_at=datetime.now(UTC),
    )


def record(
    evals_dir: Path,
    site_ids: list[str] | None = None,
    self_agreement: bool = False,
    openai_client: Any = None,
    anthropic_client: Any = None,
) -> dict[str, int]:
    evals_dir = Path(evals_dir)
    sites = [s for s in b.load_sites(evals_dir) if site_ids is None or s.id in site_ids]
    if openai_client is None and any(s.sdk == "openai" for s in sites):
        from openai import OpenAI

        openai_client = OpenAI()
    if anthropic_client is None and any(s.sdk == "anthropic" for s in sites):
        from anthropic import Anthropic

        anthropic_client = Anthropic()
    summary, agreement = {}, {}
    for site in sites:
        cases = b.load_cases(evals_dir, site.id)
        first = [call_closed(site, c, openai_client, anthropic_client) for c in cases]
        b.save_closed(
            evals_dir,
            site.id,
            {**b.load_closed(evals_dir, site.id), **{r.case_id: r for r in first}}.values(),
        )
        summary[site.id] = len(first)
        if self_agreement and cases and site.contract and site.contract.task_type != "free_text":
            second = [call_closed(site, c, openai_client, anthropic_client) for c in cases]
            scores = [
                score_case(c, a.output, z.output, site.contract)
                for c, a, z in zip(cases, first, second, strict=True)
            ]
            agreement[site.id] = sum(s.passed for s in scores) / len(scores)
    if agreement:
        path = evals_dir / "closed" / "self_agreement.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps({**old, **agreement}, indent=1, sort_keys=True) + "\n")
    b.write_manifest(evals_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repo", type=Path)
    parser.add_argument("--sites", help="comma-separated site ids (default: all prepared sites)")
    parser.add_argument("--self-agreement", action="store_true")
    args = parser.parse_args(argv)
    summary = record(args.repo / "evals", args.sites.split(",") if args.sites else None, args.self_agreement)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
