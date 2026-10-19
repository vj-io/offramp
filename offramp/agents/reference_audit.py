"""Reference audit (PLAN GC18, GC27b): every `reference:seed` is answered blind by an independent model and
scored against the seed with the site's own scorer. A disagreement drops the case and its reference; both
are kept, with the reason, in references/<site>.dropped.jsonl, which is also the human-audit queue
(scripts/audit_references.py).

Auditor per case: the closed reference (GC27: a Claude Sonnet subagent answers the exact input_messages
outside this code; its rows in closed/<site>.jsonl are loaded here), so Nemotron never grades gold that
Nemotron wrote. Only when an `llm` is passed does Ultra (role `reference_audit`, thinking on, batched) answer
the cases the closed reference has not. Oracle and fixture references are not blind-audited: their answers
come from code. Per-site counts go to evals/audit.json. Re-running never re-asks an audited reference.

    python -m offramp.agents.reference_audit <repo> --export-blind [--out DIR]  # closed auditor's bundle
    python -m offramp.agents.reference_audit <repo> [--closed answers.jsonl] [--ultra-fallback]

answers.jsonl rows: {"case_id": "cs_01:004", "output": {"text": ...} | {"tool_calls": [...]}}. References that
carry a human verdict (verified_by "human:...") are final: no auditor ever re-checks or drops them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from offramp.evals import baselines
from offramp.evals.scorers import score_case
from offramp.schemas import BaselineResult, CallSite, EvalCase, ModelOutput, Tier, ToolCall
from offramp.tf.client import LLM

ROLE = "reference_audit"
SEED = "reference:seed"
HUMAN = "human:tendem"
BATCH = 10  # cases per Ultra call: keeps the audit far inside the 40 Ultra-agent-call cap
MAX_TOKENS = 16000  # thinking is on and its tokens count toward max_tokens (GC25)


def _blind(ref: BaselineResult) -> list[str]:
    return [v for v in ref.verified_by if v.startswith("blind:")]


def dropped_row(
    case: EvalCase, ref: BaselineResult, by: str, reason: str, auditor_output: ModelOutput | None = None
) -> dict[str, Any]:
    return {
        "case": case.model_dump(mode="json"),
        "reference": ref.model_dump(mode="json"),
        "by": by,
        "auditor_output": auditor_output.model_dump(mode="json") if auditor_output else None,
        "reason": reason,
        "restored": False,
    }


# ---------------------------------------------------------------- Ultra fallback (blind, batched)


def _answer_schema(site: CallSite) -> dict[str, Any]:
    contract = site.contract
    if contract.task_type == "classification":
        return {"label": {"type": "string", **({"enum": contract.labels} if contract.labels else {})}}
    if contract.task_type == "json_extraction":
        return {"output": contract.json_schema or {"type": "object"}}
    if contract.task_type == "tool_call":
        names = [t.get("function", t)["name"] for t in site.params.get("tools") or [] if isinstance(t, dict)]
        return {
            "tool": {"type": "string", **({"enum": names} if names else {})},
            "arguments": {"type": "object"},
        }
    return {"answer": {"type": "string"}}  # reasoning: the final answer


def _to_output(site: CallSite, answer: dict[str, Any]) -> ModelOutput:
    kind = site.contract.task_type
    if kind == "classification":
        return ModelOutput(text=answer["label"])
    if kind == "json_extraction":
        return ModelOutput(text=json.dumps(answer["output"], ensure_ascii=False))
    if kind == "tool_call":
        return ModelOutput(
            tool_calls=[ToolCall(name=answer["tool"], arguments=answer.get("arguments") or {})]
        )
    return ModelOutput(text=f"ANSWER: {answer['answer']}")


_FORMAT = {
    "classification": 'Give "label": exactly the label the call should reply with.',
    "json_extraction": 'Give "output": the JSON object the call should reply with.',
    "tool_call": 'Give "tool" and "arguments": the single tool call the call should make next.',
    "reasoning": 'Give "answer": the final answer only (what the call puts on its last `ANSWER:` line).',
}


def _system(case: EvalCase) -> str | None:
    return next((m["content"] for m in case.input_messages if m["role"] == "system"), None)


def _audit_prompt(site: CallSite, cases: list[EvalCase]) -> str:
    shared = _system(cases[0])
    tools = site.params.get("tools") if site.contract.task_type == "tool_call" else None
    parts = [f"## System message of the call\n{shared or '(none)'}"]
    if tools:
        parts.append(f"## Tools the call can use\n{json.dumps(tools, ensure_ascii=False)}")
    parts.append(f"## What to give for each case\n{_FORMAT[site.contract.task_type]}")
    blocks = []
    for case in cases:
        own = _system(case)
        head = f"### case_id: {case.id}" + (
            f"\nSystem message for this case:\n{own}" if own != shared else ""
        )
        body = "\n\n".join(
            f"[{m['role']}]\n{m['content']}" for m in case.input_messages if m["role"] != "system"
        )
        blocks.append(f"{head}\n{body}")
    parts.append("## Cases\n" + "\n\n".join(blocks))
    parts.append(
        f'Reply {{"answers": [...]}} with one answer per case ({len(cases)}), each with its "case_id".'
    )
    return "\n\n".join(parts)


def _ultra_answers(site: CallSite, cases: list[EvalCase], llm: LLM) -> dict[str, tuple[str, ModelOutput]]:
    item = {
        "type": "object",
        "properties": {"case_id": {"type": "string"}, **_answer_schema(site)},
        "required": ["case_id", *_answer_schema(site)],
    }
    schema = {
        "type": "object",
        "properties": {"answers": {"type": "array", "items": item}},
        "required": ["answers"],
    }
    out: dict[str, tuple[str, ModelOutput]] = {}
    for start in range(0, len(cases), BATCH):
        batch = cases[start : start + BATCH]
        result = llm.chat(
            role=ROLE,
            tier=Tier.ULTRA,
            messages=[
                {
                    "role": "system",
                    "content": "You are an independent auditor. For each case, give the output the "
                    "call described below should produce for that case's input, judging "
                    "each case on its own. Reply with JSON only.",
                },
                {"role": "user", "content": _audit_prompt(site, batch)},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            json_schema=schema,
            reasoning="high",
        )
        parsed = result.parsed
        if parsed is None:
            try:
                parsed = json.loads(result.output.text or "")
            except ValueError:
                parsed = {}
        ids = {c.id for c in batch}
        for answer in parsed.get("answers", []) if isinstance(parsed, dict) else []:
            if isinstance(answer, dict) and answer.get("case_id") in ids:
                try:
                    out[answer["case_id"]] = (f"blind:{result.model}", _to_output(site, answer))
                except (KeyError, TypeError, ValueError):
                    pass  # malformed answer: the case stays unchecked
    return out


# ---------------------------------------------------------------- audit


def audit_site(evals_dir: Path, site: CallSite, llm: LLM | None = None) -> None:
    cases = {c.id: c for c in baselines.load_cases(evals_dir, site.id)}
    refs = baselines.load_references(evals_dir, site.id)
    todo = [
        c
        for c in cases.values()
        if c.id in refs
        and refs[c.id].model == SEED
        and not _blind(refs[c.id])
        and not any(v.startswith("human:") for v in refs[c.id].verified_by)
    ]  # an expert verdict is final
    closed = baselines.load_closed(evals_dir, site.id)
    answers = {c.id: (f"blind:{closed[c.id].model}", closed[c.id].output) for c in todo if c.id in closed}
    rest = [c for c in todo if c.id not in answers]
    if llm is not None and rest:
        answers |= _ultra_answers(site, rest, llm)
    if not answers:
        return
    dropped = baselines.load_dropped(evals_dir, site.id)
    for case_id, (tag, output) in sorted(answers.items()):
        ref = refs[case_id]
        score = score_case(cases[case_id], ref.output, output, site.contract)
        if score.passed:
            refs[case_id] = ref.model_copy(update={"verified_by": [*ref.verified_by, tag]})
        else:
            reason = f"{tag} disagrees: {score.detail or score.error_category}"
            dropped.append(dropped_row(cases.pop(case_id), refs.pop(case_id), tag, reason, output))
    baselines.save_cases(evals_dir, site.id, cases.values())
    baselines.save_references(evals_dir, site.id, refs.values())
    baselines.save_dropped(evals_dir, site.id, dropped)


def summarize(evals_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-site counts derived from the files alone (so they survive re-runs and human ingests); writes
    audit.json. checked = seed references the blind auditor answered (agreed, or dropped/overruled after a
    disagreement); dropped = currently dropped; unchecked = seed references no auditor has answered yet."""
    out = {}
    for site in baselines.load_sites(evals_dir):
        refs = baselines.load_references(evals_dir, site.id).values()
        rows = baselines.load_dropped(evals_dir, site.id)
        blind_rows = [
            d
            for d in rows
            if d["by"].startswith("blind:")
            or any(v.startswith("blind:") for v in d["reference"]["verified_by"])
        ]
        agreed = sum(1 for r in refs if r.model == SEED and _blind(r))
        out[site.id] = {
            "checked": agreed + len(blind_rows),
            "agreed": agreed,
            "dropped": sum(1 for d in rows if not d.get("restored")),
            "human_verified": sum(1 for r in refs if HUMAN in r.verified_by),
            "unchecked": sum(
                1 for r in refs if r.model == SEED and not _blind(r) and HUMAN not in r.verified_by
            ),
            "auditors": sorted(
                {t for r in refs for t in _blind(r)} | {d["by"] for d in rows if d["by"].startswith("blind:")}
            ),
        }
    (Path(evals_dir) / "audit.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def audit_repo(evals_dir: str | Path, llm: LLM | None = None, site_ids: list[str] | None = None):
    """Audit every prepared site (or `site_ids`), then write audit.json and refresh the manifest."""
    evals_dir = Path(evals_dir)
    for site in baselines.load_sites(evals_dir):
        if (site_ids is None or site.id in site_ids) and site.contract.task_type != "free_text":
            audit_site(evals_dir, site, llm)
    counts = summarize(evals_dir)
    baselines.write_manifest(evals_dir)
    return counts


def import_closed(evals_dir: Path, rows: list[dict[str, Any]], model: str = baselines.CLAUDE_CODE_REFERENCE):
    """Store closed-model answers produced outside this code (GC27): source closed_model, nothing measured."""
    by_site: dict[str, dict[str, BaselineResult]] = {}
    now = datetime.now(UTC)
    for row in rows:
        site_id = row["case_id"].rsplit(":", 1)[0]
        site_rows = by_site.setdefault(site_id, baselines.load_closed(evals_dir, site_id))
        site_rows[row["case_id"]] = BaselineResult(
            case_id=row["case_id"],
            source="closed_model",
            model=model,
            output=ModelOutput.model_validate(row["output"]),
            recorded_at=now,
        )
    for site_id, site_rows in by_site.items():
        baselines.save_closed(evals_dir, site_id, site_rows.values())
    return {k: len(v) for k, v in by_site.items()}


# ---------------------------------------------------------------- blind bundle for the closed auditor (GC27b)


_ANSWER_FORMAT = {
    "classification": '{"text": "<exactly one of the labels>"}',
    "json_extraction": '{"text": "<the JSON object the call replies with; it must match response_schema>"}',
    "tool_call": '{"tool_calls": [{"name": "<tool name>", "arguments": {...}}]} (the one call it makes next)',
    "reasoning": '{"text": "<your working, then a last line ANSWER: <number>>"}',
    "free_text": '{"text": "<the reply the assistant sends>"}',
}

_BLIND_INSTRUCTIONS = """# Blind answers: {repo}

You are the assistant inside each LLM call described by the `<site>.json` files in this folder. For every
case, read its `input_messages` (the system message is that call's own instructions) and give the output
the call should produce. Answer from the case alone: there is nothing else to look up.

Each site file has `task_type`, `answer_format` and, depending on the task, the `labels`, the
`response_schema`, or the `tools` (OpenAI function format) with `tool_choice`.

Write one JSON object per line to `answers.jsonl` in this folder:

    {{"case_id": "<case_id>", "output": <answer_format>}}
"""


def _default_bundle_dir(repo: Path, site_ids: list[str] | None = None) -> Path:
    """One folder per repo, and a separate one per site selection, so a partial bundle never mixes with
    the site files of an earlier full one."""
    from offramp.agents.case_gen import data_dir

    name = f"{repo.name}-{hashlib.sha256(str(repo).encode()).hexdigest()[:8]}"
    return data_dir() / "audit" / (name + "".join(f"-{s}" for s in sorted(site_ids or [])))


def export_blind_audit(
    repo_path: str | Path, out_dir: str | Path | None = None, site_ids: list[str] | None = None
) -> Path:
    """Write what the closed auditor (GC27: a Claude subagent) may see, outside the target repo (default
    OFFRAMP_DATA_DIR/audit/<repo>-<hash>/): per site the case ids and input_messages, the tools in OpenAI
    form, the label set or response JSON schema, and the answer format. Never a reference. Returns the
    folder.
    """
    from offramp.agents.translate import openai_tool, openai_tool_choice

    repo = Path(repo_path).resolve()
    evals = repo / "evals"
    out = Path(out_dir) if out_dir else _default_bundle_dir(repo, site_ids)
    out.mkdir(parents=True, exist_ok=True)
    sites = [s for s in baselines.load_sites(evals) if site_ids is None or s.id in site_ids]
    for site in sites:
        kind = site.contract.task_type
        bundle: dict[str, Any] = {
            "site_id": site.id,
            "task_type": kind,
            "answer_format": _ANSWER_FORMAT[kind],
        }
        if kind == "classification":
            bundle["labels"] = site.contract.labels
        if kind == "json_extraction" and site.contract.json_schema:
            bundle["response_schema"] = site.contract.json_schema
        if kind == "tool_call":
            bundle["tools"] = [t for t in map(openai_tool, site.params.get("tools") or []) if t]
            if site.params.get("tool_choice") is not None:
                bundle["tool_choice"] = openai_tool_choice(site.params["tool_choice"])
        bundle["cases"] = [
            {"case_id": c.id, "input_messages": c.input_messages}
            for c in baselines.load_cases(evals, site.id)
        ]
        (out / f"{site.id}.json").write_text(json.dumps(bundle, indent=1, ensure_ascii=False) + "\n")
    (out / "INSTRUCTIONS.md").write_text(_BLIND_INSTRUCTIONS.format(repo=repo.name))
    (out / "bundle.json").write_text(
        json.dumps(
            # the repo's name only: the auditor's folder must not point at the repo (its evals/ has answers)
            {"repo": repo.name, "sites": [s.id for s in sites], "created_at": datetime.now(UTC).isoformat()},
            indent=1,
        )
        + "\n"
    )
    return out


def ingest_blind_audit(
    answers_file: str | Path,
    repo_path: str | Path,
    model: str = baselines.CLAUDE_CODE_REFERENCE,
) -> dict[str, Any]:
    """Store the closed auditor's answers (labelled `model`, GC27) and audit every seed reference against them
    with the site's own scorer. Answers for case ids the dataset does not have are reported, not stored."""
    answers_file = Path(answers_file)
    evals = Path(repo_path) / "evals"
    known = {c.id for s in baselines.load_sites(evals) for c in baselines.load_cases(evals, s.id)}
    rows = [json.loads(line) for line in answers_file.read_text().splitlines() if line.strip()]
    stored = [r for r in rows if r["case_id"] in known]
    import_closed(evals, stored, model)
    return {
        "stored": len(stored),
        "unknown": [r["case_id"] for r in rows if r["case_id"] not in known],
        "counts": audit_repo(evals),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blind audit of seed references (GC18, GC27).")
    parser.add_argument("repo", type=Path)
    parser.add_argument("--closed", type=Path, help="JSONL of closed-reference answers to store first")
    parser.add_argument(
        "--model", default=baselines.CLAUDE_CODE_REFERENCE, help="model label of --closed rows"
    )
    parser.add_argument(
        "--ultra-fallback",
        action="store_true",
        help="Ultra answers the seed cases no closed reference answered (Token Factory calls)",
    )
    parser.add_argument("--sites", help="comma-separated site ids")
    parser.add_argument(
        "--export-blind", action="store_true", help="write the blind bundle for the closed auditor and exit"
    )
    parser.add_argument(
        "--out", type=Path, help="bundle folder for --export-blind (default under OFFRAMP_DATA_DIR)"
    )
    args = parser.parse_args(argv)
    evals = args.repo / "evals"
    if args.export_blind:
        print(export_blind_audit(args.repo, args.out, args.sites.split(",") if args.sites else None))
        return 0
    if args.closed:
        result = ingest_blind_audit(args.closed, args.repo, args.model)
        if result["unknown"]:
            print(f"ignored answers for unknown case ids: {result['unknown']}", file=sys.stderr)
    llm = None
    if args.ultra_fallback:
        from offramp.tf import Budget, TFClient

        llm = TFClient(budget=Budget())
    counts = audit_repo(evals, llm, args.sites.split(",") if args.sites else None)
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
