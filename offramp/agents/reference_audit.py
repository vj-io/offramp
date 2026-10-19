"""Reference audit (PLAN GC18, GC27b): every `reference:seed` is answered blind by an independent model and
scored against the seed with the site's own scorer. A disagreement drops the case and its reference; both
are kept, with the reason, in references/<site>.dropped.jsonl, which is also the human-audit queue
(scripts/audit_references.py).

Auditor per case: the closed reference (GC27: a Claude Sonnet subagent answers the exact input_messages
outside this code; its rows in closed/<site>.jsonl are loaded here), so Nemotron never grades gold that
Nemotron wrote. Only when an `llm` is passed does Ultra (role `reference_audit`, thinking on, batched) answer
the cases the closed reference has not. Oracle and fixture references are not blind-audited: their answers
come from code. Per-site counts go to evals/audit.json. Re-running never re-asks an audited reference.

    python -m offramp.agents.reference_audit <repo> [--closed answers.jsonl] [--ultra-fallback]

answers.jsonl rows: {"case_id": "cs_01:004", "output": {"text": ...} | {"tool_calls": [...]}}.
"""

from __future__ import annotations

import argparse
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
    todo = [c for c in cases.values() if c.id in refs and refs[c.id].model == SEED and not _blind(refs[c.id])]
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
    args = parser.parse_args(argv)
    evals = args.repo / "evals"
    if args.closed:
        rows = [json.loads(line) for line in args.closed.read_text().splitlines() if line.strip()]
        import_closed(evals, rows, args.model)
    llm = None
    if args.ultra_fallback:
        from offramp.tf import Budget, TFClient

        llm = TFClient(budget=Budget())
    counts = audit_repo(evals, llm, args.sites.split(",") if args.sites else None)
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
