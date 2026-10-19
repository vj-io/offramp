"""Human audit of references via Tendem (PLAN GC24). No network: this script only writes and reads files.

    python scripts/audit_references.py export <repo> [--sample 5] [--seed 0]
        -> <repo>/evals/human_audit/task.md: every flagged reference (dropped after the blind audit
           disagreed) plus a random sample of kept ones per site, the instructions each call follows, and
           the answer format.
    python scripts/audit_references.py ingest <repo> <verdicts.csv>
        -> applies the expert's verdicts: correct => verified_by += ["human:tendem"] (a flagged case is
           restored); wrong => the case and its reference are dropped. Refreshes evals/audit.json and the
           manifest.

Posting the task to Tendem and paying for it is a separate, manual step: every paid task needs the user's
explicit approval of its quoted price.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from offramp.agents import reference_audit as ra
from offramp.evals import baselines as b
from offramp.schemas import BaselineResult, EvalCase, ModelOutput

CELL_LIMIT = 4000

INSTRUCTIONS = """You are checking the expected outputs of test cases for AI features in a support product.
Each section below is one feature ("call site"): first the instructions the AI follows, then a table of
cases. For every row, read the input and decide whether the expected output is exactly the correct output
for that input under those instructions. Rows marked **flagged** are ones where an independent AI model
disagreed with the expected output; judge them on the merits, not by the other model's answer.

## Answer format

Return one CSV file with this header and one line per row:

```
case_id,verdict,note
```

`verdict` is `correct` or `wrong`. `note` is optional for `correct`; for `wrong`, write what the correct
output is.
"""


def _cell(text: str) -> str:
    text = text if len(text) <= CELL_LIMIT else text[:CELL_LIMIT] + " …(truncated)"
    return text.replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _shown(output: dict[str, Any] | ModelOutput | None) -> str:
    if output is None:
        return "—"
    out = output if isinstance(output, ModelOutput) else ModelOutput.model_validate(output)
    calls = [f"{c.name}({json.dumps(c.arguments, ensure_ascii=False)})" for c in out.tool_calls]
    return "; ".join(calls) or out.text or "(empty)"


def _input(case: EvalCase) -> str:
    return "\n\n".join(m["content"] for m in case.input_messages if m["role"] != "system")


def export(repo: Path, sample: int = 5, seed: int = 0) -> Path:
    evals, rng = Path(repo) / "evals", random.Random(seed)
    sections, total = [], 0
    for site in b.load_sites(evals):
        if site.contract.task_type == "free_text":
            continue
        cases = {c.id: c for c in b.load_cases(evals, site.id)}
        refs = b.load_references(evals, site.id)
        flagged = [d for d in b.load_dropped(evals, site.id) if not d.get("restored")]
        sampled = rng.sample(sorted(refs), min(sample, len(refs)))
        rows = [
            (
                d["case"]["id"],
                f"flagged: {d['by']} said {_shown(d['auditor_output'])}",
                _input(EvalCase.model_validate(d["case"])),
                _shown(d["reference"]["output"]),
            )
            for d in flagged
        ]
        rows += [(cid, "sample", _input(cases[cid]), _shown(refs[cid].output)) for cid in sorted(sampled)]
        if not rows:
            continue
        total += len(rows)
        table = ["| case_id | status | input | expected output |", "|---|---|---|---|"]
        table += [
            f"| {cid} | {_cell(status)} | {_cell(text)} | {_cell(expected)} |"
            for cid, status, text, expected in rows
        ]
        system = (site.template.system or "(none)").replace("\n", "\n> ")
        sections.append(
            f"## {site.id}: {site.contract.task_type} ({site.file})\n\n"
            f"Instructions the AI follows:\n\n> {system}\n\n" + "\n".join(table)
        )
    path = evals / "human_audit" / "task.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Reference audit: {total} cases\n\n{INSTRUCTIONS}\n" + "\n\n".join(sections) + "\n")
    return path


def ingest(repo: Path, verdicts_path: Path) -> dict[str, Any]:
    evals = Path(repo) / "evals"
    by_site: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with open(verdicts_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = (row.get("case_id") or "").strip()
            by_site[cid.rsplit(":", 1)[0]].append(
                (cid, (row.get("verdict") or "").strip().lower(), (row.get("note") or "").strip())
            )
    applied, unknown, invalid = 0, [], []
    for site_id, items in by_site.items():
        cases = {c.id: c for c in b.load_cases(evals, site_id)}
        refs = b.load_references(evals, site_id)
        dropped = b.load_dropped(evals, site_id)
        flagged = {d["case"]["id"]: d for d in dropped if not d.get("restored")}
        for cid, verdict, note in items:
            if verdict not in ("correct", "wrong"):
                invalid.append(cid)
            elif cid in refs and cid in cases:
                ref = refs[cid]
                if verdict == "correct":
                    refs[cid] = ref.model_copy(update={"verified_by": [*ref.verified_by, ra.HUMAN]})
                else:
                    dropped.append(
                        ra.dropped_row(cases.pop(cid), refs.pop(cid), ra.HUMAN, f"{ra.HUMAN}: wrong — {note}")
                    )
            elif cid in flagged:
                row = flagged[cid]
                if verdict == "correct":
                    row["restored"] = True
                    ref = BaselineResult.model_validate(row["reference"])
                    refs[cid] = ref.model_copy(update={"verified_by": [*ref.verified_by, ra.HUMAN]})
                    cases[cid] = EvalCase.model_validate(row["case"])
                else:
                    row["reason"] += f"; {ra.HUMAN} confirmed wrong — {note}"
            else:
                unknown.append(cid)
                continue
            applied += verdict in ("correct", "wrong")
        b.save_cases(evals, site_id, cases.values())
        b.save_references(evals, site_id, refs.values())
        b.save_dropped(evals, site_id, dropped)
    ra.summarize(evals)
    b.write_manifest(evals)
    return {"applied": applied, "unknown": unknown, "invalid": invalid}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tendem human audit of references (no network).")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("repo", type=Path)
    exp.add_argument("--sample", type=int, default=5)
    exp.add_argument("--seed", type=int, default=0)
    ing = sub.add_parser("ingest")
    ing.add_argument("repo", type=Path)
    ing.add_argument("verdicts", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        print(export(args.repo, args.sample, args.seed))
        return 0
    result = ingest(args.repo, args.verdicts)
    print(json.dumps(result))
    return 1 if result["unknown"] or result["invalid"] else 0


if __name__ == "__main__":
    sys.exit(main())
