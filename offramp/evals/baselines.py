"""Eval dataset I/O (PLAN Task 7, Amendment A): cases, references, closed-model outputs, manifest.

Layout under a target repo's `evals/` (every `.jsonl` row is one pydantic model dumped as JSON):

    cases/<site>.jsonl               EvalCase
    references/<site>.jsonl          BaselineResult(source="reference"): reference:seed|oracle|fixture|human
    references/<site>.dropped.jsonl  references removed by the audit or a human, with the reason
    closed/<site>.jsonl              BaselineResult(source="closed_model"): scripts/record_baselines.py (teams
                                     with keys) or a Claude Code subagent (GC27, model CLAUDE_CODE_REFERENCE)
    baselines/<site>.jsonl           legacy closed-model baselines (PRD §6.3), read only
    sites.json                       the CallSites the cases were generated for (contracts, templates)
    reference/*.py                   deterministic oracles (reference:oracle)
    audit.json                       per-site reference audit counts (offramp.agents.reference_audit)
    manifest.json                    sha256 per file above, reference method counts, generator model ids
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from offramp.schemas import BaselineResult, CallSite, EvalCase

# GC27: the only closed reference this build can produce; always stored under exactly this model string.
CLAUDE_CODE_REFERENCE = "claude-sonnet-5.5 (via Claude Code; not the repo's source model)"

MANIFEST = "manifest.json"
_UNHASHED_DIRS = {"results", "human_audit"}  # run outputs and human-task drafts change after preparation
_HASHED_SUFFIXES = {".jsonl", ".json", ".py"}


def _read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    if not path.exists():
        return []
    return [model.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    _write_lines(path, (r.model_dump_json() for r in rows))


def load_cases(evals_dir: Path, site_id: str) -> list[EvalCase]:
    return _read_jsonl(Path(evals_dir) / "cases" / f"{site_id}.jsonl", EvalCase)


def input_key(messages: list[dict[str, Any]]) -> str:
    """A case's rendered input with case and whitespace normalised: two cases with one key are one input."""
    parts = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, sort_keys=True, ensure_ascii=False)
        parts.append([m.get("role"), " ".join(content.split()).casefold()])
    return json.dumps(parts, ensure_ascii=False)


def _leaks(cases: Iterable[EvalCase]) -> str | None:
    """'the same input is in dev and holdout (<ids>)' when one is: the Iterator would tune on inputs the
    holdout verdict then reads (GC10)."""
    by_input: dict[str, dict[str, str]] = {}  # input -> split -> first case id
    for c in sorted(cases, key=lambda c: c.id):
        by_input.setdefault(input_key(c.input_messages), {}).setdefault(c.split, c.id)
    leaked = [" and ".join(sorted(ids.values())) for ids in by_input.values() if len(ids) > 1]
    return f"the same input is in dev and holdout ({'; '.join(leaked[:5])})" if leaked else None


def save_cases(evals_dir: Path, site_id: str, cases: Iterable[EvalCase]) -> None:
    """Refuses (ValueError, nothing written) cases whose input is in both splits."""
    cases = sorted(cases, key=lambda c: c.id)
    if leak := _leaks(cases):
        raise ValueError(f"{site_id}: {leak}: drop the duplicates")
    _write_jsonl(Path(evals_dir) / "cases" / f"{site_id}.jsonl", cases)


def load_references(evals_dir: Path, site_id: str) -> dict[str, BaselineResult]:
    """References by case id; rows in legacy `baselines/` are read first, so `references/` wins."""
    rows = _read_jsonl(Path(evals_dir) / "baselines" / f"{site_id}.jsonl", BaselineResult)
    rows += _read_jsonl(Path(evals_dir) / "references" / f"{site_id}.jsonl", BaselineResult)
    return {r.case_id: r for r in rows}


def save_references(evals_dir: Path, site_id: str, refs: Iterable[BaselineResult]) -> None:
    _write_jsonl(Path(evals_dir) / "references" / f"{site_id}.jsonl", sorted(refs, key=lambda r: r.case_id))


def load_closed(evals_dir: Path, site_id: str) -> dict[str, BaselineResult]:
    return {
        r.case_id: r for r in _read_jsonl(Path(evals_dir) / "closed" / f"{site_id}.jsonl", BaselineResult)
    }


def save_closed(evals_dir: Path, site_id: str, rows: Iterable[BaselineResult]) -> None:
    _write_jsonl(Path(evals_dir) / "closed" / f"{site_id}.jsonl", sorted(rows, key=lambda r: r.case_id))


def baselines_for(evals_dir: Path, site_id: str) -> dict[str, BaselineResult]:
    """What candidates are compared against: the reference when a case has one, else the closed-model output.

    Free-text sites have no reference rows (GC20), so their closed outputs (GC27a) fill in. Each row keeps its
    `source` and `model`, so a consumer can always say which kind of baseline it scored against.
    """
    return {**load_closed(evals_dir, site_id), **load_references(evals_dir, site_id)}


def load_dropped(evals_dir: Path, site_id: str) -> list[dict[str, Any]]:
    path = Path(evals_dir) / "references" / f"{site_id}.dropped.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_dropped(evals_dir: Path, site_id: str, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(evals_dir) / "references" / f"{site_id}.dropped.jsonl"
    _write_lines(path, (json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows))


def load_sites(evals_dir: Path) -> list[CallSite]:
    path = Path(evals_dir) / "sites.json"
    return [CallSite.model_validate(s) for s in json.loads(path.read_text())] if path.exists() else []


def save_sites(evals_dir: Path, sites: Iterable[CallSite]) -> None:
    """Merge by id with the sites already stored, so preparing one site never forgets the others."""
    merged = {s.id: s for s in load_sites(evals_dir)} | {s.id: s for s in sites}
    path = Path(evals_dir) / "sites.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [merged[k].model_dump(mode="json") for k in sorted(merged)]
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- manifest


def _hashed_files(evals_dir: Path) -> list[str]:
    out = []
    for path in sorted(evals_dir.rglob("*")):
        rel = path.relative_to(evals_dir)
        if not path.is_file() or rel.as_posix() == MANIFEST or path.suffix not in _HASHED_SUFFIXES:
            continue
        if rel.parts[0] in _UNHASHED_DIRS or any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
            continue
        out.append(rel.as_posix())
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _site_ids(evals_dir: Path) -> list[str]:
    stems = {
        p.name.split(".")[0]
        for d in ("references", "closed", "baselines")
        for p in (evals_dir / d).glob("*.jsonl")
    }
    return sorted(stems)


def write_manifest(evals_dir: Path, generator_models: dict[str, str] | None = None) -> dict[str, Any]:
    """Recompute hashes and method counts; generator model ids merge with those already recorded."""
    evals_dir = Path(evals_dir)
    old = json.loads((evals_dir / MANIFEST).read_text()) if (evals_dir / MANIFEST).exists() else {}
    methods = {}
    for site_id in _site_ids(evals_dir):
        counts = Counter(r.model for r in load_references(evals_dir, site_id).values())
        counts.update("closed_model" for _ in load_closed(evals_dir, site_id))
        methods[site_id] = dict(counts)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generator_models": {**old.get("generator_models", {}), **(generator_models or {})},
        "methods": methods,
        "files": {rel: _sha256(evals_dir / rel) for rel in _hashed_files(evals_dir)},
    }
    (evals_dir / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def verify_manifest(evals_dir: Path) -> list[str]:
    """Problems found (empty = the dataset is exactly what the manifest records, and no input of a site is in
    both of its splits)."""
    evals_dir = Path(evals_dir)
    if not (evals_dir / MANIFEST).exists():
        return [f"no {MANIFEST} in {evals_dir}"]
    recorded = json.loads((evals_dir / MANIFEST).read_text())["files"]
    present = set(_hashed_files(evals_dir))
    problems = [f"{rel}: missing" for rel in sorted(set(recorded) - present)]
    problems += [f"{rel}: not in manifest" for rel in sorted(present - set(recorded))]
    problems += [
        f"{rel}: hash mismatch"
        for rel in sorted(present & set(recorded))
        if _sha256(evals_dir / rel) != recorded[rel]
    ]
    for path in sorted((evals_dir / "cases").glob("*.jsonl")):  # written before save_cases refused leaks
        if leak := _leaks(_read_jsonl(path, EvalCase)):
            problems.append(f"cases/{path.name}: {leak}")
    return problems
