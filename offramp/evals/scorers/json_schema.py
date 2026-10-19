"""json_extraction scorer (PRD §6.5): schema validity plus field match against the baseline.

A case passes when the candidate parses, validates, and at least FIELD_MATCH_MIN of the compared
fields agree. "Compared fields" is the union of baseline and candidate leaf paths, so a missing key,
a wrong value and an invented key each cost a field (key-set + value equality). `CaseScore.score` is
that field match. The site rule (100% schema-valid, fraction of passing cases, mean field match, all
against the configurable threshold) lives in aggregate.site_pass.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import jsonschema

from offramp.schemas import CaseScore, Contract, EvalCase, ModelOutput

FIELD_MATCH_MIN = 0.95  # per-case bar, Global Constraint 11

INVALID_JSON = "invalid_json"
EMPTY_OUTPUT = "empty_output"
SCHEMA_VIOLATION = "schema_violation"  # candidate does not conform to the schema
SCHEMA_ERROR = "schema_error"  # the schema itself is unusable, so validity cannot be established
BASELINE_UNUSABLE = "baseline_unusable"
MISSING_KEY, WRONG_VALUE, EXTRA_KEY = "missing_key", "wrong_value", "extra_key"
# Categories meaning "the candidate did not produce schema-valid JSON" (site rule: 100% valid).
INVALID_CATEGORIES = frozenset({INVALID_JSON, EMPTY_OUTPUT, SCHEMA_VIOLATION, SCHEMA_ERROR})

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
# No leading zeros, no exponent: the string "01234" is not the number 1234.
_PLAIN_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_SHOWN = 5  # paths listed per category in `detail`


def normalise(value: Any, *, numeric_strings: bool = False) -> Any:
    """Canonical form for equality: strings trimmed + casefolded, numbers exact Decimals.

    Booleans are wrapped so `True` never equals `1`. With numeric_strings, a string that is a plain
    number ("5", "5.0") equals the number (tool-call arguments); JSON field values stay type-strict.
    """
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return Decimal("NaN")  # 1e400 parses to inf: never equal to anything, itself included
        return Decimal(repr(value))  # shortest round-trip text, so 5.1 == "5.10" without binary noise
    if isinstance(value, str):
        text = value.strip()
        if numeric_strings and _PLAIN_NUMBER.fullmatch(text):
            return Decimal(text)
        return text.casefold()
    if isinstance(value, dict):
        return {k: normalise(v, numeric_strings=numeric_strings) for k, v in value.items()}
    if isinstance(value, list):
        return [normalise(v, numeric_strings=numeric_strings) for v in value]
    return value  # None


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _parse(text: str) -> Any:
    """json.loads that tolerates one surrounding ```json fence and rejects NaN/Infinity."""
    text = text.lstrip("﻿").strip()
    fenced = _FENCE.fullmatch(text)
    return json.loads(fenced.group(1) if fenced else text, parse_constant=_reject_constant)


def _infer_schema(value: Any) -> dict[str, Any]:
    """Structure of the baseline: types everywhere, every baseline key required, extras allowed."""
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int | float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if value is None:
        return {"type": "null"}
    if isinstance(value, list):
        items = []
        for item in value:
            shape = _infer_schema(item)
            if shape not in items:
                items.append(shape)
        if not items:
            return {"type": "array"}
        return {"type": "array", "items": items[0] if len(items) == 1 else {"anyOf": items}}
    return {
        "type": "object",
        "properties": {k: _infer_schema(v) for k, v in value.items()},
        "required": list(value),
    }


def _leaves(value: Any, path: tuple = ()) -> Iterator[tuple[tuple, Any]]:
    """(path, value) for every leaf; an empty object/array is itself a leaf so it counts as a key."""
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    else:
        yield path, value


def _show(paths: list[tuple]) -> str:
    def one(path: tuple) -> str:
        return "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path).lstrip(".") or "$"

    return ", ".join(map(one, paths[:_SHOWN])) + (" ..." if len(paths) > _SHOWN else "")


def _fail(case: EvalCase, category: str, detail: str) -> CaseScore:
    return CaseScore(case_id=case.id, passed=False, score=0.0, detail=detail, error_category=category)


def score(case: EvalCase, baseline: ModelOutput, candidate: ModelOutput, contract: Contract) -> CaseScore:
    try:
        return _score(case, baseline, candidate, contract)
    except RecursionError:  # absurdly nested JSON: unscorable, never a pass
        return _fail(case, INVALID_JSON, "JSON nesting too deep to score")


def _score(case: EvalCase, baseline: ModelOutput, candidate: ModelOutput, contract: Contract) -> CaseScore:
    if candidate.text is None or not candidate.text.strip():
        return _fail(case, EMPTY_OUTPUT, "candidate returned no text")
    try:
        got = _parse(candidate.text)
    except ValueError as exc:
        return _fail(case, INVALID_JSON, f"candidate is not JSON: {exc}")
    try:
        want = _parse(baseline.text or "")
    except ValueError as exc:
        return _fail(case, BASELINE_UNUSABLE, f"baseline is not JSON: {exc}")

    schema = contract.json_schema if contract.json_schema is not None else _infer_schema(want)
    try:
        jsonschema.validate(got, schema)
    except jsonschema.ValidationError as exc:
        where = _show([tuple(exc.absolute_path)])
        return _fail(case, SCHEMA_VIOLATION, f"schema violation at {where}: {exc.message[:200]}")
    except Exception as exc:  # SchemaError, unresolvable $ref: validity cannot be established
        return _fail(case, SCHEMA_ERROR, f"schema unusable: {type(exc).__name__}: {str(exc)[:200]}")

    base_leaves, cand_leaves = dict(_leaves(normalise(want))), dict(_leaves(normalise(got)))
    missing = [p for p in base_leaves if p not in cand_leaves]
    wrong = [p for p in base_leaves if p in cand_leaves and base_leaves[p] != cand_leaves[p]]
    extra = [p for p in cand_leaves if p not in base_leaves]
    compared = len(base_leaves.keys() | cand_leaves.keys())
    field_match = (len(base_leaves) - len(missing) - len(wrong)) / compared
    passed = field_match >= FIELD_MATCH_MIN
    if field_match == 1.0:
        return CaseScore(case_id=case.id, passed=True, score=1.0)
    detail = f"field_match={field_match:.3f}; " + "; ".join(
        f"{kind}: {_show(paths)}"
        for kind, paths in (("missing", missing), ("wrong", wrong), ("extra", extra))
        if paths
    )
    category = None if passed else MISSING_KEY if missing else WRONG_VALUE if wrong else EXTRA_KEY
    return CaseScore(
        case_id=case.id, passed=passed, score=field_match, detail=detail, error_category=category
    )
