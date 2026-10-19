"""Case Generator (PRD §6.2; PLAN Task 7, Amendment A GC17): eval cases whose expected outputs are known by
construction, never recorded from a model.

Per call site, in this order:
  1. Repo fixtures (tests/fixtures/) that carry an expected output -> `reference:oracle` when the repo has an
     oracle for the site (and the fixture's value must agree with it), else `reference:fixture`. Free-text
     sites take every fixture input and get no reference (the judge scores against the evidence, GC20).
  2. Synthetic, seed-first: code picks the expected output first (a label balanced across contract.labels,
     a tool call, an extraction record from a seeded RNG, or oracle parameters whose answer the oracle
     computes), then Super (role `case_gen`, thinking off) writes a realistic input implying it.
  3. Adversarial: the same, written by Ultra (role `case_gen_adversarial`) to be hard while the expected
     output stays the correct one. Oracle sites use the oracle's hard parameters instead.
Every `reference:seed` is then checked blind by offramp.agents.reference_audit.

Inputs are fixture-shaped records. `Recipe`s turn a record into the template's variables the way the call
site's own code does (placeholder names come from expressions, e.g. `request_plan_name`, `len_messages`,
so they are mapped explicitly: by exact/suffix/len_ names when unambiguous, else by one Super call that is
checked by rendering every fixture). Synthetic records render through the same recipes as real fixtures.

Target-repo code (Pydantic schemas, oracles) runs in the target's own environment, in a subprocess.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel

from offramp.evals import baselines
from offramp.evals.scorers import score_case
from offramp.evals.scorers.json_schema import _infer_schema
from offramp.schemas import (
    BaselineResult,
    CallSite,
    EvalCase,
    ModelOutput,
    ScanResult,
    Split,
    Tier,
    ToolCall,
)
from offramp.tf.client import LLM

ROLE, ROLE_ADVERSARIAL = "case_gen", "case_gen_adversarial"
SEED, ORACLE, FIXTURE = "reference:seed", "reference:oracle", "reference:fixture"
DEFAULT_TARGET, DEFAULT_ADVERSARIAL = 24, 4  # PRD §6.2: 20-30 cases per site, 3-5 of them adversarial
BATCH = 8  # records written per LLM call
RUN_TIMEOUT_S = 60
NEEDS_RUNTIME_CAPTURE = "NEEDS_RUNTIME_CAPTURE"

EXPECTED_KIND = {
    "classification": "label",
    "json_extraction": "json",
    "tool_call": "tool_call",
    "free_text": "text",
    "reasoning": "number",
}
EXPECTED_KEYS = (
    "expected_output",
    "expected_answer",
    "expected_label",
    "expected",
)  # fixture expected outputs
STYLES = {  # diversity axes of PRD §6.2; one per written record, cycled
    "length:short": "very short: one line, the way people type on a phone",
    "length:long": "long (150-300 words), with background details that do not change the answer",
    "locale:de": "written in German",
    "locale:es": "written in Spanish",
    "locale:ja": "written in Japanese",
    "malformed": "messy: typos, broken line breaks, missing punctuation, stray pasted text",
    "empty": "as terse as possible: the bare minimum of text that still determines the answer",
    "unicode": "full of emoji, accented names, non-Latin script and special symbols",
}

MARKER = re.compile(r"<pydantic:([A-Za-z_][\w.]*)>")


def split_for(case_id: str) -> Split:
    """Global Constraint 10: deterministic 70/30 dev/holdout split by hash of the case id."""
    return "holdout" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100 < 30 else "dev"


# ---------------------------------------------------------------- running target-repo code


def _run_in_repo(repo: Path, code: str, *args: str, stdin: str | None = None) -> str:
    """Run `python -c code` in the target repo's own environment (uv project) or, without a pyproject, the
    current interpreter; cwd is the repo, so its packages import. Raises on failure or timeout."""
    python = (
        ["uv", "run", "--quiet", "--project", str(repo), "python"]
        if (repo / "pyproject.toml").exists()
        else [sys.executable]
    )
    proc = subprocess.run(
        [*python, "-B", "-c", code, *args],
        cwd=repo,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        check=True,
    )
    return proc.stdout


_SCHEMA_CODE = """
import importlib, json, sys
module = importlib.import_module(sys.argv[1])
out = {}
for name in sys.argv[2:]:
    try:
        out[name] = getattr(module, name).model_json_schema()
    except Exception:
        pass
print(json.dumps(out))
"""


def _module_name(file: str) -> str:
    parts = list(PurePosixPath(file).with_suffix("").parts)
    if parts[0] == "src":
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _marker_names(site: CallSite) -> list[str]:
    texts = [site.template.system or "", site.template.user] if site.template else []
    return sorted({name for text in texts for name in MARKER.findall(text)})


def resolved_schemas(site: CallSite, repo_path: str | Path) -> dict[str, dict[str, Any]]:
    """{ClassName: model_json_schema()} for every `<pydantic:ClassName>` marker in the site's template.

    Imports the site's module in the target repo's environment (subprocess, 60 s timeout). Names that cannot
    be resolved are absent from the result; it never raises. Also the input the Adapter Agent takes (Task 6).
    """
    names = _marker_names(site)
    if not names:
        return {}
    try:
        out = _run_in_repo(Path(repo_path), _SCHEMA_CODE, _module_name(site.file), *names)
        schemas = json.loads(out.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}
    return {k: v for k, v in schemas.items() if isinstance(v, dict)}


def resolve_pydantic_markers(site: CallSite, repo_path: str | Path) -> CallSite:
    """Copy of `site` with each `<pydantic:ClassName>` marker replaced by `json.dumps(schema)` (braces escaped
    for str.format templates), as the call site's own `json.dumps(Model.model_json_schema())` renders it.

    A json_extraction contract without a schema gets the (single) resolved one. Any unresolved marker stays in
    place and the site is flagged NEEDS_RUNTIME_CAPTURE.
    """
    names = _marker_names(site)
    if not names:
        return site
    schemas = resolved_schemas(site, repo_path)

    def sub(text: str | None) -> str | None:
        if text is None:
            return None
        return MARKER.sub(
            lambda m: (
                json.dumps(schemas[m[1]]).replace("{", "{{").replace("}", "}}") if m[1] in schemas else m[0]
            ),
            text,
        )

    update: dict[str, Any] = {
        "template": site.template.model_copy(
            update={"system": sub(site.template.system), "user": sub(site.template.user)}
        ),
    }
    if set(names) - set(schemas):
        update["flags"] = (
            [*site.flags, NEEDS_RUNTIME_CAPTURE]
            if NEEDS_RUNTIME_CAPTURE not in site.flags
            else list(site.flags)
        )
    contract = site.contract
    if (
        contract
        and contract.task_type == "json_extraction"
        and contract.json_schema is None
        and len(names) == 1
        and names[0] in schemas
    ):
        update["contract"] = contract.model_copy(update={"json_schema": schemas[names[0]]})
    return site.model_copy(update=update)


# ---------------------------------------------------------------- fixtures, recipes, rendering


def discover_fixtures(repo_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Fixture sets under tests/fixtures/: a JSON list of objects is one set (key: its relative path); a
    directory of text files (.eml/.txt/.md) is one set (key: "dir/") of records {"id": stem, "text": ...}."""
    root = Path(repo_path) / "tests" / "fixtures"
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("*")) if root.is_dir() else []:
        if not path.is_file():
            continue
        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
                out[path.relative_to(root).as_posix()] = data
        elif path.suffix in {".eml", ".txt", ".md"}:
            key = f"{path.parent.relative_to(root).as_posix()}/"
            out.setdefault(key, []).append(
                {"id": path.stem, "text": path.read_text(encoding="utf-8", errors="replace")}
            )
    return out


class Recipe(BaseModel):
    """How one template variable is computed from a record, mirroring the call site's own code."""

    var: str
    field: str  # top-level key of the record
    op: Literal["value", "len", "join"] = "value"  # str(value) | str(len(value)) | sep.join(item per element)
    item: str = ""  # join: str.format template over each element's keys, e.g. "[{id}] {title}\n{text}"
    sep: str = "\n"  # join: separator
    empty: str = ""  # join: value for an empty list (the site's `... or "(none)"`)


class _MappingReply(BaseModel):
    fixture: str | None  # fixture set key, None when no set fits the site
    recipes: list[Recipe]


@dataclass
class FixtureMapping:
    fixture: str | None
    recipes: list[Recipe]
    note: str | None = None


def _formatted(value: Any, spec: str | None) -> str:
    """str(value), or the f-string `{value<spec>}` when the scanner recorded a spec ("!r:.2f", ".2f").

    Fixtures store numbers the app parses (Decimal) as strings, so a numeric spec retries on Decimal(value).
    """
    if not spec:
        return value if isinstance(value, str) else str(value)
    template = "{0" + (spec if spec[0] in "!:" else f":{spec}") + "}"
    try:
        number = Decimal(value) if isinstance(value, str) else value
    except InvalidOperation:
        number = value
    for candidate in (value, number):
        try:
            return template.format(candidate)
        except (ValueError, TypeError):
            continue
    return str(value)


def _apply(recipe: Recipe, record: dict[str, Any], spec: str | None) -> str:
    value = record[recipe.field]
    if recipe.op == "len":
        return str(len(value))
    if recipe.op == "join":
        parts = [
            recipe.item.format_map(v) if isinstance(v, dict) else recipe.item.format(v) for v in value or []
        ]
        return recipe.sep.join(parts) or recipe.empty
    return _formatted(value, spec)


def render(
    site: CallSite, recipes: list[Recipe], record: dict[str, Any]
) -> tuple[dict[str, str], list[dict]]:
    """(variables, input_messages) for one record; raises KeyError/TypeError/ValueError if it cannot."""
    specs = site.params.get("template_format_specs") or {}
    variables = {r.var: _apply(r, record, specs.get(r.var)) for r in recipes}
    messages = []
    if site.template.system:
        messages.append({"role": "system", "content": site.template.system.format_map(variables)})
    messages.append({"role": "user", "content": site.template.user.format_map(variables)})
    return variables, messages


def _unescape(text: str) -> str:
    """Python string-literal escapes (\\n, \\t, \\u00e9) to characters; other text, real newlines included,
    is unchanged. Super copies literals from source code, so it writes the two characters \\n for a newline.
    """
    if "\\" not in text:
        return text
    try:
        return text.encode("latin-1", "backslashreplace").decode("unicode_escape")
    except UnicodeDecodeError:
        return text


def _is_expected_key(key: str) -> bool:
    return key.startswith("expected")


def _name_recipe(var: str, keys: set[str]) -> Recipe | None:
    if var in keys:
        return Recipe(var=var, field=var)
    if var.startswith("len_") and var[4:] in keys:
        return Recipe(var=var, field=var[4:], op="len")
    suffixes = [k for k in keys if var.endswith(f"_{k}")]
    return Recipe(var=var, field=max(suffixes, key=len)) if suffixes else None


def _name_matches(site: CallSite, fixtures: dict[str, list[dict]]) -> dict[str, list[Recipe | None]]:
    """Per fixture set, the recipe each placeholder gets by name alone (None where no key name fits)."""
    out = {}
    for key, records in fixtures.items():
        names = {k for k in records[0] if k != "tags" and not _is_expected_key(k)}
        out[key] = [_name_recipe(v, names) for v in site.template.variables]
    return out


_ELEMENT = re.compile(r"""\{\w+(?:\.(\w+)|\[['"]?(\w+)['"]?\])""")


def _element_keys(item: str) -> str:
    """`{m.sent_at}` / `{m['sent_at']}` (copied from the f-string) -> `{sent_at}`, the element's key."""
    return _ELEMENT.sub(lambda m: "{" + (m[1] or m[2]), item)


def _identity(site: CallSite, note: str | None) -> FixtureMapping:
    return FixtureMapping(None, [Recipe(var=v, field=v) for v in site.template.variables], note)


def _mapping_problem(reply: _MappingReply, site: CallSite, fixtures: dict[str, list[dict]]) -> str | None:
    if reply.fixture not in fixtures:
        return f"unknown fixture {reply.fixture!r}"
    got, want = [r.var for r in reply.recipes], site.template.variables
    if sorted(got) != sorted(want):
        return f"recipes cover {got}, the template needs {want} (missing variable or duplicate)"
    leaked = [r.field for r in reply.recipes if _is_expected_key(r.field)]
    if leaked:
        return f"maps the expected output field {leaked[0]!r} into the prompt"
    for record in fixtures[reply.fixture]:
        try:
            render(site, reply.recipes, record)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            return f"recipes do not render record {record.get('id')!r}: {type(e).__name__}: {e}"
    return None


def _source_text(repo: Path, site: CallSite, limit: int = 12000) -> str:
    try:
        return (repo / site.file).read_text(encoding="utf-8")[:limit]
    except OSError:
        return "(source not readable)"


def _clip(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " ...(truncated)"


_MAPPING_SYSTEM = (
    "You map test-fixture records to the placeholders of an LLM prompt template, reproducing exactly what "
    "the application code does before it formats the prompt. Reply with JSON only."
)


def _mapping_prompt(site: CallSite, fixtures: dict[str, list[dict]], source: str) -> str:
    catalog = "\n\n".join(
        f"### {key} ({len(records)} records)\nFirst record: {_clip(records[0])}"
        for key, records in fixtures.items()
    )
    return f"""## Application code ({site.file}; the LLM call is in `{site.function}`, lines {site.lines})
```python
{source}
```

## Prompt template (str.format placeholders)
System: {_clip(site.template.system, 1500)}
User: {_clip(site.template.user, 3000)}
Placeholders: {json.dumps(site.template.variables)}

## Fixture sets in tests/fixtures/
{catalog}

## Task
Pick the one fixture set whose records are the inputs of `{site.function}` (null if none is), and for EVERY
placeholder give a recipe that computes its value from one record, exactly as the code above does:
- op "value": str(record[field])
- op "len": str(len(record[field]))
- op "join": sep.join(item.format(**element) for element in record[field]), or `empty` when the list is empty
  (mirror the code's join separator, per-element f-string and any `or "..."` fallback).
`item`, `sep` and `empty` are the contents of the Python string literals in the code, escapes included (write
the two characters \\n where the code has \\n). In `item` the placeholders are the element's keys: write
{{sent_at}}, not {{m.sent_at}} or {{m['sent_at']}}.
`field` is a top-level key of the record; never use a key whose name starts with "expected" (those are
expected outputs, not inputs).
Reply {{"fixture": ..., "recipes": [{{"var", "field", "op", "item", "sep", "empty"}}]}}."""


def map_fixtures(
    site: CallSite, fixtures: dict[str, list[dict]], llm: LLM, repo_path: str | Path
) -> FixtureMapping:
    """Which fixture set feeds `site`, and how each record becomes the template's variables.

    Unambiguous names (exact, `<prefix>_<key>`, `len_<key>`) map without an LLM; otherwise one Super call
    (shown the sets whose keys name the most placeholders) proposes a mapping, accepted only if every fixture
    record renders with it. A rejected or empty answer leaves the site without fixtures (identity recipes: a
    written record holds the variables themselves).
    """
    if not fixtures or not site.template.variables:
        return _identity(site, None)
    matches = _name_matches(site, fixtures)
    full = [key for key, recipes in matches.items() if all(recipes)]
    if len(full) == 1:
        return FixtureMapping(full[0], matches[full[0]])
    # Show Super only the sets whose keys best name the placeholders (exact/len_ names count double a suffix
    # match, so `ticket_text` prefers a `ticket_text` key over a `text` key); all sets when none does.
    weight = {
        k: sum(2 if r.field == r.var or r.op == "len" else 1 for r in recipes if r)
        for k, recipes in matches.items()
    }
    best = max(weight.values())
    shown = {k: fixtures[k] for k in fixtures if weight[k] == best} if best else fixtures
    result = llm.chat(
        role=ROLE,
        tier=Tier.SUPER,
        messages=[
            {"role": "system", "content": _MAPPING_SYSTEM},
            {"role": "user", "content": _mapping_prompt(site, shown, _source_text(Path(repo_path), site))},
        ],
        max_tokens=2000,
        temperature=0.0,
        response_model=_MappingReply,
        reasoning="off",
    )
    reply = result.parsed
    if reply is None:
        return _identity(site, "fixture mapping rejected: Super's reply was not a valid mapping")
    if reply.fixture is None:
        return _identity(site, "no fixture set matches this site (Super)")
    reply.recipes = [
        r.model_copy(
            update={
                "item": _element_keys(_unescape(r.item)),
                "sep": _unescape(r.sep),
                "empty": _unescape(r.empty),
            }
        )
        for r in reply.recipes
    ]
    problem = _mapping_problem(reply, site, fixtures)
    if problem:
        return _identity(site, f"fixture mapping rejected: {problem}")
    return FixtureMapping(reply.fixture, reply.recipes)


# ---------------------------------------------------------------- seeds: the expected output, picked first


@dataclass
class Seed:
    output: ModelOutput
    shown: str  # the expected output as the writer is told it
    tags: list[str] = field(default_factory=list)


NULL_RATE = 0.25
NAMES = [
    "Priya Nair",
    "Jürgen Müller",
    "María José García",
    "田中 花子",
    "Chloé Dubois",
    "Kwame Mensah",
    "Olga Ivanova",
    "Liam O'Connor",
    "Fatima Zahra El Idrissi",
    "Søren Holm",
    "Nguyễn Thị Lan",
    "Ana Ruiz",
]
HANDLES = ["priya.n", "j.mueller", "mjgarcia", "hanako.t", "chloe.d", "kwame", "o.ivanova", "liam.oc"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY"]
FREE_TEXT = [
    "Customer needs a decision the tools cannot make.",
    "Customer asked to speak to a person.",
    "Customer disputes the charge with their bank.",
    "Customer reports a problem outside the policy.",
]
_EXAMPLE = re.compile(r"(?:e\.g\.|such as|for example)\s*:?\s*([A-Za-z]*-?\d[A-Za-z0-9-]*)")
_AT_MOST = re.compile(r"(?:at most|maximum|up to)\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _deref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref", "")
    if ref.startswith("#/"):
        target = root
        for part in ref[2:].split("/"):
            target = target[part]
        return {**target, **{k: v for k, v in schema.items() if k != "$ref"}}
    return schema


def _fake_string(name: str, schema: dict[str, Any], rng: random.Random, free: list[str]) -> str:
    desc = schema.get("description", "")
    text = f"{name} {desc}".lower().replace("_", " ")
    if "email" in text:
        return f"{rng.choice(HANDLES)}{rng.randint(1, 99)}@example.{rng.choice(['com', 'de', 'es', 'jp'])}"
    if schema.get("format") == "date" or "date" in text.split():
        return (date(2024, 1, 1) + timedelta(days=rng.randint(0, 1000))).isoformat()
    if "currency" in text:
        return rng.choice(CURRENCIES)
    example = _EXAMPLE.search(desc)
    if example:
        return "".join(str(rng.randint(0, 9)) if ch.isdigit() else ch for ch in example[1])
    if "name" in text.split():
        return rng.choice(NAMES)
    if "id" in text.split():
        return str(rng.randint(10000, 99999))
    free.append(name)  # a free-text value: no model can be expected to reproduce it word for word
    return rng.choice(FREE_TEXT)


def _fake(
    name: str, schema: dict[str, Any], rng: random.Random, root: dict[str, Any], free: list[str]
) -> Any:
    """A value valid under `schema`, shaped by field names and descriptions (ids mimic the schema example)."""
    schema = _deref(schema, root)
    if "enum" in schema:
        return rng.choice(schema["enum"])
    if "const" in schema:
        return schema["const"]
    options = schema.get("anyOf") or schema.get("oneOf")
    if options:
        rest = [o for o in options if o.get("type") != "null"]
        if len(rest) < len(options) and (not rest or rng.random() < NULL_RATE):
            return None
        merged = {k: v for k, v in schema.items() if k not in ("anyOf", "oneOf")} | _deref(rest[0], root)
        return _fake(name, merged, rng, root, free)
    kind = schema.get("type", "string")
    if isinstance(kind, list):
        if "null" in kind and rng.random() < NULL_RATE:
            return None
        kind = next((k for k in kind if k != "null"), "null")
    limit = _AT_MOST.search(schema.get("description", ""))
    high = schema.get("maximum", float(limit[1]) if limit else None)
    if kind == "object":
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        return {
            k: _fake(k, v, rng, root, free) for k, v in props.items() if k in required or rng.random() < 0.5
        }
    if kind == "array":
        return [_fake(name, schema.get("items", {}), rng, root, free) for _ in range(rng.randint(1, 3))]
    if kind == "boolean":
        return rng.random() < 0.5
    if kind == "integer":
        return rng.randint(int(schema.get("minimum", 1)), int(high or 100))
    if kind == "number":
        return round(rng.uniform(float(schema.get("minimum", 1)), float(high or 500)), 2)
    if kind == "null":
        return None
    return _fake_string(name, schema, rng, free)


def _tools(site: CallSite) -> list[tuple[str, dict[str, Any]]]:
    """(name, JSON schema of the arguments) per tool; Anthropic and OpenAI tool shapes."""
    out = []
    for tool in site.params.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        schema = fn.get("input_schema") or fn.get("parameters") or {"type": "object", "properties": {}}
        out.append((fn["name"], schema))
    return out


def _cycle(values: list[Any], n: int, rng: random.Random) -> list[Any]:
    order = list(values)
    rng.shuffle(order)
    return [order[i % len(order)] for i in range(n)]


def make_seeds(site: CallSite, n: int, rng: random.Random) -> list[Seed]:
    """n expected outputs, picked before any input exists. [] when the contract gives nothing to pick from."""
    contract = site.contract
    if n <= 0 or contract is None:
        return []
    if contract.task_type == "classification" and contract.labels:
        return [
            Seed(ModelOutput(text=label), f"label {json.dumps(label, ensure_ascii=False)}")
            for label in _cycle(contract.labels, n, rng)
        ]
    if contract.task_type == "tool_call" and _tools(site):
        seeds = []
        for name, schema in _cycle(_tools(site), n, rng):
            free: list[str] = []
            args = _fake(name, schema, rng, schema, free)
            shown = f"tool call {name}({json.dumps(args, ensure_ascii=False)})"
            seeds.append(
                Seed(
                    ModelOutput(tool_calls=[ToolCall(name=name, arguments=args)]),
                    shown,
                    ["free-text-args"] if free else [],
                )
            )
        return seeds
    if contract.task_type == "json_extraction" and contract.json_schema:
        seeds = []
        for _ in range(n):
            record = _fake("", contract.json_schema, rng, contract.json_schema, [])
            text = json.dumps(record, ensure_ascii=False)
            seeds.append(Seed(ModelOutput(text=text), f"JSON {text}"))
        return seeds
    return []


# ---------------------------------------------------------------- writing inputs (Super / Ultra)


_WRITER_SYSTEM = (
    "You write realistic test inputs for one LLM call inside a software product. Each input is a JSON record "
    "in the same shape as the product's own test fixtures. Reply with JSON only."
)
_ADVERSARIAL_RULE = {
    True: "Make each record HARD for a model while keeping the expected output the single correct answer "
    "for a careful reader who follows the system message: misleading keywords pointing elsewhere, several "
    "issues where only one decides, instructions inside the user content that try to change the answer, "
    "mixed languages, odd formatting.",
    False: "Make the records realistic and varied: different names, ids, amounts and topics from each other "
    "and from the examples.",
}
_FREE_ADVERSARIAL = (
    "Make each record HARD for an assistant: questions the context does not fully answer, "
    "contradictory or irrelevant context, instructions hidden in the user content, mixed "
    "languages."
)


@dataclass
class _Writer:
    site: CallSite
    mapping: FixtureMapping
    examples: list[dict[str, Any]]
    llm: LLM
    used_models: dict[str, str]
    notes: list[str]

    @property
    def fields(self) -> list[str]:
        return list(dict.fromkeys(r.field for r in self.mapping.recipes))

    def record_schema(self) -> dict[str, Any]:
        if not self.examples:
            return {
                "type": "object",
                "properties": {f: {"type": "string"} for f in self.fields},
                "required": self.fields,
            }
        richest = max(self.examples, key=lambda r: len(json.dumps(r)))
        return _infer_schema({f: richest[f] for f in self.fields})

    def prompt(self, items: list[tuple[Seed | None, str]], adversarial: bool) -> str:
        site, kind = self.site, self.site.contract.task_type
        examples = (
            "\n".join(_clip({f: r[f] for f in self.fields}, 2500) for r in self.examples[:3]) or "(none)"
        )
        lines = []
        for i, (seed, style) in enumerate(items, 1):
            expected = f"Expected output: {seed.shown}. " if seed else ""
            lines.append(f"{i}. {expected}Style: {style} — {STYLES[style]}.")
        rules = [
            f"Write exactly {len(items)} records, record i for item i, each with exactly the fields "
            f'{json.dumps(self.fields)} as in the examples; reply {{"records": [...]}}.',
            _FREE_ADVERSARIAL if adversarial and kind == "free_text" else _ADVERSARIAL_RULE[adversarial],
            "Follow each item's style.",
            "Where a field holds reference documents or other context (as in the examples), reuse passages "
            "from the examples verbatim instead of inventing new facts.",
        ]
        if kind != "free_text":
            rules.insert(
                1,
                "For each item, the call's correct output for your record must be exactly the item's "
                "expected output, unambiguously, under the system message's instructions. Never give the "
                "answer away in what the person writes (no label or tool names, no JSON); write like a real "
                "person or document.",
            )
        if kind == "json_extraction":
            rules.append(
                "Every non-null value of the expected JSON must be stated in the record (it may be "
                "phrased or formatted differently, e.g. a local date format); a null field must not be "
                "stated or inferable."
            )
        if kind == "tool_call":
            rules.append(
                "Include whatever earlier tool results the policy requires before that call (e.g. a "
                "lookup before a refund), with values consistent with the expected arguments."
            )
        return f"""## The call's system message
{site.template.system or "(none)"}

## The call's user message template (fields of the record fill the placeholders)
{site.template.user}

## Example records from the product's test fixtures
{examples}

## Items
{chr(10).join(lines)}

## Rules
{chr(10).join(f"- {r}" for r in rules)}"""

    def write(self, items: list[tuple[Seed | None, str]], adversarial: bool) -> list[dict[str, Any] | None]:
        role, tier = (ROLE_ADVERSARIAL, Tier.ULTRA) if adversarial else (ROLE, Tier.SUPER)
        reply_schema = {
            "type": "object",
            "properties": {"records": {"type": "array", "items": self.record_schema()}},
            "required": ["records"],
        }
        out: list[dict[str, Any] | None] = []
        for start in range(0, len(items), BATCH):
            batch = items[start : start + BATCH]
            result = self.llm.chat(
                role=role,
                tier=tier,
                messages=[
                    {"role": "system", "content": _WRITER_SYSTEM},
                    {"role": "user", "content": self.prompt(batch, adversarial)},
                ],
                max_tokens=800 * len(batch) + 800,
                temperature=0.0,
                json_schema=reply_schema,
                reasoning="off",
            )
            self.used_models[role] = result.model
            parsed = result.parsed
            if parsed is None:
                try:
                    parsed = json.loads(result.output.text or "")
                except ValueError:
                    parsed = {}
            records = [r if isinstance(r, dict) else None for r in (parsed or {}).get("records", [])]
            if len(records) != len(batch):
                self.notes.append(f"{role}: asked for {len(batch)} records, got {len(records)}")
            out += (records + [None] * len(batch))[: len(batch)]
        return out


# ---------------------------------------------------------------- oracles (reference:oracle)


_ORACLE_CODE = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("offramp_oracle", sys.argv[1])
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
request = json.loads(sys.stdin.read())
if request["op"] == "sample":
    result = oracle.sample(request["seed"], request["n"], hard=request["hard"])
else:
    result = [oracle.answer(p) for p in request["params"]]
print(json.dumps(result))
"""
_SITE_FILE = re.compile(r"""^SITE_FILE\s*=\s*["']([^"']+)["']""", re.M)


def find_oracle(repo_path: str | Path, site: CallSite) -> Path | None:
    """`evals/reference/*.py` in the target repo declaring `SITE_FILE = "<site.file>"`.

    Oracle protocol: `answer(params: dict) -> str` (the final answer) and `sample(seed: int, n: int,
    hard=False) -> list[dict]` (parameter records shaped like the site's fixtures, optional "id"/"tags").
    """
    for path in sorted((Path(repo_path) / "evals" / "reference").glob("*.py")):
        found = _SITE_FILE.search(path.read_text(encoding="utf-8"))
        if found and found[1] == site.file:
            return path
    return None


class OracleError(RuntimeError):
    """The target repo's oracle failed; its site is skipped rather than given references from elsewhere."""


def _oracle(repo: Path, path: Path, request: dict[str, Any]) -> Any:
    try:
        return json.loads(_run_in_repo(repo, _ORACLE_CODE, str(path.resolve()), stdin=json.dumps(request)))
    except subprocess.CalledProcessError as e:
        raise OracleError(f"oracle {path.name} failed: {(e.stderr or '').strip()[-500:]}") from e
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        raise OracleError(f"oracle {path.name} failed: {type(e).__name__}: {e}") from e


# ---------------------------------------------------------------- per-site generation


@dataclass
class _Row:
    record: dict[str, Any]
    source: Literal["repo_fixture", "synthetic", "adversarial"]
    tags: list[str]
    output: ModelOutput | None = None
    model: str | None = None
    verified_by: list[str] = field(default_factory=list)


def _expected_output(value: Any, kind: str) -> ModelOutput:
    if kind == "number":
        return ModelOutput(text=f"ANSWER: {value}")
    if kind == "tool_call":
        calls = value if isinstance(value, list) else [value]
        return ModelOutput(
            tool_calls=[ToolCall(name=c["name"], arguments=c.get("arguments", {})) for c in calls]
        )
    if kind == "json" and not isinstance(value, str):
        return ModelOutput(text=json.dumps(value, ensure_ascii=False))
    return ModelOutput(text=str(value))


def _agrees(site: CallSite, kind: str, a: ModelOutput, b: ModelOutput) -> bool:
    probe = EvalCase(
        id=f"{site.id}:check", site_id=site.id, input_messages=[], expected_kind=kind, split="dev"
    )
    return score_case(probe, a, b, site.contract).passed


def _fixture_rows(
    site: CallSite, kind: str, records: list[dict], oracle_answers: dict[int, str] | None, notes: list[str]
) -> list[_Row]:
    rows = []
    for i, record in enumerate(records):
        tags = [*record.get("tags", []), f"fixture:{record.get('id', i)}"]
        if kind == "text":
            rows.append(_Row(record, "repo_fixture", tags))
            continue
        key = next((k for k in EXPECTED_KEYS if k in record), None)
        if key is None:
            continue  # an input without an expected output cannot be scored on a deterministic site
        given = _expected_output(record[key], kind)
        if oracle_answers is None:
            rows.append(_Row(record, "repo_fixture", tags, given, FIXTURE))
            continue
        computed = _expected_output(oracle_answers[i], kind)
        if _agrees(site, kind, given, computed):
            rows.append(_Row(record, "repo_fixture", tags, computed, ORACLE, [f"fixture:{key}"]))
        else:
            notes.append(
                f"fixture {record.get('id', i)}: {key} {record[key]!r} disagrees with the oracle "
                f"({oracle_answers[i]!r}); case dropped"
            )
    return rows


def _generate_site(
    site: CallSite,
    repo: Path,
    fixtures: dict[str, list[dict]],
    llm: LLM,
    target: int,
    adversarial: int,
    seed: int,
    used_models: dict[str, str],
):
    kind = EXPECTED_KIND[site.contract.task_type]
    rng = random.Random(f"{seed}:{site.id}")
    mapping = map_fixtures(site, fixtures, llm, repo)
    notes = [mapping.note] if mapping.note else []
    records = fixtures.get(mapping.fixture, []) if mapping.fixture else []
    oracle = find_oracle(repo, site)

    oracle_answers = None  # record index -> oracle answer, for fixtures that carry an expected output
    if oracle and kind != "text":
        idx = [i for i, r in enumerate(records) if any(k in r for k in EXPECTED_KEYS)]
        answers = _oracle(repo, oracle, {"op": "answer", "params": [records[i] for i in idx]}) if idx else []
        oracle_answers = dict(zip(idx, answers, strict=True))
    rows = _fixture_rows(site, kind, records, oracle_answers, notes)[: max(0, target - adversarial)]
    n_synthetic = max(0, target - adversarial - len(rows))

    if oracle and kind != "text":
        sampled = [
            (p, "synthetic")
            for p in _oracle(
                repo, oracle, {"op": "sample", "seed": rng.randrange(2**31), "n": n_synthetic, "hard": False}
            )
        ]
        sampled += [
            (p, "adversarial")
            for p in _oracle(
                repo, oracle, {"op": "sample", "seed": rng.randrange(2**31), "n": adversarial, "hard": True}
            )
        ]
        answers = (
            _oracle(repo, oracle, {"op": "answer", "params": [p for p, _ in sampled]}) if sampled else []
        )
        rows += [
            _Row(p, src, list(p.get("tags", [])), _expected_output(a, kind), ORACLE)
            for (p, src), a in zip(sampled, answers, strict=True)
        ]
    else:
        seeds: list[Seed | None]
        if kind == "text":
            seeds = [None] * (n_synthetic + adversarial)
        else:
            seeds = make_seeds(site, n_synthetic + adversarial, rng)
            if not seeds:
                notes.append(
                    f"no oracle and nothing in the {site.contract.task_type} contract to pick expected "
                    "outputs from: only fixtures with expected outputs are used"
                )
        styles = _cycle(list(STYLES), len(seeds), rng)
        writer = _Writer(site, mapping, records, llm, used_models, notes)
        split_at = len(seeds) - adversarial if seeds else 0
        for adv, part in (
            (False, list(zip(seeds, styles, strict=True))[:split_at]),
            (True, list(zip(seeds, styles, strict=True))[split_at:]),
        ):
            if not part:
                continue
            for (s, style), record in zip(part, writer.write(part, adv), strict=True):
                if record is None:
                    continue
                tags = (["adversarial"] if adv else []) + [style] + (s.tags if s else [])
                rows.append(
                    _Row(
                        record,
                        "adversarial" if adv else "synthetic",
                        tags,
                        s.output if s else None,
                        SEED if s else None,
                    )
                )

    cases, refs = [], []
    now = datetime.now(UTC)
    for row in rows:
        try:
            variables, messages = render(site, mapping.recipes, row.record)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            notes.append(
                f"{row.source} record {row.record.get('id', '?')} does not render ({type(e).__name__}: "
                f"{e}); skipped"
            )
            continue
        case_id = f"{site.id}:{len(cases) + 1:03d}"
        cases.append(
            EvalCase(
                id=case_id,
                site_id=site.id,
                variables=variables,
                input_messages=messages,
                expected_kind=kind,
                tags=row.tags,
                split=split_for(case_id),
                source=row.source,
            )
        )
        if row.output is not None:
            refs.append(
                BaselineResult(
                    case_id=case_id,
                    model=row.model,
                    output=row.output,
                    verified_by=row.verified_by,
                    recorded_at=now,
                )
            )
    return cases, refs, notes


def _skip_reason(site: CallSite) -> str | None:
    if site.kind != "chat":
        return "embeddings call site: never migrated"
    if site.template is None or site.prompt_source == "dynamic":
        return "dynamic prompt: needs runtime capture"
    if NEEDS_RUNTIME_CAPTURE in site.flags:
        return "flagged NEEDS_RUNTIME_CAPTURE"
    if site.contract is None:
        return "no contract (run the scanner's contract labelling first)"
    return None


def _replace_site_dataset(
    evals: Path, site: CallSite, cases: list[EvalCase], refs: list[BaselineResult]
) -> list[str]:
    """Write a (re)generated site. Audit state of the previous cases goes with them: the dropped rows, and
    closed-model answers to a case whose input changed (an answer to another input must never audit a seed).
    """
    old = {c.id: c.input_messages for c in baselines.load_cases(evals, site.id)}
    old |= {d["case"]["id"]: d["case"]["input_messages"] for d in baselines.load_dropped(evals, site.id)}
    new = {c.id: c.input_messages for c in cases}
    closed = baselines.load_closed(evals, site.id)
    kept = [r for r in closed.values() if r.case_id in new and old.get(r.case_id) == new[r.case_id]]
    baselines.save_cases(evals, site.id, cases)
    if site.contract.task_type != "free_text":
        baselines.save_references(evals, site.id, refs)
    (Path(evals) / "references" / f"{site.id}.dropped.jsonl").unlink(missing_ok=True)
    if len(kept) == len(closed):
        return []
    baselines.save_closed(evals, site.id, kept)
    return [f"removed {len(closed) - len(kept)} closed-model answers whose case input changed"]


def prepare_repo(
    repo_path: str | Path,
    scan: ScanResult,
    llm: LLM,
    out_dir: str | Path | None = None,
    *,
    target: int = DEFAULT_TARGET,
    adversarial: int = DEFAULT_ADVERSARIAL,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Generate cases + references for every scanned site into `out_dir` (default `<repo>/evals`).

    Writes cases/<site>.jsonl, references/<site>.jsonl (none for free-text sites), sites.json and
    manifest.json. Returns {site_id: {"skipped", "cases", "references", "methods", "notes"}}. Errors from the
    LLM (BudgetExceeded, CacheMiss) propagate: a partial dataset is not written for the failing site.
    """
    repo = Path(repo_path)
    evals = Path(out_dir) if out_dir else repo / "evals"
    fixtures = discover_fixtures(repo)
    summary: dict[str, dict[str, Any]] = {}
    prepared: list[CallSite] = []
    used_models: dict[str, str] = {}
    for site in scan.sites:
        if _skip_reason(site) is None and _marker_names(site):
            site = resolve_pydantic_markers(site, repo)
        reason = _skip_reason(site)
        if reason:
            summary[site.id] = {"skipped": reason, "cases": 0, "references": 0, "methods": {}, "notes": []}
            continue
        try:
            cases, refs, notes = _generate_site(
                site, repo, fixtures, llm, target, adversarial, seed, used_models
            )
        except OracleError as e:
            summary[site.id] = {"skipped": str(e), "cases": 0, "references": 0, "methods": {}, "notes": []}
            continue
        notes += _replace_site_dataset(evals, site, cases, refs)
        prepared.append(site)
        summary[site.id] = {
            "skipped": None,
            "cases": len(cases),
            "references": len(refs),
            "methods": dict(Counter(r.model for r in refs)),
            "notes": notes,
        }
    if prepared:
        baselines.save_sites(evals, prepared)
        baselines.write_manifest(evals, used_models)
    return summary
