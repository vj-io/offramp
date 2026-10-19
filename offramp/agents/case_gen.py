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

Tool-call arguments are classed per site (classify_tool_args): input-determined values (ids, amounts, dates)
must be stated in the written input; policy enums take a value the policy can lead to; free arguments (free
text, or an enum the policy never mentions) get FREE_ARG plus the case tag `free_text_arg:<name>`, so the
scorer checks presence only, and the writer is never asked for them. Every written record must pass
record_problem (it states its seed's input values and gives no answer away) or is rewritten once, then
dropped.

Inputs are fixture-shaped records. `Recipe`s turn a record into the template's variables the way the call
site's own code does (placeholder names come from expressions, e.g. `request_plan_name`, `len_messages`,
so they are mapped explicitly: by exact/suffix/len_ names when unambiguous, else by one Super call that is
checked by rendering every fixture). Synthetic records render through the same recipes as real fixtures.

Target-repo code (Pydantic schemas, oracles) runs in the target's own environment, in a subprocess.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tomllib
import unicodedata
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
RUN_TIMEOUT_S = 60  # one run of target code (schema import, oracle)
SETUP_TIMEOUT_S = 600  # building the target's environment the first time
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


def data_dir() -> Path:
    """Offramp's own data directory (OFFRAMP_DATA_DIR, default ./var): target environments and audit bundles
    live here, never in the target repo."""
    return Path(os.environ.get("OFFRAMP_DATA_DIR", "./var")).resolve()


def _run(
    cmd: list[str],
    cwd: Path,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> str:
    """Like subprocess.run(check=True), but in a new process group that is killed as a whole on timeout, so a
    hanging target import cannot outlive the `uv` that started it. Raises TimeoutExpired or
    CalledProcessError.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        start_new_session=True,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = proc.communicate(stdin, timeout=timeout or RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, out, err)
    return out


def _target_env() -> dict[str, str]:
    """The environment target code runs in: what uv and python need, nothing of Offramp's (no *_KEY,
    *_TOKEN, *_FALLBACK, GITHUB_TOKEN or LANGSMITH_*); HOME and TMPDIR in a scratch dir, as Task 10's local
    sandbox does."""
    scratch = data_dir() / "envs" / "home"
    scratch.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or str(Path.home() / ".cache" / "uv"),
        "UV_PYTHON_INSTALL_DIR": os.environ.get("UV_PYTHON_INSTALL_DIR")
        or str(Path.home() / ".local" / "share" / "uv" / "python"),
    }


def _target_python(repo: Path) -> tuple[list[str], dict[str, str]]:
    """Command and environment for running python in the target's own dependencies without writing into the
    target. The virtualenv lives in Offramp's data dir (UV_PROJECT_ENVIRONMENT) and holds the third-party
    dependencies only: from an existing uv.lock (--frozen --no-install-workspace --no-install-local) or else
    from pyproject.toml's dependencies (no lockfile is created). The project, its workspace members and local
    path sources are never installed, since an editable setuptools install writes <name>.egg-info/ into the
    target; their code imports from PYTHONPATH (_import_roots). Every run uses --no-sync; a changed lock or
    pyproject re-syncs the env."""
    repo = repo.resolve()  # commands below run with cwd=repo or the env dir: a relative path breaks both
    path = os.pathsep.join(str(p) for p in _import_roots(repo))
    base = _target_env() | {"PYTHONPATH": path}
    if not (repo / "pyproject.toml").exists():
        return [sys.executable], base  # ponytail: no project metadata to build an env from; Offramp's python
    env_dir = data_dir() / "envs" / hashlib.sha256(str(repo.resolve()).encode()).hexdigest()[:16]
    env = base | {"UV_PROJECT_ENVIRONMENT": str(env_dir)}
    lock = repo / "uv.lock"
    inputs = b"".join(p.read_bytes() for p in (repo / "pyproject.toml", lock) if p.exists())
    stamp, synced = env_dir / ".offramp-synced", hashlib.sha256(inputs).hexdigest()
    if not stamp.exists() or stamp.read_text() != synced:
        shutil.rmtree(env_dir, ignore_errors=True)
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        if lock.exists():
            sync = [
                "uv",
                "sync",
                "--quiet",
                "--frozen",
                "--no-install-workspace",
                "--no-install-local",
                "--project",
                str(repo),
            ]
            _run(sync, env_dir.parent, env=env, timeout=SETUP_TIMEOUT_S)
        else:
            project = tomllib.loads((repo / "pyproject.toml").read_text()).get("project", {})
            python = ["--python", project["requires-python"]] if project.get("requires-python") else []
            _run(["uv", "venv", "--quiet", *python, str(env_dir)], env_dir.parent, env=env)
            if project.get("dependencies"):
                install = [
                    "uv",
                    "pip",
                    "install",
                    "--quiet",
                    "--python",
                    str(env_dir),
                    "-r",
                    str(repo / "pyproject.toml"),
                ]
                _run(install, env_dir.parent, env=env, timeout=SETUP_TIMEOUT_S)
        stamp.write_text(synced)
    return ["uv", "run", "--quiet", "--no-sync", "--project", str(repo), "python"], env


def _run_in_repo(repo: Path, code: str, *args: str, stdin: str | None = None) -> str:
    """Run `python -B -c code` in the target repo's own environment (see _target_python); cwd is the repo, so
    its packages import. Raises on failure or timeout (RUN_TIMEOUT_S; the env setup gets SETUP_TIMEOUT_S)."""
    command, env = _target_python(repo)
    return _run([*command, "-B", "-c", code, *args], repo, stdin=stdin, env=env)


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


def _import_roots(repo: Path) -> list[Path]:
    """Where the target's own code imports from: src/ or the root of the repo, of each uv workspace member
    and of each local path source (none of them is installed, see _target_python)."""
    roots: list[Path] = []

    def add(directory: Path) -> None:
        roots.extend(p for p in (directory / "src", directory) if p.is_dir() and p not in roots)

    add(repo)
    try:
        uv = tomllib.loads((repo / "pyproject.toml").read_text()).get("tool", {}).get("uv", {})
    except (OSError, tomllib.TOMLDecodeError):
        return roots
    for pattern in uv.get("workspace", {}).get("members", []):
        for member in sorted(repo.glob(pattern)):
            if (member / "pyproject.toml").exists():
                add(member)
    for source in uv.get("sources", {}).values():
        if isinstance(source, dict) and "path" in source and (repo / source["path"]).is_dir():
            add((repo / source["path"]).resolve())
    return roots


def _module_name(repo: Path, file: str) -> str:
    """Dotted module name of `file` under the deepest import root that contains it."""
    path = (repo / file).with_suffix("")
    roots = sorted((r for r in _import_roots(repo) if path.is_relative_to(r)), key=lambda r: len(r.parts))
    parts = (
        list(path.relative_to(roots[-1]).parts) if roots else list(PurePosixPath(file).with_suffix("").parts)
    )
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
        out = _run_in_repo(Path(repo_path), _SCHEMA_CODE, _module_name(Path(repo_path), site.file), *names)
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


ArgClass = Literal["input", "policy", "free", "undetermined"]
FREE_ARG = "(free: not compared)"  # reference value of an argument tagged free_text_arg:<name>


@dataclass
class Seed:
    output: ModelOutput
    shown: str  # the expected output as the writer is told it
    tags: list[str] = field(default_factory=list)
    # Checked on every written record (record_problem):
    stated: list[tuple[str, Any, str | None]] = field(default_factory=list)  # (name, value, kind) to state
    requires: list[str] = field(default_factory=list)  # tools whose earlier results the input must show
    giveaways: list[str] = field(default_factory=list)  # expected values never written as "Label: value"
    verbatim: list[str] = field(default_factory=list)  # identifiers never written in what the person says
    undetermined: list[str] = field(default_factory=list)  # values of free enums no input may dictate
    present: list[Any] = field(default_factory=list)  # the record's seeded (non-null) scalar values
    absent: list[tuple[str, str, str | None]] = field(
        default_factory=list
    )  # null fields: (name, kind, id regex)


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
CURRENCY_ALIASES = {  # how an input may write a currency: ISO code, symbol, names in major languages
    "USD": ["usd", "$", "us$", "dollar", "dollars", "dólar", "dólares", "ドル", "美元"],
    "EUR": ["eur", "€", "euro", "euros", "ユーロ", "欧元"],
    "GBP": ["gbp", "£", "pound", "pounds", "sterling", "libra", "libras", "pfund", "ポンド", "英镑"],
    "JPY": ["jpy", "¥", "円", "yen", "日元"],
    "INR": ["inr", "₹", "rupee", "rupees", "rupia", "rupias", "ルピー"],
}
FREE_TEXT = ["Customer reports a problem with the order.", "Customer asks for help with an invoice."]
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


def _flat(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """The non-null branch of a nullable schema (anyOf/oneOf with null), $refs resolved."""
    schema = _deref(schema, root)
    options = [o for o in schema.get("anyOf") or schema.get("oneOf") or [] if o.get("type") != "null"]
    if options:
        return {k: v for k, v in schema.items() if k not in ("anyOf", "oneOf")} | _deref(options[0], root)
    return schema


def _string_kind(name: str, schema: dict[str, Any]) -> str | None:
    """What a string field holds, from its name and description; None for free text."""
    desc = schema.get("description", "")
    text = f"{name} {desc}".lower().replace("_", " ")
    if "email" in text:
        return "email"
    if schema.get("format") == "date" or "date" in text.split():
        return "date"
    if "currency" in text:
        return "currency"
    if _EXAMPLE.search(desc):
        return "example"
    if "name" in text.split():
        return "name"
    if "id" in text.split():
        return "id"
    return None


def _fake_string(name: str, schema: dict[str, Any], rng: random.Random) -> str:
    kind = _string_kind(name, schema)
    if kind == "email":
        return f"{rng.choice(HANDLES)}{rng.randint(1, 99)}@example.{rng.choice(['com', 'de', 'es', 'jp'])}"
    if kind == "date":
        return (date(2024, 1, 1) + timedelta(days=rng.randint(0, 1000))).isoformat()
    if kind == "currency":
        return rng.choice(CURRENCIES)
    if kind == "example":
        example = _EXAMPLE.search(schema.get("description", ""))[1]
        return "".join(str(rng.randint(0, 9)) if ch.isdigit() else ch for ch in example)
    if kind == "name":
        return rng.choice(NAMES)
    if kind == "id":
        return str(rng.randint(10000, 99999))
    return rng.choice(FREE_TEXT)


def _email_for(name: str, rng: random.Random) -> str | None:
    """An address built from a (possibly accented) name; None for names with no Latin letters."""
    words = re.findall(
        r"[a-z]+", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    )
    if not words:
        return None
    local = rng.choice([".".join(words[:2]), f"{words[0]}{rng.randint(1, 99)}", words[0][0] + words[-1]])
    return f"{local}@example.{rng.choice(['com', 'de', 'es', 'jp'])}"


def _fake(name: str, schema: dict[str, Any], rng: random.Random, root: dict[str, Any]) -> Any:
    """A value valid under `schema`, shaped by field names and descriptions (ids mimic the schema example).
    Objects get every property: an input can say a value is unknown (null), never that a key is absent."""
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
        return _fake(name, _flat(schema, root), rng, root)
    kind = schema.get("type", "string")
    if isinstance(kind, list):
        if "null" in kind and rng.random() < NULL_RATE:
            return None
        kind = next((k for k in kind if k != "null"), "null")
    limit = _AT_MOST.search(schema.get("description", ""))
    high = schema.get("maximum", float(limit[1]) if limit else None)
    if kind == "object":
        props = schema.get("properties", {})
        obj = {k: _fake(k, v, rng, root) for k, v in props.items()}
        kinds = {k: _string_kind(k, _flat(v, root)) for k, v in props.items() if isinstance(obj[k], str)}
        name = next((obj[k] for k, kind in kinds.items() if kind == "name"), None)
        for k, kind in kinds.items():
            if kind == "email" and name:  # a sender's address fits their name; writers "fix" ones that don't
                obj[k] = _email_for(name, rng) or obj[k]
        return obj
    if kind == "array":
        return [_fake(name, schema.get("items", {}), rng, root) for _ in range(rng.randint(1, 3))]
    if kind == "boolean":
        return rng.random() < 0.5
    if kind == "integer":
        return rng.randint(int(schema.get("minimum", 1)), int(high or 100))
    if kind == "number":
        return round(rng.uniform(float(schema.get("minimum", 1)), float(high or 500)), 2)
    if kind == "null":
        return None
    return _fake_string(name, schema, rng)


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


# ---------------------------------------------------------------- tool arguments: what can determine them


@dataclass
class ToolArgs:
    """How each required argument of one tool can be known:
    `input` - stated in the input (ids, amounts, dates);
    `policy` - an enum (or boolean) the policy decides; `allowed` = the values its sentences about this tool
        mention and the policy call also lists;
    `free` - a string the policy never decides (free text, an enum it never mentions): tagged
        free_text_arg:<name>, so the scorer checks it is a non-blank string and never compares the value;
    `undetermined` - a boolean or non-string enum the policy never decides: the scorer compares it exactly
        and no input determines it, so the tool gets no seeds (`unseedable` says why)."""

    schema: dict[str, Any]
    classes: dict[str, ArgClass]
    allowed: dict[str, list[Any]] = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)  # tools the policy wants called in an earlier step
    unseedable: str | None = None


class _EnumRule(BaseModel):
    tool: str
    arg: str
    values: list[Any]  # values the policy can lead to; [] = the policy never says how to choose


class _ToolRule(BaseModel):
    tool: str
    requires: list[str] = []


class _PolicyReply(BaseModel):
    enums: list[_EnumRule]
    tools: list[_ToolRule] = []


def _schema_type(schema: dict[str, Any]) -> str:
    kind = schema.get("type", "string")
    return next((k for k in kind if k != "null"), "string") if isinstance(kind, list) else kind


def _arg_class(name: str, schema: dict[str, Any], root: dict[str, Any]) -> ArgClass:
    schema = _flat(schema, root)
    if "enum" in schema or _schema_type(schema) == "boolean":
        return "policy"  # provisional: only the policy can decide it (classify_tool_args)
    if _schema_type(schema) == "string":
        return "input" if _string_kind(name, schema) else "free"
    return "input"


def _enum_values(schema: dict[str, Any], root: dict[str, Any]) -> list[Any]:
    schema = _flat(schema, root)
    return list(schema.get("enum", [True, False] if _schema_type(schema) == "boolean" else []))


def _value_str(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _tool_text(policy: str, tool: str) -> str:
    """The policy's blocks (bullets, numbered items, paragraphs) naming `tool`; "" if none does, so a policy
    that never names the tool decides none of its values."""
    blocks = re.split(r"\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)", policy)
    return "\n".join(b for b in blocks if _verbatim(b, tool) or _verbatim(b, tool.replace("_", " ")))


_INFLECTION = r"(?:s|es|d|ed|ing)?(?!\w)"  # "charge" also matches "charged", "other" never "otherwise"


def _value_mentioned(text: str, value: Any, arg: str) -> bool:
    """Whether `text` names the value. A string: one of its words occurs (words of 4+ letters with an
    inflection, shorter ones exactly). A number or boolean: its token follows the argument's name within three
    words ("severity 3", "set notify_customer to true"); an incidental number elsewhere ("wait 2 days") does
    not count."""
    if isinstance(value, str):
        for word in re.split(r"[_\W]+", value.casefold()):
            end = _INFLECTION if len(word) >= 4 else r"(?!\w)"
            if word and re.search(rf"(?<!\w){re.escape(word)}{end}", text, re.IGNORECASE):
                return True
        return False
    names = {arg, arg.replace("_", " "), *(w for w in re.split(r"[_\W]+", arg) if len(w) >= 4)}
    alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    token = re.escape(_value_str(value))
    return re.search(rf"(?<!\w)(?:{alts})\W+(?:\w+\W+){{0,3}}?{token}(?!\w)", text, re.IGNORECASE) is not None


def _is_string_arg(schema: dict[str, Any]) -> bool:
    """A string argument: its type, or for an enum without a type, all its values."""
    if "enum" in schema and "type" not in schema:
        return all(isinstance(v, str) for v in schema["enum"])
    return _schema_type(schema) == "string"


def _booleans(schema: dict[str, Any], root: dict[str, Any], path: str = "", depth: int = 0) -> list[str]:
    """Paths of the booleans inside an object/array argument (no input can state them; see _derivable)."""
    schema = _flat(schema, root)
    if depth > 8:
        return []
    if _schema_type(schema) == "boolean" and "enum" not in schema:
        return [path or "(value)"]
    found = [
        p
        for k, v in schema.get("properties", {}).items()
        for p in _booleans(v, root, f"{path}.{k}".lstrip("."), depth + 1)
    ]
    if isinstance(schema.get("items"), dict):
        found += _booleans(schema["items"], root, f"{path}[]", depth + 1)
    return found


def classify_tool_args(
    site: CallSite, llm: LLM | None, notes: list[str] | None = None
) -> dict[str, ToolArgs]:
    """Per tool, the class of each required argument (see ToolArgs). Non-enum arguments are classed from
    their schema. An enum or boolean is `policy` with the values that pass two checks: the policy's sentences
    about that tool mention the value (deterministic: a model asked which values a policy allows tends to list
    them all, and "other" in "the other tools" belongs to another tool's sentence), and one Super call (role
    case_gen, thinking off) lists it as a value the policy can lead to. With no value left it is `free` when
    it is a string, else `undetermined` and the tool is not seeded. Without an answer (llm None, invalid
    reply) no value passes the second check."""
    rules = {}
    for name, schema in _tools(site):
        props = schema.get("properties", {})
        required = [a for a in schema.get("required", list(props)) if a in props]
        rules[name] = ToolArgs(schema, {a: _arg_class(a, props[a], schema) for a in required})
    enums = [
        (t, a, _enum_values(r.schema["properties"][a], r.schema))
        for t, r in rules.items()
        for a, c in r.classes.items()
        if c == "policy"
    ]
    reply = None
    if enums and llm is not None:
        lines = "\n".join(f"- {t}.{a}: {json.dumps(values, ensure_ascii=False)}" for t, a, values in enums)
        prompt = f"""## Policy (the call's system message)
{site.template.system or "(none)"}

## Tools
{json.dumps(site.params.get("tools"), ensure_ascii=False)}

## Task
1. For each enum argument below, list the values the policy can lead to when that tool is called. Leave out
values the policy rules out for that tool (for example a reason the policy never accepts). Give an empty list
when the policy never says how the argument's value is chosen.
{lines}
2. For each tool, list the tools the policy requires to have been called in an earlier step (empty if none).
Reply {{"enums": [{{"tool", "arg", "values"}}], "tools": [{{"tool", "requires"}}]}}."""
        result = llm.chat(
            role=ROLE,
            tier=Tier.SUPER,
            messages=[
                {
                    "role": "system",
                    "content": "You read a support policy and say what it determines. JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.0,
            response_model=_PolicyReply,
            reasoning="off",
        )
        reply = result.parsed
        if reply is None and notes is not None:
            notes.append("tool policy reply invalid: no enum or boolean argument counts as policy-decided")
    said = (
        {(e.tool, e.arg): {_value_str(v).casefold() for v in e.values} for e in reply.enums} if reply else {}
    )
    policy = site.template.system or ""
    why: dict[tuple[str, str], str] = {}
    for t, a, values in enums:
        text, listed = _tool_text(policy, t), said.get((t, a), set())
        allowed = [v for v in values if _value_mentioned(text, v, a) and _value_str(v).casefold() in listed]
        prop = _flat(rules[t].schema["properties"][a], rules[t].schema)
        if allowed:
            rules[t].allowed[a] = allowed
        elif _is_string_arg(prop):
            rules[t].classes[a] = "free"
        else:
            rules[t].classes[a] = "undetermined"
            why[t, a] = f"{a} ({_schema_type(prop)}): not decided by the policy"
    for t, r in rules.items():
        for a, c in r.classes.items():
            nested = _booleans(r.schema["properties"][a], r.schema) if c == "input" else []
            if nested:  # e.g. ticket.notify_customer: _fake would pick it at random
                r.classes[a] = "undetermined"
                why[t, a] = f"{a} holds boolean {', '.join(nested)}, which no input states"
        undetermined = [why[t, a] for a, c in r.classes.items() if c == "undetermined"]
        if undetermined:
            r.unseedable = (
                f"required {'; '.join(undetermined)}; only string arguments can be compared "
                "presence-only, so these values could only be guessed"
            )
            if notes is not None:
                notes.append(f"{t}: not seeded: {r.unseedable}")
    for rule in reply.tools if reply else []:
        if rule.tool in rules:
            rules[rule.tool].requires = [x for x in rule.requires if x in rules and x != rule.tool]
    return rules


def describe_tool_args(rules: dict[str, ToolArgs]) -> str:
    """One site note: per tool, its arguments by class, the allowed values of policy ones, prior tools."""
    parts = []
    for tool, r in rules.items():
        groups = []
        for cls in ("input", "policy", "free", "undetermined"):
            names = [a for a, c in r.classes.items() if c == cls]
            if names and cls == "policy":
                groups.append("policy " + ", ".join(f"{a} {r.allowed[a]}" for a in names))
            elif names:
                groups.append(f"{cls} {', '.join(names)}")
        if r.requires:
            groups.append(f"requires {', '.join(r.requires)}")
        parts.append(f"{tool}: {'; '.join(groups) or 'no arguments'}")
    return "tool arguments: " + " | ".join(parts)


def _tool_seed(name: str, rules: ToolArgs, rng: random.Random) -> Seed:
    props = rules.schema["properties"]
    args, free, stated, giveaways, undetermined = {}, [], [], [], []
    for arg, cls in rules.classes.items():
        if cls == "free":
            args[arg] = FREE_ARG
            free.append(arg)
            undetermined += [str(v) for v in _enum_values(props[arg], rules.schema)]
        elif cls == "policy":
            args[arg] = rng.choice(rules.allowed[arg])
            giveaways.append(str(args[arg]))
        else:
            args[arg] = _fake(arg, props[arg], rng, rules.schema)
            stated.append((arg, args[arg], _string_kind(arg, _flat(props[arg], rules.schema))))
    shown_args = {a: v for a, v in args.items() if a not in free}
    shown = f"tool call {name}({json.dumps(shown_args, ensure_ascii=False)})"
    if free:
        shown += f"; leave {', '.join(free)} unstated: they are chosen freely"
    return Seed(
        ModelOutput(tool_calls=[ToolCall(name=name, arguments=args)]),
        shown,
        tags=[f"free_text_arg:{a}" for a in free],
        stated=stated,
        requires=list(rules.requires),
        giveaways=giveaways,
        verbatim=[name, *(v for v in giveaways if "_" in v)],
        undetermined=undetermined,
    )


def _json_stated(record: dict[str, Any], schema: dict[str, Any]) -> list[tuple[str, Any, str | None]]:
    """Top-level non-null values an input must state (enums and free text are judged by the audit)."""
    out = []
    for key, value in record.items():
        prop = _flat(schema.get("properties", {}).get(key, {}), schema)
        if value is None or "enum" in prop or isinstance(value, dict | list | bool):
            continue
        kind = _string_kind(key, prop) if isinstance(value, str) else None
        if isinstance(value, str) and kind is None:
            continue
        out.append((key, value, kind))
    return out


def _absent_kind(key: str, prop: dict[str, Any]) -> tuple[str, str | None] | None:
    """How to detect a field in an input when its seed value is null: (detector kind, id regex)."""
    if _schema_type(prop) in ("number", "integer"):
        text = f"{key} {prop.get('description', '')}".lower()
        return ("amount", None) if any(w in text for w in _AMOUNT_WORDS) else None
    kind = _string_kind(key, prop)
    if kind == "example":
        return "id", _shape_pattern(_EXAMPLE.search(prop.get("description", ""))[1])
    if kind == "name":  # the sender-name detector fits a person, not a product or a shop
        text = f"{key} {prop.get('description', '')}".lower().replace("_", " ")
        return ("name", None) if any(w in text for w in _PERSON_WORDS) else None
    return (kind, None) if kind in ("email", "date", "currency") else None


def json_seed(record: dict[str, Any], schema: dict[str, Any]) -> Seed:
    """The seed for one extraction record: its non-null values must be stated in the input, and its null
    fields must not appear anywhere in it (the writer is given their names)."""
    text = json.dumps(record, ensure_ascii=False)
    nulls = [k for k, v in record.items() if v is None]
    shown = f"JSON {text}"
    if nulls:
        shown += f"; null fields, which must not appear anywhere in the record: {', '.join(nulls)}"
    absent = []
    for key in nulls:
        found = _absent_kind(key, _flat(schema.get("properties", {}).get(key, {}), schema))
        if found:
            absent.append((key, *found))
    present = [v for v in record.values() if v is not None and not isinstance(v, dict | list)]
    return Seed(
        ModelOutput(text=text), shown, stated=_json_stated(record, schema), present=present, absent=absent
    )


def make_seeds(
    site: CallSite, n: int, rng: random.Random, tool_args: dict[str, ToolArgs] | None = None
) -> list[Seed]:
    """n expected outputs, picked before any input exists. [] when the contract gives nothing to pick from.
    Tool sites use `tool_args` (classify_tool_args); without it every enum argument counts as free."""
    contract = site.contract
    if n <= 0 or contract is None:
        return []
    if contract.task_type == "classification" and contract.labels:
        return [
            Seed(
                ModelOutput(text=label),
                f"label {json.dumps(label, ensure_ascii=False)}",
                giveaways=[label],
                verbatim=[label] if "_" in label else [],
            )
            for label in _cycle(contract.labels, n, rng)
        ]
    if contract.task_type == "tool_call" and _tools(site):
        rules = tool_args or classify_tool_args(site, None)
        seedable = [name for name, r in rules.items() if r.unseedable is None]
        return [_tool_seed(name, rules[name], rng) for name in _cycle(seedable, n, rng)] if seedable else []
    if contract.task_type == "json_extraction" and contract.json_schema:
        schema = contract.json_schema
        return [json_seed(_fake("", schema, rng, schema), schema) for _ in range(n)]
    return []


# ---------------------------------------------------------------- checking a written record


_NUMBER_RUN = re.compile(r"\d(?:[\d.,'   ]*\d)?")
_MONTHS = [
    "january jan januar enero janvier janeiro gennaio",
    "february feb februar febrero février fevereiro febbraio",
    "march mar märz marzo mars março",
    "april apr abril avril aprile",
    "may mai mayo maio maggio",
    "june jun juni junio juin junho giugno",
    "july jul juli julio juillet julho luglio",
    "august aug agosto août",
    "september sep sept septiembre septembre setembro settembre",
    "october oct oktober octubre octobre outubro ottobre",
    "november nov noviembre novembre novembro",
    "december dec dezember diciembre décembre dezembro dicembre",
]
_QUOTES = "\"'`“”‘’«»"


def _squash(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _numbers(text: str) -> set[Decimal]:
    """Every number the text states, read with either decimal separator ("34,90", "1.234,56", "1,234.56")."""
    out = set()
    for run in _NUMBER_RUN.findall(text):
        raw = re.sub(r"['   ]", "", run)
        for candidate in (raw.replace(",", ""), raw.replace(".", "").replace(",", ".")):
            try:
                out.add(Decimal(candidate))
            except InvalidOperation:
                pass
    return out


def _date_stated(iso: str, text: str) -> bool:
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return _squash(iso) in _squash(text)
    forms = [
        d.isoformat(),
        f"{d.year}/{d.month:02}/{d.day:02}",
        f"{d.day:02}.{d.month:02}.{d.year}",
        f"{d.day}.{d.month}.{d.year}",
        f"{d.day:02}/{d.month:02}/{d.year}",
        f"{d.month:02}/{d.day:02}/{d.year}",
        f"{d.day}/{d.month}/{d.year}",
        f"{d.month}/{d.day}/{d.year}",
        f"{d.year}年{d.month}月{d.day}日",
    ]
    if any(f in text for f in forms):
        return True
    words = set(re.findall(r"\w+", text.casefold()))
    return (
        str(d.year) in words
        and str(d.day) in {w.lstrip("0") for w in words if w.isdigit()}
        and bool(words & set(_MONTHS[d.month - 1].split()))
    )


def _derivable(value: Any, kind: str | None, text: str) -> bool:
    if isinstance(value, dict | list):  # every scalar inside must be stated
        items = value.values() if isinstance(value, dict) else value
        return all(_derivable(v, kind if isinstance(v, str) else None, text) for v in items if v is not None)
    if isinstance(value, bool):
        return False  # no evidence rule can show an input "states" True/False: the seed is rejected
    if isinstance(value, int | float):
        return Decimal(str(value)) in _numbers(text)
    if kind == "currency":
        return _currency_named(str(value), text)
    if kind == "date":
        return _date_stated(str(value), _without_date_headers(text))
    return _squash(str(value)) in _squash(text)


def _verbatim(text: str, value: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text, re.IGNORECASE) is not None


def _given_away(text: str, value: str) -> bool:
    """`value` written as an answer: after a colon/equals sign ("Category: billing", "Prioridad: urgente";
    any word continuing it counts) or alone in quotes ("classify as 'billing'")."""
    v = re.escape(value).replace("_", "[_ ]")
    q = re.escape(_QUOTES)
    pattern = rf"[:=]\s*[*_{q}]*\s*{v}|[{q}]\s*{v}\s*[{q}]"
    return re.search(pattern, text, re.IGNORECASE) is not None


_CJK = r"[぀-ヿ一-鿿가-힯]"
_OTHER_CURRENCY_CODES = ["chf", "cny", "aud", "cad", "mxn", "brl", "krw", "sek", "nok", "dkk", "pln"]
_EMAIL = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|[\w.+-]+\s*[\[(]at[\])]\s*[\w-]+(?:\s*[\[(]dot[\])]\s*[\w-]+)+"
)
_AMOUNT_WORDS = ("amount", "total", "price", "cost", "sum", "paid", "fee", "charge")
_PERSON_WORDS = (
    "customer",
    "sender",
    "contact",
    "person",
    "client",
    "buyer",
    "recipient",
    "user",
    "author",
    "full name",
    "signer",
    "requester",
    "kunde",
    "cliente",
)

# ---- email headers: keys compared without inner spaces ("Dat e:" is a Date header the writer mangled)
_HEADER_KEYS = {
    "from",
    "to",
    "cc",
    "bcc",
    "reply-to",
    "subject",
    "date",
    "sent",
    "de",
    "para",
    "asunto",
    "fecha",
    "enviado",
    "von",
    "an",
    "betreff",
    "datum",
    "gesendet",
    "à",
    "objet",
    "envoyé",
    "da",
    "van",
    "oggetto",
    "data",
    "差出人",
    "送信者",
    "宛先",
    "件名",
    "日付",
    "送信日時",
}
_ADDRESS_KEYS = _HEADER_KEYS - {
    "date",
    "sent",
    "fecha",
    "enviado",
    "datum",
    "gesendet",
    "envoyé",
    "data",
    "日付",
    "送信日時",
}
_DATE_KEYS = _HEADER_KEYS - _ADDRESS_KEYS
_RECIPIENT_KEYS = {"to", "cc", "bcc", "para", "an", "à", "宛先"}
_KEY = re.compile(r"^\s*>?\s*([^\W\d_][^\n:：]{0,24}?)\s*[:：]")

# ---- currencies: names and symbols that mean money anywhere; short or ambiguous ones only next to a number;
# a bare three-letter code only next to a number, minus time zones and units. "pound(s)" alone is a weight
# ("3 pounds"), so only the reverse check (CURRENCY_ALIASES) accepts it.
_NULL_CURRENCY_WORDS = (
    ({a for names in CURRENCY_ALIASES.values() for a in names} - {"pound", "pounds"})
    | set(_OTHER_CURRENCY_CODES)
    | {
        "franc",
        "francs",
        "franken",
        "yuan",
        "peso",
        "pesos",
        "reais",
        "rublo",
        "rublos",
        "ruble",
        "rubles",
        "rouble",
        "roubles",
        "livre",
        "livres",
        "sterlina",
        "sterline",
        "krona",
        "kronor",
        "krone",
        "kroner",
        "złoty",
        "zloty",
        "₩",
        "₽",
        "₺",
        "₪",
        "฿",
        "₱",
        "₫",
        "r$",
    }
)
_NEAR_NUMBER_WORDS = {"kr", "rs", "rs.", "zł", "元", "원"}
_NOT_CURRENCY = {
    "UTC",
    "GMT",
    "CET",
    "CEST",
    "EST",
    "EDT",
    "PST",
    "PDT",
    "JST",
    "KST",
    "IST",
    "BST",
    "MSK",
    "AEST",
    "KGS",
    "LBS",
    "PCS",
    "MHZ",
    "GHZ",
    "RPM",
    "DPI",
    "SKU",
    "PIN",
    "ETA",
    "VAT",
    "USB",
}
_NUM = r"\d[\d.,']*\d|\d"


def _alias_pattern(alias: str) -> str:
    """Words match whole ("eur" is not in "Europe"); symbols and CJK aliases match anywhere."""
    if alias.isalpha() and not re.search(_CJK, alias):
        return rf"(?<![^\W\d_]){re.escape(alias)}(?![^\W\d_])"
    return re.escape(alias)


def _alternation(words: set[str]) -> str:
    return "|".join(_alias_pattern(a) for a in sorted(words, key=len, reverse=True))


_CURRENCY_NAMES = re.compile(_alternation(_NULL_CURRENCY_WORDS), re.IGNORECASE)
_NEAR = _alternation(_NEAR_NUMBER_WORDS)
_CODE = rf"(?<![A-Za-z])(?!(?:{'|'.join(_NOT_CURRENCY)})(?![A-Za-z]))[A-Z]{{3}}(?![A-Za-z])"
_NEAR_NUMBER = re.compile(rf"(?:{_NUM})\s?(?:(?i:{_NEAR})|{_CODE})|(?:(?i:{_NEAR})|{_CODE})\s?(?:{_NUM})")
_CURRENCY_MARKER = rf"(?i:{_alternation(_NULL_CURRENCY_WORDS)}|{_NEAR})|{_CODE}"
_AMOUNT = re.compile(rf"(?:{_CURRENCY_MARKER})\s?(?:{_NUM})|(?:{_NUM})\s?(?:{_CURRENCY_MARKER})")

# ---- dates: written forms in en/de/es/fr/pt/it
_MONTH_WORDS = "|".join(sorted({w for m in _MONTHS for w in m.split()}, key=len, reverse=True))
_DATE_FORMS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?!\d)",
        r"(?<!\d)\d{4}/\d{1,2}/\d{1,2}(?!\d)",
        r"(?<![\d.])\d{4}\.\d{1,2}\.\d{1,2}(?![\d.])",
        r"(?<![\d.,])\d{1,2}[./]\d{1,2}[./]\d{2,4}(?!\d)",
        r"\d{4}\s*年\s*\d{1,2}\s*月",
        r"\d{1,2}\s*月\s*\d{1,2}\s*日",
        rf"(?<!\d)\d{{1,2}}(?:st|nd|rd|th)?\s+of\s+(?:{_MONTH_WORDS})(?![^\W\d_])",
        rf"(?<!\d)\d{{1,2}}(?:st|nd|rd|th)?\.?\s+(?:de\s+)?(?:{_MONTH_WORDS})(?![^\W\d_])",
        rf"(?<![^\W\d_])(?:{_MONTH_WORDS})\.?\s+\d{{1,2}}(?!\d)",
    )
]

# ---- names: where a sender is named, and what is not a name
_SIGNOFF = re.compile(
    r"^(?:thanks|thank you|many thanks|thx|regards|best|best regards|kind regards|warm regards|cheers|"
    r"sincerely|yours|gracias|muchas gracias|saludos|un saludo|atentamente|danke|vielen dank|grüße|gruß|"
    r"viele grüße|liebe grüße|mit freundlichen grüßen|lg|merci|cordialement|bien à vous|obrigad[oa]|"
    r"cumprimentos|grazie|cordiali saluti|saluti|よろしく[^\n]*|ありがとう[^\n]*|宜しく[^\n]*)"
    r"(?![^\W\d_])[\s,.!。、]*(?P<rest>.*)$",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^(?:hi|hello|hey|dear|hola|hallo|bonjour|ciao|olá|ola|oi|liebe|lieber|estimad[oa]|querid[oa]|caro|cara|"
    r"sehr geehrte[rs]?|good (?:morning|afternoon|evening))(?![^\W\d_])[\s,]*(?P<rest>[^,\n!:;]+)",
    re.IGNORECASE,
)
_LABEL = re.compile(
    r"^(?:name|customer|customer name|client|kunde|kundin|nombre|cliente|nom|nome|氏名|名前|お名前)"
    r"\s*[:：]\s*(?P<rest>.+)$",
    re.IGNORECASE,
)
_INTRODUCTION = re.compile(
    r"(?:\bmy name is|\bi am|\bi'm|\bthis is|\bme llamo|\bmi nombre es|\bsoy|\bich bin|\bich heiße|"
    r"\bmein name ist|\bje m'appelle|\bje suis)\s+([^\W\d_][\w'’-]*(?:\s+[^\W\d_][\w'’-]*)?)",
    re.IGNORECASE,
)
_HERE = re.compile(r"(?:^|[.!?,]\s*)([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)\s+here\b")
_HONORIFIC = re.compile(rf"({_CJK}{{1,4}})\s?(?:様|さん|殿|님)")
_DEVICE = re.compile(
    r"^(?:sent from|enviado desde|enviado do|von meinem|envoyé de|inviato da|送信元|发自)", re.IGNORECASE
)
_NOT_NAMES = {
    "team",
    "support",
    "customer",
    "customers",
    "service",
    "services",
    "care",
    "desk",
    "staff",
    "admin",
    "billing",
    "sales",
    "everyone",
    "all",
    "there",
    "friend",
    "friends",
    "sir",
    "madam",
    "colleagues",
    "equipo",
    "soporte",
    "atención",
    "cliente",
    "clientes",
    "kundenservice",
    "kundendienst",
    "zusammen",
    "alle",
    "équipe",
    "client",
    "assistance",
    "supporto",
    "equipe",
    "atendimento",
    "the",
    "urgent",
    "important",
    "regarding",
    "about",
    "order",
    "ticket",
    "request",
    "help",
    "hilfe",
    "ayuda",
    "aide",
    "aiuto",
    "ajuda",
    "damen",
    "herren",
    "お客",
}
_CLOSERS = {
    "以上",
    "谢谢",
    "謝謝",
    "감사합니다",
    "お願いします",
    "よろしくお願いします",
    "よろしく",
    "ありがとう",
    "ありがとうございます",
    "敬具",
    "草々",
    "再见",
    "此致",
    "敬礼",
    "お客",
}
_TITLES = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "prof",
    "herr",
    "frau",
    "sr",
    "sra",
    "srta",
    "m",
    "mme",
    "mlle",
    "sig",
    "dott",
}


def _header_key(line: str) -> str | None:
    """A line's normalised header key ("Dat e:" -> "date"; a quoted "> From:" -> "from"), if it has one."""
    match = _KEY.match(line)
    if not match:
        return None
    key = re.sub(r"\s+", "", match[1]).casefold()
    return key if key in _HEADER_KEYS or key.startswith("x-") else None


def _header_block(text: str) -> tuple[list[str], str]:
    """(the leading email header lines, the rest). The leading run of header lines is a header block when it
    has an address-type key (From, To, Subject, their translations) or ends at a blank line, as headers do;
    a note such as "Date: ...\nOrder: ..." or "注文番号: ..." has none."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and lines[i].strip() and _header_key(lines[i]):
        i += 1
    blank_after = i < len(lines) and not lines[i].strip()
    if not i or not ({_header_key(line) for line in lines[:i]} & _ADDRESS_KEYS or blank_after):
        return [], text
    return lines[:i], "\n".join(lines[i:])


def _without_date_headers(text: str) -> str:
    """The text minus the email's own Date header: when the message was sent, never a value of the record
    (Task 14: seeds whose order date sat only in that header were rightly read as null). A "日付:" line in
    a note's body stays: there it is a value."""
    headers, rest = _header_block(text)
    return "\n".join([h for h in headers if _header_key(h) not in _DATE_KEYS] + [rest])


def _customer_emails(text: str) -> list[str]:
    """Email addresses in `text` that are not the support desk's: not on a recipient line (To/Cc/Para/An,
    quoted ones included), not at a recipient's domain, not a role address such as support@ or info@."""
    recipients = {
        a.casefold()
        for line in text.splitlines()
        if _header_key(line) in _RECIPIENT_KEYS
        for a in _EMAIL.findall(line)
    }
    desk = {a.split("@")[-1] for a in recipients}
    roles = {
        "support",
        "soporte",
        "suporte",
        "help",
        "helpdesk",
        "info",
        "service",
        "services",
        "kundenservice",
        "kundendienst",
        "noreply",
        "no-reply",
        "contact",
        "billing",
        "sales",
        "ops",
        "admin",
        "care",
        "hilfe",
        "atendimento",
        "assistance",
    }
    found = []
    for address in _EMAIL.findall(text):
        local, _, domain = address.casefold().partition("@")
        if address.casefold() not in recipients and domain not in desk and local not in roles:
            found.append(address)
    return found


def _currency_named(code: str, text: str) -> bool:
    aliases = CURRENCY_ALIASES.get(code.upper(), [code.casefold()])
    return any(re.search(_alias_pattern(a), text, re.IGNORECASE) for a in aliases)


def _currency_matches(text: str) -> list[str]:
    """Currency mentions in text order: names, symbols and known codes anywhere; ambiguous short words and
    other three-letter codes only next to a number."""
    found = [(m.start(), m[0]) for m in _CURRENCY_NAMES.finditer(text)]
    found += [(m.start(), m[0]) for m in _NEAR_NUMBER.finditer(text)]
    return [m for _, m in sorted(found)]


def _shape_pattern(example: str) -> str:
    """A regex for ids shaped like the schema's example: "HH-48213" -> two letters, a hyphen, digits."""
    parts = re.findall(r"[A-Za-z]+|\d+|[^A-Za-z\d]", example)
    body = "".join(
        f"[A-Za-z]{{{len(p)}}}" if p.isalpha() else r"\d+" if p.isdigit() else re.escape(p) for p in parts
    )
    return rf"(?<![\w-]){body}(?![\w-])"


def _name_like(text: str, brands: set[str]) -> bool:
    """Whether `text` is a person's name: 1-4 capitalised words (titles and initials allowed), none of them
    boilerplate ("Team", "Support"), a brand from the call's own prompt ("HelpHub") or a lowercase word; or
    a CJK/Hangul run of at most 4 characters that is not a closing phrase ("以上", "谢谢")."""
    if re.search(_CJK, text):
        core = "".join(re.findall(_CJK, text))
        return bool(core) and core not in _CLOSERS and len(core) <= 4
    words = re.findall(r"[^\W\d_][\w'’.-]*", text)
    names = [w for w in words if w.rstrip(".").casefold() not in _TITLES]
    if not names or len(words) > 4:
        return False
    for word in names:
        bare = word.rstrip(".'’")
        if bare.casefold() in _NOT_NAMES or bare.casefold() in brands:
            return False
        if not word[0].isupper() or (bare.isupper() and len(bare) > 1):
            return False
    return True


def _person_names(text: str, brands: set[str]) -> list[str]:
    """Where an input names a person: a From display name, a greeting ("Hi Priya,", "> Dear Mr. Tanaka"), a
    label line ("Name: Priya Nair"), a signature (after a sign-off, or a short last line), "Priya here", a
    self-introduction ("I'm Priya"), a CJK name with an honorific ("田中様")."""
    found = []
    lines = [line.strip().lstrip(">").strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    for line in lines:
        key = _header_key(line)
        if key in ("from", "de", "von", "da", "van", "差出人", "送信者"):
            value = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            display = value.split("<")[0] if "<" in value else ("" if "@" in value else value)
            if display and _name_like(display, brands):
                found.append(display.strip())
            continue
        for pattern in (_GREETING, _LABEL):
            match = pattern.match(line)
            if match and _name_like(match["rest"].strip(), brands):
                found.append(match["rest"].strip())
    for i, line in enumerate(lines):
        signoff = _SIGNOFF.match(line)
        if not signoff:
            continue
        rest = signoff["rest"].strip()
        if rest and _name_like(rest, brands):
            found.append(rest)
        elif not rest and i + 1 < len(lines) and not _DEVICE.match(lines[i + 1]):
            if len(lines[i + 1]) <= 40 and _name_like(lines[i + 1], brands):
                found.append(lines[i + 1])
    last = lines[-1] if lines else ""
    if (
        len(last) <= 30
        and not re.search(r"[.!?。:：\d]", last)
        and not _SIGNOFF.match(last)
        and not _DEVICE.match(last)
        and not _header_key(last)
        and _name_like(last, brands)
    ):
        found.append(last)
    found += [m[1] for m in _HERE.finditer(text) if _name_like(m[1], brands)]
    found += [m[1] for m in _INTRODUCTION.finditer(text) if m[1][0].isupper() and _name_like(m[1], brands)]
    found += [m[1] for m in _HONORIFIC.finditer(text) if m[1] not in _CLOSERS]
    return found


def _date_matches(text: str) -> list[str]:
    """Dates in common written forms (ISO, numeric, 年月日, en/de/es/fr/pt/it month names), in text order."""
    found = sorted((m.start(), m[0]) for pattern in _DATE_FORMS for m in pattern.finditer(text))
    return [m for _, m in found]


def _same_value(kind: str, found: str, value: Any) -> bool:
    """Whether a detected mention is just a seeded (non-null) value of the record, e.g. the order date when
    the delivery date is null, or the customer's own address when a cc address is null."""
    if value is None or isinstance(value, bool | dict | list):
        return False
    if kind == "date":  # a mention without a year ("12 March") compares on day and month
        if not (isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)):
            return False
        return _date_stated(value, found if re.search(r"\d{4}", found) else f"{found} {value[:4]}")
    if kind == "amount":
        return isinstance(value, int | float) and Decimal(str(value)) in _numbers(found)
    if kind == "currency":
        return isinstance(value, str) and len(value) == 3 and _currency_named(value, found)
    seeded = _squash(str(value))
    return bool(seeded) and (_squash(found) == seeded or (kind == "name" and _squash(found) in seeded))


def _null_found(
    kind: str, pattern: str | None, text: str, present: list[Any] = (), brands: set[str] = frozenset()
) -> str | None:
    """The first mention in `text` of a field of this kind that is not one of the record's seeded values, or
    None. Dates ignore the message's own Date header."""
    if kind == "email":
        matches = _customer_emails(text)
    elif kind == "name":
        matches = _person_names(text, brands)
    elif kind == "date":
        matches = _date_matches(_without_date_headers(text))
    elif kind == "currency":
        matches = _currency_matches(text)
    elif kind == "amount":
        matches = [m[0] for m in _AMOUNT.finditer(text)]
    elif kind == "id" and pattern:
        matches = [m[0] for m in re.finditer(pattern, text, re.IGNORECASE)]
    else:
        matches = []
    return next((m for m in matches if not any(_same_value(kind, m, v) for v in present)), None)


def _brands(site: CallSite) -> set[str]:
    """Words the call's own prompt capitalises (the product, the desk): not a customer's name."""
    texts = [site.template.system or "", site.template.user] if site.template else []
    return {w.casefold() for t in texts for w in re.findall(r"[^\W\d_]+", t) if any(c.isupper() for c in w)}


def record_problem(
    site: CallSite, recipes: list[Recipe], seed: Seed | None, record: dict[str, Any]
) -> str | None:
    """Why a written record cannot be a case for `seed`, or None. Deterministic: it must render; it must state
    every input-determined value of the seed (anywhere, earlier tool results included) and show the earlier
    tool results the policy requires; and what the person writes (the record's plain-value fields) must not
    give the answer away: no expected label/enum value as "Label: value" or in quotes, no tool or label
    identifier verbatim, no value of an argument the input should not decide. A field the seed sets to
    null must not be detectable anywhere (see _null_found): an email address, a sender's name, a date
    outside the Date header, a currency, an amount with a currency, an id shaped like the schema's example."""
    try:
        variables, _ = render(site, recipes, record)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
        return f"does not render ({type(e).__name__}: {e})"
    if seed is None:
        return None
    everything = "\n".join(variables.values())
    written = "\n".join(variables[r.var] for r in recipes if r.op == "value")
    problems = [
        f"shows no earlier {t} result, which the policy requires first"
        for t in seed.requires
        if not _verbatim(everything, t)
    ]
    problems += [
        f"does not state {name}={value!r}"
        for name, value, kind in seed.stated
        if not _derivable(value, kind, everything)
    ]
    problems += [f"gives the expected output away ({v!r})" for v in seed.verbatim if _verbatim(written, v)]
    problems += [
        f"gives the expected output away ({v!r})"
        for v in seed.giveaways
        if v not in seed.verbatim and _given_away(written, v)
    ]
    problems += [
        f"states a value no input should decide ({v!r})" for v in seed.undetermined if _given_away(written, v)
    ]
    brands = _brands(site) if seed.absent else set()
    for name, kind, pattern in seed.absent:
        found = _null_found(kind, pattern, everything, seed.present, brands)
        if found:
            problems.append(f"states {name} although it is null ({found!r})")
    return "; ".join(problems) or None


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

    def prompt(
        self, items: list[tuple[Seed | None, str]], adversarial: bool, rejected: list[str] | None = None
    ) -> str:
        site, kind = self.site, self.site.contract.task_type
        examples = (
            "\n".join(_clip({f: r[f] for f in self.fields}, 2500) for r in self.examples[:3]) or "(none)"
        )
        lines = []
        for i, (seed, style) in enumerate(items, 1):
            expected = f"Expected output: {seed.shown}. " if seed else ""
            again = (
                f" Your previous record for this item was rejected: it {rejected[i - 1]}." if rejected else ""
            )
            lines.append(f"{i}. {expected}Style: {style} — {STYLES[style]}.{again}")
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
                "phrased or formatted differently, e.g. a local date format). A field that is null must "
                "not appear ANYWHERE in the record: not in headers (From, To, Subject), signatures, "
                "greetings, sign-offs or quoted text. For example: no email address when the email is "
                "null, not even a first name when the name is null, no currency code, symbol or name when "
                "the currency is null, no order number when the order id is null. A Date: header is when "
                "the message was sent, never a value of the record: write a non-null date in the body, and "
                "write no date in the body when the date is null."
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

    def write(
        self, items: list[tuple[Seed | None, str]], adversarial: bool, check: Any
    ) -> list[dict[str, Any] | None]:
        """Records for `items`. A record `check(seed, record)` rejects is asked for once more, with reason,
        and dropped (None, with a note) if the rewrite is rejected too."""
        role = ROLE_ADVERSARIAL if adversarial else ROLE
        records = self._ask(items, adversarial)
        problems = [
            check(seed, r) if r is not None else "was missing"
            for (seed, _), r in zip(items, records, strict=True)
        ]
        bad = [i for i, p in enumerate(problems) if p]
        if bad:
            again = self._ask([items[i] for i in bad], adversarial, [problems[i] for i in bad])
            for i, record in zip(bad, again, strict=True):
                problem = check(items[i][0], record) if record is not None else "was missing"
                records[i] = None if problem else record
                if problem:
                    self.notes.append(f"{role}: record dropped after one rewrite: it {problem}")
        return records

    def _ask(
        self, items: list[tuple[Seed | None, str]], adversarial: bool, rejected: list[str] | None = None
    ) -> list[dict[str, Any] | None]:
        role, tier = (ROLE_ADVERSARIAL, Tier.ULTRA) if adversarial else (ROLE, Tier.SUPER)
        reply_schema = {
            "type": "object",
            "properties": {"records": {"type": "array", "items": self.record_schema()}},
            "required": ["records"],
        }
        out: list[dict[str, Any] | None] = []
        for start in range(0, len(items), BATCH):
            batch = items[start : start + BATCH]
            reasons = rejected[start : start + BATCH] if rejected else None
            result = self.llm.chat(
                role=role,
                tier=tier,
                messages=[
                    {"role": "system", "content": _WRITER_SYSTEM},
                    {"role": "user", "content": self.prompt(batch, adversarial, reasons)},
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
        if kind == "json":
            try:
                json.loads(given.text)
            except ValueError:
                notes.append(f"fixture {record.get('id', i)}: {key} is not JSON; case dropped")
                continue
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
            tool_args = classify_tool_args(site, llm, notes) if kind == "tool_call" else None
            if tool_args:
                notes.append(describe_tool_args(tool_args))
            seeds = make_seeds(site, n_synthetic + adversarial, rng, tool_args)
            if not seeds:
                notes.append(
                    f"no oracle and nothing in the {site.contract.task_type} contract to pick expected "
                    "outputs from: only fixtures with expected outputs are used"
                )
        styles = _cycle(list(STYLES), len(seeds), rng)
        writer = _Writer(site, mapping, records, llm, used_models, notes)

        def check(seed: Seed | None, record: dict[str, Any]) -> str | None:
            return record_problem(site, mapping.recipes, seed, record)

        split_at = len(seeds) - adversarial if seeds else 0
        for adv, part in (
            (False, list(zip(seeds, styles, strict=True))[:split_at]),
            (True, list(zip(seeds, styles, strict=True))[split_at:]),
        ):
            if not part:
                continue
            for (s, style), record in zip(part, writer.write(part, adv, check), strict=True):
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
