"""Patch Composer (PRD §6.8): each MIGRATE site's winning adapter becomes a code change; the repo stays green.

Per run, on a sandbox copy of the target repo:
0. copy the repo without following symlinks (a link leaving the repo is dropped, with a warning) and check the
   untouched copy with the repo's own ruff + pytest;
1. add the Token Factory client module (`<pkg>/offramp_client.py` inside the package holding the migrated
   sites, so it ships with it; else `offramp/client.py`) and mark KEEP_CLOSED calls with
   `# offramp: kept on <model> — parity <p>% < <threshold>%`; check again;
2. for each MIGRATE site, against that same base (so no site depends on another's patch): the request
   parameters come from `build_request` (Task 6, the function the eval runner used), so what ships is what was
   evaluated. Large literals (tools, response_format, few-shot, a changed system prompt) are written into the
   file deterministically; Nemotron 3 Super (role `patcher`) rewires the call, its reply handling, callers
   and provider-pinning tests as search/replace edits; over-long string literals on patched lines are wrapped
   deterministically;
3. the edits are checked statically (allowed files, no removed tests, the call carries exactly the adapter's
   parameters), then alone by the repo's ruff + pytest in the sandbox, with up to two repair rounds. Any
   failure: the site becomes MANUAL_REVIEW;
4. the sites green alone are merged in site order and checked once together; if that fails they are added
   one at a time, and a site that breaks the others is dropped (MANUAL_REVIEW).
"""

import ast
import difflib
import io
import json
import os
import re
import shutil
import string
import tokenize
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from offramp.agents.pr_writer import TIER_NAMES, is_test_path, threshold
from offramp.agents.translate import UnresolvedCallSite, build_request
from offramp.sandbox import NO_SANDBOX, CheckResult, Sandbox
from offramp.schemas import Adapter, AdapterSpec, CallSite, EvalCase, RunConfig, Tier, Verdict
from offramp.tf.budget import BudgetExceeded
from offramp.tf.client import LLM, _merge, plan_structured
from offramp.tf.tiers import get_tier, reasoning_params

ROLE = "patcher"
MAX_TOKENS = 8000
REPAIR_MAX_TOKENS = 12000  # a complete second set of blocks, with room to spare (a recorded repair ran out)
REPAIRS = 2  # tries after the first, each shown the last failure; a repeated patch stops at once
REPAIR = """Your edits were not accepted: {failure}
The files are unchanged from my first message. Reply with SEARCH/REPLACE edit blocks only, no analysis or \
explanation: a complete new set (all blocks, not only the fix), then the NOTES line. Begin with `FILE:`."""
ROOT_CLIENT = ("offramp/client.py", "offramp.client")  # when the sites are not inside one package
TEMPLATE = Path(__file__).parents[1] / "templates" / "client.py.tmpl"
TIER_CONSTANTS = {Tier.LIGHTNING: "LIGHTNING", Tier.SUPER: "SUPER", Tier.ULTRA: "ULTRA"}
REQUEST_KWARGS = ("max_tokens", "temperature", "tools", "tool_choice", "response_format", "extra_body")
IGNORE = shutil.ignore_patterns(
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    "node_modules",
    ".env",
    ".env.*",
)
SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", "node_modules", "offramp"}

BuildRequest = Callable[[AdapterSpec, EvalCase, Tier], dict[str, Any]]
# Names a patch may use without importing them; added mechanically, like an editor's auto-import.
AUTO_IMPORTS = {
    "token_factory_client": "from {client} import token_factory_client",
    "OpenAI": "from openai import OpenAI",
    "json": "import json",
}
# pytest's built-in fixtures: a patched test may use one without requesting it; added like an import.
PYTEST_FIXTURES = ("monkeypatch", "capsys", "capfd", "caplog", "tmp_path", "tmp_path_factory")
_FILE = re.compile(r"^FILE: *`?([^`\s]+)`?[ \t]*$", re.M)
_BLOCK = re.compile(r"^<{7} SEARCH\n(.*?)\n={7}\n(.*?)>{7} REPLACE", re.S | re.M)


class Edit(BaseModel):
    path: str
    search: str
    replace: str


class PatchEdits(BaseModel):
    edits: list[Edit] = Field(default_factory=list)
    notes: str = ""


def parse_edits(text: str) -> PatchEdits:
    """The patcher replies in plain SEARCH/REPLACE blocks: code inside JSON strings needs every Python escape
    escaped again, and the model dropped some of them (a recorded run broke 3 of 5 patches that way)."""
    files = [(m.start(), m.group(1)) for m in _FILE.finditer(text)]
    edits = []
    for block in _BLOCK.finditer(text):  # a block without its own FILE line belongs to the previous file
        path = next((p for at, p in reversed(files) if at < block.start()), None)
        search, replace = block.group(1), block.group(2)
        if path:
            edits.append(
                Edit(path=path, search=search, replace=replace[:-1] if replace.endswith("\n") else replace)
            )
    notes = re.search(r"^NOTES:(.*)$", text, re.M)
    return PatchEdits(edits=edits, notes=notes.group(1).strip() if notes else "")


def edits_text(patch: PatchEdits) -> str:
    """Inverse of parse_edits (the reply format, as shown to the model)."""
    blocks = [
        f"FILE: {e.path}\n<<<<<<< SEARCH\n{e.search}\n=======\n{e.replace}\n>>>>>>> REPLACE"
        for e in patch.edits
    ]
    return "\n\n".join([*blocks, f"NOTES: {patch.notes}"])


class PatchResult(BaseModel):
    files: dict[str, str]  # repo-relative path -> full new text: the PR's commit (added and modified files)
    diffs: dict[
        str, str
    ]  # site_id -> unified diff of that site's change (MIGRATE patch or kept-site comment)
    site_files: dict[str, list[str]]  # site_id -> paths that change touched
    verdicts: list[Verdict]  # input order; MIGRATE sites whose patch failed are MANUAL_REVIEW
    adapters: dict[str, Adapter]  # copies, with .patch set for the migrated sites
    sandbox_mode: str  # "docker" (network off) | "local" (opt-in, not isolated) | "none"
    client_path: str | None = None  # where the Token Factory client module was added, when it was
    alone: dict[str, bool] = Field(default_factory=dict)  # MIGRATE site -> its patch passed the checks alone
    combined: list[str] = Field(default_factory=list)  # sites whose patches are in the final, checked tree
    warnings: list[str] = Field(default_factory=list)  # e.g. symlinks dropped from the sandbox copy


def client_location(repo: Path, files: list[str]) -> tuple[str, str] | None:
    """(path, module) of the client: `<pkg>/offramp_client.py` in the one top-level package that holds every
    migrated site, so it ships in that package's wheel; else `offramp/client.py` at the repository root.

    None when a site sits directly in a repository root that is itself a package (`__init__.py`): there is
    no package below the root to hold the module, and the root's own import name is not known here."""
    repo = repo.resolve()
    tops = set()
    for f in files:
        top = (repo / f).resolve().parent
        if top == repo:
            return None if (repo / "__init__.py").is_file() else ROOT_CLIENT
        if not (top / "__init__.py").is_file():
            return ROOT_CLIENT
        while top.parent != repo and (top.parent / "__init__.py").is_file():  # never climb to the root
            top = top.parent
        tops.add(top)
    if len(tops) != 1:
        return ROOT_CLIENT
    top = tops.pop()
    return f"{top.relative_to(repo).as_posix()}/offramp_client.py", f"{top.name}.offramp_client"


def copy_repo(src: Path, dst: Path) -> list[str]:
    """Copy without following symlinks; keep a link only if it resolves inside the copy. A link to a host
    file (e.g. /proc/self/environ) would otherwise copy its content in, for the repo's tests to print."""
    shutil.copytree(src, dst, ignore=IGNORE, symlinks=True)
    root, warnings = dst.resolve(), []
    for dirpath, dirnames, filenames in os.walk(dst):
        for name in [*dirnames, *filenames]:
            path = Path(dirpath) / name
            if not path.is_symlink():
                continue
            why = None
            try:
                path.resolve(strict=True)  # 3.12 raises RuntimeError on a loop, 3.13 OSError (ELOOP)
            except FileNotFoundError:  # dangling: judged by where it points
                pass
            except (OSError, RuntimeError):
                why = "it is a symlink loop"
            # os.path.realpath never raises on a loop; 3.12's non-strict Path.resolve can (dangling + loop)
            if why is None and not Path(os.path.realpath(path)).is_relative_to(root):
                why = "it points outside the repository"
            if why:
                path.unlink()
                warnings.append(f"dropped symlink {path.relative_to(dst).as_posix()}: {why}")
    return sorted(warnings)


# ---------------------------------------------------------------- request parameters


def call_params(request: dict[str, Any]) -> dict[str, Any]:
    """The `chat.completions.create` keyword arguments (minus messages) that `TFClient.chat(**request)` sends.

    Mirrors TFClient.chat: the same plan_structured, reasoning_params and extra_body merge; a test pins the
    two against each other.
    """
    info = get_tier(request["tier"])
    plan = plan_structured(
        request.get("json_schema"), None, request.get("json_mode_fallback", False), info.supports_json_schema
    )
    params: dict[str, Any] = {
        "model": info.model_id,
        "max_tokens": request["max_tokens"],
        "temperature": request.get("temperature", 0.0),
    }
    if plan.response_format is not None:
        params["response_format"] = plan.response_format
    if request.get("tools") is not None:
        params["tools"] = request["tools"]
    if request.get("tool_choice") is not None:
        params["tool_choice"] = request["tool_choice"]
    thinking = reasoning_params(info.tier, request.get("reasoning", "off"))["extra_body"]
    params["extra_body"] = _merge(request.get("extra_body") or {}, thinking)
    return params


def _probe_case(site: CallSite) -> EvalCase:
    """build_request needs a case; only its parameters are used, the messages come from the code."""
    return EvalCase(
        id=f"{site.id}:patch",
        site_id=site.id,
        input_messages=[{"role": "user", "content": ""}],
        expected_kind="text",
        split="dev",
    )


# ---------------------------------------------------------------- python literals


def _str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)  # a JSON string is a valid Python string literal


def _flat(v: Any) -> str:
    if isinstance(v, str):
        return _str(v)
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_flat(k)}: {_flat(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_flat(x) for x in v) + "]"
    return repr(v)  # bool, None, int, float


def _chunks(s: str, limit: int) -> list[str]:
    """Split `s` after whitespace so each chunk's literal fits `limit` columns; hard-cut over-long words."""
    limit = max(limit, 12)  # a hard cut must always make progress
    out, cur = [], ""
    for word in re.findall(r"\S*\s*", s):
        while len(_str(word)) > limit:
            n = limit - 2
            while n > 1 and len(_str(word[:n])) > limit:
                n -= 1
            if cur:
                out.append(cur)
                cur = ""
            out.append(word[:n])
            word = word[n:]
        if cur and len(_str(cur + word)) > limit:
            out.append(cur)
            cur = word
        else:
            cur += word
    return [*out, cur] if cur else out


def py_literal(value: Any, indent: int = 0, width: int = 100, start: int | None = None) -> str:
    """Python source for a JSON-like value: one line if it fits, else 4-space blocks; long strings are split
    into implicit concatenations. `start` is the column the literal begins at (default `indent`)."""
    flat = _flat(value)
    if (indent if start is None else start) + len(flat) + 1 <= width:
        return flat
    pad, inner = " " * indent, " " * (indent + 4)
    if isinstance(value, dict):
        items = [
            f"{inner}{_flat(k)}: {py_literal(v, indent + 4, width, indent + 6 + len(_flat(k)))},"
            for k, v in value.items()
        ]
        return "{\n" + "\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list | tuple):
        return "[\n" + "\n".join(f"{inner}{py_literal(v, indent + 4, width)}," for v in value) + f"\n{pad}]"
    if isinstance(value, str):
        return (
            "(\n" + "\n".join(f"{inner}{_str(c)}" for c in _chunks(value, width - indent - 5)) + f"\n{pad})"
        )
    return flat


def prompt_literal(s: str, width: int = 100, start: int = 0) -> str:
    """Triple-quoted literal equal to `s`; long lines wrapped with backslash-newline continuations."""
    body = s.replace("\\", "\\\\").replace("\r", "\\r").replace('"""', '\\"\\"\\"')
    if body.endswith('"'):
        body = body[:-1] + '\\"'
    lines, first = [], True
    for line in body.split("\n"):
        room = width - 1 - (start + 3 if first else 0)
        while len(line) > room and (cut := line.rfind(" ", 0, room)) > 0:
            lines.append(line[: cut + 1] + "\\")
            line, first, room = line[cut + 1 :], False, width - 1
        lines.append(line)
        first = False
    return '"""' + "\n".join(lines) + '"""'


# ---------------------------------------------------------------- source edits


def keep_comment(site: CallSite, v: Verdict, config: RunConfig) -> str:
    parity = f"{v.parity_pct:.1f}%" if v.parity_pct is not None else "n/a"
    return (
        f"# offramp: kept on {site.source_model or 'the original model'} — parity {parity} "
        f"< {threshold(site, config) * 100:g}%"
    )


def _insert_comment(source: str, site: CallSite, comment: str) -> str:
    lines = source.splitlines(keepends=True)
    i = site.lines[0] - 1
    indent = re.match(r"\s*", lines[i]).group(0)
    if i > 0 and lines[i - 1].strip().startswith("# offramp: kept on"):  # re-run: refresh, don't stack
        lines[i - 1] = f"{indent}{comment}\n"
    else:
        lines.insert(i, f"{indent}{comment}\n")
    return "".join(lines)


def _module_strings(tree: ast.Module) -> dict[str, str]:
    """Top-level `NAME = "<str>"` constants."""
    return {
        n.targets[0].id: n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str)
    }


def _module_values(tree: ast.Module) -> dict[str, Any]:
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


def _prepare(source: str, site: CallSite, spec: AdapterSpec, params: dict[str, Any], width: int):
    """Write the adapter's large literals into the site's module. Returns (source, constants, system_name):
    constants maps request keyword / "few_shot" -> constant name; system_name holds the system prompt."""
    tree = ast.parse(source)
    taken = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | set(_module_values(tree))

    def name(base: str) -> str:
        return base if base not in taken else f"{base}_{site.id.upper()}"

    system_name = None
    if spec.system_prompt is not None:
        strings = _module_strings(tree)
        same = [k for k, s in strings.items() if s == spec.system_prompt]
        if same:
            system_name = same[0]
        elif site.prompt_ref in strings:  # the adapter changed the prompt: change it where it lives
            system_name = site.prompt_ref
            node = next(
                n
                for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == system_name
            )
            lines = source.splitlines(keepends=True)
            new = f"{system_name} = {prompt_literal(spec.system_prompt, width, len(system_name) + 3)}\n"
            source = "".join(lines[: node.lineno - 1]) + new + "".join(lines[node.end_lineno :])
            tree = ast.parse(source)

    blocks, constants = [], {}
    if spec.system_prompt is not None and system_name is None:
        system_name = name("NEMOTRON_SYSTEM_PROMPT")
        blocks.append(f"{system_name} = {prompt_literal(spec.system_prompt, width, len(system_name) + 3)}")
    for key, base, value in [
        ("few_shot", "NEMOTRON_FEW_SHOT", spec.few_shot or None),
        ("tools", "NEMOTRON_TOOLS", params.get("tools")),
        ("response_format", "NEMOTRON_RESPONSE_FORMAT", params.get("response_format")),
    ]:
        if value is not None:
            constants[key] = name(base)
            blocks.append(f"{constants[key]} = {py_literal(value, 0, width, len(constants[key]) + 3)}")
    if blocks:
        top = _top_node(tree, site)
        at = min([top.lineno, *(d.lineno for d in getattr(top, "decorator_list", []))]) - 1
        lines = source.splitlines(keepends=True)
        source = "".join(lines[:at]) + "\n\n".join(blocks) + "\n\n\n" + "".join(lines[at:])
    return source, constants, system_name


def _top_node(tree: ast.Module, site: CallSite) -> ast.stmt:
    """The top-level statement holding the site: by function name (lines shift as files are edited)."""
    for n in tree.body:
        if any(
            isinstance(f, ast.FunctionDef | ast.AsyncFunctionDef) and f.name == site.function
            for f in ast.walk(n)
        ):
            return n
    return next((n for n in tree.body if n.lineno <= site.lines[0] <= n.end_lineno), tree.body[-1])


def _apply(edits: list[Edit], files: dict[str, str]) -> list[str]:
    shown = dict(files)
    seen = set()
    for e in edits:
        if (e.path, e.search, e.replace) in seen:  # the model repeated an edit: already applied
            continue
        seen.add((e.path, e.search, e.replace))
        if e.path not in files:
            return [f"{e.path} is not editable for this site"]
        n = files[e.path].count(e.search) if e.search else 0
        elsewhere = [p for p in files if p != e.path and e.search and files[p].count(e.search) == 1]
        if (
            n == 0 and len(elsewhere) == 1
        ):  # labelled with the wrong editable file: its text is unique elsewhere
            e = e.model_copy(update={"path": elsewhere[0]})
            n = 1
        if n != 1:
            hint = (
                " (an earlier block already changed these lines: merge the two)"
                if e.search and (e.search in shown[e.path])
                else ""
            )
            return [
                f"search text {'not found' if n == 0 else f'found {n} times'} in {e.path}{hint}: "
                f"{e.search[:80]!r}"
            ]
        files[e.path] = files[e.path].replace(e.search, e.replace, 1)
    return []


def _bound(tree: ast.Module) -> set[str]:
    """Names a module binds at module level, wherever: imports, defs, classes, (annotated) assignments, also
    inside top-level `if` / `try` / `with` / `for`. Function and class bodies are not entered."""
    bound, stack = set(), list(tree.body)
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Import | ast.ImportFrom):
            bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(n.name)
            continue
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        stack.extend(ast.iter_child_nodes(n))
    return bound


def add_imports(source: str, client: str) -> str:
    """Import AUTO_IMPORTS names the module uses but never binds (after its last top-level import); `client`
    is the module path of the Token Factory client."""
    tree = ast.parse(source)
    bound = _bound(tree)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = [line.format(client=client) for name, line in AUTO_IMPORTS.items() if name in used - bound]
    if not missing:
        return source
    imports = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    docstring = (
        tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)
    )
    at = imports[-1].end_lineno if imports else (tree.body[0].end_lineno if docstring else 0)
    lines = source.splitlines(keepends=True)
    return "".join(lines[:at]) + "".join(f"{m}\n" for m in missing) + "".join(lines[at:])


def _parameter_list(head: str) -> tuple[int, int]:
    """Offsets of the `(` and matching `)` of a `def` header's parameter list (comments are skipped)."""
    starts = [0]
    for line in head.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    depth, opened = 0, None
    for tok in tokenize.generate_tokens(io.StringIO(head).readline):
        if tok.type == tokenize.OP and tok.string in "()":
            depth += 1 if tok.string == "(" else -1
            at = starts[tok.start[0] - 1] + tok.start[1]
            if tok.string == "(" and opened is None:
                opened = at
            elif depth == 0:
                return opened, at
    raise ValueError("no parameter list")


def add_fixtures(source: str) -> str:
    """Add a built-in pytest fixture as a parameter to a test function that uses it without requesting or
    binding it (e.g. `monkeypatch.setattr(...)` in `def test_x():`)."""
    tree = ast.parse(source)
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in tree.body):
        return source  # a star import may well provide the name: never guess
    lines = source.splitlines(keepends=True)
    module = _bound(tree)  # a module-level name of the same spelling is what the test means
    tests = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
             and n.name.startswith("test")]  # fmt: skip
    for fn in sorted(tests, key=lambda n: n.lineno, reverse=True):
        args = fn.args
        given = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
        stored = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)}
        loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        missing = [f for f in PYTEST_FIXTURES if f in loaded and f not in given | stored | module]
        body_at = fn.body[0].lineno - 1  # the signature ends on the line before the body
        if not missing or args.vararg or args.kwarg or args.kwonlyargs or body_at < fn.lineno:
            continue
        head = "".join(lines[fn.lineno - 1 : body_at])
        try:
            opened, close = _parameter_list(head)
        except (tokenize.TokenError, ValueError, SyntaxError):  # e.g. the body on the `):` line: leave it
            continue
        sep = "" if head[opened + 1 : close].strip() == "" else ", "
        head = head[:close].rstrip().rstrip(",") + sep + ", ".join(missing) + head[close:]
        lines[fn.lineno - 1 : body_at] = [head]
    return "".join(lines)


def _test_names(source: str) -> set[str]:
    return {
        n.name
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name.startswith("test")
    }


def _call_name(node: ast.expr) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join(reversed(parts))


def conformance(
    source: str,
    site: CallSite,
    tier: Tier,
    params: dict[str, Any],
    constants: dict[str, str],
    system_name: str | None,
    spec: AdapterSpec,
) -> list[str]:
    """Problems with the patched site file: its call must carry exactly the adapter's request."""
    tree = ast.parse(source)
    scope = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == site.function
        ),
        tree,
    )
    where = f"`{site.function}`" if scope is not tree else site.file
    calls = [n for n in ast.walk(scope) if isinstance(n, ast.Call)]
    tf = [c for c in calls if _call_name(c.func).endswith("chat.completions.create")]
    if any(_call_name(c.func).endswith("messages.create") for c in calls):
        return [f"the old messages.create call is still in {where}"]
    if len(tf) != 1:
        return [f"expected one chat.completions.create call in {where}, found {len(tf)}"]
    kw = {k.arg: k.value for k in tf[0].keywords}
    if None in kw:
        return ["the call unpacks **kwargs; pass the adapter's arguments explicitly"]
    problems = [
        f"unexpected keyword argument {k}" for k in kw if k not in {"model", "messages", *REQUEST_KWARGS}
    ]
    module = _module_values(tree)

    def value(node: ast.expr) -> Any:
        try:
            return ast.literal_eval(node)
        except ValueError:
            return module.get(node.id, "<unresolved>") if isinstance(node, ast.Name) else "<unresolved>"

    model = kw.get("model")
    named = model is not None and TIER_CONSTANTS[tier] in (
        getattr(model, "id", None),
        getattr(model, "attr", None),
    )
    if not (named or (model is not None and value(model) == params["model"])):
        problems.append(f"model must be {params['model']!r}")
    if "messages" not in kw:
        problems.append("messages missing")
    for key in REQUEST_KWARGS:
        expected = params.get(key)
        if expected is None:
            if key in kw:
                problems.append(f"{key} is not part of the adapter's request")
        elif key not in kw:
            problems.append(f"{key} missing (adapter: {_flat(expected)[:120]})")
        elif value(kw[key]) != expected:
            problems.append(f"{key} = {_flat(value(kw[key]))[:120]}, adapter: {_flat(expected)[:120]}")
    used = {n.id for n in ast.walk(scope) if isinstance(n, ast.Name)}
    if system_name and system_name not in used:
        problems.append(f"system prompt constant {system_name} is not used in {where}")
    if "few_shot" in constants and constants["few_shot"] not in used:
        problems.append(f"few-shot constant {constants['few_shot']} is not used in {where}")
    texts = [n.value for n in ast.walk(scope) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    for label, piece in (("user prefix", spec.user_prefix), ("user suffix", spec.user_suffix)):
        if piece and not any(piece in t for t in texts):
            problems.append(f"{label} {piece!r} is not in {where}")
    return problems


# ---------------------------------------------------------------- context for the patcher prompt


def _context(repo: Path, site: CallSite, client_path: str) -> list[str]:
    """Editable files for one site: its module, callers of its function, tests touching them, test helpers."""
    files = sorted(
        p.relative_to(repo)
        for p in repo.rglob("*.py")
        if not SKIP_PARTS & set(p.relative_to(repo).parts[:-1])
        and p.relative_to(repo).as_posix() != client_path
    )
    text = {f: (repo / f).read_text() for f in files}
    own = Path(site.file)
    callers = [
        f
        for f in files
        if f != own
        and not is_test_path(f.as_posix())
        and site.function
        and re.search(rf"\b{re.escape(site.function)}\b", text[f])
    ]
    words = {own.stem, *(f.stem for f in callers), *([site.function] if site.function else [])}
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(words))) + r")\b")
    tests = [
        f for f in files if is_test_path(f.as_posix()) and f.name != "conftest.py" and pattern.search(text[f])
    ]
    helpers = set()
    for t in tests:
        for node in ast.walk(ast.parse(text[t])):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else (
                    [node.module]
                    if isinstance(node, ast.ImportFrom) and node.module and not node.level
                    else []
                )
            )
            for mod in names:
                helper = t.parent / f"{mod.split('.')[0]}.py"
                if helper in text and helper not in tests:
                    helpers.add(helper)
        if (t.parent / "conftest.py") in text:
            helpers.add(t.parent / "conftest.py")
    return [f.as_posix() for f in [own, *callers, *tests, *sorted(helpers)]]


SYSTEM = """You migrate one LLM call site of a Python repository to NVIDIA Nemotron on Nebius Token Factory, \
which is OpenAI-compatible: the call becomes `client.chat.completions.create(...)` on an `openai.OpenAI` \
client. Offramp has measured that the request given below reaches parity; make the code send exactly that \
request and keep the repository's ruff and tests green.

Reply with edit blocks only, then one NOTES line:

FILE: path/of/file.py
<<<<<<< SEARCH
exact lines copied from the file
=======
the lines that replace them
>>>>>>> REPLACE

NOTES: one or two sentences on what you changed.

- Each block replaces the one occurrence of its SEARCH lines in that file. Copy them verbatim from the file \
as shown (same indentation, same escapes), with enough lines to be unique. Blocks apply in order, each to \
the result of the previous ones; never repeat a block. Only files marked (editable); no new files.

Rules:
1. Keep every function's name, parameters and return value. Only the type of an injected client parameter \
may change, to `OpenAI` (from openai).
2. The call passes exactly the keyword arguments under "Required request", using the constants named there \
(they are already defined in the file): no other arguments, no **kwargs.
3. Read the reply the OpenAI way: text is `response.choices[0].message.content` (may be None); tool calls \
are `response.choices[0].message.tool_calls or []`, each with `.function.name` and `.function.arguments` \
(a JSON string: json.loads it). Map them onto exactly what the function returned before.
4. Code in this repo that constructs the old SDK client to call this function must pass \
`token_factory_client()` (imported from the client module named under "Required request") instead. \
Tests that replaced that constructor in such a module with a fake must replace `token_factory_client` \
there instead: check the editable tests for such patches, since one left on the old constructor makes the \
code call the real client.
5. Tests: you may update the tests that pin the old provider - assertions on the request's shape (model, \
SDK keyword arguments, message layout, checks of the call's source such as `"X" in ast.unparse(call)`) \
and the repo's provider fakes: extend its existing OpenAI-style fake helper when it cannot yet return what \
the new code reads (e.g. tool calls), building real openai response objects the way the helper already \
does, instead of writing ad-hoc fake classes inside tests. Never weaken a behavioural or parsing \
assertion: expected return values and parsing checks stay exactly as they are. Never delete or rename a \
test.
6. Add the imports your code needs and remove the ones it leaves unused (import order is fixed \
automatically afterwards).
7. Keep lines within the repo's line length."""


def _prompt(
    site: CallSite,
    tier: Tier,
    spec: AdapterSpec,
    params: dict[str, Any],
    constants: dict[str, str],
    system_name: str | None,
    files: dict[str, str],
    client: str,
    client_at: tuple[str, str],
    width: int,
) -> str:
    args = [f'    model="{params["model"]}",', "    messages=[...],  # see below"]
    for key in REQUEST_KWARGS:
        if key in params:
            shown = constants.get(key) or py_literal(params[key], 4, width, 5 + len(key))
            args.append(f"    {key}={shown},")
    messages = [
        f'{{"role": "system", "content": {system_name}}}'
        if system_name
        else "the system prompt the code sends today, as the first message (an Anthropic `system=` argument "
        'becomes {"role": "system", "content": ...})'
    ]
    if "few_shot" in constants:
        messages.append(f"*{constants['few_shot']}")
    last = "the user message(s) the code sends today"
    wraps = [
        f"{_str(text)} {how}"
        for text, how in ((spec.user_prefix, "prepended"), (spec.user_suffix, "appended"))
        if text
    ]
    if wraps:
        last += f", with {' and '.join(wraps)} to the content of the last one"
    messages.append(last)
    task = site.contract.task_type if site.contract else "unknown"
    lines = [
        f"## Call site {site.id}",
        f"`{site.file}` lines {site.lines[0]}-{site.lines[-1]}, function `{site.function}`: {site.sdk} "
        f"`{site.call}` on `{site.source_model}` ({task}). Target: {TIER_NAMES[tier]} on Token "
        "Factory.",
        "",
        "## Required request",
        "client.chat.completions.create(",
        *args,
        ")",
        "messages, in this order:",
        *(f"{i}. {m}" for i, m in enumerate(messages, 1)),
        "",
        f"Token Factory client: `from {client_at[1]} import token_factory_client` (module shown below).",
        f"Repo ruff line length: {width}.",
        "",
        "## Files",
        f"### {client_at[0]} (read-only)",
        "```python",
        client.rstrip("\n"),
        "```",
    ]
    for path, text in files.items():
        lines += [f"### {path} (editable)", "```python", text.rstrip("\n"), "```"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- composer


def line_length(repo: Path) -> int:
    """The target's ruff line length: `.ruff.toml`, `ruff.toml`, then pyproject `[tool.ruff]` (ruff's own
    precedence); 88, ruff's default, when none sets it."""
    for name, table in ((".ruff.toml", ()), ("ruff.toml", ()), ("pyproject.toml", ("tool", "ruff"))):
        try:
            cfg = tomllib.loads((repo / name).read_text())
        except (FileNotFoundError, tomllib.TOMLDecodeError):
            continue
        for key in table:
            cfg = cfg.get(key, {})
        if "line-length" in cfg:
            return int(cfg["line-length"])
        if name != "pyproject.toml":
            return 88  # a ruff.toml without line-length: ruff stops looking there
    return 88


def _summary(output: str) -> str:
    """The lines that say what failed (pytest FAILED/ERROR, ruff findings), else the output's tail."""
    # run-specific tokens (object addresses, durations) out: the summary feeds the next request's cache key
    output = re.sub(r"\b\d+(\.\d+)?s\b", "<t>s", re.sub(r"0x[0-9a-fA-F]{6,}", "0x…", output))
    keep = [ln for ln in output.splitlines() if re.match(r"(FAILED|ERROR) |\S+\.py:\d+:\d+: |\$ |E\s{3}", ln)]
    return (" | ".join(keep[:8]) if keep else output.strip()[-500:])[:1200]


def changed_lines(before: str, after: str) -> set[int]:
    """1-based numbers of the lines in `after` that are new or changed relative to `before`."""
    sm = difflib.SequenceMatcher(None, before.splitlines(), after.splitlines(), autojunk=False)
    return {j + 1 for tag, _, _, j1, j2 in sm.get_opcodes() if tag != "equal" for j in range(j1, j2)}


def _fstring_pieces(segment: str, room: int) -> list[str] | None:
    """Split one single-quoted f-string at spaces outside its replacement fields into literals that each fit
    `room` columns; a piece with no field drops the `f` (ruff F541) and its doubled braces."""
    m = re.match(r"([A-Za-z]*)(['\"])", segment)
    prefix, q = m.group(1), m.group(2)
    if segment[len(prefix) : len(prefix) + 3] == q * 3 or set(prefix.lower()) - {"f"} or "\\N{" in segment:
        return None
    body, words, depth, start, i = segment[len(prefix) + 1 : -1], [], 0, 0, 0
    while i < len(body):
        if depth == 0 and body[i : i + 2] in ("{{", "}}"):
            i += 2
            continue
        if body[i] == "{":
            depth += 1
        elif body[i] == "}" and depth:
            depth -= 1
        elif body[i] == "\\":
            i += 1  # an escape is never a split point
        elif depth == 0 and body[i] == " " and i + 1 < len(body) and body[i + 1] != " ":
            words.append(body[start : i + 1])
            start = i + 1
        i += 1
    words.append(body[start:])

    def literal(text: str) -> str:
        has_field = "{" in text.replace("{{", "").replace("}}", "")
        return f"{prefix}{q}{text}{q}" if has_field else f"{q}{text.replace('{{', '{').replace('}}', '}')}{q}"

    pieces, cur = [], ""
    for word in words:
        if cur and len(literal(cur + word)) > room:
            pieces.append(literal(cur))
            cur = word
        else:
            cur += word
    return [*pieces, literal(cur)] if pieces else None


def _literals(source: str) -> list[tuple[int, int, int, bool]]:
    """(row, start col, end col, part of a concatenation) of every single-line, top-level string literal
    token; strings nested inside f-string fields are not listed. Literals separated only by line breaks or
    comments form one implicit concatenation."""
    out, depth, start, prev_end = [], 0, None, None
    fs, fe = getattr(tokenize, "FSTRING_START", -1), getattr(tokenize, "FSTRING_END", -1)
    between: list[int] = []
    spans: list[tuple[tuple[int, int], tuple[int, int], bool]] = []  # start, end, joined to the previous
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == fs:
            if depth == 0:
                start = tok.start
            depth += 1
            continue
        if tok.type == fe:
            depth -= 1
            if depth == 0:
                spans.append(
                    (start, tok.end, prev_end is not None and set(between) <= {tokenize.NL, tokenize.COMMENT})
                )
                prev_end, between = tok.end, []
            continue
        if depth:
            continue
        if tok.type == tokenize.STRING:
            spans.append(
                (tok.start, tok.end, prev_end is not None and set(between) <= {tokenize.NL, tokenize.COMMENT})
            )
            prev_end, between = tok.end, []
        else:
            between.append(tok.type)
            if tok.type not in (tokenize.NL, tokenize.COMMENT):
                prev_end = None
    joined_next = [spans[k + 1][2] if k + 1 < len(spans) else False for k in range(len(spans))]
    for k, ((r1, c1), (r2, c2), joined) in enumerate(spans):
        if r1 == r2:
            out.append((r1, c1, c2, joined or joined_next[k]))
    return out


def wrap_long_strings(source: str, lines: set[int], width: int) -> str:
    """Split each over-long string literal on the given (patched) lines at spaces: a plain string is
    re-rendered from its value, an f-string is split outside its fields. A literal that stands alone becomes a
    parenthesised implicit concatenation; a literal that is already one piece of a bracketed concatenation is
    split in place. A split is kept only if the module's AST is unchanged, i.e. every string still evaluates
    to the same value; anything else (bytes, triple-quoted f-strings) is left as it is. Never adds noqa."""
    expected = ast.dump(ast.parse(source))
    text = source.splitlines(keepends=True)
    for row, c1, c2, joined in sorted(
        _literals(source), reverse=True
    ):  # bottom-right first: offsets stay valid
        line = text[row - 1]
        if row not in lines or len(line.rstrip("\r\n")) <= width:
            continue
        segment = line[c1:c2]
        indent = re.match(r"[ \t]*", line).group(0)
        lead = " " * c1 if joined else indent + "    "
        if width - len(lead) < 20:  # no room to split into (e.g. a second literal far along one line)
            continue
        if segment.lstrip("rRuU")[:1] in "'\"":  # plain string: re-render from its value
            value = ast.literal_eval(segment)
            chunks = _chunks(value, width - len(lead)) if isinstance(value, str) else []
            pieces = [_str(c) for c in chunks] if len(chunks) > 1 else None
        else:
            pieces = _fstring_pieces(segment, width - len(lead))
        if not pieces:
            continue
        if joined:  # already inside brackets, next to other pieces: more adjacent pieces
            split = f"\n{lead}".join(pieces)
        else:
            split = "(\n" + "".join(f"{lead}{p}\n" for p in pieces) + f"{indent})"
        candidate = [*text[: row - 1], line[:c1] + split + line[c2:], *text[row:]]
        try:
            if ast.dump(ast.parse("".join(candidate))) != expected:
                continue
        except SyntaxError:
            continue
        text = candidate
    return "".join(text)


def merge_onto(base: str, ours: str, theirs: str) -> str | None:
    """Apply `theirs`'s line changes against `base` onto `ours`, which holds other sites' changes against the
    same base. A change `ours` already holds is skipped, so two sites adding the same import or the same fake
    compose; a base line both changed differently is a conflict (None). Unlike diff3, edits on adjacent lines
    (one site's row next to another's) are not a conflict."""
    b, o, t = (x.splitlines(keepends=True) for x in (base, ours, theirs))
    where: dict[int, int] = {}  # base line -> its index in ours, for base lines ours left unchanged
    for tag, i1, i2, j1, _ in difflib.SequenceMatcher(None, b, o, autojunk=False).get_opcodes():
        if tag == "equal":
            where.update({i1 + k: j1 + k for k in range(i2 - i1)})
    out = list(o)

    def present(new: list[str], i1: int, i2: int) -> bool:  # ours already holds `new` around base[i1:i2]
        lo = next((where[k] + 1 for k in range(i1 - 1, -1, -1) if k in where), 0)
        hi = next((where[k] for k in range(i2, len(b)) if k in where), len(o))
        window = o[lo:hi]
        return (
            any(window[k : k + len(new)] == new for k in range(len(window) - len(new) + 1))
            if new
            else not window
        )

    hunks = [h for h in difflib.SequenceMatcher(None, b, t, autojunk=False).get_opcodes() if h[0] != "equal"]
    for _, i1, i2, j1, j2 in reversed(hunks):  # last first: earlier positions in `out` stay valid
        new = t[j1:j2]
        if present(new, i1, i2):
            continue
        if i1 == i2:  # an insertion before base line i1
            if i1 in where:
                pos = where[i1]
            elif i1 == len(b) or i1 - 1 in where:
                pos = where[i1 - 1] + 1 if i1 - 1 in where else len(o)
            else:
                return None
            out[pos:pos] = new
        elif all(k in where for k in range(i1, i2)) and where[i2 - 1] - where[i1] == i2 - i1 - 1:
            out[where[i1] : where[i2 - 1] + 1] = new
        else:  # ours changed these base lines differently
            return None
    return "".join(out)


def _diff(before: dict[str, str | None], after: dict[str, str]) -> str:
    out: list[str] = []
    for p in sorted(after):
        a = before.get(p)
        if a != after[p]:
            out += difflib.unified_diff(
                (a or "").splitlines(keepends=True),
                after[p].splitlines(keepends=True),
                "/dev/null" if a is None else f"a/{p}",
                f"b/{p}",
            )
    return "".join(out)


def client_module() -> str:
    ids = {t.name.lower(): get_tier(t).model_id for t in Tier}
    return string.Template(TEMPLATE.read_text()).substitute(ids)


@dataclass
class _Job:
    """One MIGRATE site being patched: what the edits must achieve and the files they may touch."""

    site: CallSite
    tier: Tier
    spec: AdapterSpec
    params: dict[str, Any]
    constants: dict[str, str]
    system_name: str | None
    editable: dict[str, str]  # as shown to the patcher (the site's module with the adapter's constants)
    before: dict[str, str]  # on disk before the site's patch: the base every site is patched against
    client_module: str  # import path of the Token Factory client module
    width: int  # the target's ruff line length


def _wrap(job: _Job, path: str, text: str) -> str:
    if not path.endswith(".py"):
        return text
    return wrap_long_strings(text, changed_lines(job.before[path], text), job.width)


def _attempt(
    job: _Job, reply: str, repo: Path, box: Sandbox, formatted: bool, write
) -> tuple[str | None, dict]:
    """Apply one reply to the base, check it alone, then put the base back. (None, the site's patched files)
    when the repo is green with it; else (why not, {})."""
    patch = parse_edits(reply)
    if not patch.edits:
        return "the reply holds no edit blocks", {}
    new = dict(job.editable)
    problems = _apply(patch.edits, new)
    for path in new if not problems else []:
        try:
            if new[path] != job.editable[path]:
                new[path] = add_imports(new[path], job.client_module)
                if is_test_path(path):
                    new[path] = add_fixtures(new[path])
        except SyntaxError as e:
            problems.append(f"{path} does not parse: {e.msg} (line {e.lineno})")
    if not problems:
        for path in new:
            if is_test_path(path) and (gone := _test_names(job.editable[path]) - _test_names(new[path])):
                problems.append(f"patch removes tests from {path}: {sorted(gone)}")
        problems += conformance(new[job.site.file], job.site, job.tier, job.params, job.constants,
                                job.system_name, job.spec)  # fmt: skip
    if problems:
        return f"patch rejected (adapter spec or patch rules): {'; '.join(problems)}", {}
    changed = [p for p in new if new[p] != job.before[p]]
    for p in changed:
        write(p, new[p])
    box.tidy(changed, format_code=formatted)
    rewrapped = []
    for p in changed:  # after formatting, whose indentation is final: wrap over-long strings
        text = (repo / p).read_text()
        if (wrapped := _wrap(job, p, text)) != text:
            (repo / p).write_text(wrapped)
            rewrapped.append(p)
    if formatted and rewrapped:  # the pieces no longer fit on one line, so formatting keeps them split
        box.tidy(rewrapped, format_code=True)
    result: CheckResult = box.check()
    patched = {p: (repo / p).read_text() for p in changed}
    for p in changed:  # every site is checked against the same base
        (repo / p).write_text(job.before[p])
    if result.ok:
        return None, patched
    return f"repo checks failed on the patch ({result.mode}): {_summary(result.output)}", {}


def compose_patches(
    repo_dir: Path,
    sites: list[CallSite],
    verdicts: list[Verdict],
    adapters: dict[str, Adapter],
    llm: LLM,
    work_dir: Path,
    *,
    config: RunConfig,
    sandbox_factory: Callable[[Path], Sandbox] = Sandbox,
    build_request: BuildRequest = build_request,
) -> PatchResult:
    repo = Path(work_dir) / "repo"
    shutil.rmtree(repo, ignore_errors=True)
    warnings = copy_repo(repo_dir, repo)
    width = line_length(repo)
    originals: dict[str, str | None] = {}

    def read(p: str) -> str | None:
        f = repo / p
        return f.read_text() if f.is_file() else None

    def write(p: str, text: str) -> None:
        originals.setdefault(p, read(p))
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(text)

    by_id = {s.id: s for s in sites}
    adapters = {k: a.model_copy(deep=True) for k, a in adapters.items()}
    decided: dict[str, Verdict] = {}
    diffs: dict[str, str] = {}
    site_files: dict[str, list[str]] = {}
    box = sandbox_factory(repo)
    to_migrate = sorted((v for v in verdicts if v.decision == "MIGRATE"), key=lambda v: v.site_id)

    def downgrade(v: Verdict, reason: str) -> None:
        decided[v.site_id] = v.model_copy(update={"decision": "MANUAL_REVIEW", "reason": reason})

    def stop(reason: str) -> PatchResult:  # nothing can be validated: no site is patched, no file changes
        for v in to_migrate:
            downgrade(v, reason)
        verdicts_out = [decided.get(v.site_id, v) for v in verdicts]
        return PatchResult(files={}, diffs={}, site_files={}, verdicts=verdicts_out, adapters=adapters,
                           sandbox_mode=box.mode, warnings=warnings)  # fmt: skip

    if box.mode == "none":
        return stop(NO_SANDBOX)
    untouched = box.check()
    if not untouched.ok:
        return stop(
            f"repo checks fail on the untouched repo ({untouched.mode}): {_summary(untouched.output)}"
        )

    client_at = client_location(repo, [by_id[v.site_id].file for v in to_migrate if v.site_id in by_id])
    if client_at is None:
        return stop(
            "no package to hold Offramp's client module: the migrated call sites sit in the repository root, "
            "which is itself a Python package (it has __init__.py)"
        )
    client = client_module()
    write(client_at[0], client)
    if client_at == ROOT_CLIENT and read("offramp/__init__.py") is None:
        write("offramp/__init__.py", "")
    kept = [by_id[v.site_id] for v in verdicts if v.decision == "KEEP_CLOSED" and v.site_id in by_id]
    for site in sorted(kept, key=lambda s: (s.file, -s.lines[0])):  # bottom-up: line numbers stay valid
        v = next(v for v in verdicts if v.site_id == site.id)
        before = read(site.file)
        write(site.file, _insert_comment(before, site, keep_comment(site, v, config)))
        diffs[site.id] = _diff({site.file: before}, {site.file: read(site.file)})
        site_files[site.id] = [site.file]

    base = box.check()
    if not base.ok:
        return stop(
            f"repo checks fail after adding Offramp's client module ({client_at[0]}) and kept-site comments "
            f"({base.mode}): {_summary(base.output)}"
        )
    formatted = box.formatted()  # the repo keeps `ruff format` clean: so do the patches

    versions: dict[str, dict[str, str]] = {}  # site -> its patched files, green alone
    alone: dict[str, bool] = {}
    for v in to_migrate:
        site, adapter = by_id.get(v.site_id), adapters.get(v.site_id)
        alone[v.site_id] = False
        if site is None or adapter is None or v.tier is None:
            downgrade(v, "no call site, adapter or tier to build the patch from")
            continue
        tier = Tier(v.tier)
        try:
            params = call_params(build_request(adapter.spec, _probe_case(site), tier))
        except UnresolvedCallSite as e:
            downgrade(v, f"patch not composed: {e}")
            continue
        if params.get("response_format") == {"type": "json_object"}:
            why = "evaluated as json_object plus a validation repair retry, which a patch cannot reproduce"
            downgrade(v, f"patch not composed: the adapter was {why}")
            continue
        prepared, constants, system_name = _prepare(read(site.file), site, adapter.spec, params, width)
        context = _context(repo, site, client_at[0])
        editable = {p: (prepared if p == site.file else read(p)) for p in context}
        before = {p: read(p) for p in editable}
        job = _Job(site, tier, adapter.spec, params, constants, system_name, editable, before,
                   client_module=client_at[1], width=width)  # fmt: skip
        prompt = _prompt(
            site, tier, adapter.spec, params, constants, system_name, editable, client, client_at, width
        )
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        failure, patched, tried = None, {}, []
        for attempt in range(1 + REPAIRS):
            try:
                res = llm.chat(
                    role=ROLE,
                    tier=Tier.SUPER,
                    messages=messages,
                    max_tokens=REPAIR_MAX_TOKENS if attempt else MAX_TOKENS,
                    temperature=0.0,
                    reasoning="off",  # GC25; recorded with thinking on, 2 of 5 replies hit the token cap
                )
            except BudgetExceeded as e:
                failure = f"patch not composed: {e}"
                break
            edits = parse_edits(res.output.text or "").edits
            if edits in tried:  # a patch already tried (A-B-A too): the sandbox would say the same again
                failure = f"repair produced no change; last failure: {failure}"
                break
            tried.append(edits)
            failure, patched = _attempt(job, res.output.text or "", repo, box, formatted, write)
            if failure is None:
                break
            messages = [
                *messages,
                {"role": "assistant", "content": res.output.text or ""},
                {"role": "user", "content": REPAIR.format(failure=failure)},
            ]
        if failure is not None:
            downgrade(v, failure)
            continue
        alone[v.site_id] = True
        versions[v.site_id] = patched
        diffs[v.site_id] = adapters[v.site_id].patch = _diff(job.before, patched)
        site_files[v.site_id] = sorted(patched)

    # all green-alone patches together, in site order (disk is at the base again)
    base_files = {p: read(p) for files_ in versions.values() for p in files_}
    by_verdict = {v.site_id: v for v in to_migrate}

    def merge(sids: list[str]) -> tuple[dict[str, str], list[str]]:  # (merged files, sites merged)
        merged: dict[str, str] = {}
        taken = []
        for sid in sids:
            trial = dict(merged)
            for path, text in versions[sid].items():
                if (m := merge_onto(base_files[path], trial.get(path, base_files[path]), text)) is None:
                    downgrade(
                        by_verdict[sid], f"green alone, but its patch conflicts with another site's in {path}"
                    )
                    break
                trial[path] = m
            else:
                merged, taken = trial, [*taken, sid]
        return merged, taken

    merged, combined = merge([v.site_id for v in to_migrate if v.site_id in versions])
    for path, text in merged.items():
        write(path, text)
    if len(combined) > 1 and not (together := box.check()).ok:
        for path in merged:  # one at a time: keep each site that the others can live with
            write(path, base_files[path])
        kept: list[str] = []
        for sid in combined:
            trial, _ = merge([*kept, sid])  # a superset of the kept sites' files
            for path, text in trial.items():
                write(path, text)
            result = box.check()
            if result.ok:
                kept.append(sid)
                continue
            current, _ = merge(kept)
            for path in trial:
                write(path, current.get(path, base_files[path]))
            others = ", ".join(kept) or "the base"
            why = _summary(result.output or together.output)
            downgrade(by_verdict[sid], f"green alone, but not together with {others} ({result.mode}): {why}")
        combined = kept
    for sid in versions:
        if sid not in combined:
            diffs.pop(sid, None)
            site_files.pop(sid, None)
            adapters[sid].patch = None
    migrated = len(combined)

    files = {p: text for p in originals if (text := read(p)) is not None and text != originals[p]}
    if not migrated:  # nothing uses the client module
        files.pop(client_at[0], None)
        if originals.get("offramp/__init__.py", "") is None:  # added for the root client only
            files.pop("offramp/__init__.py", None)
    return PatchResult(
        files=files,
        diffs=diffs,
        site_files=site_files,
        verdicts=[decided.get(v.site_id, v) for v in verdicts],
        adapters=adapters,
        sandbox_mode=box.mode,
        client_path=client_at[0] if migrated else None,
        alone=alone,
        combined=combined,
        warnings=warnings,
    )
