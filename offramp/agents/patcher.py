"""Patch Composer (PRD §6.8): each MIGRATE site's winning adapter becomes a code change; the repo stays green.

Per run, on a sandbox copy of the target repo:
1. add `offramp/client.py` (Token Factory client + tier constants) and mark KEEP_CLOSED calls with
   `# offramp: kept on <model> — parity <p>% < <threshold>%`; the repo's own ruff + pytest must pass
   (baseline);
2. for each MIGRATE site, in order: the request parameters come from `build_request` (Task 6, the function the
   eval runner used), so what ships is what was evaluated. Large literals (tools, response_format, few-shot,
   a changed system prompt) are written into the file deterministically; Nemotron 3 Super (role `patcher`)
   rewires the call, its reply handling, callers and provider-pinning tests as search/replace edits;
3. the edits are checked statically (allowed files, no removed tests, the call carries exactly the adapter's
   parameters), then by the repo's ruff + pytest in the sandbox. Any failure: the site becomes MANUAL_REVIEW
   and its patch is dropped; later sites build on the last green state.
"""

import ast
import difflib
import json
import re
import shutil
import string
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from offramp.agents.pr_writer import TIER_NAMES, threshold
from offramp.sandbox import CheckResult, Sandbox
from offramp.schemas import Adapter, AdapterSpec, CallSite, EvalCase, RunConfig, Tier, Verdict
from offramp.tf.budget import BudgetExceeded
from offramp.tf.client import LLM, _merge, plan_structured
from offramp.tf.tiers import get_tier, reasoning_params

ROLE = "patcher"
MAX_TOKENS = 8000
REPAIRS = 1  # one more try per site, shown its failure (like Global Constraint 6's single repair retry)
REPAIR = """Your edits were not accepted: {failure}
The files are unchanged from my first message. Reply with a complete new set of edit blocks (all of them, \
not only the fix) and the NOTES line."""
CLIENT_PATH = "offramp/client.py"
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
    "token_factory_client": "from offramp.client import token_factory_client",
    "OpenAI": "from openai import OpenAI",
    "json": "import json",
}
_BLOCK = re.compile(r"^FILE: *`?([^`\s]+)`?[ \t]*\n<{7} SEARCH\n(.*?)\n={7}\n(.*?)>{7} REPLACE", re.S | re.M)


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
    edits = [
        Edit(path=p, search=s, replace=r[:-1] if r.endswith("\n") else r) for p, s, r in _BLOCK.findall(text)
    ]
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
    sandbox_mode: str  # "docker" (network off) | "local" (not isolated)


def _default_build_request() -> BuildRequest:
    from offramp.agents.translate import build_request  # Task 6

    return build_request


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


def add_imports(source: str) -> str:
    """Import AUTO_IMPORTS names the module uses but never binds (after its last top-level import)."""
    tree = ast.parse(source)
    bound = set()
    for n in tree.body:
        if isinstance(n, ast.Import | ast.ImportFrom):
            bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(n.name)
        elif isinstance(n, ast.Assign):
            bound |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = [line for name, line in AUTO_IMPORTS.items() if name in used - bound]
    if not missing:
        return source
    imports = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    docstring = (
        tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)
    )
    at = imports[-1].end_lineno if imports else (tree.body[0].end_lineno if docstring else 0)
    lines = source.splitlines(keepends=True)
    return "".join(lines[:at]) + "".join(f"{m}\n" for m in missing) + "".join(lines[at:])


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


def _is_test(rel: Path) -> bool:
    return (
        any(p in ("tests", "test") for p in rel.parts[:-1])
        or rel.name.startswith("test_")
        or rel.name.endswith("_test.py")
        or rel.name == "conftest.py"
    )


def _context(repo: Path, site: CallSite) -> list[str]:
    """Editable files for one site: its module, callers of its function, tests touching them, test helpers."""
    files = sorted(
        p.relative_to(repo)
        for p in repo.rglob("*.py")
        if not SKIP_PARTS & set(p.relative_to(repo).parts[:-1])
    )
    text = {f: (repo / f).read_text() for f in files}
    own = Path(site.file)
    callers = [
        f
        for f in files
        if f != own
        and not _is_test(f)
        and site.function
        and re.search(rf"\b{re.escape(site.function)}\b", text[f])
    ]
    words = {own.stem, *(f.stem for f in callers), *([site.function] if site.function else [])}
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(words))) + r")\b")
    tests = [f for f in files if _is_test(f) and f.name != "conftest.py" and pattern.search(text[f])]
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
`token_factory_client()` (from offramp.client import token_factory_client) instead. Tests that replaced \
that constructor in such a module with a fake must replace `token_factory_client` there instead: check the \
editable tests for such patches, since one left on the old constructor makes the code call the real client.
5. Tests: change only what pins the old provider - fakes of the old SDK and assertions on the old request \
(model, old SDK arguments, message layout). Use the repo's existing OpenAI-style fake helper; if it cannot \
yet return what the new code reads (e.g. tool calls), extend that helper the way it already builds real \
openai response objects, instead of writing ad-hoc fake classes inside tests. Never delete or rename a \
test, never change an expected return value or a parsing check.
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
        f"Repo ruff line length: {width}.",
        "",
        "## Files",
        f"### {CLIENT_PATH} (read-only)",
        "```python",
        client.rstrip("\n"),
        "```",
    ]
    for path, text in files.items():
        lines += [f"### {path} (editable)", "```python", text.rstrip("\n"), "```"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- composer


def _line_length(repo: Path) -> int:
    try:
        cfg = tomllib.loads((repo / "pyproject.toml").read_text())
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return 88
    return int(cfg.get("tool", {}).get("ruff", {}).get("line-length", 88))


def _summary(output: str) -> str:
    """The lines that say what failed (pytest FAILED/ERROR, ruff findings), else the output's tail."""
    keep = [ln for ln in output.splitlines() if re.match(r"(FAILED|ERROR) |\S+\.py:\d+:\d+: |\$ |E\s{3}", ln)]
    return (" | ".join(keep[:8]) if keep else output.strip()[-500:])[:1200]


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
    before: dict[str, str]  # on disk before the site's patch


def _attempt(job: _Job, reply: str, repo: Path, box: Sandbox, formatted: bool, write) -> str | None:
    """Apply one patcher reply. None when the patch is in place and the repo is green; otherwise why not, with
    the files back as they were."""
    patch = parse_edits(reply)
    if not patch.edits:
        return "the reply holds no edit blocks"
    new = dict(job.editable)
    problems = _apply(patch.edits, new)
    for path in new if not problems else []:
        try:
            new[path] = add_imports(new[path]) if new[path] != job.editable[path] else new[path]
        except SyntaxError as e:
            problems.append(f"{path} does not parse: {e.msg} (line {e.lineno})")
    if not problems:
        for path in new:
            if _is_test(Path(path)) and (gone := _test_names(job.editable[path]) - _test_names(new[path])):
                problems.append(f"patch removes tests from {path}: {sorted(gone)}")
        problems += conformance(new[job.site.file], job.site, job.tier, job.params, job.constants,
                                job.system_name, job.spec)  # fmt: skip
    if problems:
        return f"patch rejected (adapter spec or patch rules): {'; '.join(problems)}"
    changed = [p for p in new if new[p] != job.before[p]]
    for p in changed:
        write(p, new[p])
    box.tidy(changed, format_code=formatted)
    result: CheckResult = box.check()
    if result.ok:
        return None
    for p in changed:
        (repo / p).write_text(job.before[p])
    return f"repo checks failed on the patch ({result.mode}): {_summary(result.output)}"


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
    build_request: BuildRequest | None = None,
) -> PatchResult:
    build = build_request or _default_build_request()
    repo = Path(work_dir) / "repo"
    shutil.rmtree(repo, ignore_errors=True)
    shutil.copytree(repo_dir, repo, ignore=IGNORE)
    width = _line_length(repo)
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

    def downgrade(v: Verdict, reason: str) -> None:
        decided[v.site_id] = v.model_copy(update={"decision": "MANUAL_REVIEW", "reason": reason})

    client = client_module()
    write(CLIENT_PATH, client)
    if read("offramp/__init__.py") is None:
        write("offramp/__init__.py", "")
    kept = [by_id[v.site_id] for v in verdicts if v.decision == "KEEP_CLOSED" and v.site_id in by_id]
    for site in sorted(kept, key=lambda s: (s.file, -s.lines[0])):  # bottom-up: line numbers stay valid
        v = next(v for v in verdicts if v.site_id == site.id)
        before = read(site.file)
        write(site.file, _insert_comment(before, site, keep_comment(site, v, config)))
        diffs[site.id] = _diff({site.file: before}, {site.file: read(site.file)})
        site_files[site.id] = [site.file]

    base = box.check()
    formatted = base.ok and box.formatted()  # the repo keeps `ruff format` clean: so do the patches
    if not base.ok:
        reason = f"repo checks fail before any call-site patch ({base.mode}): {_summary(base.output)}"
        for v in verdicts:
            if v.decision == "MIGRATE":
                downgrade(v, reason)
        return PatchResult(
            files={},
            diffs={},
            site_files={},
            verdicts=[decided.get(v.site_id, v) for v in verdicts],
            adapters=adapters,
            sandbox_mode=box.mode,
        )

    migrated = 0
    for v in sorted((v for v in verdicts if v.decision == "MIGRATE"), key=lambda v: v.site_id):
        site, adapter = by_id.get(v.site_id), adapters.get(v.site_id)
        if site is None or adapter is None or v.tier is None:
            downgrade(v, "no call site, adapter or tier to build the patch from")
            continue
        tier = Tier(v.tier)
        try:
            params = call_params(build(adapter.spec, _probe_case(site), tier))
            prepared, constants, system_name = _prepare(read(site.file), site, adapter.spec, params, width)
        except ValueError as e:
            downgrade(v, f"patch not composed: {e}")
            continue
        editable = {p: (prepared if p == site.file else read(p)) for p in _context(repo, site)}
        job = _Job(site, tier, adapter.spec, params, constants, system_name, editable,
                   before={p: read(p) for p in editable})  # fmt: skip
        prompt = _prompt(site, tier, adapter.spec, params, constants, system_name, editable, client, width)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        failure = None
        for _ in range(1 + REPAIRS):
            try:
                res = llm.chat(
                    role=ROLE,
                    tier=Tier.SUPER,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                    reasoning="off",  # GC25; recorded with thinking on, 2 of 5 replies hit the token cap
                )
            except BudgetExceeded as e:
                failure = f"patch not composed: {e}"
                break
            failure = _attempt(job, res.output.text or "", repo, box, formatted, write)
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
        changed = sorted(p for p in editable if read(p) != job.before[p])
        diffs[v.site_id] = adapters[v.site_id].patch = _diff(job.before, {p: read(p) for p in changed})
        site_files[v.site_id] = changed
        migrated += 1

    files = {p: text for p in originals if (text := read(p)) is not None and text != originals[p]}
    if not migrated:  # nothing uses the client module
        files.pop(CLIENT_PATH, None)
        if originals.get("offramp/__init__.py") is None:
            files.pop("offramp/__init__.py", None)
    return PatchResult(
        files=files,
        diffs=diffs,
        site_files=site_files,
        verdicts=[decided.get(v.site_id, v) for v in verdicts],
        adapters=adapters,
        sandbox_mode=box.mode,
    )
