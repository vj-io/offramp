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
import textwrap
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
Reply with SEARCH/REPLACE edit blocks only, no analysis or explanation, then the NOTES line. Each file you \
name is taken as shown in my first message with your new blocks applied, so give every block that file \
needs, not only the fix; a file you do not name keeps your previous edits. Begin with `FILE:`."""
ROOT_CLIENT = ("offramp/client.py", "offramp.client")  # when the sites are not inside one package
TEMPLATE = Path(__file__).parents[1] / "templates" / "client.py.tmpl"
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
    "ChatCompletion": "from openai.types.chat import ChatCompletion",
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


SCHEMA_PARAMS = {"tools": "NEMOTRON_TOOLS", "response_format": "NEMOTRON_RESPONSE_FORMAT"}  # long: constants


def _prepare(source: str, site: CallSite, spec: AdapterSpec, params: dict[str, Any], width: int):
    """Write the evaluated request into the site's module, mechanically. Returns (source, constants,
    system_name, rewritten, fixed): constants maps "few_shot", "prefix", "suffix", "tools",
    "response_format" -> module constant name; system_name holds the system prompt; rewritten: the call
    already sends the request as literal keyword arguments; fixed: the reply is read the OpenAI way too, so
    the module is final."""
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
        ("prefix", "NEMOTRON_USER_PREFIX", spec.user_prefix or None),
        ("suffix", "NEMOTRON_USER_SUFFIX", spec.user_suffix or None),
        *((key, base, params.get(key)) for key, base in SCHEMA_PARAMS.items()),
    ]:
        if value is not None:
            constants[key] = name(base)
            blocks.append(f"{constants[key]} = {py_literal(value, 0, width, len(constants[key]) + 3)}")
    if blocks:
        top = _top_node(tree, site)
        at = min([top.lineno, *(d.lineno for d in getattr(top, "decorator_list", []))]) - 1
        lines = source.splitlines(keepends=True)
        source = "".join(lines[:at]) + "\n\n".join(blocks) + "\n\n\n" + "".join(lines[at:])
    rewritten = _rewrite_call(source, site, constants, system_name, params, width)
    if rewritten is None:
        return source, constants, system_name, False, False
    mapped = _map_reply(rewritten, site)
    return (mapped or rewritten), constants, system_name, True, mapped is not None


def _offset(lines: list[str], lineno: int, col: int) -> int:
    """Character offset of an AST (line, UTF-8 byte column) position."""
    return sum(map(len, lines[: lineno - 1])) + len(lines[lineno - 1].encode()[:col].decode())


def _role(node: ast.expr) -> str | None:
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values, strict=True):
            if isinstance(k, ast.Constant) and k.value == "role" and isinstance(v, ast.Constant):
                return v.value
    return None


def _rewrite_call(
    source: str,
    site: CallSite,
    constants: dict[str, str],
    system_name: str | None,
    params: dict[str, Any],
    width: int,
) -> str | None:
    """The site's call rewritten to `<client>.chat.completions.create(model=..., <every evaluated parameter as
    a literal keyword argument, schemas by constant name>, messages=[system, *few-shot, ...the call's own
    messages, the last user content wrapped in prefix/suffix])`, from an OpenAI `chat.completions.create` or
    an Anthropic `messages.create` (its `system=` becomes the first message). None when the call is not a
    single call with a literal `messages=[...]` list: the model then writes it."""
    tree = ast.parse(source)
    scope = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                  and n.name == site.function), tree)  # fmt: skip
    kinds = ("chat.completions.create", "messages.create")
    calls = [c for c in ast.walk(scope) if isinstance(c, ast.Call) and _call_name(c.func).endswith(kinds)]
    if len(calls) != 1 or calls[0].args:
        return None
    call = calls[0]
    kw = {k.arg: k.value for k in call.keywords}
    messages = kw.get("messages")
    if None in kw or not isinstance(messages, ast.List):
        return None
    anthropic = not _call_name(call.func).endswith("chat.completions.create")
    receiver = call.func.value.value if anthropic else call.func.value.value.value

    def seg(node: ast.AST) -> str:
        return ast.get_source_segment(source, node)

    items = list(messages.elts)
    head = []
    if items and _role(items[0]) == "system":
        old = items.pop(0)
        head.append(f'{{"role": "system", "content": {system_name}}}' if system_name else seg(old))
    elif system_name:
        head.append(f'{{"role": "system", "content": {system_name}}}')
    elif anthropic and "system" in kw:
        head.append(f'{{"role": "system", "content": {seg(kw["system"])}}}')
    if "few_shot" in constants:
        head.append(f"*{constants['few_shot']}")
    body = [seg(e) for e in items]
    if "prefix" in constants or "suffix" in constants:
        users = [k for k, e in enumerate(items) if _role(e) == "user"]
        if not users:
            return None
        last = items[users[-1]]
        keys = [k.value if isinstance(k, ast.Constant) else None for k in last.keys]
        if sorted(map(str, keys)) != ["content", "role"]:
            return None
        content = f"({seg(last.values[keys.index('content')])})"
        if "prefix" in constants:
            content = f"{constants['prefix']} + {content}"
        if "suffix" in constants:
            content = f"{content} + {constants['suffix']}"
        body[users[-1]] = f'{{"role": "user", "content": {content}}}'
    lines = source.splitlines(keepends=True)
    start = _offset(lines, call.lineno, call.col_offset)
    end = _offset(lines, call.end_lineno, call.end_col_offset)
    indent = re.match(r"[ \t]*", lines[call.lineno - 1]).group(0)
    i1, i2 = indent + "    ", indent + "        "
    entries = "".join(f"{i2}{x},\n" for x in [*head, *body])
    args = "".join(
        f"{i1}{k}={constants.get(k) or py_literal(v, len(i1), width, len(i1) + len(k) + 1)},\n"
        for k, v in params.items()
    )
    text = f"{seg(receiver)}.chat.completions.create(\n{args}{i1}messages=[\n{entries}{i1}],\n{indent})"
    rewritten = source[:start] + text + source[end:]
    try:
        ast.parse(rewritten)
    except SyntaxError:
        return None
    return rewritten


def _calls(fn: ast.AST) -> list[ast.Call]:
    kinds = ("chat.completions.create", "messages.create")
    return [c for c in ast.walk(fn) if isinstance(c, ast.Call) and _call_name(c.func).endswith(kinds)]


def _function(tree: ast.Module, site: CallSite) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                 and n.name == site.function), None)  # fmt: skip


def _is_content(node: ast.AST, reply: str) -> bool:  # `<reply>.content`
    return (isinstance(node, ast.Attribute) and node.attr == "content" and isinstance(node.value, ast.Name)
            and node.value.id == reply)  # fmt: skip


def _is_type(node: ast.AST, var: str, kind: str) -> bool:  # `<var>.type == "<kind>"`
    return (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)
            and isinstance(node.left, ast.Attribute) and node.left.attr == "type"
            and isinstance(node.left.value, ast.Name) and node.left.value.id == var
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value == kind)  # fmt: skip


def _text_read(node: ast.AST, reply: str) -> bool:
    """`"".join(b.text for b in <reply>.content if b.type == "text")` or `<reply>.content[0].text`."""
    if (isinstance(node, ast.Attribute) and node.attr == "text" and isinstance(node.value, ast.Subscript)
            and _is_content(node.value.value, reply) and isinstance(node.value.slice, ast.Constant)
            and node.value.slice.value == 0):  # fmt: skip
        return True
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join"
            and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str)
            and len(node.args) == 1 and not node.keywords
            and isinstance(node.args[0], ast.GeneratorExp | ast.ListComp)):  # fmt: skip
        return False
    gen = node.args[0]
    if len(gen.generators) != 1 or not isinstance(gen.generators[0].target, ast.Name):
        return False
    var = gen.generators[0].target.id
    return (_is_content(gen.generators[0].iter, reply) and isinstance(gen.elt, ast.Attribute)
            and gen.elt.attr == "text" and isinstance(gen.elt.value, ast.Name) and gen.elt.value.id == var
            and all(_is_type(c, var, "text") for c in gen.generators[0].ifs))  # fmt: skip


def _map_reply(source: str, site: CallSite) -> str | None:
    """The migrated function reads the reply the OpenAI way, by code. An OpenAI site already does. For an
    Anthropic site: `"".join(b.text for b in r.content if b.type == "text")` and `r.content[0].text` become
    `(r.choices[0].message.content or "")`; a `for b in r.content:` loop over tool_use blocks walks
    `r.choices[0].message.tool_calls or []` (`b.type == "tool_use"` -> `"function"`, `b.name` ->
    `b.function.name`, `b.input` -> `json.loads(b.function.arguments)`); an `Anthropic` annotation of a
    parameter becomes `OpenAI`. None when the reply is used any other way: the model maps it then."""
    if site.sdk == "openai":
        return source
    tree = ast.parse(source)
    fn = _function(tree, site)
    calls = _calls(fn) if fn else []
    if len(calls) != 1:
        return None
    holder = next((n for n in ast.walk(fn) if isinstance(n, ast.Assign) and n.value is calls[0]), None)
    if holder is None or len(holder.targets) != 1 or not isinstance(holder.targets[0], ast.Name):
        return None
    reply = holder.targets[0].id
    lines = source.splitlines(keepends=True)
    spans: list[tuple[int, int, str]] = []

    def put(node: ast.AST, text: str) -> None:
        start = _offset(lines, node.lineno, node.col_offset)
        spans.append((start, _offset(lines, node.end_lineno, node.end_col_offset), text))

    parents = {child: node for node in ast.walk(fn) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(fn):
        if _text_read(node, reply):
            text = f'{reply}.choices[0].message.content or ""'
            bare = isinstance(parents.get(node), ast.Assign | ast.Return | ast.Call | ast.keyword)
            put(node, text if bare else f"({text})")
        elif (
            isinstance(node, ast.For) and _is_content(node.iter, reply) and isinstance(node.target, ast.Name)
        ):
            block = node.target.id
            put(node.iter, f"{reply}.choices[0].message.tool_calls or []")
            for use in (n for part in [*node.body, *node.orelse] for n in ast.walk(part)):
                if not (isinstance(use, ast.Name) and use.id == block):
                    continue
                attr = parents[use]
                if not isinstance(attr, ast.Attribute) or attr.attr not in ("type", "name", "input", "id"):
                    return None
                if attr.attr == "type":
                    if not _is_type(parents[attr], block, "tool_use"):
                        return None
                    put(parents[attr].comparators[0], '"function"')
                elif attr.attr != "id":
                    new = "function.name" if attr.attr == "name" else None
                    put(attr, f"{block}.{new}" if new else f"json.loads({block}.function.arguments)")
    for arg in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]:
        if isinstance(arg.annotation, ast.Name) and arg.annotation.id == "Anthropic":
            put(arg.annotation, "OpenAI")
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:], strict=False)):
        return None
    for start, end, text in reversed(spans):
        source = source[:start] + text + source[end:]
    fn = _function(ast.parse(source), site)  # every use of the reply is now an OpenAI one
    parents = {child: node for node in ast.walk(fn) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == reply and isinstance(node.ctx, ast.Load):
            parent = parents[node]
            if not (isinstance(parent, ast.Attribute) and parent.attr in ("choices", "id", "model")):
                return None
    return source


def _top_node(tree: ast.Module, site: CallSite) -> ast.stmt:
    """The top-level statement holding the site: by function name (lines shift as files are edited)."""
    for n in tree.body:
        if any(
            isinstance(f, ast.FunctionDef | ast.AsyncFunctionDef) and f.name == site.function
            for f in ast.walk(n)
        ):
            return n
    return next((n for n in tree.body if n.lineno <= site.lines[0] <= n.end_lineno), tree.body[-1])


def _near(text: str, search: str) -> tuple[int, int] | None:
    """Where a long SEARCH block that the model copied with a slip sits: the one run of as many lines whose
    first and last lines match exactly and at least 90% of whose lines do. None when it is short (5 lines or
    fewer), or no run or more than one run qualifies."""
    want = search.split("\n")
    have = text.split("\n")
    n = len(want)
    if n <= 5:
        return None
    runs = [i for i in range(len(have) - n + 1) if have[i] == want[0] and have[i + n - 1] == want[-1]
            and sum(a == b for a, b in zip(have[i : i + n], want, strict=True)) >= 0.9 * n]  # fmt: skip
    if len(runs) != 1:
        return None
    start = sum(len(line) + 1 for line in have[: runs[0]])
    return start, start + len("\n".join(have[runs[0] : runs[0] + n]))


def _apply(edits: list[Edit], files: dict[str, str], locked: frozenset[str] = frozenset()) -> list[str]:
    """Apply the edit blocks in order; `locked` files are shown but read-only."""
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
        if e.path in locked:
            return [f"{e.path} is migrated by Offramp and read-only: edit only the tests and fakes"]
        near = _near(files[e.path], e.search) if n == 0 and e.search not in shown[e.path] else None
        if near:  # a slip in copying a long block: it still names one place
            files[e.path] = files[e.path][: near[0]] + e.replace + files[e.path][near[1] :]
            continue
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


OLD_CLIENTS = ("OpenAI", "Anthropic")


def rewrite_callers(source: str, function: str, client: str) -> str:
    """In a caller module: an `OpenAI()` / `Anthropic()` built right in a call of the migrated function
    becomes `token_factory_client()` (imported from `client`). Calls of other functions keep their client."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    spans = []
    for call in ast.walk(tree):
        if (
            not isinstance(call, ast.Call)
            or getattr(call.func, "id", getattr(call.func, "attr", None)) != function
        ):
            continue
        for arg in [*call.args, *(k.value for k in call.keywords)]:
            if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id in OLD_CLIENTS
                    and not arg.args and not arg.keywords):  # fmt: skip
                spans.append(
                    (
                        _offset(lines, arg.lineno, arg.col_offset),
                        _offset(lines, arg.end_lineno, arg.end_col_offset),
                    )
                )
    if not spans:
        return source
    for start, end in sorted(spans, reverse=True):
        source = source[:start] + "token_factory_client()" + source[end:]
    return add_imports(source, client)


def mirror_fakes(source: str, modules: set[str], old: str = "OpenAI") -> str:
    """In a test: `monkeypatch.setattr(<module>, "<old>", <fake>)` for a module whose code now builds
    `token_factory_client()` gets a twin for `token_factory_client`, with the same fake. An OpenAI fake
    fits the Token Factory client as it is; an Anthropic fake does once `convert_fakes` taught it the
    OpenAI call."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    setattrs = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and len(c.args) == 3
                and ast.unparse(c.func).endswith("monkeypatch.setattr")]  # fmt: skip
    present = {
        (ast.unparse(c.args[0]), c.args[1].value) for c in setattrs if isinstance(c.args[1], ast.Constant)
    }
    inserts = []
    for stmt in ast.walk(tree):
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            continue
        call = stmt.value
        if (ast.unparse(call.func).endswith("monkeypatch.setattr") and len(call.args) == 3
                and isinstance(call.args[0], ast.Name) and call.args[0].id in modules
                and isinstance(call.args[1], ast.Constant) and call.args[1].value == old
                and (call.args[0].id, "token_factory_client") not in present):  # fmt: skip
            head = ast.get_source_segment(source, call.func)
            fake = ast.get_source_segment(source, call.args[2])
            indent = re.match(r"[ \t]*", lines[stmt.lineno - 1]).group(0)
            inserts.append(
                (stmt.end_lineno, f'{indent}{head}({call.args[0].id}, "token_factory_client", {fake})\n')
            )
    for row, text in sorted(inserts, reverse=True):
        lines.insert(row, text)
    return "".join(lines)


FAKE_HELPER = '''def chat_completion_from_blocks(blocks) -> ChatCompletion:
    """Anthropic content blocks as the OpenAI chat completion that a call site
    migrated by Offramp reads: text blocks joined, tool_use blocks as tool calls."""
    text = "".join(b.text for b in blocks if b.type == "text")
    calls = [
        {
            "id": b.id,
            "type": "function",
            "function": {"name": b.name, "arguments": json.dumps(b.input)},
        }
        for b in blocks
        if b.type == "tool_use"
    ]
    message = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = calls
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if calls else "stop",
                    "message": message,
                }
            ],
        }
    )
'''


def _fake_messages_method(cls: ast.ClassDef) -> tuple[ast.stmt, ast.FunctionDef, ast.expr] | None:
    """(the `self.messages = SimpleNamespace(create=self.<m>)` statement, method <m>, the expression its
    `return Message(..., content=<expr>)` answers with) for an Anthropic fake class, else None."""
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    init = methods.get("__init__")
    stmts = [n for n in ast.walk(init) if isinstance(n, ast.Assign)] if init else []
    if any(ast.unparse(t) == "self.chat" for n in stmts for t in n.targets):
        return None  # already answers the OpenAI call
    for n in stmts:
        if ([ast.unparse(t) for t in n.targets] == ["self.messages"] and isinstance(n.value, ast.Call)
                and len(n.value.keywords) == 1 and n.value.keywords[0].arg == "create"
                and ast.unparse(n.value.keywords[0].value).startswith("self.")):  # fmt: skip
            method = methods.get(ast.unparse(n.value.keywords[0].value).removeprefix("self."))
            last = method.body[-1] if method else None
            if (isinstance(last, ast.Return) and isinstance(last.value, ast.Call)
                    and ast.unparse(last.value.func).split(".")[-1] == "Message"):  # fmt: skip
                content = next((k.value for k in last.value.keywords if k.arg == "content"), None)
                if content is not None:
                    return n, method, content
    return None


def convert_fakes(source: str, client: str) -> str:
    """In a test helper: an Anthropic fake (`self.messages = SimpleNamespace(create=self.<m>)`, <m> returning
    `Message(..., content=<blocks>)`) also answers `chat.completions.create`, with the same canned blocks as
    an OpenAI chat completion (`chat_completion_from_blocks`, added once to the module). <m>'s other steps
    are kept, an argument check against `Messages.create` checking against `Completions.create` instead."""
    tree = ast.parse(source)
    found = [
        (cls, hit)
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and (hit := _fake_messages_method(cls))
    ]
    if not found:
        return source
    lines = source.splitlines(keepends=True)
    bound = _bound(tree)
    inserts: list[tuple[int, str]] = []  # (0-based line index to insert before, text)
    for _, (assign, method, content) in found:
        pad = re.match(r"[ \t]*", lines[assign.lineno - 1]).group(0)
        inserts.append(
            (
                assign.end_lineno,
                f"{pad}self.chat = SimpleNamespace(\n"
                f"{pad}    completions=SimpleNamespace(create=self._chat_completion)\n{pad})\n",
            )
        )
        indent = " " * method.col_offset
        body = " " * method.body[0].col_offset
        steps = []
        for stmt in method.body[:-1]:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # its docstring
            text = "".join(lines[stmt.lineno - 1 : stmt.end_lineno])
            if re.search(r"\bMessages\b", text):
                if "Completions" not in bound:
                    continue
                text = re.sub(r"\bMessages\b", "Completions", text)
            steps.append(text)
        expr = ast.get_source_segment(source, content)
        inserts.append(
            (
                method.end_lineno,
                (
                    f"\n{indent}def _chat_completion(self, **kwargs):\n"
                    f'{body}"""The same canned reply, as an OpenAI chat completion (Offramp)."""\n'
                    + "".join(steps)
                    + f"{body}return chat_completion_from_blocks({expr})\n"
                ),
            )
        )
    if "chat_completion_from_blocks" not in bound:
        first = min(found, key=lambda f: f[0].lineno)[0]
        at = min([first.lineno, *(d.lineno for d in first.decorator_list)]) - 1
        inserts.append((at, FAKE_HELPER + "\n\n"))
    for row, text in sorted(inserts, key=lambda i: i[0], reverse=True):
        lines.insert(row, text)
    return add_imports("".join(lines), client)


def _signature(source: str, function: str | None) -> str | None:
    """The parameter names of `function` in `source` (None when it is not found)."""
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == function:
            a = n.args
            return ", ".join(x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs])
    return None


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
    source: str, site: CallSite, params: dict[str, Any], constants: dict[str, str], system_name: str | None
) -> list[str]:
    """Problems with the patched site file: the call must pass exactly the evaluated parameters as written-out
    keyword arguments (a schema may be a module constant holding it) plus messages, and use the prompt
    constants."""
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
    call, problems = tf[0], []
    if call.args or any(k.arg is None for k in call.keywords):
        problems.append("the call takes written-out keyword arguments only: no positional arguments, no **")
    named = {k.arg: k.value for k in call.keywords if k.arg}
    if set(named) != {*params, "messages"}:
        problems.append(f"the call must pass exactly {', '.join([*params, 'messages'])}, nothing else")
    values = _module_values(tree)
    for key, want in params.items():
        if (node := named.get(key)) is None:
            continue
        try:
            got = values.get(node.id, ...) if isinstance(node, ast.Name) else ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            got = ...
        if got != want:
            problems.append(f"`{key}` must be exactly the evaluated {_flat(want)[:200]}")
    used = {n.id for n in ast.walk(scope) if isinstance(n, ast.Name)}
    named_constants = (
        ("system prompt", system_name),
        ("few-shot", constants.get("few_shot")),
        ("user prefix", constants.get("prefix")),
        ("user suffix", constants.get("suffix")),
    )
    for label, const in named_constants:
        if const and const not in used:
            problems.append(f"{label} constant {const} is not used in {where}")
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


SYSTEM = """Offramp migrates one LLM call site of a Python repository to NVIDIA Nemotron on Nebius Token \
Factory, which is OpenAI-compatible: the call becomes `client.chat.completions.create(...)` on an \
`openai.OpenAI` client, sending a request Offramp measured to reach parity. Offramp's code has already \
rewritten what it could (see "The call" and "Already changed by Offramp"). You make the rest of the \
repository fit, keeping its ruff and tests green.

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
1. Files marked (read-only) are final: never edit them. When the migrated module is read-only, edit only \
the tests and their fakes.
2. Never change the request: the call's keyword arguments, their values and its messages stay exactly as \
Offramp wrote them. Keep every function's name, parameters and return value.
3. Only when the migrated module is editable: make the code send exactly the request under "The call" \
(written-out keyword arguments, no ** unpacking) and read the reply the OpenAI way: text is \
`response.choices[0].message.content` (may be None); tool calls are \
`response.choices[0].message.tool_calls or []`, each with `.function.name` and `.function.arguments` (a \
JSON string: json.loads it). Map them onto exactly what the function returned before.
4. Code that builds the old SDK client to call the migrated function passes `token_factory_client()` \
instead, and a test that replaced that constructor with a fake replaces `token_factory_client` the same \
way (Offramp did this where the client is built inside the call).
5. Tests: update what pins the old provider for THIS call site so it asserts the new request: the model ID, \
SDK keyword arguments (`system=`, `max_completion_tokens`, a `tool_choice` or `tools` value, parameter \
rules of the old model), the message layout, and checks of the call's source (tables of call sites, \
`ast` checks of keyword arguments). Every other call site keeps its old pins. The system prompt is now the \
first message: a test that read the user message as `messages[0]` reads it as `messages[-1]`, asserting \
the same things. Never weaken a behavioural or parsing assertion: expected return values and parsing \
checks stay exactly as they are. Never delete or rename a test.
6. Fakes: when a provider fake still cannot answer `chat.completions.create` the way the new code reads \
it, extend the repo's existing fake helper, building real openai response objects the way it already \
does, instead of writing ad-hoc fake classes inside tests.
7. Add the imports your code needs and remove the ones it leaves unused (import order is fixed \
automatically afterwards). Keep lines within the repo's line length.
8. When "What fails now" lists failing checks, your edits must make every one of them pass."""


def _call_text(source: str, site: CallSite) -> str | None:
    """The source of the site's single LLM call, dedented to its statement's indentation."""
    tree = ast.parse(source)
    calls = _calls(_function(tree, site) or tree)
    if len(calls) != 1:
        return None
    lines = source.splitlines(keepends=True)
    stmt = min((n for n in ast.walk(tree) if isinstance(n, ast.stmt) and calls[0] in ast.walk(n)),
               key=lambda n: n.end_lineno - n.lineno)  # fmt: skip
    return textwrap.dedent("".join(lines[stmt.lineno - 1 : stmt.end_lineno])).rstrip("\n")


def _prompt(
    site: CallSite,
    tier: Tier,
    params: dict[str, Any],
    constants: dict[str, str],
    system_name: str | None,
    files: dict[str, str],
    client: str,
    client_at: tuple[str, str],
    width: int,
    calls: tuple[str | None, str | None],
    fixed: bool,
    done: list[str],
    failing: str | None = None,
) -> str:
    before, now = calls
    if fixed:
        call = [
            f"`{site.file}` is migrated by Offramp and read-only: `{site.function}` now sends the evaluated "
            "request with written-out keyword arguments and reads the reply the OpenAI way.",
        ]
    elif now != before:
        call = [
            f"Offramp rewrote the call in `{site.file}` (below; keep it exactly). Map how `{site.function}` "
            "reads the reply onto the OpenAI shape (rule 3).",
        ]
    else:
        messages = [
            f'{{"role": "system", "content": {system_name}}}'
            if system_name
            else "the system prompt the code sends today, as the first message (an Anthropic `system=` "
            'argument becomes {"role": "system", "content": ...})'
        ]
        if "few_shot" in constants:
            messages.append(f"*{constants['few_shot']}")
        last = "the user message(s) the code sends today"
        wraps = [f"{constants[k]} {how}" for k, how in (("prefix", "prepended"), ("suffix", "appended"))
                 if k in constants]  # fmt: skip
        if wraps:
            last += f", with {' and '.join(wraps)} to the content of the last one"
        messages.append(last)
        written = ", ".join(f"{k}={constants.get(k) or _flat(v)}" for k, v in params.items())
        call = [
            f"Rewrite the call in `{site.file}` to `client.chat.completions.create({written}, "
            "messages=[...])`, these keyword arguments written out exactly, nothing else passed; the "
            "NEMOTRON_* constants are defined in the file.",
            "messages, in this order:",
            *(f"{i}. {m}" for i, m in enumerate(messages, 1)),
        ]
    if before:
        call += ["", "Before (what the tests pin today):", "```python", before, "```"]
    if now and now != before:
        call += ["Now:", "```python", now, "```"]
    if done:
        call += ["", "Already changed by Offramp (keep these):", *(f"- {d}" for d in done)]
    if failing:
        if failing.startswith("repo checks failed"):
            failing = "this repo's checks fail:\n- " + failing.partition("): ")[2].replace(" | ", "\n- ")
        call += ["", "## What fails now", f"With Offramp's changes alone, {failing}", "Make these pass."]
    task = site.contract.task_type if site.contract else "unknown"
    lines = [
        f"## Call site {site.id}",
        f"`{site.file}` lines {site.lines[0]}-{site.lines[-1]}, function `{site.function}`: {site.sdk} "
        f"`{site.call}` on `{site.source_model}` ({task}). Target: {TIER_NAMES[tier]} on Token "
        "Factory.",
        "",
        "## The call",
        *call,
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
        mode = "read-only" if fixed and path == site.file else "editable"
        lines += [f"### {path} ({mode})", "```python", text.rstrip("\n"), "```"]
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
    lines = output.splitlines()
    failed: dict[str, list[str]] = {}  # a parametrised test's cases count once
    for ln in lines:
        if re.match(r"(FAILED|ERROR) |\S+\.py:\d+:\d+: |\$ ", ln):
            failed.setdefault(re.sub(r"\[[^\]]*\]( - .*)?$", "[…]", ln), []).append(ln)
    tests = [hits[0] if len(hits) == 1 else f"{key} ({len(hits)} cases)" for key, hits in failed.items()]
    keep = tests[:14] + [ln for ln in lines if re.match(r"E\s{3}", ln)][:6]  # which tests, then why
    return (" | ".join(keep) if keep else output.strip()[-500:])[:2400]


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
    return [*pieces, literal(cur)]


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
        line = text[row - 1]  # may hold several physical lines once a literal to its right was split
        if row not in lines or len(line.split("\n", 1)[0].rstrip("\r")) <= width:
            continue
        segment = line[c1:c2]
        indent = re.match(r"[ \t]*", line).group(0)
        lead = " " * c1 if joined else indent + "    "
        if width - len(lead) < 20:  # no room to split into (e.g. a second literal far along one line)
            continue
        if segment.lstrip("rRuU")[:1] in "'\"":  # plain string: re-render from its value
            value = ast.literal_eval(segment)
            chunks = _chunks(value, width - len(lead)) if isinstance(value, str) else []
            pieces = [_str(c) for c in chunks]
        else:
            pieces = _fstring_pieces(segment, width - len(lead))
        if not pieces or (joined and len(pieces) < 2):  # a lone literal may just move to its own line
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
    rewritten: bool  # the call already sends the request: no edit may be needed
    fixed: bool  # the site's module is final (call and reply rewritten by code): the model edits tests only


def _settle(box: Sandbox, repo: Path, paths: list[str], before: dict[str, str], width: int, formatted: bool):
    """Mechanical clean-up of changed files on disk: the repo's ruff fixes (and format), then over-long
    strings on changed lines wrapped (formatted again: the pieces no longer fit on one line, so they stay)."""
    box.tidy(paths, format_code=formatted)
    rewrapped = []
    for p in paths:  # after formatting, whose indentation is final
        text = (repo / p).read_text()
        if (
            p.endswith(".py")
            and (wrapped := wrap_long_strings(text, changed_lines(before[p], text), width)) != text
        ):
            (repo / p).write_text(wrapped)
            rewrapped.append(p)
    if formatted and rewrapped:
        box.tidy(rewrapped, format_code=True)


def _attempt(
    job: _Job, edits: list[Edit], repo: Path, box: Sandbox, formatted: bool, write
) -> tuple[str | None, dict]:
    """Apply one set of edits to the base, check it alone, then put the base back. (None, the site's patched
    files) when the repo is green with it; else (why not, {})."""
    if not edits and not job.rewritten:
        return "the reply holds no edit blocks", {}
    new = dict(job.editable)
    problems = _apply(edits, new, frozenset([job.site.file] if job.fixed else []))
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
        problems += conformance(new[job.site.file], job.site, job.params, job.constants, job.system_name)
        if (was := _signature(job.editable[job.site.file], job.site.function)) != (
            now := _signature(new[job.site.file], job.site.function)
        ):
            problems.append(
                f"the signature of `{job.site.function}` changed ({was} -> {now}): keep every parameter"
            )
    if problems:
        return f"patch rejected (adapter spec or patch rules): {'; '.join(problems)}", {}
    changed = [p for p in new if new[p] != job.before[p]]
    for p in changed:
        write(p, new[p])
    _settle(box, repo, changed, job.before, job.width, formatted)
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
        original = read(site.file)
        prepared, constants, system_name, rewritten, fixed = _prepare(
            original, site, adapter.spec, params, width
        )
        (repo / site.file).write_text(add_imports(prepared, client_at[1]))
        _settle(box, repo, [site.file], {site.file: original}, width, formatted)  # shown as it will ship
        prepared = read(site.file)
        (repo / site.file).write_text(original)
        context = _context(repo, site, client_at[0])
        editable = {p: (prepared if p == site.file else read(p)) for p in context}
        for p in context:  # mechanical: callers hand the migrated function a Token Factory client
            if p != site.file and not is_test_path(p) and site.function and p.endswith(".py"):
                editable[p] = rewrite_callers(editable[p], site.function, client_at[1])
        changed_callers = {Path(p).stem for p in context if p != site.file and editable[p] != read(p)}
        done = [f"`{p}` passes `token_factory_client()` to `{site.function}`" for p in context
                if p != site.file and not is_test_path(p) and editable[p] != read(p)]  # fmt: skip
        old = "OpenAI" if site.sdk == "openai" else "Anthropic"
        for p in context:  # mechanical: the repo's provider fakes answer the Token Factory client's call
            if not (is_test_path(p) and p.endswith(".py")):
                continue
            if (
                site.sdk == "anthropic"
                and (converted := convert_fakes(editable[p], client_at[1])) != editable[p]
            ):
                editable[p] = converted
                done.append(f"`{p}`: its Anthropic fake also answers `chat.completions.create`, with the "
                            "same canned blocks as an OpenAI chat completion")  # fmt: skip
            if (
                changed_callers
                and (mirrored := mirror_fakes(editable[p], changed_callers, old)) != editable[p]
            ):
                editable[p] = mirrored
                done.append(f"`{p}` also fakes `token_factory_client` where it faked `{old}`")
        before = {p: read(p) for p in editable}
        job = _Job(site, tier, adapter.spec, params, constants, system_name, editable, before,
                   client_module=client_at[1], width=width, rewritten=rewritten, fixed=fixed)  # fmt: skip
        failure, patched, tried = None, {}, []
        if (
            rewritten
        ):  # Offramp's own changes alone: green needs no model call; red shows the model what fails
            failure, patched = _attempt(job, [], repo, box, formatted, write)
            tried.append([])
        checked = failure  # what the last check said
        calls = (_call_text(read(site.file), site), _call_text(prepared, site))
        prompt = _prompt(site, tier, params, constants, system_name, editable, client, client_at, width,
                         calls, fixed, done, failure)  # fmt: skip
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        for attempt in range(1 + REPAIRS if failure or not rewritten else 0):
            try:
                res = llm.chat(
                    role=ROLE,
                    tier=Tier.ULTRA
                    if attempt
                    else Tier.SUPER,  # repairs on Ultra (Task 10c; PRD §6.11: Super)
                    messages=messages,
                    max_tokens=REPAIR_MAX_TOKENS if attempt else MAX_TOKENS,
                    temperature=0.0,
                    reasoning="off",  # GC25; recorded with thinking on, 2 of 5 replies hit the token cap
                )
            except BudgetExceeded as e:
                failure = f"patch not composed: {e}" + (f"; last failure: {failure}" if failure else "")
                break
            edits = parse_edits(res.output.text or "").edits
            if not edits and (attempt or rewritten):  # empty or malformed: ask again, no check needed
                failure = "the reply holds no complete edit block" + (
                    f"; last failure: {checked}" if checked else ""
                )
            else:
                named = {e.path for e in edits}
                edits += [e for e in (tried[-1] if attempt else []) if e.path not in named]  # kept as is
                if edits in tried:  # a patch already tried (A-B-A too): the sandbox would say the same again
                    failure = f"repair produced no change; last failure: {failure}"
                    break
                tried.append(edits)
                failure, patched = _attempt(job, edits, repo, box, formatted, write)
                checked = failure
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
