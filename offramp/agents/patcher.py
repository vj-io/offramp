"""Patch Composer (PRD §6.8): each MIGRATE site's winning adapter becomes a code change; the repo stays green.

Per run, on a sandbox copy of the target repo:
0. copy the repo without following symlinks (a link leaving the repo is dropped, with a warning) and check the
   untouched copy with the repo's own ruff + pytest;
1. add the Token Factory client module beside the migrated sites (`<pkg>/offramp_client.py` inside the
   package holding them, so it ships with it; see `client_location`; with an async client too when an async
   site is migrated), declare the openai SDK where the repo declares the one it replaces, and mark
   KEEP_CLOSED calls with
   `# offramp: kept on <model> — parity <p>% < <threshold>%`; check again;
2. for each MIGRATE site, against that same base (so no site depends on another's patch): the request
   parameters come from `build_request` (Task 6, the function the eval runner used), so what ships is what was
   evaluated. Offramp's code writes the call (the evaluated request as written-out keyword arguments), maps
   the reply, rewires callers and provider fakes; a call or a reply it cannot rewrite is a manual review.
   Code that ships is code-only: Nemotron 3 Super (role `patcher`) edits only the tests and fakes that pin
   the old provider, as search/replace edits (or a short test file whole, offered after a SEARCH miss);
3. every patch is checked: the backstop (each file that ships as code, on disk, AST-identical to what
   Offramp's code wrote; no new ones), no removed tests; the repo's ruff + pytest in the sandbox; and, where
   an eval case can drive it, the probe (the migrated function, run there, must send exactly build_request's
   request). Up to REPAIRS repair rounds (Ultra), each applied to, and shown, the files as the last checked
   try left them. Any failure: the site becomes MANUAL_REVIEW;
4. the sites green alone are merged in site order and checked together, the backstop and the probe again
   too: a site whose code changed drops with every site whose patch edited code it reads; a red check gets
   one combined repair (tests and fakes only), then sites are added one at a time (MANUAL_REVIEW if dropped).
"""

import ast
import difflib
import fnmatch
import io
import json
import logging
import re
import shutil
import string
import sys
import textwrap
import tokenize
import tomllib
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from openai import APIStatusError
from packaging.requirements import InvalidRequirement, Requirement
from pydantic import BaseModel, Field

from offramp.agents.pr_writer import NO_TEST_FILE, TIER_NAMES, TIMED_OUT, is_test_path, threshold
from offramp.agents.translate import UnresolvedCallSite, build_request
from offramp.evals.baselines import load_cases
from offramp.sandbox import (
    NO_SANDBOX,
    NOT_COPIED,
    OPENAI_REQUIREMENT,
    TIMEOUT_S,
    CheckResult,
    Sandbox,
    copy_repo,
    git_files,
)
from offramp.schemas import (
    Adapter,
    AdapterSpec,
    CallSite,
    EvalCase,
    PatchFailure,
    PatchFailureKind,
    RunConfig,
    Tier,
    Verdict,
)
from offramp.tf.budget import BudgetExceeded
from offramp.tf.client import LLM, _merge, plan_structured
from offramp.tf.tiers import get_tier, reasoning_params

log = logging.getLogger(__name__)

ROLE = "patcher"
MAX_TOKENS = 8000
REPAIR_MAX_TOKENS = 12000  # a complete second set of blocks, with room to spare (a recorded repair ran out)
REPAIRS = 3  # tries after the first, each shown the last failure; a repeated patch stops at once
REPAIR = """Your edits were not accepted: {failure}
Reply with SEARCH/REPLACE edit blocks only, no analysis or explanation, then the NOTES line. Your blocks \
apply to each file as it is now: as shown under "Files now" below where it changed since my first message \
(your checked edits, the imports Offramp added and the repo's formatting are in it) or where a SEARCH \
missed, else as shown in my first message. Copy every SEARCH from that text and give only the blocks still \
needed: edits already checked stay in, and leaving one out does not drop it; to drop one, send a block that \
restores the earlier text. Begin with `FILE:`."""
DELETED = """Your patch deletes {tests}. Never delete a test: keep each one under its name and rewrite it \
so it asserts the new Nemotron request (the model ID, keyword arguments and message layout Offramp now \
sends) where it pinned the old provider's."""
WHOLE_FILE_LINES = 300  # a test file shorter than this may come back whole (offered after a SEARCH miss)
FILES_NOW_CHARS = 40_000  # "Files now" holds the failure's own files first, then others while they fit
LEFT_OUT = (
    "Also changed, not shown again to keep this message short, so not editable in this repair: {files}."
)
NOT_SHOWN = "was not shown again in its current text, so it is not editable in this repair"
SUPERSEDED = '### {path} (editable): superseded, its current text is under the latest "Files now"'
ALREADY = "every block of the repair is already"  # its checked blocks re-sent, the one to drop left out
RETURNED = "the repair returned to a patch already checked"
# account- or config-wide (auth, out of credit, unknown model) or transient (what the SDK itself retries):
# these fail the run, which can be resumed; any other 4xx fails only the site whose request it refused
NOT_A_REFUSAL = (401, 402, 403, 404, 408, 409, 429)
STALE = "internal: the conversation would show a stale copy of {path}"
TOO_LARGE = (
    "context length",
    "context window",
    "maximum context",
    "too long",
    "too many tokens",
    "too large",
)
READ_ONLY = "is read-only: it ships as Offramp's code wrote it; edit only tests and fakes"
WHOLE = """Your SEARCH text did not match exactly one place in {files}. Instead of SEARCH/REPLACE blocks, \
you may give such a file whole, as shown under "Files now" with all your changes, in one block:
FILE: {path}
<<<<<<< WHOLE FILE
every line of the new file
>>>>>>> WHOLE FILE
Every test in it stays: rewritten where it pins the old provider, never deleted."""
ROOT_CLIENT = ("offramp_client.py", "offramp_client")  # when no one package or folder holds the sites
TEMPLATE = Path(__file__).parents[1] / "templates" / "client.py.tmpl"
SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", "node_modules", "offramp"}

BuildRequest = Callable[[AdapterSpec, EvalCase, Tier], dict[str, Any]]
# Names a patch may use without importing them; added mechanically, like an editor's auto-import.
AUTO_IMPORTS = {
    "token_factory_client": "from {client} import token_factory_client",
    "token_factory_async_client": "from {client} import token_factory_async_client",
    "OpenAI": "from openai import OpenAI",
    "AsyncOpenAI": "from openai import AsyncOpenAI",
    "json": "import json",
    "ChatCompletion": "from openai.types.chat import ChatCompletion",
    "Completions": "from openai.resources.chat.completions import Completions",
}
# pytest's built-in fixtures: a patched test may use one without requesting it; added like an import.
PYTEST_FIXTURES = ("monkeypatch", "capsys", "capfd", "caplog", "tmp_path", "tmp_path_factory")
# `FILES:` is a slip seen live: read only as a fallback (parse_edits)
_FILE = re.compile(r"^(FILES?): *`?([^`\s]+)`?[ \t]*$", re.M)
_OPENER = re.compile(r"^<{7} (?:SEARCH|WHOLE FILE)$", re.M)
_BOUND = re.compile(r"^(?:<{7} SEARCH$|FILES?:)", re.M)  # a SEARCH block's own text ends before these
_WHOLE = r"|<{7} WHOLE FILE\n(?P<whole>.*?)^>{7} WHOLE FILE[ \t]*$"
_STRICT = re.compile(
    r"<{7} SEARCH\n(?P<search>.*?)\n={7}\n(?P<replace>.*?)>{7} REPLACE" + _WHOLE, re.S | re.M
)
# Format slips seen live, read only for a block that does not parse strictly: a bare `REPLACE` closing line,
# and a second `=======` line right before the closing marker. The SEARCH text is never loosened.
_TOLERANT = re.compile(
    r"<{7} SEARCH\n(?P<search>.*?)\n={7}\n(?P<replace>.*?)(?:^={7}\n)?(?:>{7} REPLACE|^REPLACE[ \t]*$)"
    + _WHOLE,
    re.S | re.M,
)


class Edit(BaseModel):
    path: str
    search: str
    replace: str
    whole: bool = False  # `replace` is the whole new file (test files under WHOLE_FILE_LINES only)


class PatchEdits(BaseModel):
    edits: list[Edit] = Field(default_factory=list)
    notes: str = ""


def parse_edits(text: str) -> PatchEdits:
    """The patcher replies in plain SEARCH/REPLACE blocks: code inside JSON strings needs every Python escape
    escaped again, and the model dropped some of them (a recorded run broke 3 of 5 patches that way).

    Each block is read strictly first, exactly as before any slip was tolerated. Only a block that does not
    parse strictly (its file line is `FILES:`, or no `>>>>>>> REPLACE` closes it before the next SEARCH block
    or file line) is read with the tolerant closers; when that fails too, the strict reading stands."""
    files = [(m.start(), m[1], m[2]) for m in _FILE.finditer(text)]
    edits, at = [], 0
    while opener := _OPENER.search(text, at):
        start = opener.start()  # a block without its own file line belongs to the previous file
        label, path = next(((k, p) for pos, k, p in reversed(files) if pos < start), (None, None))
        bound = b.start() if (b := _BOUND.search(text, opener.end())) else len(text)
        block = _STRICT.match(text, start)
        if label != "FILE" or block is None or block.end() > bound:
            tolerant = _TOLERANT.match(text, start)
            if tolerant is not None and tolerant.end() <= bound:
                block = tolerant
            elif label != "FILE":
                block = None
        if block is None or path is None:
            at = opener.end()
            continue
        at = block.end()
        if block["whole"] is not None:
            edits.append(Edit(path=path, search="", replace=block["whole"], whole=True))
        else:
            replace = block["replace"]
            edits.append(
                Edit(
                    path=path,
                    search=block["search"],
                    replace=replace[:-1] if replace.endswith("\n") else replace,
                )
            )
    notes = re.search(r"^NOTES:(.*)$", text, re.M)
    return PatchEdits(edits=edits, notes=notes.group(1).strip() if notes else "")


def edits_text(patch: PatchEdits) -> str:
    """Inverse of parse_edits (the reply format, as shown to the model)."""
    blocks = [
        f"FILE: {e.path}\n<<<<<<< WHOLE FILE\n{e.replace.removesuffix('\n')}\n>>>>>>> WHOLE FILE"
        if e.whole
        else f"FILE: {e.path}\n<<<<<<< SEARCH\n{e.search}\n=======\n{e.replace}\n>>>>>>> REPLACE"
        for e in patch.edits
    ]
    return "\n\n".join([*blocks, f"NOTES: {patch.notes}"])


class Problem(NamedTuple):
    """One patch-rule refusal: its kind, the message the repair and the verdict reason get, and its file."""

    kind: PatchFailureKind
    text: str
    file: str


class Unsupported(ValueError):
    """The call has a shape the rewrite cannot carry exactly (a streamed reply, a keyword argument the
    evaluated request does not send, a system prompt that is not provably text): manual review."""


# keyword arguments of the source call: the translation turns these into the evaluated request; transport
# options are carried through as written; anything else makes the site a manual review
TRANSLATED_KWARGS = {"model", "messages", "system", "tools", "tool_choice", "response_format", "max_tokens",
                     "max_completion_tokens", "temperature", "reasoning_effort", "stream"}  # fmt: skip
CARRIED_KWARGS = ("timeout", "extra_headers")
# The only client classes a rewrite replaces, resolved through the module's imports (aliases too), -> the
# Token Factory client function that replaces each (an async client stays async). Any other client (Bedrock,
# Vertex, Azure, a helper's, an attribute) is a manual review: nothing ships on a client the rewrite does not
# know.
CLIENT_CLASSES = {
    "openai.OpenAI": "token_factory_client",
    "anthropic.Anthropic": "token_factory_client",
    "openai.AsyncOpenAI": "token_factory_async_client",
    "anthropic.AsyncAnthropic": "token_factory_async_client",
}
OLD_CLIENTS = {k.split(".")[1]: v for k, v in CLIENT_CLASSES.items()}  # by class name, as tests fake them
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}  # the replaced SDK's key name
CLIENT_OPTIONS = ("timeout", "max_retries")  # options of an old client that the async Token Factory one takes


UNCHANGED = "the code Offramp wrote, unchanged"  # how a shipped site's request was checked, when not run


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
    combined_repair: list[str] = Field(default_factory=list)  # tests and fakes the combined repair changed
    combined_failure: str | None = None  # why the merged tree stayed red (then sites were dropped one by one)
    verification: dict[str, str] = Field(default_factory=dict)  # shipped site -> how its request was checked
    tests_run: dict[str, list[str]] = Field(default_factory=dict)  # shipped site -> its tests that ran
    untested: dict[str, str] = Field(default_factory=dict)  # shipped site -> why none of its tests ran
    tests_edited: dict[str, list[str]] = Field(default_factory=dict)  # shipped site -> its edited tests
    warnings: list[str] = Field(default_factory=list)  # e.g. symlinks dropped from the sandbox copy
    linted: bool = True  # the checks ran the repo's ruff (only a repo that configures ruff gets it)
    sandbox_notes: list[str] = Field(default_factory=list)  # Sandbox.notes: the PR states them
    async_client: bool = False  # the client module also holds token_factory_async_client
    client_packaged: bool = True  # the client module sits inside a package (it ships in its wheel)
    change_notes: list[str] = Field(default_factory=list)  # dependency edits, lockfiles, key reads left as is
    merged_diff: str = ""  # the shipped tree against the per-site diffs merged: what only merging changed


def client_location(repo: Path, files: list[str]) -> tuple[str, str] | None:
    """(path, module) of the client, beside the migrated sites; never a top-level `offramp` package (Offramp's
    own name, and perhaps the target's):
    - one top-level package holds every site: `<pkg>/offramp_client.py`, imported as `<pkg>.offramp_client`,
      so it ships in that package's wheel;
    - one plain folder (no `__init__.py`) holds them: `<folder>/offramp_client.py`, imported the way the
      folder's modules import each other: by bare name when the folder is an import root (the repository root,
      `src/`) or one of them imports a sibling so (`import db` beside db.py: the folder is on sys.path), else
      by the folder's dotted path;
    - else `offramp_client.py` at the import root (`src/` when every site is under it), by bare name.

    None when a site sits directly in a repository root that is itself a package (`__init__.py`): there is
    no package below the root to hold the module, and the root's own import name is not known here.
    Unsupported when a bare `offramp_client` import may not find this module: another `offramp_client.py` in
    the repo could shadow it, or sites in several folders share the root's while one of those folders runs
    with itself on sys.path (its modules import each other by bare name)."""
    place = _client_place(repo, files)
    if place is None or place[1] != ROOT_CLIENT[1]:
        return place
    root = repo.resolve()
    for other in sorted(p.relative_to(root) for p in root.rglob(ROOT_CLIENT[0])):
        if other.as_posix() != place[0] and not set(NOT_COPIED) & set(other.parts[:-1]):
            why = "which the sites import by bare name `offramp_client`: remove or rename it, then run again"
            raise Unsupported(f"`{other.as_posix()}` could shadow this run's `{place[0]}`, {why}")
    return place


def _client_place(repo: Path, files: list[str]) -> tuple[str, str] | None:
    """`client_location` before its shadowing check."""
    repo = repo.resolve()
    tops, folders = set(), set()
    for f in files:
        top = (repo / f).resolve().parent
        if top == repo and (repo / "__init__.py").is_file():
            return None
        if top == repo or not (top / "__init__.py").is_file():
            folders.add(top)
            continue
        while top.parent != repo and (top.parent / "__init__.py").is_file():  # never climb to the root
            top = top.parent
        tops.add(top)

    def root(folder: Path) -> Path:  # the folder an import starts from: src/ for a src layout
        return repo / "src" if folder == repo / "src" or (repo / "src") in folder.parents else repo

    if len(tops) == 1 and not folders:
        top = tops.pop()
        return f"{top.relative_to(repo).as_posix()}/offramp_client.py", f"{top.name}.offramp_client"
    if len(folders) == 1 and not tops:
        folder = folders.pop()
        path = (folder / "offramp_client.py").relative_to(repo).as_posix()
        if folder == root(folder) or _imports_siblings(folder):
            return path, "offramp_client"
        return path, ".".join([*folder.relative_to(root(folder)).parts, "offramp_client"])
    roots = {root(f) for f in tops | folders}
    base = roots.pop() if len(roots) == 1 else repo
    on_path = sorted(f.relative_to(repo).as_posix() for f in folders if f != base and _imports_siblings(f))
    if on_path:  # its scripts run with that folder first on sys.path, perhaps without the import root
        why = "it runs with itself on sys.path, where the import may miss it"
        raise Unsupported(f"the sites span several folders, which would share one `offramp_client` module "
                          f"at the import root by bare name, but `{on_path[0]}/` imports its own modules by "
                          f"bare name: {why}")  # fmt: skip
    return (base / "offramp_client.py").relative_to(repo).as_posix(), "offramp_client"


def _imports_siblings(folder: Path) -> bool:
    """Whether a module in this plain folder imports another one by bare name (`import db` beside db.py), so
    the folder is on sys.path when its code runs."""
    for f in folder.glob("*.py"):
        try:
            tree = ast.parse(f.read_text())
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                names = [n.module]
            else:
                continue
            if any((folder / f"{name.split('.')[0]}.py").is_file() for name in names):
                return True
    return False


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


def _prepare(
    source: str,
    site: CallSite,
    spec: AdapterSpec,
    params: dict[str, Any],
    width: int,
    client: str = ROOT_CLIENT[1],  # import path of the Token Factory client module
):
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
    rewritten = _rewrite_call(source, site, constants, system_name, params, width, client)
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
    client: str = ROOT_CLIENT[1],
) -> str | None:
    """The site's call rewritten to `<client>.chat.completions.create(model=..., <every evaluated parameter as
    a literal keyword argument, schemas by constant name>, messages=[system, *few-shot, ...the call's own
    messages, the last user content wrapped in prefix/suffix])`, from an OpenAI `chat.completions.create` or
    an Anthropic `messages.create` (its `system=` becomes the first message); `await` stays as it was. When
    the function builds the old SDK client itself, that becomes the Token Factory client (`_built_client`).
    None when the call is not a single call with a literal `messages=[...]` list, or its built client serves
    other code too: then nobody writes it."""
    tree = ast.parse(source)
    scope = _function(tree, site) if site.function else tree
    if scope is None:
        return None
    kinds = ("chat.completions.create", "messages.create")
    calls = [c for c in ast.walk(scope) if isinstance(c, ast.Call) and _call_name(c.func).endswith(kinds)]
    if len(calls) != 1 or calls[0].args:
        return None
    call = calls[0]
    kw = {k.arg: k.value for k in call.keywords}
    stream = kw.get("stream")  # refused whoever would write the call: the code reading it would not fit
    if stream is not None and not (isinstance(stream, ast.Constant) and stream.value is False):
        raise Unsupported("the call streams its reply (stream=); Offramp evaluated the non-streaming call")
    if unknown := sorted(set(kw) - TRANSLATED_KWARGS - set(CARRIED_KWARGS) - {None}):
        names = ", ".join(f"`{k}=`" for k in unknown)
        raise Unsupported(f"the call passes {names}, which the evaluated request does not carry")
    messages = kw.get("messages")
    if None in kw or not isinstance(messages, ast.List):
        return None
    anthropic = not _call_name(call.func).endswith("chat.completions.create")
    receiver = call.func.value.value if anthropic else call.func.value.value.value

    def seg(node: ast.AST) -> str:
        return ast.get_source_segment(source, node)

    lines = source.splitlines(keepends=True)
    spans: list[tuple[int, int, str]] = []  # (start, end, new text) besides the call itself

    def put(node: ast.AST, text: str) -> None:
        start = _offset(lines, node.lineno, node.col_offset)
        spans.append((start, _offset(lines, node.end_lineno, node.end_col_offset), text))

    head_text = seg(receiver)
    built = _built_client(tree, scope, receiver, source)
    if built is not None:
        factory = CLIENT_CLASSES[_client_class(built, tree, scope)]
        if built is receiver:  # `AsyncAnthropic().messages.create(...)`
            head_text = _factory_call(built, factory, source)
        else:  # `client = AsyncAnthropic(...)` in the function
            put(built, _factory_call(built, factory, source))
        root = getattr(built.func.value if isinstance(built.func, ast.Attribute) else built.func, "id", None)
        imports = [n for n in ast.walk(scope) if isinstance(n, ast.Import | ast.ImportFrom)
                   and any((a.asname or a.name).split(".")[0] == root for a in n.names)]  # fmt: skip
        reads = [
            n
            for n in ast.walk(scope)
            if isinstance(n, ast.Name) and n.id == root and isinstance(n.ctx, ast.Load)
        ]
        if len(imports) == 1 and len(imports[0].names) == 1 and len(reads) == 1:
            put(imports[0], f"from {client} import {factory}")  # imported where the SDK was: lazily too
    # the function's own checks of the old provider's key (`os.environ.get("ANTHROPIC_API_KEY")`, `in
    # os.environ`) check the Token Factory key: the literal names only, outside the call and its old client
    key, skip = PROVIDER_KEYS.get(site.sdk), {*ast.walk(call), *(ast.walk(built) if built else ())}
    body = [n for stmt in scope.body for n in ast.walk(stmt)] if key and scope is not tree else []
    skip |= {v for n in body if isinstance(n, ast.JoinedStr) for v in ast.walk(n)}  # an f-string's text stays
    for n in body:  # the body only: a decorator's or a parameter default's literal stays (listed in the PR)
        if isinstance(n, ast.Constant) and n.value == key and n not in skip and key in (seg(n) or ""):
            put(n, seg(n).replace(key, "NEBIUS_API_KEY"))

    def system_text(node: ast.expr) -> str:  # what translate.system_message sent: a string, or blocks joined
        text = isinstance(node, ast.JoinedStr) or (
            isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        if text or (isinstance(node, ast.Name) and node.id in _module_strings(tree)):
            return seg(node)
        blocks = isinstance(node, ast.List | ast.Tuple) or (
            isinstance(node, ast.Name) and isinstance(_module_values(tree).get(node.id), list | tuple)
        )
        if blocks:
            return f'"\\n".join(b["text"] for b in {seg(node)} if b.get("type") == "text")'
        raise Unsupported("its Anthropic `system=` is not provably a string or a list of text blocks")

    items = list(messages.elts)
    head = []
    if items and _role(items[0]) == "system":
        old = items.pop(0)
        head.append(f'{{"role": "system", "content": {system_name}}}' if system_name else seg(old))
    elif system_name:
        head.append(f'{{"role": "system", "content": {system_name}}}')
    elif anthropic and "system" in kw:
        head.append(f'{{"role": "system", "content": {system_text(kw["system"])}}}')
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
    start = _offset(lines, call.lineno, call.col_offset)
    end = _offset(lines, call.end_lineno, call.end_col_offset)
    indent = re.match(r"[ \t]*", lines[call.lineno - 1]).group(0)
    i1, i2 = indent + "    ", indent + "        "
    entries = "".join(f"{i2}{x},\n" for x in [*head, *body])
    args = "".join(
        f"{i1}{k}={constants.get(k) or py_literal(v, len(i1), width, len(i1) + len(k) + 1)},\n"
        for k, v in params.items()
    )
    args += "".join(f"{i1}{k}={seg(kw[k])},\n" for k in CARRIED_KWARGS if k in kw)  # as the code wrote them
    text = f"{head_text}.chat.completions.create(\n{args}{i1}messages=[\n{entries}{i1}],\n{indent})"
    rewritten = source
    for a, b, new in sorted([*spans, (start, end, text)], reverse=True):  # bottom first: offsets stay valid
        rewritten = rewritten[:a] + new + rewritten[b:]
    try:
        ast.parse(rewritten)
    except SyntaxError:
        return None
    return rewritten


def _imported_names(tree: ast.Module, scope: ast.AST) -> dict[str, str]:
    """Local name -> the dotted name it imports (`Claude` -> `anthropic.AsyncAnthropic` for `from anthropic
    import AsyncAnthropic as Claude`), from the module's top-level imports and the function's own (which
    shadow them). A name the module or the function also binds another way (a parameter too) is left out: it
    may not be the import. `X = None` (an optional import's fallback) does not count."""
    imports, rebound, stack = {}, set(), list(tree.body)
    nodes = []
    while stack:  # module level, outside functions and classes
        n = stack.pop()
        nodes.append(n)
        if not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            stack.extend(ast.iter_child_nodes(n))
    nodes += list(ast.walk(scope)) if scope is not tree else []
    if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
        rebound |= {a.arg for a in [*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs]}
    fallback = {t for n in nodes if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                and n.value.value is None for t in n.targets}  # fmt: skip
    for n in nodes:
        if isinstance(n, ast.Import):
            for a in n.names:
                imports[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            imports |= {a.asname or a.name: f"{n.module}.{a.name}" for a in n.names}
        elif isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load) and n not in fallback:
            rebound.add(n.id)
        elif isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            rebound.add(n.name)
    return {k: v for k, v in imports.items() if k not in rebound}


def _client_class(node: ast.AST | None, tree: ast.Module, scope: ast.AST) -> str | None:
    """The dotted name of what a call constructs, resolved through the imports (`anthropic.AsyncAnthropic`
    for `anthropic.AsyncAnthropic(...)`, `Claude(...)` or `a.AsyncAnthropic(...)`); None when it does not
    resolve to an import."""
    if not isinstance(node, ast.Call):
        return None
    parts, func = [], node.func
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    base = _imported_names(tree, scope).get(func.id) if isinstance(func, ast.Name) else None
    return ".".join([base, *reversed(parts)]) if base else None


def _built_client(tree: ast.Module, scope: ast.AST, receiver: ast.expr, source: str) -> ast.Call | None:
    """The constructor of the client the site's call is made on, when the function builds it from a class in
    CLIENT_CLASSES: the receiver itself (`AsyncAnthropic().messages.create(...)`) or the one plain assignment
    to the receiver's name in the function. None when the client is a parameter of the function with no
    default but None: its callers hand it one (`rewrite_callers`; `conformance` checks every one of them).
    Unsupported for every other client, so nothing ships on a client the rewrite does not know: another class
    (`AsyncAnthropicBedrock`, `AsyncAzureOpenAI`), a helper's result, an attribute (`self.client`), a
    module-level or class-level client, a parameter's default, one built by `with`, by an annotated
    assignment or more than once, or one that also serves other code in the function."""

    def unrecognised(expr: ast.AST, why: str = "") -> Unsupported:
        text = (ast.get_source_segment(source, expr) or ast.unparse(expr)).replace("\n", " ")
        return Unsupported(f"client type not recognised: `{text[:80]}`{why}")

    function = isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef)
    if isinstance(receiver, ast.Call):
        if function and _client_class(receiver, tree, scope) in CLIENT_CLASSES:
            return receiver
        raise unrecognised(receiver)
    if not function or not isinstance(receiver, ast.Name):
        raise unrecognised(receiver)
    name = receiver.id
    if name in {a.arg for a in [*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs]}:
        if not _no_default(scope, name):
            raise unrecognised(_default(scope, name), f" (the default of its `{name}` parameter)")
        return None

    def binds(n: ast.AST) -> bool:  # `n` binds the receiver's name
        if isinstance(n, ast.Assign):
            return any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
        if isinstance(n, ast.AnnAssign | ast.NamedExpr | ast.For | ast.AsyncFor):
            return isinstance(n.target, ast.Name) and n.target.id == name
        return (
            isinstance(n, ast.withitem)
            and isinstance(n.optional_vars, ast.Name)
            and n.optional_vars.id == name
        )

    local = [n for n in ast.walk(scope) if binds(n)]
    if not local:  # a module-level (or global) client
        raise unrecognised(receiver, " (built outside the function)")
    if len(local) != 1 or not isinstance(local[0], ast.Assign) or len(local[0].targets) != 1:
        why = "Offramp rewrites a client the function builds once, or one it is given"
        raise Unsupported(
            f"its client `{name}` is built by `with`, an annotated assignment or more than once: {why}"
        )
    value = local[0].value
    if _client_class(value, tree, scope) not in CLIENT_CLASSES:
        raise unrecognised(value)
    uses = [
        n for n in ast.walk(scope) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
    ]
    if uses != [receiver]:
        why = "which the Token Factory client would not answer the same way"
        raise Unsupported(f"its client `{name}` also serves other code in the function, {why}")
    return value


def _factory_call(built: ast.Call, factory: str, source: str) -> str:
    """The Token Factory client call replacing a client constructor: `api_key=` goes (the client reads
    NEBIUS_API_KEY); `timeout=` and `max_retries=` stay as the code wrote them (async client only)."""
    name = ast.unparse(built.func)
    if built.args or any(k.arg is None for k in built.keywords):
        raise Unsupported(f"its `{name}` client is built with positional or ** arguments")
    if other := [k.arg for k in built.keywords if k.arg not in ("api_key", *CLIENT_OPTIONS)]:
        names = ", ".join(f"`{k}=`" for k in other)
        raise Unsupported(
            f"its `{name}` client is built with {names}, which the Token Factory client does not take"
        )
    options = [
        f"{k.arg}={ast.get_source_segment(source, k.value)}" for k in built.keywords if k.arg != "api_key"
    ]
    if options and factory == "token_factory_client":
        option = options[0].partition("=")[0]
        raise Unsupported(f"its `{name}` client is built with `{option}=`, and the sync Token Factory client "
                          "takes no options")  # fmt: skip
    return f"{factory}({', '.join(options)})"


def _calls(fn: ast.AST) -> list[ast.Call]:
    kinds = ("chat.completions.create", "messages.create")
    return [c for c in ast.walk(fn) if isinstance(c, ast.Call) and _call_name(c.func).endswith(kinds)]


def _defs(tree: ast.AST, prefix: str = "") -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """(qualified name, node) of every function and method, e.g. ("AnthropicBackend.complete", ...)."""
    out = []
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            name = f"{prefix}{child.name}"
            out += [(name, child)] if not isinstance(child, ast.ClassDef) else []
            out += _defs(child, f"{name}.")
        else:
            out += _defs(child, prefix)
    return out


def _function(tree: ast.Module, site: CallSite) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The site's function: by qualified name ("Cls.method"), or by plain name when only one def has it."""
    hits = [n for q, n in _defs(tree) if q == site.function or q.rsplit(".", 1)[-1] == site.function]
    exact = [n for q, n in _defs(tree) if q == site.function]
    return exact[0] if len(exact) == 1 else (hits[0] if len(hits) == 1 else None)


def site_qualname(source: str, site: CallSite) -> str | None:
    """The qualified name of the site's function, told apart by the site's line range when several functions
    or methods share the scanner's plain name; None when that still leaves more than one."""
    if not site.function:
        return site.function
    hits = [(q, n) for q, n in _defs(ast.parse(source)) if q.rsplit(".", 1)[-1] == site.function]
    if len(hits) <= 1:
        return hits[0][0] if hits else site.function
    holding = [(q, n) for q, n in hits if n.lineno <= site.lines[0] <= n.end_lineno]
    if not holding:
        return None
    q = min(holding, key=lambda h: h[1].end_lineno - h[1].lineno)[0]  # the innermost
    return q if [name for name, _ in hits].count(q) == 1 else None


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


def _awaited(node: ast.expr) -> ast.expr:
    return node.value if isinstance(node, ast.Await) else node


def _map_reply(source: str, site: CallSite) -> str | None:
    """The migrated function reads the reply the OpenAI way, by code. An OpenAI site already does. For an
    Anthropic site (`r = client.messages.create(...)`, awaited or not): `"".join(b.text for b in r.content if
    b.type == "text")` and `r.content[0].text` become `(r.choices[0].message.content or "")`; a `for b in
    r.content:` loop over tool_use blocks walks `r.choices[0].message.tool_calls or []` (`b.type ==
    "tool_use"` -> `"function"`, `b.name` -> `b.function.name`, `b.input` -> `json.loads(b.function.
    arguments)`); an `Anthropic` (`AsyncAnthropic`) annotation of a parameter becomes `OpenAI`
    (`AsyncOpenAI`). None when the reply is used any other way: the model maps it then."""
    if site.sdk == "openai":
        return source
    tree = ast.parse(source)
    fn = _function(tree, site)
    calls = _calls(fn) if fn else []
    if len(calls) != 1:
        return None
    holder = next(
        (n for n in ast.walk(fn) if isinstance(n, ast.Assign) and _awaited(n.value) is calls[0]), None
    )
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
        if isinstance(arg.annotation, ast.Name) and arg.annotation.id in ("Anthropic", "AsyncAnthropic"):
            put(arg.annotation, arg.annotation.id.replace("Anthropic", "OpenAI"))
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
    fn = _function(tree, site) if site.function else None
    for n in tree.body if fn else []:
        if any(f is fn for f in ast.walk(n)):
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


def _file_block(path: str, mode: str, text: str) -> str:
    """One file as every prompt shows it (and as `_supersede` finds it again)."""
    return f"### {path} ({mode})\n```python\n{text.rstrip('\n')}\n```"


def _repair(
    failure: str, now: dict[str, str], visible: dict[str, str]
) -> tuple[str, list[str], frozenset[str]]:
    """The repair turn: the last failure, plus what to do about the two that a bare retry repeated live: each
    test the patch deleted, named, to be rewritten; after a SEARCH miss, the short test files it missed in
    offered whole. `failure` may carry the last checked try's failure ("...; last failure: ...").

    `now` is each editable file as the next blocks meet it, `visible` the text of its one copy in the
    conversation so far. A file whose copy is not its current text (a checked try's edits, Offramp's imports,
    the repo's formatting, a restore to the first text), or that a SEARCH missed, is shown again, so every
    SEARCH can be copied from the text it is applied to (live run 6: repairs searched a blend of the first
    text and their own earlier edits, or a file never shown again). The files the failure names are always
    shown; others while they fit in FILES_NOW_CHARS (at least one). A file left out whose copy is stale is
    named, and not editable this turn. Returns (the turn, the files it shows again, those left out stale).
    """
    text = REPAIR.format(failure=failure)
    removed = re.findall(r"patch removes tests from (\S+): \[([^\]]*)\]", failure)
    deleted = [f"`{path}::{name}`" for path, names in removed for name in re.findall(r"'(\w+)'", names)]
    if deleted:
        text += "\n\n" + DELETED.format(tests=", ".join(deleted))
    misses = {p for p in re.findall(r"search text (?:not found|found \d+ times) in (\S+?)(?::| \()", failure)
              if p in now}  # fmt: skip
    named = {p for p in now if re.search(rf"(?<![\w./-]){re.escape(p)}(?![\w/-]|\.\w)", failure)}
    again, left, size = [], [], 0
    for p in sorted(misses | {p for p in now if now[p] != visible[p]}, key=lambda p: (p not in named, p)):
        if p in named or not again or size + len(now[p]) <= FILES_NOW_CHARS:
            again, size = [*again, p], size + len(now[p])
        else:  # stale (a miss is named, so always shown): listed, and not editable this turn
            left.append(p)
    missed = [
        p for p in again if p in misses and is_test_path(p) and len(now[p].splitlines()) < WHOLE_FILE_LINES
    ]
    if missed:
        text += "\n\n" + WHOLE.format(files=", ".join(f"`{p}`" for p in missed), path=missed[0])
    if again:
        text += "\n\n## Files now\n" + "\n".join(_file_block(p, "editable", now[p]) for p in again)
    if left:
        text += "\n\n" + LEFT_OUT.format(files=", ".join(f"`{p}`" for p in left))
    return text, again, frozenset(left)


@dataclass
class _Copy:
    """One copy of an editable file that Offramp put in the conversation: its message, its text."""

    path: str
    turn: int  # index of the message holding it
    text: str
    superseded: bool = False


def _supersede(messages: list[dict[str, Any]], ledger: list[_Copy], paths: set[str]) -> list[dict[str, Any]]:
    """`messages` with the live copy of each of `paths` replaced by a one-line marker, so the conversation
    holds one copy of each file: the one under the latest "Files now" (copies of one file in several versions
    were what cs_01's repairs blended, live run 6). A copy is marked superseded in `ledger` only once its own
    block was found, and replaced, in its own message."""
    out = [dict(m) for m in messages]
    for c in ledger:
        block = _file_block(c.path, "editable", c.text)
        if c.path in paths and not c.superseded and out[c.turn]["content"].count(block) == 1:
            out[c.turn]["content"] = out[c.turn]["content"].replace(block, SUPERSEDED.format(path=c.path))
            c.superseded = True
    return out


def _stale(ledger: list[_Copy], now: dict[str, str], hidden: frozenset[str]) -> str | None:
    """The rule every model request meets, read from Offramp's own record of the copies it placed (never from
    the messages' text, which holds the target repo's files): each editable file not left out has exactly one
    live copy, and it is the file's current text. The first file that breaks it, else None."""
    for p, text in now.items():
        live = [c.text for c in ledger if c.path == p and not c.superseded]
        if p not in hidden and live != [text]:
            return p
    return None


def _live(ledger: list[_Copy]) -> dict[str, str]:
    """Each file's live copy's text (what the model sees of it now)."""
    return {c.path: c.text for c in ledger if not c.superseded}


def _resent(edits: list[Edit], applied: list[Edit], base: dict[str, str]) -> tuple[list[Edit], list[Edit]]:
    """(the blocks still to apply, the checked blocks the reply leaves out of a file it names). A block a
    checked try applied whose change is still in the text is not applied twice; one the model since undid is.
    Leaving a block out does not drop it: a block that restores the earlier text does."""

    def on(e: Edit) -> bool:  # its change is still in the text
        text = base.get(e.path, "")
        return e.search not in text or bool(e.replace) and e.replace in text

    named = {e.path for e in edits}
    fresh = [e for e in edits if e not in applied or not on(e)]
    return fresh, [a for a in applied if a.path in named and a not in edits and on(a)]


def _refusal(e: APIStatusError) -> str | None:
    """A model request Token Factory refused for what it is (a 4xx: too large, malformed), in Offramp's words:
    its HTTP status (its text is read for a size cause, never quoted). None for a transient failure (429, 5xx,
    what the SDK retries) or an auth failure (401/403): those fail the run, which can then be resumed."""
    if not 400 <= e.status_code < 500 or e.status_code in NOT_A_REFUSAL:
        return None
    big = e.status_code == 413 or any(w in f"{e.message} {e.body}".lower() for w in TOO_LARGE)
    return f"the model request was {'too large' if big else 'refused'} (HTTP {e.status_code})"


CHECK_CALLS = ("raises", "warns", "deprecated_call")  # pytest's, as a call or as a `with` item


def _checks(node: ast.AST, source: str) -> list[tuple[str, str, str]]:
    """The checks under `node` of `source`, each (its `ast.dump`, its whole statement's text, how a refusal
    names it): an `assert`, a call statement whose final name starts with `assert` (`m.assert_called_once()`,
    `self.assertEqual(...)`) or is `pytest.raises/warns/deprecated_call(...)`, and each such `with` item
    (its statement: the `with ...:` line)."""

    def final(e: ast.expr) -> str:
        f = e.func if isinstance(e, ast.Call) else None
        return getattr(f, "attr", None) or getattr(f, "id", None) or ""

    out = []
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Assert)
            or isinstance(n, ast.Expr)
            and (final(n.value).startswith("assert") or final(n.value) in CHECK_CALLS)
        ):
            seg = ast.get_source_segment(source, n) or ast.unparse(n)
            out.append((ast.dump(n), seg, seg if "\n" not in seg else ast.unparse(n)))
        elif isinstance(n, ast.With | ast.AsyncWith):
            line = source.splitlines()[n.lineno - 1] if n.lineno <= len(source.splitlines()) else ""
            out += [(ast.dump(i.context_expr), line, f"with {ast.unparse(i.context_expr)}")
                    for i in n.items if final(i.context_expr) in CHECK_CALLS]  # fmt: skip
    return out


def _quotes(search: str, statement: str) -> bool:
    """Whether `search` holds `statement` as whole lines, whitespace aside (review m1: `assert r` is not
    quoted by a SEARCH holding `assert rate == 2`)."""
    want, lines, k = " ".join(statement.split()), search.splitlines(), statement.count("\n") + 1
    return any(" ".join(" ".join(lines[i : i + k]).split()) == want for i in range(len(lines) - k + 1))


def _by_name(text: str, replace: str, search: str) -> tuple[str, list[str], list[str]] | None:
    """A test file's `text` with a REPLACE whose SEARCH matched nowhere applied by name: when the REPLACE
    parses as a module whose top-level statements are each a def or class whose name the file has exactly
    once at top level, or a statement the file already has (by `ast.dump`), each def replaces the file's
    own of that name, from its first decorator line (its `def` line when the REPLACE's carries none: the
    file's decorators stay, as a SEARCH starting at the `def` leaves them) to its end; nothing else changes.
    A def replaces only one of its own kind (def, async def, class). (the new text, the names, the checks
    (`_checks`) of the file's defs, and their decorators when the REPLACE carries its own, that the REPLACE
    drops and its SEARCH never quoted) or None; one dropped refuses the edit (the caller's miss names it): the
    model never saw that line, so it never chose to drop it (review I1: repair 2 shipped without `... is
    Queue.SHIPPING`; I3: a mock or unittest check, a `pytest.raises` block, a narrowed `parametrize`). Live
    run 7, cs_01: repair 2's SEARCH quoted a test that never existed, repair 3's the same one under the file's
    first top, each REPLACE that test whole. ponytail: top level only, so a method is replaced only with its
    whole class."""
    have, new = _parsed(text), _parsed(replace)
    kinds = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    if have is None or new is None:
        return None
    old = {n.name: n for n in have.body if isinstance(n, kinds)}
    tops = [n.name for n in have.body if isinstance(n, kinds)]
    defs = [n for n in new.body if isinstance(n, kinds)]
    names = [d.name for d in defs]
    same = {ast.dump(n) for n in have.body}
    if not defs or len(set(names)) < len(names) or any(tops.count(name) != 1 for name in names):
        return None
    if any(type(old[d.name]) is not type(d) for d in defs):
        return None
    if any(not isinstance(n, kinds) and ast.dump(n) not in same for n in new.body):
        return None

    def first(n: ast.AST) -> int:
        return n.decorator_list[0].lineno if n.decorator_list else n.lineno

    found = _parsed(textwrap.dedent(search))
    quoted = {c[0] for c in _checks(found, textwrap.dedent(search))} if found else set()
    quoted |= {
        ast.dump(x) for n in ast.walk(found or ast.Module(body=[])) for x in getattr(n, "decorator_list", [])
    }

    def decorated(file_def: ast.AST, its: ast.AST | None) -> list[tuple[str, str, str, set[str]]]:
        """The file def's decorators, each with the REPLACE def's (`its`) it may be kept among."""
        theirs = {ast.dump(x) for x in getattr(its, "decorator_list", [])}
        segs = [(x, ast.get_source_segment(text, x) or ast.unparse(x)) for x in file_def.decorator_list]
        return [
            (ast.dump(x), "@" + seg, "@" + (seg if "\n" not in seg else ast.unparse(x)), theirs)
            for x, seg in segs
        ]

    dropped = []
    for d in defs:
        o, kept = old[d.name], {c[0] for c in _checks(d, replace)}
        found = [(dump, statement, shown, kept) for dump, statement, shown in _checks(o, text)]
        found += (
            decorated(o, d) if d.decorator_list else []
        )  # the REPLACE's replace the file's: each kept or quoted
        if isinstance(o, ast.ClassDef):  # a whole class: each method's too, paired by name (re-review 2, n1)
            mine = dict(_defs(d))
            found += [f for qual, fn in _defs(o) for f in decorated(fn, mine.get(qual))]
        dropped += [shown for dump, statement, shown, kept_in in found
                    if dump not in kept_in | quoted and not _quotes(search, statement)]  # fmt: skip
    lines, mine = text.splitlines(keepends=True), replace.splitlines(keepends=True)
    for d in sorted(defs, key=lambda d: old[d.name].lineno, reverse=True):
        o = old[d.name]
        piece = mine[first(d) - 1 : d.end_lineno]
        piece[-1] = piece[-1].rstrip("\n") + "\n"
        lines[(first(o) if d.decorator_list else o.lineno) - 1 : o.end_lineno] = piece
    return "".join(lines), names, dropped


def _union(*found: dict[str, set[str]]) -> dict[str, set[str]]:
    """path -> names, merged."""
    out: dict[str, set[str]] = {}
    for f in found:
        for p, names in f.items():
            out.setdefault(p, set()).update(names)
    return out


def _by_name_note(path: str, names: tuple[str, ...]) -> str:
    """The run's warning for an edit `_apply` placed by name (`_by_name`)."""
    defs = ", ".join(f"`{x}`" for x in names)
    return f"an edit to `{path}` was applied by function name ({defs}) after its SEARCH matched nowhere"


def _apply(
    edits: list[Edit],
    files: dict[str, str],
    locked: frozenset[str] = frozenset(),
    hidden: frozenset[str] = frozenset(),
    notes: list[tuple[str, tuple[str, ...]]] | None = None,
) -> list[Problem]:
    """Apply the edit blocks in order; `locked` files (code that ships) are shown but read-only, `hidden` ones
    (left out of the last repair turn in their current text) are not editable, whatever a block's label. A
    block whose SEARCH matched nowhere in a test file is applied by name when its REPLACE is whole
    definitions (`_by_name`), and `notes` says so; the checks after it are the same."""
    shown = dict(files)
    seen = set()
    for e in edits:
        if (e.path, e.search, e.replace) in seen:  # the model repeated an edit: already applied
            continue
        seen.add((e.path, e.search, e.replace))
        if e.path not in files:
            return [Problem("not_editable", f"{e.path} is not editable for this site", e.path)]
        if e.whole:  # the same checks follow as for a block: no test removed, then the backstop
            lines = len(shown[e.path].splitlines())
            if e.path in locked:
                return [Problem("locked_file", f"{e.path} {READ_ONLY}", e.path)]
            if e.path in hidden:
                return [Problem("not_editable", f"{e.path} {NOT_SHOWN}", e.path)]
            if not is_test_path(e.path):
                return [
                    Problem("not_editable", f"{e.path}: a whole-file block is for test files only", e.path)
                ]
            if lines >= WHOLE_FILE_LINES:
                why = f"a whole-file block is for test files under {WHOLE_FILE_LINES} lines"
                return [
                    Problem(
                        "not_editable",
                        f"{e.path} has {lines} lines: give SEARCH/REPLACE blocks ({why})",
                        e.path,
                    )
                ]
            files[e.path] = e.replace
            continue
        n = files[e.path].count(e.search) if e.search else 0
        elsewhere = [p for p in files if p != e.path and e.search and files[p].count(e.search) == 1]
        if (
            n == 0 and len(elsewhere) == 1
        ):  # labelled with the wrong editable file: its text is unique elsewhere
            e = e.model_copy(update={"path": elsewhere[0]})
            n = 1
        if e.path in locked:
            return [Problem("locked_file", f"{e.path} {READ_ONLY}", e.path)]
        if e.path in hidden:  # labelled so, or relabelled to it above
            return [Problem("not_editable", f"{e.path} {NOT_SHOWN}", e.path)]
        near = _near(files[e.path], e.search) if n == 0 and e.search not in shown[e.path] else None
        # a slip in copying a long block still names one place; only the model's intended changes (its SEARCH
        # to its REPLACE) land there, never its mis-copied lines over the file's real ones
        merged = near and merge_onto(e.search, files[e.path][near[0] : near[1]], e.replace)
        if merged is not None and near:
            files[e.path] = files[e.path][: near[0]] + merged + files[e.path][near[1] :]
            continue
        named = (
            _by_name(files[e.path], e.replace, e.search)
            if n == 0 and e.search not in shown[e.path] and is_test_path(e.path)
            else None
        )  # a SEARCH an earlier block of this reply changed is a merge for the model, never a guess
        if named is not None and not named[2]:
            files[e.path] = named[0]
            if notes is not None:
                notes.append((e.path, tuple(named[1])))
            continue
        if n != 1:
            hint = (
                " (an earlier block already changed these lines: merge the two)"
                if e.search and (e.search in shown[e.path])
                else ""
            )
            where = "not found" if n == 0 else f"found {n} times"
            it = "it" if named and len(named[2]) == 1 else "them"
            drops = ", ".join(f"`{a}`" for a in named[2]) if named else ""
            why = (
                f"; applied by name, this edit would drop {drops}, which your SEARCH never quoted: "
                f"keep {it}, or quote {it} to change {it}"
                if named
                else ""
            )
            return [
                Problem(
                    "search_miss", f"search text {where} in {e.path}{hint}: {e.search[:80]!r}{why}", e.path
                )
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


def _free_reads(tree: ast.Module) -> set[str]:
    """Names read where no import in their own function (or one enclosing it) binds them."""
    out, stack = set(), [(tree, frozenset())]
    while stack:
        node, local = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            local = local | {(a.asname or a.name).split(".")[0] for n in ast.walk(node)
                             if isinstance(n, ast.Import | ast.ImportFrom) for a in n.names}  # fmt: skip
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in local:
            out.add(node.id)
        stack.extend((child, local) for child in ast.iter_child_nodes(node))
    return out


def add_imports(source: str, client: str, extra: dict[str, str] | None = None) -> str:
    """Import AUTO_IMPORTS names (and `extra`'s, name -> import line) the module uses but never binds, after
    its last top-level import; `client` is the module path of the Token Factory client."""
    tree = ast.parse(source)
    bound = _bound(tree)
    used = _free_reads(tree)
    known = {**{name: line.format(client=client) for name, line in AUTO_IMPORTS.items()}, **(extra or {})}
    missing = [line for name, line in known.items() if name in used - bound]
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


def _dotted(path: str) -> list[str]:
    """Every dotted name a module file may be imported as: `a.b.c`, `b.c` and `c` for a/b/c.py (`a.b`, `b`
    for a/b/__init__.py). Which folder is an import root is not known here, so every suffix counts; a module
    elsewhere that shares one only errs toward a manual review."""
    parts = Path(path).with_suffix("").parts
    parts = parts[:-1] if parts[-1:] == ("__init__",) else parts
    return [".".join(parts[i:]) for i in range(len(parts))]


def _imported(node: ast.ImportFrom, path: str) -> str:
    """The module a `from ... import` in the file at `path` names; a relative one resolved against the file's
    own folder, as a dotted path from the repository root."""
    if not node.level:
        return node.module or ""
    package = Path(path).parent.parts
    return ".".join([*package[: max(len(package) - node.level + 1, 0)], *filter(None, [node.module])])


def _parsed(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _targets(sources: dict[str, str], site_file: str, function: str) -> set[str]:
    """Every dotted name the site's module-level `function` is importable as: `<its module>.<function>`,
    and `<module>.<name>` for each module that imports it as `name` (a re-export, such as a package
    `__init__`'s `from .summarize import summarize`, or a star import), to a fixed point."""
    targets = {f"{d}.{function}" for d in _dotted(site_file)}
    trees = [(path, _parsed(text)) for path, text in sources.items()]
    imports = [
        (path, n) for path, tree in trees if tree for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    ]
    size = 0
    while size != len(targets):
        size = len(targets)
        for path, node in imports:
            base = _imported(node, path)
            for a in node.names:
                if a.name == "*":
                    names = [t.rpartition(".")[2] for t in targets if t.rpartition(".")[0] == base]
                else:
                    names = [a.asname or a.name] if f"{base}.{a.name}" in targets else []
                targets |= {f"{d}.{name}" for d in _dotted(path) for name in names}
    return targets


def _module_aliases(tree: ast.Module, path: str) -> dict[str, str]:
    """Local name -> the dotted name an import binds it to, perhaps a module (`s` -> `app.summarize` after
    `import app.summarize as s` or `from app import summarize as s`; `app` -> `app` after `import app.x`)."""
    modules = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.partition(".")[0]
                modules[a.asname or top] = a.name if a.asname else top
        elif isinstance(node, ast.ImportFrom):
            base = _imported(node, path)
            modules |= {a.asname or a.name: f"{base}.{a.name}" for a in node.names if a.name != "*"}
    return modules


def _chain(n: ast.expr, modules: dict[str, str]) -> str | None:
    """`app.summarize.summarize` for an attribute chain on an imported name (`s.summarize`), else None."""
    attrs = []
    while isinstance(n, ast.Attribute):
        attrs.append(n.attr)
        n = n.value
    return (
        ".".join([modules[n.id], *reversed(attrs)]) if isinstance(n, ast.Name) and n.id in modules else None
    )


def _references(path: str, tree: ast.Module, targets: set[str], defined: str | None) -> list[ast.expr]:
    """The expressions in a module that refer to the function `targets` names, resolved through the module's
    imports, never by name alone: a name a `from ... import` (or a star import) binds to it, an attribute
    chain through an imported module (`s.summarize` after `import app.summarize as s`), and its own name in
    the module that defines it (`defined`)."""
    names, modules = {defined} - {None}, _module_aliases(tree, path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = _imported(node, path)
            for a in node.names:
                if a.name == "*":
                    names |= {t.rpartition(".")[2] for t in targets if t.rpartition(".")[0] == base}
                elif f"{base}.{a.name}" in targets:
                    names.add(a.asname or a.name)

    def refers(n: ast.AST) -> bool:
        if isinstance(n, ast.Name):
            return isinstance(n.ctx, ast.Load) and n.id in names
        return isinstance(n, ast.Attribute) and _chain(n, modules) in targets

    return [n for n in ast.walk(tree) if refers(n)]


def _function_references(
    sources: dict[str, str], site_file: str, function: str
) -> dict[str, tuple[ast.Module | None, list[ast.expr]]]:
    """path -> (its module, None when it does not parse; the expressions in it that refer to the site's
    function). A module-level function's are resolved through imports (`_targets`, `_references`). A
    method's or a nested function's, which only an object or the enclosing code reaches, are matched by
    name (`<name>`, `x.<name>`), which errs toward a manual review."""
    last = function.rpartition(".")[2]
    targets = set() if "." in function else _targets(sources, site_file, function)

    def by_name(n: ast.AST) -> bool:
        if isinstance(n, ast.Name):
            return isinstance(n.ctx, ast.Load) and n.id == last
        return isinstance(n, ast.Attribute) and n.attr == last

    out = {}
    for path, text in sources.items():
        tree = _parsed(text)
        if tree is None:
            out[path] = (None, [])
        elif "." in function:
            out[path] = (tree, [n for n in ast.walk(tree) if by_name(n)])
        else:
            out[path] = (tree, _references(path, tree, targets, function if path == site_file else None))
    return out


def rewrite_callers(sources: dict[str, str], site_file: str, function: str, client: str) -> dict[str, str]:
    """In each module of `sources` (path -> source: the repo's non-test code, the site's own module
    included): an `OpenAI()` / `Anthropic()` built right in a call of the migrated function, the call found
    through the module's imports (aliases too, `_function_references`), becomes `token_factory_client()`
    (an `AsyncOpenAI()` / `AsyncAnthropic()`: `token_factory_async_client()`), imported from `client`.
    Returns the modules it changed."""
    out = {}
    for path, (tree, refs) in _function_references(sources, site_file, function).items():
        heads, source, spans = {id(r) for r in refs}, sources[path], []
        lines = source.splitlines(keepends=True)
        for call in ast.walk(tree) if tree else ():
            if not isinstance(call, ast.Call) or id(call.func) not in heads:
                continue
            for arg in [*call.args, *(k.value for k in call.keywords)]:
                cls = _client_class(arg, tree, tree)
                if cls in CLIENT_CLASSES and not arg.args and not arg.keywords:
                    start = _offset(lines, arg.lineno, arg.col_offset)
                    spans.append(
                        (start, _offset(lines, arg.end_lineno, arg.end_col_offset), CLIENT_CLASSES[cls])
                    )
        for start, end, factory in sorted(spans, reverse=True):
            source = source[:start] + f"{factory}()" + source[end:]
        if spans:
            out[path] = add_imports(source, client)
    return out


def _import_bindings(tree: ast.AST | None) -> set[str]:
    """Every name an import in `tree` binds (`from a import B as C` -> C, `import a.b` -> a)."""
    if tree is None:
        return set()
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import | ast.ImportFrom)]
    return {a.asname or (a.name if isinstance(n, ast.ImportFrom) else a.name.split(".")[0])
            for n in imports for a in n.names}  # fmt: skip


def drop_orphaned_imports(
    before: str, after: str, dropped: list[str] | None = None, keep: frozenset[str] = frozenset()
) -> str:
    """`after` (a file Offramp's code or a model edit changed from `before`) without the import names the
    change orphaned: read in `before` and nowhere in `after` (a name, or a string equal to it: `__all__`, a
    string annotation), or added by the change to an import and read nowhere in `after` (fix round 2's replay:
    cs_04's repair imported `NEMOTRON_FEW_SHOT` and never used it). A name imported before stays unless read
    before; a `__future__` import always stays. Only those names go (`from anthropic import Anthropic,
    APIError` keeps `APIError`); a statement left with none goes whole, and a run of such statements takes the
    blank line below it when it stood between two (or at the top). Every other line is byte-identical: an
    import unused before is the repo's own lint state, and a dotted `import a.b` may be there for its side
    effect, so both stay. Live run 7: three Anthropic sites each rewired one `Anthropic()` in helphub/cli.py;
    merged, nothing read the import, the repo's ruff failed (F401) and cs_06 dropped. This replaced a lint
    autofix that removed every unused import, the repo's own included (review I5). `dropped` gets the names
    removed (for `drop_dead_patches`); `keep` names stay (another module takes them from this one:
    `_used_elsewhere`). ponytail: an import sharing its line with another statement, or the only statement of
    a block, stays."""
    old, new = _parsed(before), _parsed(after)
    if old is None or new is None:
        return after

    def reads(tree: ast.Module, name: str) -> bool:
        return any(isinstance(n, ast.Name) and n.id == name or isinstance(n, ast.Constant) and n.value == name
                   for n in ast.walk(tree))  # fmt: skip

    had = _import_bindings(old)
    lines = after.splitlines(keepends=True)
    cuts, rows, names = [], [], []  # (start, end) offsets of names; whole statements' (first, last) lines
    for node in ast.walk(new):
        if not isinstance(node, ast.Import | ast.ImportFrom) or getattr(node, "level", 0):
            continue
        frm = isinstance(node, ast.ImportFrom)
        bound = [a.asname or (a.name if frm else a.name.split(".")[0]) for a in node.names]
        gone = [
            (frm or a.asname or "." not in a.name)
            and not (frm and node.module == "__future__")  # never read, yet never unused
            and not reads(new, b)
            and (reads(old, b) or b not in had)  # it lost its last use, or the change added it unused
            and b not in keep
            for a, b in zip(node.names, bound, strict=True)
        ]
        if all(gone):  # the whole statement, on lines of its own
            head = lines[node.lineno - 1][: node.col_offset]
            tail = lines[node.end_lineno - 1].encode()[node.end_col_offset :].decode().strip()
            if head.strip() or tail[:1] not in ("", "#"):  # shares its lines with another statement
                continue
            rows.append((node.lineno - 1, node.end_lineno - 1))
            names += bound
            continue
        at = [(_offset(lines, a.lineno, a.col_offset), _offset(lines, a.end_lineno, a.end_col_offset))
              for a in node.names]  # fmt: skip
        # each name with a comma: up to the next kept name, else from the end of the last kept one
        for i in (i for i, g in enumerate(gone) if g):
            cuts.append((at[i][0], at[i + 1][0]) if not all(gone[i:]) else (at[i - 1][1], at[i][1]))
            names.append(bound[i])
    runs: list[list[int]] = []  # adjacent whole statements, as one run of lines
    for first, last in sorted(rows):
        if runs and first == runs[-1][1] + 1:
            runs[-1][1] = last
        else:
            runs.append([first, last])
    for first, last in runs:
        blank = last + 1 < len(lines) and not lines[last + 1].strip()
        last += blank and (first == 0 or not lines[first - 1].strip())  # the run stood between two blanks
        cuts.append((sum(map(len, lines[:first])), sum(map(len, lines[: last + 1]))))
    for start, end in sorted(cuts, reverse=True):
        after = after[:start] + after[end:]
    if _parsed(after) is None:
        return "".join(lines)
    if dropped is not None:
        dropped += names
    return after


def _drop_unused(
    path: str, before: str, after: str, others: dict[str, str], kept: dict[str, None], who: str
) -> tuple[str, set[str]]:
    """`drop_orphaned_imports` on `who`'s change of `path`, keeping a name another module of `others` still
    takes from it (`_used_elsewhere`, review I2), said in `kept` (the run's warnings): (the text, the names
    dropped)."""
    names: list[str] = []
    out = drop_orphaned_imports(before, after, names)
    if names and (uses := _used_elsewhere(path, out, set(names), others)):
        kept.update(dict.fromkeys(
            f"kept the `{n}` import in `{path}`, which {who} left unused: {why}" for n, why in uses.items()
        ))  # fmt: skip
        names = []
        out = drop_orphaned_imports(before, after, names, keep=frozenset(uses))
    return out, set(names)


def _patch_target(node: ast.AST, modules: dict[str, str]) -> tuple[str, str, bool] | None:
    """(dotted module, name, whether as a decorator it hands the test a mock) for a patch of a module's
    attribute: `<fixture>.setattr(module, "N", value)` or `.setattr("pkg.mod.N", value)`, `patch("pkg.mod.N",
    ...)` and `patch.object(module, "N", ...)`, bare or through `mock.`, also started (`.start()`). A module
    argument is resolved through the file's imports (`modules`), never by its name; None for anything else."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "start":
        node = node.func.value
    if not isinstance(node, ast.Call):
        return None
    f, args, new = node.func, node.args, any(k.arg == "new" for k in node.keywords)
    text = [a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else None for a in args]
    kind = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
    on = getattr(f, "value", None)
    if kind == "object" and (getattr(on, "id", None) or getattr(on, "attr", None)) != "patch":
        return None
    if (kind == "setattr" and isinstance(f, ast.Attribute) and len(args) == 3 or kind == "object"
            and len(args) >= 2) and text[1] and (module := _chain(args[0], modules)):  # fmt: skip
        return module, text[1], len(args) < 3 and not new
    if (
        kind == "setattr" and isinstance(f, ast.Attribute) and len(args) == 2 or kind == "patch" and args
    ) and (text[0] and "." in text[0]):
        module, _, name = text[0].rpartition(".")
        return module, name, len(args) < 2 and not new
    return None


def _used_elsewhere(path: str, text: str, names: set[str], sources: dict[str, str]) -> dict[str, str]:
    """Each of `names` (SDK names about to leave the module at `path`, whose text is `text`) that another
    module of `sources` still takes from it -> how: `from <it> import N` (relative too), `from <it> import *`
    when it exports N (no `__all__`, or one naming N), or `<it>.N` read through an import of it (review I2:
    the cleanup is per file, and `from app.llm import Anthropic` in app/admin.py lost its name). Every
    suffix of the module's dotted name counts (`_dotted`): a shared one errs toward keeping the import."""
    dotted, tree = set(_dotted(path)), _parsed(text)
    listed = [n.value for n in (tree.body if tree else []) if isinstance(n, ast.Assign)
              and any(getattr(t, "id", None) == "__all__" for t in n.targets)]  # fmt: skip
    exported = (
        {e.value for v in listed for e in ast.walk(v) if isinstance(e, ast.Constant)} if listed else None
    )
    reads = {f"{d}.{n}": n for d in dotted for n in names}
    out: dict[str, str] = {}
    for q, source in sorted(sources.items()):
        other = _parsed(source) if q != path else None
        aliases = _module_aliases(other, q) if other else {}
        for node in ast.walk(other) if other else ():
            if isinstance(node, ast.ImportFrom) and _imported(node, q) in dotted:
                for a in node.names:
                    if a.name in names:
                        out.setdefault(a.name, f"`{q}` imports it")
                    elif a.name == "*":
                        for n in names if exported is None else names & exported:
                            out.setdefault(n, f"`{q}` imports `*` from it")
            elif isinstance(node, ast.Attribute) and (chain := _chain(node, aliases)) in reads:
                out.setdefault(reads[chain], f"`{q}` reads `{chain}`")
    return out


def drop_dead_patches(
    source: str, path: str, dropped: dict[str, set[str]], tracked: Iterable[str] = ()
) -> tuple[str, dict[str, int]]:
    """A test file (`path`) without its patches of a module attribute that `drop_orphaned_imports` removed
    (`dropped`: module path -> names), which would now raise: a statement (`monkeypatch.setattr(cli,
    "Anthropic", fake)`, a started `patch`) or a decorator that passes no mock to the test. Only in a test
    that also patches the Token Factory client on that module (mirror_fakes' twin), so the test still fakes
    the call. Returns (the source, `module.name` -> patches removed). Live run 7, replayed: tests/test_cli.py
    patched `cli.Anthropic` beside its twins; merged, cli.py no longer imported it and pytest failed.
    ponytail: a `with patch(...)` block, a decorator that injects a mock, a statement sharing its line or the
    only one of its block stay, for the repair or a manual review."""
    tree = _parsed(source)
    if tree is None or not dropped:
        return source, {}
    modules = _module_aliases(tree, path)
    owners: dict[str, set[str]] = {}  # dotted name -> the tracked modules it may name
    for m in {*tracked, *dropped}:
        for d in _dotted(m):
            owners.setdefault(d, set()).add(m)
    # the full name, or a shorter one no other tracked module shares (review M2: `import cli` may be tools/)
    by_dotted = {
        d: m for m in dropped for d in _dotted(m) if d in (_dotted(m)[0], _module_name(m)) or owners[d] == {m}
    }
    lines = source.splitlines(keepends=True)
    gone: dict[int, tuple[int, str]] = {}  # first line -> (last line, `module.name`)
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)):
        stmts = [s for s in ast.walk(fn) if isinstance(s, ast.Expr)]
        found = [
            (n, t)
            for n in [*fn.decorator_list, *stmts]
            if (t := _patch_target(getattr(n, "value", n), modules))
        ]
        twins = {by_dotted[m] for _, (m, name, _) in found if m in by_dotted and name in FACTORIES}
        lone = {id(b[0]) for n in ast.walk(fn) for f in ("body", "orelse", "finalbody")
                if isinstance(b := getattr(n, f, None), list) and len(b) == 1}  # fmt: skip
        for node, (module, name, injects) in found:
            m = by_dotted.get(module)
            alone = (
                not lines[node.lineno - 1][: node.col_offset].strip(" \t@")
                and not lines[node.end_lineno - 1][node.end_col_offset :].split("#")[0].strip()
            )
            if (
                m in twins
                and name in dropped[m]
                and alone
                and id(node) not in lone
                and not (injects and node in fn.decorator_list)
            ):
                gone[node.lineno] = (node.end_lineno, f"{_module_name(m)}.{name}")
    drop = {i for first, (last, _) in gone.items() for i in range(first, last + 1)}
    out = "".join(line for i, line in enumerate(lines, 1) if i not in drop)
    if not gone or _parsed(out) is None:
        return source, {}
    counts: dict[str, int] = {}
    for _, key in gone.values():
        counts[key] = counts.get(key, 0) + 1
    return out, counts


def mirror_fakes(
    source: str, modules: set[str], old: str = "OpenAI", new: str = "token_factory_client"
) -> str:
    """In a test: `monkeypatch.setattr(<module>, "<old>", <fake>)` for a module whose code now builds
    `<new>()` (`token_factory_client()`, or `token_factory_async_client()` for an async client) gets a twin
    for `<new>`, with the same fake. An OpenAI fake fits the Token Factory client as it is; an Anthropic fake
    does once `convert_fakes` taught it the OpenAI call."""
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
                and (call.args[0].id, new) not in present):  # fmt: skip
            head = ast.get_source_segment(source, call.func)
            fake = ast.get_source_segment(source, call.args[2])
            indent = re.match(r"[ \t]*", lines[stmt.lineno - 1]).group(0)
            inserts.append((stmt.end_lineno, f'{indent}{head}({call.args[0].id}, "{new}", {fake})\n'))
    for row, text in sorted(inserts, reverse=True):
        lines.insert(row, text)
    return "".join(lines)


def reach_handles(source: str, function: str) -> set[str]:
    """What a test names to reach `function` through this caller module: the function itself, the key of a
    dict entry holding the call (a command table), the strings an enclosing `if` compares against (an
    argument dispatch), or, when neither exists, the name of the function that makes the call."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    handles = {function}
    for call in ast.walk(tree):
        if (
            not isinstance(call, ast.Call)
            or getattr(call.func, "id", getattr(call.func, "attr", None)) != function
        ):
            continue
        keys, node, outer = set(), call, None
        while node in parents:
            parent = parents[node]
            if isinstance(parent, ast.Dict) and node in parent.values:
                key = parent.keys[parent.values.index(node)]
                keys |= {key.value} if isinstance(key, ast.Constant) and isinstance(key.value, str) else set()
            elif isinstance(parent, ast.If) and node in parent.body:
                keys |= {c.value for c in ast.walk(parent.test)
                         if isinstance(c, ast.Constant) and isinstance(c.value, str)}  # fmt: skip
            elif isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef) and outer is None:
                outer = parent.name
            node = parent
        handles |= keys or ({outer} if outer else set())
    return handles


def _twin_dumps(lines: set[str]) -> set[str]:
    """The AST of each fake line mirror_fakes added: it survives the repo's formatter splitting the line."""
    out = set()
    for line in lines:
        try:
            node = ast.parse(line.strip()).body[0]
        except (SyntaxError, IndexError):
            continue
        out |= {ast.dump(node.value)} if isinstance(node, ast.Expr) else set()
    return out


def _tests(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            and n.name.startswith("test")]  # fmt: skip


def prune_twins(source: str, twins: set[str], handles: set[str]) -> str:
    """Drop the `token_factory_client` fakes Offramp added (`twins`) from test functions that never name a
    handle of the migrated function (see `reach_handles`): those tests do not reach it."""
    dumps, tree, drop = _twin_dumps(twins), ast.parse(source), set()
    for fn in _tests(tree):
        named = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        named |= {getattr(n, "id", None) or getattr(n, "attr", None) for n in ast.walk(fn)}
        if (
            named & handles
        ):  # ponytail: a test reaching it only through a helper loses the fake; checks catch it
            continue
        for stmt in ast.walk(fn):
            if isinstance(stmt, ast.Expr) and ast.dump(stmt.value) in dumps:
                drop |= set(range(stmt.lineno, stmt.end_lineno + 1))
    lines = source.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines, 1) if i not in drop)


def _parameter_values(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, set[str]]:
    """parameter name -> the string values a `pytest.mark.parametrize` decorator gives it."""
    out: dict[str, set[str]] = {}
    for d in fn.decorator_list:
        if not (isinstance(d, ast.Call) and _call_name(d.func).endswith("parametrize") and len(d.args) >= 2):
            continue
        spec = d.args[0]
        if isinstance(spec, ast.Constant):
            names = [n.strip() for n in str(spec.value).split(",")]
        else:
            names = [e.value for e in getattr(spec, "elts", []) if isinstance(e, ast.Constant)]
        for row in getattr(d.args[1], "elts", []):
            cells = row.elts if len(names) > 1 and isinstance(row, ast.Tuple) else [row]
            for name, cell in zip(names, cells, strict=False):
                if isinstance(cell, ast.Constant) and isinstance(cell.value, str):
                    out.setdefault(name, set()).add(cell.value)
    return out


def unify_twins(source: str, sites: list[tuple[set[str], set[str]]]) -> str:
    """In a merged test file: a parametrised test that sets `token_factory_client` once per migrated site
    with different fakes (Offramp's twins: the last one would serve every case) gets one setattr whose fake
    is chosen per case, from the parameter naming each site's handle. `sites`: (twin lines, handles) per
    site. Left as it is when no single parameter tells the sites apart."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []  # (first line, last line, replacement), 1-based

    def seg(node: ast.AST) -> str:
        return ast.get_source_segment(source, node)

    for fn in _tests(tree):
        stmts = [st for st in fn.body if isinstance(st, ast.Expr)]
        twin_of: dict[str, ast.Expr] = {}  # a fake's AST -> the first statement setting it
        handles_of: dict[str, set[str]] = {}  # a fake's AST -> handles of the sites that use it
        for twins, handles in sites:
            dumps = _twin_dumps(twins)
            for st in (st for st in stmts if ast.dump(st.value) in dumps):
                fake = ast.dump(st.value.args[2])
                twin_of.setdefault(fake, st)
                handles_of[fake] = handles_of.get(fake, set()) | handles
        values = _parameter_values(fn)
        arg = next((a for a, vs in values.items() if all(vs & h for h in handles_of.values())), None)
        if len(twin_of) < 2 or arg is None:
            continue
        picked = [values[arg] & h for h in handles_of.values()]
        if sum(map(len, picked)) != len(
            set().union(*picked)
        ):  # a case two fakes claim: not for code to settle
            continue
        order = sorted(twin_of, key=lambda f: twin_of[f].lineno)  # the first set fake becomes the default
        rows = []
        for fake in order[1:]:
            vs = [_str(v) for v in sorted(values[arg] & handles_of[fake])]
            choices = ", ".join(vs) + ("," if len(vs) == 1 else "")
            rows += [f"({seg(twin_of[fake].value.args[2])})", f"if {arg} in ({choices})", "else"]
        rows.append(f"({seg(twin_of[order[0]].value.args[2])}),")
        folded: list[str] = []
        for row in rows:  # "else" joins the row after it
            if folded and folded[-1] == "else":
                folded[-1] = f"else {row}"
            else:
                folded.append(row)
        call, pad = twin_of[order[0]].value, " " * twin_of[order[0]].col_offset
        head = [f"{seg(call.func)}(", f"    {seg(call.args[0])},", f"    {seg(call.args[1])},"]
        text = "".join(f"{pad}{row}\n" for row in [*head, *(f"    {r}" for r in folded), ")"])
        last = max(twin_of.values(), key=lambda st: st.lineno)
        edits += [(st.lineno, st.end_lineno, text if st is last else "") for st in twin_of.values()]
    for first, last_line, text in sorted(edits, reverse=True):
        lines[first - 1 : last_line] = [text] if text else []
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
            steps.append(re.sub(r"\bMessages\b", "Completions", text))  # Completions imported when missing
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


def _drift_text(off: dict[str, list[str]]) -> str:
    return "; ".join(f"{sid}: {', '.join(problems)}" for sid, problems in off.items())


def _test_bodies_changed(before: str, after: str) -> set[str]:
    """Names of the test functions whose code differs between two versions of a module."""

    def tests(source: str) -> dict[str, str]:  # by qualified name: same-named methods of two classes differ
        return {q: ast.dump(n) for q, n in _defs(ast.parse(source)) if n.name.startswith("test")}

    old, new = tests(before), tests(after)
    return {name for name in old.keys() & new.keys() if old[name] != new[name]}


SKIP_MARKS = ("skip", "skipif", "xfail")  # pytest.mark.<x>
SKIP_DECORATORS = ("skip", "skipIf", "skipUnless", "expectedFailure")  # unittest's, bare or `unittest.<x>`
SKIP_CALLS = ("pytest.skip", "pytest.xfail", "pytest.importorskip")  # and any `<x>.skipTest(...)`


def _skip_marks(source: str) -> dict[str, set[str]]:
    """Each test (function or method by qualified name, `Test*` class) and a module's `pytestmark` -> the
    skips and xfails it carries: a `pytest.mark.skip/skipif/xfail` mark (a decorator, or in a `pytestmark`),
    a unittest `skip*` / `expectedFailure` decorator, a `pytest.skip/xfail/importorskip(...)` or
    `.skipTest(...)` call in its body."""
    tree = _parsed(source)

    def marks(e: ast.AST) -> set[str]:
        found = set()
        for n in ast.walk(e):
            parts = ast.unparse(n).split(".") if isinstance(n, ast.Attribute | ast.Name) else []
            if (
                parts[-2:-1] == ["mark"]
                and parts[-1] in SKIP_MARKS
                or parts
                and parts[-1] in SKIP_DECORATORS
                and (len(parts) == 1 or parts[0] == "unittest")
            ):
                found.add(".".join(parts))
        return found

    def calls(fn: ast.AST) -> set[str]:
        heads = (ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call))
        return {h for h in heads if h in SKIP_CALLS or h.endswith(".skipTest")}

    def pytestmark(body: list[ast.stmt]) -> set[str]:
        return set().union(*(marks(n.value) for n in body if isinstance(n, ast.Assign)
                             and any(getattr(t, "id", None) == "pytestmark" for t in n.targets)))  # fmt: skip

    out = {"pytestmark": pytestmark(tree.body)} if tree else {}
    for qual, fn in _defs(tree) if tree else []:
        if fn.name.startswith("test"):
            out[qual] = set().union(calls(fn), *(marks(d) for d in fn.decorator_list))
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)) if tree else ():
        if cls.name.startswith("Test"):
            out[cls.name] = pytestmark(cls.body).union(*(marks(d) for d in cls.decorator_list))
    return out


def _new_skips(before: str, after: str) -> dict[str, list[str]]:
    """The skips and xfails (`_skip_marks`) a test or `pytestmark` has in `after` and lacked in `before`: test
    -> them. A model edit may not switch a test off (review I4: `pytest.skip()` or `@pytest.mark.skip` turned
    a red test green, and the sandbox counted it as passed)."""
    old, new = _skip_marks(before), _skip_marks(after)
    return {k: sorted(v - old.get(k, set())) for k, v in new.items() if v - old.get(k, set())}


def _skips_text(path: str, skips: dict[str, list[str]]) -> str:
    gains = "; ".join(f"`{k}` gains {', '.join(f'`{m}`' for m in v)}" for k, v in skips.items())
    return f"patch makes tests skip in {path}: {gains}; never switch a test off: make it pass"


def _skip_count(output: str) -> int:
    """Skipped plus xfailed tests in pytest's summary line (`3 passed, 2 skipped, 1 xfailed in 0.4s`)."""
    counts = r"\b\d+ (passed|failed|skipped|xfailed|xpassed|errors?)\b.* in [\d.]+s"
    summary = [ln for ln in output.splitlines() if re.search(counts, ln)]
    return sum(int(n) for n in re.findall(r"(\d+) (?:skipped|xfailed)\b", summary[-1])) if summary else 0


def _no_more_skips(result: CheckResult, baseline: int) -> CheckResult:
    """A green run that skips (or xfails) more tests than the untouched repo's is red: the backstop under
    `_new_skips` for a skip no rule sees (a fixture's, a conftest's)."""
    more = _skip_count(result.output) - baseline if result.ok else 0
    return (
        CheckResult(ok=False, mode=result.mode, output=f"the patch makes {more} more tests skip")
        if more > 0
        else result
    )


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


# ---------------------------------------------------------------- the backstop (code that ships)


def _root(node: ast.AST) -> str | None:
    """The name a call receiver ultimately reads from: `X` of `X[0].a`."""
    while isinstance(node, ast.Attribute | ast.Subscript | ast.Starred | ast.Call):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _py_files(repo: Path) -> list[Path]:
    """The copy's .py files, outside the folders a copy leaves out (e.g. the `.venv` a local check makes)."""
    return [f for f in repo.rglob("*.py") if not set(NOT_COPIED) & set(f.relative_to(repo).parts[:-1])]


def shipped_code(repo: Path) -> dict[str, str]:
    """Every file of the copy that ships as code -> its text: each non-test .py file (for a link, the file it
    points to), and each test-path file one of them imports, directly or through another (with the packages'
    `__init__.py` Python runs first). The model may edit none of them."""
    root, real = repo.resolve(), {}
    for f in _py_files(repo):
        target = f.resolve()
        if target.is_file() and target.is_relative_to(root):
            real[f.relative_to(repo).as_posix()] = target.relative_to(root).as_posix()
    regular = set(real.values())
    todo, out = [real[p] for p in real if not is_test_path(p)], {}
    while todo:
        p = todo.pop()
        if p in out:
            continue
        out[p] = (repo / p).read_text()
        for q in _imported_files(out[p], p, regular):
            parts = q.split("/")[:-1]
            packages = ("/".join([*parts[:k], "__init__.py"]) for k in range(1, len(parts) + 1))
            todo += [q, *(i for i in packages if i in regular)]
    return out


def _same_code(a: str | None, b: str | None) -> bool:
    if a == b:
        return True
    try:
        return a is not None and b is not None and ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def backstop(repo: Path, reference: dict[str, str | None]) -> dict[str, str]:
    """Every file that ships as code, as it is on disk, against `reference` (the tree Offramp's code wrote):
    path -> why it is not that tree (its AST differs, it is new, it is gone). The disk is compared, not the
    model's edits, so whatever wrote the file (an edit, a formatter, a test run) is caught."""
    paths = {f.relative_to(repo).as_posix(): f for f in _py_files(repo)}
    present = {p for p, f in paths.items() if f.is_file() and not f.is_symlink() and not is_test_path(p)}
    out = {}
    for p in sorted(present | set(reference)):
        now = (repo / p).read_text(errors="replace") if (repo / p).is_file() else None
        if p not in reference:
            out[p] = f"{p} is a new file outside the tests"
        elif now is None:
            out[p] = f"{p} was removed"
        elif not _same_code(now, reference[p]):
            out[p] = f"{p} is not the code Offramp wrote"
    return out


def _restore(repo: Path, reference: dict[str, str | None], paths) -> None:
    """Put the files the backstop flagged back as `reference` has them; a new one is removed."""
    for p in paths:
        f = repo / p
        f.unlink(missing_ok=True)  # first: never written through a link a test run may have left there
        if reference.get(p) is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(reference[p])


def source_carried(source: str, site: CallSite) -> dict[str, ast.expr]:
    """The transport options (CARRIED_KWARGS) the site's LLM call passes, as the code has them."""
    calls = _calls((_function(ast.parse(source), site) if site.function else None) or ast.parse(source))
    kwargs = {k.arg: k.value for k in calls[0].keywords if k.arg} if len(calls) == 1 else {}
    return {k: v for k, v in kwargs.items() if k in CARRIED_KWARGS}


def probe_parameters(source: str, site: CallSite) -> tuple[list[str], list[str], str | None] | None:
    """(the parameters a case must name, every parameter a case may name, the client parameter: the one the
    call's receiver names); None when the function cannot be called that way (a method, *args, not found)."""
    fn = _function(ast.parse(source), site) if site.function else None
    if (
        fn is None
        or "." in (site.function or "")
        or fn.args.vararg
        or fn.args.kwonlyargs
        or fn.args.posonlyargs
    ):
        return None
    calls = _calls(fn)
    names = [a.arg for a in fn.args.args]
    client = receiver if len(calls) == 1 and (receiver := _root(calls[0].func)) in names else None
    required = [a.arg for a in fn.args.args[: len(names) - len(fn.args.defaults)] if a.arg != client]
    return required, [n for n in names if n != client], client


PROBE = r"""
import asyncio, importlib, inspect, json, os, sys, types
sys.path[:0] = [".", "src"]
os.environ.setdefault("NEBIUS_API_KEY", "offramp-probe")  # a key check in the function lets the call through
JOBS, MARK = json.loads(%(jobs)r), %(mark)r


class _Sent(Exception):
    pass


def _recorder(box):
    def create(**kwargs):
        box.append(kwargs)
        raise _Sent()
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


out = {}
for job in JOBS:
    box = []
    try:
        try:
            client = importlib.import_module(job["client_module"])
        except ImportError:  # e.g. no openai package here: a stand-in, as a test's fake module would be
            client = sys.modules[job["client_module"]] = types.ModuleType(job["client_module"])
        mod = importlib.import_module(job["module"])
        for holder in (mod, client):  # the site's module may have imported a factory by name
            for name in ("token_factory_client", "token_factory_async_client"):
                if holder is client or hasattr(holder, name):
                    setattr(holder, name, lambda box=box, **options: _recorder(box))
        fn = getattr(mod, job["function"])
        args = {}
        for name, param in inspect.signature(fn).parameters.items():
            if name == job["client_param"]:
                args[name] = _recorder(box)
            elif name in job["variables"]:
                args[name] = job["variables"][name]
            elif param.default is inspect.Parameter.empty:
                raise LookupError("no value for parameter " + name)
        try:
            result = fn(**args)
            if inspect.iscoroutine(result):
                asyncio.run(result)
        except _Sent:
            pass
        out[job["site"]] = {"request": box[0]} if box else {"error": "the function sent no request"}
    except Exception as e:
        out[job["site"]] = {"error": type(e).__name__ + ": " + str(e)[:300]}
print(MARK + json.dumps(out, default=repr))
"""


def run_probe(box: Sandbox, jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Call each site's migrated function once, in the sandbox, with a recorder for its client: site ->
    {"request": the keyword arguments it sent} or {"error": why}. One container for all the jobs."""
    if not jobs:
        return {}
    mark = f"OFFRAMP-PROBE-{uuid.uuid4().hex}:"
    code, output = box.probe(PROBE % {"jobs": json.dumps(jobs), "mark": mark})
    line = next((ln for ln in output.splitlines() if ln.startswith(mark)), None)
    if line is None:
        why = f"the probe did not run (exit {code}): {output.strip()[-300:]}"
        return {job["site"]: {"error": why} for job in jobs}
    found = json.loads(line[len(mark) :])
    return {job["site"]: found.get(job["site"], {"error": "no result"}) for job in jobs}


TESTS_RUN = (  # a summary line per test: PASSED / FAILED / SKIPPED [n] path:line: reason / XFAIL / ERROR
    "import sys, pytest\nsys.exit(pytest.main(['-rfEpsxX', '--tb=line', '-p', 'no:cacheprovider', "
    "'--continue-on-collection-errors', *%r]))"
)


def names_function(source: str, function: str) -> bool:
    """Whether `source` names `function` as a Python name: not in a string, a comment or a longer name."""
    try:
        return any(
            t.type == tokenize.NAME and t.string == function
            for t in tokenize.generate_tokens(io.StringIO(source).readline)
        )
    except (tokenize.TokenError, SyntaxError):  # a file that does not tokenize does not run either
        return False


def _test_holders(path: str, text: str) -> list[str]:
    """Per line of `text`: the pytest node id of the test function holding it (decorators included), else
    `path` (module level, a fake, a fixture, or a file that does not parse)."""
    at = [path] * len(text.splitlines())
    try:
        body = ast.parse(text).body
    except SyntaxError:
        return at

    def walk(nodes: list[ast.stmt], prefix: str) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                walk(node.body, f"{prefix}::{node.name}")
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test"):
                start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                at[start - 1 : node.end_lineno] = [f"{prefix}::{node.name}"] * (node.end_lineno - start + 1)

    walk(body, path)
    return at


def edited_tests(path: str, before: str | None, after: str) -> list[str]:
    """The tests of a test file that a change edited: the node ids of the test functions holding a non-blank
    line it removed (in the old text) or added (in the new). A changed line outside every test function, a
    fake or fixture at module level, names the file itself: any of its tests may use it (review M-3)."""
    old, new = (before or "").splitlines(), after.splitlines()
    held = _test_holders(path, before or ""), _test_holders(path, after)
    out: set[str] = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag != "equal":
            out |= {held[0][k] for k in range(i1, i2) if old[k].strip()}
            out |= {held[1][k] for k in range(j1, j2) if new[k].strip()}
    return sorted(out)


def _collection_error(output: str, path: str) -> str:
    """The last `E ` line of pytest's "ERROR collecting <path>" section, the error itself after any source
    lines a SyntaxError shows (one line, capped), or a fallback."""
    section = re.search(
        rf"ERROR collecting {re.escape(path)} _*\n(.*?)(?:\n_{{3,}} |\n={{3,}} )", output, re.S
    )
    found = [
        ln[1:].strip() for ln in (section.group(1).splitlines() if section else []) if ln.startswith("E ")
    ]
    return (found[-1] if found else "an error during collection")[:200]


def run_site_tests(
    box: Sandbox, site_tests: dict[str, list[str]]
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """The tests of each site's test files (those that name its function) run once more on the tree that
    ships, in one container: (site -> the tests that ran, passed or failed, as pytest node ids; site -> why
    none did, in plain words). A green check says nothing of a site whose tests were all skipped, and the
    skip-count backstop passes a module skipped before and after the patch alike (review I-2)."""
    files = sorted({f for fs in site_tests.values() for f in fs})
    code, output = box.probe(TESTS_RUN % (files,)) if files else (0, "")
    ran = re.findall(r"^(?:PASSED|FAILED) (\S+)", output, re.M)  # an xfail, skip or deselection is no run
    skipped = dict.fromkeys(re.findall(r"^SKIPPED \[\d+\] (.+?):\d+: (.*)$", output, re.M))
    xfailed = Counter(n.split("::")[0] for n in re.findall(r"^(?:XFAIL|XPASS) (\S+)", output, re.M))
    # a module that did not collect: `ERROR <path>`, or `ERROR <path> - <error>` when it raised at import
    broken = set(re.findall(r"^ERROR ([^\s:]+)(?: - .*)?$", output, re.M))
    setup = re.findall(r"^ERROR ([^\s:]+)::(\S+)(?: - (.*))?$", output, re.M)  # a test that errored at setup
    counts = r"\b(\d+ (passed|failed|skipped|xfailed|xpassed|errors?|deselected)|no tests ran)\b.* in [\d.]+s"
    started = re.search(counts, output)  # pytest's last line: absent at -qq, or when the run never got there
    out: dict[str, list[str]] = {}
    why: dict[str, str] = {}
    for sid, fs in site_tests.items():
        out[sid] = [n for n in ran if n.split("::")[0] in fs]
        if out[sid]:
            continue
        reasons = [f"`{f}` could not be collected: {_collection_error(output, f)}" for f in fs if f in broken]
        reasons += [f"`{path}` was skipped: {reason[:200]}" for path, reason in skipped if path in fs]
        for f, n in sorted(xfailed.items()):
            if f in fs:
                tests = "1 test marked xfail, which does" if n == 1 else f"{n} tests marked xfail, which do"
                reasons.append(f"`{f}`: {tests} not count as a run")
        for f in fs:
            if errors := [(test, error) for path, test, error in setup if path == f]:
                (test, error), n = errors[0], len(errors)
                if n == 1:
                    head = f"`{f}::{test}` errored at setup"
                else:
                    head = f"`{f}`: {n} tests errored at setup, the first `{test}`"
                reasons.append(head + (f": {error[:200]}" if error else ""))
        listed = ", ".join(f"`{f}`" for f in fs)
        if not fs:
            why[sid] = NO_TEST_FILE
        elif code == 124:  # the sandbox returns no output: whether any test ran is unknown
            why[sid] = f"{TIMED_OUT} {getattr(box, 'timeout_s', TIMEOUT_S)} s"
        elif reasons:
            why[sid] = "; ".join(reasons)
        elif not started and not (ran or skipped or xfailed or broken):
            why[sid] = f"the run of its tests did not start (exit {code}): {output.strip()[-200:]}"
        elif re.search(r"\b\d+ deselected\b", output):  # nothing of the site's own explains it: say "some"
            why[sid] = f"none of the tests in {listed} ran (this repo's pytest options deselect some tests)"
        else:
            why[sid] = f"none of the tests in {listed} ran (pytest exit {code})"
    return out, why


def _request_difference(got: dict[str, Any], expected: dict[str, Any]) -> str | None:
    """None when the captured request equals the evaluated one (key order aside), else what differs first."""
    got, expected = (json.loads(json.dumps(x, default=repr)) for x in (got, expected))
    for key in [*expected, *(k for k in got if k not in expected)]:
        if got.get(key, ...) != expected.get(key, ...):
            return (f"it sends `{key}` = {json.dumps(got.get(key))[:160]}, the evaluated request "
                    f"{json.dumps(expected.get(key))[:160]}")  # fmt: skip
    return None


def _imported_files(source: str, path: str, tracked: set[str]) -> set[str]:
    """The repository files a module imports (absolute or relative imports that resolve to tracked files)."""
    out = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names = [a.name.replace(".", "/") for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            base = Path(path).parent
            for _ in range(max(n.level - 1, 0)):
                base = base.parent
            head = (
                (base / (n.module or "").replace(".", "/")).as_posix()
                if n.level
                else (n.module or "").replace(".", "/")
            )
            names = [head, *(f"{head}/{a.name}" for a in n.names)]
        else:
            continue
        for name in (name.removeprefix("./") for name in names):
            shapes = (f"{name}.py", f"{name}/__init__.py", f"src/{name}.py", f"src/{name}/__init__.py")
            out |= {f for f in shapes if f in tracked}
    return out


def _literal(node: ast.expr) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return ...


def _module_name(path: str) -> str:
    return path.removeprefix("src/").removesuffix(".py").replace("/", ".").removesuffix(".__init__")


def _default(fn: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> ast.expr | None:
    """The default value of the function's parameter `name`; None when it has none."""
    a = fn.args
    positional = [p.arg for p in [*a.posonlyargs, *a.args]]
    defaults = dict(zip(positional[len(positional) - len(a.defaults) :], a.defaults, strict=True))
    kw = {p.arg: d for p, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d is not None}
    return (defaults | kw).get(name)


def _no_default(fn: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """The parameter has no default, or None: only a caller's argument is ever the client."""
    default = _default(fn, name)
    return default is None or (isinstance(default, ast.Constant) and default.value is None)


def _handed_clients(
    scope: ast.AST, site: CallSite, receiver: ast.expr, callers: dict[str, str]
) -> list[str] | None:
    """None unless the call's client is a parameter the function never rebinds. Then, each place in
    `callers` (path -> source as it ships: the repo's non-test code) that does not hand the function a
    Token Factory client, as `path:line` and what it does. Every reference to the function is found through
    each module's imports, aliases and re-exports included (`_function_references`), and must be a call
    handing `token_factory_client()` / `token_factory_async_client(...)` written in it (`rewrite_callers`
    makes one of an inline `OpenAI()` / `Anthropic()`). Not known: a variable, an attribute, a helper's
    result, a parameter passed on, `*args` / `**kwargs`, an omitted argument; the function used as a value
    (passed, stored, wrapped); a lookup by a literal name (`getattr(m, "summarize")`, `importlib`); a module
    `__getattr__` where the function lives. A lookup by a computed name cannot be seen (the PR says so)."""
    if not isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef) or not isinstance(receiver, ast.Name):
        return None
    a, name = scope.args, receiver.id
    if name not in {p.arg for p in [*a.posonlyargs, *a.args, *a.kwonlyargs]} or any(
        isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load) for n in ast.walk(scope)
    ):
        return None
    positional = [p.arg for p in [*a.posonlyargs, *a.args]]
    method = "." in (site.function or "") and "staticmethod" not in map(ast.unparse, scope.decorator_list)
    function = site.function or scope.name
    targets = set() if "." in function else _targets(callers, site.file, function)
    literal = {scope.name, *_dotted(site.file)}  # names a lookup by string would use

    def handed(call: ast.Call) -> ast.expr | str:  # the argument bound to `name`, or how it is not known
        if name in (kw := {k.arg: k.value for k in call.keywords if k.arg}):
            return kw[name]
        shift = method and isinstance(call.func, ast.Attribute)  # a bound method's `self` is not in the call
        at = positional.index(name) - shift if name in positional else -1
        if at >= 0 and any(isinstance(x, ast.Starred) for x in call.args[: at + 1]):
            return "hands it on through `*`"
        if 0 <= at < len(call.args):
            return call.args[at]
        return "hands it on through `**`" if any(k.arg is None for k in call.keywords) else f"omits `{name}`"

    found: list[tuple[str, int, str]] = []  # (path, line, what), sorted below
    for path, (tree, refs) in _function_references(callers, site.file, function).items():
        if tree is None:
            found.append((path, 0, "does not parse"))
            continue
        calls = {id(c.func): c for c in ast.walk(tree) if isinstance(c, ast.Call)}
        for ref in refs:
            if (call := calls.get(id(ref))) is None:  # passed, stored or wrapped: other code calls it
                found.append((path, ref.lineno, f"uses `{ast.unparse(ref)}` without calling it"))
                continue
            got = handed(call)
            if isinstance(got, ast.Call) and isinstance(got.func, ast.Name) and got.func.id in FACTORIES:
                continue
            what = got if isinstance(got, str) else f"hands `{ast.unparse(got)}`"
            found.append((path, call.lineno, f"`{ast.unparse(ref)}(...)` {what}"))
        for call in calls.values():  # getattr(m, "summarize"), importlib.import_module("app.summarize")
            lookup = ast.unparse(call.func).rpartition(".")[2] in ("getattr", "import_module", "__import__")
            names = [x.value for x in call.args if isinstance(x, ast.Constant) and x.value in literal]
            if lookup and names:
                found.append((path, call.lineno, f"looks up `{names[0]}` by name"))
        exports = path == site.file or any(t.rpartition(".")[0] in _dotted(path) for t in targets)
        for node in tree.body if exports else ():
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                what = f"defines a module `__getattr__`, which may hand out `{scope.name}` under any name"
                found.append((path, node.lineno, what))
    return [f"`{path}{f':{line}' if line else ''}` {what}" for path, line, what in sorted(found)]


def _client_parameter(source: str, site: CallSite) -> bool:
    """Whether the site's migrated call is made on a parameter of its function: its callers hand it the
    client (`_handed_clients`)."""
    tree = ast.parse(source)
    fn = _function(tree, site) if site.function else None
    calls = _calls(fn) if fn else []
    if len(calls) != 1:
        return False
    receiver = calls[0].func.value.value.value  # <receiver>.chat.completions.create
    return _handed_clients(fn, site, receiver, {}) is not None


def _token_factory_client(receiver: ast.expr, scope: ast.AST) -> bool:
    """Whether a call's client is the Token Factory client: a `token_factory_client()` /
    `token_factory_async_client(...)` call, a name bound once to one, or a parameter of the function with no
    default but None that the code never rebinds (its callers hand it one: `_handed_clients`)."""

    def factory(n: ast.AST | None) -> bool:
        return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FACTORIES

    if factory(receiver):
        return True
    if not isinstance(receiver, ast.Name):
        return False
    bound = [n for n in ast.walk(scope) if isinstance(n, ast.Name) and n.id == receiver.id
             and not isinstance(n.ctx, ast.Load)]  # fmt: skip
    values = [n.value for n in ast.walk(scope) if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == receiver.id]  # fmt: skip
    if bound:
        return len(bound) == 1 and len(values) == 1 and factory(values[0])
    args = scope.args if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef) else None
    return (
        args is not None
        and receiver.id in {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
        and _no_default(scope, receiver.id)
    )


def conformance(
    source: str,
    site: CallSite,
    params: dict[str, Any],
    constants: dict[str, str],
    system_name: str | None,
    carried: dict[str, str] | None = None,  # the source call's transport options -> their AST, kept as is
    callers: dict[str, str] | None = None,  # the repo's non-test code calling the function, as it ships
) -> list[str]:
    """Problems with the patched site file: the call must be made on the Token Factory client (one a caller
    hands the function too, checked in `callers`) and pass exactly the evaluated parameters as written-out
    keyword arguments (a schema may be a module constant holding it) plus messages, and use the prompt
    constants."""
    tree = ast.parse(source)
    scope = (_function(tree, site) if site.function else None) or tree
    where = f"`{site.function}`" if scope is not tree else site.file
    calls = [n for n in ast.walk(scope) if isinstance(n, ast.Call)]
    tf = [c for c in calls if _call_name(c.func).endswith("chat.completions.create")]
    if any(_call_name(c.func).endswith("messages.create") for c in calls):
        return [f"the old messages.create call is still in {where}"]
    if len(tf) != 1:
        return [f"expected one chat.completions.create call in {where}, found {len(tf)}"]
    sdk = next(
        (c for n in calls if (c := _client_class(n, tree, scope)) and c.split(".")[0] in PROVIDER_KEYS), None
    )
    if sdk:
        return [f"an SDK client (`{sdk}`) is still built in {where}"]
    call, problems = tf[0], []
    receiver = call.func.value.value.value  # <receiver>.chat.completions.create
    if not _token_factory_client(receiver, scope):
        problems.append(f"the call's client `{ast.unparse(receiver)}` is not the Token Factory client")
    elif callers is not None and (unknown := _handed_clients(scope, site, receiver, callers)):
        problems.append(f"the client `{ast.unparse(receiver)}` its callers hand `{site.function}` is not "
                        f"known to be the Token Factory client: {'; '.join(unknown)}")  # fmt: skip
    if call.args or any(k.arg is None for k in call.keywords):
        problems.append("the call takes written-out keyword arguments only: no positional arguments, no **")
    named = {k.arg: k.value for k in call.keywords if k.arg}
    carried = carried or {}
    if set(named) - set(carried) != {*params, "messages"}:
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
    for key, was in carried.items():
        if key not in named or ast.dump(named[key]) != was:
            problems.append(f"`{key}=` must stay exactly as the code passed it")
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
    """Files shown for one site: its module and the callers of its function, then the tests touching them and
    their helpers. Only test files are editable: the code that ships is read-only context (see `_attempt`)."""
    files = sorted(
        p.relative_to(repo)
        for p in repo.rglob("*.py")
        if not SKIP_PARTS & set(p.relative_to(repo).parts[:-1])
        and p.relative_to(repo).as_posix() != client_path
        and not p.is_symlink()  # never edited through a link
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
rewritten the call, how its reply is read, and the callers (see "The call" and "Already changed by \
Offramp"); that code is final. You update the tests and their fakes so the repository's ruff and tests \
stay green.

Reply with edit blocks only, no analysis or explanation, then one NOTES line. Begin with `FILE:`.

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
1. Only test files are editable: tests, their fakes and test helpers, marked (editable). The migrated \
module, its callers and every other file marked (read-only) are final as Offramp's code wrote them: never \
edit them. An edit to one is rejected.
2. A test that replaced the old SDK client's constructor with a fake replaces `token_factory_client` the \
same way (Offramp did this where it could).
3. Tests: update what pins the old provider for THIS call site so it asserts the new request: the model ID, \
SDK keyword arguments (`system=`, `max_completion_tokens`, a `tool_choice` or `tools` value, parameter \
rules of the old model), the message layout, and checks of the call's source (tables of call sites, \
`ast` checks of keyword arguments). Every other call site keeps its old pins. The system prompt is now the \
first message: a test that read the user message as `messages[0]` reads it as `messages[-1]`, asserting \
the same things. Never weaken a behavioural or parsing assertion: expected return values and parsing \
checks stay exactly as they are. Never delete or rename a test.
4. Fakes: when a provider fake still cannot answer `chat.completions.create` the way the new code reads \
it, extend the repo's existing fake helper, building real openai response objects the way it already \
does, instead of writing ad-hoc fake classes inside tests.
5. Add the imports your test code needs and remove the ones it leaves unused (import order is fixed \
automatically afterwards). Keep lines within the repo's line length.
6. When "What fails now" lists failing checks, your edits must make every one of them pass."""


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
    files: dict[str, str],
    read_only: set[str],
    client: str,
    client_at: tuple[str, str],
    width: int,
    calls: tuple[str | None, str | None],
    done: list[str],
    failing: str | None = None,
) -> str:
    before, now = calls
    call = [
        f"`{site.file}` is migrated by Offramp and read-only: `{site.function}` now sends the evaluated "
        "request with written-out keyword arguments and reads the reply the OpenAI way.",
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
    factories = ", ".join(f for f in FACTORIES if f"def {f}(" in client)
    lines = [
        f"## Call site {site.id}",
        f"`{site.file}` lines {site.lines[0]}-{site.lines[-1]}, function `{site.function}`: {site.sdk} "
        f"`{site.call}` on `{site.source_model}` ({task}). Target: {TIER_NAMES[tier]} on Token "
        "Factory.",
        "",
        "## The call",
        *call,
        "",
        f"Token Factory client: `from {client_at[1]} import {factories}` (module shown below).",
        f"Repo ruff line length: {width}.",
        "",
        "## Files",
        f"### {client_at[0]} (read-only)",
        "```python",
        client.rstrip("\n"),
        "```",
    ]
    for path, text in files.items():
        lines.append(_file_block(path, "read-only" if path in read_only else "editable", text))
    return "\n".join(lines) + "\n"


TOGETHER = (
    """Offramp migrated several LLM call sites of a Python repository to NVIDIA Nemotron on Nebius Token \
Factory, which is OpenAI-compatible (`client.chat.completions.create`). Each site's patch passes the \
repository's ruff and tests on its own; merged together, some tests fail. You make them pass together by \
editing the failing tests and the fakes only.

"""
    + SYSTEM[SYSTEM.index("Reply with edit blocks only") : SYSTEM.index("Rules:")]
    + """Rules:
1. Files marked (read-only) are final: never edit them. In the editable test files, change only the tests \
listed as failing, plus helpers and fakes; every other test stays exactly as it is.
2. Keep what each site's tests assert about its own request and behaviour. Never weaken a behavioural or \
parsing assertion; never delete or rename a test.
3. Offramp gave each site's tests a fake for `token_factory_client`, next to the fake of that site's old \
SDK client. Where one test now sets `token_factory_client` more than once, only the last one counts, so the \
cases of the other SDK get the wrong fake: set it once, to the fake each case needs (for example chosen \
from the test's own parameters), built from the repo's existing fakes.
4. Add the imports your code needs. Keep lines within the repo's line length."""
)
TOGETHER_REPAIRS = 2  # Ultra calls at most for the merged tree, each shown the last failure


def _failing_tests(output: str) -> set[tuple[str, str]]:
    """(file, test's qualified name, e.g. "TestX.test_y") of each FAILED / ERROR line, parameters dropped."""
    found = re.findall(r"^(?:FAILED|ERROR) ([^\s:]+)::(?:(\w+)::)?(\w+)", output, re.M)
    return {(f, f"{cls}.{name}" if cls else name) for f, cls, name in found}


def _together_prompt(
    sites: list[CallSite],
    reach: dict[str, set[str]],
    failing: str,
    files: dict[str, str],
    editable: set[str],
    width: int,
) -> str:
    lines = ["## The sites, each green alone"]
    for s in sites:
        names = ", ".join(f"`{h}`" for h in sorted(reach.get(s.id, set()) - {s.function}))
        via = f"; tests reach it through {names}" if names else ""
        lines.append(f"- {s.id} `{s.file}` `{s.function}`: was {s.sdk} on `{s.source_model}`, so its tests "
                     f"fake the client with the repo's {s.sdk} fake{via}")  # fmt: skip
    if failing.startswith("repo checks failed"):
        failing = "this repo's checks fail:\n- " + failing.partition("): ")[2].replace(" | ", "\n- ")
    lines += ["", "## What fails together", f"With all of them merged, {failing}", "", f"Repo ruff line "
              f"length: {width}.", "", "## Files"]  # fmt: skip
    for path in sorted(files, key=lambda p: p not in editable):
        lines.append(_file_block(path, "editable" if path in editable else "read-only", files[path]))
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


ASYNC_CLIENT = '''

def token_factory_async_client(**options) -> AsyncOpenAI:
    """Async Token Factory client for the call sites that were async (they
    still await the call): reads NEBIUS_API_KEY (and NEBIUS_BASE_URL).
    `options` are the replaced client's own, such as timeout=.

    A test fakes it the way it faked the old SDK client: patch
    `token_factory_async_client` where a module imports it by name, or, for a
    call site that imports it inside the function, put a module holding a
    fake one in sys.modules["$module"] (patch.dict).
    """
    return AsyncOpenAI(
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.environ.get("NEBIUS_BASE_URL", NEBIUS_BASE_URL),
        **options,
    )
'''


# OPENAI_REQUIREMENT (sandbox.py): the floor Offramp's own pyproject declares; the client module and the
# evaluated calls ran on it (the API they use, OpenAI / AsyncOpenAI with base_url and extra_body, is older).
COMPILED = re.compile(
    r"autogenerated by (pip-compile|uv)|uv pip compile|pip-compile", re.I
)  # lockfile header
LOCKS = {"uv.lock": "uv lock", "poetry.lock": "poetry lock"}  # lockfile -> the command that regenerates it


def _requirement_name(spec: str) -> str | None:
    """The normalised project name of a requirement (`Anthropic[vertex]>=1` -> `anthropic`); None for a
    comment, an option line (`-r`, `--index-url`) or a blank."""
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
    return re.sub(r"[-_.]+", "-", m[1]).lower() if m and not spec.lstrip().startswith(("#", "-")) else None


def _requirement_places(data: dict) -> dict[tuple[str, ...], list | dict]:
    """pyproject's requirement declarations: [project] dependencies, optional dependencies and dependency
    groups (lists of requirement strings), and Poetry's dependency tables (name -> version spec)."""
    project, poetry = data.get("project", {}), data.get("tool", {}).get("poetry", {})
    out = {("project", "dependencies"): project.get("dependencies")}
    out |= {
        ("project", "optional-dependencies", k): v
        for k, v in project.get("optional-dependencies", {}).items()
    }
    out |= {("dependency-groups", k): v for k, v in data.get("dependency-groups", {}).items()}
    out |= {("tool", "poetry", "dependencies"): poetry.get("dependencies")}
    out |= {
        ("tool", "poetry", "group", g, "dependencies"): v.get("dependencies")
        for g, v in poetry.get("group", {}).items()
        if isinstance(v, dict)
    }
    return {k: v for k, v in out.items() if isinstance(v, list | dict)}


def _names(place: list | dict) -> list[str | None]:
    return [_requirement_name(x) if isinstance(x, str) else None for x in place]  # a table's keys are names


def _pyproject_with_openai(text: str, sdks: set[str]) -> tuple[str, list[tuple[str, ...]]]:
    """pyproject text with openai added to each requirement declaration naming a replaced SDK and not openai,
    right after that SDK's entry and in the file's own layout: in a list, an entry per line or inline; in a
    Poetry table, a `openai = ">=..."` line. The declarations it changed come back too. One whose text this
    cannot change exactly (checked by parsing the result) is left as it is."""
    data = tomllib.loads(text)
    for key, place in _requirement_places(data).items():
        names = _names(place)
        at = next((i for i, n in enumerate(names) if n in sdks), None)
        if at is None or "openai" in names:
            continue
        want = tomllib.loads(text)
        target = want
        for part in key[:-1]:
            target = target[part]
        if isinstance(place, dict):  # Poetry: `anthropic = "^0.40"` -> a line after it
            sdk = list(place)[at]
            target[key[-1]] = {**place, "openai": OPENAI_REQUIREMENT.removeprefix("openai")}
            found = re.finditer(rf"^[ \t]*([\"']?){re.escape(sdk)}\1[ \t]*=.*$", text, re.M)
            line = 'openai = "' + OPENAI_REQUIREMENT.removeprefix("openai") + '"'
            candidates = [  # a CRLF file keeps its line endings: `.*` took the line's \r
                text[: m.end()] + "\n" + line + ("\r" if m[0].endswith("\r") else "") + text[m.end() :]
                for m in found
            ]
        else:
            items = place
            target[key[-1]] = [*items[: at + 1], OPENAI_REQUIREMENT, *items[at + 1 :]]
            candidates = []
            for m in re.finditer(rf"([\"']){re.escape(items[at])}\1", text):
                line_end = text.find("\n", m.end())
                rest = text[m.end() : len(text) if line_end < 0 else line_end]
                quote, line_start = m[1], text.rfind("\n", 0, m.start()) + 1
                indent = re.match(r"[ \t]*", text[line_start:]).group(0)
                if text[line_start : m.start()].strip() == "" and re.fullmatch(r"\s*,?\s*(#.*)?", rest):
                    comma = "" if rest.lstrip().startswith(",") else ","  # the last entry gets one
                    cr = "\r" if rest.endswith("\r") else ""  # a CRLF file keeps its line endings
                    entry = f"\n{indent}{quote}{OPENAI_REQUIREMENT}{quote},{cr}"
                    candidates.append(text[: m.end()] + comma + rest + entry + text[m.end() + len(rest) :])
                else:  # an inline list
                    candidates.append(
                        text[: m.end()] + f", {quote}{OPENAI_REQUIREMENT}{quote}" + text[m.end() :]
                    )
        for new in candidates:
            try:
                if tomllib.loads(new) == want:
                    text = new
                    break
            except tomllib.TOMLDecodeError:
                continue
    places = _requirement_places(data)
    changed = [k for k, v in _requirement_places(tomllib.loads(text)).items() if v != places.get(k)]
    return text, changed


def _logical_lines(lines: list[str]) -> list[tuple[int, int, str]]:
    """(first, last physical index, the joined text) of each logical line of a requirements file: a line
    ending in a backslash continues on the next."""
    out, start = [], 0
    for i, line in enumerate(lines):
        if not line.rstrip("\r\n").endswith("\\") or i == len(lines) - 1:
            out.append((start, i, "".join(x.rstrip("\r\n").removesuffix("\\") for x in lines[start : i + 1])))
            start = i + 1
    return out


def _parses(line: str) -> bool:
    """A requirements line (its comment stripped) that packaging reads as one requirement."""
    try:
        Requirement(re.sub(r"(^|\s)#.*$", "", line).strip())
    except InvalidRequirement:
        return False
    return True


def _compile_command(text: str, kind: str) -> str:
    """The command a compiled requirements file's header names (`pip-compile ...`, `uv pip compile ...`)."""
    lines = [ln.lstrip("#").strip() for ln in text.splitlines()[:20] if ln.startswith("#")]
    at = next((i for i, ln in enumerate(lines) if ln.endswith("following command:")), None)
    found = next((ln for ln in lines[at + 1 :] if ln), None) if at is not None else None
    return found or ("uv pip compile" if kind == "uv" else "pip-compile")


def _requirements_with_openai(
    repo: Path, path: Path, text: str, sdks: set[str]
) -> tuple[str | None, list[str]]:
    """A requirements file with OPENAI_REQUIREMENT as a new logical line after the replaced SDK's (its
    continuation lines, comment and markers kept whole, its line endings too), checked by re-reading the
    logical lines and parsing the SDK's; (None, a note) when it must not or cannot be edited exactly. A
    compiled file (a pip-compile or uv header, or a sibling .in) and a hashed one are lockfiles: never
    edited, the note says which line to add where and how to recompile."""
    rel, named = path.relative_to(repo).as_posix(), ", ".join(f"`{s}`" for s in sorted(sdks))
    lines = text.splitlines(keepends=True)
    logical = _logical_lines(lines)
    at = next(k for k, (*_, t) in enumerate(logical) if _requirement_name(t) in sdks)
    source = path.with_suffix(".in")
    header = COMPILED.search("".join(lines[:20]))
    if header or source.is_file():
        kind = "uv" if header and "uv" in header[0].lower() else "pip-compile"
        where = (
            f"`{source.relative_to(repo).as_posix()}`" if source.is_file() else "the file it is compiled from"
        )
        return None, [f"`{rel}` is compiled ({kind}), so Offramp did not edit it: add the line "
                      f"`{OPENAI_REQUIREMENT}` to {where} beside {named}, then recompile with "
                      f"`{_compile_command(text, kind)}`"]  # fmt: skip
    if "--hash" in text:
        return None, [f"`{rel}` pins hashes, so Offramp did not edit it: add `{OPENAI_REQUIREMENT}` with its "
                      "`--hash` values (pip requires a hash on every line of such a file)"]  # fmt: skip
    _, last, sdk_line = logical[at]
    nl = "\r\n" if lines[last].endswith("\r\n") else "\n"
    new_lines = [*lines[:last], lines[last] if lines[last].endswith("\n") else lines[last] + nl,
                 OPENAI_REQUIREMENT + nl, *lines[last + 1 :]]  # fmt: skip
    old = [t for *_, t in logical]
    if [t for *_, t in _logical_lines(new_lines)] == [
        *old[: at + 1],
        OPENAI_REQUIREMENT,
        *old[at + 1 :],
    ] and (_parses(sdk_line)):
        return "".join(new_lines), []
    return None, [f"`{rel}` declares {named} in a line Offramp cannot extend exactly: add "
                  f"`{OPENAI_REQUIREMENT}` to it by hand"]  # fmt: skip


def declare_openai(repo: Path, sdks: set[str]) -> tuple[dict[str, str], list[str], bool]:
    """The migrated code imports the openai SDK (through the client module). Wherever the repo declares a
    replaced SDK and not openai, the same declaration gains OPENAI_REQUIREMENT: a requirements*.txt file
    (`_requirements_with_openai`) or a pyproject declaration (`_pyproject_with_openai`). A file declaring
    openai anywhere already is left as it is (its own pin may conflict with the floor). Lockfiles (uv.lock,
    poetry.lock, compiled or hashed requirements) are never edited: a note says what to add and how to
    regenerate them. Line endings are kept. Returns (path -> new text, notes for the PR, whether the file the
    sandbox image installs declares a replaced SDK: pyproject.toml, else the root requirements.txt)."""
    sdks = sdks - {"openai"}  # a replaced openai SDK is declared already
    edits, notes, declared, direct = {}, [], False, False
    installed = repo / ("pyproject.toml" if (repo / "pyproject.toml").is_file() else "requirements.txt")
    named = ", ".join(f"`{s}`" for s in sorted(sdks))
    reqs = sorted(f for f in repo.rglob("requirements*.txt") if f.is_file() and not f.is_symlink()
                  and not set(NOT_COPIED) & set(f.relative_to(repo).parts[:-1]))  # fmt: skip
    for f in reqs:
        try:
            text = f.read_bytes().decode()  # not read_text(): a CRLF file stays CRLF
        except (OSError, UnicodeDecodeError):
            continue
        names = [_requirement_name(t) for *_, t in _logical_lines(text.splitlines(keepends=True))]
        if not sdks & set(names):
            continue
        declared, direct = True, direct or f == installed
        if "openai" in names:
            continue
        new, why = _requirements_with_openai(repo, f, text, sdks)
        if new is not None:
            edits[f.relative_to(repo).as_posix()] = new
        notes += why
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file() and not pyproject.is_symlink():
        try:
            original = pyproject.read_bytes().decode()
            data = tomllib.loads(original)
            text, changed = _pyproject_with_openai(original, sdks)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
            data, text, changed = {}, "", []
        names = {k: set(_names(v)) for k, v in _requirement_places(data).items()}
        declared |= any(sdks & n for n in names.values())
        direct |= any(sdks & n for n in names.values())
        elsewhere = [".".join(k) for k, n in names.items() if "openai" in n]
        if changed and not elsewhere:
            edits["pyproject.toml"] = text
            locks = [(lock, cmd) for lock, cmd in LOCKS.items() if (repo / lock).is_file()]
            notes += [f"`{lock}` is not regenerated by Offramp: run `{cmd}` before merging, so the lockfile "
                      f"includes `{OPENAI_REQUIREMENT}`" for lock, cmd in locks]  # fmt: skip
        missed = [k for k, n in names.items() if sdks & n and "openai" not in n]
        if elsewhere:
            notes += [f"`pyproject.toml` `{'.'.join(k)}` declares {named} but not openai, which "
                      f"`{elsewhere[0]}` declares: Offramp left the file as it is (that openai pin decides); "
                      "check the migrated code's environment installs it" for k in missed]  # fmt: skip
        else:
            notes += [f"`pyproject.toml` `{'.'.join(k)}` declares {named}, but Offramp could not add "
                      f"`{OPENAI_REQUIREMENT}` to it exactly: add it by hand"
                      for k in missed if k not in changed]  # fmt: skip
    added, why = [], "since the migrated code imports the OpenAI SDK through Offramp's client module"
    for p in sorted(edits):
        added.append(f"`{p}`: added `{OPENAI_REQUIREMENT}` beside {named}, {why}")
        if repo / p != installed:  # the checks never installed the edit: say what they ran on
            ran = f"ran on the environment from `{installed.name}`, which does not install that file"
            ran = (
                ran
                if installed.is_file()
                else "ran without it: the sandbox image installs no dependency file"
            )
            added.append(f"`{p}` gains `{OPENAI_REQUIREMENT}`; the checks {ran}")
    notes[:0] = added
    if sdks and not declared:
        why = "since the migrated code imports it"
        notes.append(f"No requirements*.txt or pyproject.toml here declares {named}: add "
                     f"`{OPENAI_REQUIREMENT}` where this repo declares its dependencies, {why}")  # fmt: skip
    return edits, notes, direct


def client_module(asynchronous: bool = False, module: str = ROOT_CLIENT[1]) -> str:
    """The Token Factory client module; with `asynchronous`, also `token_factory_async_client` (an async site
    was migrated), documented for its import path `module`. Without it, the module is exactly as before."""
    ids = {t.name.lower(): get_tier(t).model_id for t in Tier}
    text = string.Template(TEMPLATE.read_text()).substitute(ids)
    if not asynchronous:
        return text
    text = text.replace("from openai import OpenAI\n", "from openai import AsyncOpenAI, OpenAI\n", 1)
    return text + ASYNC_CLIENT.replace("$module", module)


FACTORIES = ("token_factory_client", "token_factory_async_client")


def _built_note(site: CallSite, code: str, original: str, client: str) -> str | None:
    """For the patcher prompt: the site's function now builds a Token Factory client where it built the old
    SDK client, and how a test fakes it the way it faked that SDK."""
    factory = next((f for f in FACTORIES if f"{f}(" in code and f not in original), None)
    if factory is None:
        return None
    if re.search(rf"^[ \t]+from {re.escape(client)} import {factory}$", code, re.M):  # inside the function
        how = (f"imported inside the function as the SDK was; a test fakes it the same way, e.g. "
               f'`patch.dict("sys.modules", {{"{client}": fake_module}})` with `fake_module.{factory}` '
               "returning the fake client")  # fmt: skip
    else:
        module = _module_name(site.file)
        how = f'a test fakes it by name, e.g. `monkeypatch.setattr({module}, "{factory}", fake)`'
    return (
        f"`{site.file}`: `{site.function}` builds its client with `{factory}(...)` from `{client}` where "
        f"it built the old SDK client, {how}; the fake client's `chat.completions.create` returns an OpenAI "
        "chat completion"
    )


CONFIG_FILES = ("*.yaml", "*.yml", "*.toml", "*.cfg", "*.ini", "*.json", "*.sh", "Dockerfile*", "Procfile",
                "docker-compose*", "*.example", "*.sample")  # fmt: skip  # where a deployment sets its keys


def _key_reads(repo: Path, keys: set[str]) -> list[str]:
    """Where the repo (tests aside) still names a replaced provider's key, as PR notes: Offramp renames it
    only inside the migrated functions. Code: each exact read, at most 10, then a count. Deployment and config
    files (CONFIG_FILES) that mention it, and the .py files that could not be read here, are named too."""
    found, unread, configs = [], [], []
    for f in sorted(repo.rglob("*")) if keys else []:
        path = f.relative_to(repo).as_posix()
        if (f.is_symlink() or not f.is_file() or set(NOT_COPIED) & set(f.relative_to(repo).parts[:-1])
                or path == "offramp_client.py" or f.name == "offramp_client.py"):  # fmt: skip
            continue
        if f.suffix != ".py":
            if any(fnmatch.fnmatch(f.name, g) for g in CONFIG_FILES):
                text = f.read_bytes()[:1_000_000].decode(errors="replace")
                configs += [(path, k) for k in sorted(keys) if re.search(rf"\b{k}\b", text)]
            continue
        if is_test_path(path):
            continue
        try:
            tree = ast.parse(f.read_text())
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
            unread.append(path)
            continue
        defs = _defs(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and n.value in keys:
                inside = [
                    (d.end_lineno - d.lineno, q) for q, d in defs if d.lineno <= n.lineno <= d.end_lineno
                ]
                where = f"in `{min(inside)[1]}`" if inside else "at module level"
                found.append((path, n.lineno, f"`{path}:{n.lineno}` ({where}) still reads `{n.value}`"))
    notes = [text for *_, text in sorted(found)]
    tail = [f"and {len(notes) - 10} more such reads"] if len(notes) > 10 else []
    why = "Offramp renamed the key to `NEBIUS_API_KEY` only inside the migrated functions"
    out = [f"{note}: {why}" for note in notes[:10]] + tail
    out += [f"`{path}` mentions `{k}`: wherever this repo deploys, set `NEBIUS_API_KEY` for the migrated code"
            for path, k in configs]  # fmt: skip
    if unread:
        version = ".".join(map(str, sys.version_info[:2]))
        named = ", ".join(f"`{u}`" for u in unread[:5])
        out.append(f"{len(unread)} .py file(s) could not be parsed here (Python {version}), so key reads in "
                   f"them are not listed: {named}")  # fmt: skip
    return out


def _sdk_left(source: str, site: CallSite) -> list[str]:
    """What the migrated function still reads from the anthropic SDK (`anthropic.APIError` in a handler)."""
    tree = ast.parse(source)
    fn = _function(tree, site)
    names = _imported_names(tree, fn) if fn else {}
    sdk = {k for k, v in names.items() if v.split(".")[0] == "anthropic"}
    attrs = {ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id in sdk} if fn else set()  # fmt: skip
    bare = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id in sdk
            and names[n.id] != "anthropic"} if fn else set()  # fmt: skip
    return sorted(attrs | bare)


def is_async_site(source: str, site: CallSite) -> bool:
    """Whether the site's function is async (its callers then hand it an async client) or builds one."""
    tree = ast.parse(source)
    fn = _function(tree, site) if site.function else None
    built = (CLIENT_CLASSES.get(_client_class(n, tree, fn or tree) or "") for n in ast.walk(fn or tree))
    return isinstance(fn, ast.AsyncFunctionDef) or "token_factory_async_client" in built


@dataclass
class _Job:
    """One MIGRATE site being patched: the files shown to the model and the code that must ship as written."""

    site: CallSite
    shown: dict[str, str]  # path -> text as shown to the patcher: the code Offramp wrote, then the tests
    before: dict[str, str]  # on disk before the site's patch: the base every site is patched against
    reference: dict[str, str]  # every file that ships as code, as Offramp's code wrote it: read-only
    client_module: str  # import path of the Token Factory client module
    width: int  # the target's ruff line length
    constants: dict[str, str] = field(default_factory=dict)  # Offramp's NEMOTRON_* name -> its import line
    locked: bool = True  # shipped code read-only to the model (switched off only to test the backstop alone)
    verify: Callable[[], str | None] | None = None  # the probe: why the patched call is not the evaluated one
    notes: list = field(default_factory=list)  # the last attempt's edits applied by name: (path, names)
    skips: int = 0  # the untouched repo's skipped + xfailed tests: a patched run may not have more
    others: dict[str, str] = field(default_factory=dict)  # every module, as this site's code leaves it
    kept: dict[str, None] = field(default_factory=dict)  # the run's "kept the ... import" warnings
    gone: dict[str, set[str]] = field(default_factory=dict)  # the last attempt's: imports its edit orphaned


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
    job: _Job,
    edits: list[Edit],
    repo: Path,
    box: Sandbox,
    formatted: bool,
    write,
    base: dict[str, str] | None = None,
    tried: list | None = None,
    hidden: frozenset[str] = frozenset(),
) -> tuple[str | None, dict | None, PatchFailure | None]:
    """Apply one set of edits to `base` (the files as shown in the first message unless given), check it
    alone, then put the disk back. (None, the site's patched files, None) when the repo is green with it and
    its code is the code Offramp wrote; when the checks ran red, (why, the changed files as they were checked,
    the failure); else (why not, {}, the failure's kind and the files it names). `tried` holds every tree
    checked so far, with its outcome: a tree in it is not checked again (the sandbox would say the same) and
    comes back as (its why, None, its failure). Blocks for `hidden` files (not shown in their current text in
    the last repair turn) are refused."""
    base = job.shown if base is None else base
    new = dict(base)
    locked = frozenset(p for p in new if p in job.reference) if job.locked else frozenset()
    job.notes, job.gone = [], {}  # this attempt's: its caller keeps those of a tree that ships or is built on
    problems = _apply(edits, new, locked, hidden, job.notes)
    for path in new if not problems else []:
        try:
            if new[path] != job.shown[path]:
                new[path] = add_imports(new[path], job.client_module)
                if is_test_path(path):  # a test may read the constants Offramp wrote, imported like any other
                    new[path] = add_fixtures(add_imports(new[path], job.client_module, job.constants))
            if new[path] != base[path] and is_test_path(path):  # imports this edit left unused (review I5)
                others = {**job.others, **new}  # the other files as this edit leaves them (re-review 2, n3)
                new[path], names = _drop_unused(
                    path, base[path], new[path], others, job.kept, "the model's edit"
                )
                job.gone |= {path: names} if names else {}
        except SyntaxError as e:
            problems.append(
                Problem("does_not_parse", f"{path} does not parse: {e.msg} (line {e.lineno})", path)
            )
    deleted = []
    for path in new if not problems else []:
        if is_test_path(path) and (gone := _test_names(job.shown[path]) - _test_names(new[path])):
            problems.append(
                Problem("deletes_tests", f"patch removes tests from {path}: {sorted(gone)}", path)
            )
            deleted += [f"{path}::{name}" for name in sorted(gone)]
        if is_test_path(path) and (skips := _new_skips(job.shown[path], new[path])):
            problems.append(Problem("skips_tests", _skips_text(path, skips), path))
            deleted += [f"{path}::{name}" for name in skips]
    if problems:
        files = list(dict.fromkeys(p.file for p in problems))
        rejected = PatchFailure(kind=problems[0].kind, files=files, tests=deleted)
        return f"patch rejected (patch rules): {'; '.join(p.text for p in problems)}", {}, rejected
    if (seen := next((t for t in tried or [] if t[0] == new), None)) is not None:
        return seen[1], None, seen[2]
    changed = [p for p in new if new[p] != job.before[p]]
    for p in changed:
        write(p, new[p])
    _settle(box, repo, changed, job.before, job.width, formatted)
    result: CheckResult = _no_more_skips(box.check(), job.skips)
    why = job.verify() if result.ok and job.verify else None  # run while the patch is on disk
    off = backstop(repo, job.reference)  # after the run: whatever wrote code, an edit or a test, is caught
    patched = {p: (repo / p).read_text() for p in changed}
    _restore(repo, job.reference, off)
    for p in changed:  # every site is checked against the same base
        (repo / p).write_text(job.before[p])
    if off:
        text = f"the patch changes code that ships (only tests may change): {'; '.join(off.values())}"
        out = text, {}, PatchFailure(kind="backstop", files=sorted(off))
    elif why:
        out = (
            f"the patched code does not send the evaluated request: {why}",
            {},
            PatchFailure(kind="probe_mismatch"),
        )
    elif result.ok:
        return None, patched, None
    else:  # the files as checked: what a repair is shown and applied to
        out = (
            f"repo checks failed on the patch ({result.mode}): {_summary(result.output)}",
            patched,
            PatchFailure(kind="checks_failed"),
        )
    if tried is not None:
        tried.append((new, out[0], out[2].model_copy(update={"model_edit": bool(edits)})))
    return out


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
    locked: bool = True,  # shipped code read-only to the model; off only to prove the backstop alone in tests
    probed: bool = True,  # the probe in the sandbox; tests switch it off likewise
) -> PatchResult:
    repo = Path(work_dir) / "repo"
    shutil.rmtree(repo, ignore_errors=True)
    listing = git_files(Path(repo_dir))  # a git checkout: only its tracked files reach the sandbox
    tracked_list, filtered = listing or (None, [])
    warnings = copy_repo(repo_dir, repo, tracked_list)
    paths = ", ".join(f"`{p}`" for p in filtered[:10]) + (
        f" and {len(filtered) - 10} more" * (len(filtered) > 10)
    )
    left_out = (  # paths only, never content
        f"Offramp left {len(filtered)} tracked file(s) out of the sandbox, which git filters (a `filter` "
        f"attribute, such as git-crypt's; a fresh clone holds them encrypted): {paths}"
    )
    width = line_length(repo)
    originals: dict[str, str | None] = {}

    root = repo.resolve()

    def inside(p: str) -> bool:  # a relative path to a regular file (or a new one) inside the copy
        f = repo / p
        return not Path(p).is_absolute() and not f.is_symlink() and f.resolve().is_relative_to(root)

    def read(p: str) -> str | None:
        if not inside(p):
            warnings.append(f"ignored a path outside the repository copy or a symlink: {p}")
            return None
        f = repo / p
        return f.read_text() if f.is_file() else None

    def write(p: str, text: str) -> None:
        if not inside(p):
            warnings.append(f"refused to write a path outside the repository copy or a symlink: {p}")
            return
        originals.setdefault(p, read(p))
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(text)

    tracked = {f.relative_to(repo).as_posix() for f in repo.rglob("*.py")  # the target's own files, as copied
               if f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(root)}  # fmt: skip

    by_id = {s.id: s for s in sites}
    adapters = {k: a.model_copy(deep=True) for k, a in adapters.items()}
    decided: dict[str, Verdict] = {}
    diffs: dict[str, str] = {}
    site_files: dict[str, list[str]] = {}
    box = sandbox_factory(repo)
    to_migrate = sorted((v for v in verdicts if v.decision == "MIGRATE"), key=lambda v: v.site_id)
    box.files = tracked_list  # the copy holds those files only; a plain directory needs the sandbox's opt-in

    def downgrade(v: Verdict, reason: str, failure: PatchFailure | None = None) -> None:
        update = {"decision": "MANUAL_REVIEW", "reason": reason, "patch_failure": failure}
        decided[v.site_id] = v.model_copy(update=update)

    def stop(reason: str, failure: PatchFailure | None = None) -> PatchResult:  # nothing patched or changed
        for v in to_migrate:
            downgrade(v, reason, failure)
        verdicts_out = [decided.get(v.site_id, v) for v in verdicts]
        return PatchResult(files={}, diffs={}, site_files={}, verdicts=verdicts_out, adapters=adapters,
                           sandbox_mode=box.mode, warnings=warnings, linted=box.ruff,
                           sandbox_notes=box.notes)  # fmt: skip

    if box.mode == "none":
        return stop(NO_SANDBOX)
    untouched = box.check()
    baseline_skips = _skip_count(untouched.output)  # a patched run may not skip or xfail more (review I4)
    if not untouched.ok:
        return stop(
            f"repo checks fail on the untouched repo ({untouched.mode}): {_summary(untouched.output)}"
            + (f"; {left_out}" if filtered else ""),
            PatchFailure(kind="repo_red"),
        )

    try:
        client_at = client_location(repo, [by_id[v.site_id].file for v in to_migrate if v.site_id in by_id])
    except Unsupported as e:
        return stop(f"no place for Offramp's client module: {e}")
    if client_at is None:
        return stop(
            "no package to hold Offramp's client module: the migrated call sites sit in the repository root, "
            "which is itself a Python package (it has __init__.py)"
        )
    asynchronous = any(  # an async site is migrated: the client module holds the async client too
        is_async_site(text, by_id[v.site_id])
        for v in to_migrate
        if v.site_id in by_id and by_id[v.site_id].file in tracked and (text := read(by_id[v.site_id].file))
    )
    client = client_module(asynchronous, client_at[1])
    replaced = read(client_at[0])  # a module of that name already there (an earlier run's, or the repo's own)
    write(client_at[0], client)
    sdks = {by_id[v.site_id].sdk for v in to_migrate if v.site_id in by_id}
    dependencies, change_notes, direct = declare_openai(repo, sdks)  # code-written like the client module
    box.sdks = sdks & {"anthropic"} if direct else set()  # the patched checks' image: see OPENAI_LAYER
    if replaced not in (None, client):
        change_notes.insert(0, f"`{client_at[0]}` existed and was replaced by this run's client module")
    for path, text in dependencies.items():
        write(path, text)
    kept = [by_id[v.site_id] for v in verdicts if v.decision == "KEEP_CLOSED" and v.site_id in by_id]
    for site in sorted(kept, key=lambda s: (s.file, -s.lines[0])):  # bottom-up: line numbers stay valid
        v = next(v for v in verdicts if v.site_id == site.id)
        before = read(site.file) if site.file in tracked else None
        if before is None:  # a link or a file outside the copy: no comment, never written through
            warnings.append(
                f"no keep comment for {site.id}: {site.file} is not a regular file of the repository"
            )
            continue
        write(site.file, _insert_comment(before, site, keep_comment(site, v, config)))
        diffs[site.id] = _diff({site.file: before}, {site.file: read(site.file)})
        site_files[site.id] = [site.file]

    base = box.check()
    if not base.ok:
        return stop(
            f"repo checks fail after adding Offramp's client module ({client_at[0]}) and kept-site comments "
            f"({base.mode}): {_summary(base.output)}",
            PatchFailure(kind="repo_red"),
        )
    formatted = box.formatted()  # the repo keeps `ruff format` clean: so do the patches

    base_code = shipped_code(repo)  # the base's code (client module and kept-site comments in): read-only
    base_code |= {
        p: read(p) for p in dependencies
    }  # the dependency edit too: the backstop keeps it as written
    qual_of: dict[str, CallSite] = {}  # MIGRATE site -> the site with its function's qualified name
    for v in to_migrate:
        site = by_id.get(v.site_id)
        source = read(site.file) if site and site.file in tracked else None
        if source is not None and (qual := site_qualname(source, site)):
            qual_of[v.site_id] = site.model_copy(update={"function": qual})
    carried_of: dict[str, dict[str, str]] = {}  # site -> the source call's transport options -> their AST
    carried_values: dict[str, dict[str, Any]] = {}  # site -> those whose value is a literal
    probes: dict[str, tuple[dict[str, Any], dict[str, Any], str] | None] = {}  # site -> (job, expected, case)
    verification: dict[str, str] = {}  # site -> how its shipped request was checked

    def site_cases(sid: str) -> list[EvalCase]:
        try:
            return sorted(load_cases(repo / "evals", sid), key=lambda c: c.id)
        except (ValueError, OSError):  # unreadable cases: nothing to drive the function with
            return []

    def probe_job(sid: str, local: CallSite, source: str, spec: AdapterSpec, tier: Tier):
        """(probe job, the request build_request evaluates for that case, case id), or (None, why not)."""
        shape = probe_parameters(source, local)
        if shape is None:
            return None, "a method, or a function taking *args or keyword-only arguments"
        need, every, client = shape
        case = next((c for c in site_cases(sid) if set(need) <= set(c.variables)), None)
        if case is None:
            return None, f"no eval case's variables name its parameters ({', '.join(need) or 'none'})"
        try:
            request = build_request(spec, case, tier)
        except (ValueError, UnresolvedCallSite) as e:
            return None, f"the case does not build a request: {e}"
        expected = {**call_params(request), "messages": request["messages"]}
        job = {"site": sid, "module": _module_name(local.file), "function": local.function,
               "client_module": client_at[1], "client_param": client,
               "variables": {k: case.variables[k] for k in every if k in case.variables}}  # fmt: skip
        return (job, expected, case.id), None

    def verify(sids: list[str]) -> dict[str, str]:
        """Run each site's migrated function in the sandbox and compare what it sends with the evaluated
        request: site -> why not (absent when it matches, or when the probe cannot run it)."""
        jobs = {sid: probes[sid] for sid in sids if probes.get(sid)}
        found = run_probe(box, [job for job, _, _ in jobs.values()])
        out = {}
        for sid, (_, expected, case_id) in jobs.items():
            if "request" not in found[sid]:
                verification[sid] = f"{UNCHANGED}; the probe could not run it: {found[sid]['error']}"
                continue
            literal = carried_values[sid]  # transport options compared by value when they are literals
            got = {k: v for k, v in found[sid]["request"].items() if k not in carried_of[sid] or k in literal}
            if (diff := _request_difference(got, {**expected, **literal})) is not None:
                out[sid] = f"for case {case_id}, {diff}"
            else:
                verification[sid] = (
                    f"probe: run in the sandbox with case {case_id}, it sends the evaluated request"
                )
        return out

    # every module at base, and the warnings (in order) for an orphaned SDK import another module still reads
    base_sources = {p: text for p in sorted(tracked) if (text := read(p)) is not None}
    kept_imports: dict[str, None] = {}

    def clean(path: str, before: str, after: str, others: dict[str, str]) -> tuple[str, set[str]]:
        return _drop_unused(path, before, after, others, kept_imports, "Offramp's rewrite")

    versions: dict[str, dict[str, str]] = {}  # site -> its patched files, green alone
    unchecked = False  # a green-alone patch was pruned after its check
    reach: dict[str, set[str]] = {}  # site -> what its tests name to reach the migrated function
    site_tests: dict[str, list[str]] = {}  # site -> its test files: those in its context naming its function
    raw: dict[str, dict[str, str]] = {}  # site -> its patched files as checked alone, before pruning
    befores: dict[str, dict[str, str]] = {}  # site -> its files before its patch
    code_of: dict[str, dict[str, str]] = {}  # site -> the code files Offramp's code changed, as written
    site_twins: dict[str, dict[str, set[str]]] = {}  # site -> test file -> the fake lines mirror_fakes added
    site_notes: dict[str, list] = {}  # site -> the by-name edits (path, names) of the tree it ships alone
    site_dropped: dict[str, dict[str, set[str]]] = {}  # site -> code file -> SDK names its rewrite dropped
    site_dead: dict[str, Counter] = {}  # site -> `module.name` -> its tests' patches of it removed alone
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
        original = read(site.file) if site.file in tracked else None
        if original is None:
            downgrade(v, f"patch not composed: {site.file} is not a regular file inside the repository")
            continue
        local = qual_of.get(site.id)  # module-local lookups use the qualified name; callers the plain one
        if local is None:
            why = (
                f"more than one function named `{site.function}` in {site.file}, and the site's lines do not"
            )
            downgrade(v, f"patch not composed: {why} tell which one makes the call")
            continue
        try:
            prepared, constants, system_name, rewritten, fixed = _prepare(
                original, local, adapter.spec, params, width, client_at[1]
            )
        except Unsupported as e:
            downgrade(v, f"patch not composed: {e}")
            continue
        if not rewritten:  # code that ships is code-only: a call the model writes is never shipped
            downgrade(v, "Offramp could not rewrite this call; a model-written call is not shipped")
            continue
        if not fixed:
            downgrade(v, "Offramp could not map how this call's reply is read; a model-written mapping is "
                         "not shipped")  # fmt: skip
            continue
        context = _context(repo, site, client_at[0])
        code = {site.file: add_imports(prepared, client_at[1])}  # what ships for it: Offramp's code only
        own = code[site.file]  # before its own callers are rewired
        modules = {p: text for p in sorted(tracked) if not is_test_path(p) and p != client_at[0]
                   and (text := read(p)) is not None} | code  # fmt: skip  # the repo's code, as it would ship
        if site.function:  # mechanical: callers hand the migrated function a Token Factory client
            code |= rewrite_callers(modules, site.file, local.function, client_at[1])
        dropped: dict[str, set[str]] = {}  # code file -> the SDK names its rewrite orphaned, dropped
        others = base_sources | code  # the tree as this site's patch leaves it
        for p, text in code.items():
            code[p], names = clean(p, read(p) or "", text, others)
            dropped |= {p: names} if names else {}
        site_dropped[site.id] = dropped
        context += [p for p in code if p not in context]  # a caller reached through an alias is shown too
        base = {p: read(p) for p in code}
        for p, text in code.items():
            (repo / p).write_text(text)
        _settle(box, repo, list(code), base, width, formatted)  # tidied as it will ship, and shown so
        code = {p: read(p) for p in code}
        for p, text in base.items():
            (repo / p).write_text(text)
        carried = source_carried(original, local)
        carried_of[site.id] = {k: ast.dump(node) for k, node in carried.items()}
        literal = {k: _literal(node) for k, node in carried.items()}
        carried_values[site.id] = {k: x for k, x in literal.items() if x is not ...}
        callers = modules | code  # every module that may call it, as it ships
        problems = conformance(
            code[site.file], local, params, constants, system_name, carried_of[site.id], callers
        )
        if problems:
            why = f"Offramp's own rewrite does not carry the evaluated request: {'; '.join(problems)}"
            downgrade(v, f"patch not composed: {why}")
            continue
        probes[site.id], unprobed = probe_job(site.id, local, original, adapter.spec, tier)
        if unprobed and probed:
            verification[site.id] = f"{UNCHANGED}; not probed: {unprobed}"
        shown = {p: code[p] if p in code else read(p) for p in context}
        changed_callers = {Path(p).stem for p in code if p != site.file}
        done = [
            f"`{p}` passes `{' / '.join(f'{f}()' for f in FACTORIES if f'{f}()' in code[p])}` to "
            f"`{site.function}`"
            for p in code
            if p != site.file
        ]
        if site.function and (note := _built_note(site, own, original, client_at[1])):
            done.append(note)
        old = "OpenAI" if site.sdk == "openai" else "Anthropic"
        handles = {site.function or ""}.union(
            *(reach_handles(code[p], site.function) for p in code if p != site.file)
        )
        reach[v.site_id] = handles
        name = (site.function or "").rsplit(".", 1)[-1]  # what the PR says was measured: files naming it
        site_tests[v.site_id] = [p for p in context if is_test_path(p) and Path(p).name != "conftest.py"
                                 and name and names_function(read(p) or "", name)]  # fmt: skip
        twins: dict[str, set[str]] = {}  # test file -> the fake lines mirror_fakes added
        for p in context:  # mechanical: the repo's provider fakes answer the Token Factory client's call
            if not is_test_path(p) or p in base_code:  # a test file shipped code imports stays as it is
                continue
            if site.sdk == "anthropic" and (converted := convert_fakes(shown[p], client_at[1])) != shown[p]:
                shown[p] = converted
                done.append(f"`{p}`: its Anthropic fake also answers `chat.completions.create`, with the "
                            "same canned blocks as an OpenAI chat completion")  # fmt: skip
            for name in (n for n in OLD_CLIENTS if n.endswith(old)) if changed_callers else ():  # sync, async
                if (mirrored := mirror_fakes(shown[p], changed_callers, name, OLD_CLIENTS[name])) != shown[p]:
                    added = set(mirrored.splitlines(keepends=True)) - set(shown[p].splitlines(keepends=True))
                    twins[p] = twins.get(p, set()) | added
                    shown[p] = mirrored
                    done.append(f"`{p}` also fakes `{OLD_CLIENTS[name]}` where it faked `{name}`")
            if dropped and (cut := drop_dead_patches(shown[p], p, dropped, tracked))[1]:
                shown[p] = cut[0]
                site_dead[site.id] = Counter(site_dead.get(site.id)) + Counter(cut[1])
                done += [f"`{p}` no longer patches `{k}`, which the rewrite removed" for k in cut[1]]
        before = {p: read(p) for p in shown}
        probe = (lambda sid=site.id: verify([sid]).get(sid)) if probed else None
        names = [*constants.values(), *([system_name] if (system_name or "").startswith("NEMOTRON_") else [])]
        imports = {name: f"from {_module_name(site.file)} import {name}" for name in names}
        job = _Job(local, shown, before, {**base_code, **code}, client_module=client_at[1], width=width,
                   constants=imports, locked=locked, verify=probe, skips=baseline_skips, others=others,
                   kept=kept_imports)  # fmt: skip
        # Offramp's own changes alone: green needs no model call; red shows the model what fails
        tried: list = []  # every tree checked for this site, with its outcome (see _attempt)
        carried: list = []  # by-name edits of the checked tries the next blocks build on
        carried_gone: dict[str, set[str]] = {}  # and the test imports those tries' edits orphaned
        failure, patched, last = _attempt(job, [], repo, box, formatted, write, tried=tried)
        if last is not None and last.kind == "probe_mismatch":  # the model may not touch it
            why = f"Offramp's own rewrite was run and {failure.partition(': ')[2]}"
            downgrade(v, f"patch not composed: {why}", last)
            continue
        checked, applied, sent = failure, [], []  # the last check's failure; checked tries' blocks; replies
        base = dict(shown)  # what the next blocks meet: the first message's text, then the last checked try's
        hidden: frozenset[str] = frozenset()  # changed files the last repair turn had no room to show
        ended: PatchFailureKind | None = None  # how the loop ended when not on a checked try's failure
        replies = 0  # the model's replies for this site
        calls = (_call_text(original, local), _call_text(code[site.file], local))
        read_only = {p for p in shown if p in job.reference} if locked else set()
        prompt = _prompt(local, tier, shown, read_only, client, client_at, width, calls, done, failure)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        editable = [p for p in shown if p not in read_only]
        ledger = [_Copy(p, 1, shown[p]) for p in editable]  # the copies of editable files in the conversation
        for attempt in range(1 + REPAIRS if failure else 0):
            if (stale := _stale(ledger, {p: base[p] for p in editable}, hidden)) is not None:
                failure, ended = STALE.format(path=stale), "internal_check"
                log.error("%s: %s; nothing was sent to the model", site.id, failure)
                break
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
                ended = "budget_stop"
                break
            except APIStatusError as e:  # a refusal (too large, a 4xx) fails this site only, never cached
                if (refused := _refusal(e)) is None:
                    raise  # transient or auth: the run fails and can be resumed
                failure = f"patch not composed: {refused}" + (f"; last failure: {failure}" if failure else "")
                ended = "provider_error"
                break
            replies += 1
            edits = parse_edits(res.output.text or "").edits
            if not edits:  # empty or malformed: ask again, no check needed
                failure = "the reply holds no complete edit block" + (
                    f"; last failure: {checked}" if checked else ""
                )
                ended = "no_edit_block"
            else:
                fresh, omitted = _resent(edits, applied, base)
                # the same reply on the same text again, or only checked blocks: nothing new to check
                if (base, hidden, edits) in sent or not (fresh or omitted):
                    failure = f"repair produced no change; last failure: {failure}"
                    ended = "no_change"
                    break
                sent.append((base, hidden, edits))
                if not fresh:  # checked blocks left out to drop them: the tree is still the one checked last
                    _, was, last = next(t for t in tried if t[0] == base)
                    failure = f"{ALREADY} applied, and leaving one out does not drop it; last failure: {was}"
                    ended = None
                    break
                why, found_patched, found = _attempt(
                    job, fresh, repo, box, formatted, write, base, tried, hidden
                )
                if found_patched is None:  # a tree checked before (A-B-A, an undo): its own failure, its kind
                    failure, last, ended = f"{RETURNED}; its failure: {why}", found, None
                    break
                failure, patched, checked, ended = why, found_patched, why, None
                last = found.model_copy(update={"model_edit": True}) if found else last
                if found and found.kind == "checks_failed":  # it applied: a repair builds on it, as checked
                    base = {p: patched.get(p, job.before[p]) for p in base}
                    tried.append((base, why, last))
                    applied += fresh
                    carried += job.notes
                    carried_gone = _union(carried_gone, job.gone)
                if failure is None:
                    site_notes[site.id] = [*carried, *job.notes]  # review M3: only what ships is reported
                    site_dropped[site.id] = _union(site_dropped.get(site.id, {}), carried_gone, job.gone)
                    break
            repair, again, hidden = _repair(failure, {p: base[p] for p in editable}, _live(ledger))
            messages = [
                *_supersede(messages, ledger, set(again)),
                {"role": "assistant", "content": res.output.text or ""},
                {"role": "user", "content": repair},
            ]
            ledger += [_Copy(p, len(messages) - 1, base[p]) for p in again]
        if failure is not None:
            outer = {"kind": ended, "last": last.kind} if ended else {}
            downgrade(v, failure, last.model_copy(update={**outer, "tries": replies}))
            continue
        alone[v.site_id] = True
        unpruned = dict(patched)  # the version the check passed: the fallback if pruning breaks it
        for p in set(patched) & set(twins):  # a twin in a test that never reaches the function is noise
            pruned = prune_twins(patched[p], twins[p], handles)
            if pruned != patched[p]:
                patched[p], unchecked = pruned, True  # checked together below, even alone
                if pruned == job.before[p]:
                    del patched[p]
        versions[v.site_id] = patched
        raw[v.site_id] = unpruned
        site_twins[v.site_id] = twins
        befores[v.site_id] = job.before
        code_of[v.site_id] = code

    # all green-alone patches together, in site order (disk is at the base again)
    base_files = {p: read(p) for files_ in versions.values() for p in files_}
    by_verdict = {v.site_id: v for v in to_migrate}

    unified: set[str] = set()  # test files where merge() gave one fake per case to several sites' twins
    dead: Counter = Counter()  # `module.name` -> the merged tests' patches of it removed (drop_dead_patches)
    plain: dict[str, str] = {}  # the last merged tree before its own edits (cleanup, unified twins, patches)
    gone_imports: dict[str, set[str]] = {}  # code file -> SDK names the last merged tree no longer imports
    together_by_name: list = []  # the combined repair's by-name edits, when its tree ships
    together_gone: dict[str, set[str]] = {}  # and the test imports its edits orphaned

    def merge(sids: list[str]) -> tuple[dict[str, str], list[str]]:  # (merged files, sites merged)
        merged: dict[str, str] = {}
        taken = []
        for sid in sids:
            trial = dict(merged)
            for path, text in versions[sid].items():
                if (m := merge_onto(base_files[path], trial.get(path, base_files[path]), text)) is None:
                    why = f"green alone, but its patch conflicts with another site's in {path}"
                    downgrade(by_verdict[sid], why, PatchFailure(kind="merge_conflict", files=[path]))
                    break
                trial[path] = m
            else:
                merged, taken = trial, [*taken, sid]
        plain.clear()  # the per-site diffs merged: what the report's site diffs show together
        plain.update(merged)
        dropped: dict[str, set[str]] = {}  # merged code file -> the SDK names dropped from it
        for path in {p for sid in taken for p in code_of[sid]} & set(merged):  # Offramp's code, merged
            merged[path], names = clean(path, base_files[path], merged[path], base_sources | merged)
            dropped |= {path: names} if names else {}
        for path in merged:  # sites of different SDKs faked in one test: one fake per case (Offramp's twins)
            twins = [(site_twins[sid][path], reach[sid]) for sid in taken if path in site_twins[sid]]
            if len(twins) > 1 and (one := unify_twins(merged[path], twins)) != merged[path]:
                merged[path] = one
                unified.add(path)
        dead.clear()  # what the tree last merged (the one placed) removed
        for path in [p for p in merged if is_test_path(p)] if dropped else []:
            merged[path], counts = drop_dead_patches(merged[path], path, dropped, tracked)
            dead.update(counts)
        dead.update(sum((Counter(site_dead.get(sid)) for sid in taken), Counter()))
        gone_imports.clear()  # every SDK import the placed tree dropped: merged, and each site's alone
        for found in [dropped, *(site_dropped.get(sid, {}) for sid in taken)]:
            for path, names in found.items():
                gone_imports.setdefault(path, set()).update(names)
        return merged, taken

    def drift(sids: list[str]) -> dict[str, list[str]]:
        """The backstop and the probe on the merged tree, as it is on disk: site -> why its code or its
        request is not what was evaluated."""
        reference: dict[str, str | None] = dict(base_code)  # the merged tree as Offramp's code wrote it
        for sid in sids:
            for path, text in code_of[sid].items():
                was = reference.get(path)
                reference[path] = None if was is None else merge_onto(base_code[path], was, text)
        on_disk = {p: (repo / p).read_text() for p in sorted(tracked) if (repo / p).is_file()}  # as placed
        for path in {p for sid in sids for p in code_of[sid]}:  # as merge() leaves it: no orphaned SDK import
            if reference.get(path) is not None:
                reference[path] = clean(path, base_code[path], reference[path], on_disk)[0]
        out: dict[str, list[str]] = {}
        for path, why in backstop(repo, reference).items():
            for sid in [s for s in sids if path in code_of[s] or path in versions[s]] or sids:
                out.setdefault(sid, []).append(why)
        for sid, why in (verify([sid for sid in sids if sid not in out]) if probed else {}).items():
            out[sid] = [f"the merged code does not send the evaluated request: {why}"]
        return out

    def culprits(off: dict[str, list[str]], sids: list[str]) -> dict[str, str]:
        """Each site whose code or request changed on the merged tree, and every site whose patch edited a
        non-test module that site reads (its own module, or one it imports): site -> why it drops."""
        out = {}
        for b, problems in off.items():
            out[b] = f"its code or its evaluated call changed on the merged tree: {'; '.join(problems)}"
            module = by_id[b].file
            reads = {module} | _imported_files(read(module) or "", module, tracked)
            for a in sids:
                touched = sorted(p for p in versions[a] if p in reads and not is_test_path(p))
                if a != b and touched and a not in off:
                    out.setdefault(a, f"its patch edits {', '.join(touched)}, which {b}'s request reads, "
                                      f"and {b}'s code or request changed on the merged tree")  # fmt: skip
        return out

    def place(files: dict[str, str]) -> None:  # a tree on disk; a unified test formatted like the repo
        _restore(repo, base_code, backstop(repo, base_code))  # code at the base first: a check may write it
        for path, text in files.items():
            write(path, text)
        if formatted and (tidy := [p for p in files if p in unified]):
            box.tidy(tidy, format_code=True)

    def repair_together(merged: dict[str, str], failure: str, output: str) -> tuple[list[str], str | None]:
        """Up to TOGETHER_REPAIRS Ultra calls editing the failing tests and the fakes they import, the merged
        tree shown: (files changed, None) once it is green, else ([], why not). Disk is back at `merged`
        when it fails."""
        failing = _failing_tests(output)  # the target repo's output: every path in it is untrusted
        modules = {s.file for s in sites}  # call-site modules stay read-only in every layout, flat included

        def fake_or_test(path: str) -> bool:  # a test file no shipped code imports
            return path in tracked and is_test_path(path) and path not in modules and path not in base_code

        editable = {f for f, _ in failing if fake_or_test(f)}
        warnings.extend(f"combined repair: ignored {f} from the check output: not a tracked test or fake file"
                        for f in sorted({f for f, _ in failing} - editable))  # fmt: skip
        for t in sorted(editable):  # the test helpers (fakes) they import from their own directory
            try:
                imports = [n for n in ast.walk(ast.parse(read(t) or "")) if isinstance(n, ast.ImportFrom)]
            except SyntaxError:
                continue
            for node in (n for n in imports if n.module and not n.level):
                helper = (Path(t).parent / f"{node.module.split('.')[0]}.py").as_posix()
                editable |= {helper} if fake_or_test(helper) else set()
        if not editable:  # nothing it may change (a ruff-only or collection failure): no call
            return [], f"{failure} (no failing test or fake the combined repair may edit)"
        shown = {p: read(p) for p in [*merged, *sorted(editable - set(merged))]}
        read_only = frozenset(set(shown) - editable) if locked else frozenset()
        prompt = _together_prompt([by_id[sid] for sid in combined], reach, failure, shown, editable, width)
        messages = [{"role": "system", "content": TOGETHER}, {"role": "user", "content": prompt}]
        tried, checked, applied, base = [(dict(shown), failure)], failure, [], dict(shown)  # as per site
        carried: list = []  # by-name edits of the checked tries the next blocks build on (review M3)
        carried_gone: dict[str, set[str]] = {}  # and the test imports their edits orphaned
        sent: list = []
        hidden: frozenset[str] = frozenset()
        ledger = [_Copy(p, 1, shown[p]) for p in editable]
        for _ in range(TOGETHER_REPAIRS):
            if (stale := _stale(ledger, {p: base[p] for p in editable}, hidden)) is not None:
                log.error("combined repair: %s; nothing was sent to the model", STALE.format(path=stale))
                return [], STALE.format(path=stale)
            try:
                res = llm.chat(role=ROLE, tier=Tier.ULTRA, messages=messages, max_tokens=REPAIR_MAX_TOKENS,
                               temperature=0.0, reasoning="off")  # fmt: skip
            except BudgetExceeded as e:
                return [], f"combined repair not composed: {e}; last failure: {checked}"
            except APIStatusError as e:  # a refusal fails the combined repair; anything else, the run
                if (refused := _refusal(e)) is None:
                    raise
                return [], f"combined repair not composed: {refused}; last failure: {checked}"
            edits = parse_edits(res.output.text or "").edits
            if not edits:
                failure = f"the reply holds no complete edit block; last failure: {checked}"
            else:
                fresh, omitted = _resent(edits, applied, base)
                if (base, hidden, edits) in sent or not (fresh or omitted):
                    return [], f"combined repair produced no change; last failure: {checked}"
                if not fresh:
                    why = f"{ALREADY} applied, and leaving one out does not drop it; last failure: {checked}"
                    return [], why
                sent.append((base, hidden, edits))
                new, notes, gone_now = dict(base), [], {}
                problems = [p.text for p in _apply(fresh, new, read_only, hidden, notes)]
                for p in [p for p in new if new[p] != base[p] and is_test_path(p)] if not problems else []:
                    new[p], names = _drop_unused(
                        p, base[p], new[p], base_sources | new, kept_imports, "the model's edit"
                    )
                    gone_now |= {p: names} if names else {}
                gone = {p: _test_names(shown[p]) - _test_names(new[p]) for p in new if is_test_path(p)}
                problems += [f"patch removes tests from {p}: {sorted(g)}" for p, g in gone.items() if g]
                problems += [
                    _skips_text(p, sk)
                    for p in new
                    if is_test_path(p) and (sk := _new_skips(shown[p], new[p]))
                ]
                for p in sorted(editable) if not problems else []:
                    for name in sorted(
                        n for n in _test_bodies_changed(shown[p], new[p]) if (p, n) not in failing
                    ):
                        problems.append(f"only the failing tests may change: {p}::{name} did")
                if not problems and (seen := next((w for t, w in tried if t == new), None)):
                    return [], f"{RETURNED}; its failure: {seen}"
                if problems:
                    failure = f"patch rejected (patch rules): {'; '.join(problems)}"
                else:
                    changed = [p for p in new if new[p] != shown[p]]
                    for p in changed:
                        write(p, add_fixtures(add_imports(new[p], client_at[1])))
                    _settle(box, repo, changed, shown, width, formatted)
                    result = _no_more_skips(box.check(), baseline_skips)
                    if result.ok and not (off := drift(combined)):
                        together_by_name.extend(dict.fromkeys([*carried, *notes]))
                        together_gone.update(_union(carried_gone, gone_now))
                        warnings.extend(f"combined repair: {_by_name_note(*n)}" for n in together_by_name)
                        return changed, None
                    if result.ok:  # the backstop and the probe: the merged tree may not change shipped code
                        result = CheckResult(ok=False, mode=result.mode, output=_drift_text(off))
                    failure = checked = (
                        f"repo checks failed on the merged patches ({result.mode}): {_summary(result.output)}"
                    )
                    # as checked, the next blocks meet it (after a backstop or probe failure too, unlike per
                    # site: every later merged tree is drift-checked again)
                    base = {**shown, **{p: read(p) for p in changed}}
                    carried += notes  # the next blocks build on this tree
                    carried_gone = _union(carried_gone, gone_now)
                    tried += [(new, failure), (base, failure)]
                    for p in changed:
                        write(p, shown[p])
                    applied += fresh
            repair, again, hidden = _repair(failure, {p: base[p] for p in editable}, _live(ledger))
            messages = [*_supersede(messages, ledger, set(again)),
                        {"role": "assistant", "content": res.output.text or ""},
                        {"role": "user", "content": repair}]  # fmt: skip
            ledger += [_Copy(p, len(messages) - 1, base[p]) for p in again]
        return [], failure

    merged, combined = merge([v.site_id for v in to_migrate if v.site_id in versions])
    place(merged)
    repaired: list[str] = []
    together_failure = None
    together = None
    while combined and (len(combined) > 1 or unchecked):
        together = _no_more_skips(box.check(), baseline_skips)
        if not (off := drift(combined)):
            break
        drop = culprits(off, combined)  # never repaired: only code may change an evaluated call
        for sid, why in drop.items():
            downgrade(by_verdict[sid], f"green alone, but {why}", PatchFailure(kind="merged_drift"))
        for path in merged:
            write(path, base_files[path])
        merged, combined = merge([sid for sid in combined if sid not in drop])
        place(merged)
        unchecked = True  # what is left is checked again
    if together is not None and combined and not together.ok:
        why = f"repo checks failed on the merged patches ({together.mode}): {_summary(together.output)}"
        if len(combined) > 1:
            repaired, together_failure = repair_together(merged, why, together.output)
        else:
            together_failure = why
    if together_failure:
        for path in merged:  # one at a time: keep each site that the others can live with
            write(path, base_files[path])
        kept: list[str] = []
        for sid in combined:
            for version in [versions[sid], *([raw[sid]] if raw[sid] != versions[sid] else [])]:
                versions[sid] = version  # pruned first; as checked alone if pruning broke it
                trial, _ = merge([*kept, sid])  # a superset of the kept sites' files
                place(trial)
                result, off = _no_more_skips(box.check(), baseline_skips), drift([*kept, sid])
                if result.ok and not off:
                    kept.append(sid)
                    break
                current, _ = merge(kept)
                place({path: current.get(path, base_files[path]) for path in trial})
            else:
                others = ", ".join(kept) or "the base"
                alongside = f"not together with {others}" if kept else "not on the merged tree"
                output = result.output or (together.output if together else "")
                why = (
                    f": its code or its evaluated call changed: {_drift_text(off)}"
                    if off
                    else f" ({result.mode}): {_summary(output)}"
                )
                kind = "merged_drift" if off else "not_together"
                downgrade(by_verdict[sid], f"green alone, but {alongside}{why}", PatchFailure(kind=kind))
        combined = kept
    tests_edited: dict[str, list[str]] = {}
    for sid in combined:  # what ships: each site's own change against the base
        diffs[sid] = adapters[sid].patch = _diff(befores[sid], versions[sid])
        site_files[sid] = sorted(versions[sid])
        tests_edited[sid] = [
            t
            for p in site_files[sid]
            if is_test_path(p)
            for t in edited_tests(p, befores[sid].get(p), versions[sid][p])
        ]
    for sid in versions:
        if sid not in combined:
            adapters[sid].patch = None
    migrated = len(combined)
    tests_run, untested = run_site_tests(box, {sid: site_tests[sid] for sid in combined})  # the shipped tree

    warnings += [
        f"removed {n} patch(es) of `{k}`, which Offramp's rewrite removed" for k, n in sorted(dead.items())
    ]
    by_name = [(sid, n) for sid in combined for n in dict.fromkeys(site_notes.get(sid, []))]
    by_name += [("combined repair", n) for n in together_by_name]
    warnings += [f"{who}: {_by_name_note(*n)}" for who, n in by_name if who != "combined repair"]
    warnings += list(kept_imports) if migrated else []
    files = {p: text for p in originals if (text := read(p)) is not None and text != originals[p]}
    files |= {p: text for p, text in dependencies.items() if p in files}  # as written: a CRLF file stays CRLF
    if not migrated:  # nothing uses the client module, nor the openai SDK it imports
        for path in [client_at[0], *dependencies]:
            files.pop(path, None)
        change_notes = []
    keys = {PROVIDER_KEYS[by_id[sid].sdk] for sid in combined if by_id[sid].sdk in PROVIDER_KEYS}
    for sid in combined:  # the migrated function's own key checks, renamed by `_rewrite_call`
        site, key = qual_of[sid], PROVIDER_KEYS.get(by_id[sid].sdk, "")
        if by_id[sid].sdk == "anthropic" and (left := _sdk_left(code_of[sid][site.file], site)):
            why = "the Token Factory client raises openai's errors, which a handler for anthropic's misses"
            refs = ", ".join(f"`{x}`" for x in left)
            change_notes.append(f"`{site.function}` (`{site.file}`) still refers to {refs}: {why}")
        if (
            key
            and "NEBIUS_API_KEY" in code_of[sid][site.file]
            and "NEBIUS_API_KEY" not in befores[sid][site.file]
        ):
            now = f"now checks `NEBIUS_API_KEY` where it checked `{key}`"
            why = "so its Nemotron call runs when the Token Factory key is set"
            change_notes.append(f"`{site.function}` (`{site.file}`) {now}, {why}")
    change_notes += _key_reads(repo, keys)
    handed = [
        qual_of[sid] for sid in combined if _client_parameter(code_of[sid][by_id[sid].file], qual_of[sid])
    ]
    if handed:  # their callers were checked (`_handed_clients`): say how far that goes
        names = ", ".join(f"`{s.function}`" for s in handed)
        reflection = "getattr, importlib, sys.modules, vars/__dict__"
        change_notes.append(f"{names} get{'s' * (len(handed) == 1)} the client from callers: callers were "
                            f"checked statically; lookups through reflection ({reflection}) were not, except "
                            "the literal getattr/import_module forms Offramp refuses")  # fmt: skip
    if filtered:
        change_notes.append(left_out)
    # the placed tree's and a shipped combined repair's, as the repo's own file had them: a name the change
    # added and Offramp dropped is no line of the PR's diff (re-review 2, n5)
    removed = {p: own for p, names in _union(gone_imports, together_gone).items()
               if (own := names & _import_bindings(_parsed(originals.get(p) or "")))}  # fmt: skip
    if migrated and removed:  # review M4 and I5: imports a change orphaned, said
        gone = "; ".join(
            f"{', '.join(f'`{n}`' for n in sorted(ns))} from `{p}`" for p, ns in sorted(removed.items())
        )
        change_notes.append(f"imports that Offramp's or the model's changes left unused were removed: {gone}")
    if migrated and dead:
        patches = ", ".join(f"{n} of `{k}`" for k, n in sorted(dead.items()))
        change_notes.append(f"test patches of those names were removed: {patches}")
    if migrated and by_name:
        edits = "; ".join(
            f"{', '.join(f'`{x}`' for x in names)} in `{p}` ({who})" for who, (p, names) in by_name
        )
        change_notes.append(f"a model edit was applied by function name after its SEARCH missed: {edits}")
    paths = sorted(set(plain) | set(repaired)) if migrated else []
    merged_diff = _diff({p: plain.get(p, originals.get(p)) for p in paths},
                        {p: text for p in paths if (text := read(p)) is not None})  # fmt: skip
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
        combined_repair=repaired,
        combined_failure=together_failure,
        verification={sid: verification.get(sid, UNCHANGED) for sid in combined},
        tests_run=tests_run,
        untested=untested,
        tests_edited=tests_edited,
        warnings=warnings,
        linted=box.ruff,
        sandbox_notes=box.notes,
        async_client=any(FACTORIES[1] in text for p, text in files.items() if p != client_at[0]),
        client_packaged=(repo / client_at[0]).parent != repo
        and ((repo / client_at[0]).parent / "__init__.py").is_file(),
        change_notes=change_notes,
        merged_diff=merged_diff,
    )
