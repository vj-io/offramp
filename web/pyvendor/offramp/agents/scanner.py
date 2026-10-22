"""Scanner Agent (PRD §6.1): deterministic AST scan for closed-LLM call sites, then Super contract labelling.

Rendered prompts use str.format syntax: `{name}` placeholders with literal braces doubled, so
`site.template.user.format(**values)` gives the text the code would send.
"""

from __future__ import annotations

import ast
import errno
import json
import logging
import operator
import os
import re
import string
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from warnings import catch_warnings, simplefilter

from pydantic import BaseModel, Field

from offramp.schemas import CallSite, Contract, PromptTemplate, ScanResult, Tier

if TYPE_CHECKING:  # the static scan imports only pydantic and the schemas, so /api/scan can vendor it
    from offramp.tf.client import LLM

EXPR = "<expr>"  # a value the scanner could not resolve statically
PARAM_KEYS = (
    "tools",
    "tool_choice",
    "response_format",
    "stream",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "temperature",
    "reasoning_effort",
    "system",
    "instructions",
    "text",
)
SKIP_DIRS = {"tests", "venv", "site-packages", "node_modules", "__pycache__"}
SDK_EXPORTS = {  # name an SDK module exports -> what it is
    "openai": {"OpenAI": "client", "AsyncOpenAI": "client"},
    "anthropic": {"Anthropic": "client", "AsyncAnthropic": "client"},
    "litellm": {"completion": "completion", "acompletion": "completion"},
}
LANGCHAIN_CLASSES = {"ChatOpenAI", "ChatAnthropic"}
METHODS = {  # attribute chain after a client -> (sdk, kind)
    ("chat", "completions", "create"): ("openai", "chat"),
    ("responses", "create"): ("openai", "chat"),
    ("embeddings", "create"): ("openai", "embedding"),
    ("messages", "create"): ("anthropic", "chat"),
    # the SDKs' streaming helpers (`with client.messages.stream(...) as s:`) send the same request, streamed
    ("chat", "completions", "stream"): ("openai", "chat"),
    ("responses", "stream"): ("openai", "chat"),
    ("messages", "stream"): ("anthropic", "chat"),
}
MUTATORS = {
    "append", "extend", "insert", "update", "pop", "popitem", "remove", "clear", "setdefault", "add",
    "discard", "sort", "reverse",
}  # fmt: skip
ARITH = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
CONVERSIONS = {"r": repr, "s": str, "a": ascii}
KINDS = ["literal", "variable", "fstring"]  # prompt_source of renderable prompts, weakest first
PARAM = object()  # _lookup result for an unassigned function parameter
# Bounds for untrusted repos (/api/scan): past them a file is skipped, or a value becomes unknown (EXPR).
MAX_FILE_BYTES = 1_000_000
MAX_SITES = 2000  # per scan: each site keeps its rendered prompt, so the whole scan stops here
MAX_SIZE = 100_000  # characters of a resolved string; characters plus items of a resolved list, tuple or dict
MAX_INT_BITS = 10_000  # str() of an int over 4,300 digits raises ValueError
SPEC_NUMBER = re.compile(r"\d+")  # widths and precisions in a format spec

Fn = ast.FunctionDef | ast.AsyncFunctionDef
Scope = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None  # None: module level
log = logging.getLogger(__name__)


# ---------------------------------------------------------------- scan


def scan_repo(path: str | Path, deadline: float | None = None) -> ScanResult:
    """Every closed-LLM call site under `path`. Deterministic, no LLM. Raises TimeoutError once
    time.monotonic() passes `deadline`."""
    root = Path(path)
    found: list[CallSite] = []
    warnings: list[str] = []
    for file in _py_files(root):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"scan of {path} passed its deadline")
        if len(found) > MAX_SITES:
            break
        rel = file.relative_to(root).as_posix()
        if not file.resolve().is_relative_to(root.resolve()):  # its code would go out as a site's context
            warnings.append(f"skipped {rel}: a symlink to outside the repository")
            continue
        try:
            size = file.stat().st_size
            if size > MAX_FILE_BYTES:  # parse and analysis cost grow with size; call sites sit in small files
                warnings.append(f"skipped {rel}: over 1 MB ({size} bytes)")
                continue
            source = file.read_bytes()
        except OSError as exc:  # unreadable, or a broken symlink named *.py
            warnings.append(f"skipped {rel}: not readable ({type(exc).__name__})")
            continue
        try:
            with catch_warnings():  # the target's invalid escapes ("\s") are not ours to report
                simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)  # bytes, so a BOM or a coding cookie is honoured
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            if isinstance(exc, MemoryError) and not exc.args:
                raise  # out of memory, not the parser's "stack overflowed" limit: the caller's bound decides
            version = f"{sys.version_info.major}.{sys.version_info.minor}"  # this interpreter's grammar
            warnings.append(f"skipped {rel}: not parseable as Python {version} ({type(exc).__name__})")
            continue
        try:
            found += _Module(tree, deadline).sites(rel, MAX_SITES + 1 - len(found))
        except (TimeoutError, MemoryError):
            raise
        except RecursionError:
            warnings.append(f"skipped {rel}: nested too deeply to analyse (RecursionError)")
        except Exception as exc:  # one file never stops the scan, whatever it holds
            if isinstance(exc, OSError) and exc.errno == errno.ENOMEM:  # an mmap refused by RLIMIT_AS
                raise MemoryError from exc
            warnings.append(f"skipped {rel}: could not be analysed ({type(exc).__name__})")
    if len(found) > MAX_SITES:
        found = found[:MAX_SITES]
        warnings.append(f"truncated at {MAX_SITES:,} call sites: the rest of the repository was not scanned")
    for warning in warnings:
        log.warning("scanner: %s", warning)
    found.sort(key=lambda s: (s.file, s.lines))
    chat = [s for s in found if s.kind == "chat"]
    embeddings = [s for s in found if s.kind == "embedding"]
    for prefix, group in (("cs", chat), ("emb", embeddings)):
        for i, site in enumerate(group, 1):
            site.id = f"{prefix}_{i:02d}"
    return ScanResult(repo=str(path), sites=chat + embeddings, warnings=warnings)


def _py_files(root: Path) -> Iterator[Path]:
    """In sorted order, so a scan cut at MAX_SITES keeps the same sites on every filesystem."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not (
                d.startswith(".") or d in SKIP_DIRS or os.path.exists(os.path.join(dirpath, d, "pyvenv.cfg"))
            )
        )
        yield from (Path(dirpath, f) for f in sorted(filenames) if f.endswith(".py"))


def _walk(
    node: ast.AST, fn: Fn | None = None, scope: Scope = None, cls: ast.ClassDef | None = None
) -> Iterator[tuple[ast.AST, Fn | None, Scope, ast.ClassDef | None]]:
    """Every node with its enclosing function, name scope (function or class body) and class, in order."""
    for child in ast.iter_child_nodes(node):
        yield child, fn, scope, cls
        if isinstance(child, Fn):
            yield from _walk(child, child, child, cls)
        elif isinstance(child, ast.ClassDef):
            yield from _walk(child, fn, child, child)
        else:
            yield from _walk(child, fn, scope, cls)


def _mutated(node: ast.AST) -> list[str]:
    """Names whose binding or object `node` changes: x.append(), x[k] = v, x.a = v, del x, global x."""
    if isinstance(node, ast.Global):
        return node.names
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
        return [node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in MUTATORS:
        target = node.func.value
    elif isinstance(node, ast.Subscript | ast.Attribute) and isinstance(node.ctx, ast.Store | ast.Del):
        target = node.value
    else:
        return []
    while isinstance(target, ast.Subscript | ast.Attribute):
        target = target.value
    return [target.id] if isinstance(target, ast.Name) else []


def _assigned(node: ast.AST) -> tuple[str, ast.expr] | None:
    """(name, value) for `name = value` and `name: T = value`."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id, node.value
    if isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
        return node.target.id, node.value
    return None


def _dict(node: ast.AST) -> dict[Any, ast.expr] | None:
    """Key -> value node of a dict display whose keys are all literals."""
    if not isinstance(node, ast.Dict) or not all(isinstance(k, ast.Constant) for k in node.keys):
        return None
    return {k.value: v for k, v in zip(node.keys, node.values, strict=True)}


def _scalar(value: Any) -> bool:
    return isinstance(value, str | int | float) and value != EXPR


def _known(value: Any) -> bool:
    """Fully resolved: no EXPR anywhere inside."""
    if isinstance(value, str):
        return value != EXPR
    if isinstance(value, list | tuple):
        return all(map(_known, value))
    if isinstance(value, dict):
        return all(map(_known, [*value, *value.values()]))
    return value is None or isinstance(value, int | float)


def _large(value: Any) -> bool:
    """An int or string json.dumps would turn into a big output (indent, separators)."""
    if isinstance(value, str):
        return len(value) > 16
    return isinstance(value, int) and not isinstance(value, bool) and value > 16


def _wide(spec: str) -> bool:
    """A format spec whose width or precision would build a text over MAX_SIZE: never evaluated."""
    return any(len(n) > 6 or int(n) > MAX_SIZE for n in SPEC_NUMBER.findall(spec))


def _formatted(value: Any, conversion: str | None, spec: Any) -> str | None:
    """The text of one f-string field, `format(value!conversion, spec)`, or None when it is not known."""
    if not (_known(value) and isinstance(spec, str) and spec != EXPR) or _wide(spec):
        return None
    try:
        text = format(CONVERSIONS.get(conversion, lambda v: v)(value), spec)
    except (TypeError, ValueError):
        return None
    return text if len(text) <= MAX_SIZE else None


def _conversion(code: int) -> str | None:
    return None if code == -1 else chr(code)


class _Text(NamedTuple):
    """A rendered prompt fragment."""

    text: str  # str.format template
    slots: dict[str, tuple[str, str]]  # placeholder -> (source expression, "!r:.2f"-style format or "")
    kind: str  # one of KINDS
    ref: str | None = None  # module constant that holds this fragment


def _lit(text: str, kind: str = "literal") -> _Text:
    return _Text(text.replace("{", "{{").replace("}", "}}"), {}, kind)


def _slot(source: str, kind: str, fmt: str = "") -> _Text:
    """One placeholder for a runtime expression, named from its source text."""
    name = re.sub(r"\W+", "_", source).strip("_")
    name = name if name.isidentifier() else f"v_{name}"
    return _Text("{" + name + "}", {name: (source, fmt)}, kind)


def _join(parts: list[_Text | None], sep: str = "", kind: str | None = None) -> _Text | None:
    if any(p is None for p in parts):
        return None
    if sum(len(p.text) for p in parts) + len(sep) * len(parts) > MAX_SIZE:
        return None
    slots: dict[str, tuple[str, str]] = {}
    for p in parts:
        for name, info in p.slots.items():
            if slots.setdefault(name, info) != info:
                return None  # one placeholder name for two expressions or two formats
    return _Text(
        sep.join(p.text for p in parts),
        slots,
        kind or max((p.kind for p in parts), key=KINDS.index, default="literal"),
        next((p.ref for p in parts if p.ref), None),
    )


class _Module:
    """One parsed file: its SDK imports, client variables, module constants and the call sites they reveal."""

    def __init__(self, tree: ast.Module, deadline: float | None = None):
        self.deadline = deadline
        # Memos, keyed by (id of a definition node, id of its scope): each name is resolved and rendered once,
        # so a chain like `a1 = a0 + a0` ... `a30 = a29 + a29` costs 30 steps, not 2**30.
        self._resolved: dict[tuple[int, int], Any] = {}
        self._rendered: dict[tuple[int, int], _Text | None] = {}
        self._sizes: dict[int, tuple[Any, int]] = {}  # id -> (container kept alive, so ids stay unique, size)
        self.refs: dict[str, tuple[str, str]] = {}  # local name -> (sdk, "module"|"client"|"completion"|...)
        self.imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    top = a.name.split(".")[0]
                    if top in SDK_EXPORTS:
                        self.imported.add(top)
                        if a.asname is None or a.name == top:
                            self.refs[a.asname or top] = (top, "module")
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in SDK_EXPORTS:
                    self.imported.add(top)
                for a in node.names:
                    if what := SDK_EXPORTS.get(node.module, {}).get(a.name):
                        self.refs[a.asname or a.name] = (node.module, what)
                    elif top.startswith("langchain") and a.name in LANGCHAIN_CLASSES:
                        self.refs[a.asname or a.name] = ("langchain", "constructor")

        stores: Counter[str] = Counter()
        consts: dict[str, ast.expr] = {}
        for stmt in tree.body:
            if isinstance(stmt, Fn | ast.ClassDef):
                continue
            stores.update(
                n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            )
            if assigned := _assigned(stmt):
                consts[assigned[0]] = assigned[1]
        # A constant mutated or rebound anywhere in the file (also inside functions) is not a constant.
        changed = {name for n in ast.walk(tree) for name in _mutated(n)}
        self.consts = {k: v for k, v in consts.items() if stores[k] == 1 and k not in changed}

        self.nodes = list(_walk(tree))
        self.parent: dict[Fn, Fn | None] = {n: fn for n, fn, _, _ in self.nodes if isinstance(n, Fn)}
        self._scopes: dict[Fn, tuple[dict[str, ast.expr], Counter[str], set[str]]] = {}
        # Client bindings per scope, in source order: `client = OpenAI()` (module or function),
        # `self.client = client or Anthropic()` (class), parameters annotated `client: OpenAI | None`.
        self.clients: dict[tuple[Scope, str], set[str]] = {}  # (scope, receiver source text) -> sdks
        for n, fn, scope, cls in self.nodes:
            self._tick()
            if isinstance(n, ast.arg):
                pairs = [(n.arg, n.annotation)]
            elif isinstance(n, ast.AnnAssign):
                pairs = [(ast.unparse(n.target), n.annotation), (ast.unparse(n.target), n.value)]
            elif isinstance(n, ast.Assign):
                pairs = [
                    (ast.unparse(t), n.value) for t in n.targets if isinstance(t, ast.Name | ast.Attribute)
                ]
            else:
                continue
            for text, value in pairs:
                if value is not None and (sdks := self._client_sdks(value, fn, cls)):
                    self.clients.setdefault(self._key(text, scope, cls), set()).update(sdks)

    # ------------------------------------------------------------ detection

    def _ref(self, node: ast.AST) -> tuple[str, str] | None:
        """What an SDK-imported name, or `sdk_module.Name`, refers to."""
        if isinstance(node, ast.Name):
            return self.refs.get(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            mod = self.refs.get(node.value.id)
            if mod and mod[1] == "module" and node.attr in SDK_EXPORTS[mod[0]]:
                return mod[0], SDK_EXPORTS[mod[0]][node.attr]
        return None

    @staticmethod
    def _key(text: str, scope: Scope, cls: ast.ClassDef | None) -> tuple[Scope, str]:
        if text.startswith("self."):
            return cls, text
        if isinstance(scope, ast.ClassDef):  # class attribute, read through self
            return scope, f"self.{text}"
        return scope, text

    def _bound(self, node: ast.Name | ast.Attribute, fn: Fn | None, cls: ast.ClassDef | None) -> set[str]:
        """SDKs a receiver was bound to as a client, resolved like Python from the function outwards."""
        text = ast.unparse(node)
        if text.startswith("self."):
            return self.clients.get((cls, text), set())
        head = node
        while isinstance(head, ast.Attribute):
            head = head.value
        scope = fn
        while scope is not None:
            if (scope, text) in self.clients:
                return self.clients[(scope, text)]
            _, counts, params = self._scope(scope)
            if isinstance(head, ast.Name) and (head.id in counts or head.id in params):
                return set()  # a local of this function that holds no known client
            scope = self.parent[scope]
        return self.clients.get((None, text), set())

    def _client_sdks(self, node: ast.AST, fn: Fn | None, cls: ast.ClassDef | None) -> set[str]:
        """SDKs a client expression can hold: `OpenAI()`, a bound name, `x or Anthropic()`, `T | None`."""
        if isinstance(node, ast.Call):
            ref = self._ref(node.func)
            return {ref[0]} if ref and ref[1] == "client" else set()
        if isinstance(node, ast.Name | ast.Attribute):
            ref = self._ref(node)
            return {ref[0]} if ref and ref[1] == "client" else self._bound(node, fn, cls)
        if isinstance(node, ast.BoolOp):
            kids = node.values
        elif isinstance(node, ast.IfExp):
            kids = [node.body, node.orelse]
        elif isinstance(node, ast.BinOp):
            kids = [node.left, node.right]
        elif isinstance(node, ast.Subscript):
            kids = [node.slice]
        else:
            return set()
        return set().union(*(self._client_sdks(k, fn, cls) for k in kids))

    def _receiver_is(self, node: ast.expr, sdk: str, fn: Fn | None, cls: ast.ClassDef | None) -> bool:
        ref = self._ref(node)
        if ref:
            return ref == (sdk, "module")  # openai.chat.completions.create(...)
        bound = self._client_sdks(node, fn, cls)
        if bound:
            return sdk in bound
        # ponytail: an untracked receiver (injected client, factory result) counts when this file imports the
        # SDK; a client used in a file that never imports its SDK is missed (add cross-module tracking then).
        return sdk in self.imported

    def _match(self, call: ast.Call, fn: Fn | None, cls: ast.ClassDef | None) -> tuple[str, str] | None:
        ref = self._ref(call.func)
        if ref and ref[1] == "completion":
            return ref[0], "chat"
        if ref and ref[1] == "constructor":
            return "langchain", "chat"
        chain: tuple[str, ...] = ()
        node = call.func
        while isinstance(node, ast.Attribute):
            chain, node = (node.attr, *chain), node.value
            hit = METHODS.get(chain)
            if hit and self._receiver_is(node, hit[0], fn, cls):
                return hit
        return None

    def sites(self, file: str, limit: int | None = None) -> list[CallSite]:
        """The file's call sites in source order, at most `limit` of them."""
        out: list[CallSite] = []
        for node, fn, _, cls in self.nodes:
            if limit is not None and len(out) >= limit:
                break
            if isinstance(node, ast.Call):
                self._tick()
                if hit := self._match(node, fn, cls):
                    out.append(self._site(file, node, fn, *hit))
        return out

    def _tick(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError("the scan passed its deadline")

    def _size(self, value: Any) -> int:
        """Characters plus items of a resolved value, counted as a tree (resolved values share structure)."""
        if isinstance(value, str):
            return len(value)
        if not isinstance(value, list | tuple | dict):
            return 1
        if (hit := self._sizes.get(id(value))) is not None:
            return hit[1]
        kids = [*value, *value.values()] if isinstance(value, dict) else value
        size = 1 + sum(self._size(k) for k in kids)
        self._sizes[id(value)] = (value, size)
        return size

    def _capped(self, value: Any) -> Any:
        return value if self._size(value) <= MAX_SIZE else EXPR

    def _site(self, file: str, call: ast.Call, fn: Fn | None, sdk: str, kind: str) -> CallSite:
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        model_node = kw.get("model", kw.get("model_name"))
        model = None if model_node is None else self.resolve(model_node, fn)
        params = {k: self.resolve(kw[k], fn) for k in PARAM_KEYS if k in kw}
        if isinstance(call.func, ast.Attribute) and call.func.attr == "stream":
            params["stream"] = True  # a streaming helper: the SDK sends stream=True
        rendered = self._template(kw, fn) if kind == "chat" and sdk != "langchain" else None
        flags = []
        if params.get("stream") in (True, EXPR):
            flags.append("MANUAL_REVIEW")
        if kind == "embedding":
            flags.append("EMBEDDINGS")
        elif rendered is None:
            flags.append("NEEDS_RUNTIME_CAPTURE")
        template, source, ref, specs = rendered or (None, "dynamic", None, {})
        if specs:  # case rendering pre-formats these values the way the code would
            params["template_format_specs"] = specs
        return CallSite(
            id="",
            file=file,
            lines=[call.lineno, call.end_lineno],
            kind=kind,
            sdk=sdk,
            call=ast.unparse(call.func),
            function=fn.name if fn else None,
            source_model=model if _scalar(model) and isinstance(model, str) else None,
            prompt_source=source,
            prompt_ref=ref,
            template=template,
            params=params,
            flags=flags,
        )

    # ------------------------------------------------------------ name resolution

    def _scope(self, fn: Fn) -> tuple[dict[str, ast.expr], Counter[str], set[str]]:
        """Single-assignment values, store counts (mutations count double) and parameters of a function."""
        if fn not in self._scopes:
            values: dict[str, ast.expr] = {}
            counts: Counter[str] = Counter()
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    counts[n.id] += 1
                for name in _mutated(n):
                    counts[name] += 2
                if assigned := _assigned(n):
                    values[assigned[0]] = assigned[1]
            a = fn.args
            params = {p.arg for p in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg] if p}
            self._scopes[fn] = values, counts, params
        return self._scopes[fn]

    def _lookup(self, name: str, fn: Fn | None) -> tuple[ast.expr, Fn | None] | object | None:
        """(value, scope) for a name assigned exactly once, PARAM for an unassigned parameter, else None."""
        if fn is not None:
            values, counts, params = self._scope(fn)
            if counts[name]:
                return (values[name], fn) if counts[name] == 1 and name in values else None
            if name in params:
                return PARAM
        return (self.consts[name], None) if name in self.consts else None

    def _deref(self, node: ast.expr, fn: Fn | None) -> tuple[ast.expr, Fn | None]:
        """Follow names to the expression they were assigned once."""
        seen = set()
        while isinstance(node, ast.Name) and node.id not in seen:
            found = self._lookup(node.id, fn)
            if found is None or found is PARAM:
                break
            seen.add(node.id)
            node, fn = found
        return node, fn

    def resolve(self, node: ast.expr, fn: Fn | None, seen: tuple[str, ...] = ()) -> Any:
        """Constant-fold an expression from literals and single-assignment names; EXPR where that fails or the
        value would pass MAX_SIZE."""
        self._tick()
        if isinstance(node, ast.Name):
            found = None if node.id in seen else self._lookup(node.id, fn)
            if found is None or found is PARAM:
                return EXPR
            key = (id(found[0]), id(found[1]))
            if key not in self._resolved:
                self._resolved[key] = self.resolve(found[0], found[1], (*seen, node.id))
            return self._resolved[key]
        if isinstance(node, ast.List):
            return self._capped([self.resolve(e, fn, seen) for e in node.elts])
        if isinstance(node, ast.Tuple):
            return self._capped(tuple(self.resolve(e, fn, seen) for e in node.elts))
        if isinstance(node, ast.Dict):
            d = _dict(node)
            return EXPR if d is None else self._capped({k: self.resolve(v, fn, seen) for k, v in d.items()})
        if isinstance(node, ast.JoinedStr):
            texts, size = [], 0
            for v in node.values:
                if isinstance(v, ast.FormattedValue):
                    spec = "" if v.format_spec is None else self.resolve(v.format_spec, fn, seen)
                    text = _formatted(self.resolve(v.value, fn, seen), _conversion(v.conversion), spec)
                else:
                    text = v.value
                if text is None or (size := size + len(text)) > MAX_SIZE:
                    return EXPR
                texts.append(text)
            return "".join(texts)
        if isinstance(node, ast.BinOp) and type(node.op) in ARITH:
            a, b = self.resolve(node.left, fn, seen), self.resolve(node.right, fn, seen)
            if all(isinstance(x, int | float) and not isinstance(x, bool) for x in (a, b)):
                value = ARITH[type(node.op)](a, b)
                return EXPR if isinstance(value, int) and value.bit_length() > MAX_INT_BITS else value
            if isinstance(node.op, ast.Add) and type(a) is type(b) and isinstance(a, str | list | tuple):
                if EXPR in (a, b) or len(a) + len(b) > MAX_SIZE:
                    return EXPR
                return self._capped(a + b)
            return EXPR
        if isinstance(node, ast.Call):
            return self._fold_call(node, fn, seen)
        try:
            return self._capped(ast.literal_eval(node))
        except (ValueError, TypeError, SyntaxError):
            return EXPR

    def _fold_call(self, call: ast.Call, fn: Fn | None, seen: tuple[str, ...]) -> Any:
        f = call.func
        if isinstance(f, ast.Attribute) and f.attr == "model_json_schema":
            return f"<pydantic:{ast.unparse(f.value).split('.')[-1]}>"
        args = [self.resolve(a, fn, seen) for a in call.args]
        kwargs = {k.arg: self.resolve(k.value, fn, seen) for k in call.keywords if k.arg}
        name = ast.unparse(f)
        if name == "json.dumps" and len(args) == 1:
            if isinstance(args[0], str) and args[0].startswith("<pydantic:"):
                return args[0]  # stands for that model's schema text
            if any(_large(v) for v in kwargs.values()):
                return EXPR  # indent=10**9 would build gigabytes
            try:
                text = json.dumps(args[0], **kwargs)
            except (TypeError, ValueError):
                return EXPR
            return EXPR if EXPR in text or len(text) > MAX_SIZE else text
        if name in ("os.getenv", "os.environ.get") and len(args) == 2:
            return args[1]
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "join"
            and len(args) == 1
            and isinstance(args[0], list | tuple)
        ):
            sep = self.resolve(f.value, fn, seen)
            if (
                _scalar(sep)
                and isinstance(sep, str)
                and all(_scalar(a) and isinstance(a, str) for a in args[0])
                and sum(map(len, args[0])) + len(sep) * len(args[0]) <= MAX_SIZE
            ):
                return sep.join(args[0])
        return EXPR

    # ------------------------------------------------------------ prompt rendering

    def _template(
        self, kw: dict[str, ast.expr], fn: Fn | None
    ) -> tuple[PromptTemplate, str, str | None, dict[str, str]] | None:
        """(template, prompt_source, prompt_ref, placeholder formats) with the prompt as one system + one user
        message; None when it cannot be known statically."""
        msgs = kw.get("messages", kw.get("input"))
        turns = None if msgs is None else self._messages(msgs, fn)
        if turns is None and "input" in kw:  # Responses API: input="..."
            turns = [("user", self._content(kw["input"], fn))]
        if turns is None:
            return None
        system = [self._content(kw[k], fn) for k in ("system", "instructions") if k in kw]
        system += [t for role, t in turns if role in ("system", "developer")]
        user = [t for role, t in turns if role == "user"]
        # ponytail: PromptTemplate holds one system + one user message, so multi-turn and assistant-prefill
        # prompts go to runtime capture. Extend the schema if real repos need them.
        if len(user) != 1 or any(role not in ("system", "developer", "user") for role, _ in turns):
            return None
        system_text = _join(system, "\n\n") if system else None
        whole = _join([system_text, *user] if system else user)
        if whole is None:
            return None
        template = PromptTemplate(
            system=system_text.text if system_text else None, user=user[0].text, variables=list(whole.slots)
        )
        specs = {name: fmt for name, (_, fmt) in whole.slots.items() if fmt}
        return template, whole.kind, whole.ref, specs

    def _messages(self, node: ast.expr, fn: Fn | None) -> list[tuple[Any, _Text | None]] | None:
        node, fn = self._deref(node, fn)
        if not isinstance(node, ast.List | ast.Tuple):
            return None
        turns = []
        for elt in node.elts:
            msg, scope = self._deref(elt, fn)
            d = _dict(msg)
            if d is None or "role" not in d or "content" not in d:
                return None
            turns.append((self.resolve(d["role"], scope), self._content(d["content"], scope)))
        return turns

    def _content(self, node: ast.expr, fn: Fn | None) -> _Text | None:
        """A message body: a string expression or a list of text blocks."""
        blocks, scope = self._deref(node, fn)
        if isinstance(blocks, ast.List | ast.Tuple):
            texts = [_dict(b) for b in blocks.elts]
            return _join([self._render(d["text"], scope) if d and "text" in d else None for d in texts], "\n")
        return self._render(node, fn)

    def _render(self, node: ast.expr, fn: Fn | None, seen: tuple[str, ...] = ()) -> _Text | None:
        """A string expression as a str.format template; None when its text cannot be known statically."""
        self._tick()
        if isinstance(node, ast.Name) and node.id not in seen:
            found = self._lookup(node.id, fn)
            if found is PARAM:
                return _slot(node.id, "variable")
            if found is None:
                return None
            value, scope = found
            if scope is not None:  # local variable assigned once
                key = (id(value), id(scope))
                if key not in self._rendered:
                    self._rendered[key] = self._render(value, scope, (*seen, node.id))
                return self._rendered[key]
            text = self.resolve(value, None)
            return _lit(str(text), "variable")._replace(ref=node.id) if _scalar(text) else None
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.FormattedValue):
                    spec = "" if v.format_spec is None else self.resolve(v.format_spec, fn)
                    parts.append(self._field(v.value, _conversion(v.conversion), spec, fn))
                else:
                    parts.append(_lit(v.value))
            return _join(parts, kind="fstring")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):  # both sides render, so both are str
            t = _join([self._render(node.left, fn, seen), self._render(node.right, fn, seen)])
            return t._replace(kind="fstring") if t and t.slots else t
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
            return self._format(node, fn)
        text = self.resolve(node, fn)
        return _lit(str(text)) if _scalar(text) else None

    def _field(self, expr: ast.expr, conversion: str | None, spec: Any, fn: Fn | None) -> _Text | None:
        """One f-string or str.format field: its exact text when every name in it is a known constant, else
        one placeholder named from its source, with any conversion/spec recorded for case rendering."""
        if not (isinstance(spec, str) and spec != EXPR and "{" not in spec) or _wide(spec):
            return None  # the format spec is decided at runtime, or too wide to ever apply
        text = _formatted(self.resolve(expr, fn), conversion, spec)
        if text is not None:
            return _lit(text, "fstring")
        return _slot(
            ast.unparse(expr),
            "fstring",
            (f"!{conversion}" if conversion else "") + (f":{spec}" if spec else ""),
        )

    def _format(self, call: ast.Call, fn: Fn | None) -> _Text | None:
        """TEMPLATE.format(name=value): literal text kept, each field rendered like an f-string field."""
        base = call.func.value
        template = self.resolve(base, fn)
        kwargs = {k.arg: k.value for k in call.keywords}
        if not (_scalar(template) and isinstance(template, str)) or call.args or None in kwargs:
            return None
        try:
            fields = list(string.Formatter().parse(template))
        except ValueError:
            return None
        parts = []
        for literal, field, spec, conversion in fields:
            parts.append(_lit(literal, "fstring"))
            if field is not None:
                if field not in kwargs:  # positional, indexed or attribute field, or a KeyError at runtime
                    return None
                parts.append(self._field(kwargs[field], conversion, spec, fn))
        t = _join(parts, kind="fstring")
        ref = base.id if isinstance(base, ast.Name) and base.id in self.consts else None
        return t and t._replace(ref=ref)


# ---------------------------------------------------------------- Super labelling

SCANNER_MAX_TOKENS = 1024  # thinking is off, so this budget holds only the ContractLabel JSON
SNIPPET_BEFORE, SNIPPET_AFTER = 20, 15  # lines of code around the call shown to the labeller

LABEL_SYSTEM = """\
You label one LLM call site from a Python codebase with its output contract, so its behaviour can be
re-tested on another model. Return JSON with `contract` and `confidence`.

contract.task_type, one of:
- classification: picks one label from a fixed set (put the labels, verbatim, in contract.labels)
- json_extraction: returns JSON with a fixed set of fields
- tool_call: is given tools and must call one
- free_text: prose such as answers, replies or summaries
- reasoning: multi-step math or logic whose final answer is checked
contract.output: label | json_schema | json | tool_call | text (json_schema when the code supplies a schema).
contract.tools / contract.streaming: whether the call passes tools / streams its response.
contract.reasoning_heavy: true when the task needs multi-step reasoning.
contract.latency_budget_ms: only when the code states a timeout, else null.
contract.json_schema: always null (the scanner sets it from the code).
confidence: your probability, 0 to 1, that task_type is right.

In the prompt template, {name} placeholders are runtime inputs. "<expr>" marks a value the scanner could not
resolve and "<pydantic:Name>" stands for that Pydantic model's JSON schema."""


class ContractLabel(BaseModel):
    """Super's answer for one call site."""

    contract: Contract
    confidence: float = Field(ge=0.0, le=1.0)  # the labeller's confidence that task_type is right


def label_contracts(sites: list[CallSite], llm: LLM, root: str | Path | None = None) -> list[CallSite]:
    """Fill `contract` and `confidence` on every chat site, one Super call each (PRD §6.1 step 3).

    `root` is the scanned repo (ScanResult.repo); when given, the code around each call goes into the prompt.
    Parameters that settle the task type override the model. A reply that does not validate leaves the
    contract empty, confidence 0 and the site flagged MANUAL_REVIEW.
    """
    out = []
    for site in sites:
        if site.kind != "chat":
            out.append(site)
            continue
        result = llm.chat(
            role="scanner",
            tier=Tier.SUPER,
            messages=[
                {"role": "system", "content": LABEL_SYSTEM},
                {"role": "user", "content": _describe(site, root)},
            ],
            max_tokens=SCANNER_MAX_TOKENS,
            temperature=0.0,
            response_model=ContractLabel,
            reasoning="off",  # Nemotron thinks by default (docs/plan/tf-probe.md); labelling needs none
        )
        label: ContractLabel | None = result.parsed
        hints = _hints(site)
        certain = "task_type" in hints
        if root is not None and "tools" not in hints and (parsed := json_reply(root, site)):
            hints["reply_json"] = parsed  # from the code, like the other hints: the labeller is not asked
        if label is None:  # keep what the code settles, and ask a human for the rest
            flags = site.flags if "MANUAL_REVIEW" in site.flags else [*site.flags, "MANUAL_REVIEW"]
            update = {
                "contract": Contract(**hints) if certain else None,
                "confidence": 1.0 if certain else 0.0,
            }
            out.append(site.model_copy(update={**update, "flags": flags}))
            continue
        # The labeller never supplies json_schema: it is set only from a literal schema in the code (or later
        # from the resolved Pydantic model), since a model-written schema can dangle ($ref with no $defs).
        contract = label.contract.model_copy(update={"json_schema": None, **hints})
        out.append(
            site.model_copy(update={"contract": contract, "confidence": 1.0 if certain else label.confidence})
        )
    return out


# ---------------------------------------------------------------- how the code parses the reply

REPLY_ATTRS = {
    "content",
    "text",
    "output_text",
}  # the attribute a chain from the response ends in at the text
STRIPS = {"strip", "lstrip", "rstrip"}
UNFENCE = {"removeprefix", "removesuffix", "replace"}  # a step of the reply's flow only with a ``` argument
FENCE = "```"


def _has_fence(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Constant) and isinstance(n.value, str) and FENCE in n.value for n in ast.walk(node)
    )


def _own_nodes(fn: ast.AST) -> Iterator[ast.AST]:
    """The nodes of `fn` itself: not those of functions, lambdas or classes defined inside it."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, Fn | ast.Lambda | ast.ClassDef):
            stack.extend(ast.iter_child_nodes(node))


def _bindings(fn: ast.AST) -> dict[str, list[ast.AST | None]]:
    """Each local name and every value `fn` binds to it; None for a binding of any other kind (a loop or
    with target, an unpacking, +=), so such a name is never taken for the reply."""
    out: dict[str, list[ast.AST | None]] = {}
    plain: set[int] = set()
    for node in _own_nodes(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign | ast.NamedExpr) and isinstance(node.target, ast.Name):
            target, value = node.target, node.value
        else:
            continue
        plain.add(id(target))
        out.setdefault(target.id, []).append(value)
    for node in _own_nodes(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and id(node) not in plain:
            out.setdefault(node.id, []).append(None)
    return out


class _Flow:
    """The reply's flow through one function: names bound to the response (`resp`), to the reply text after
    nothing but strips and fence removal (`text`), and to its lines while a fence is being removed (`lines`).
    `text` and `lines` map a name to whether a ``` fence step is on its flow."""

    def __init__(self, call: ast.AST | None, resp: set[str], text: dict[str, bool], lines: dict[str, bool]):
        self.call, self.resp, self.text, self.lines = call, resp, text, lines

    def is_resp(self, e: ast.AST) -> bool:
        if self.call is not None and (e is self.call or (isinstance(e, ast.Await) and e.value is self.call)):
            return True
        if isinstance(e, ast.Name):
            return e.id in self.resp
        return isinstance(e, ast.Attribute | ast.Subscript) and self.is_resp(e.value)

    def text_of(self, e: ast.AST) -> bool | None:
        """Whether `e` is the reply text itself (stripped, or with a ``` fence removed): None if it is not,
        else whether a fence step is on its flow."""
        if isinstance(e, ast.Name):
            return self.text.get(e.id)
        if isinstance(e, ast.Attribute) and e.attr in REPLY_ATTRS and self.is_resp(e.value):
            return False
        if not (isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute)):
            return None
        method, base = e.func.attr, e.func.value
        if method in STRIPS and (inner := self.text_of(base)) is not None:
            return inner or any(_has_fence(a) or "`" in str(getattr(a, "value", "")) for a in e.args)
        if method in UNFENCE and e.args and _has_fence(e.args[0]) and self.text_of(base) is not None:
            return True
        if method == "join" and isinstance(base, ast.Constant) and len(e.args) == 1:
            return self.lines_of(e.args[0])
        if method == "sub" and isinstance(base, ast.Name) and base.id == "re" and len(e.args) >= 3:
            if _has_fence(e.args[0]) and self.text_of(e.args[2]) is not None:
                return True
        return None

    def lines_of(self, e: ast.AST) -> bool | None:
        if isinstance(e, ast.Name):
            return self.lines.get(e.id)
        if (
            isinstance(e, ast.Call)
            and isinstance(e.func, ast.Attribute)
            and e.func.attr in ("split", "splitlines")
        ):
            return self.text_of(e.func.value)
        if isinstance(e, ast.ListComp | ast.GeneratorExp) and len(e.generators) == 1:
            g = e.generators[0]
            if not (
                isinstance(g.target, ast.Name) and isinstance(e.elt, ast.Name) and e.elt.id == g.target.id
            ):
                return None  # each line changed: not a filter
            inner = self.lines_of(g.iter)
            if inner is None or (g.ifs and not all(_has_fence(c) for c in g.ifs)):
                return None  # dropping lines for another reason than a fence changes the reply
            return inner or bool(g.ifs)
        return None


def _flow(fn: ast.AST, call: ast.AST | None, seeds: set[str]) -> _Flow:
    """The reply's flow in `fn`, `seeds` being names that hold the reply text on entry (a helper's parameter).
    A name is in a set only if every binding of it is such a value (the largest such sets) and it gets its
    value from the call or a seed (the smallest such set): one other binding keeps it out."""
    binds = _bindings(fn)
    names = {n for n, values in binds.items() if None not in values}
    flow = _Flow(call, set(names), dict.fromkeys(names | seeds, False), dict.fromkeys(names, False))
    while True:  # largest sets closed under "every value is of the kind"
        resp = {n for n in flow.resp if all(flow.is_resp(v) for v in binds[n])}
        text = {n for n in flow.text if all(flow.text_of(v) is not None for v in binds.get(n, []))}
        lines = {n for n in flow.lines if all(flow.lines_of(v) is not None for v in binds[n])}
        if (resp, text, lines) == (flow.resp, set(flow.text), set(flow.lines)):
            break
        flow.resp, flow.text, flow.lines = resp, dict.fromkeys(text, False), dict.fromkeys(lines, False)
    grounded, grew = set(seeds), True
    while grew:  # names that get a value from the call or a seed, to a fixed point
        before = len(grounded)
        for n, values in binds.items():
            for v in values:
                if v is not None and any(
                    x is call or (isinstance(x, ast.Name) and x.id in grounded) for x in ast.walk(v)
                ):
                    grounded.add(n)
        grew = len(grounded) > before
    flow.resp &= grounded
    flow.text = {n: False for n in flow.text if n in grounded}
    flow.lines = {n: False for n in flow.lines if n in grounded}
    changed = True
    while changed:  # whether a fence step is on each name's flow
        changed = False
        for kind, of in ((flow.text, flow.text_of), (flow.lines, flow.lines_of)):
            for n in kind:
                if not kind[n] and any(of(v) for v in binds.get(n, [])):
                    kind[n] = changed = True
    return flow


def _tolerated(fn: ast.AST, node: ast.AST, flow: _Flow) -> bool:
    """Whether `node` sits in the body of a try whose handler accepts a reply that is not JSON: it returns or
    stores the reply text, or carries on after the try. Only a handler ending in raise, or in a return of
    something else, gives the call up (logging the reply on the way is no use of it)."""
    parents = {id(c): p for p in [fn, *_own_nodes(fn)] for c in ast.iter_child_nodes(p)}
    reply = set(flow.text) | set(flow.lines)

    def keeps_reply(handler: ast.ExceptHandler) -> bool:
        values = [
            n.value
            for stmt in handler.body
            for n in ast.walk(stmt)
            if isinstance(n, ast.Return | ast.Assign | ast.AnnAssign | ast.AugAssign | ast.NamedExpr)
            and n.value is not None
        ]
        return any(isinstance(x, ast.Name) and x.id in reply for v in values for x in ast.walk(v))

    child, parent = node, parents.get(id(node))
    while parent is not None:
        if isinstance(parent, ast.Try) and child in parent.body:
            for handler in parent.handlers:
                if keeps_reply(handler) or not isinstance(handler.body[-1], ast.Return | ast.Raise):
                    return True
        child, parent = parent, parents.get(id(parent))
    return False


def _parses_reply(fn: ast.AST, call: ast.AST | None, seeds: set[str], loads: tuple[set[str], set[str]]):
    """True/False (a ``` fence step on the reply's flow or not) when `fn` runs json.loads on the reply text
    itself, directly or through a helper defined inside it, where a reply that is not JSON breaks the code;
    None otherwise. ponytail: one function's local names only; a missed site is still judged."""
    flow = _flow(fn, call, seeds)
    modules, functions = loads
    helpers = {
        h.name: fenced
        for h in _own_nodes(fn)
        if isinstance(h, Fn) and h.args.args
        if (fenced := _parses_reply(h, None, {h.args.args[0].arg}, loads)) is not None
    }
    found: list[bool] = []
    for node in _own_nodes(fn):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        f = node.func
        is_loads = (isinstance(f, ast.Name) and f.id in functions) or (
            isinstance(f, ast.Attribute)
            and f.attr == "loads"
            and isinstance(f.value, ast.Name)
            and f.value.id in modules
        )
        helper = isinstance(f, ast.Name) and f.id in helpers
        if not (is_loads or helper) or (fenced := flow.text_of(node.args[0])) is None:
            continue
        if not _tolerated(fn, node, flow):
            found.append(fenced or (helper and helpers[f.id]))
    return any(found) if found else None


def _json_loads_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """(names bound to the json module, names bound to json.loads) by the file's imports."""
    modules, functions = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.asname or a.name for a in node.names if a.name == "json"}
        elif isinstance(node, ast.ImportFrom) and node.module == "json":
            functions |= {a.asname or a.name for a in node.names if a.name == "loads"}
    return modules, functions


def json_reply(root: str | Path, site: CallSite) -> str | None:
    """How the function around `site`'s call parses the reply, read from the code: "exact" when it runs
    json.loads on the reply text itself (stripped at most), "fence_stripped" when a ``` fence is removed
    from that text first; None when it never does, parses something else (a regex match, a slice, a
    document built around the reply), falls back to the text when json.loads fails, or cannot be read.
    Conservative on purpose: a site it misses is still judged; one it flags wrongly fails good replies."""
    try:
        with catch_warnings():
            simplefilter("ignore", SyntaxWarning)
            tree = ast.parse((Path(root) / site.file).read_bytes())
    except (OSError, SyntaxError, ValueError, RecursionError):
        return None
    found = next(
        (
            (node, fn)
            for node, fn, _, _ in _walk(tree)
            if isinstance(node, ast.Call)
            and [node.lineno, node.end_lineno] == site.lines
            and ast.unparse(node.func) == site.call
        ),
        None,
    )
    if found is None or found[1] is None:
        return None
    fenced = _parses_reply(found[1], found[0], set(), _json_loads_names(tree))
    return None if fenced is None else "fence_stripped" if fenced else "exact"


def _describe(site: CallSite, root: str | Path | None) -> str:
    facts = site.model_dump(
        mode="json",
        include={
            "file",
            "lines",
            "function",
            "sdk",
            "call",
            "source_model",
            "prompt_source",
            "template",
            "params",
            "flags",
        },
    )
    text = "Call site:\n" + json.dumps(facts, indent=2)
    if root is not None:
        lines = (Path(root) / site.file).read_text(encoding="utf-8").splitlines()
        start, end = site.lines
        code = "\n".join(lines[max(0, start - 1 - SNIPPET_BEFORE) : end + SNIPPET_AFTER])
        text += f"\n\nCode:\n```python\n{code}\n```"
    return text


def _hints(site: CallSite) -> dict[str, Any]:
    """Contract fields the call's own parameters settle; they override the labeller (task_type => certain)."""
    p = site.params
    fmt = p.get("response_format")
    if not isinstance(fmt, dict) and isinstance(p.get("text"), dict):
        fmt = p["text"].get("format")  # Responses API: text={"format": {...}}
    update: dict[str, Any] = {}
    tools = p.get("tools")
    if isinstance(tools, list | tuple) and tools:  # an unresolved (EXPR) list may be empty at runtime
        update = {"task_type": "tool_call", "output": "tool_call", "tools": True}
    elif isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        inner = fmt.get("json_schema")  # chat: {"json_schema": {"schema": ...}}; Responses: {"schema": ...}
        schema = (inner if isinstance(inner, dict) else fmt).get("schema")
        update = {"task_type": "json_extraction", "output": "json_schema"}
        if isinstance(schema, dict):
            update["json_schema"] = schema
    if p.get("stream") is True:
        update["streaming"] = True
    return update
