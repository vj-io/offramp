"""The `offramp scan` table: each call site's shape and its list-price ratio to the cheapest Nemotron tier.

Static, like the scan: no key, no model call, the scanned code is never run. The shape comes from the call's
own parameters when they settle it (tools, response_format); otherwise it is guessed from the prompt text and
marked as guessed. A full run labels every site with a model and measures parity; this table does neither.
"""

from __future__ import annotations

import re
from typing import Any

from offramp.agents.scanner import EXPR
from offramp.pricing import entry, load_pricing
from offramp.schemas import TIER_ORDER, CallSite, ScanResult

RATIO_LABEL = "list-price ratio, not measured parity"
JSON_ASK = re.compile(r"\bjson\b", re.IGNORECASE)
CLASSIFY_ASK = re.compile(
    r"\b(classif(y|ies|ication)|categori[sz]e|exactly one of|one of the following)\b", re.IGNORECASE
)


def reference() -> dict[str, Any]:
    """The cheapest Nemotron tier's pricing.yaml entry: what every ratio divides by."""
    tier = TIER_ORDER[0].value
    model, e = next((m, e) for m, e in load_pricing().items() if e.get("tier") == tier)
    return {"model": model, "tier": tier} | {
        k: e[k] for k in ("input_usd_per_m", "output_usd_per_m", "price_status", "as_of", "source")
    }


def _shape(site: CallSite) -> tuple[str, bool]:
    """(shape, guessed): tool call / JSON / classification / free text / embedding / unknown."""
    if site.kind == "embedding":
        return "embedding", False
    p = site.params
    tools = p.get("tools")
    if tools == EXPR or (isinstance(tools, list | tuple) and tools):
        return "tool call", False
    fmt = p.get("response_format")
    if not fmt and isinstance(p.get("text"), dict):
        fmt = p["text"].get("format")  # Responses API: text={"format": {...}}
    if fmt and not (isinstance(fmt, dict) and fmt.get("type") == "text"):
        return "JSON", False
    t = site.template
    parts = [t.system, t.user] if t else [p.get("system"), p.get("instructions")]
    text = "\n".join(x for x in parts if isinstance(x, str) and x != EXPR)  # not system blocks (a list)
    if not text:
        return "unknown", False
    if JSON_ASK.search(text):
        return "JSON", True
    if CLASSIFY_ASK.search(text):
        return "classification", True
    return "free text", True


def _ratio(site: CallSite, ref: dict[str, Any]) -> tuple[dict[str, float] | None, str | None]:
    if site.kind == "embedding":
        return None, "no Nemotron embedding tier"
    if not site.source_model:
        return None, "model set at runtime"
    try:
        e = entry(site.source_model)
    except KeyError:
        return None, "no list price on file"
    return {
        "input": round(e["input_usd_per_m"] / ref["input_usd_per_m"], 1),
        "output": round(e["output_usd_per_m"] / ref["output_usd_per_m"], 1),
    }, None


def table(result: ScanResult) -> dict[str, Any]:
    """The table as JSON: what `/api/scan` returns and what `render` prints."""
    ref = reference()
    rows = []
    for s in result.sites:
        shape, guessed = _shape(s)
        stream = s.params.get("stream")
        ratio, note = _ratio(s, ref)
        flags = ["streams"] if stream is True else ["may stream"] if stream == EXPR else []
        if "NEEDS_RUNTIME_CAPTURE" in s.flags:
            flags.append("prompt built at runtime")
        rows.append(
            {
                "id": s.id,
                "file": s.file,
                "line": s.lines[0],
                "sdk": s.sdk,
                "model": s.source_model,
                "shape": shape,
                "shape_guessed": guessed,
                "streaming": True if stream is True else None if stream == EXPR else False,
                "flags": flags,
                "price_ratio": ratio,
                "price_note": note,
            }
        )
    return {"reference": ref, "ratio_label": RATIO_LABEL, "rows": rows, "warnings": result.warnings}


def _x(v: float) -> str:
    return f"{v:g}x"


def render(result: ScanResult) -> str:
    t = table(result)
    ref, rows = t["reference"], t["rows"]
    if not rows:
        lines = [f"No OpenAI, Anthropic, LiteLLM or LangChain call sites found in {result.repo}."]
    else:
        tier = ref["tier"].capitalize()
        head = ["id", "file:line", "provider / model", "shape", "flags", f"list-price ratio vs {tier}*"]
        body = [
            [
                r["id"],
                f"{r['file']}:{r['line']}",
                f"{r['sdk']} {r['model'] or '(model set at runtime)'}",
                r["shape"] + ("~" if r["shape_guessed"] else ""),
                ", ".join(r["flags"]) or "-",
                (
                    r["price_note"]
                    if r["price_ratio"] is None
                    else _x(r["price_ratio"]["input"])
                    if r["price_ratio"]["input"] == r["price_ratio"]["output"]
                    else f"{_x(r['price_ratio']['input'])} in / {_x(r['price_ratio']['output'])} out"
                ),
            ]
            for r in rows
        ]
        widths = [max(len(row[i]) for row in [head, *body]) for i in range(len(head))]
        status = ref["price_status"].replace("-", " ")
        lines = [
            f"{len(rows)} call sites in {result.repo} "
            "(static scan: no key, no model call, the code was not run)",
            "",
            *(
                "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
                for row in [head, *body]
            ),
            "",
            f"* {RATIO_LABEL}: the source model's list price per token over Nemotron {tier}'s "
            f"({ref['model']}, ${ref['input_usd_per_m']} in / ${ref['output_usd_per_m']} out per 1M tokens, "
            f"{status}, as of {ref['as_of']}).",
            "~ guessed from the prompt text. "
            "A full run (`offramp run`) labels every call and measures parity.",
        ]
    lines += [f"warning: {w}" for w in t["warnings"]]
    return "\n".join(lines)
