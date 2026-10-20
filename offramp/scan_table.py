"""The `offramp scan` table: each call site's shape and its list-price ratio to each Nemotron tier.

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


def references() -> list[dict[str, Any]]:
    """The Nemotron tiers' pricing.yaml entries, cheapest first: what the ratios divide by."""
    out = []
    for tier in TIER_ORDER:
        model, e = next((m, e) for m, e in load_pricing().items() if e.get("tier") == tier.value)
        keys = ("input_usd_per_m", "output_usd_per_m", "price_status", "as_of", "source")
        out.append({"model": model, "tier": tier.value} | {k: e[k] for k in keys})
    return out


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


def _ratio(
    site: CallSite, refs: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, float]] | None, str | None]:
    if site.kind == "embedding":
        return None, "no Nemotron embedding tier"
    if not site.source_model:
        return None, "model set at runtime"
    try:
        e = entry(site.source_model)
    except KeyError:
        return None, "no list price on file"
    return {
        ref["tier"]: {
            "input": round(e["input_usd_per_m"] / ref["input_usd_per_m"], 2),
            "output": round(e["output_usd_per_m"] / ref["output_usd_per_m"], 2),
        }
        for ref in refs
    }, None


def table(result: ScanResult) -> dict[str, Any]:
    """The table as JSON: what `/api/scan` returns and what `render` prints."""
    refs = references()
    rows = []
    for s in result.sites:
        shape, guessed = _shape(s)
        stream = s.params.get("stream")
        ratio, note = _ratio(s, refs)
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
    return {"references": refs, "ratio_label": RATIO_LABEL, "rows": rows, "warnings": result.warnings}


def _x(v: float) -> str:
    return f"{v:.3g}"


def ratio_text(ratio: dict[str, dict[str, float]]) -> str:
    """Per tier, cheapest first: the input–output ratio, one number when equal ("2.5× / 0.5–0.67×")."""
    parts = []
    for r in ratio.values():
        lo, hi = sorted((r["input"], r["output"]))
        parts.append(f"{_x(lo)}×" if lo == hi else f"{_x(lo)}–{_x(hi)}×")
    return " / ".join(parts)


def render(result: ScanResult) -> str:
    t = table(result)
    refs, rows = t["references"], t["rows"]
    if not rows:
        lines = [f"No OpenAI, Anthropic, LiteLLM or LangChain call sites found in {result.repo}."]
    else:
        tiers = " / ".join(ref["tier"].capitalize() for ref in refs)
        head = ["id", "file:line", "provider / model", "shape", "flags", f"list-price ratio vs {tiers}*"]
        body = [
            [
                r["id"],
                f"{r['file']}:{r['line']}",
                f"{r['sdk']} {r['model'] or '(model set at runtime)'}",
                r["shape"] + ("~" if r["shape_guessed"] else ""),
                ", ".join(r["flags"]) or "-",
                r["price_note"] if r["price_ratio"] is None else ratio_text(r["price_ratio"]),
            ]
            for r in rows
        ]
        widths = [max(len(row[i]) for row in [head, *body]) for i in range(len(head))]
        prices = ", ".join(
            f"{ref['tier'].capitalize()} ${ref['input_usd_per_m']:.2f}/${ref['output_usd_per_m']:.2f}"
            for ref in refs
        )
        status = " and ".join(sorted({ref["price_status"].replace("-", " ") for ref in refs}))
        lines = [
            f"{len(rows)} call sites in {result.repo} "
            "(static scan: no key, no model call, the code was not run)",
            "",
            *(
                "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
                for row in [head, *body]
            ),
            "",
            f"* {RATIO_LABEL}: the source model's list price per token over each Nemotron tier's "
            f"({prices} per 1M input/output tokens; {status}, as of {refs[0]['as_of']}). "
            "A range is the input and the output token ratio; below 1× the source model lists cheaper.",
            "~ guessed from the prompt text. "
            "A full run (`offramp run`) labels every call and measures parity.",
        ]
    lines += [f"warning: {w}" for w in t["warnings"]]
    return "\n".join(lines)
