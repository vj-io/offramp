"""Token Factory by the numbers: aggregate the Token Factory calls Offramp recorded. No network, no keys.

    uv run python scripts/tf_stats.py --cache CACHE_DIR --runs RUNS_DIR > docs/evidence/tf-production.json
    uv run python scripts/tf_stats.py --cache CACHE_DIR --runs RUNS_DIR --format md \
        > docs/evidence/tf-production.md

CACHE_DIR is var/tf-cache and RUNS_DIR is var/runs (neither is in the repo).

Sources (all optional; the committed cassettes are the default, so a clean clone reproduces their part):
  --cache      the response cache: one JSON per live call, written by TFClient.
  --cassettes  tests/fixtures/cassettes: the same entry shape; hand-authored ones carry "synthetic": true
               and are excluded. A call present in both is counted once.
  --runs       var/runs/*: events.jsonl and budget.json per run. They hold no per-call data; they are used
               to attribute cache entries to runs (by time window), to reconcile token totals and to list
               errors.

Only fields that exist in the entries are used. The role of a call is NOT recorded: it is read off the
system prompt (ROLE_PREFIXES); a call whose system prompt matches none of them is the migrated app's own
call, which the budget names "eval_target". The output holds aggregates, the 60-character heads of the
system prompts that matched no role (so the role reading can be checked) and the run-log messages; never
request or response text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from offramp.tf.budget import EVAL_TARGET_ROLE
from offramp.tf.client import strip_think, validate_structured
from offramp.tf.tiers import DEFAULT_MODEL_IDS

ROOT = Path(__file__).resolve().parents[1]
REPAIR_ASK = "Your previous reply is not valid."  # the repair retry's last user message (_repair_messages)
ROLE_PREFIXES = (  # system-prompt start -> the role passed to TFClient.chat (offramp/agents/*.py)
    ("You grade answers written by an AI assistant", "judge"),
    ("You write realistic test inputs", "case_gen"),
    ("You map test-fixture records", "case_gen"),
    ("You read a support policy", "case_gen"),
    ("You are an independent auditor", "reference_audit"),
    ("You migrate one LLM call site", "patcher"),  # the patcher's first system prompt
    ("Offramp migrates one LLM call site", "patcher"),
    ("Offramp migrated several LLM call sites", "patcher"),
    ("You are Offramp's Iterator Agent", "iterator"),
    ("You are Offramp's Adapter Agent", "adapter"),
    ("You write one qualitative sentence", "verdict"),
    ("You write the prose of a GitHub pull request", "pr_writer"),
    ("You extract facts about ONE AI model", "pricing_extract"),
    ("You label one LLM call site", "scanner"),
)
TIER_OF = {model: tier.value for tier, model in DEFAULT_MODEL_IDS.items()}  # insertion order = tier order


def role_of(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """(role, system-prompt head). Role is EVAL_TARGET_ROLE when no agent prefix matches."""
    system = next((str(m.get("content") or "") for m in messages if m.get("role") == "system"), "")
    for prefix, role in ROLE_PREFIXES:
        if system.startswith(prefix):
            return role, system[:60]
    return EVAL_TARGET_ROLE, system[:60]


def reply_text(entry: dict[str, Any]) -> str:
    message = ((entry["response"].get("choices") or [{}])[0]).get("message") or {}
    return strip_think(message.get("content") or "")


def to_call(entry: dict[str, Any], source: str) -> dict[str, Any]:
    """One flat record from a cache or cassette entry."""
    request, response = entry["request"], entry["response"]
    params, messages = request.get("params") or {}, request["messages"]
    choice = (response.get("choices") or [{}])[0]
    usage = entry.get("usage") or response.get("usage") or {}
    raw_usage = response.get("usage") or {}
    role, head = role_of(messages)
    fmt = params.get("response_format") or {}
    kind = fmt.get("type") if fmt.get("type") in ("json_schema", "json_object") else None
    valid = None
    if kind and not (choice.get("message") or {}).get("tool_calls"):
        schema = fmt["json_schema"].get("schema") if kind == "json_schema" else None
        valid = validate_structured(reply_text(entry), schema, None)[1] is None
    last = messages[-1] if messages else {}
    repair = last.get("role") == "user" and str(last.get("content") or "").startswith(REPAIR_ASK)
    thinking = ((params.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking")
    return {
        "source": source,
        "model": request["model"],
        "role": role,
        "system_head": head,
        "created_at": entry.get("created_at"),
        "latency_ms": entry.get("latency_ms") or None,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "prompt_cache_hit_tokens": raw_usage.get("prompt_cache_hit_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "thinking": thinking,
        "max_tokens": params.get("max_tokens"),
        "tool_choice": params.get("tool_choice"),
        "json_kind": kind,
        "json_valid": valid,
        "repair_retry": repair,
        "repair_prev": strip_think(str(messages[-2].get("content") or "")) if repair else None,
        "repair_of_synthetic": False,
    }


def load_calls(directories: list[tuple[str, Path]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Every distinct recorded call in the (source, directory) pairs, in order. Synthetic entries and
    non-entry JSON are skipped; a copy of the same call (same response id and created_at) is counted once.
    A repair retry whose rejected reply is a synthetic entry's reply is flagged repair_of_synthetic."""
    calls: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    synthetic_replies: set[str] = set()
    meta: Counter[str] = Counter()
    for source, directory in directories:
        for path in sorted(directory.rglob("*.json")):
            try:
                entry = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if not (isinstance(entry, dict) and {"request", "response"} <= entry.keys()):
                continue
            if entry.get("synthetic"):
                meta["synthetic_excluded"] += 1
                synthetic_replies.add(reply_text(entry))
                continue
            ident = (entry["response"].get("id"), entry.get("created_at"))
            if ident in seen:
                meta["duplicate_copies_dropped"] += 1
                continue
            seen.add(ident)
            calls.append(to_call(entry, source))
    for c in calls:
        c["repair_of_synthetic"] = c.pop("repair_prev") in synthetic_replies
    return calls, dict(meta)


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Per run: its time window, budget.json, and the error / warning events."""
    runs = []
    for run in sorted(p for p in runs_dir.iterdir() if (p / "events.jsonl").exists()):
        events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line]
        if not events:
            continue
        budget = run / "budget.json"
        logs = [e["payload"] for e in events if e["type"] == "log"]
        warnings = [p["message"] for p in logs if p["level"] == "warn"]
        runs.append(
            {
                "run_id": run.name,
                "start": events[0]["ts"],
                "end": events[-1]["ts"],
                "budget": json.loads(budget.read_text()) if budget.exists() else None,
                "errors": [e["payload"] for e in events if e["type"] == "error"],
                "warnings": warnings,
            }
        )
    return runs


def assign_scopes(calls: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    """A cache call made inside a run's window belongs to that run (the token sums match budget.json)."""
    for c in calls:
        c["scope"] = "cassette" if c["source"] == "cassette" else "cache_outside_runs"
        if c["source"] == "cache" and c["created_at"]:
            for r in runs:
                if _ts(r["start"]) <= _ts(c["created_at"]) <= _ts(r["end"]):
                    c["scope"] = r["run_id"]
                    break


def percentile(values: list[int], p: int) -> int | None:
    """Nearest-rank percentile: always one of the recorded values, never interpolated."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(p / 100 * len(ordered)), 1) - 1]


def summarise(calls: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(calls)
    lat = [c["latency_ms"] for c in calls if c["latency_ms"]]
    tin, tout = [c["input_tokens"] for c in calls], [c["output_tokens"] for c in calls]
    finish = Counter(c["finish_reason"] for c in calls)
    reporting = [c for c in calls if c["prompt_cache_hit_tokens"] is not None]
    judged = [c for c in calls if c["json_valid"] is not None]
    invalid = Counter(c["finish_reason"] for c in judged if not c["json_valid"])
    latency = {"n": len(lat), "p50": percentile(lat, 50), "p95": percentile(lat, 95)}
    latency["max"] = max(lat, default=None)
    return {
        "calls": n,
        "input_tokens": {"total": sum(tin), "p50": percentile(tin, 50)},
        "output_tokens": {"total": sum(tout), "p50": percentile(tout, 50)},
        "latency_ms": latency,
        "finish_reasons": dict(sorted(finish.items(), key=lambda kv: str(kv[0]))),
        "length_share": finish["length"] / n if n else None,
        "prompt_cache": {
            "hit_tokens": sum(c["prompt_cache_hit_tokens"] for c in reporting),
            "prompt_tokens_of_reporting_calls": sum(c["input_tokens"] for c in reporting),
        },
        "json": {
            "calls_with_response_format": len(judged),
            "valid": sum(c["json_valid"] for c in judged),
            "invalid_by_finish_reason": dict(invalid),
            "repair_retry_calls": sum(c["repair_retry"] for c in calls),
            "repair_retries_of_synthetic_replies": sum(c["repair_of_synthetic"] for c in calls),
        },
    }


def group(calls: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        out[tuple(c[k] for k in keys)].append(c)
    return out


def tier_rank(model: str) -> int:
    return list(TIER_OF).index(model) if model in TIER_OF else len(TIER_OF)


def reconcile(calls: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per run: what the cache holds for its window against what its budget.json charged."""
    ultra = DEFAULT_MODEL_IDS[next(t for t in DEFAULT_MODEL_IDS if t.value == "ultra")]
    rows = []
    for r in runs:
        mine = [c for c in calls if c["scope"] == r["run_id"]]
        b = r["budget"] or {}
        agent = sum(1 for c in mine if c["model"] == ultra and c["role"] != EVAL_TARGET_ROLE)
        row = {
            "run_id": r["run_id"],
            "cache_calls": len(mine),
            "cache_tokens_in": sum(c["input_tokens"] for c in mine),
            "cache_tokens_out": sum(c["output_tokens"] for c in mine),
            "cache_ultra_agent_calls": agent,
            "budget_tokens_in": b.get("tokens_in"),
            "budget_tokens_out": b.get("tokens_out"),
            "budget_ultra_agent_calls": b.get("ultra_agent_calls"),
            "budget_tf_spend_usd": b.get("tf_spend_usd"),
        }
        cached = (row["cache_tokens_in"], row["cache_tokens_out"], agent)
        charged = (b.get("tokens_in"), b.get("tokens_out"), b.get("ultra_agent_calls"))
        row["matches_budget"] = cached == charged
        rows.append(row)
    return rows


def aggregate(
    calls: list[dict[str, Any]], runs: list[dict[str, Any]], meta: dict[str, int]
) -> dict[str, Any]:
    run_ids = {r["run_id"] for r in runs}
    in_runs = [c for c in calls if c["scope"] in run_ids]
    cache_only = [c for c in calls if c["source"] == "cache"]
    by_model_role: dict[str, dict[str, Any]] = defaultdict(dict)
    for (m, role), v in group(calls, "model", "role").items():
        by_model_role[m][role] = summarise(v)
    cassette_only = [c for c in calls if c["source"] == "cassette"]
    eval_target = [c for c in calls if c["role"] == EVAL_TARGET_ROLE]

    def by_thinking(sel: list[dict[str, Any]]) -> dict[str, Any]:
        flags = sorted(group(sel, "model", "thinking").items(), key=lambda kv: str(kv[0]))
        return {
            f"{m} | {'on' if t else 'off' if t is False else 'unset'}": summarise(v) for (m, t), v in flags
        }

    length_keys = (
        "model",
        "role",
        "source",
        "thinking",
        "max_tokens",
        "output_tokens",
        "tool_choice",
        "json_kind",
        "scope",
    )
    return {
        "inputs": {
            "runs": len(runs),
            "calls_by_source": dict(Counter(c["source"] for c in calls)),
            "calls_by_scope": dict(sorted(Counter(c["scope"] for c in calls).items())),
            "created_at_first": min(c["created_at"] for c in calls),
            "created_at_last": max(c["created_at"] for c in calls),
            **meta,
        },
        "tier_of_model": TIER_OF,
        "overall": summarise(calls),
        "by_model": {m: summarise(v) for (m,), v in sorted(group(calls, "model").items())},
        "by_model_in_runs": {m: summarise(v) for (m,), v in sorted(group(in_runs, "model").items())},
        "by_model_cache_only": {m: summarise(v) for (m,), v in sorted(group(cache_only, "model").items())},
        "by_model_cassette_only": {
            m: summarise(v) for (m,), v in sorted(group(cassette_only, "model").items())
        },
        "by_model_role": dict(by_model_role),
        "by_model_thinking": by_thinking(calls),
        "by_model_thinking_eval_target": by_thinking(eval_target),
        "by_scope": {k[0]: summarise(v) for k, v in sorted(group(calls, "scope").items())},
        "length_calls": [{k: c[k] for k in length_keys} for c in calls if c["finish_reason"] == "length"],
        "run_reconciliation": reconcile(calls, runs),
        "run_events": {
            "runs_failed": [{"run_id": r["run_id"], **e} for r in runs for e in r["errors"]],
            "warnings": [{"run_id": r["run_id"], "message": w} for r in runs for w in r["warnings"]],
        },
        "role_evidence": {
            "eval_target_system_prompt_heads": dict(
                Counter(c["system_head"] for c in calls if c["role"] == EVAL_TARGET_ROLE).most_common()
            ),
        },
    }


# ---------------------------------------------------------------- markdown


def tier_name(model: str) -> str:
    return TIER_OF.get(model, model).capitalize()


def seconds(ms: int | None) -> str:
    return "n/a" if ms is None else f"{ms / 1000:.1f}"


def stat_cells(s: dict[str, Any]) -> list[str]:
    """The columns every table shares, from one summary."""
    n, f, j, lat = s["calls"], s["finish_reasons"], s["json"], s["latency_ms"]
    other = n - f.get("stop", 0) - f.get("tool_calls", 0) - f.get("length", 0)
    finish = f"{f.get('stop', 0)} / {f.get('tool_calls', 0)} / {f.get('length', 0)}"
    finish += f" (+{other})" if other else ""
    return [
        f"{n:,}",
        f"{s['input_tokens']['total']:,} / {s['input_tokens']['p50']:,}",
        f"{s['output_tokens']['total']:,} / {s['output_tokens']['p50']:,}",
        f"{seconds(lat['p50'])} / {seconds(lat['p95'])}{'†' if n < 20 else ''}",
        finish,
        f"{s['length_share']:.1%}{'†' if n < 20 else ''}",
        f"{j['valid']} / {j['calls_with_response_format']}" if j["calls_with_response_format"] else "n/a",
    ]


STAT_HEADERS = [
    "Calls",
    "Input tokens (total / p50)",
    "Output tokens (total / p50)",
    "Latency s (p50 / p95)",
    "Finish: stop / tool_calls / length",
    "`length` share",
    "JSON valid",
]


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def reflow(md: str, width: int = 110) -> str:
    """Re-wrap prose and bullet paragraphs at `width`; headings, tables and code fences pass through."""
    out = []
    for block in md.split("\n\n"):
        if not block:
            continue
        if block.startswith(("#", "|", "```")):
            out.append(block)
            continue
        items = re.split(r"\n(?=- )", block)
        out.append(
            "\n".join(
                textwrap.fill(
                    " ".join(i.split()),
                    width,
                    subsequent_indent="  " if i.startswith("- ") else "",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                for i in items
            )
        )
    return "\n\n".join(out)


def render_md(r: dict[str, Any]) -> str:
    inp, o, j, ev = r["inputs"], r["overall"], r["overall"]["json"], r["run_events"]
    runs, by_scope = r["run_reconciliation"], r["by_scope"]
    models = sorted(r["by_model"], key=tier_rank)
    ids = ", ".join(f"{tier_name(m)} = `{m}`" for m in models)
    first, last = inp["created_at_first"][:10], inp["created_at_last"][:10]
    n_runs = len(runs)
    n_cache, n_cassette = (inp["calls_by_source"].get(s, 0) for s in ("cache", "cassette"))
    in_runs = sum(x["cache_calls"] for x in runs)
    all_ok = all(x["matches_budget"] for x in runs)
    spend = sum(x["budget_tf_spend_usd"] or 0 for x in runs)
    fin = o["finish_reasons"]
    if n_cache:
        scope = (
            f"all {o['calls']:,} recorded calls: {n_cache:,} from the response cache plus {n_cassette:,} "
            f"from the committed cassettes, with {n_runs} run directories for the run checks."
        )
    else:
        scope = (
            f"the {n_cassette:,} calls in the committed cassettes only. No response cache was given, so "
            "this is not the full set: the figures differ from a document generated with the cache and "
            "the run files, and there are no run tables."
        )

    def calls_in(scope: str) -> str:
        return f"{by_scope.get(scope, {}).get('calls', 0):,}"

    sources = table(
        ["Where the calls were recorded", "Calls"],
        [[f"Run `{x['run_id']}` (cache entries inside its time window)", calls_in(x["run_id"])] for x in runs]
        + [
            ["Response cache, not inside any recorded run window", calls_in("cache_outside_runs")],
            [
                "Committed cassettes in `tests/fixtures/cassettes` (committed test and dev recordings, "
                "recorded by TFClient, not marked synthetic)",
                calls_in("cassette"),
            ],
            ["**Total**", f"**{o['calls']:,}**"],
        ],
    )

    def model_table(key: str) -> str:
        rows = [[tier_name(m), *stat_cells(r[key][m])] for m in models if m in r[key]]
        return table(["Model", *STAT_HEADERS], rows)

    more_models = "\n\n".join(
        f"{caption}\n\n{model_table(key)}"
        for caption, key in (
            (f"Only the calls inside the runs' time windows ({in_runs:,} calls):", "by_model_in_runs"),
            (f"Only the response cache ({n_cache:,} calls, no cassettes):", "by_model_cache_only"),
            (
                f"Only the committed cassettes ({n_cassette:,} calls; what a clean clone reproduces):",
                "by_model_cassette_only",
            ),
        )
        if r[key] and sum(s["calls"] for s in r[key].values()) != o["calls"]
    )

    role_table = table(
        ["Model", "Role", *STAT_HEADERS],
        [
            [tier_name(m), role, *stat_cells(s)]
            for m in models
            for role, s in sorted(r["by_model_role"][m].items(), key=lambda kv: -kv[1]["calls"])
        ],
    )

    def split_flags(key: str) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (m, flag): s
            for (m, flag), s in sorted(
                ((tuple(k.split(" | ")), s) for k, s in r[key].items()),
                key=lambda kv: (tier_rank(kv[0][0]), kv[0][1]),
            )
        }

    def thinking_table(key: str) -> str:
        return table(
            ["Model", "Thinking", *STAT_HEADERS],
            [[tier_name(m), flag, *stat_cells(s)] for (m, flag), s in split_flags(key).items()],
        )

    eval_flags = split_flags("by_model_thinking_eval_target")

    def off_vs_on(key: str) -> str:
        """Per model, one percentile of the thinking-off group against the same percentile of the on group."""
        return ", ".join(
            f"{tier_name(m)} {seconds(eval_flags[m, 'off']['latency_ms'][key])} s vs "
            f"{seconds(eval_flags[m, 'on']['latency_ms'][key])} s"
            for m in models
            if (m, "off") in eval_flags and (m, "on") in eval_flags
        )

    n_on = sum(s["calls"] for (m, flag), s in split_flags("by_model_thinking").items() if flag == "on")
    n_on_eval = sum(s["calls"] for (m, flag), s in eval_flags.items() if flag == "on")
    patcher_p95 = ", ".join(
        f"{tier_name(m)} {seconds(r['by_model_role'][m]['patcher']['latency_ms']['p95'])} s"
        for m in models
        if "patcher" in r["by_model_role"][m]
    )
    on_off = {True: "on", False: "off"}
    length = r["length_calls"]
    length_table = table(
        [
            "Model",
            "Role",
            "Source",
            "Thinking",
            "max_tokens",
            "Output tokens",
            "tool_choice",
            "response_format",
        ],
        [
            [
                tier_name(c["model"]),
                c["role"],
                c["source"],
                on_off.get(c["thinking"], "unset"),
                f"{c['max_tokens']:,}" if c["max_tokens"] is not None else "unset",
                f"{c['output_tokens']:,}",
                str(c["tool_choice"] or "none"),
                c["json_kind"] or "none",
            ]
            for c in length
        ],
    )
    lengths = Counter((tier_name(c["model"]), c["role"]) for c in length)
    length_desc = ", ".join(f"{n} {t} {role}" for (t, role), n in sorted(lengths.items()))
    n_think = sum(1 for c in length if c["thinking"])
    n_required = sum(1 for c in length if c["tool_choice"] == "required")
    recon_table = table(
        [
            "Run",
            "Calls",
            "Tokens in (cache / budget)",
            "Tokens out (cache / budget)",
            "Ultra agent calls (cache / budget)",
            "Budget spend",
            "Match",
        ],
        [
            [
                f"`{x['run_id']}`",
                f"{x['cache_calls']:,}",
                f"{x['cache_tokens_in']:,} / {x['budget_tokens_in']:,}",
                f"{x['cache_tokens_out']:,} / {x['budget_tokens_out']:,}",
                f"{x['cache_ultra_agent_calls']} / {x['budget_ultra_agent_calls']}",
                f"${x['budget_tf_spend_usd']:.2f}",
                "yes" if x["matches_budget"] else "**no**",
            ]
            for x in runs
        ],
    )
    cache_lines = []
    for m in models:
        pc = r["by_model"][m]["prompt_cache"]
        total, hits = pc["prompt_tokens_of_reporting_calls"], pc["hit_tokens"]
        cache_lines.append(
            f"{tier_name(m)} {hits:,} of {total:,} ({hits / total:.1%})" if total else f"{tier_name(m)} n/a"
        )
    invalid = ", ".join(f"{n} with finish_reason `{k}`" for k, n in j["invalid_by_finish_reason"].items())
    failed = "; ".join(f"`{e['run_id']}` at {e['stage']}: {e['reason']}" for e in ev["runs_failed"])
    warns = "; ".join(f"`{w['run_id']}`: {w['message']}" for w in ev["warnings"])
    judge_note = (
        " That failure is Offramp's own `ProviderError`: a judge reply that left a case unscored,"
        " not an HTTP error from Token Factory (`docs/FEEDBACK.md` item 11)."
        if any("judge:" in e["reason"] for e in ev["runs_failed"])
        else ""
    )
    synthetic_repairs = j["repair_retries_of_synthetic_replies"]
    match_word = "match" if all_ok else "do not all match"
    role_check = (
        f"The per-run Ultra agent-call counts below {match_word} `budget.json`, which checks the split "
        "between agent roles and `eval_target` for the Ultra calls."
        if runs
        else ""
    )
    run_errors = (
        f"Runs that ended FAILED: {failed or 'none'}. Warning events: {warns or 'none'}.{judge_note}"
        if runs
        else "No run files were given, so run errors are not listed."
    )
    budget_section = (
        "## Runs against their budgets\n\n"
        "For each run, the cache entries inside its time window, summed, against what `budget.json` charged "
        "(tokens in, tokens out, Ultra agent calls). "
        + ("All runs match." if all_ok else "Not all runs match; see the `no` rows.")
        + " A run that replays cached calls charges only its new ones, so later runs show few calls or none. "
        f"`budget.json` also records spend: ${spend:.2f} across the {n_runs} runs.\n\n{recon_table}\n\n"
        if runs
        else ""
    )
    synthetic, copies = inp.get("synthetic_excluded", 0), inp.get("duplicate_copies_dropped", 0)
    n_rest = len(length) - n_think - n_required
    return reflow(f"""# Token Factory by the numbers

**Scope: {scope}**

Measured from the Token Factory calls Offramp actually made: {o["calls"]:,} distinct recorded chat
completions ({o["input_tokens"]["total"]:,} prompt tokens, {o["output_tokens"]["total"]:,} completion
tokens) on the three tiers ({ids}), recorded between {first} and {last} (UTC). Nothing here is an estimate
or a re-run: every figure is computed by `scripts/tf_stats.py` from three kinds of files, and the raw
numbers are in [`tf-production.json`](tf-production.json).

- **Response cache** (`var/tf-cache`, one JSON per live call, not committed): the call's request, the raw
  response, the client-measured `latency_ms`, token usage and a timestamp.
- **Cassettes** (`tests/fixtures/cassettes`, committed): the same entry shape. {synthetic} are
  hand-authored copies marked `"synthetic": true` (a recorded reply with its tail cut off, to test the
  repair retry); they are excluded. {copies} cassettes were exact copies of another recorded call and are
  counted once.
- **Run files** (`var/runs/*`: `events.jsonl`, `budget.json`): no per-call data. They are used to
  attribute cache entries to pipeline runs by time window, to check the token totals, and to list errors.

A "call" is one response that Token Factory returned and Offramp recorded. A request replayed from the
cache is not counted again. Latency is the client's wall-clock time around the OpenAI SDK call
(non-streaming), so it includes the SDK's automatic retries, the one retry on the fallback key, and any
queueing from Offramp's parallel eval threads. Percentiles are nearest-rank: each is an actual recorded
value. † marks a group of fewer than 20 calls, where p95 is close to the maximum and one call moves the
`length` share by 5 points or more. A call's role is not recorded; it is read from the system prompt (the
table in the script), and `eval_target` is the migrated app's own call on the candidate tier.
{role_check}

## Reproduce

```
uv run python scripts/tf_stats.py --cache var/tf-cache --runs var/runs --format md \\
  > docs/evidence/tf-production.md
uv run python scripts/tf_stats.py --cache var/tf-cache --runs var/runs \\
  > docs/evidence/tf-production.json
```

`var/` is not committed, so a clean clone has no cache or run files and reproduces only the cassette part
({n_cassette:,} calls; its scope line says so):

```
uv run python scripts/tf_stats.py --format md
```

## Where the calls come from

{sources}

## By model

All recorded calls:

{model_table("by_model")}

{more_models}

## By model and role

{role_table}

## Thinking on and off

Latency by the request's `enable_thinking` flag, over all calls. Thinking is on for {n_on} calls, {n_on_eval}
of them `eval_target` calls, while the thinking-off group also holds the agent roles that write long
outputs (the patcher's p95 is {patcher_p95}). The two groups are therefore confounded by role, and p95 of
one is not comparable with p50 of the other.

{thinking_table("by_model_thinking")}

The same split over `eval_target` calls only. The files do not say why thinking was on for a given call,
so this is still not a controlled comparison.

{thinking_table("by_model_thinking_eval_target")}

## The `length` finishes

{len(length)} of {o["calls"]:,} calls ({o["length_share"]:.1%}) ended on `finish_reason: length`:
{length_desc}. {n_think} of them had thinking on (the reasoning-budget case in `docs/FEEDBACK.md` item 2);
{n_required} used `tool_choice="required"` (item 4, Lightning looping on tool calls until the cap); the
other {n_rest} ran to the cap with thinking off. The files record no cause. Where the reply had to be
JSON, a `length` finish leaves it truncated and invalid (next section).

{length_table}

## Validity, retries, errors, caching

- **JSON validity.** {j["valid"]} of {j["calls_with_response_format"]} calls that asked for `json_schema`
  or `json_object` returned JSON that parses and, for `json_schema`, satisfies the schema in the request
  (checked here with `jsonschema`; a `json_object` call is checked only for parsing as an object, because
  its schema is not in the request). The invalid replies: {invalid or "none"}.
- **Repair retries.** {j["repair_retry_calls"]} calls were repair retries ("Your previous reply is not
  valid"): {synthetic_repairs} answered a synthetic reply (the excluded hand-cut ones, so not provider
  failures) and {j["repair_retry_calls"] - synthetic_repairs} a real reply that failed validation.
- **Run errors.** {run_errors}
- **Not recorded.** HTTP errors, 429s, timeouts, the SDK's automatic retries (up to 3 per call) and the
  one retry on the fallback key: the client caches only successful responses, so a failed request leaves
  no file. The provider error rate is unknown here, not zero.
- **Provider prompt-cache hits** (`prompt_cache_hit_tokens` in the response usage):
  {"; ".join(cache_lines)}.

{budget_section}## What this does and doesn't show

It shows that the three Nemotron tiers on Token Factory returned {o["calls"]:,} responses to Offramp's own
calls between {first} and {last}: {fin.get("stop", 0):,} `stop` finishes, {fin.get("tool_calls", 0):,}
`tool_calls` (the normal end of a tool-use call) and {fin.get("length", 0)} `length`, and {j["valid"]} of
{j["calls_with_response_format"]} structured-output replies were valid JSON. On `eval_target` calls alone,
p50 latency with thinking off against on is {off_vs_on("p50")}, and p95 is {off_vs_on("p95")}; those are
different calls, not a controlled experiment. It does not show uptime or an error rate (failed requests are
not recorded), throughput or behaviour under sustained load (the calls came from Offramp's pipeline and
tests in bursts, with at most 8 eval calls in flight), time to first token (nothing was streamed), or cost
per call (`budget.json` has run totals only). All of it comes from one demo app's call sites, the
cassettes are committed test and dev recordings rather than customer traffic, the roles are inferred from
system prompts, and latency was measured on the client side only.""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, help="response cache directory (var/tf-cache)")
    ap.add_argument("--cassettes", type=Path, default=ROOT / "tests" / "fixtures" / "cassettes")
    ap.add_argument("--runs", type=Path, help="runs directory (var/runs)")
    ap.add_argument("--format", choices=("json", "md"), default="json")
    args = ap.parse_args()
    sources = [("cache", args.cache)] if args.cache else []
    sources += [("cassette", args.cassettes)] if args.cassettes.is_dir() else []
    calls, meta = load_calls(sources)
    if not calls:
        sys.exit("no recorded calls found")
    runs = load_runs(args.runs) if args.runs else []
    assign_scopes(calls, runs)
    result = aggregate(calls, runs, meta)
    print(render_md(result) if args.format == "md" else json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
