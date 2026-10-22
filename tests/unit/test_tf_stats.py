"""scripts/tf_stats.py: the aggregation over a tiny hand-built cache (numbers worked out by hand)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
JSON_SCHEMA = {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}}
JUDGE = "You grade answers written by an AI assistant. Be strict."
APP = "You are HelpHub's support assistant."


def load_script():
    spec = importlib.util.spec_from_file_location("tf_stats", ROOT / "scripts" / "tf_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry(
    content,
    *,
    system=APP,
    latency=100,
    tin=10,
    tout=1,
    finish="stop",
    fmt=None,
    rid="c-1",
    created="2026-10-19T10:00:00+00:00",
    last=None,
    **extra,
):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "hi"}]
    if last:
        messages += [{"role": "assistant", "content": last[0]}, {"role": "user", "content": last[1]}]
    params = {"max_tokens": 50, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if fmt:
        params["response_format"] = fmt
    usage = {"prompt_tokens": tin, "completion_tokens": tout}
    return {
        "request": {"model": MODEL, "messages": messages, "params": params},
        "response": {
            "id": rid,
            "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}],
            "usage": {**usage, "prompt_cache_hit_tokens": 4},
        },
        "latency_ms": latency,
        "usage": usage,
        "created_at": created,
        **extra,
    }


def test_summarise_counts_tokens_percentiles_finish_and_json_validity():
    stats = load_script()
    calls = [
        stats.to_call(entry('{"a": 1}', latency=100, tin=10, tout=1, fmt=JSON_SCHEMA, system=JUDGE), "cache"),
        stats.to_call(entry('{"a": 2}', latency=200, tin=20, tout=2, fmt=JSON_SCHEMA, system=JUDGE), "cache"),
        stats.to_call(
            entry('{"a": ', latency=1000, tin=30, tout=50, finish="length", fmt=JSON_SCHEMA), "cache"
        ),
        stats.to_call(entry("hello", latency=300, tin=40, tout=4), "cassette"),
    ]
    assert [c["role"] for c in calls] == ["judge", "judge", "eval_target", "eval_target"]

    s = stats.summarise(calls)

    assert s["calls"] == 4
    assert s["input_tokens"] == {"total": 100, "p50": 20}  # nearest rank: ceil(0.5 * 4) = 2nd of 4
    assert s["output_tokens"] == {"total": 57, "p50": 2}
    assert s["latency_ms"] == {"n": 4, "p50": 200, "p95": 1000, "max": 1000}  # ceil(0.95 * 4) = 4th
    assert s["finish_reasons"] == {"length": 1, "stop": 3}
    assert s["length_share"] == 0.25
    assert s["prompt_cache"] == {"hit_tokens": 16, "prompt_tokens_of_reporting_calls": 100}
    assert s["json"]["calls_with_response_format"] == 3
    assert s["json"]["valid"] == 2  # the truncated reply (finish length) fails the schema
    assert s["json"]["invalid_by_finish_reason"] == {"length": 1}

    stats.assign_scopes(calls, [])
    result = stats.aggregate(calls, [], {})
    assert result["by_model_role"][MODEL]["judge"]["calls"] == 2
    assert result["length_calls"][0]["output_tokens"] == 50
    md = stats.render_md(result)
    assert "# Token Factory by the numbers" in md
    assert "Scope: all 4 recorded calls" in md
    assert "uv run python scripts/tf_stats.py --cache var/tf-cache --runs var/runs --format md" in md
    cassettes = [c for c in calls if c["source"] == "cassette"]  # a clean clone has no cache
    stats.assign_scopes(cassettes, [])
    assert "committed cassettes only" in stats.render_md(stats.aggregate(cassettes, [], {}))


def test_load_calls_skips_synthetic_and_non_entries_and_counts_a_copy_once(tmp_path):
    stats = load_script()
    cache, cassettes = tmp_path / "cache", tmp_path / "cassettes" / "sub"
    cache.mkdir(), cassettes.mkdir(parents=True)
    real = entry("hello", rid="c-real")
    cut = entry('{"a": ', rid="c-cut", created="2026-10-19T11:00:00+00:00", synthetic=True)
    repair = entry(
        '{"a": 1}',
        rid="c-fix",
        created="2026-10-19T12:00:00+00:00",
        fmt=JSON_SCHEMA,
        last=('{"a": ', stats.REPAIR_ASK + " Validation error: eof"),
    )
    (cache / "real.json").write_text(json.dumps(real))
    (cassettes / "copy.json").write_text(json.dumps(real))  # same response id and created_at: one call
    (cassettes / "cut.json").write_text(json.dumps(cut))
    (cassettes / "fix.json").write_text(json.dumps(repair))
    (cassettes / "scan.json").write_text(json.dumps({"sites": []}))  # a fixture, not a cassette entry

    calls, meta = stats.load_calls([("cache", cache), ("cassette", tmp_path / "cassettes")])

    assert [c["source"] for c in calls] == ["cache", "cassette"]
    assert meta == {"synthetic_excluded": 1, "duplicate_copies_dropped": 1}
    assert calls[1]["repair_retry"] and calls[1]["repair_of_synthetic"]
    assert stats.summarise(calls)["json"]["repair_retries_of_synthetic_replies"] == 1


def test_role_of_maps_every_prefix_and_each_prefix_is_still_in_the_source():
    stats = load_script()
    retired = {"You migrate one LLM call site"}  # the patcher's first prompt, kept for old recordings
    source = "".join(p.read_text() for p in (ROOT / "offramp").rglob("*.py"))
    for prefix, role in stats.ROLE_PREFIXES:
        assert stats.role_of([{"role": "system", "content": prefix + " and more"}])[0] == role
        assert (prefix in source) != (prefix in retired), prefix  # a reworded agent prompt fails here
    assert stats.role_of([{"role": "system", "content": APP}]) == ("eval_target", APP)
    assert stats.role_of([{"role": "user", "content": "no system prompt"}])[0] == "eval_target"


def test_assign_scopes_puts_only_cache_calls_inside_a_run_window_into_the_run():
    stats = load_script()
    runs = [{"run_id": "r1", "start": "2026-10-19T10:00:00.000000Z", "end": "2026-10-19T10:10:00.000000Z"}]
    calls = [
        stats.to_call(entry("x", created="2026-10-19T10:00:00+00:00"), "cache"),  # start bound, inclusive
        stats.to_call(entry("x", created="2026-10-19T10:10:00+00:00"), "cache"),  # end bound, inclusive
        stats.to_call(entry("x", created="2026-10-19T10:10:01+00:00"), "cache"),  # a second too late
        stats.to_call(entry("x", created="2026-10-19T10:05:00+00:00"), "cassette"),  # inside, but a cassette
    ]

    stats.assign_scopes(calls, runs)

    assert [c["scope"] for c in calls] == ["r1", "r1", "cache_outside_runs", "cassette"]


def test_load_runs_skips_non_runs_and_reconcile_compares_the_window_with_the_budget(tmp_path):
    stats = load_script()

    def event(ts, kind, payload):
        return {"ts": ts, "type": kind, "payload": payload}

    run = tmp_path / "r1"
    run.mkdir()
    events = [
        event("2026-10-19T10:00:00.000000Z", "stage", {}),
        event("2026-10-19T10:03:00.000000Z", "log", {"level": "warn", "message": "slow judge"}),
        event(
            "2026-10-19T10:04:00.000000Z", "error", {"stage": "ITERATING", "reason": "ProviderError: judge"}
        ),
        event("2026-10-19T10:10:00.000000Z", "stage", {}),
    ]
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (run / "budget.json").write_text(json.dumps({"tokens_in": 30, "tokens_out": 3, "ultra_agent_calls": 1}))
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "events.jsonl").write_text("")  # a run with no events has no window
    (tmp_path / "notes.txt").write_text("not a run")

    runs = stats.load_runs(tmp_path)

    assert [r["run_id"] for r in runs] == ["r1"]
    assert runs[0]["warnings"] == ["slow judge"]
    assert runs[0]["errors"] == [{"stage": "ITERATING", "reason": "ProviderError: judge"}]
    calls = [  # an Ultra agent call and an Ultra eval_target call, both inside the window
        stats.to_call(
            entry("a", system=JUDGE, tin=10, tout=1, rid="c-1", created="2026-10-19T10:01:00+00:00"), "cache"
        ),
        stats.to_call(
            entry("b", system=APP, tin=20, tout=2, rid="c-2", created="2026-10-19T10:02:00+00:00"), "cache"
        ),
    ]
    stats.assign_scopes(calls, runs)

    row = stats.reconcile(calls, runs)[0]

    assert (row["cache_calls"], row["cache_tokens_in"], row["cache_tokens_out"]) == (2, 30, 3)
    assert row["cache_ultra_agent_calls"] == 1  # eval_target is not an agent call
    assert row["matches_budget"]
    runs[0]["budget"]["ultra_agent_calls"] = 2
    assert not stats.reconcile(calls, runs)[0]["matches_budget"]


def test_a_judge_request_with_twice_the_room_is_counted_as_a_retry_after_a_deleted_reply():
    """Task 31 D (review M-4): the judge sends a batch again with twice `max_tokens` only after the first
    reply did not parse, and deleted that first reply from the cache, so the counts never saw it."""
    stats = load_script()
    batch = "Case 1\n<evidence>\nq\n</evidence>\n\nCase 2\n<evidence>\nq\n</evidence>"  # two cases

    def judge(max_tokens, system=JUDGE):
        e = entry('{"a": 1}', system=system, fmt=JSON_SCHEMA)
        e["request"]["messages"][-1]["content"] = batch
        e["request"]["params"]["max_tokens"] = max_tokens
        return stats.to_call(e, "cache")

    assert [judge(2 * (400 + 150 * 2))["room_retry"], judge(400 + 150 * 2)["room_retry"]] == [True, False]
    assert judge(2 * (400 + 150 * 2), system=APP)["room_retry"] is False  # not a judge call
