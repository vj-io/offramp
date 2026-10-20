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


def entry(content, *, system=APP, latency=100, tin=10, tout=1, finish="stop", fmt=None, rid="c-1",
          created="2026-10-19T10:00:00+00:00", last=None, **extra):
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
    assert "# Token Factory by the numbers" in stats.render_md(result)


def test_load_calls_skips_synthetic_and_non_entries_and_counts_a_copy_once(tmp_path):
    stats = load_script()
    cache, cassettes = tmp_path / "cache", tmp_path / "cassettes" / "sub"
    cache.mkdir(), cassettes.mkdir(parents=True)
    real = entry("hello", rid="c-real")
    cut = entry('{"a": ', rid="c-cut", created="2026-10-19T11:00:00+00:00", synthetic=True)
    repair = entry(
        '{"a": 1}', rid="c-fix", created="2026-10-19T12:00:00+00:00", fmt=JSON_SCHEMA,
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
