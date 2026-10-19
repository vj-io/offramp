import json
import os
import threading

from offramp.tf.cache import Cache, cache_key

MESSAGES = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "héllo 🌍"}]
PARAMS = {"max_tokens": 64, "temperature": 0.0, "response_format": {"type": "json_object"}}


def test_key_is_a_sha256_hex_digest():
    key = cache_key("m", MESSAGES, PARAMS)
    assert len(key) == 64 and int(key, 16) >= 0


def test_key_stable_across_dict_key_order():
    reordered_msgs = [dict(reversed(list(m.items()))) for m in MESSAGES]
    reordered_params = dict(reversed(list(PARAMS.items())))
    assert cache_key("m", MESSAGES, PARAMS) == cache_key("m", reordered_msgs, reordered_params)


def test_key_ignores_json_whitespace():
    compact = '{"messages":[{"role":"user","content":"x"}],"params":{"max_tokens":8,"temperature":0}}'
    spaced = """{
        "params": { "temperature": 0,   "max_tokens": 8 },
        "messages": [ { "content": "x", "role": "user" } ]
    }"""
    a, b = json.loads(compact), json.loads(spaced)
    assert cache_key("m", a["messages"], a["params"]) == cache_key("m", b["messages"], b["params"])


def test_key_changes_with_model_messages_or_params():
    base = cache_key("m", MESSAGES, PARAMS)
    assert cache_key("other", MESSAGES, PARAMS) != base
    assert cache_key("m", MESSAGES[:1], PARAMS) != base
    assert cache_key("m", MESSAGES, {**PARAMS, "max_tokens": 65}) != base
    assert cache_key("m", MESSAGES, {**PARAMS, "temperature": 0.1}) != base


def test_key_distinguishes_message_order():
    assert cache_key("m", MESSAGES, PARAMS) != cache_key("m", MESSAGES[::-1], PARAMS)


def test_miss_then_hit_round_trip(tmp_path):
    cache = Cache(tmp_path / "nested" / "cache")  # directory is created on first put
    key = cache_key("m", MESSAGES, PARAMS)
    assert cache.get(key) is None
    entry = {
        "request": {"model": "m", "messages": MESSAGES, "params": PARAMS},
        "response": {"choices": []},
        "latency_ms": 12,
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        "created_at": "2026-10-19T00:00:00+00:00",
        "synthetic": True,
    }
    cache.put(key, entry)
    assert cache.get(key) == entry
    assert (tmp_path / "nested" / "cache" / f"{key}.json").exists()


def test_one_file_per_response_and_no_temp_files_left(tmp_path):
    cache = Cache(tmp_path)
    cache.put("a" * 64, {"x": 1})
    cache.put("b" * 64, {"x": 2})
    cache.put("a" * 64, {"x": 3})  # overwrite
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{'a' * 64}.json", f"{'b' * 64}.json"]
    assert cache.get("a" * 64) == {"x": 3}


def test_no_ttl_old_entries_still_hit(tmp_path):
    cache = Cache(tmp_path)
    cache.put("k", {"created_at": "2000-01-01T00:00:00+00:00"})
    os.utime(tmp_path / "k.json", (0, 0))  # mtime = 1970
    assert cache.get("k") == {"created_at": "2000-01-01T00:00:00+00:00"}


def test_unicode_survives_round_trip(tmp_path):
    cache = Cache(tmp_path)
    cache.put("k", {"text": "héllo 🌍 日本語"})
    assert cache.get("k") == {"text": "héllo 🌍 日本語"}
    assert "日本語" in (tmp_path / "k.json").read_text()  # stored readable, not \u-escaped


def test_concurrent_puts_to_one_key_never_corrupt(tmp_path):
    cache = Cache(tmp_path)
    errors = []

    def work(i):
        try:
            for _ in range(20):
                cache.put("k", {"writer": i, "pad": "x" * 5000})
                assert cache.get("k")["pad"] == "x" * 5000
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
