"""Content-addressed response cache = cassette (Global Constraint 7). One JSON file per response, no TTL."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _normalise(obj: Any) -> Any:
    """Whole-number floats become ints, so temperature 0 and 0.0 (or max_tokens 64.0) share one key."""
    if isinstance(obj, dict):
        return {k: _normalise(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_normalise(v) for v in obj]
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(_normalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
    """sha256 over the canonical JSON of the parsed request: independent of dict key order and whitespace."""
    request = {"model": model, "messages": messages, "params": params}
    return hashlib.sha256(canonical_json(request).encode()).hexdigest()


class Cache:
    """Entry shape (written by TFClient; hand-authored cassettes follow it and set "synthetic": true):
    {"request": {model, messages, params}, "response": <raw chat.completion JSON>, "latency_ms": int,
     "usage": {"prompt_tokens": int, "completion_tokens": int}, "created_at": iso8601, "synthetic"?: true}
    """

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            return json.loads(self._path(key).read_text())
        except FileNotFoundError:
            return None

    def put(self, key: str, entry: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")  # write-then-rename: no partial reads
        with os.fdopen(fd, "w") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._path(key))
