"""Token Factory's own prices for the Nemotron tiers, read from its model API (GET /v1/models?verbose=true:
`pricing.prompt` / `pricing.completion` per model, in USD per token).

Read once per UTC day and cached on disk like the Tavily lookup; a cache-only run never calls the API and
takes the newest cached read. pricing.yaml is the fallback, labelled with why. The run's own cost figures
(budget, $/1k calls) use pricing.yaml's tier prices, which matched the API on 2026-10-21; an API price that
differs is shown with pricing.yaml's next to it.
"""

import json
import logging
import math
import os
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from offramp.integrations import keys
from offramp.pricing import load_pricing
from offramp.schemas import AfterPrice, Tier
from offramp.tf.client import DEFAULT_BASE_URL
from offramp.tf.tiers import get_tier

log = logging.getLogger(__name__)

SOURCE = "Token Factory model API"
TIMEOUT_S = 30.0


def _dir(cache_dir: str | Path | None) -> Path:
    return Path(cache_dir or Path(os.environ.get("OFFRAMP_DATA_DIR", "./var")) / "tf-prices")


def _per_m(per_token: object) -> float | None:
    """USD per token -> USD per 1M tokens; None unless finite and 0 < x < 1000 (a NaN, infinite, negative or
    absurd price is never shown, and NaN would not survive the run's JSON event log)."""
    x = round(float(per_token) * 1e6, 6)
    return x if math.isfinite(x) and 0 < x < 1000 else None


def _fetch(http: httpx.Client) -> dict[str, list[float] | None]:
    """model id -> [input, output] USD per 1M tokens for every model the API prices; None for a model whose
    price is not plausible."""
    key = keys.current("nebius", os.environ.get("NEBIUS_API_KEY"))
    if not key:
        raise RuntimeError("NEBIUS_API_KEY is not set")
    base = (os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    resp = http.get(f"{base}/models", params={"verbose": "true"}, headers={"Authorization": f"Bearer {key}"})
    resp.raise_for_status()
    per_m = {}
    for m in resp.json()["data"]:
        p = m.get("pricing") or {}
        if p.get("prompt") is not None and p.get("completion") is not None:
            pair = [_per_m(p["prompt"]), _per_m(p["completion"])]
            per_m[m["id"]] = None if None in pair else pair
    return per_m


def _why(e: Exception) -> str:
    """A short fixed reason (it reaches the PR): never the response body."""
    if isinstance(e, httpx.HTTPStatusError):
        return f"HTTP {e.response.status_code}"
    if isinstance(e, httpx.TransportError):
        return type(e).__name__
    if isinstance(e, RuntimeError):
        return str(e)
    return "unexpected response"


def _read(path: Path) -> dict[str, list[float]] | None:
    try:
        return json.loads(path.read_text())["prices"]
    except (OSError, ValueError, KeyError, TypeError):  # missing, partial or corrupt: a miss
        return None


def _write(path: Path, prices: dict[str, list[float]], now: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")  # write-then-rename: no partial reads
    with os.fdopen(fd, "w") as f:
        json.dump({"fetched_at": now.isoformat(), "prices": prices}, f, indent=2)
    os.replace(tmp, path)


def from_yaml(tier: Tier, why: str) -> AfterPrice:
    """The tier's pricing.yaml price, labelled with its as_of date and why it is used."""
    info = get_tier(tier)
    as_of = next(e["as_of"] for e in load_pricing().values() if e.get("tier") == Tier(tier).value)
    return AfterPrice(
        model=info.model_id,
        input_usd_per_m=info.input_usd_per_m,
        output_usd_per_m=info.output_usd_per_m,
        via="pricing.yaml",
        source=f"pricing.yaml as of {as_of}; {why}",
    )


def _price(
    tier: Tier, listed: dict[str, list[float] | None] | None, label: str, why: str | None
) -> AfterPrice:
    info = get_tier(tier)  # the model id (an OFFRAMP_MODEL_* override included) and pricing.yaml's price
    if not listed or info.model_id not in listed:
        return from_yaml(tier, why or f"{info.model_id} is not priced by the {SOURCE}")
    if listed[info.model_id] is None:
        return from_yaml(tier, f"the {SOURCE} lists no plausible price for {info.model_id}")
    api, yaml = tuple(listed[info.model_id]), (info.input_usd_per_m, info.output_usd_per_m)
    if api != yaml:
        label += f"; this run's cost figures use pricing.yaml's ${yaml[0]:g} / ${yaml[1]:g}"
    return AfterPrice(
        model=info.model_id,
        input_usd_per_m=api[0],
        output_usd_per_m=api[1],
        via="token_factory_api",
        source=label,
    )


def after_prices(
    tiers: Iterable[Tier],
    *,
    live: bool,
    http: httpx.Client | None = None,
    cache_dir: str | Path | None = None,
    now: datetime | None = None,
) -> dict[Tier, AfterPrice]:
    """Each tier's Token Factory price: today's model API read (one call per UTC day, cached), or with
    `live` False the newest cached read and no call at all; pricing.yaml's, labelled with why, otherwise."""
    now = now or datetime.now(UTC)
    base = _dir(cache_dir)
    listed, why = None, None
    if live:
        path, label = base / f"{now:%Y-%m-%d}.json", f"{SOURCE}, {now:%Y-%m-%d}"
        listed = _read(path)
        if listed is None:
            own = http is None
            http = http or httpx.Client(timeout=TIMEOUT_S)
            try:
                listed = _fetch(http)
            except (httpx.HTTPError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as e:
                why = f"the {SOURCE} failed ({_why(e)})"
                log.warning("token factory prices: %s", why)
            finally:
                if own:
                    http.close()
            try:
                if listed is not None:
                    _write(path, listed, now)
            except OSError as e:  # an unwritable data dir costs a read next time, never the run
                log.warning("token factory prices: cache not written: %s", type(e).__name__)
    else:  # the newest cached read that is readable
        cached = ((p, _read(p)) for p in sorted(base.glob("????-??-??.json"), reverse=True))
        day, listed = next(((p.stem, prices) for p, prices in cached if prices is not None), (None, None))
        label = f"{SOURCE}, {day} (cached; cache-only run)"
        why = None if listed is not None else f"cache-only run with no cached {SOURCE} read"
    return {Tier(t): _price(Tier(t), listed, label, why) for t in tiers}
