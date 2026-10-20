"""USD pricing from the timestamped table in offramp/evals/pricing.yaml (PRD §6.5)."""

import functools
import re
from pathlib import Path
from typing import Any

import yaml

PRICING_PATH = Path(__file__).parent / "evals" / "pricing.yaml"
_SNAPSHOT_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")  # gpt-4o-mini-2024-07-18, ...-20250929


@functools.cache
def load_pricing() -> dict[str, dict[str, Any]]:
    """model id -> entry {input_usd_per_m, output_usd_per_m, as_of, source, [tier], [note]}."""
    return yaml.safe_load(PRICING_PATH.read_text())["models"]


def entry(model: str) -> dict[str, Any]:
    table = load_pricing()
    if model in table:
        return table[model]
    # Dated snapshots and provider prefixes resolve to the base model; "o3-mini" must NOT resolve to "o3".
    base = _SNAPSHOT_SUFFIX.sub("", model.rsplit("/", 1)[-1])
    if base not in table:
        raise KeyError(f"no price for model {model!r}; add it to {PRICING_PATH.name}")
    return table[base]


def price(model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens."""
    e = entry(model)
    return e["input_usd_per_m"], e["output_usd_per_m"]


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = price(model)
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


def cost_per_1k_calls(model: str, mean_tokens_in: float, mean_tokens_out: float) -> float:
    return 1000 * cost_usd(model, mean_tokens_in, mean_tokens_out)
