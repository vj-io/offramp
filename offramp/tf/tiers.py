"""Tier -> Token Factory model mapping (Global Constraint 4). Prices come from pricing.yaml."""

import os
from dataclasses import dataclass
from typing import Any

from offramp.pricing import load_pricing
from offramp.schemas import Tier

# Ids confirmed against GET /v1/models (docs/plan/tf-probe.md, 2026-10-19).
# Override per tier with OFFRAMP_MODEL_{LIGHTNING,SUPER,ULTRA}. Each default must be a key in pricing.yaml.
DEFAULT_MODEL_IDS: dict[Tier, str] = {
    Tier.LIGHTNING: "nvidia/Nemotron-3_5-Lightning",
    Tier.SUPER: "nvidia/nemotron-3-super-120b-a12b",
    Tier.ULTRA: "nvidia/Nemotron-3-Ultra-550b-a55b",
}

# Verified by the live probe (docs/plan/tf-probe.md): json_schema and json_object work on all three tiers.
# Set a tier to False and the client uses json_object + validation + one repair retry for it.
SUPPORTS_JSON_SCHEMA: dict[Tier, bool] = {Tier.LIGHTNING: True, Tier.SUPER: True, Tier.ULTRA: True}


@dataclass(frozen=True)
class TierInfo:
    tier: Tier
    model_id: str
    input_usd_per_m: float
    output_usd_per_m: float
    supports_json_schema: bool
    reasoning_style: str


def get_tier(tier: Tier) -> TierInfo:
    tier = Tier(tier)
    priced = next(e for e in load_pricing().values() if e.get("tier") == tier.value)
    return TierInfo(
        tier=tier,
        model_id=os.environ.get(f"OFFRAMP_MODEL_{tier.name}") or DEFAULT_MODEL_IDS[tier],
        input_usd_per_m=priced["input_usd_per_m"],
        output_usd_per_m=priced["output_usd_per_m"],
        supports_json_schema=SUPPORTS_JSON_SCHEMA[tier],
        reasoning_style="chat_template_kwargs",
    )


def reasoning_params(tier: Tier, level: str) -> dict[str, Any]:
    """Map AdapterSpec.reasoning ("off"|"low"|"medium"|"high") to chat.completions.create kwargs.

    The single place the reasoning-control mechanism lives. Nemotron thinks by default; the live probe
    (docs/plan/tf-probe.md) showed chat_template_kwargs.enable_thinking=false turns it off, while
    reasoning_effort="low" did not shorten Ultra output, so the levels low/medium/high all mean "on".
    """
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": level != "off"}}}
