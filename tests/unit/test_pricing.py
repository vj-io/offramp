import pytest

from offramp.pricing import cost_per_1k_calls, cost_usd, load_pricing, price
from offramp.schemas import TIER_ORDER, Tier
from offramp.tf.tiers import DEFAULT_MODEL_IDS, get_tier, reasoning_params


def test_token_factory_prices_from_the_prd():
    assert price("nvidia/Nemotron-3-Ultra-550b-a55b") == (1.00, 3.00)
    assert price("nvidia/nemotron-3-super-120b-a12b") == (0.30, 0.90)
    assert price("nvidia/Nemotron-3_5-Lightning") == (0.06, 0.24)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-4o-mini", (0.15, 0.60)),
        ("gpt-4o", (2.50, 10.00)),
        ("gpt-4.1-mini", (0.40, 1.60)),
        ("o3", (2.00, 8.00)),
        ("o4-mini", (1.10, 4.40)),
        ("claude-sonnet-4-5", (3.00, 15.00)),
        ("claude-haiku-4-5", (1.00, 5.00)),
    ],
)
def test_closed_model_prices(model, expected):
    assert price(model) == expected


def test_every_entry_is_timestamped_and_sourced():
    table = load_pricing()
    assert len(table) == 10
    for model, e in table.items():
        assert e["as_of"] == "2026-10-19", model
        assert e["source"].startswith("https://"), model
        assert e["input_usd_per_m"] > 0 and e["output_usd_per_m"] > 0, model


def test_cost_usd_is_per_million_arithmetic():
    # 2000 in * $0.15/M + 500 out * $0.60/M
    assert cost_usd("gpt-4o-mini", 2000, 500) == pytest.approx(0.0006)
    assert cost_usd("gpt-4o-mini", 0, 0) == 0.0
    assert cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_cost_per_1k_calls_scales_by_a_thousand():
    # claude-sonnet-4-5: 800 in * $3/M + 200 out * $15/M = $0.0054 per call -> $5.40 per 1k calls
    assert cost_per_1k_calls("claude-sonnet-4-5", 800, 200) == pytest.approx(5.40)
    # nemotron super, same shape: 800*0.30 + 200*0.90 = 420 / 1e6 per call -> $0.42 per 1k calls
    assert cost_per_1k_calls("nvidia/nemotron-3-super-120b-a12b", 800, 200) == pytest.approx(0.42)
    assert cost_per_1k_calls("gpt-4o", 123.5, 45.25) == pytest.approx(1000 * cost_usd("gpt-4o", 123.5, 45.25))


@pytest.mark.parametrize(
    "alias,base",
    [
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
        ("openai/gpt-4o", "gpt-4o"),
        ("anthropic/claude-haiku-4-5-20251001", "claude-haiku-4-5"),
    ],
)
def test_dated_snapshots_and_provider_prefixes_resolve_to_the_base_model(alias, base):
    assert price(alias) == price(base)


@pytest.mark.parametrize("model", ["o3-mini", "gpt-4o-nano", "mystery-model", ""])
def test_unknown_models_raise_instead_of_guessing(model):
    with pytest.raises(KeyError, match="pricing.yaml"):
        price(model)


# ----------------------------------------------------------------------------- tiers


def test_tier_info_prices_and_ids():
    ultra = get_tier(Tier.ULTRA)
    assert (ultra.tier, ultra.model_id) == (Tier.ULTRA, "nvidia/Nemotron-3-Ultra-550b-a55b")
    assert (ultra.input_usd_per_m, ultra.output_usd_per_m) == (1.00, 3.00)
    sup, lightning = get_tier(Tier.SUPER), get_tier(Tier.LIGHTNING)
    assert (sup.input_usd_per_m, sup.output_usd_per_m) == (0.30, 0.90)
    assert (lightning.input_usd_per_m, lightning.output_usd_per_m) == (0.06, 0.24)


def test_tiers_are_ordered_cheapest_first_by_price():
    outputs = [get_tier(t).output_usd_per_m for t in TIER_ORDER]
    assert outputs == sorted(outputs)


def test_default_model_ids_are_priced_under_the_matching_tier():
    table = load_pricing()
    for tier, model_id in DEFAULT_MODEL_IDS.items():
        assert table[model_id]["tier"] == tier.value
        assert get_tier(tier).model_id == model_id


@pytest.mark.parametrize("tier", list(Tier))
def test_env_override_changes_the_model_id_but_keeps_the_tier_price(tier, monkeypatch):
    before = get_tier(tier)
    monkeypatch.setenv(f"OFFRAMP_MODEL_{tier.name}", "nvidia/some-other-id")
    after = get_tier(tier)
    assert after.model_id == "nvidia/some-other-id"
    assert (after.input_usd_per_m, after.output_usd_per_m) == (
        before.input_usd_per_m,
        before.output_usd_per_m,
    )


def test_empty_env_override_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("OFFRAMP_MODEL_SUPER", "")
    assert get_tier(Tier.SUPER).model_id == DEFAULT_MODEL_IDS[Tier.SUPER]


def test_reasoning_params_maps_levels_onto_enable_thinking():
    off = reasoning_params(Tier.SUPER, "off")
    assert off == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    for level in ("low", "medium", "high"):
        kwargs = reasoning_params(Tier.ULTRA, level)["extra_body"]["chat_template_kwargs"]
        assert kwargs["enable_thinking"] is True
