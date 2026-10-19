from offramp.schemas import STAGE_ORDER, TIER_ORDER, RunConfig, Stage, Tier


def test_tier_order_cheapest_first():
    assert TIER_ORDER == [Tier.LIGHTNING, Tier.SUPER, Tier.ULTRA]


def test_stage_order_excludes_failed():
    assert STAGE_ORDER[0] is Stage.QUEUED and STAGE_ORDER[-1] is Stage.DONE
    assert Stage.FAILED not in STAGE_ORDER


def test_run_config_defaults():
    cfg = RunConfig(repo_url="https://github.com/x/y")
    assert (cfg.threshold, cfg.reasoning_threshold, cfg.max_iterations) == (0.95, 0.90, 3)
