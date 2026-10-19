import threading

import pytest

from offramp.schemas import Tier
from offramp.tf.budget import Budget, BudgetExceeded


def test_defaults_match_the_prd():
    b = Budget()
    assert (b.max_in, b.max_out, b.max_ultra_agent_calls) == (1_500_000, 300_000, 40)


def test_charge_accumulates_and_snapshot_reports_it():
    b = Budget()
    b.charge("scanner", Tier.SUPER, 100, 20, 0.5)
    b.charge("eval", Tier.LIGHTNING, 50, 10, 0.25)
    snap = b.snapshot()
    assert snap == {
        "tokens_in": 150,
        "tokens_out": 30,
        "ultra_agent_calls": 0,
        "tf_spend_usd": 0.75,
        "max_in": 1_500_000,
        "max_out": 300_000,
        "max_ultra_agent_calls": 40,
    }


def test_ultra_agent_cap_is_40_calls():
    b = Budget()
    for _ in range(40):
        b.check("adapter", Tier.ULTRA)
        b.charge("adapter", Tier.ULTRA, 1, 1)
    assert b.snapshot()["ultra_agent_calls"] == 40
    with pytest.raises(BudgetExceeded, match="Ultra agent call cap"):
        b.check("judge", Tier.ULTRA)


def test_ultra_cap_applies_to_every_agent_role_but_not_eval_targets():
    b = Budget(max_ultra_agent_calls=2)
    for role in ("adapter", "iterator"):
        b.charge(role, Tier.ULTRA, 1, 1)
    for _ in range(5):  # eval-target calls on Ultra: token caps only
        b.check("eval_target", Tier.ULTRA)
        b.charge("eval_target", Tier.ULTRA, 1, 1)
    assert b.snapshot()["ultra_agent_calls"] == 2
    b.check("scanner", Tier.SUPER)  # other tiers are never subject to the Ultra cap
    with pytest.raises(BudgetExceeded):
        b.check("case_gen", Tier.ULTRA)  # adversarial case gen runs on Ultra and is an agent call


def test_charge_past_the_ultra_cap_raises_but_records_the_real_spend():
    b = Budget(max_ultra_agent_calls=1)
    b.charge("judge", Tier.ULTRA, 5, 5)
    with pytest.raises(BudgetExceeded):
        b.charge("judge", Tier.ULTRA, 5, 5)
    assert b.snapshot()["tokens_in"] == 10


@pytest.mark.parametrize("tin,tout", [(11, 0), (0, 6)])
def test_token_caps_raise_on_the_charge_that_crosses_them(tin, tout):
    b = Budget(max_in=10, max_out=5)
    with pytest.raises(BudgetExceeded, match="token cap exceeded"):
        b.charge("scanner", Tier.SUPER, tin, tout)
    with pytest.raises(BudgetExceeded):  # and the budget stays closed afterwards
        b.check("scanner", Tier.SUPER)


def test_exactly_at_the_cap_is_allowed_to_charge_but_not_to_call_again():
    b = Budget(max_in=10, max_out=5)
    b.charge("scanner", Tier.SUPER, 10, 5)  # at the cap, not over it
    with pytest.raises(BudgetExceeded):
        b.check("scanner", Tier.SUPER)


def test_thread_safe_accounting():
    b = Budget(max_in=10**9, max_out=10**9, max_ultra_agent_calls=10**9)

    def work():
        for _ in range(2000):
            b.charge("adapter", Tier.ULTRA, 3, 2, 0.001)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    snap = b.snapshot()
    assert (snap["tokens_in"], snap["tokens_out"], snap["ultra_agent_calls"]) == (48_000, 32_000, 16_000)
    assert snap["tf_spend_usd"] == pytest.approx(16.0)
