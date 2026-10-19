"""Known-fail canary (PRD §12.3): a cs_05-like reasoning site that never reaches 90% must stay KEEP_CLOSED.

Everything below the models is real: route_site, run_tier, build_request, the reasoning scorer, the site rule,
the Iterator and the verdict rule. Eval targets answer from FIXTURE refund outputs that are right on a fixed
subset of cases per tier whatever the adapter says (Lightning 50%, Super 70%, Ultra 80%), so a scorer or
routing bug that over-passes turns this test red.
"""

import hashlib
import math
import re
from datetime import UTC, datetime
from fractions import Fraction

from offramp.agents.iterator import ITERATOR_ROLE, Iterator
from offramp.agents.verdict import VERDICT_ROLE, pricing_yaml_price
from offramp.evals.route import route_site
from offramp.schemas import (
    Adapter,
    AdapterSpec,
    BaselineResult,
    CallSite,
    Contract,
    EvalCase,
    ModelOutput,
    PromptTemplate,
    RunConfig,
    Tier,
)
from offramp.tf import ChatResult, FakeLLM
from offramp.tf.tiers import get_tier

NOW = datetime(2026, 10, 19, tzinfo=UTC)
CASE = re.compile(r"Request (cs_05:\d{3})")
RIGHT = {
    Tier.LIGHTNING: {0, 2, 4, 6, 8},
    Tier.SUPER: {0, 1, 2, 3, 4, 5, 6},
    Tier.ULTRA: {0, 1, 2, 3, 4, 5, 6, 7},
}


def gc10(case_id: str) -> str:
    return "holdout" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100 < 30 else "dev"


SITE = CallSite(
    id="cs_05",
    file="helphub/refunds/plan.py",
    lines=[60, 75],
    sdk="openai",
    call="client.chat.completions.create",
    source_model="o3",
    prompt_source="fstring",
    template=PromptTemplate(
        system="You compute subscription refunds. Show your working, then `ANSWER: <number>` last.",
        user="Request {request_id}: {seats} seats at {price} USD, cancelled after {days} of 365 days.",
        variables=["request_id", "seats", "price", "days"],
    ),
    contract=Contract(task_type="reasoning", output="text", reasoning_heavy=True),
)


def refund(n: int) -> str:
    """FIXTURE reference answers (made-up amounts with two decimals)."""
    return f"{(n * 37.17) % 900 + 12.05:.2f}"


CASES, REFS = [], {}
for n in range(1, 41):
    cid = f"cs_05:{n:03d}"
    user = f"Request {cid}: {n % 60 + 3} seats at 99.00 USD, cancelled after {n * 7 % 300 + 15} of 365 days."
    CASES.append(
        EvalCase(
            id=cid,
            site_id="cs_05",
            input_messages=[
                {"role": "system", "content": SITE.template.system},
                {"role": "user", "content": user},
            ],
            expected_kind="number",
            split=gc10(cid),
        )
    )
    REFS[cid] = BaselineResult(
        case_id=cid,
        model="reference:oracle",
        output=ModelOutput(text=f"ANSWER: {refund(n)}"),
        recorded_at=NOW,
    )


def right(tier: Tier, cid: str) -> bool:
    return int(cid.split(":")[1]) % 10 in RIGHT[tier]


class Models:
    def __init__(self, agents: FakeLLM):
        self.agents = agents

    def chat(self, *, role, tier, messages, **kw):
        if role != "eval_target":
            return self.agents.chat(role=role, tier=tier, messages=messages, **kw)
        [cid] = CASE.findall(messages[-1]["content"])
        amount = refund(int(cid.split(":")[1]))
        if not right(Tier(tier), cid):  # a half-even rounding slip on the processing fee
            amount = f"{float(amount) + 0.01:.2f}"
        text = f"Prorated amount, minus the 3% fee, converted.\nANSWER: {amount}"
        return ChatResult(
            output=ModelOutput(text=text),
            model=get_tier(tier).model_id,
            tier=tier,
            latency_ms=4000,
            input_tokens=700,
            output_tokens=900,
        )


def iterator_reply(n: int) -> dict:
    return {
        "diagnosis": "The processing fee is rounded half-even instead of half up.",
        "system_prompt": None,
        "user_prefix": "",
        "user_suffix": f"Round every amount half up to 2 decimals (attempt {n}).",
        "few_shot": [],
        "reasoning": "high",
        "output_mode": "text",
        "max_tokens": None,
        "change_log": [f"Spelled out half-up rounding (attempt {n})"],
    }


def run(config: RunConfig):
    agents = FakeLLM(
        {
            ITERATOR_ROLE: [iterator_reply(n) for n in range(9)],
            VERDICT_ROLE: [{"reason": "Nemotron matches the refund references."}],
        }
    )
    v0 = Adapter(site_id="cs_05", version=0, spec=AdapterSpec(reasoning="high", max_tokens=2000))
    out = route_site(
        SITE,
        CASES,
        REFS,
        v0,
        Models(agents),
        config,
        iterator=Iterator(agents),
        price_lookup=pricing_yaml_price,
    )  # hermetic: no Tavily lookup
    return out, agents


def floor_pct(k: int, n: int) -> str:
    return f"{math.floor(Fraction(1000 * k, n)) / 10:.1f}"


def test_cs05_like_reasoning_site_stays_closed_with_an_honest_reason():
    holdout = [c.id for c in CASES if c.split == "holdout"]
    dev = [c.id for c in CASES if c.split == "dev"]
    for tier in Tier:  # precondition: no tier reaches 90% on either split
        assert sum(right(tier, c) for c in dev) / len(dev) < 0.9
        assert sum(right(tier, c) for c in holdout) / len(holdout) < 0.9

    out, agents = run(RunConfig(repo_url="x"))
    v = out.verdict
    assert v.decision == "KEEP_CLOSED" and v.tier is None and v.model is None
    assert v.best_tier == Tier.ULTRA and v.iterations == 3
    k = sum(right(Tier.ULTRA, c) for c in holdout)
    expected = f"parity vs reference {floor_pct(k, len(holdout))}% < 90%"
    assert v.parity_pct == float(floor_pct(k, len(holdout)))
    # the Super sentence lacked the measured phrase, so the rule sentence stands
    assert v.reason == f"{expected} on Ultra after 3 adapter iterations."
    assert sum(c["role"] == ITERATOR_ROLE for c in agents.calls) == 9


def test_canary_stays_closed_under_a_stricter_threshold_too():
    out, _ = run(RunConfig(repo_url="x", threshold=0.99, reasoning_threshold=0.95))
    assert out.verdict.decision == "KEEP_CLOSED" and "< 95%" in out.verdict.reason
