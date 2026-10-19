"""Holdout leakage (PRD §12.3, Global Constraint 10): the Iterator never receives a holdout case, id or input.

The loop runs for real: route_site, run_tier, build_request, the label scorer, the site rule, the Iterator and
the verdict rule. Only the models are fakes: eval targets answer each case by the marker in its input (FIXTURE
outputs), and the Iterator's Ultra replies are canned. The scenario makes holdout failures exist before and
between iterations (Super passes dev and fails holdout, then Ultra iterates), so a leak has something to leak.
"""

import hashlib
import re
from datetime import UTC, datetime

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
from offramp.tf import BudgetExceeded, ChatResult, FakeLLM
from offramp.tf.tiers import get_tier

NOW = datetime(2026, 10, 19, tzinfo=UTC)
MARK = re.compile(r"MARK-(cs_09:\d{3})")


def gc10(case_id: str) -> str:
    """Global Constraint 10, restated: the test does not use the code under test to decide splits."""
    return "holdout" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100 < 30 else "dev"


SITE = CallSite(
    id="cs_09",
    file="helphub/inbox/triage.py",
    lines=[30, 41],
    sdk="openai",
    call="client.chat.completions.create",
    source_model="gpt-4o-mini",
    prompt_source="fstring",
    template=PromptTemplate(system="Route the ticket.", user="{body}", variables=["body"]),
    contract=Contract(task_type="classification", output="label", labels=["billing", "shipping", "general"]),
)
CASES = [
    EvalCase(
        id=cid,
        site_id="cs_09",
        input_messages=[
            {"role": "system", "content": "Route the ticket."},
            {"role": "user", "content": f"Ticket MARK-{cid}: my parcel number {i} is late."},
        ],
        expected_kind="label",
        split=gc10(cid),
    )
    for i, cid in ((i, f"cs_09:{i:03d}") for i in range(1, 41))
]
REFS = {
    c.id: BaselineResult(
        case_id=c.id, model="reference:seed", output=ModelOutput(text="shipping"), recorded_at=NOW
    )
    for c in CASES
}
HOLDOUT = {c.id for c in CASES if c.split == "holdout"}
DEV = {c.id for c in CASES if c.split == "dev"}


def right(tier: Tier, cid: str) -> bool:
    """FIXTURE behaviour: Lightning and Ultra fail dev; Super passes dev and fails holdout."""
    n = int(cid.split(":")[1])
    if tier == Tier.LIGHTNING:
        return n % 5 > 1
    if tier == Tier.SUPER:
        return cid in DEV or n % 2 == 1
    return n % 5 != 0


class Models:
    """Eval targets answer by case marker; every other role is served (and recorded) by a FakeLLM."""

    def __init__(self, agents: FakeLLM):
        self.agents = agents

    def chat(self, *, role, tier, messages, **kw):
        if role != "eval_target":
            return self.agents.chat(role=role, tier=tier, messages=messages, **kw)
        [cid] = MARK.findall(messages[-1]["content"])
        text = "shipping" if right(Tier(tier), cid) else "billing"
        return ChatResult(
            output=ModelOutput(text=text),
            model=get_tier(tier).model_id,
            tier=tier,
            latency_ms=500,
            input_tokens=120,
            output_tokens=2,
        )


def decision(n: int) -> dict:
    return {
        "diagnosis": "Late parcels are routed to billing.",
        "system_prompt": None,
        "user_prefix": "",
        "user_suffix": f"Late or missing parcels belong to shipping (attempt {n}).",
        "few_shot": [],
        "reasoning": "off",
        "output_mode": "text",
        "max_tokens": None,
        "change_log": [f"Clarified the shipping queue (attempt {n})"],
    }


def test_the_iterator_never_sees_a_holdout_case():
    assert HOLDOUT and DEV
    agents = FakeLLM(
        {ITERATOR_ROLE: [decision(n) for n in range(6)], VERDICT_ROLE: [BudgetExceeded("rule text")]}
    )
    iterator = Iterator(agents)
    events = []
    out = route_site(
        SITE,
        CASES,
        REFS,
        Adapter(site_id="cs_09", version=0, spec=AdapterSpec(max_tokens=16)),
        Models(agents),
        RunConfig(repo_url="x"),
        iterator=iterator,
        price_lookup=pricing_yaml_price,  # hermetic: no Tavily lookup
        on_event=lambda t, p: events.append((t, p)),
    )

    # the scenario: 3 iterations on Lightning, Super dev pass + holdout fail, 3 iterations on Ultra
    runs = [
        (p["tier"], p["split"], p["adapter_version"], p["passed"]) for t, p in events if t == "tier_result"
    ]
    assert ("super", "holdout", 0, False) in runs
    holdout_failures = {
        cid
        for t, p in events
        if t == "tier_result" and p["split"] == "holdout"
        for cid in p["failing_case_ids"]
    }
    assert holdout_failures  # something existed to leak
    iterator_calls = [c for c in agents.calls if c["role"] == ITERATOR_ROLE]
    assert (
        len(iterator_calls) == 6
        and out.verdict.decision == "KEEP_CLOSED"
        and out.verdict.best_tier == Tier.SUPER
    )

    # instrumented: every case id the Iterator saw is a dev id
    assert iterator.seen_case_ids and set(iterator.seen_case_ids) <= DEV
    assert not set(iterator.seen_case_ids) & HOLDOUT
    # and nothing of a holdout case reached its prompts: no id, no input marker
    for call in iterator_calls:
        text = "\n".join(str(m["content"]) for m in call["messages"])
        assert not set(MARK.findall(text)) & HOLDOUT
        assert not any(cid in text for cid in HOLDOUT)
        assert set(MARK.findall(text)) & DEV  # the check would see a marker if one were there
