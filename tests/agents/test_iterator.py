"""Iterator Agent contract (PRD §6.6): FakeLLM unit tests, then cassettes recorded live from Token Factory."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from offramp.agents.adapter import JSON_ONLY, LIGHTNING_LEAK
from offramp.agents.iterator import (
    ITERATOR_MAX_TOKENS,
    ITERATOR_ROLE,
    MAX_CASES,
    DevCase,
    Iterator,
    IteratorDecision,
    failing_dev_cases,
)
from offramp.agents.translate import build_request, strict_json_schema
from offramp.evals.scorers import score_case
from offramp.schemas import (
    Adapter,
    BaselineResult,
    CallSite,
    CandidateResult,
    CaseScore,
    EvalCase,
    ModelOutput,
    Tier,
    TierMetrics,
    TierResult,
)
from offramp.tf import Budget, BudgetExceeded, ChatResult, FakeLLM, TFClient
from offramp.tf.tiers import get_tier

CASSETTES = Path(__file__).parents[1] / "fixtures/cassettes/iterator"
REPAIR = CASSETTES / "repair"  # one synthetic (hand-corrupted) cassette + the live repair reply
INPUTS = json.loads((CASSETTES / "inputs.json").read_text())
SCHEMAS = INPUTS["schemas"]
RECORDED = ["cs_01", "cs_03", "cs_05"]
NOW = datetime(2026, 10, 19, tzinfo=UTC)


def gc10(case_id: str) -> str:
    """Global Constraint 10, restated here so the tests do not use the module's own split function."""
    return "holdout" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100 < 30 else "dev"


def ids(split: str, n: int, prefix: str = "cs_01") -> list[str]:
    found = (f"{prefix}:{i:03d}" for i in range(1, 1000))
    return [cid for cid in found if gc10(cid) == split][:n]


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Model overrides would change cassette keys; tracing would wrap the SDK."""
    for name in [f"OFFRAMP_MODEL_{t.name}" for t in Tier] + ["LANGSMITH_TRACING", "OFFRAMP_DATA_DIR"]:
        monkeypatch.delenv(name, raising=False)


def fixture(sid: str):
    """One site's recorded inputs: site, adapter v0, cases, references, candidate calls, scored dev result."""
    data = INPUTS["sites"][sid]
    site, adapter = CallSite(**data["site"]), Adapter(**data["adapter_v0"])
    tier = Tier(INPUTS["tiers"][sid])
    cases = [EvalCase(**row["case"]) for row in data["cases"]]
    refs = {
        row["case"]["id"]: BaselineResult(
            case_id=row["case"]["id"],
            model="reference:fixture",
            output=ModelOutput(text=row["reference"]),
            recorded_at=NOW,
        )
        for row in data["cases"]
    }
    calls = [
        CandidateResult(
            case_id=row["case"]["id"],
            tier=tier,
            adapter_version=0,
            output=ModelOutput(text=row["candidate"]),
            latency_ms=900,
            input_tokens=600,
            output_tokens=40,
            cost_usd=0.0,
        )
        for row in data["cases"]
    ]
    scores = [
        score_case(c, refs[c.id].output, r.output, site.contract) for c, r in zip(cases, calls, strict=True)
    ]
    return site, adapter, cases, refs, calls, dev_result(site.id, tier, scores), tier


def dev_result(site_id: str, tier: Tier, scores: list[CaseScore], split: str = "dev") -> TierResult:
    return TierResult(
        site_id=site_id,
        tier=tier,
        adapter_version=0,
        split=split,
        passed=False,
        score=sum(s.passed for s in scores) / len(scores),
        metrics=TierMetrics(p50_ms=900, p95_ms=900, tokens_in=0, tokens_out=0, cost_per_1k_calls_usd=0.0),
        failing_case_ids=[s.case_id for s in scores if not s.passed],
        case_scores=scores,
    )


def decision(**over) -> dict:
    return {
        "diagnosis": "The model adds prose and capitals around the queue name.",
        "system_prompt": None,
        "user_prefix": "",
        "user_suffix": "",
        "few_shot": [],
        "reasoning": "off",
        "output_mode": "text",
        "max_tokens": None,
        "change_log": ["Pinned the reply to the bare lower-case queue name"],
    } | over


def dev_cases(sid: str = "cs_01") -> tuple:
    site, adapter, cases, refs, calls, result, tier = fixture(sid)
    return site, adapter, failing_dev_cases(result, cases, refs, calls), tier


# ---------------------------------------------------------------- the dev-only view (Global Constraint 10)


def test_a_holdout_case_cannot_be_a_dev_case():
    [holdout] = ids("holdout", 1)
    with pytest.raises(ValidationError, match="holdout"):
        DevCase(id=holdout, input_messages=[{"role": "user", "content": "x"}])
    [dev] = ids("dev", 1)
    assert DevCase(id=dev, input_messages=[{"role": "user", "content": "x"}]).id == dev


def test_failing_dev_cases_carry_input_reference_candidate_and_category():
    site, adapter, cases, refs, calls, result, tier = fixture("cs_03")
    view = failing_dev_cases(result, cases, refs, calls)
    assert [c.id for c in view] == result.failing_case_ids == [c.id for c in cases]
    first = view[0]
    assert first.input_messages == cases[0].input_messages
    assert first.reference == refs[cases[0].id].output and first.candidate == calls[0].output
    assert first.error_category == result.case_scores[0].error_category == "schema_violation"
    assert first.detail == result.case_scores[0].detail


def test_failing_dev_cases_refuse_a_holdout_result():
    site, adapter, cases, refs, calls, result, tier = fixture("cs_01")
    with pytest.raises(ValueError, match="dev"):
        failing_dev_cases(result.model_copy(update={"split": "holdout"}), cases, refs, calls)


def test_failing_dev_cases_keep_the_call_error_when_there_was_no_output():
    site, adapter, cases, refs, calls, result, tier = fixture("cs_01")
    calls[1] = calls[1].model_copy(update={"output": None, "error": "APITimeoutError: Request timed out."})
    [view] = [c for c in failing_dev_cases(result, cases, refs, calls) if c.id == calls[1].case_id]
    assert view.candidate is None and view.error == "APITimeoutError: Request timed out."


def test_failing_dev_cases_show_at_most_ten_round_robin_over_error_categories():
    dev = ids("dev", 14)
    cases = [
        EvalCase(
            id=cid,
            site_id="cs_01",
            input_messages=[{"role": "user", "content": cid}],
            expected_kind="label",
            split="dev",
        )
        for cid in dev
    ]
    categories = ["wrong_label"] * 11 + ["invalid_label", "empty_output", "empty_output"]
    scores = [
        CaseScore(case_id=cid, passed=False, score=0.0, error_category=cat)
        for cid, cat in zip(dev, categories, strict=True)
    ]
    view = failing_dev_cases(dev_result("cs_01", Tier.LIGHTNING, scores), cases, {}, [])
    assert len(view) == MAX_CASES == 10
    shown = Counter(c.error_category for c in view)
    assert shown == {"wrong_label": 7, "invalid_label": 1, "empty_output": 2}  # turns: w i e, w e, w w w w w
    assert [c.id for c in view][:3] == [dev[0], dev[11], dev[12]]  # first of each category leads
    assert all(c.reference is None and c.candidate is None for c in view)


# ------------------------------------------------------------ FakeLLM: the call, the new adapter, the guards


def test_one_structured_ultra_call_that_shows_the_failing_dev_cases():
    site, adapter, failing, tier = dev_cases("cs_01")
    llm = FakeLLM({ITERATOR_ROLE: [decision()]})
    Iterator(llm, SCHEMAS)(adapter, site, failing, tier, {"wrong_label": 1, "invalid_label": 2})
    [call] = llm.calls
    assert call["role"] == ITERATOR_ROLE and call["tier"] == Tier.ULTRA
    assert call["response_model"] is IteratorDecision and call["max_tokens"] == ITERATOR_MAX_TOKENS
    assert call["reasoning"] == "off" and call["temperature"] == 0
    facts = json.loads(call["messages"][1]["content"])
    assert facts["tier"] == "lightning" and facts["contract"]["task_type"] == "classification"
    assert facts["current_adapter"]["version"] == adapter.version
    shown = facts["failing_dev_cases"]
    assert shown["count_by_error_category"] == {"wrong_label": 1, "invalid_label": 2}
    assert [c["id"] for c in shown["shown"]] == [c.id for c in failing]
    [view] = [c for c in failing if c.id == "cs_01:003"]
    [entry] = [c for c in shown["shown"] if c["id"] == "cs_01:003"]
    assert entry["input_messages"] == view.input_messages
    assert (entry["reference"]["text"], entry["candidate"]["text"]) == ("account_access", "Account Access")
    assert entry["error_category"] == "invalid_label" and entry["detail"] == view.detail


def test_new_adapter_is_the_next_version_for_the_tier_with_its_change_log():
    site, adapter, failing, tier = dev_cases("cs_01")
    reply = decision(
        system_prompt="Route the ticket. Reply with one queue name in lower case.",
        user_prefix="Ticket follows.",
        user_suffix="Reply with the queue name only.",
        few_shot=[
            {"role": "user", "content": "Subject: Refund please\n\nThe kettle broke, I want my money back."},
            {"role": "assistant", "content": "returns_refunds"},
        ],
        max_tokens=24,
        change_log=["Rewrote the system prompt", "Added one example"],
    )
    new = Iterator(FakeLLM({ITERATOR_ROLE: [reply]}), SCHEMAS)(adapter, site, failing, tier)
    assert (new.site_id, new.version, new.tier) == (site.id, adapter.version + 1, Tier.LIGHTNING)
    assert new.change_log == ["Rewrote the system prompt", "Added one example"]
    spec = new.spec
    assert spec.system_prompt == "Route the ticket. Reply with one queue name in lower case."
    assert (
        spec.user_prefix == "Ticket follows.\n\n"
        and spec.user_suffix == "\n\nReply with the queue name only."
    )
    assert spec.few_shot == reply["few_shot"] and spec.max_tokens == 24
    for kept in ("tools", "tool_choice", "temperature", "extra_body", "response_format"):
        assert getattr(spec, kept) == getattr(adapter.spec, kept)
    assert new.notes[: len(adapter.notes)] == adapter.notes
    request = build_request(spec, EvalCase(**INPUTS["sites"]["cs_01"]["cases"][0]["case"]), tier)
    assert [m["role"] for m in request["messages"]] == ["system", "user", "assistant", "user"]
    assert request["messages"][-1]["content"].endswith("Reply with the queue name only.")


def test_blank_fields_clear_and_null_system_prompt_keeps_the_current_one():
    site, adapter, failing, tier = dev_cases("cs_01")
    adapter = adapter.model_copy(
        update={
            "spec": adapter.spec.model_copy(
                update={"system_prompt": "Earlier rewrite.", "user_suffix": "\n\nX"}
            )
        }
    )
    new = Iterator(FakeLLM({ITERATOR_ROLE: [decision()]}), SCHEMAS)(adapter, site, failing, tier)
    assert new.spec.system_prompt == "Earlier rewrite." and new.spec.user_suffix == ""
    assert new.spec.max_tokens == adapter.spec.max_tokens


def test_system_prompt_with_case_variables_is_not_replaced():
    site, adapter, failing, tier = dev_cases("cs_01")
    templ = site.template.model_copy(update={"system": "Route tickets for {subject}."})
    site = site.model_copy(update={"template": templ})
    new = Iterator(FakeLLM({ITERATOR_ROLE: [decision(system_prompt="A fixed prompt.")]}), SCHEMAS)(
        adapter, site, failing, tier
    )
    assert new.spec.system_prompt is None and any("system_prompt" in n for n in new.notes)


def test_few_shot_must_be_user_assistant_pairs():
    site, adapter, failing, tier = dev_cases("cs_01")
    odd = [{"role": "user", "content": "Subject: hi"}]
    new = Iterator(FakeLLM({ITERATOR_ROLE: [decision(few_shot=odd)]}), SCHEMAS)(adapter, site, failing, tier)
    assert new.spec.few_shot == adapter.spec.few_shot and any("few_shot" in n for n in new.notes)


def test_examples_copying_a_failing_dev_input_are_dropped():
    """Recorded live: Ultra copied failing cases into few-shot, memorising dev instead of fixing the cause."""
    site, adapter, failing, tier = dev_cases("cs_05")
    copied = failing[0].input_messages[-1]["content"].replace("\n", "<br>")  # as Ultra wrote it
    fresh = (
        "Plan: Starter Monthly, billing period 30 days\nList price: 19.00 USD per seat per period\nSeats: 4\n"
        "Period start: 2026-01-05\nCancelled on: 2026-01-20\nCoupon: 10% off\nRefunds already issued: 0 USD\n"
        "Customer paid in GBP at 0.79 GBP per 1 USD\n\nHow much do we refund, in GBP?"
    )
    shots = [
        {"role": "user", "content": copied},
        {"role": "assistant", "content": "ANSWER: 1069.23"},
        {"role": "user", "content": fresh},
        {"role": "assistant", "content": "Working.\nANSWER: 23.45"},
    ]
    new = Iterator(FakeLLM({ITERATOR_ROLE: [decision(few_shot=shots, reasoning="high")]}), SCHEMAS)(
        adapter, site, failing, tier
    )
    assert new.spec.few_shot == shots[2:]
    assert any("1 few-shot example(s) dropped" in n for n in new.notes)


def test_json_site_examples_must_be_valid_records():
    """Recorded live: Ultra's cs_03 example reply was just "{"."""
    site, adapter, failing, tier = dev_cases("cs_03")
    record = {
        "order_id": "HH-10001",
        "customer_name": "John Doe",
        "customer_email": "john@example.com",
        "order_date": "2024-01-01",
        "total_amount": 49.99,
        "currency": "USD",
        "issue": "damaged",
    }
    email = (
        "JSON schema: ...\n\nEmail:\nHi, order HH-10001 from 2024-01-01 (49.99 USD) arrived damaged. John Doe"
    )
    good = [{"role": "user", "content": email}, {"role": "assistant", "content": json.dumps(record)}]
    bad = [good[0], {"role": "assistant", "content": "{"}]
    off_schema = [good[0], {"role": "assistant", "content": json.dumps(record | {"total_amount": "49,99"})}]
    for shots, kept in ((good, good), (bad, []), (off_schema, [])):
        new = Iterator(FakeLLM({ITERATOR_ROLE: [decision(few_shot=shots)]}), SCHEMAS)(
            adapter, site, failing, tier
        )
        assert new.spec.few_shot == kept
        assert any("not a valid JSON reply" in n for n in new.notes) == (not kept)


def test_reasoning_heavy_site_keeps_thinking_on_and_flags_lightning():
    site, adapter, failing, tier = dev_cases("cs_05")
    new = Iterator(FakeLLM({ITERATOR_ROLE: [decision(reasoning="off")]}), SCHEMAS)(
        adapter, site, failing, Tier.LIGHTNING
    )
    assert new.spec.reasoning == "medium" and LIGHTNING_LEAK in new.risk_flags


def test_turning_thinking_off_drops_the_stale_lightning_leak_flag():
    site, adapter, failing, tier = dev_cases("cs_01")
    adapter = adapter.model_copy(update={"risk_flags": [*adapter.risk_flags, LIGHTNING_LEAK]})
    new = Iterator(FakeLLM({ITERATOR_ROLE: [decision(reasoning="off")]}), SCHEMAS)(
        adapter, site, failing, tier
    )
    assert LIGHTNING_LEAK not in new.risk_flags


def test_json_schema_mode_on_a_json_site_uses_the_resolved_schema():
    site, adapter, failing, tier = dev_cases("cs_03")
    adapter = adapter.model_copy(
        update={"spec": adapter.spec.model_copy(update={"response_format": {"type": "json_object"}})}
    )
    new = Iterator(FakeLLM({ITERATOR_ROLE: [decision(output_mode="json_schema")]}), SCHEMAS)(
        adapter, site, failing, tier
    )
    assert new.spec.response_format == strict_json_schema(SCHEMAS["OrderFields"], "OrderFields")
    assert new.spec.json_mode_fallback is False


def test_json_modes_are_ignored_where_the_code_does_not_parse_json():
    site, adapter, failing, tier = dev_cases("cs_01")
    new = Iterator(FakeLLM({ITERATOR_ROLE: [decision(output_mode="json_object")]}), SCHEMAS)(
        adapter, site, failing, tier
    )
    assert new.spec.response_format is None and any("ignored" in n for n in new.notes)


def test_a_rewrite_never_drops_the_json_request_where_the_code_json_loads_the_reply():
    site, adapter, failing, tier = dev_cases("cs_01")
    site = site.model_copy(update={"contract": site.contract.model_copy(update={"reply_json": "exact"})})
    llm = FakeLLM({ITERATOR_ROLE: [decision(user_suffix="Answer briefly.")]})
    new = Iterator(llm, SCHEMAS)(adapter, site, failing, tier)
    assert new.spec.user_suffix == f"\n\nAnswer briefly. {JSON_ONLY}"
    assert json.loads(llm.calls[0]["messages"][1]["content"])["contract"]["reply_json"] == "exact"


def test_after_a_refused_rewrite_the_iterator_is_told_why():
    site, adapter, failing, tier = dev_cases("cs_01")
    llm = FakeLLM({ITERATOR_ROLE: [decision(), decision()]})
    Iterator(llm, SCHEMAS)(adapter, site, failing, tier)
    Iterator(llm, SCHEMAS)(
        adapter, site, failing, tier, refused="system_prompt repeats 'utility' 752 times in a row"
    )
    plain, told = (json.loads(c["messages"][1]["content"]) for c in llm.calls)
    assert "previous_rewrite_refused" not in plain  # an ordinary call is the request the cassettes recorded
    assert told["previous_rewrite_refused"].startswith("system_prompt repeats 'utility' 752 times in a row")


def test_records_every_case_id_it_saw_and_shows_at_most_ten():
    site, adapter, _, tier = dev_cases("cs_01")
    many = [DevCase(id=cid, input_messages=[{"role": "user", "content": cid}]) for cid in ids("dev", 12)]
    llm = FakeLLM({ITERATOR_ROLE: [decision(), decision()]})
    iterator = Iterator(llm, SCHEMAS)
    iterator(adapter, site, many, tier)
    assert iterator.seen_case_ids == [c.id for c in many[:MAX_CASES]]
    prompt = llm.calls[0]["messages"][1]["content"]
    assert many[MAX_CASES - 1].id in prompt and many[MAX_CASES].id not in prompt
    iterator(adapter, site, many[:2], tier)
    assert iterator.seen_case_ids == [c.id for c in many[:MAX_CASES]] + [c.id for c in many[:2]]


def test_accepts_dev_case_views_only():
    site, adapter, failing, tier = dev_cases("cs_01")
    site_cases = [EvalCase(**row["case"]) for row in INPUTS["sites"]["cs_01"]["cases"]]
    llm = FakeLLM({ITERATOR_ROLE: [decision()]})
    iterator = Iterator(llm, SCHEMAS)
    with pytest.raises(TypeError, match="DevCase"):
        iterator(adapter, site, site_cases, tier)
    assert llm.calls == [] and iterator.seen_case_ids == []


def test_error_counts_default_to_the_shown_cases():
    site, adapter, failing, tier = dev_cases("cs_01")
    llm = FakeLLM({ITERATOR_ROLE: [decision()]})
    Iterator(llm, SCHEMAS)(adapter, site, failing, tier)
    facts = json.loads(llm.calls[0]["messages"][1]["content"])
    assert facts["failing_dev_cases"]["count_by_error_category"] == {"invalid_label": 2, "wrong_label": 1}


def test_malformed_reply_is_repaired_exactly_once():
    site, adapter, failing, tier = dev_cases("cs_01")
    llm = FakeLLM({ITERATOR_ROLE: ['{"diagnosis": "x", "reasoning": ', decision()]})
    new = Iterator(llm, SCHEMAS)(adapter, site, failing, tier)
    assert new is not None and len(llm.calls) == 2
    assert "not valid" in llm.calls[1]["messages"][-1]["content"]


def test_no_second_repair_when_the_client_already_repaired():
    site, adapter, failing, tier = dev_cases("cs_01")
    already = ChatResult(
        output=ModelOutput(text="nope"),
        parsed=None,
        model="m",
        tier=Tier.ULTRA,
        json_path="json_object",
        repaired=True,
    )
    llm = FakeLLM({ITERATOR_ROLE: [already]})
    assert Iterator(llm, SCHEMAS)(adapter, site, failing, tier) is None and len(llm.calls) == 1


def test_still_malformed_after_one_repair_gives_no_adapter():
    site, adapter, failing, tier = dev_cases("cs_01")
    llm = FakeLLM({ITERATOR_ROLE: ["nope", "still nope"]})
    assert Iterator(llm, SCHEMAS)(adapter, site, failing, tier) is None and len(llm.calls) == 2


def test_budget_exceeded_propagates():
    site, adapter, failing, tier = dev_cases("cs_01")
    llm = FakeLLM({ITERATOR_ROLE: [BudgetExceeded("Ultra agent call cap reached (40)")]})
    with pytest.raises(BudgetExceeded):
        Iterator(llm, SCHEMAS)(adapter, site, failing, tier)


# ---------------------------------------------------------------- cassettes recorded live from Token Factory


def replay(directory: Path = CASSETTES) -> TFClient:
    return TFClient(cache_dir=directory, live=False)


def run_recorded(sid: str, client) -> tuple[Adapter, Adapter | None, CallSite, Tier]:
    site, adapter, cases, refs, calls, result, tier = fixture(sid)
    failing = failing_dev_cases(result, cases, refs, calls)
    counts = Counter(s.error_category for s in result.case_scores if not s.passed)
    return adapter, Iterator(client, SCHEMAS)(adapter, site, failing, tier, counts), site, tier


@pytest.mark.parametrize("sid", RECORDED)
def test_iterator_on_recorded_cassettes(sid):
    adapter, new, site, tier = run_recorded(sid, replay())
    assert new is not None and (new.version, new.tier) == (1, tier)
    assert 1 <= len(new.change_log) <= 5 and all(e.strip() for e in new.change_log)
    for kept in ("tools", "tool_choice", "temperature", "extra_body"):
        assert getattr(new.spec, kept) == getattr(adapter.spec, kept)
    for row in INPUTS["sites"][sid]["cases"]:
        build_request(new.spec, EvalCase(**row["case"]), tier)  # the new spec is sendable
    # What Ultra chose in the recorded replies, after the guards (re-recording may change these pins)
    copies = [n for n in new.notes if "copy a failing dev case" in n]
    if sid == "cs_01":  # a rewritten system prompt and three new examples
        assert new.spec.system_prompt.startswith("You are the ticket router") and len(new.spec.few_shot) == 6
        assert copies == [] and new.spec.max_tokens == 16
    if sid == "cs_03":  # its one example reply was "{": refused, the stricter system prompt stays
        assert new.spec.few_shot == [] and any("not a valid JSON reply" in n for n in new.notes)
        assert new.spec.response_format["type"] == "json_schema" and "ISO 4217" in new.spec.system_prompt
    if sid == "cs_05":  # three examples copied the failing refunds: dropped; half-up rounding spelled out
        assert new.spec.few_shot == [] and copies == [
            "iterator: 3 few-shot example(s) dropped, they copy a failing dev case's input"
        ]
        assert new.spec.reasoning == "high" and "half-up" in new.spec.system_prompt


def test_malformed_cassette_is_repaired():
    """repair/ holds a hand-corrupted copy of cs_01's recorded reply (synthetic) and the live repair reply."""
    synthetic = [p for p in REPAIR.glob("*.json") if json.loads(p.read_text()).get("synthetic")]
    assert len(synthetic) == 1 and len(list(REPAIR.glob("*.json"))) == 2
    _, new, _, _ = run_recorded("cs_01", replay(REPAIR))
    assert new is not None and new.version == 1
    assert new.spec.few_shot == []  # the repaired reply copied all three failing tickets


@pytest.mark.live
def test_record_cassettes():
    """pytest --live: records every cassette above (4 Ultra calls; hits are free)."""
    client = TFClient(cache_dir=CASSETTES, live=True, budget=Budget(max_ultra_agent_calls=len(RECORDED)))
    for sid in RECORDED:
        run_recorded(sid, client)

    ultra = get_tier(Tier.ULTRA).model_id
    [source] = [
        p
        for p in CASSETTES.glob("*.json")
        if p.name != "inputs.json"
        and (e := json.loads(p.read_text()))["request"]["model"] == ultra
        and '"id": "cs_01"' in e["request"]["messages"][1]["content"]
    ]
    entry = json.loads(source.read_text())
    entry["response"]["choices"][0]["message"]["content"] = '{"diagnosis": "The labels drift", "reasoning": '
    entry["synthetic"] = True  # hand-corrupted: exercises the one repair retry
    REPAIR.mkdir(exist_ok=True)
    (REPAIR / source.name).write_text(json.dumps(entry, indent=2, ensure_ascii=False))
    run_recorded("cs_01", TFClient(cache_dir=REPAIR, live=True, budget=Budget(max_ultra_agent_calls=1)))
