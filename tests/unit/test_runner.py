"""Eval runner (PRD §6.5): every TierResult number on the dashboard and in the PR comes from run_tier.

Eval-target replies are keyed by case id because the thread pool calls cases in any order; judge replies
come from a FakeLLM queue (the judge runs after the pool, in order). build_request is a local stub with the
shape of Task 6's offramp.agents.translate.build_request.
"""

import threading
import time
from datetime import UTC, datetime

import httpx
import openai
import pytest

from offramp.evals.errors import MissingReference, ProviderError
from offramp.evals.runner import MAX_CONCURRENCY, BudgetStop, run_tier
from offramp.evals.scorers.aggregate import CANDIDATE_ERROR, Thresholds
from offramp.pricing import load_pricing
from offramp.schemas import (
    Adapter,
    AdapterSpec,
    BaselineResult,
    CallSite,
    CandidateResult,
    Contract,
    EvalCase,
    ModelOutput,
    Tier,
    TierResult,
)
from offramp.tf import BudgetExceeded, CacheMiss, ChatResult, FakeLLM, get_tier
from offramp.tf.budget import EVAL_TARGET_ROLE


def _site(task_type: str = "classification", contract: bool = True) -> CallSite:
    output = {"classification": "label", "free_text": "text"}[task_type]
    return CallSite(
        id="cs_01",
        file="app/x.py",
        lines=[1, 9],
        sdk="openai",
        call="client.chat.completions.create",
        prompt_source="literal",
        contract=Contract(task_type=task_type, output=output, labels=["billing", "bug"])
        if contract
        else None,
    )


def _case(n: int, split: str = "dev") -> EvalCase:
    case_id = f"cs_01:{n:03d}"
    return EvalCase(
        id=case_id,
        site_id="cs_01",
        input_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": case_id}],
        expected_kind="label",
        split=split,
    )


def _ref(n: int, text: str | None = "billing") -> BaselineResult:
    return BaselineResult(
        case_id=f"cs_01:{n:03d}",
        model="reference:seed",
        output=ModelOutput(text=text),
        recorded_at=datetime.now(UTC),
    )


def _adapter(reasoning: str = "off") -> Adapter:
    return Adapter(site_id="cs_01", version=2, spec=AdapterSpec(reasoning=reasoning))


def stub_build_request(spec: AdapterSpec, case: EvalCase, tier: Tier) -> dict:
    return {
        "role": EVAL_TARGET_ROLE,
        "tier": tier,
        "messages": case.input_messages,
        "max_tokens": 64,
        "temperature": spec.temperature,
        "json_schema": spec.response_format,
        "json_mode_fallback": spec.json_mode_fallback,
        "tools": spec.tools,
        "tool_choice": spec.tool_choice,
        "reasoning": spec.reasoning,
        "extra_body": dict(spec.extra_body),
    }


def _ok(text: str, latency: int = 100, tin: int = 10, tout: int = 5):
    """Reply factory: built at call time so `model` is whatever get_tier says for the called tier."""

    def make(tier: Tier) -> ChatResult:
        return ChatResult(
            output=ModelOutput(text=text),
            model=get_tier(tier).model_id,
            tier=tier,
            latency_ms=latency,
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=123.0,  # sentinel: the runner prices tokens with get_tier(tier), not this field
        )

    return make


_REQ = httpx.Request("POST", "https://example.invalid/v1/chat/completions")


def _timeout() -> openai.APITimeoutError:
    """A per-request failure of the model service: scored as a failed case."""
    return openai.APITimeoutError(request=_REQ)


def _status(status: int, body: str = "") -> openai.APIStatusError:
    return openai.APIStatusError(
        f"Error code: {status} - {body}", response=httpx.Response(status, request=_REQ, text=body), body=None
    )


class Slow:
    """Reply that takes `seconds` and then raises `exc` (a call that timed out after a while)."""

    def __init__(self, exc: BaseException, seconds: float):
        self.exc, self.seconds = exc, seconds


class CaseClient:
    """eval_target replies keyed by case id; every other role goes to a FakeLLM queue."""

    def __init__(self, replies: dict, judge_replies: tuple = (), delay: float = 0.0):
        self.replies = replies
        self.fake = FakeLLM({"judge": list(judge_replies)})
        self.target_calls: list[dict] = []
        self.delay = delay
        self.active = self.max_active = 0
        self._lock = threading.Lock()

    def chat(self, **kw) -> ChatResult:
        if kw["role"] != EVAL_TARGET_ROLE:
            return self.fake.chat(**kw)
        with self._lock:
            self.target_calls.append(kw)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            reply = self.replies[kw["messages"][-1]["content"]]
            if isinstance(reply, Slow):
                time.sleep(reply.seconds)
                raise reply.exc
            if isinstance(reply, BaseException):
                raise reply
            return reply(kw["tier"])
        finally:
            with self._lock:
                self.active -= 1

    @property
    def judge_calls(self) -> list[dict]:
        return self.fake.calls


def _run(client, cases, baselines, *, site=None, tier=Tier.LIGHTNING, split="dev", adapter=None, **kw):
    return run_tier(
        site or _site(),
        cases,
        baselines,
        adapter or _adapter(),
        tier,
        client,
        split,
        kw.pop("thresholds", Thresholds()),
        build_request=kw.pop("build_request", stub_build_request),
        **kw,
    )


# ------------------------------------------------------------------ deterministic sites


def test_scores_every_case_including_failed_calls():
    cases = [_case(i) for i in range(1, 5)]
    client = CaseClient(
        {
            "cs_01:001": _ok("billing"),
            "cs_01:002": _ok("billing"),
            "cs_01:003": _ok("bug"),
            "cs_01:004": _timeout(),
        }
    )
    result = _run(client, cases, [_ref(i) for i in range(1, 5)])

    assert isinstance(result, TierResult)
    assert [c.case_id for c in result.case_scores] == [c.id for c in cases]  # none dropped, case order kept
    assert result.failing_case_ids == ["cs_01:003", "cs_01:004"]
    assert result.case_scores[3].error_category == CANDIDATE_ERROR
    assert result.case_scores[3].detail.startswith("candidate call failed: APITimeoutError")
    assert (result.score, result.passed) == (0.5, False)
    assert (result.site_id, result.tier, result.split, result.adapter_version) == (
        "cs_01",
        Tier.LIGHTNING,
        "dev",
        2,
    )


def test_all_cases_matching_pass():
    cases = [_case(i) for i in range(1, 4)]
    client = CaseClient({c.id: _ok("Billing.") for c in cases})
    result = _run(client, cases, [_ref(i) for i in range(1, 4)])
    assert (result.passed, result.score, result.failing_case_ids) == (True, 1.0, [])


def test_threshold_comes_from_caller():
    cases = [_case(i) for i in range(1, 5)]
    client = CaseClient({**{c.id: _ok("billing") for c in cases[:3]}, "cs_01:004": _ok("bug")})
    refs = [_ref(i) for i in range(1, 5)]
    assert _run(client, cases, refs, thresholds=Thresholds(threshold=0.95)).passed is False
    assert _run(client, cases, refs, thresholds=Thresholds(threshold=0.75)).passed is True


def test_eval_target_role_tier_and_reasoning_are_enforced():
    def sloppy_build_request(spec, case, tier):  # wrong role, no tier, no reasoning
        return {"role": "adapter", "messages": case.input_messages, "max_tokens": 64}

    case = _case(1)
    client = CaseClient({case.id: _ok("billing")})
    _run(
        client,
        [case],
        [_ref(1)],
        tier=Tier.ULTRA,
        adapter=_adapter("high"),
        build_request=sloppy_build_request,
    )
    (call,) = client.target_calls
    assert call["role"] == "eval_target"  # the only Ultra role exempt from the agent-call cap
    assert call["tier"] == Tier.ULTRA
    assert call["reasoning"] == "high"  # enable_thinking follows AdapterSpec.reasoning (GC25)


def test_build_request_output_is_passed_through():
    case = _case(1)
    client = CaseClient({case.id: _ok("billing")})
    _run(client, [case], [_ref(1)], tier=Tier.SUPER)
    (call,) = client.target_calls
    assert call == stub_build_request(_adapter().spec, case, Tier.SUPER)


def test_runs_only_the_requested_split():
    cases = [_case(1, "dev"), _case(2, "holdout"), _case(3, "holdout")]
    client = CaseClient({c.id: _ok("billing") for c in cases})
    result = _run(client, cases, [_ref(i) for i in range(1, 4)], split="holdout")
    called = sorted(c["messages"][-1]["content"] for c in client.target_calls)
    assert called == ["cs_01:002", "cs_01:003"]
    assert [c.case_id for c in result.case_scores] == ["cs_01:002", "cs_01:003"]
    assert result.split == "holdout"


def test_case_without_reference_raises_before_any_call():
    cases = [_case(1)] + [_case(i, "holdout") for i in range(2, 12)]
    client = CaseClient({c.id: _ok("billing") for c in cases})
    with pytest.raises(MissingReference) as err:
        _run(client, cases, [_ref(2)], split="holdout")  # 10 holdout cases, 1 reference
    assert err.value.case_ids == [f"cs_01:{i:03d}" for i in range(3, 12)]
    assert "cs_01:003" in str(err.value)
    assert client.target_calls == []  # nothing spent on a split that cannot be scored


def test_reference_check_covers_only_the_requested_split():
    cases = [_case(1), _case(2, "holdout")]  # the holdout case has no reference; a dev run does not need it
    client = CaseClient({c.id: _ok("billing") for c in cases})
    result = _run(client, cases, {"cs_01:001": _ref(1)})  # a mapping works as well as a list
    assert [c.case_id for c in result.case_scores] == ["cs_01:001"]


def test_no_cases_never_passes():
    result = _run(CaseClient({}), [], [])
    assert (result.passed, result.score, result.case_scores) == (False, 0.0, [])
    assert result.metrics.p50_ms == 0 and result.metrics.cost_per_1k_calls_usd == 0.0


def test_missing_contract_raises():
    with pytest.raises(ValueError, match="contract"):
        _run(CaseClient({}), [_case(1)], [_ref(1)], site=_site(contract=False))


# ------------------------------------------------------------------ metrics


def _lightning_price() -> tuple[float, float]:
    """Read pricing.yaml directly (not through get_tier) so the oracle is independent of the runner."""
    (entry,) = [e for e in load_pricing().values() if e.get("tier") == "lightning"]
    return entry["input_usd_per_m"], entry["output_usd_per_m"]


def test_metrics_latency_percentiles_tokens_and_cost():
    lat = [400, 100, 1000, 300, 200]
    toks = [(100, 10), (200, 20), (300, 30), (400, 40), (500, 50)]
    cases = [_case(i) for i in range(1, 7)]
    replies = {f"cs_01:{i:03d}": _ok("billing", lat[i - 1], *toks[i - 1]) for i in range(1, 6)}
    replies["cs_01:006"] = _timeout()  # fails at once: in the percentiles (~0 ms), no tokens or cost
    result = _run(CaseClient(replies), cases, [_ref(i) for i in range(1, 7)])

    m = result.metrics
    assert (m.p50_ms, m.p95_ms) == (200, 1000)  # nearest rank over [~0, 100, 200, 300, 400, 1000]
    assert (m.tokens_in, m.tokens_out) == (1500, 150)
    p_in, p_out = _lightning_price()
    mean_cost = sum(i * p_in + o * p_out for i, o in toks) / 1_000_000 / 5
    assert m.cost_per_1k_calls_usd == pytest.approx(1000 * mean_cost)


def test_percentiles_are_nearest_rank_on_even_counts():
    cases = [_case(i) for i in range(1, 5)]
    replies = {c.id: _ok("billing", lat) for c, lat in zip(cases, [400, 100, 300, 200], strict=True)}
    m = _run(CaseClient(replies), cases, [_ref(i) for i in range(1, 5)]).metrics
    assert (m.p50_ms, m.p95_ms) == (200, 400)  # rank ceil(0.5 * 4) = 2 -> 200, not an interpolated 250


def test_failed_call_counts_in_percentiles_at_its_time_until_failure():
    cases = [_case(i) for i in range(1, 5)]
    replies = {c.id: _ok("billing", 100) for c in cases[:3]}
    replies["cs_01:004"] = Slow(_timeout(), 0.3)
    m = _run(CaseClient(replies), cases, [_ref(i) for i in range(1, 5)]).metrics
    assert m.p50_ms == 100
    assert 300 <= m.p95_ms < 2000  # the timed-out call is the tail, not 0 ms


def test_cost_uses_tier_price_even_when_model_id_is_overridden(monkeypatch):
    monkeypatch.setenv("OFFRAMP_MODEL_LIGHTNING", "acme/model-not-in-pricing-yaml")
    case = _case(1)
    result = _run(CaseClient({case.id: _ok("billing", tin=1000, tout=100)}), [case], [_ref(1)])
    p_in, p_out = _lightning_price()
    assert result.metrics.cost_per_1k_calls_usd == pytest.approx(
        1000 * (1000 * p_in + 100 * p_out) / 1_000_000
    )


def test_concurrency_is_bounded():
    cases = [_case(i) for i in range(1, 25)]
    client = CaseClient({c.id: _ok("billing") for c in cases}, delay=0.02)
    _run(client, cases, [_ref(i) for i in range(1, 25)])
    assert len(client.target_calls) == 24
    assert client.max_active == MAX_CONCURRENCY == 8


# ------------------------------------------------------------------ budget and replay errors


def test_budget_exceeded_mid_matrix_returns_budget_stop():
    cases = [_case(i, "holdout") for i in range(1, 4)]
    replies = {c.id: _ok("billing") for c in cases}
    replies["cs_01:002"] = BudgetExceeded("token cap reached (1500000/1500000 in, 0/300000 out)")
    result = _run(
        CaseClient(replies), cases, [_ref(i) for i in range(1, 4)], tier=Tier.SUPER, split="holdout"
    )
    assert result == BudgetStop(
        site_id="cs_01",
        tier=Tier.SUPER,
        split="holdout",
        reason="budget: token cap reached (1500000/1500000 in, 0/300000 out)",
    )


def test_cache_miss_propagates():
    case = _case(1)
    with pytest.raises(CacheMiss):
        _run(CaseClient({case.id: CacheMiss("no cached response")}), [case], [_ref(1)])


# ------------------------------------------------------------------ provider errors are run errors

PROVIDER_ERRORS = [
    _status(401, '{"error": "invalid api key"}'),
    _status(402, '{"error": "insufficient balance"}'),
    _status(403, '{"error": "forbidden"}'),
    _status(404, '{"error": "model nvidia/unknown not found"}'),
    _status(429, '{"error": "quota exceeded: credit balance is 0"}'),
    _status(429, '{"error": "rate limit reached, retry later"}'),  # still limited after the SDK retries
    _status(502),  # gateway / service outage after the SDK retries: says nothing about the model
    _status(503),
    _status(504),
    openai.APIConnectionError(request=_REQ),  # network down: says nothing about the model
]
CASE_FAILURES = [_timeout(), _status(400, '{"error": "context length exceeded"}'), _status(422), _status(500)]


@pytest.mark.parametrize(
    "exc", PROVIDER_ERRORS, ids=lambda e: type(e).__name__ + str(getattr(e, "status_code", ""))
)
def test_provider_error_is_raised_never_scored(exc):
    cases = [_case(i) for i in range(1, 4)]
    replies = {c.id: _ok("billing") for c in cases}
    replies["cs_01:002"] = exc
    with pytest.raises(ProviderError) as err:
        _run(CaseClient(replies), cases, [_ref(i) for i in range(1, 4)])
    assert err.value.__cause__ is exc
    assert err.value.status == getattr(exc, "status_code", None)


@pytest.mark.parametrize(
    "exc", CASE_FAILURES, ids=lambda e: type(e).__name__ + str(getattr(e, "status_code", ""))
)
def test_model_side_failure_is_a_scored_failed_case(exc):
    cases = [_case(1), _case(2), _case(3)]  # 1 failure in 3: below the outage fraction
    replies = {"cs_01:001": _ok("billing"), "cs_01:002": exc, "cs_01:003": _ok("billing")}
    result = _run(CaseClient(replies), cases, [_ref(i) for i in range(1, 4)])
    assert result.failing_case_ids == ["cs_01:002"]
    assert result.case_scores[1].error_category == CANDIDATE_ERROR


@pytest.mark.parametrize(
    ("failures", "exc", "outage"),
    [
        (1, _timeout(), False),  # 1 of 4 timed out: a slow model, scored
        (2, _timeout(), True),  # 2 of 4 (50%): a run that broad is an outage, not a measurement
        (2, _status(500), True),  # 5xx count toward it as well
        (3, _status(500), True),
        (2, _status(400), False),  # request-level rejections are the case's own: always scored
    ],
)
def test_broad_timeouts_or_5xx_are_an_outage(failures, exc, outage):
    cases = [_case(i) for i in range(1, 5)]
    replies = {c.id: (exc if i < failures else _ok("billing")) for i, c in enumerate(cases)}
    refs = [_ref(i) for i in range(1, 5)]
    if outage:
        with pytest.raises(ProviderError, match=f"{failures}/4"):
            _run(CaseClient(replies), cases, refs)
    else:
        assert len(_run(CaseClient(replies), cases, refs).failing_case_ids) == failures


def test_judge_timeout_is_a_run_error_not_a_candidate_score():
    case = _case(1)
    client = CaseClient({case.id: _ok("x")}, judge_replies=(_timeout(),))
    with pytest.raises(ProviderError, match="judge"):
        _run(client, [case], [], site=_site("free_text"))


def test_provider_error_from_the_judge_is_raised():
    case = _case(1)
    client = CaseClient({case.id: _ok("x")}, judge_replies=(_status(402, "insufficient balance"),))
    with pytest.raises(ProviderError):
        _run(client, [case], [], site=_site("free_text"))


# ------------------------------------------------------------------ candidate outputs for the Iterator


def test_return_candidates_gives_every_raw_output_in_case_order():
    cases = [_case(1), _case(2), _case(3)]
    client = CaseClient({"cs_01:001": _ok("billing"), "cs_01:002": _ok("bug"), "cs_01:003": _timeout()})
    result, candidates = _run(client, cases, [_ref(i) for i in range(1, 4)], return_candidates=True)
    assert isinstance(result, TierResult) and result.failing_case_ids == ["cs_01:002", "cs_01:003"]
    assert all(isinstance(c, CandidateResult) for c in candidates)
    assert [c.case_id for c in candidates] == ["cs_01:001", "cs_01:002", "cs_01:003"]
    assert [c.output.text if c.output else None for c in candidates] == ["billing", "bug", None]
    assert candidates[2].error.startswith("APITimeoutError")
    assert {(c.tier, c.adapter_version) for c in candidates} == {(Tier.LIGHTNING, 2)}


def test_return_candidates_on_budget_stop():
    case = _case(1)
    client = CaseClient({case.id: BudgetExceeded("token cap reached")})
    stop, candidates = _run(client, [case], [_ref(1)], return_candidates=True)
    assert isinstance(stop, BudgetStop) and candidates == []


# ------------------------------------------------------------------ free_text sites (judge)


def _judge_reply(*items):
    return {"scores": list(items)}


def _rub(n: int) -> dict:
    return {"groundedness": n, "completeness": n, "format": n}


def test_free_text_without_reference_text_uses_absolute_judge_and_floor():
    cases = [_case(1), _case(2)]
    client = CaseClient(
        {c.id: _ok(f"answer {c.id}") for c in cases},
        judge_replies=(
            _judge_reply({"case": 1, "note": "", "a": _rub(5)}, {"case": 2, "note": "", "a": _rub(4)}),
        ),
    )
    refs = [_ref(1, text=None), _ref(2, text=None)]  # free-text references without text: no pairwise
    result = _run(client, cases, refs, site=_site("free_text"))

    (call,) = client.judge_calls
    assert call["role"] == "judge" and call["tier"] == Tier.ULTRA
    assert [c.score for c in result.case_scores] == [5.0, 4.0]
    assert (result.passed, result.score, result.failing_case_ids) == (True, 4.5, [])


def test_free_text_failed_call_scores_one_and_is_not_judged():
    cases = [_case(1), _case(2), _case(3)]
    client = CaseClient(
        {"cs_01:001": _ok("a1"), "cs_01:002": _ok("a2"), "cs_01:003": _timeout()},
        judge_replies=(
            _judge_reply({"case": 1, "note": "", "a": _rub(5)}, {"case": 2, "note": "", "a": _rub(4)}),
        ),
    )
    result = _run(client, cases, [], site=_site("free_text"))  # free text needs no reference to run

    assert "cs_01:003" not in client.judge_calls[0]["messages"][-1]["content"]
    by_id = {c.case_id: c for c in result.case_scores}
    assert by_id["cs_01:003"].score == 1.0 and by_id["cs_01:003"].error_category == CANDIDATE_ERROR
    assert result.score == pytest.approx(10 / 3)
    assert result.passed is False
    assert result.failing_case_ids == ["cs_01:003"]


def test_free_text_with_reference_is_pairwise_and_reference_mean_gates_pass():
    cases = [_case(1), _case(2)]
    pair = [{"case": i, "note": "", "a": _rub(4), "b": _rub(5)} for i in (1, 2)]  # A/B: candidate 4, ref 5
    swap = [{"case": i, "note": "", "a": _rub(5), "b": _rub(4)} for i in (1, 2)]  # B/A: ref 5, candidate 4
    client = CaseClient(
        {c.id: _ok("candidate") for c in cases}, judge_replies=(_judge_reply(*pair), _judge_reply(*swap))
    )
    refs = [_ref(1, text="human reference"), _ref(2, text="human reference")]
    result = _run(client, cases, refs, site=_site("free_text"))

    assert len(client.judge_calls) == 2  # A/B then B/A
    assert result.score == 4.0  # clears the 4.0 floor ...
    assert result.passed is False  # ... but not reference mean 5.0 - 0.3
    assert result.failing_case_ids == ["cs_01:001", "cs_01:002"]


def test_free_text_case_scores_keep_the_judge_score_of_the_reference():
    cases = [_case(1), _case(2)]
    pair = [
        {"case": 1, "note": "", "a": _rub(5), "b": _rub(1)},
        {"case": 2, "note": "", "a": _rub(5), "b": _rub(4)},
    ]
    swap = [
        {"case": 1, "note": "", "a": _rub(1), "b": _rub(5)},
        {"case": 2, "note": "", "a": _rub(4), "b": _rub(5)},
    ]
    client = CaseClient(
        {c.id: _ok("candidate") for c in cases}, judge_replies=(_judge_reply(*pair), _judge_reply(*swap))
    )
    refs = [_ref(1, text="a weak reference"), _ref(2, text="a good reference")]

    result = _run(client, cases, refs, site=_site("free_text"))

    assert [(c.case_id, c.score, c.reference_score) for c in result.case_scores] == [
        ("cs_01:001", 5.0, 1.0),
        ("cs_01:002", 5.0, 4.0),
    ]


def test_budget_exceeded_in_judge_returns_budget_stop():
    case = _case(1)
    client = CaseClient(
        {case.id: _ok("x")}, judge_replies=(BudgetExceeded("Ultra agent call cap reached (40)"),)
    )
    result = _run(client, [case], [], site=_site("free_text"))
    assert isinstance(result, BudgetStop)
    assert result.reason == "budget: Ultra agent call cap reached (40)"


# ------------------------------------------------------------------ default build_request (Task 6)


def test_default_build_request_is_task6_translate():
    from offramp.agents.translate import max_tokens_for

    spec = AdapterSpec(
        system_prompt="Reply with one label: billing or bug.", user_suffix="\nLabel:", max_tokens=16
    )
    adapter = Adapter(site_id="cs_01", version=1, spec=spec)
    case = _case(1)
    fake = FakeLLM({EVAL_TARGET_ROLE: ["billing"]})
    result = run_tier(_site(), [case], [_ref(1)], adapter, Tier.SUPER, fake, "dev")  # no build_request=

    assert (result.passed, result.score) == (True, 1.0)
    (call,) = fake.calls
    assert call["role"] == EVAL_TARGET_ROLE and call["tier"] == Tier.SUPER and call["reasoning"] == "off"
    assert call["messages"] == [
        {"role": "system", "content": "Reply with one label: billing or bug."},  # the spec replaced "sys"
        {"role": "user", "content": "cs_01:001\nLabel:"},
    ]
    assert call["max_tokens"] == max_tokens_for(spec) == 16


def test_unresolved_call_site_propagates_and_is_never_scored():
    from offramp.agents.translate import UnresolvedCallSite

    spec = AdapterSpec(
        response_format={"type": "json_schema", "json_schema": {"name": "T", "schema": "<pydantic:T>"}}
    )
    fake = FakeLLM()  # a call would raise: none must be made
    with pytest.raises(UnresolvedCallSite):
        run_tier(
            _site(),
            [_case(1)],
            [_ref(1)],
            Adapter(site_id="cs_01", version=0, spec=spec),
            Tier.LIGHTNING,
            fake,
            "dev",
        )
    assert fake.calls == []
