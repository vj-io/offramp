"""Token Factory after-prices from its model API (offramp/tf/prices.py), on httpx.MockTransport: no network.

The API response below is a made-up fixture in the shape GET /v1/models?verbose=true returned on 2026-10-21
(per-model `pricing` in USD per token; the values are the ones read that day, docs/plan progress #4).
"""

import json
from datetime import UTC, datetime

import httpx
import pytest

from offramp.integrations import keys
from offramp.schemas import RunEvent, Stage, Tier, Verdict
from offramp.tf import prices

NOW = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)
TIERS = [Tier.LIGHTNING, Tier.SUPER, Tier.ULTRA]
LISTING = {
    "object": "list",
    "data": [
        {"id": "nvidia/Nemotron-3_5-Lightning", "pricing": {"prompt": "6e-08", "completion": "2.4e-07"}},
        {"id": "nvidia/nemotron-3-super-120b-a12b", "pricing": {"prompt": 3e-07, "completion": 9e-07}},
        {"id": "nvidia/Nemotron-3-Ultra-550b-a55b", "pricing": {"prompt": "1e-06", "completion": "3e-06"}},
        {"id": "some/unpriced-model"},
    ],
}


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    for name in ["NEBIUS_BASE_URL", "NEBIUS_API_KEY_FALLBACK", "OFFRAMP_DATA_DIR"] + [
        f"OFFRAMP_MODEL_{t.name}" for t in Tier
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEBIUS_API_KEY", "nb-secret-key")
    monkeypatch.setattr(keys, "_exhausted", set())


class API:
    """Scripted model API: each reply is a body (dict), an HTTP status (int) or an exception to raise."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, json={"detail": "nb-secret-key is not valid"})
        return httpx.Response(200, json=reply)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def read(tmp_path, api, live=True, now=NOW):
    return prices.after_prices(TIERS, live=live, http=api.client(), cache_dir=tmp_path, now=now)


def test_live_prices_come_from_the_model_api_per_1m_tokens_and_are_labelled_with_the_day(tmp_path):
    api = API(LISTING)
    got = read(tmp_path, api)
    (request,) = api.requests
    assert request.method == "GET"
    assert str(request.url) == "https://api.tokenfactory.nebius.com/v1/models?verbose=true"
    assert request.headers["authorization"] == "Bearer nb-secret-key"
    assert {t: (p.input_usd_per_m, p.output_usd_per_m) for t, p in got.items()} == {
        Tier.LIGHTNING: (0.06, 0.24),
        Tier.SUPER: (0.30, 0.90),
        Tier.ULTRA: (1.00, 3.00),
    }
    lightning = got[Tier.LIGHTNING]
    assert lightning.model == "nvidia/Nemotron-3_5-Lightning" and lightning.via == "token_factory_api"
    assert {p.source for p in got.values()} == {"Token Factory model API, 2026-10-21"}


def test_one_read_per_utc_day(tmp_path):
    api = API(LISTING, LISTING)
    first = read(tmp_path, api)
    assert read(tmp_path, api) == first and len(api.requests) == 1  # cached: no second call today
    later = read(tmp_path, api, now=datetime(2026, 10, 22, 0, 5, tzinfo=UTC))
    assert len(api.requests) == 2 and later[Tier.SUPER].source == "Token Factory model API, 2026-10-22"
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["2026-10-21.json", "2026-10-22.json"]
    assert "nb-secret-key" not in (tmp_path / "2026-10-21.json").read_text()


def test_cache_only_never_calls_the_api_and_uses_the_newest_cached_read(tmp_path):
    read(tmp_path, API(LISTING), now=datetime(2026, 10, 20, tzinfo=UTC))
    read(tmp_path, API(LISTING))
    offline = API()  # no reply to give: any call would fail the test
    got = read(tmp_path, offline, live=False, now=datetime(2026, 12, 1, tzinfo=UTC))
    assert offline.requests == []
    assert got[Tier.ULTRA].via == "token_factory_api"
    assert got[Tier.ULTRA].source == "Token Factory model API, 2026-10-21 (cached; cache-only run)"


def test_cache_only_without_a_cached_read_uses_pricing_yaml_and_says_so(tmp_path):
    offline = API()
    got = read(tmp_path, offline, live=False)
    assert offline.requests == [] and {p.via for p in got.values()} == {"pricing.yaml"}
    assert got[Tier.LIGHTNING].source == (
        "pricing.yaml as of 2026-10-21; cache-only run with no cached Token Factory model API read"
    )
    assert (got[Tier.LIGHTNING].input_usd_per_m, got[Tier.LIGHTNING].output_usd_per_m) == (0.06, 0.24)


@pytest.mark.parametrize(
    "reply,why",
    [
        (401, "HTTP 401"),
        (httpx.ConnectTimeout("slow"), "ConnectTimeout"),
        ({"data": "not a list"}, "unexpected response"),
    ],
)
def test_a_failed_read_falls_back_to_pricing_yaml_with_a_fixed_reason_and_is_not_cached(tmp_path, reply, why):
    got = read(tmp_path, API(reply))
    assert got[Tier.SUPER].via == "pricing.yaml"
    assert (
        got[Tier.SUPER].source == f"pricing.yaml as of 2026-10-21; the Token Factory model API failed ({why})"
    )
    assert "nb-secret-key" not in json.dumps({t: p.model_dump() for t, p in got.items()})
    assert not list(tmp_path.glob("*.json"))


def test_without_a_key_nothing_is_sent(tmp_path, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY")
    api = API()
    got = read(tmp_path, api)
    assert api.requests == [] and "NEBIUS_API_KEY is not set" in got[Tier.ULTRA].source


def test_a_model_the_api_does_not_price_falls_back_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFRAMP_MODEL_SUPER", "some/unpriced-model")
    got = read(tmp_path, API(LISTING))
    assert got[Tier.LIGHTNING].via == "token_factory_api"
    assert got[Tier.SUPER].model == "some/unpriced-model" and got[Tier.SUPER].via == "pricing.yaml"
    assert got[Tier.SUPER].source.endswith("some/unpriced-model is not priced by the Token Factory model API")
    assert (got[Tier.SUPER].input_usd_per_m, got[Tier.SUPER].output_usd_per_m) == (0.30, 0.90)  # the tier's


def test_an_api_price_that_differs_from_pricing_yaml_is_shown_with_the_one_the_costs_use(tmp_path):
    listing = json.loads(json.dumps(LISTING))
    listing["data"][0]["pricing"] = {"prompt": "7e-08", "completion": "2.4e-07"}
    got = read(tmp_path, API(listing))[Tier.LIGHTNING]
    assert (got.input_usd_per_m, got.output_usd_per_m) == (0.07, 0.24)
    assert got.source == (
        "Token Factory model API, 2026-10-21; this run's cost figures use pricing.yaml's $0.06 / $0.24"
    )


@pytest.mark.parametrize("bad", ["NaN", "inf", "-3e-07", "1e300", 0])
def test_an_implausible_api_price_is_never_shown_and_the_run_record_survives(tmp_path, bad):
    """Review T19 I2: a NaN, infinite, negative, zero or absurd price falls back to pricing.yaml and says
    why, and the verdict carrying it survives the JSON event log and re-validation (NaN would not)."""
    listing = json.loads(json.dumps(LISTING))
    listing["data"][1]["pricing"] = {"prompt": bad, "completion": "9e-07"}
    got = read(tmp_path, API(listing))
    assert got[Tier.LIGHTNING].via == "token_factory_api"  # the others are unaffected
    sup = got[Tier.SUPER]
    assert (sup.via, sup.input_usd_per_m, sup.output_usd_per_m) == ("pricing.yaml", 0.30, 0.90)
    assert sup.source == (
        "pricing.yaml as of 2026-10-21; the Token Factory model API lists no plausible price for "
        "nvidia/nemotron-3-super-120b-a12b"
    )
    assert read(tmp_path, API()) == got  # the cached read says the same, with no second call
    verdict = Verdict(site_id="cs_02", decision="MIGRATE", reason="r", after_price=sup)
    event = RunEvent(
        run_id="r", seq=0, stage=Stage.VERDICTS, type="verdict", payload=verdict.model_dump(mode="json")
    )
    assert Verdict.model_validate(RunEvent.model_validate_json(event.model_dump_json()).payload) == verdict


def test_an_unwritable_cache_costs_a_read_next_time_not_the_run(tmp_path, monkeypatch):
    def unwritable(path, prices, now):
        raise PermissionError("read-only")

    monkeypatch.setattr(prices, "_write", unwritable)
    assert read(tmp_path, API(LISTING))[Tier.SUPER].via == "token_factory_api"


def test_cache_only_skips_an_unreadable_newest_read(tmp_path):
    read(tmp_path, API(LISTING), now=datetime(2026, 10, 20, tzinfo=UTC))
    (tmp_path / "2026-10-21.json").write_text("{partial")
    got = read(tmp_path, API(), live=False)
    assert got[Tier.SUPER].source == "Token Factory model API, 2026-10-20 (cached; cache-only run)"
