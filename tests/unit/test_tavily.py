"""Tavily source-model lookup (PRD §6.10, GC21) against httpx.MockTransport and FakeLLM: no network, no key.

The pages in SYNTHETIC_* are made-up test fixtures (they only look like vendor pages). The recorded real
responses live in tests/fixtures/tavily/ (Tavily, one search per model) and tests/fixtures/tavily/cassettes/
(Nemotron Super's extraction replies) and are replayed further down.

Re-record (spends one basic Tavily search per deleted <model>.json, plus a few cents of Super tokens):
delete the files to refresh, then `pytest tests/unit/test_tavily.py --live -k record` with NEBIUS_API_KEY and
TAVILY_API_KEY set. Changing the search query, the Super prompt or the schema invalidates the cassettes
(delete tests/fixtures/tavily/cassettes/ and re-run: the search fixtures are replayed, not re-fetched).
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from offramp import pricing
from offramp.integrations import keys, tavily
from offramp.integrations.tavily import PricingExtraction, SourceModelInfo, TavilyError, lookup_source_model
from offramp.schemas import Tier
from offramp.tf import BudgetExceeded, CacheMiss, FakeLLM, TFClient

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tavily"
CASSETTES = FIXTURES / "cassettes"  # the real Super extraction replies, replayed by TFClient(live=False)
MODELS = ["gpt-4o-mini", "gpt-4o", "o3", "claude-sonnet-4-5", "claude-haiku-4-5"]
LIVE_TAVILY_KEY = os.environ.get("TAVILY_API_KEY")  # read before hermetic_env replaces it
NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
MINI_URL = "https://developers.openai.com/api/docs/models/gpt-4o-mini"
O3_URL = "https://platform.openai.com/docs/models/o3"
CLAUDE_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
SYNTHETIC_MINI = "gpt-4o-mini  Input: $0.15 / 1M tokens  Output: $0.60 / 1M tokens  Cached input: $0.075"
SYNTHETIC_DEPRECATION = (
    "| claude-sonnet-4-5 | Deprecated | 2026-10-01 | Retired 2027-04-01 | claude-sonnet-4-6 |\n"
    "| claude-3-haiku | Deprecated | 2026-08-01 | Retired 2027-02-01 | claude-haiku-4-5 |"
)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch, tmp_path):
    for name in ("TAVILY_API_KEY_FALLBACK", "OFFRAMP_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-primary")
    monkeypatch.setattr(keys, "_exhausted", set())
    monkeypatch.setattr(tavily.time, "sleep", lambda s: None)  # the retry backoff


class Tavily:
    """Scripted Tavily: each reply is a response body (dict), an HTTP status (int), (status, body text) or an
    exception to raise. Records every request (JSON body + bearer token)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.tokens: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.tavily.com/search" and request.method == "POST"
        self.requests.append(json.loads(request.content))
        self.tokens.append(request.headers["authorization"].removeprefix("Bearer "))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple):
            return httpx.Response(reply[0], text=reply[1])
        if isinstance(reply, int):
            return httpx.Response(reply, json={"detail": {"error": "scripted failure"}})
        return httpx.Response(200, json=reply)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def body(*pages, answer="summary"):
    return {
        "answer": answer,
        "results": [{"url": u, "title": "t", "content": c, "score": 0.9} for u, c in pages],
    }


def mini(**over) -> PricingExtraction:
    fields = dict(input_usd_per_m=0.15, output_usd_per_m=0.60, price_source_url=MINI_URL)
    return PricingExtraction(**(fields | over))


def lookup(tmp_path, server, extraction, model="gpt-4o-mini", **kw):
    llm = kw.pop("llm", None) or FakeLLM({tavily.ROLE: [extraction]})
    info = lookup_source_model(
        model, llm=llm, http=server.client(), cache_dir=tmp_path / "cache", now=NOW, **kw
    )
    return info, llm


# ------------------------------------------------------------------ the happy path


def test_search_request_is_a_basic_answer_search_restricted_to_the_vendors_official_domains(tmp_path):
    server = Tavily(body((MINI_URL, SYNTHETIC_MINI)))
    lookup(tmp_path, server, mini())
    (request,) = server.requests
    assert server.tokens == ["tvly-primary"]
    assert request["search_depth"] == "basic" and request["include_answer"] is True
    assert request["include_domains"] == tavily.OPENAI_DOMAINS
    assert request["exclude_domains"] == ["community.openai.com"]  # include_domains also matches subdomains
    assert "exclude_domains" not in tavily._search_payload("claude-haiku-4-5")  # nothing to exclude there
    assert "developers.openai.com" in request["include_domains"]
    assert "gpt-4o-mini" in request["query"] and "deprecation" in request["query"]
    assert "api_key" not in request  # the key travels in the Authorization header only


@pytest.mark.parametrize(
    "model,domains",
    [
        ("o3", tavily.OPENAI_DOMAINS),
        ("openai/gpt-4o", tavily.OPENAI_DOMAINS),
        ("claude-haiku-4-5", tavily.ANTHROPIC_DOMAINS),
        ("some-other-vendor-model", tavily.OPENAI_DOMAINS + tavily.ANTHROPIC_DOMAINS),
    ],
)
def test_allow_list_follows_the_models_vendor(model, domains):
    assert tavily._search_payload(model.rsplit("/", 1)[-1])["include_domains"] == domains
    assert {"docs.claude.com", "claude.com", "anthropic.com", "docs.anthropic.com"} <= set(
        tavily.ANTHROPIC_DOMAINS
    )
    assert {"openai.com", "platform.openai.com", "developers.openai.com"} <= set(tavily.OPENAI_DOMAINS)


def test_a_verified_extraction_becomes_the_source_model_info(tmp_path):
    server = Tavily(body((MINI_URL, SYNTHETIC_MINI), ("https://openai.com/other", "unrelated $99.00")))
    info, llm = lookup(tmp_path, server, mini())
    assert info == SourceModelInfo(
        model="gpt-4o-mini",
        input_usd_per_m=0.15,
        output_usd_per_m=0.60,
        price_source_url=MINI_URL,
        fetched_at=NOW,
    )
    assert info.via == "tavily" and info.note is None and info.sunset_risk() is None
    assert info.cost_before_note() == (
        "est.: gpt-4o-mini $0.15/$0.60 per 1M (developers.openai.com/api/docs/models/gpt-4o-mini, "
        "fetched via Tavily 2026-10-19) x tokens measured on the Token Factory call"
    )


def test_extraction_is_a_super_call_with_thinking_off_and_the_pages_in_the_user_message(tmp_path):
    server = Tavily(body((MINI_URL, SYNTHETIC_MINI), answer="Tavily says $0.15 and $0.60"))
    _, llm = lookup(tmp_path, server, mini())
    (call,) = llm.calls
    assert call["role"] == "pricing_extract" and call["tier"] == Tier.SUPER and call["reasoning"] == "off"
    assert (
        call["response_model"] is PricingExtraction
        and call["temperature"] == 0
        and call["max_tokens"] <= 1000
    )
    system, user = (m["content"] for m in call["messages"])
    assert "untrusted" in system and SYNTHETIC_MINI not in system
    assert MINI_URL in user and SYNTHETIC_MINI in user
    assert "Tavily says" not in user  # the answer is requested, but Super only reads the pages


def test_provider_prefix_is_dropped_from_the_model_id(tmp_path):
    server = Tavily(body((MINI_URL, SYNTHETIC_MINI)))
    info, _ = lookup(tmp_path, server, mini(), model="openai/gpt-4o-mini")
    assert info.model == "gpt-4o-mini" and "openai/" not in server.requests[0]["query"]


def test_results_off_the_vendors_domains_are_ignored_even_if_tavily_returned_them(tmp_path):
    server = Tavily(
        body(("https://evil.example.com/gpt-4o-mini", "gpt-4o-mini $0.01 $0.02"), (MINI_URL, SYNTHETIC_MINI))
    )
    info, llm = lookup(
        tmp_path,
        server,
        mini(
            price_source_url="https://evil.example.com/gpt-4o-mini",
            input_usd_per_m=0.01,
            output_usd_per_m=0.02,
        ),
    )
    assert info.via == "pricing.yaml"  # the cited page was not an allowed result
    assert "evil.example.com" not in llm.calls[0]["messages"][1]["content"]


# ------------------------------------------------------------------ numbers must be on the cited page


@pytest.mark.parametrize(
    "over,reason",
    [
        (dict(output_usd_per_m=0.66), "prices 0.15/0.66 are not printed next to the model's name"),
        (dict(input_usd_per_m=0.10), "prices 0.1/0.6 are not printed next to the model's name"),
        (dict(price_source_url="https://openai.com/never-returned"), "not one of the search results"),
        (dict(price_source_url=None), "not one of the search results"),
        (dict(input_usd_per_m=None), "no plausible input price"),
        (dict(output_usd_per_m=0.0), "no plausible output price"),
    ],
)
def test_an_unverifiable_extraction_is_rejected_and_pricing_yaml_is_used_with_the_reason(
    tmp_path, over, reason
):
    server = Tavily(body((MINI_URL, SYNTHETIC_MINI)))
    info, _ = lookup(tmp_path, server, mini(**over))
    assert (info.via, info.input_usd_per_m, info.output_usd_per_m) == ("pricing.yaml", 0.15, 0.60)
    assert reason in info.note and info.note.startswith("fallback: Tavily price not used")
    assert info.price_source_url == "https://developers.openai.com/api/docs/pricing"
    assert info.fetched_at == datetime(2026, 10, 19, tzinfo=UTC)
    assert info.cost_before_note().startswith(
        "est.: gpt-4o-mini $0.15/$0.60 per 1M (developers.openai.com/api/docs/pricing, "
        "pricing.yaml as of 2026-10-19; fallback:"
    )


def test_a_number_on_a_different_result_than_the_cited_one_does_not_count(tmp_path):
    server = Tavily(body((MINI_URL, "gpt-4o-mini Input: $0.15"), (O3_URL, "Output: $0.60 per 1M")))
    info, _ = lookup(tmp_path, server, mini())
    assert info.via == "pricing.yaml" and "not printed next to the model's name" in info.note


def test_a_page_that_prints_the_numbers_but_not_next_to_this_model_does_not_count(tmp_path):
    # Seen in the real claude-sonnet-4-5 search: Sonnet 5's pages mention 4.5 yet print Sonnet 5's price.
    page = (
        "Migrate from Claude Sonnet 4.5 ... "
        + "x" * 600
        + " Input pricing: $2 / MTok Output pricing: $10 / MTok"
    )
    info, _ = lookup(
        tmp_path,
        Tavily(body((CLAUDE_URL, page))),
        PricingExtraction(input_usd_per_m=2.0, output_usd_per_m=10.0, price_source_url=CLAUDE_URL),
        model="claude-sonnet-4-5",
    )
    assert info.via == "pricing.yaml" and "not printed next to the model's name" in info.note
    assert (info.input_usd_per_m, info.output_usd_per_m) == (3.0, 15.0)


def test_a_cited_page_that_never_names_the_model_is_rejected(tmp_path):
    info, _ = lookup(tmp_path, Tavily(body((MINI_URL, "Input: $0.15  Output: $0.60"))), mini())
    assert info.via == "pricing.yaml" and "the cited page does not name the model" in info.note


@pytest.mark.parametrize(
    "model,page",
    [
        (
            "claude-sonnet-4-5",
            "| Claude Sonnet 4.5 | $3 / MTok | $3.75 / MTok | $6 / MTok | $0.30 / MTok | $15 / MTok |",
        ),
        ("claude-sonnet-4-5", "| claude-sonnet-4-5-20250929 | $3 / MTok | $15 / MTok |"),
        ("claude-sonnet-4-5-20250929", "| Claude Sonnet 4.5 | $3 / MTok | $15 / MTok |"),
        ("gpt-4o-mini", "GPT-4o mini ... Input $3 Output $15"),
        ("o3", "| o3-mini | $0.50 | $9 |\n| o3 | $3 | $15 |"),
        (
            "claude-haiku-4-5",
            "| Claude Haiku 4.5This modelLegacy | 200K | 64K | $3 / $15 | Extended |",
        ),  # real page
    ],
)
def test_the_model_is_found_however_the_page_spells_it(tmp_path, model, page):
    url = CLAUDE_URL if model.startswith("claude") else O3_URL
    ext = PricingExtraction(input_usd_per_m=3.0, output_usd_per_m=15.0, price_source_url=url)
    info, _ = lookup(tmp_path, Tavily(body((url, page))), ext, model=model)
    assert info.via == "tavily"


@pytest.mark.parametrize(
    "model,page",
    [
        ("claude-sonnet-4", "| Claude Sonnet 4.5 | $3 / MTok | $15 / MTok |"),
        ("claude-sonnet-5", "| Claude Sonnet 5.5 | $3 / MTok | $15 / MTok |"),
        ("claude-sonnet-4-5", "| Claude Sonnet 4 | $3 / MTok | $15 / MTok |"),
        ("o3", "| o3-mini | $3 | $15 |"),
        ("gpt-4o", "| gpt-4o-mini | $3 | $15 |"),
        ("gpt-4o-mini", "| GPT-4o mini Realtime (per 1M tokens) | $0.60 | $2.40 |"),
        ("gpt-4o", "| GPT-4o mini | $3 | $15 |"),
        ("claude-haiku-4-5", "Claude Haiku 4.5 Thinking: $1 / MTok, $5 / MTok"),
        ("gpt-4o", "| gpt-4o-2024-08-06x | $3 | $15 |"),
    ],
)
def test_a_sibling_model_is_not_this_model(model, page):
    assert tavily._mentions(page, model) == []


def test_user_written_forum_pages_are_ignored_even_if_tavily_returns_them(tmp_path):
    # Observed: the real gpt-4o-mini search returned community.openai.com posts, one with the wrong price.
    forum = "https://community.openai.com/t/gpt-realtime-mini-pricing/1372904"
    server = Tavily(body((forum, "GPT-4o mini $0.60 $2.40"), (MINI_URL, SYNTHETIC_MINI)))
    info, llm = lookup(
        tmp_path, server, mini(price_source_url=forum, input_usd_per_m=0.6, output_usd_per_m=2.4)
    )
    assert info.via == "pricing.yaml" and "community.openai.com" not in llm.calls[0]["messages"][1]["content"]


def test_result_urls_with_trailing_spaces_still_match_the_citation(tmp_path):
    server = Tavily(body((MINI_URL + "  ", SYNTHETIC_MINI)))  # seen in a real result: "…/models/gpt-4o "
    info, _ = lookup(tmp_path, server, mini())
    assert info.via == "tavily" and info.price_source_url == MINI_URL


def test_the_vendor_docs_return_one_page_under_many_urls_and_super_sees_it_once(tmp_path):
    twin = "https://platform.openai.com/docs/models/gpt-4o-mini"
    server = Tavily(body((MINI_URL, SYNTHETIC_MINI), (twin, SYNTHETIC_MINI), (O3_URL, "o3 $2 $8")))
    _, llm = lookup(tmp_path, server, mini())
    shown = llm.calls[0]["messages"][1]["content"]
    assert shown.count(SYNTHETIC_MINI) == 1 and MINI_URL in shown and twin not in shown and O3_URL in shown


def test_numbers_match_regardless_of_formatting(tmp_path):
    server = Tavily(body((O3_URL, "o3 pricing — input $2, output $8.00 per 1M tokens; 1,000 free")))
    info, _ = lookup(
        tmp_path,
        server,
        PricingExtraction(input_usd_per_m=2.0, output_usd_per_m=8.0, price_source_url=O3_URL + "/"),
        model="o3",
    )
    assert info.via == "tavily" and (info.input_usd_per_m, info.output_usd_per_m) == (2.0, 8.0)


def test_pricing_yaml_fallback_for_an_unknown_model_is_a_keyerror_naming_both_failures(tmp_path):
    server = Tavily(500, 500, 500)
    with pytest.raises(KeyError, match="no price for model 'mystery-1'.*Tavily: Tavily price not used"):
        lookup(tmp_path, server, mini(), model="mystery-1")


# ------------------------------------------------------------------ deprecation notices


DEP = dict(
    deprecation_quote="| claude-sonnet-4-5 | Deprecated | 2026-10-01 | Retired 2027-04-01 |",
    deprecation_url="https://docs.claude.com/en/docs/about-claude/model-deprecations",
)
SONNET_PRICE = dict(
    input_usd_per_m=3.0, output_usd_per_m=15.0, price_source_url="https://docs.claude.com/pricing"
)


def sonnet_body():
    return body(
        ("https://docs.claude.com/pricing", "claude-sonnet-4-5 $3 input $15 output"),
        (DEP["deprecation_url"], SYNTHETIC_DEPRECATION),
    )


def test_a_verbatim_deprecation_notice_becomes_the_sunset_risk(tmp_path):
    info, _ = lookup(
        tmp_path, Tavily(sonnet_body()), PricingExtraction(**SONNET_PRICE, **DEP), model="claude-sonnet-4-5"
    )
    assert info.deprecation == DEP["deprecation_quote"] and info.deprecation_url == DEP["deprecation_url"]
    assert info.sunset_risk() == f"{DEP['deprecation_quote']} ({DEP['deprecation_url']})"


@pytest.mark.parametrize(
    "over",
    [
        dict(deprecation_quote="claude-sonnet-4-5 will be retired next week"),  # not on the page
        dict(deprecation_url="https://docs.claude.com/never-returned"),  # not a result
        dict(
            deprecation_quote="| claude-3-haiku | Deprecated | 2026-08-01 | Retired 2027-02-01 |"
        ),  # other model
        dict(deprecation_url=None),
        dict(deprecation_quote=None),
        dict(deprecation_quote="", deprecation_url=None),
        dict(deprecation_quote="claude-sonnet-4-5"),  # a model name is not a notice
        dict(deprecation_quote="| claude-sonnet-4-5 | Active | N/A |"),  # not on the page either
    ],
)
def test_an_unverifiable_deprecation_notice_is_dropped_but_the_verified_price_stays(tmp_path, over):
    info, _ = lookup(
        tmp_path,
        Tavily(sonnet_body()),
        PricingExtraction(**SONNET_PRICE, **(DEP | over)),
        model="claude-sonnet-4-5",
    )
    assert info.via == "tavily" and info.input_usd_per_m == 3.0
    assert info.deprecation is None and info.deprecation_url is None and info.sunset_risk() is None


def test_o3_is_not_flagged_for_a_notice_about_o3_mini(tmp_path):
    page = "| o3-mini | Deprecated | 2026-09-01 | Shutdown 2027-01-01 |"
    server = Tavily(body((O3_URL, "o3 Input $2.00 Output $8.00 " + page)))
    ext = PricingExtraction(
        input_usd_per_m=2.0,
        output_usd_per_m=8.0,
        price_source_url=O3_URL,
        deprecation_quote=page,
        deprecation_url=O3_URL,
    )
    info, _ = lookup(tmp_path, server, ext, model="o3")
    assert info.via == "tavily" and info.deprecation is None


def test_a_verified_notice_survives_a_rejected_price(tmp_path):
    info, _ = lookup(
        tmp_path,
        Tavily(sonnet_body()),
        PricingExtraction(**(SONNET_PRICE | dict(input_usd_per_m=9.0)), **DEP),
        model="claude-sonnet-4-5",
    )
    assert info.via == "pricing.yaml" and (info.input_usd_per_m, info.output_usd_per_m) == (3.0, 15.0)
    assert info.deprecation == DEP["deprecation_quote"]


# ------------------------------------------------------------------ disk cache per (model, day)


def test_a_tavily_result_is_cached_per_model_and_day(tmp_path):
    server = Tavily(body((MINI_URL, SYNTHETIC_MINI)), body((MINI_URL, SYNTHETIC_MINI)))
    first, llm = lookup(tmp_path, server, mini())
    again = lookup_source_model(
        "gpt-4o-mini", llm=FakeLLM(), http=server.client(), cache_dir=tmp_path / "cache", now=NOW
    )
    assert (
        again == first and len(server.requests) == 1
    )  # no second search, and FakeLLM() has no reply to give

    later = datetime(2026, 10, 20, 0, 5, tzinfo=UTC)
    llm2 = FakeLLM({tavily.ROLE: [mini()]})
    next_day = lookup_source_model(
        "gpt-4o-mini", llm=llm2, http=server.client(), cache_dir=tmp_path / "cache", now=later
    )
    assert len(server.requests) == 2 and next_day.fetched_at == later
    assert sorted(p.name for p in (tmp_path / "cache").glob("*.json")) == [
        "gpt-4o-mini-2026-10-19.json",
        "gpt-4o-mini-2026-10-20.json",
    ]


def test_a_fallback_result_is_not_cached_so_the_next_call_tries_tavily_again(tmp_path):
    server = Tavily(body((MINI_URL, SYNTHETIC_MINI)), body((MINI_URL, SYNTHETIC_MINI)))
    first, _ = lookup(tmp_path, server, mini(output_usd_per_m=0.66))
    assert first.via == "pricing.yaml" and not list((tmp_path / "cache").glob("*.json"))
    second, _ = lookup(tmp_path, server, mini())
    assert second.via == "tavily" and len(server.requests) == 2


def test_default_cache_dir_is_under_offramp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    server = Tavily(body((MINI_URL, SYNTHETIC_MINI)))
    lookup_source_model("gpt-4o-mini", llm=FakeLLM({tavily.ROLE: [mini()]}), http=server.client(), now=NOW)
    assert (tmp_path / "data" / "tavily-cache" / "gpt-4o-mini-2026-10-19.json").exists()


# ------------------------------------------------------------------ when Tavily or Super is unavailable


def test_without_a_tavily_key_pricing_yaml_is_used_and_nothing_is_sent(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY")
    server = Tavily()
    llm = FakeLLM()
    info = lookup_source_model("gpt-4o", llm=llm, http=server.client(), cache_dir=tmp_path, now=NOW)
    assert (info.via, info.input_usd_per_m, info.output_usd_per_m) == ("pricing.yaml", 2.50, 10.00)
    assert "TAVILY_API_KEY is not set" in info.note and server.requests == [] and llm.calls == []


@pytest.mark.parametrize("failures", [[500, 502, 503], [httpx.ConnectTimeout("x")] * 3, [429, 429, 429]])
def test_transient_failures_back_off_three_times_then_fall_back(tmp_path, failures):
    server = Tavily(*failures)
    info, llm = lookup(tmp_path, server, mini())
    assert info.via == "pricing.yaml" and len(server.requests) == 3 and llm.calls == []


def test_a_plain_rate_limit_backs_off_and_retries_on_the_same_key(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(tavily.time, "sleep", sleeps.append)
    monkeypatch.setenv("TAVILY_API_KEY_FALLBACK", "tvly-fallback")
    server = Tavily((429, "Too many requests, slow down"), body((MINI_URL, SYNTHETIC_MINI)))
    info, _ = lookup(tmp_path, server, mini())
    assert info.via == "tavily" and server.tokens == ["tvly-primary"] * 2 and sleeps == [0.5]


def test_a_client_error_is_not_retried(tmp_path):
    server = Tavily(400, body((MINI_URL, SYNTHETIC_MINI)))
    info, _ = lookup(tmp_path, server, mini())
    assert info.via == "pricing.yaml" and "HTTP 400" in info.note and len(server.requests) == 1


@pytest.mark.parametrize(
    "error",
    [
        CacheMiss("no cached response"),
        BudgetExceeded("token cap reached"),
        RuntimeError("NEBIUS_API_KEY is not set"),
    ],
)
def test_super_being_unavailable_falls_back_instead_of_crashing_the_run(tmp_path, error):
    info, _ = lookup(
        tmp_path, Tavily(body((MINI_URL, SYNTHETIC_MINI))), None, llm=FakeLLM({tavily.ROLE: [error]})
    )
    assert info.via == "pricing.yaml" and type(error).__name__ in info.note


def test_a_super_reply_that_is_not_valid_json_falls_back(tmp_path):
    info, _ = lookup(
        tmp_path, Tavily(body((MINI_URL, SYNTHETIC_MINI))), None, llm=FakeLLM({tavily.ROLE: ["not json"]})
    )
    assert info.via == "pricing.yaml" and "not valid JSON" in info.note


def test_tavily_error_never_mentions_the_key(tmp_path, caplog):
    server = Tavily(401)
    with caplog.at_level(logging.INFO), pytest.raises(TavilyError) as err:
        tavily._post(server.client(), {"query": "q"})
    assert str(err.value) == "HTTP 401" and "tvly-primary" not in caplog.text


# ------------------------------------------------------------------ fallback key (Global Constraint 28)


class KeyedTavily(Tavily):
    """Primary key is out of credit (`primary_status`), the fallback key works."""

    def __init__(self, primary_status, primary_text, *ok):
        super().__init__(*ok)
        self.primary = (primary_status, primary_text)

    def __call__(self, request):
        if request.headers["authorization"] == "Bearer tvly-primary":
            self.tokens.append("tvly-primary")
            return httpx.Response(self.primary[0], text=self.primary[1])
        return super().__call__(request)


@pytest.mark.parametrize(
    "status,text",
    [
        (401, "Unauthorized"),
        (402, "pay"),
        (403, "Forbidden"),
        (429, "Your quota is exceeded"),
        (432, "plan limit exceeded"),
        (433, "pay-as-you-go limit exceeded"),
    ],
)
def test_an_exhausted_primary_key_retries_once_on_the_fallback_for_the_rest_of_the_process(
    tmp_path, monkeypatch, caplog, status, text
):
    monkeypatch.setenv("TAVILY_API_KEY_FALLBACK", "tvly-fallback")
    server = KeyedTavily(status, text, body((MINI_URL, SYNTHETIC_MINI)), body((O3_URL, "o3 $2.00 $8.00")))
    with caplog.at_level(logging.INFO):
        info, _ = lookup(tmp_path, server, mini())
    assert info.via == "tavily" and server.tokens == ["tvly-primary", "tvly-fallback"]
    assert [r.getMessage() for r in caplog.records if r.name == keys.log.name] == [
        "tavily: primary key exhausted, using fallback"
    ]
    assert "tvly-" not in caplog.text

    info, _ = lookup(
        tmp_path,
        server,
        PricingExtraction(input_usd_per_m=2.0, output_usd_per_m=8.0, price_source_url=O3_URL),
        model="o3",
    )
    assert info.via == "tavily" and server.tokens[2:] == [
        "tvly-fallback"
    ]  # no failing request first this time


def test_without_a_fallback_an_exhausted_key_falls_back_to_pricing_yaml(tmp_path):
    server = KeyedTavily(401, "Unauthorized")
    info, _ = lookup(tmp_path, server, mini())
    assert info.via == "pricing.yaml" and "HTTP 401" in info.note and server.tokens == ["tvly-primary"]


def test_a_rejected_fallback_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY_FALLBACK", "tvly-fallback")
    server = Tavily(401, 401, body((MINI_URL, SYNTHETIC_MINI)))
    info, _ = lookup(tmp_path, server, mini())
    assert info.via == "pricing.yaml" and server.tokens == ["tvly-primary", "tvly-fallback"]


# ------------------------------------------------------------------ recorded real responses (replayed)


class Recorded(httpx.BaseTransport):
    """Replays tests/fixtures/tavily/<model>.json. With `record` (--live) it forwards to the real Tavily
    instead and saves the exchange: one basic search per model, never repeated while the fixture exists."""

    def __init__(self, model: str, record: bool = False):
        self.path, self.record, self.sent = FIXTURES / f"{model}.json", record, []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        self.sent.append(sent)
        if self.record and not self.path.exists():
            real = httpx.HTTPTransport().handle_request(request)
            real.read()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "recorded_at": datetime.now(UTC).isoformat(),
                        "status": real.status_code,
                        "request": sent,
                        "response": real.json(),
                    },
                    indent=2,
                )
                + "\n"
            )
            return real
        fixture = json.loads(self.path.read_text())
        assert sent == fixture["request"], (
            f"the search request changed: re-record {self.path.name} with --live"
        )
        return httpx.Response(fixture["status"], json=fixture["response"])


def replay(model: str, tmp_path, **kw) -> SourceModelInfo:
    """The recorded Tavily response + the recorded Super reply (cassette), through the production path."""
    return lookup_source_model(
        model,
        llm=TFClient(cache_dir=CASSETTES, live=False),
        http=httpx.Client(transport=Recorded(model)),
        cache_dir=tmp_path,
        now=NOW,
        **kw,
    )


# Recorded 2026-10-19: 7 basic searches in total (5 kept; 2 discarded while the query and the allow-list were
# tuned against real results). What came back for real:
VERIFIED_LIVE = ["gpt-4o-mini", "claude-sonnet-4-5", "claude-haiku-4-5"]
FELL_BACK = [
    "gpt-4o",
    "o3",
]  # Tavily returned no page printing this model's own price; Super answered from memory


@pytest.mark.parametrize("model", VERIFIED_LIVE)
def test_recorded_search_gives_a_verified_price_that_matches_the_vendors_published_table(model, tmp_path):
    info = replay(model, tmp_path)
    assert info.via == "tavily" and info.note is None
    assert (info.input_usd_per_m, info.output_usd_per_m) == pricing.price(
        model
    )  # = pricing.yaml, read by hand
    assert tavily._allowed(info.price_source_url, tavily._vendor_domains(model))
    assert (
        info.cost_before_note().startswith(f"est.: {model} ")
        and "fetched via Tavily 2026-10-19" in info.cost_before_note()
    )


def test_recorded_sonnet_search_found_the_vendors_deprecation_notice(tmp_path):
    info = replay("claude-sonnet-4-5", tmp_path)
    assert info.deprecation == (
        "Claude Sonnet 4.5 is deprecated: it is still functional but no longer recommended, "
        "and retires on November 30, 2026."
    )
    assert info.deprecation_url == "https://platform.claude.com/docs/en/models/sonnet-4-5"
    assert info.sunset_risk() == f"{info.deprecation} ({info.deprecation_url})"


@pytest.mark.parametrize("model", ["gpt-4o-mini", "claude-haiku-4-5"])
def test_recorded_searches_with_no_notice_have_no_sunset_risk(model, tmp_path):
    assert replay(model, tmp_path).sunset_risk() is None


@pytest.mark.parametrize("model", FELL_BACK)
def test_recorded_searches_where_super_answered_from_memory_are_rejected_and_fall_back(model, tmp_path):
    info = replay(model, tmp_path)
    assert info.via == "pricing.yaml" and info.note.startswith("fallback: Tavily price not used (")
    assert (info.input_usd_per_m, info.output_usd_per_m) == pricing.price(model)
    assert "pricing.yaml as of 2026-10-19" in info.cost_before_note()


def test_recorded_o3_search_still_found_its_deprecation_notice_while_the_price_fell_back(tmp_path):
    info = replay("o3", tmp_path)
    assert info.via == "pricing.yaml"
    assert (
        info.deprecation == "o3 and o4-mini are no longer available in ChatGPT and are deprecated in the API."
    )
    assert info.deprecation_url.startswith("https://openai.com/index/introducing-o3-and-o4-mini")


def test_recorded_requests_are_what_the_code_sends_today():
    for model in MODELS:
        assert json.loads((FIXTURES / f"{model}.json").read_text())["request"] == tavily._search_payload(
            model
        )


def test_fixtures_hold_no_api_key_and_are_marked_as_recordings():
    for path in [*FIXTURES.glob("*.json"), *CASSETTES.glob("*.json")]:
        text = path.read_text()
        assert "tvly-" not in text and "Bearer" not in text and "api_key" not in text.lower(), path.name
    assert {p.stem for p in FIXTURES.glob("*.json")} == set(MODELS)
    assert all("recorded_at" in json.loads(p.read_text()) for p in FIXTURES.glob("*.json"))


@pytest.mark.live
@pytest.mark.parametrize("model", MODELS)
def test_record_the_tavily_fixture_and_the_super_cassette(model, monkeypatch, tmp_path):
    monkeypatch.setenv("TAVILY_API_KEY", LIVE_TAVILY_KEY)
    http = httpx.Client(transport=Recorded(model, record=True))
    info = lookup_source_model(
        model, llm=TFClient(cache_dir=CASSETTES, live=True), http=http, cache_dir=tmp_path
    )
    print(info.model_dump_json(indent=2), info.cost_before_note(), info.sunset_risk(), sep="\n")
