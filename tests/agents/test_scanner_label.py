import json
from pathlib import Path

import pytest

from offramp.agents.scanner import ContractLabel, json_reply, label_contracts, scan_repo
from offramp.schemas import CallSite, Contract, Tier
from offramp.tf.client import TFClient
from offramp.tf.fake import FakeLLM

FIXTURES = Path(__file__).parents[1] / "fixtures"
REPO = FIXTURES / "scanner" / "repo"
# Call-site files of offramp-demo-app at 2e746d9, copied so the cassette keys (prompt = facts + code snippet)
# do not move when the demo repo changes.
HELPHUB = FIXTURES / "scanner" / "helphub"
CASSETTES = FIXTURES / "cassettes" / "scanner"
FREE_TEXT = Contract(task_type="free_text", output="text")


def fake(label: ContractLabel | str, n: int = 1) -> FakeLLM:
    return FakeLLM({"scanner": [label] * n})


@pytest.fixture(scope="module")
def sites():
    return scan_repo(REPO).sites


@pytest.fixture
def labelled(sites):
    llm = fake(ContractLabel(contract=FREE_TEXT, confidence=0.7), n=18)
    out = label_contracts(sites, llm, root=REPO)
    return llm, {s.id: s for s in out}, out


def test_one_super_call_per_chat_site(sites, labelled):
    llm, _, out = labelled
    chat = [s for s in sites if s.kind == "chat"]
    assert len(llm.calls) == len(chat) == 18
    for call in llm.calls:
        assert (call["role"], call["tier"], call["temperature"], call["reasoning"]) == (
            "scanner",
            Tier.SUPER,
            0.0,
            "off",
        )
        assert call["response_model"] is ContractLabel
        assert 0 < call["max_tokens"] <= 2048
    assert [s.id for s in out] == [s.id for s in sites]


def test_prompt_carries_snippet_and_params(labelled):
    llm, _, _ = labelled
    [user] = [
        c["messages"][-1]["content"]
        for c in llm.calls
        if '"file": "shopdesk/inbox/classify.py"' in c["messages"][-1]["content"]
    ]
    assert "def classify_ticket" in user and "client.chat.completions.create(" in user
    assert '"max_tokens": 5' in user and "Subject: {subject}" in user


def _site(**params) -> CallSite:
    return CallSite(
        id="cs_01",
        file="x.py",
        lines=[1, 1],
        sdk="openai",
        call="c.chat.completions.create",
        prompt_source="literal",
        params=params,
    )


@pytest.mark.parametrize(
    "params",
    [{"tools": "<expr>"}, {"response_format": "<expr>"}, {"response_format": {"type": "<expr>"}}],
    ids=["tools", "response_format", "response_format-type"],
)
def test_unresolved_params_do_not_force_a_contract(params):
    [out] = label_contracts([_site(**params)], fake(ContractLabel(contract=FREE_TEXT, confidence=0.4)))
    assert (out.contract, out.confidence) == (FREE_TEXT, 0.4)


SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}}}


@pytest.mark.parametrize(
    ("params", "contract", "confidence"),
    [
        ({"tools": [{"name": "t"}]}, Contract(task_type="tool_call", output="tool_call", tools=True), 1.0),
        (
            {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}}},
            Contract(task_type="json_extraction", output="json_schema", json_schema=SCHEMA),
            1.0,
        ),
        ({}, None, 0.0),
    ],
    ids=["tools", "json_schema", "no-hint"],
)
def test_invalid_reply_keeps_certain_hints_and_flags_manual_review(params, contract, confidence):
    [out] = label_contracts([_site(**params)], fake("not a contract"))
    assert (out.contract, out.confidence, out.flags) == (contract, confidence, ["MANUAL_REVIEW"])


def test_labeller_never_supplies_json_schema():
    label = ContractLabel(
        contract=Contract(task_type="json_extraction", output="json_schema", json_schema={"type": "object"}),
        confidence=0.9,
    )
    [out] = label_contracts([_site()], fake(label))
    assert (out.contract.task_type, out.contract.json_schema, out.confidence) == (
        "json_extraction",
        None,
        0.9,
    )


# The first live recording (before the labeller prompt stopped asking for json_schema) gave cs_03 a dangling
# {"$ref": "#/$defs/OrderFields"}; that real reply stays in the cassette directory as a regression input.
CS03_DANGLING_REF = CASSETTES / "be7a13287797e7eae4cc11a8b0f626f955126afbc1497b975c28f49370a3a139.json"


def test_recorded_dangling_ref_reply_loses_its_schema():
    reply = json.loads(CS03_DANGLING_REF.read_text())["response"]["choices"][0]["message"]["content"]
    assert json.loads(reply)["contract"]["json_schema"] == {"$ref": "#/$defs/OrderFields"}
    cs03 = next(s for s in scan_repo(HELPHUB).sites if s.id == "cs_03")
    [out] = label_contracts([cs03], fake(reply))
    assert (out.contract.task_type, out.contract.json_schema) == ("json_extraction", None)


def test_embeddings_are_not_labelled(labelled):
    _, by_id, _ = labelled
    assert by_id["emb_01"].contract is None


def test_llm_label_kept_when_no_deterministic_hint(labelled):
    _, by_id, _ = labelled
    assert (by_id["cs_04"].contract, by_id["cs_04"].confidence) == (FREE_TEXT, 0.7)


def test_tools_force_tool_call(labelled):
    _, by_id, _ = labelled
    c = by_id["cs_09"].contract
    assert (c.task_type, c.output, c.tools, by_id["cs_09"].confidence) == (
        "tool_call",
        "tool_call",
        True,
        1.0,
    )


def test_response_format_json_schema_forces_json_extraction(labelled):
    _, by_id, _ = labelled
    c = by_id["cs_11"].contract
    assert (c.task_type, c.output, by_id["cs_11"].confidence) == ("json_extraction", "json_schema", 1.0)
    assert c.json_schema is None  # pydantic marker, not a literal schema


def test_literal_json_schema_is_copied_and_stream_sets_streaming(labelled):
    _, by_id, _ = labelled
    assert by_id["cs_03"].contract.streaming is True
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}
    site = _site(text={"format": {"type": "json_schema", "name": "x", "schema": schema}})
    [out] = label_contracts([site], fake(ContractLabel(contract=FREE_TEXT, confidence=0.4)))
    assert (out.contract.task_type, out.contract.json_schema, out.confidence) == (
        "json_extraction",
        schema,
        1.0,
    )


# ---------------------------------------------------------------- Super on the demo repo (recorded cassette)

DEMO_TASK_TYPES = {
    "cs_01": "classification",
    "cs_02": "tool_call",
    "cs_03": "json_extraction",
    "cs_04": "free_text",
    "cs_05": "reasoning",
    "cs_06": "free_text",
}


def _label_helphub(live: bool) -> dict[str, CallSite]:
    scan = scan_repo(HELPHUB)
    return {
        s.id: s for s in label_contracts(scan.sites, TFClient(cache_dir=CASSETTES, live=live), root=scan.repo)
    }


@pytest.mark.live
def test_record_helphub_cassette():
    """Re-record: `uv run pytest --live -k record_helphub` with NEBIUS_API_KEY set (cache hits are free)."""
    by_id = _label_helphub(live=True)
    assert all(by_id[site_id].contract is not None for site_id in DEMO_TASK_TYPES)


def test_helphub_labels_replay_from_cassette():
    by_id = _label_helphub(live=False)
    assert {site_id: by_id[site_id].contract.task_type for site_id in DEMO_TASK_TYPES} == DEMO_TASK_TYPES
    assert by_id["cs_03"].contract.json_schema is None  # set later from the resolved Pydantic model (Task 7)


JSON_REPLY = FIXTURES / "scanner" / "json_reply"


def test_a_reply_parsed_with_json_loads_is_recorded_on_the_contract():
    """idea-reality-mcp's cs_01, cs_02 and cs_04 run json.loads on the reply, but they were judged as free
    text, so a reply the code cannot parse showed only in a rubric. Reading each site's function now records
    it on the contract: "fence_stripped" where the reply's own flow strips a ``` fence first, else "exact".
    Conservative (task 28 review, I2): only json.loads of the reply text itself (stripped or unfenced) counts,
    and not where the code falls back to the text; a missed site is still judged, a wrong one fails good
    replies."""
    scan = scan_repo(JSON_REPLY)
    chat = [s for s in scan.sites if s.kind == "chat"]
    llm = fake(ContractLabel(contract=FREE_TEXT, confidence=0.9), n=len(chat))
    found = {s.function: s.contract.reply_json for s in label_contracts(scan.sites, llm, root=JSON_REPLY)}
    assert found == {
        "extract_keywords": "fence_stripped",  # cs_01
        "pivot_hints": "fence_stripped",  # cs_02
        "search_angles": "fence_stripped",  # cs_04
        "tags_direct": "exact",
        "tags_through_a_helper": "exact",
        "summary": None,  # its json.loads never sees the reply
        "lookup_order": None,  # tool-call arguments are not the reply text
        "fallback_to_text": None,  # the except branch returns the text: the code accepts prose
        "regex_extracts_object": None,  # json.loads of a regex match out of the reply
        "slices_from_brace": None,  # json.loads of a slice of the reply
        "logs_the_response_object": None,  # json.loads of the response object's own JSON
        "wraps_then_loads": None,  # json.loads of a document built around the reply
        "reply_as_lookup_key": None,  # the reply (lower-cased) is only a key
        "summarise": None,  # an attribute store of the reply; json.loads reads settings
        "call_in_try_parse_after": "exact",  # the call sits deeper in the tree than the parse
        "strip_in_its_own_statement": "exact",
        "fence_in_prompt_only": "exact",  # the ``` is in the prompt, not a step on the reply
    }
    # the labeller is never asked for it: its request schema is what the cassettes recorded
    assert all("reply_json" not in json.dumps(c["response_model"].model_json_schema()) for c in llm.calls)


def test_tool_call_arguments_are_never_the_reply_text():
    """lookup_order's tools= settles it as a tool call before json_reply is asked; read directly, its
    json.loads of the call's arguments is still not a json.loads of the reply."""
    (site,) = [s for s in scan_repo(JSON_REPLY).sites if s.function == "lookup_order"]
    assert json_reply(JSON_REPLY, site) is None
