from pathlib import Path
from types import SimpleNamespace

import pytest

from offramp.agents.scanner import ContractLabel, label_contracts, scan_repo
from offramp.schemas import CallSite, Contract, Tier

REPO = Path(__file__).parents[1] / "fixtures" / "scanner" / "repo"
FREE_TEXT = Contract(task_type="free_text", output="text")


class FakeLLM:
    """Minimal stand-in for Task 1's LLM protocol (controller swaps in offramp.tf.fake.FakeLLM at merge)."""

    def __init__(self, label: ContractLabel):
        self.label = label
        self.calls: list[dict] = []

    def chat(
        self,
        *,
        role,
        tier,
        messages,
        max_tokens,
        temperature=0.0,
        response_model=None,
        json_schema=None,
        tools=None,
        tool_choice=None,
        reasoning="off",
        extra_body=None,
    ):
        self.calls.append(
            dict(
                role=role,
                tier=tier,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_model=response_model,
            )
        )
        return SimpleNamespace(parsed=self.label, cached=False)


@pytest.fixture(scope="module")
def sites():
    return scan_repo(REPO).sites


@pytest.fixture
def labelled(sites):
    llm = FakeLLM(ContractLabel(contract=FREE_TEXT, confidence=0.7))
    out = label_contracts(sites, llm, root=REPO)
    return llm, {s.id: s for s in out}, out


def test_one_super_call_per_chat_site(sites, labelled):
    llm, _, out = labelled
    chat = [s for s in sites if s.kind == "chat"]
    assert len(llm.calls) == len(chat) == 14
    for call in llm.calls:
        assert (call["role"], call["tier"], call["temperature"]) == ("scanner", Tier.SUPER, 0.0)
        assert call["response_model"] is ContractLabel
        assert 0 < call["max_tokens"] <= 2048
    assert [s.id for s in out] == [s.id for s in sites]


def test_prompt_carries_snippet_and_params(labelled):
    llm, _, _ = labelled
    user = llm.calls[3]["messages"][-1]["content"]  # cs_04, classify.py
    assert "def classify_ticket" in user and "client.chat.completions.create(" in user
    assert '"max_tokens": 5' in user and "Subject: {subject}" in user


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
    site = CallSite(
        id="cs_01",
        file="x.py",
        lines=[1, 1],
        sdk="openai",
        call="c.responses.create",
        prompt_source="literal",
        params={"text": {"format": {"type": "json_schema", "name": "x", "schema": schema}}},
    )
    [out] = label_contracts([site], FakeLLM(ContractLabel(contract=FREE_TEXT, confidence=0.4)))
    assert (out.contract.task_type, out.contract.json_schema, out.confidence) == (
        "json_extraction",
        schema,
        1.0,
    )
