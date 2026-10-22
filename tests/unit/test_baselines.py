"""Baselines I/O (evals/references, legacy evals/baselines, evals/closed), manifest, closed-model recorder."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from offramp.evals import baselines as b
from offramp.schemas import (
    BaselineResult,
    CallSite,
    Contract,
    EvalCase,
    ModelOutput,
    PromptTemplate,
    ToolCall,
)

NOW = datetime(2026, 10, 19, tzinfo=UTC)


def case(i: int, site: str = "cs_01") -> EvalCase:
    return EvalCase(
        id=f"{site}:{i:03d}",
        site_id=site,
        input_messages=[{"role": "user", "content": f"q{i}"}],
        expected_kind="label",
        split="dev",
    )


def ref(i: int, label: str = "billing", site: str = "cs_01", **kw) -> BaselineResult:
    return BaselineResult(
        case_id=f"{site}:{i:03d}",
        model=kw.pop("model", "reference:seed"),
        output=ModelOutput(text=label),
        recorded_at=NOW,
        **kw,
    )


def test_cases_and_references_round_trip(tmp_path):
    b.save_cases(tmp_path, "cs_01", [case(1), case(2)])
    b.save_references(tmp_path, "cs_01", [ref(1), ref(2, "shipping")])

    assert [c.id for c in b.load_cases(tmp_path, "cs_01")] == ["cs_01:001", "cs_01:002"]
    refs = b.load_references(tmp_path, "cs_01")
    assert refs["cs_01:002"].output.text == "shipping"
    assert (tmp_path / "references" / "cs_01.jsonl").read_text().count("\n") == 2
    assert b.load_cases(tmp_path, "cs_09") == [] and b.load_references(tmp_path, "cs_09") == {}


def test_an_input_in_both_dev_and_holdout_is_refused_and_nothing_is_written(tmp_path):
    """Run 20261022-023459043-590031: identical inputs under different case ids landed in both splits, so the
    Iterator tuned on inputs the holdout verdict then read. The writer refuses that, whatever wrote the cases;
    case and whitespace do not make an input different."""
    dev = case(1).model_copy(update={"input_messages": [{"role": "user", "content": "AI code review bot"}]})
    holdout = case(9).model_copy(
        update={"split": "holdout", "input_messages": [{"role": "user", "content": " ai  code review BOT"}]}
    )
    with pytest.raises(ValueError, match="cs_01:001 and cs_01:009"):
        b.save_cases(tmp_path, "cs_01", [dev, holdout, case(2)])
    assert b.load_cases(tmp_path, "cs_01") == []

    b.save_cases(tmp_path, "cs_01", [dev, holdout.model_copy(update={"split": "dev"})])  # one split: allowed
    assert len(b.load_cases(tmp_path, "cs_01")) == 2


def test_legacy_baselines_are_read_and_references_override_them(tmp_path):
    legacy = tmp_path / "baselines" / "cs_01.jsonl"
    legacy.parent.mkdir()
    legacy.write_text("".join(r.model_dump_json() + "\n" for r in [ref(1, "other"), ref(2, "other")]))
    b.save_references(tmp_path, "cs_01", [ref(1, "billing")])

    refs = b.load_references(tmp_path, "cs_01")
    assert refs["cs_01:001"].output.text == "billing"
    assert refs["cs_01:002"].output.text == "other"


def test_dropped_file_is_not_a_reference_file(tmp_path):
    b.save_references(tmp_path, "cs_01", [ref(1)])
    b.save_dropped(tmp_path, "cs_01", [{"case": case(2).model_dump(mode="json"), "reason": "x"}])

    assert list(b.load_references(tmp_path, "cs_01")) == ["cs_01:001"]
    assert b.load_dropped(tmp_path, "cs_01")[0]["reason"] == "x"


def test_baselines_for_prefers_references_and_fills_gaps_from_closed(tmp_path):
    b.save_references(tmp_path, "cs_01", [ref(1)])
    closed = [
        ref(1, "shipping", model=b.CLAUDE_CODE_REFERENCE, source="closed_model"),
        ref(2, "other", model=b.CLAUDE_CODE_REFERENCE, source="closed_model"),
    ]
    b.save_closed(tmp_path, "cs_01", closed)

    got = b.baselines_for(tmp_path, "cs_01")
    assert got["cs_01:001"].source == "reference" and got["cs_01:001"].output.text == "billing"
    assert got["cs_01:002"].source == "closed_model"
    assert got["cs_01:002"].model == "claude-sonnet-5.5 (via Claude Code; not the repo's source model)"


def test_sites_round_trip_and_merge_by_id(tmp_path):
    site = CallSite(
        id="cs_01",
        file="a.py",
        lines=[1, 2],
        sdk="openai",
        call="c",
        prompt_source="literal",
        template=PromptTemplate(user="hi"),
        contract=Contract(task_type="classification", output="label"),
    )
    b.save_sites(tmp_path, [site])
    b.save_sites(tmp_path, [site.model_copy(update={"id": "cs_02"})])

    assert [s.id for s in b.load_sites(tmp_path)] == ["cs_01", "cs_02"]


def test_manifest_records_hashes_methods_and_models_and_detects_tampering(tmp_path):
    b.save_cases(tmp_path, "cs_01", [case(1), case(2)])
    b.save_references(tmp_path, "cs_01", [ref(1), ref(2, model="reference:oracle")])
    b.save_closed(tmp_path, "cs_01", [ref(1, model=b.CLAUDE_CODE_REFERENCE, source="closed_model")])
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "oracle.py").write_text("x = 1\n")
    (tmp_path / "reference" / "__pycache__").mkdir()
    (tmp_path / "reference" / "__pycache__" / "oracle.cpython-312.pyc").write_bytes(b"\0")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "cs_01.json").write_text("{}")

    m = b.write_manifest(tmp_path, {"case_gen": "nvidia/super"})
    b.write_manifest(tmp_path, {"case_gen_adversarial": "nvidia/ultra"})
    m = json.loads((tmp_path / "manifest.json").read_text())

    assert set(m["files"]) == {
        "cases/cs_01.jsonl",
        "references/cs_01.jsonl",
        "closed/cs_01.jsonl",
        "reference/oracle.py",
    }
    assert m["methods"]["cs_01"] == {"reference:seed": 1, "reference:oracle": 1, "closed_model": 1}
    assert m["generator_models"] == {"case_gen": "nvidia/super", "case_gen_adversarial": "nvidia/ultra"}
    assert m["generated_at"]
    assert b.verify_manifest(tmp_path) == []

    (tmp_path / "references" / "cs_01.jsonl").write_text(ref(1, "other").model_dump_json() + "\n")
    (tmp_path / "cases" / "cs_02.jsonl").write_text(case(1, "cs_02").model_dump_json() + "\n")
    (tmp_path / "reference" / "oracle.py").unlink()
    problems = b.verify_manifest(tmp_path)
    assert any("references/cs_01.jsonl" in p and "hash" in p for p in problems)
    assert any("cases/cs_02.jsonl" in p and "not in manifest" in p for p in problems)
    assert any("reference/oracle.py" in p and "missing" in p for p in problems)


def test_verify_manifest_without_a_manifest_is_a_problem(tmp_path):
    assert b.verify_manifest(tmp_path) == [f"no manifest.json in {tmp_path}"]


# ---------------------------------------------------------------- closed-model recorder (GC22)


def load_recorder():
    path = Path(__file__).resolve().parents[2] / "scripts" / "record_baselines.py"
    spec = importlib.util.spec_from_file_location("record_baselines", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeOpenAI:
    """Mimics client.chat.completions.create; records kwargs."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.replies.pop(0)
        message = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5),
        )


class FakeAnthropic:
    def __init__(self, blocks):
        self.blocks, self.calls = list(blocks), []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=self.blocks.pop(0), usage=SimpleNamespace(input_tokens=200, output_tokens=20)
        )


def site(sid, sdk, task_type, **params):
    return CallSite(
        id=sid,
        file="a.py",
        lines=[1, 2],
        sdk=sdk,
        call="c",
        prompt_source="literal",
        source_model="gpt-4o-mini" if sdk == "openai" else "claude-sonnet-4-5",
        template=PromptTemplate(system="sys", user="{x}"),
        params=params,
        contract=Contract(task_type=task_type, output="label", labels=["billing", "other"]),
    )


def test_recorder_records_closed_outputs_with_measured_fields(tmp_path):
    rec = load_recorder()
    s1, s2 = (
        site("cs_01", "openai", "classification", max_tokens=8, temperature=0),
        site(
            "cs_02",
            "anthropic",
            "tool_call",
            max_tokens=64,
            tools=[{"name": "lookup_order"}],
            tool_choice={"type": "any"},
            system="sys",
        ),
    )
    b.save_sites(tmp_path, [s1, s2])
    b.save_cases(tmp_path, "cs_01", [case(1), case(2)])
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "where is HH-1"}]
    b.save_cases(
        tmp_path,
        "cs_02",
        [
            EvalCase(
                id="cs_02:001", site_id="cs_02", input_messages=msgs, expected_kind="tool_call", split="dev"
            )
        ],
    )
    oa = FakeOpenAI(["billing", "other"])
    an = FakeAnthropic([[SimpleNamespace(type="tool_use", name="lookup_order", input={"order_id": "HH-1"})]])

    summary = rec.record(tmp_path, openai_client=oa, anthropic_client=an)

    assert summary == {"cs_01": 2, "cs_02": 1}
    assert oa.calls[0] == {
        "model": "gpt-4o-mini",
        "messages": case(1).input_messages,
        "max_tokens": 8,
        "temperature": 0,
    }
    assert an.calls[0]["system"] == "sys" and an.calls[0]["messages"] == msgs[1:]
    assert an.calls[0]["tools"] == [{"name": "lookup_order"}] and an.calls[0]["max_tokens"] == 64
    one = b.load_closed(tmp_path, "cs_01")["cs_01:001"]
    assert one.source == "closed_model" and one.model == "gpt-4o-mini" and one.output.text == "billing"
    assert one.input_tokens == 100 and one.output_tokens == 5 and one.latency_ms is not None
    assert one.cost_usd == pytest.approx((100 * 0.15 + 5 * 0.60) / 1e6)
    tool = b.load_closed(tmp_path, "cs_02")["cs_02:001"]
    assert tool.output.tool_calls == [ToolCall(name="lookup_order", arguments={"order_id": "HH-1"})]
    assert b.verify_manifest(tmp_path) == []


def test_recorder_self_agreement_scores_a_second_pass_against_the_first(tmp_path):
    rec = load_recorder()
    b.save_sites(tmp_path, [site("cs_01", "openai", "classification", max_tokens=8)])
    b.save_cases(tmp_path, "cs_01", [case(1), case(2)])
    oa = FakeOpenAI(["billing", "other", "billing", "billing"])

    rec.record(tmp_path, self_agreement=True, openai_client=oa)

    assert json.loads((tmp_path / "closed" / "self_agreement.json").read_text()) == {"cs_01": 0.5}


def test_recorder_refuses_a_site_with_unresolved_scanner_placeholders(tmp_path):
    rec = load_recorder()
    fmt = {"type": "json_schema", "json_schema": {"name": "Order", "schema": "<pydantic:Order>"}}
    b.save_sites(tmp_path, [site("cs_01", "openai", "classification", max_tokens=8, response_format=fmt)])
    b.save_cases(tmp_path, "cs_01", [case(1)])
    oa = FakeOpenAI(["billing"])

    from offramp.agents.translate import UnresolvedCallSite

    with pytest.raises(UnresolvedCallSite, match="response_format"):
        rec.record(tmp_path, openai_client=oa)

    assert oa.calls == []  # refused before any call: never recorded without the parameter


def test_a_refused_site_stops_the_recorder_before_any_call_or_write(tmp_path):
    rec = load_recorder()
    from offramp.agents.translate import UnresolvedCallSite

    fmt = {"type": "json_schema", "json_schema": {"name": "Order", "schema": "<pydantic:Order>"}}
    b.save_sites(
        tmp_path,
        [
            site("cs_01", "openai", "classification", max_tokens=8),
            site("cs_02", "openai", "classification", response_format=fmt),
        ],
    )
    b.save_cases(tmp_path, "cs_01", [case(1)])
    b.save_cases(tmp_path, "cs_02", [case(1, "cs_02")])
    b.write_manifest(tmp_path)
    oa = FakeOpenAI(["billing", "billing"])

    with pytest.raises(UnresolvedCallSite, match="cs_02"):
        rec.record(tmp_path, openai_client=oa)

    assert oa.calls == [] and not (tmp_path / "closed").exists()
    assert b.verify_manifest(tmp_path) == []  # nothing was written, so the manifest still matches
