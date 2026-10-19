"""Reference audit (GC18, GC27): blind answers checked against seed references with the site's own scorer;
Tendem export/ingest (GC24)."""

import importlib.util
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from offramp.agents import case_gen as cg
from offramp.agents import reference_audit as ra
from offramp.evals import baselines as b
from offramp.schemas import (
    BaselineResult,
    CallSite,
    Contract,
    EvalCase,
    ModelOutput,
    PromptTemplate,
    ScanResult,
    Tier,
    ToolCall,
)
from offramp.tf import FakeLLM, TFClient

ROOT = Path(__file__).resolve().parents[2]
CASSETTES = ROOT / "tests" / "fixtures" / "cassettes"
NOW = datetime(2026, 10, 19, tzinfo=UTC)
SONNET = b.CLAUDE_CODE_REFERENCE
LABELS = ["billing", "shipping", "other"]


def triage_site(sid="cs_01") -> CallSite:
    return CallSite(
        id=sid,
        file="t.py",
        lines=[1, 2],
        sdk="openai",
        call="c",
        prompt_source="fstring",
        template=PromptTemplate(
            system="Route to billing, shipping or other.", user="{body}", variables=["body"]
        ),
        contract=Contract(task_type="classification", output="label", labels=LABELS),
    )


def extract_site(sid="cs_02") -> CallSite:
    schema = {
        "type": "object",
        "properties": {"order_id": {"type": ["string", "null"]}},
        "required": ["order_id"],
    }
    return CallSite(
        id=sid,
        file="e.py",
        lines=[1, 2],
        sdk="anthropic",
        call="c",
        prompt_source="fstring",
        template=PromptTemplate(system="Extract JSON.", user="{email}", variables=["email"]),
        contract=Contract(task_type="json_extraction", output="json_schema", json_schema=schema),
    )


def dataset(evals: Path, site: CallSite, outputs: list[str], model="reference:seed", kind="label"):
    cases, refs = [], []
    for i, text in enumerate(outputs, 1):
        cid = f"{site.id}:{i:03d}"
        msgs = [
            {"role": "system", "content": site.template.system},
            {"role": "user", "content": f"input number {i}"},
        ]
        cases.append(
            EvalCase(
                id=cid,
                site_id=site.id,
                input_messages=msgs,
                expected_kind=kind,
                split=cg.split_for(cid),
                source="synthetic",
            )
        )
        refs.append(BaselineResult(case_id=cid, model=model, output=ModelOutput(text=text), recorded_at=NOW))
    b.save_sites(evals, [site])
    b.save_cases(evals, site.id, cases)
    b.save_references(evals, site.id, refs)
    return cases


def closed(site_id, answers: dict[int, str]) -> list[BaselineResult]:
    return [
        BaselineResult(
            case_id=f"{site_id}:{i:03d}",
            source="closed_model",
            model=SONNET,
            output=ModelOutput(text=t),
            recorded_at=NOW,
        )
        for i, t in answers.items()
    ]


def test_closed_reference_answers_audit_seed_references(tmp_path):
    dataset(tmp_path, triage_site(), ["billing", "shipping", "other"])
    b.save_closed(tmp_path, "cs_01", closed("cs_01", {1: "Billing.", 2: "shipping", 3: "billing"}))

    counts = ra.audit_repo(tmp_path)

    assert counts["cs_01"] == {
        "checked": 3,
        "agreed": 2,
        "dropped": 1,
        "human_verified": 0,
        "unchecked": 0,
        "auditors": [f"blind:{SONNET}"],
    }
    assert json.loads((tmp_path / "audit.json").read_text()) == counts
    assert [c.id for c in b.load_cases(tmp_path, "cs_01")] == ["cs_01:001", "cs_01:002"]
    refs = b.load_references(tmp_path, "cs_01")
    assert set(refs) == {"cs_01:001", "cs_01:002"} and refs["cs_01:001"].verified_by == [f"blind:{SONNET}"]
    [row] = b.load_dropped(tmp_path, "cs_01")
    assert row["case"]["id"] == "cs_01:003" and row["reference"]["output"]["text"] == "other"
    assert row["by"] == f"blind:{SONNET}" and row["auditor_output"]["text"] == "billing"
    assert "expected 'other', got 'billing'" in row["reason"]
    assert b.verify_manifest(tmp_path) == []


def test_without_an_auditor_seed_references_stay_unchecked(tmp_path):
    dataset(tmp_path, triage_site(), ["billing", "other"])

    counts = ra.audit_repo(tmp_path)

    assert counts["cs_01"]["unchecked"] == 2 and counts["cs_01"]["checked"] == 0
    assert len(b.load_cases(tmp_path, "cs_01")) == 2


def test_oracle_references_are_not_blind_audited(tmp_path):
    dataset(tmp_path, triage_site(), ["billing"], model="reference:oracle")
    b.save_closed(tmp_path, "cs_01", closed("cs_01", {1: "other"}))

    counts = ra.audit_repo(tmp_path)

    assert counts["cs_01"]["checked"] == 0 and counts["cs_01"]["dropped"] == 0
    assert len(b.load_cases(tmp_path, "cs_01")) == 1


def test_ultra_fallback_answers_blind_in_batches_with_thinking_on(tmp_path):
    seeds = [json.dumps({"order_id": f"SH-{1000 + i}"}) for i in range(12)]
    dataset(tmp_path, extract_site(), seeds, kind="json")
    answers1 = {
        "answers": [
            {"case_id": f"cs_02:{i:03d}", "output": {"order_id": f"SH-{999 + i}"}} for i in range(1, 11)
        ]
    }
    answers1["answers"][0]["output"]["order_id"] = "SH-0000"  # one disagreement
    answers2 = {
        "answers": [{"case_id": f"cs_02:{i:03d}", "output": {"order_id": f"SH-{999 + i}"}} for i in (11, 12)]
    }
    llm = FakeLLM({"reference_audit": [answers1, answers2]})

    counts = ra.audit_repo(tmp_path, llm=llm)

    assert len(llm.calls) == 2
    for call in llm.calls:
        assert call["tier"] == Tier.ULTRA and call["reasoning"] != "off" and call["max_tokens"] >= 2000
        prompt = json.dumps(call["messages"])
        assert not any(json.dumps(s)[1:-1] in prompt for s in seeds)  # blind: seeds never shown
        assert "Extract JSON." in prompt and "input number" in prompt
    ultra = "blind:nvidia/Nemotron-3-Ultra-550b-a55b"
    assert counts["cs_02"] == {
        "checked": 12,
        "agreed": 11,
        "dropped": 1,
        "human_verified": 0,
        "unchecked": 0,
        "auditors": [ultra],
    }
    assert b.load_dropped(tmp_path, "cs_02")[0]["case"]["id"] == "cs_02:001"


def test_closed_answers_win_over_the_ultra_fallback_and_rerun_is_idempotent(tmp_path):
    dataset(tmp_path, triage_site(), ["billing", "other"])
    b.save_closed(tmp_path, "cs_01", closed("cs_01", {1: "billing"}))
    llm = FakeLLM({"reference_audit": [{"answers": [{"case_id": "cs_01:002", "label": "other"}]}]})

    first = ra.audit_repo(tmp_path, llm=llm)
    second = ra.audit_repo(tmp_path, llm=FakeLLM())

    assert "cs_01:001" not in json.dumps(llm.calls[0]["messages"])
    assert first == second and first["cs_01"]["agreed"] == 2 and first["cs_01"]["checked"] == 2


def test_cli_imports_closed_answers_with_the_gc27_label_and_audits(tmp_path):
    repo = tmp_path / "repo"
    dataset(repo / "evals", triage_site(), ["billing", "shipping"])
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        "".join(
            json.dumps({"case_id": f"cs_01:{i:03d}", "output": {"text": t}}) + "\n"
            for i, t in [(1, "billing"), (2, "other")]
        )
    )

    assert ra.main([str(repo), "--closed", str(answers)]) == 0

    stored = b.load_closed(repo / "evals", "cs_01")
    assert {r.model for r in stored.values()} == {SONNET} and {r.source for r in stored.values()} == {
        "closed_model"
    }
    assert all(r.latency_ms is r.cost_usd is None for r in stored.values())
    assert json.loads((repo / "evals" / "audit.json").read_text())["cs_01"]["dropped"] == 1


def test_human_verdicts_survive_a_re_audit(tmp_path):
    script, repo = load_script(), tmp_path / "repo"
    evals = repo / "evals"
    dataset(evals, triage_site(), ["billing", "shipping"])
    b.save_closed(evals, "cs_01", closed("cs_01", {1: "billing", 2: "other"}))
    ra.audit_repo(evals)
    verdicts = tmp_path / "v.csv"
    verdicts.write_text("case_id,verdict,note\ncs_01:002,correct,an expert says shipping\n")
    script.ingest(repo, verdicts)
    files = {p: p.read_bytes() for p in sorted(evals.rglob("*.jsonl"))}
    counts = json.loads((evals / "audit.json").read_text())

    assert ra.audit_repo(evals) == counts  # the closed answer that disagreed does not overrule the expert
    assert {p: p.read_bytes() for p in sorted(evals.rglob("*.jsonl"))} == files
    assert counts["cs_01"] == {
        "checked": 2,
        "agreed": 1,
        "dropped": 0,
        "human_verified": 1,
        "unchecked": 0,
        "auditors": [f"blind:{SONNET}"],
    }


def test_a_repeated_correct_verdict_does_not_duplicate_the_human_tag(tmp_path):
    script, repo = load_script(), tmp_path / "repo"
    dataset(repo / "evals", triage_site(), ["billing"])
    verdicts = tmp_path / "v.csv"
    verdicts.write_text("case_id,verdict,note\ncs_01:001,correct,\n")

    script.ingest(repo, verdicts)
    script.ingest(repo, verdicts)

    assert b.load_references(repo / "evals", "cs_01")["cs_01:001"].verified_by == ["human:tendem"]


def tool_site(sid="cs_03") -> CallSite:
    tools = [
        {
            "name": "lookup_order",
            "description": "Fetch an order.",
            "input_schema": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }
    ]
    return CallSite(
        id=sid,
        file="a.py",
        lines=[1, 2],
        sdk="anthropic",
        call="c",
        prompt_source="fstring",
        template=PromptTemplate(system="Pick a tool.", user="{t}", variables=["t"]),
        params={"tools": tools, "tool_choice": {"type": "any"}},
        contract=Contract(task_type="tool_call", output="tool_call", tools=True),
    )


def blind_dataset(repo: Path) -> list[str]:
    evals = repo / "evals"
    dataset(evals, triage_site(), ["billing", "shipping"])
    seeds = [json.dumps({"order_id": "SH-4242"}), json.dumps({"order_id": None})]
    dataset(evals, extract_site(), seeds, kind="json")
    site = tool_site()
    case = EvalCase(
        id="cs_03:001",
        site_id="cs_03",
        split="dev",
        source="synthetic",
        expected_kind="tool_call",
        input_messages=[
            {"role": "system", "content": "Pick a tool."},
            {"role": "user", "content": "where is HH-77?"},
        ],
    )
    ref = BaselineResult(
        case_id="cs_03:001",
        model="reference:seed",
        recorded_at=NOW,
        output=ModelOutput(tool_calls=[ToolCall(name="lookup_order", arguments={"order_id": "HH-77"})]),
    )
    b.save_sites(evals, [site])
    b.save_cases(evals, "cs_03", [case])
    b.save_references(evals, "cs_03", [ref])
    return seeds


def test_blind_bundle_holds_inputs_and_formats_but_never_references(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    repo = tmp_path / "repo"
    seeds = blind_dataset(repo)

    out = ra.export_blind_audit(repo)

    assert out.is_relative_to(tmp_path / "data" / "audit") and not out.is_relative_to(repo)
    assert json.loads((out / "bundle.json").read_text())["repo"] == "repo"  # a name, never a path
    assert str(tmp_path) not in "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    text = "".join(p.read_text() for p in sorted(out.rglob("*")) if p.is_file())
    assert not any(seed in text for seed in seeds) and "reference:seed" not in text
    assert '"expected' not in text and "references" not in text
    tools = json.loads((out / "cs_03.json").read_text())
    assert tools["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Fetch an order.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ]
    assert tools["cases"] == [
        {
            "case_id": "cs_03:001",
            "input_messages": [
                {"role": "system", "content": "Pick a tool."},
                {"role": "user", "content": "where is HH-77?"},
            ],
        }
    ]
    assert json.loads((out / "cs_01.json").read_text())["labels"] == LABELS
    assert (
        json.loads((out / "cs_02.json").read_text())["response_schema"] == extract_site().contract.json_schema
    )
    assert "case_id" in (out / "INSTRUCTIONS.md").read_text()


def test_ingest_blind_audit_scores_answers_with_the_exact_model_label(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    repo = tmp_path / "repo"
    blind_dataset(repo)
    out = ra.export_blind_audit(repo)
    rows = [
        {"case_id": "cs_01:001", "output": {"text": "billing"}},
        {"case_id": "cs_01:002", "output": {"text": "other"}},
        {"case_id": "cs_02:001", "output": {"text": '{"order_id": "SH-4242"}'}},
        {
            "case_id": "cs_03:001",
            "output": {"tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "HH-77"}}]},
        },
        {"case_id": "cs_09:001", "output": {"text": "x"}},
    ]
    answers = out / "answers.jsonl"
    answers.write_text("".join(json.dumps(r) + "\n" for r in rows))

    result = ra.ingest_blind_audit(answers, repo)

    assert result["unknown"] == ["cs_09:001"] and result["stored"] == 4
    counts = result["counts"]
    assert (counts["cs_01"]["agreed"], counts["cs_01"]["dropped"]) == (1, 1)
    assert (counts["cs_02"]["agreed"], counts["cs_02"]["unchecked"]) == (1, 1)
    assert counts["cs_03"]["agreed"] == 1 and counts["cs_03"]["auditors"] == [f"blind:{SONNET}"]
    assert {r.model for r in b.load_closed(repo / "evals", "cs_01").values()} == {SONNET}
    assert not (repo / "evals" / "closed" / "cs_09.jsonl").exists()
    assert "disagrees" in b.load_dropped(repo / "evals", "cs_01")[0]["reason"]


# ---------------------------------------------------------------- Tendem export / ingest (GC24)


def load_script():
    spec = importlib.util.spec_from_file_location(
        "audit_references", ROOT / "scripts" / "audit_references.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_writes_a_human_task_with_flagged_and_sampled_references(tmp_path):
    script, repo = load_script(), tmp_path / "repo"
    dataset(repo / "evals", triage_site(), ["billing", "shipping", "other", "billing"])
    b.save_closed(
        repo / "evals", "cs_01", closed("cs_01", {1: "billing", 2: "shipping", 3: "billing", 4: "billing"})
    )
    ra.audit_repo(repo / "evals")

    path = script.export(repo, sample=2, seed=0)

    text = path.read_text()
    assert path == repo / "evals" / "human_audit" / "task.md"
    assert "cs_01:003" in text and "flagged" in text  # the disagreement is always included
    assert sum(f"cs_01:00{i}" in text for i in (1, 2, 4)) == 2  # plus a sample of the agreed ones
    assert "case_id,verdict,note" in text and "Route to billing, shipping or other." in text
    assert "| " in text  # a markdown table


def test_ingest_applies_expert_verdicts(tmp_path):
    script, repo = load_script(), tmp_path / "repo"
    evals = repo / "evals"
    dataset(evals, triage_site(), ["billing", "shipping", "other"])
    b.save_closed(evals, "cs_01", closed("cs_01", {1: "billing", 2: "shipping", 3: "billing"}))
    ra.audit_repo(evals)
    verdicts = tmp_path / "verdicts.csv"
    verdicts.write_text(
        "case_id,verdict,note\ncs_01:003,correct,the ticket is not billing\n"
        "cs_01:002,wrong,it is about an invoice\ncs_01:001,correct,\ncs_01:404,correct,\n"
    )

    result = script.ingest(repo, verdicts)

    assert result["unknown"] == ["cs_01:404"]
    refs = b.load_references(evals, "cs_01")
    assert set(refs) == {"cs_01:001", "cs_01:003"}
    assert refs["cs_01:003"].verified_by == ["human:tendem"]
    assert refs["cs_01:001"].verified_by == [f"blind:{SONNET}", "human:tendem"]
    assert [c.id for c in b.load_cases(evals, "cs_01")] == ["cs_01:001", "cs_01:003"]
    dropped = {r["case"]["id"]: r for r in b.load_dropped(evals, "cs_01")}
    assert dropped["cs_01:002"]["by"] == "human:tendem" and "invoice" in dropped["cs_01:002"]["reason"]
    assert dropped["cs_01:003"]["restored"] is True
    counts = json.loads((evals / "audit.json").read_text())["cs_01"]
    assert counts == {
        "checked": 3,
        "agreed": 1,
        "dropped": 1,
        "human_verified": 2,
        "unchecked": 0,
        "auditors": [f"blind:{SONNET}"],
    }
    assert b.verify_manifest(evals) == []


# ---------------------------------------------------------------- recorded Ultra fallback


def prepared_minirepo(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(CASSETTES / "case_gen" / "minirepo", repo)
    scan = ScanResult.model_validate_json((CASSETTES / "case_gen" / "minirepo.scan.json").read_text())
    triage = next(s for s in scan.sites if s.file == "shop/triage.py")
    cg.prepare_repo(
        repo, scan, TFClient(cache_dir=CASSETTES / "case_gen", live=False), target=8, adversarial=2, seed=0
    )
    return repo, triage


def run_ultra_audit(tmp_path, live):
    repo, triage = prepared_minirepo(tmp_path)
    client = TFClient(cache_dir=CASSETTES / "reference_audit", live=live)
    counts = ra.audit_repo(repo / "evals", llm=client, site_ids=[triage.id])
    assert counts[triage.id]["checked"] == 8 and counts[triage.id]["unchecked"] == 0
    assert counts[triage.id]["agreed"] + counts[triage.id]["dropped"] == 8


def test_ultra_fallback_replays_recorded_calls(tmp_path):
    run_ultra_audit(tmp_path, live=False)


@pytest.mark.live
def test_record_reference_audit_cassettes(tmp_path):
    run_ultra_audit(tmp_path, live=True)
