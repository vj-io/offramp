"""Baseline integrity (PRD §12.3, amended by PLAN Amendment A): the eval dataset is what the manifest says,
every scored case has exactly one reference, and no reference carries a number nobody measured.

Runs on a dataset prepared from the recorded mini repo; set OFFRAMP_EVALS_DIR to also check a real one
(e.g. the demo repo's committed evals/).
"""

import os
import shutil
from pathlib import Path

import pytest

from offramp.agents import case_gen as cg
from offramp.evals import baselines as b
from offramp.schemas import ScanResult
from offramp.tf import TFClient

CASSETTES = Path(__file__).resolve().parents[1] / "fixtures" / "cassettes" / "case_gen"
REFERENCE_METHODS = {"reference:seed", "reference:oracle", "reference:fixture", "reference:human"}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> Path:
    repo = tmp_path_factory.mktemp("integrity") / "repo"
    shutil.copytree(CASSETTES / "minirepo", repo)
    scan = ScanResult.model_validate_json((CASSETTES / "minirepo.scan.json").read_text())
    cg.prepare_repo(repo, scan, TFClient(cache_dir=CASSETTES, live=False), target=8, adversarial=2, seed=0)
    return repo / "evals"


def datasets(prepared: Path) -> list[Path]:
    extra = os.environ.get("OFFRAMP_EVALS_DIR")
    return [prepared, *([Path(extra)] if extra else [])]


def test_files_match_the_manifest(prepared):
    for evals in datasets(prepared):
        assert b.verify_manifest(evals) == [], evals


def test_a_tampered_file_fails_verification(prepared, tmp_path):
    copy = tmp_path / "evals"
    shutil.copytree(prepared, copy)
    path = next((copy / "references").glob("cs_*.jsonl"))
    path.write_text(path.read_text().replace("reference:", "reference:x"))

    assert any(str(path.relative_to(copy)) in p for p in b.verify_manifest(copy))


def test_every_scored_case_has_exactly_one_reference(prepared):
    for evals in datasets(prepared):
        for site in b.load_sites(evals):
            cases = b.load_cases(evals, site.id)
            ids = [c.id for c in cases]
            assert len(ids) == len(set(ids)) and all(i.startswith(f"{site.id}:") for i in ids)
            assert all(c.split == cg.split_for(c.id) for c in cases)  # GC10, never hand-assigned
            refs = b.baselines_for(evals, site.id)
            assert set(refs) <= set(ids), f"{site.id}: references without a case"
            if site.contract.task_type != "free_text":
                assert set(refs) == set(ids), f"{site.id}: cases without a reference"


def test_references_carry_no_measured_numbers_and_closed_rows_are_labelled(prepared):
    for evals in datasets(prepared):
        for site in b.load_sites(evals):
            for ref in b.load_references(evals, site.id).values():
                if ref.source == "reference":
                    assert ref.model in REFERENCE_METHODS
                    assert ref.latency_ms is ref.input_tokens is ref.output_tokens is ref.cost_usd is None
            for row in b.load_closed(evals, site.id).values():
                assert row.source == "closed_model" and row.model
                if row.model == b.CLAUDE_CODE_REFERENCE:  # GC27: nothing measured, so nothing reported
                    assert row.latency_ms is row.cost_usd is None
                else:  # recorded by scripts/record_baselines.py from the source model
                    assert row.latency_ms is not None and row.input_tokens is not None


def test_json_references_are_clean_json(prepared):
    import json

    for evals in datasets(prepared):
        for site in b.load_sites(evals):
            if site.contract.task_type == "json_extraction":
                for ref in b.load_references(evals, site.id).values():
                    json.loads(ref.output.text)
