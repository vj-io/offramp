"""SQLite persistence (PRD §7). Thin on purpose: one table per PRD entity, JSON-valued fields stored as text.

Columns mirror the offramp.schemas models so the API can build rows with model_dump(); every per-run
table has a surrogate `id` plus `run_id` because site and case ids ("cs_01", "cs_01:003") repeat across runs.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import Field, SQLModel, create_engine


def _now() -> datetime:
    return datetime.now(UTC)


class Run(SQLModel, table=True):
    id: str = Field(primary_key=True)
    repo_url: str
    commit_sha: str | None = None
    stage: str = "QUEUED"
    created_at: datetime = Field(default_factory=_now)
    totals_json: str = "{}"


class CallSite(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    site_id: str  # "cs_01"
    file: str
    line_start: int
    line_end: int
    sdk: str
    source_model: str | None = None
    contract_json: str | None = None
    prompt_ref: str | None = None


class EvalCase(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    case_id: str  # "cs_01:003"
    site_id: str
    split: str
    input_json: str
    tags_json: str = "[]"


class Baseline(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    case_id: str
    model: str
    output_json: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class Adapter(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    site_id: str
    version: int
    tier: str | None = None
    spec_json: str = "{}"
    patch: str | None = None
    notes_json: str = "[]"


class TierResult(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    site_id: str
    adapter_version: int
    tier: str
    split: str
    passed: bool
    score: float
    p50_ms: int
    p95_ms: int
    tokens_in: int
    tokens_out: int
    cost_per_1k_calls_usd: float
    failing_case_ids_json: str = "[]"


class Verdict(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    site_id: str
    decision: str
    tier: str | None = None
    model: str | None = None
    parity_pct: float | None = None
    best_tier: str | None = None
    best_score: float | None = None
    iterations: int = 0
    latency_delta_ms: int | None = None
    cost_before_per_1k: float | None = None
    cost_after_per_1k: float | None = None
    reason: str


class Artifact(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    kind: str  # pr_url | report | serverless_job
    path: str


def engine(path: str | Path | None = None) -> Engine:
    """SQLite engine for `path` (default $OFFRAMP_DATA_DIR/offramp.db). Safe to share with FastAPI threads."""
    path = Path(path) if path else Path(os.environ.get("OFFRAMP_DATA_DIR", "./var")) / "offramp.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})


def init_db(eng: Engine | None = None) -> Engine:
    """Create all tables (idempotent) and return the engine used."""
    eng = eng or engine()
    SQLModel.metadata.create_all(eng)
    return eng
