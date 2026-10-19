"""Global Constraint 23: eval-target calls made from the eval runner's and the eval matrix's thread pools are
traced into the run's LangSmith project `offramp-<run_id>`, not the default project.

Pattern of tests/unit/test_langsmith.py: the real `langsmith` SDK and TFClient, the SDK's HTTP session
replaced by a fake that records what would be sent, Token Factory answered by an httpx MockTransport.
"""

import json
from datetime import UTC, datetime

import httpx
import pytest

from offramp.evals.runner import run_tier
from offramp.evals.serverless_job.matrix import Matrix, run_matrix
from offramp.integrations import langsmith
from offramp.schemas import (
    Adapter,
    AdapterSpec,
    BaselineResult,
    CallSite,
    Contract,
    EvalCase,
    ModelOutput,
    PromptTemplate,
    Tier,
)
from offramp.tf import TFClient

COMPLETION = {
    "id": "c",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "billing"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
SITE = CallSite(
    id="cs_01",
    file="app/triage.py",
    lines=[1, 5],
    sdk="openai",
    call="client.chat.completions.create",
    prompt_source="fstring",
    template=PromptTemplate(user="Classify: {ticket}", variables=["ticket"]),
    contract=Contract(task_type="classification", output="label", labels=["billing", "bug"]),
)
CASES = [
    EvalCase(
        id=f"cs_01:{i:03d}",
        site_id="cs_01",
        input_messages=[{"role": "user", "content": f"Classify: charged twice ({i})"}],
        expected_kind="label",
        split="dev",
    )
    for i in range(4)
]
REFS = {
    c.id: BaselineResult(
        case_id=c.id,
        model="reference:seed",
        output=ModelOutput(text="billing"),
        recorded_at=datetime.now(UTC),
    )
    for c in CASES
}
ADAPTER = Adapter(site_id="cs_01", version=0, spec=AdapterSpec(max_tokens=8))


@pytest.fixture
def traced(monkeypatch, tmp_path):
    """(TFClient, sent()): sent() flushes the tracing client and returns the session names uploaded."""
    sdk = pytest.importorskip("langsmith")
    requests = pytest.importorskip("requests")

    class Session(requests.Session):
        def __init__(self):
            super().__init__()
            self.posts = []

        def request(self, method, url, **kw):
            response = requests.Response()
            response.status_code, response.url, response._content = 200, url, b"{}"
            if url.endswith("/info"):
                response._content = b'{"version": "test", "batch_ingest_config": {}}'
            else:
                self.posts.append(kw.get("data") or kw.get("json"))
            return response

    built = []

    class Client(sdk.Client):
        def __init__(self, *args, **kw):
            super().__init__(*args, session=Session(), **kw)
            built.append(self)

    monkeypatch.setattr(sdk, "Client", Client)
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-primary")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "default-project")
    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=COMPLETION)))
    tf = TFClient(api_key="k", base_url="https://tf.test/v1", cache_dir=tmp_path, http_client=http)

    def sent() -> str:
        for client in built:
            client.flush()
        posts = [p for c in built for p in c.session.posts]
        return b"".join(p if isinstance(p, bytes) else json.dumps(p).encode() for p in posts).decode()

    return tf, sent


def projects(sent: str) -> set[str]:
    """The session (project) names the uploads name."""
    compact = sent.replace(" ", "")
    return {part.split('"', 1)[0] for part in compact.split('"session_name":"')[1:]}


@pytest.mark.filterwarnings("ignore:langsmith.wrappers._openai_agents is deprecated:DeprecationWarning")
def test_eval_calls_from_the_runner_pool_are_traced_into_the_run_project(traced):
    tf, sent = traced
    with langsmith.run_project("r1"):
        result = run_tier(SITE, CASES, REFS, ADAPTER, Tier.LIGHTNING, tf, "dev")
    assert result.passed
    uploads = sent().replace(" ", "")
    assert uploads.count('"session_name":"offramp-r1"') >= len(CASES)  # one traced run per eval call
    assert projects(uploads) == {"offramp-r1"}


@pytest.mark.filterwarnings("ignore:langsmith.wrappers._openai_agents is deprecated:DeprecationWarning")
def test_eval_calls_from_the_matrix_pool_are_traced_into_the_run_project(traced):
    tf, sent = traced
    budget = {"max_in": 10_000, "max_out": 10_000, "max_ultra_agent_calls": 40}
    matrix = Matrix(
        sites=[SITE],
        adapters={"cs_01": {t: ADAPTER for t in (Tier.LIGHTNING, Tier.SUPER)}},
        cases=CASES,
        references=list(REFS.values()),
        tiers=[Tier.LIGHTNING, Tier.SUPER],
        splits=["dev"],
        budget=budget,
    )
    with langsmith.run_project("r1"):
        entries = run_matrix(matrix, tf, workers=2)
    assert [e["kind"] for e in entries] == ["tier_result", "tier_result"]
    assert projects(sent()) == {"offramp-r1"}
