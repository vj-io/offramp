"""LangSmith helpers (Global Constraint 23) against a fake client object: no network, key or SDK needed."""

import inspect
import json
import logging
import os
import sys
import threading
import time
import types
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest

from offramp.integrations import keys, langsmith
from offramp.integrations.langsmith import CaseResult
from offramp.schemas import (
    BaselineResult,
    CallSite,
    CandidateResult,
    CaseScore,
    EvalCase,
    ModelOutput,
    Tier,
)
from offramp.tf import TFClient

COMPLETION = {
    "id": "c",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
LIVE_KEY = os.environ.get("LANGSMITH_API_KEY")  # read before hermetic_env removes it
SITE = CallSite(
    id="cs_01",
    file="app/support.py",
    lines=[10, 20],
    sdk="openai",
    call="client.chat.completions.create",
    source_model="gpt-4o-mini",
    prompt_source="literal",
)


def case(n: int, split="dev", **kw) -> EvalCase:
    return EvalCase(
        id=f"cs_01:{n:03d}",
        site_id="cs_01",
        input_messages=[{"role": "user", "content": f"ticket {n}"}],
        expected_kind="label",
        split=split,
        tags=["unicode"],
        **kw,
    )


def reference(n: int, label="billing") -> BaselineResult:
    return BaselineResult(
        case_id=f"cs_01:{n:03d}",
        model="reference:seed",
        output=ModelOutput(text=label),
        verified_by=["blind:nemotron-ultra"],
        recorded_at=datetime(2026, 10, 19, tzinfo=UTC),
    )


def result(n: int, passed=True, split="dev", output="billing", error=None) -> CaseResult:
    return CaseResult(
        case(n, split),
        CandidateResult(
            case_id=f"cs_01:{n:03d}",
            tier=Tier.SUPER,
            adapter_version=1,
            output=None if error else ModelOutput(text=output),
            error=error,
            latency_ms=250,
            input_tokens=40,
            output_tokens=3,
            cost_usd=0.00002,
        ),
        CaseScore(
            case_id=f"cs_01:{n:03d}",
            passed=passed,
            score=1.0 if passed else 0.0,
            detail="label ok" if passed else "wrong_label",
        ),
    )


class FakeClient:
    """What the helpers use of langsmith.Client. `fail` is an exception raised by the first API call."""

    def __init__(self, key, log, fail=None):
        self.key, self.log, self.fail, self.datasets = key, log, fail, log.datasets

    def _record(self, op, /, *args, **kw):
        self.log.calls.append((op, args, kw, self.key))
        if self.fail:
            fail, self.fail = self.fail, None
            raise fail

    def has_dataset(self, **kw):
        self._record("has_dataset", **kw)
        return kw["dataset_name"] in self.datasets

    def read_dataset(self, **kw):
        self._record("read_dataset", **kw)
        return self.datasets[kw["dataset_name"]]

    def create_dataset(self, name, **kw):
        self._record("create_dataset", name, **kw)
        ds = SimpleNamespace(id=f"ds-{len(self.datasets) + 1}", url=f"https://smith.test/datasets/{name}")
        self.datasets[name] = ds
        return ds

    def create_examples(self, **kw):
        self._record("create_examples", **kw)

    def create_project(self, name, **kw):
        self._record("create_project", name, **kw)
        return SimpleNamespace(id="p-1", url=f"https://smith.test/projects/{name}")

    def create_run(self, **kw):
        self._record("create_run", **kw)

    def create_feedback(self, run_id, key, **kw):
        self._record("create_feedback", run_id, key, **kw)

    def share_dataset(self, **kw):
        self._record("share_dataset", **kw)
        return {"share_token": "tok", "url": f"https://smith.test/public/tok/d?n={kw['dataset_name']}"}


class Clients:
    """Replaces langsmith._new_client: records the key (and whether it is the tracing client) of every client
    built; `fail` goes to the first one."""

    def __init__(self, *fails):
        self.fails, self.keys, self.tracing, self.calls, self.datasets = list(fails), [], [], [], {}

    def __call__(self, key, *, tracing=False):
        self.keys.append(key)
        self.tracing.append(tracing)
        return FakeClient(key, self, self.fails.pop(0) if self.fails else None)

    def named(self, name):
        return [(args, kw) for n, args, kw, _ in self.calls if n == name]


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    for name in ("LANGSMITH_API_KEY", "LANGSMITH_API_KEY_FALLBACK", "LANGSMITH_TRACING"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(keys, "_exhausted", set())


@pytest.fixture
def clients(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-primary")
    fake = Clients()
    monkeypatch.setattr(langsmith, "_new_client", fake)
    return fake


def http_error(status, text=""):
    err = RuntimeError("scripted")
    err.response = SimpleNamespace(status_code=status, text=text)
    return err


# ------------------------------------------------------------------ without a key everything is a no-op


def test_without_a_key_every_helper_returns_none_and_builds_no_client(monkeypatch):
    def boom(key):
        raise AssertionError("no client may be built without a key")

    monkeypatch.setattr(langsmith, "_new_client", boom)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert langsmith.tracing_enabled() is False
    with langsmith.run_project("r1") as project:
        assert project is None
    assert langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1") is None
    assert langsmith.log_experiment(SITE, Tier.SUPER, "dev", [result(1)], run_id="r1") is None
    assert langsmith.share_link(SITE, run_id="r1") is None


def test_with_a_key_but_without_the_sdk_the_helpers_are_no_ops(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-primary")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setitem(sys.modules, "langsmith", None)  # `import langsmith` raises ImportError
    assert langsmith.tracing_enabled() is False
    with langsmith.run_project("r1") as project:
        assert project is None
    assert langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1") is None
    assert langsmith.share_link(SITE, run_id="r1") is None


# ------------------------------------------------------------------ tracing


@pytest.fixture
def sdk(monkeypatch):
    """A stand-in `langsmith` package whose tracing_context records how it was entered."""
    entered = []

    @contextmanager
    def tracing_context(**kw):
        entered.append(("enter", kw))
        yield
        entered.append(("exit", None))

    fake = types.ModuleType("langsmith")
    fake.tracing_context = tracing_context
    monkeypatch.setitem(sys.modules, "langsmith", fake)
    return entered


@pytest.mark.parametrize(
    "key,tracing,expected",
    [
        ("k", "true", True),
        ("k", "TRUE", True),
        ("k", "false", False),
        ("k", None, False),
        (None, "true", False),
    ],
)
def test_tracing_needs_a_key_and_langsmith_tracing_true(monkeypatch, sdk, key, tracing, expected):
    for name, value in (("LANGSMITH_API_KEY", key), ("LANGSMITH_TRACING", tracing)):
        if value is not None:
            monkeypatch.setenv(name, value)
    assert langsmith.tracing_enabled() is expected


def test_run_project_routes_every_traced_call_inside_it_to_offramp_run_id(monkeypatch, sdk, clients):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with langsmith.run_project("r1") as project:
        assert project == "offramp-r1"
        assert [e[0] for e in sdk] == ["enter"]
    (_, kw), (last, _) = sdk
    assert last == "exit"
    assert (
        kw["project_name"] == "offramp-r1" and kw["enabled"] is True and isinstance(kw["client"], FakeClient)
    )
    assert clients.keys == ["ls-primary"]
    assert clients.tracing == [True]  # the batching client: tracing never blocks on LangSmith


def test_run_project_is_a_no_op_when_tracing_is_off(monkeypatch, sdk, clients):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    with langsmith.run_project("r1") as project:
        assert project is None
    assert sdk == [] and clients.keys == []


@pytest.mark.filterwarnings("ignore:langsmith.wrappers._openai_agents is deprecated:DeprecationWarning")
def test_token_factory_calls_inside_run_project_are_traced_into_that_project(monkeypatch, tmp_path):
    """The real SDK and TFClient with the SDK's HTTP session replaced: what would be sent to LangSmith."""
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

    class Client(sdk.Client):  # the production client (as run_project builds it) over a fake session
        def __init__(self, *args, **kw):
            super().__init__(*args, session=Session(), **kw)
            built.append(self)

    monkeypatch.setattr(sdk, "Client", Client)
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-primary")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=COMPLETION)))
    tf = TFClient(api_key="k", base_url="https://tf.test/v1", cache_dir=tmp_path, http_client=http)

    with langsmith.run_project("r1") as project:
        reply = tf.chat(
            role="scanner", tier=Tier.SUPER, messages=[{"role": "user", "content": "hi"}], max_tokens=5
        )

    assert project == "offramp-r1" and reply.output.text == "hello"
    (client,) = built
    client.flush()  # tracing batches in the background; what it uploads:
    sent = b"".join(b if isinstance(b, bytes) else json.dumps(b).encode() for b in client.session.posts)
    assert b'"session_name":"offramp-r1"' in sent.replace(b" ", b"")
    assert sent.count(b"session_name") == sent.count(b'"session_name":"offramp-r1"')  # no other project


@pytest.mark.filterwarnings("ignore:langsmith.wrappers._openai_agents is deprecated:DeprecationWarning")
def test_a_hanging_langsmith_endpoint_does_not_slow_a_traced_token_factory_call(monkeypatch, tmp_path):
    """Review finding I2: the tracing client used to make two synchronous round trips per traced call."""
    sdk = pytest.importorskip("langsmith")
    requests = pytest.importorskip("requests")
    hang_s = 1.5
    recovered = threading.Event()

    class Hanging(requests.Session):
        def request(self, method, url, **kw):
            if not recovered.wait(hang_s):  # LangSmith is up but unresponsive, for every request
                raise requests.ConnectionError("down")
            response = requests.Response()
            response.status_code, response.url, response._content = 200, url, b"{}"
            return response

    built = []

    class Client(sdk.Client):  # the production client (as run_project builds it) over a hanging session
        def __init__(self, *args, **kw):
            super().__init__(*args, session=Hanging(), **kw)
            built.append(self)

    monkeypatch.setattr(sdk, "Client", Client)
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-primary")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=COMPLETION)))
    tf = TFClient(api_key="k", base_url="https://tf.test/v1", cache_dir=tmp_path, http_client=http)

    with langsmith.run_project("r1") as project:
        # The first traced call lazily imports langsmith's wrappers (~2 s on a loaded machine, independent of
        # the hang); time only the calls after it, which is where a synchronous round trip would show.
        tf.chat(role="scanner", tier=Tier.SUPER, messages=[{"role": "user", "content": "warm"}], max_tokens=5)
        started = time.perf_counter()
        for n in range(3):
            reply = tf.chat(
                role="scanner",
                tier=Tier.SUPER,
                messages=[{"role": "user", "content": f"hi {n}"}],
                max_tokens=5,
            )
        elapsed = time.perf_counter() - started
    recovered.set()  # let the background uploads finish: nothing may keep running or logging after the test
    for client in built:
        client.flush()

    assert project == "offramp-r1" and reply.output.text == "hello"
    assert elapsed < 0.5, (
        f"3 traced calls took {elapsed:.2f}s against a LangSmith that hangs {hang_s}s per request"
    )


def test_the_tracing_client_batches_in_the_background_and_never_asks_the_server_for_info(monkeypatch):
    sdk = pytest.importorskip("langsmith")
    requests = pytest.importorskip("requests")
    asked = []

    class Session(requests.Session):
        def request(self, method, url, **kw):
            asked.append(url)
            raise requests.ConnectionError("down")

    class Client(sdk.Client):
        def __init__(self, *args, **kw):
            super().__init__(*args, session=Session(), **kw)

    monkeypatch.setattr(sdk, "Client", Client)
    tracing, sync = langsmith._new_client("k", tracing=True), langsmith._new_client("k")
    assert tracing.tracing_queue is not None and sync.tracing_queue is None  # batching vs synchronous
    assert tracing.timeout_ms == langsmith.TRACE_TIMEOUT_MS
    assert tracing.info.version == "" and asked == []  # GET /info would block whichever thread traces first


# ------------------------------------------------------------------ datasets


def test_upload_dataset_creates_one_dataset_per_run_and_site_with_cases_and_references(clients):
    cases = [case(1, "dev"), case(2, "holdout"), case(3, "dev")]
    url = langsmith.upload_dataset(SITE, cases, [reference(1), reference(2, "refund")], run_id="r1")
    assert url == "https://smith.test/datasets/offramp-r1-cs_01"
    ((name,), meta) = clients.named("create_dataset")[0]
    assert name == "offramp-r1-cs_01" and meta["metadata"]["site_id"] == "cs_01"
    assert meta["metadata"]["file"] == "app/support.py" and "3 cases" in meta["description"]
    ((), kw) = clients.named("create_examples")[0]
    assert kw["dataset_id"] == "ds-1" and len(kw["examples"]) == 3
    one, two, three = kw["examples"]
    assert one["inputs"] == {"messages": [{"role": "user", "content": "ticket 1"}]}
    assert one["outputs"] == {"reference": {"text": "billing", "tool_calls": []}}
    assert two["outputs"] == {"reference": {"text": "refund", "tool_calls": []}}
    assert three["outputs"] == {}  # no reference for this case: nothing invented
    assert [e["split"] for e in kw["examples"]] == ["dev", "holdout", "dev"]
    assert one["metadata"] == {
        "case_id": "cs_01:001",
        "site_id": "cs_01",
        "tags": ["unicode"],
        "source": "synthetic",
        "expected_kind": "label",
        "reference_model": "reference:seed",
        "reference_source": "reference",
        "verified_by": ["blind:nemotron-ultra"],
    }
    assert len({e["id"] for e in kw["examples"]}) == 3 and all(
        isinstance(e["id"], UUID) for e in kw["examples"]
    )


def test_dataset_and_experiment_helpers_use_the_synchronous_client(clients):
    langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1")
    langsmith.log_experiment(SITE, Tier.SUPER, "dev", [result(1)], run_id="r1")
    langsmith.share_link(SITE, run_id="r1")
    assert clients.tracing == [False, False, False]  # errors must reach the caller (fallback key, logging)


def test_upload_dataset_is_idempotent_per_run_and_site(clients):
    first = langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1")
    again = langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1")
    assert (
        first == again
        and len(clients.named("create_dataset")) == 1
        and len(clients.named("create_examples")) == 1
    )
    other = langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r2")
    assert other.endswith("offramp-r2-cs_01") and len(clients.named("create_dataset")) == 2


def test_upload_dataset_with_no_cases_creates_nothing(clients):
    assert langsmith.upload_dataset(SITE, [], [], run_id="r1") is None
    assert clients.calls == []


# ------------------------------------------------------------------ experiments


def test_log_experiment_is_one_project_per_site_tier_split_linked_to_the_dataset_examples(clients):
    langsmith.upload_dataset(SITE, [case(1), case(2)], [reference(1), reference(2)], run_id="r1")
    examples = clients.named("create_examples")[0][1]["examples"]
    url = langsmith.log_experiment(
        SITE, Tier.SUPER, "dev", [result(1), result(2, passed=False, output="refund")], run_id="r1"
    )
    assert url == "https://smith.test/projects/offramp-r1-cs_01-super-dev"
    ((name,), kw) = clients.named("create_project")[0]
    assert name == "offramp-r1-cs_01-super-dev" and kw["reference_dataset_id"] == "ds-1"
    assert kw["metadata"] == {
        "site_id": "cs_01",
        "tier": "super",
        "split": "dev",
        "cases": 2,
        "passed": 1,
        "run_id": "r1",
    }

    runs = [kw for _, kw in clients.named("create_run")]
    assert [r["reference_example_id"] for r in runs] == [
        e["id"] for e in examples
    ]  # linked to the right examples
    assert all(r["project_name"] == name and r["run_type"] == "llm" for r in runs)
    assert runs[0]["inputs"] == {"messages": [{"role": "user", "content": "ticket 1"}]}
    assert runs[1]["outputs"] == {"text": "refund", "tool_calls": []} and runs[1]["error"] is None
    assert runs[0]["extra"]["metadata"] == {
        "tier": "super",
        "split": "dev",
        "adapter_version": 1,
        "latency_ms": 250,
        "input_tokens": 40,
        "output_tokens": 3,
        "cost_usd": 0.00002,
        "cached": False,
    }
    assert (runs[0]["end_time"] - runs[0]["start_time"]).total_seconds() == 0.25

    feedback = {(args[0], args[1]): kw for args, kw in clients.named("create_feedback")}
    assert len(feedback) == 4 and {k[1] for k in feedback} == {"passed", "score"}
    assert [feedback[(runs[i]["id"], "passed")]["score"] for i in (0, 1)] == [1.0, 0.0]
    assert feedback[(runs[1]["id"], "passed")]["comment"] == "wrong_label"
    assert {kw["session_id"] for kw in feedback.values()} == {"p-1"}  # tied to the experiment, not a bare run


def test_a_candidate_that_errored_is_logged_with_its_error_and_no_output(clients):
    langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1")
    langsmith.log_experiment(
        SITE, Tier.SUPER, "dev", [result(1, passed=False, error="invalid_json")], run_id="r1"
    )
    (run,) = [kw for _, kw in clients.named("create_run")]
    assert run["outputs"] is None and run["error"] == "invalid_json"


def test_log_experiment_for_a_dataset_that_was_never_uploaded_logs_and_returns_none(clients, caplog):
    with caplog.at_level(logging.WARNING):
        assert langsmith.log_experiment(SITE, Tier.SUPER, "dev", [result(1)], run_id="r1") is None
    assert clients.named("create_project") == [] and "log_experiment failed" in caplog.text


def test_log_experiment_with_no_results_creates_nothing(clients):
    assert langsmith.log_experiment(SITE, Tier.SUPER, "dev", [], run_id="r1") is None
    assert clients.calls == []


# ------------------------------------------------------------------ share links


def test_share_link_shares_the_runs_dataset_and_returns_the_public_url(clients):
    langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1")
    assert langsmith.share_link(SITE, run_id="r1") == "https://smith.test/public/tok/d?n=offramp-r1-cs_01"
    assert clients.named("share_dataset") == [((), {"dataset_name": "offramp-r1-cs_01"})]


# ------------------------------------------------------------------ telemetry never fails a run


def test_an_api_failure_is_logged_without_details_and_returns_none_not_an_exception(monkeypatch, caplog):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-primary")
    monkeypatch.setattr(langsmith, "_new_client", Clients(http_error(500, "boom ls-primary")))
    with caplog.at_level(logging.WARNING):
        assert langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1") is None
    assert "upload_dataset failed: RuntimeError (HTTP 500)" in caplog.text
    assert "ls-primary" not in caplog.text and "boom" not in caplog.text


# ------------------------------------------------------------------ fallback key (Global Constraint 28)


@pytest.mark.parametrize(
    "status,text",
    [(401, "auth"), (402, "pay"), (403, "forbidden"), (429, "usage limit exceeded: quota"), (432, "limit")],
)
def test_a_rejected_primary_key_retries_once_on_the_fallback_for_the_rest_of_the_process(
    monkeypatch, caplog, status, text
):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-primary")
    monkeypatch.setenv("LANGSMITH_API_KEY_FALLBACK", "ls-fallback")
    fake = Clients(http_error(status, text))
    monkeypatch.setattr(langsmith, "_new_client", fake)
    with caplog.at_level(logging.INFO):
        url = langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1")
    assert url == "https://smith.test/datasets/offramp-r1-cs_01"
    assert fake.keys == ["ls-primary", "ls-fallback"]
    assert [r.getMessage() for r in caplog.records if r.name == keys.log.name] == [
        "langsmith: primary key exhausted, using fallback"
    ]
    assert "ls-primary" not in caplog.text and "ls-fallback" not in caplog.text

    assert langsmith.share_link(SITE, run_id="r1")  # the next call starts on the fallback
    assert fake.keys == ["ls-primary", "ls-fallback", "ls-fallback"]


def test_the_error_comes_from_the_exception_chain_like_the_sdk_raises_it(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-primary")
    monkeypatch.setenv("LANGSMITH_API_KEY_FALLBACK", "ls-fallback")
    sdk_error = RuntimeError(
        "Authentication failed for /datasets"
    )  # the SDK's own error has no .response ...
    sdk_error.__cause__ = http_error(401)  # ... but chains the HTTPError that has
    fake = Clients(sdk_error)
    monkeypatch.setattr(langsmith, "_new_client", fake)
    assert langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1")
    assert fake.keys == ["ls-primary", "ls-fallback"]


def test_a_plain_rate_limit_or_server_error_keeps_the_same_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-primary")
    monkeypatch.setenv("LANGSMITH_API_KEY_FALLBACK", "ls-fallback")
    for error in (http_error(429, "rate limit, retry later"), http_error(500), ValueError("not http")):
        fake = Clients(error)
        monkeypatch.setattr(langsmith, "_new_client", fake)
        assert langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1") is None
        assert fake.keys == ["ls-primary"]


def test_a_rejected_fallback_is_not_retried(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-primary")
    monkeypatch.setenv("LANGSMITH_API_KEY_FALLBACK", "ls-fallback")
    fake = Clients(http_error(401), http_error(401))
    monkeypatch.setattr(langsmith, "_new_client", fake)
    assert langsmith.upload_dataset(SITE, [case(1)], [reference(1)], run_id="r1") is None
    assert fake.keys == ["ls-primary", "ls-fallback"]


# ------------------------------------------------------------------ the real SDK still has what we call


def test_the_sdk_calls_we_make_still_exist_with_those_arguments():
    sdk = pytest.importorskip("langsmith")

    def params(name):
        return set(inspect.signature(getattr(sdk.Client, name)).parameters)

    assert {"dataset_name", "description", "metadata"} <= params("create_dataset")
    assert {"dataset_id", "examples"} <= params("create_examples")
    assert {"project_name", "reference_dataset_id", "metadata", "upsert"} <= params("create_project")
    assert {"run_id", "key", "score", "comment"} <= params("create_feedback")
    assert "dataset_name" in params("read_dataset") | params("has_dataset") | params("share_dataset")
    assert {"project_name", "client", "enabled"} <= set(inspect.signature(sdk.tracing_context).parameters)
    assert sdk.Client(api_key="x", auto_batch_tracing=False)  # the constructor flag we rely on


# ------------------------------------------------------------------ live roundtrip (--live)


@pytest.mark.live
def test_live_roundtrip_dataset_experiment_trace_then_delete(monkeypatch):
    """Real LangSmith: upload, experiment, a traced call under run_project, then delete everything created."""
    sdk = pytest.importorskip("langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", LIVE_KEY)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    run_id = f"selftest-{datetime.now(UTC):%Y%m%d%H%M%S}"
    client = sdk.Client(api_key=LIVE_KEY, auto_batch_tracing=False)
    dataset = langsmith.dataset_name(run_id, SITE.id)
    experiment = langsmith.experiment_name(run_id, SITE.id, Tier.SUPER, "dev")
    traced = f"offramp-{run_id}"
    try:
        url = langsmith.upload_dataset(
            SITE, [case(1), case(2, "holdout")], [reference(1), reference(2)], run_id=run_id
        )
        assert url and dataset in client.read_dataset(dataset_name=dataset).name
        assert len(list(client.list_examples(dataset_name=dataset))) == 2
        again = langsmith.upload_dataset(
            SITE, [case(1), case(2, "holdout")], [reference(1), reference(2)], run_id=run_id
        )
        assert again == url and len(list(client.list_examples(dataset_name=dataset))) == 2

        exp_url = langsmith.log_experiment(
            SITE, Tier.SUPER, "dev", [result(1), result(2, passed=False)], run_id=run_id
        )
        assert exp_url

        def landed(project, n):
            runs = []
            for _ in range(
                40
            ):  # LangSmith ingests runs asynchronously; a traced project appears with its first run
                try:
                    runs = list(client.list_runs(project_name=project))
                except sdk.utils.LangSmithNotFoundError:
                    runs = []
                if len(runs) >= n:
                    break
                time.sleep(1)
            return runs

        runs = landed(experiment, 2)
        assert len(runs) == 2 and all(r.reference_example_id for r in runs)

        @sdk.traceable(name="selftest")
        def traced_call():
            return "ok"

        with langsmith.run_project(run_id) as project:
            assert project == traced and traced_call() == "ok"
        client.flush()
        assert landed(traced, 1), "the traced call did not land in offramp-<run_id>"
    finally:
        for fn, kw in (
            (client.delete_dataset, {"dataset_name": dataset}),
            (client.delete_project, {"project_name": experiment}),
            (client.delete_project, {"project_name": traced}),
        ):
            try:
                fn(**kw)
            except Exception as e:  # already gone, or never created
                print("cleanup:", type(e).__name__)
