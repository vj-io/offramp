"""Patch Composer (PRD §6.8): contract tests with FakeLLM + a fake sandbox, and the demo repo on cassettes.

The demo-repo tests replay tests/fixtures/cassettes/patcher (recorded live with `pytest --live`) and run the
patched copy's own ruff + pytest in the real sandbox. They skip when the demo repo checkout is absent.
"""

import ast
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai.types.chat import ChatCompletion

from offramp import sandbox as sandbox_module
from offramp.agents.patcher import (
    FAKE_HELPER,
    TOGETHER,
    Edit,
    PatchEdits,
    Unsupported,
    _apply,
    _map_reply,
    _prepare,
    _rewrite_call,
    _summary,
    add_fixtures,
    add_imports,
    call_params,
    client_location,
    compose_patches,
    conformance,
    convert_fakes,
    copy_repo,
    edits_text,
    keep_comment,
    line_length,
    merge_onto,
    mirror_fakes,
    parse_edits,
    prune_twins,
    py_literal,
    reach_handles,
    rewrite_callers,
    site_qualname,
    unify_twins,
    wrap_long_strings,
)
from offramp.agents.pr_writer import is_test_path
from offramp.agents.translate import build_request, system_message
from offramp.sandbox import CheckResult, Sandbox, docker_available
from offramp.schemas import Adapter, AdapterSpec, CallSite, EvalCase, RunConfig, Tier, Verdict
from offramp.tf import BudgetExceeded, FakeLLM, TFClient, get_tier

BUILD = build_request


ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
TINY = FIXTURES / "pr" / "target_repo"
CASSETTES = FIXTURES / "cassettes" / "patcher"
LIGHTNING = get_tier(Tier.LIGHTNING).model_id
THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}
CONFIG = RunConfig(repo_url="https://github.com/example/tinyapp")


def demo_repo() -> Path | None:
    env = os.environ.get("OFFRAMP_DEMO_REPO_DIR")
    candidates = [Path(env)] if env else [p / "offramp-demo-app" for p in ROOT.parents]
    return next((p for p in candidates if (p / "helphub").is_dir()), None)


def work_dir(name: str) -> Path:
    """Sandbox scratch inside the worktree's gitignored var/ (never the system tmp)."""
    path = ROOT / "var" / "test-sandbox" / name
    shutil.rmtree(path, ignore_errors=True)
    return path


SITE_MODULES = ("tinyapp/greet.py", "tinyapp/shout.py", "greet.py", "shout.py")  # nested and flat


class FakeSandbox:
    """Stands in for offramp.sandbox.Sandbox: canned check results, records what it was asked. The check of
    Offramp's own changes alone (a site module changed, the tests not yet) answers `precheck` and leaves
    `results` to the other checks."""

    mode = "docker"

    def __init__(self, repo: Path, results: list[bool] | None = None, precheck: bool = False,
                 output: str = "FAILED tests/test_tinyapp.py::x"):  # fmt: skip
        self.repo, self.results, self.checks, self.fixed = repo, list(results or []), [], []
        self.output = output
        self.precheck, self.start = precheck, self._files()

    def _files(self) -> dict[str, str]:
        return {p.relative_to(self.repo).as_posix(): p.read_text() for p in self.repo.rglob("*.py")}

    def check(self) -> CheckResult:
        files = self._files()
        self.checks.append(files)
        tests_as_before = all(files.get(p) == t for p, t in self.start.items() if is_test_path(p))
        if tests_as_before and any(files.get(p) != self.start.get(p) for p in SITE_MODULES):
            ok = self.precheck
        else:
            ok = self.results.pop(0) if self.results else True
        return CheckResult(ok=ok, mode="docker", output="" if ok else self.output)

    def formatted(self) -> bool:
        return False

    def tidy(self, paths: list[str], format_code: bool = False) -> None:
        self.fixed.append(paths)


def tiny_sites() -> list[CallSite]:
    def site(i, file, function, model, prompt_ref):
        return CallSite(
            id=f"cs_{i:02d}",
            file=file,
            lines=[7, 14] if i != 2 else [5, 8],
            sdk="openai",
            call="client.chat.completions.create",
            function=function,
            source_model=model,
            prompt_source="fstring",
            prompt_ref=prompt_ref,
            params={"max_tokens": 32},
        )

    return [
        site(1, "tinyapp/greet.py", "greet", "gpt-4o-mini", "GREET_SYSTEM_PROMPT"),
        site(2, "tinyapp/motto.py", "motto", "o3", None),
        site(3, "tinyapp/shout.py", "shout", "gpt-4o-mini", "SHOUT_SYSTEM_PROMPT"),
    ]


def tiny_verdicts() -> list[Verdict]:
    return [
        Verdict(
            site_id="cs_01",
            decision="MIGRATE",
            tier=Tier.LIGHTNING,
            model=LIGHTNING,
            parity_pct=100.0,
            reason="r1",
        ),
        Verdict(site_id="cs_02", decision="KEEP_CLOSED", best_tier=Tier.ULTRA, parity_pct=60.0, reason="r2"),
        Verdict(
            site_id="cs_03",
            decision="MIGRATE",
            tier=Tier.LIGHTNING,
            model=LIGHTNING,
            parity_pct=100.0,
            reason="r3",
        ),
    ]


def tiny_adapters(**spec_over) -> dict[str, Adapter]:
    def adapter(sid, prompt):
        spec = AdapterSpec(system_prompt=prompt, max_tokens=32, temperature=0.0).model_copy(update=spec_over)
        return Adapter(site_id=sid, version=0, tier=Tier.LIGHTNING, spec=spec)

    return {
        "cs_01": adapter("cs_01", "You write one-line greetings."),
        "cs_03": adapter("cs_03", "You rewrite text as a short headline."),
    }


MIGRATED = f'model="{LIGHTNING}",'  # the call is rewritten by code, the evaluated request written out
TESTS = "tests/test_tinyapp.py"
BREAK_GREET = Edit(path=TESTS, search='"Hi Ana!"', replace='"Hi Ana"')  # a test edit the sandbox may fail
BREAK_GREET_2 = Edit(path=TESTS, search='"Hi Ana!"', replace='"hi ana!"')
BREAK_SHOUT = Edit(path=TESTS, search='"BIG NEWS"', replace='"big news"')


MARKS = {"tinyapp/greet.py": "def test_greet_strips_the_reply():", "tinyapp/shout.py": "def test_shout_"}


def migrate(path: str, extra: list[Edit] = ()) -> PatchEdits:
    """A patcher reply for a tiny-repo site: the code rewrote the module already; the reply touches its test
    (a mark standing in for updated pins) plus any `extra` edits."""
    mark = Edit(
        path="tests/test_tinyapp.py", search=MARKS[path], replace=f"# migrated: {path}\n{MARKS[path]}"
    )
    return PatchEdits(edits=[mark, *extra], notes="pins updated")


def patcher_llm(*replies) -> FakeLLM:
    """FakeLLM answering role `patcher` in the plain edit-block format."""
    return FakeLLM().add("patcher", *(edits_text(r) if isinstance(r, PatchEdits) else r for r in replies))


def run(llm, results=None, sandboxes=None, adapters=None, verdicts=None, name="tiny", build=BUILD, repo=TINY,
        precheck=False, output="FAILED tests/test_tinyapp.py::x"):  # fmt: skip
    def factory(repo):
        box = FakeSandbox(repo, results, precheck, output)
        if sandboxes is not None:
            sandboxes.append(box)
        return box

    return compose_patches(
        repo,
        tiny_sites(),
        verdicts or tiny_verdicts(),
        adapters or tiny_adapters(),
        llm,
        work_dir(name),
        config=CONFIG,
        sandbox_factory=factory,
        build_request=build,
    )


def verdict(result, site_id):
    return next(v for v in result.verdicts if v.site_id == site_id)


# ---------------------------------------------------------------- composition


def test_migrates_passing_sites_and_comments_kept_ones():
    llm = patcher_llm()  # the tiny repo's tests do not pin the provider: Offramp's own changes are green
    boxes = []

    result = run(llm, sandboxes=boxes, precheck=True)

    assert llm.calls == []
    assert [v.decision for v in result.verdicts] == ["MIGRATE", "KEEP_CLOSED", "MIGRATE"]
    assert set(result.files) == {
        "tinyapp/offramp_client.py",  # inside the package that holds the sites: it ships in the wheel
        "tinyapp/greet.py",
        "tinyapp/motto.py",
        "tinyapp/shout.py",
    }
    assert result.client_path == "tinyapp/offramp_client.py"
    assert (
        MIGRATED in result.files["tinyapp/greet.py"]
        and "temperature=0.0," in result.files["tinyapp/greet.py"]
    )
    assert (
        "    # offramp: kept on o3 — parity 60.0% < 95%\n    response = client"
        in result.files["tinyapp/motto.py"]
    )
    assert result.adapters["cs_01"].patch.startswith("--- a/tinyapp/greet.py\n+++ b/tinyapp/greet.py\n")
    assert result.diffs["cs_01"] == result.adapters["cs_01"].patch
    assert "+    # offramp: kept on o3" in result.diffs["cs_02"]
    assert result.site_files == {
        "cs_01": ["tinyapp/greet.py"],
        "cs_02": ["tinyapp/motto.py"],
        "cs_03": ["tinyapp/shout.py"],
    }
    assert result.sandbox_mode == "docker"
    assert len(boxes[0].checks) == 5  # untouched, + client and comments, each site alone, then combined
    assert "tinyapp/offramp_client.py" not in boxes[0].checks[0]
    # each site's module tidied as it will ship (shown to the model so), then each patch
    assert boxes[0].fixed == [["tinyapp/greet.py"]] * 2 + [["tinyapp/shout.py"]] * 2


def test_client_module_reads_the_key_and_pins_verified_tier_ids():
    llm = patcher_llm(migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))

    client = run(llm).files["tinyapp/offramp_client.py"]

    assert 'os.environ["NEBIUS_API_KEY"]' in client
    assert '"https://api.tokenfactory.nebius.com/v1"' in client
    module = ast.parse(client)
    consts = {
        n.targets[0].id: n.value.value
        for n in module.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
    }
    assert {k: consts[k] for k in ("LIGHTNING", "SUPER", "ULTRA")} == {
        t.name: get_tier(t).model_id for t in Tier
    }
    assert max(len(line) for line in client.splitlines()) <= 79  # clean under ruff's default line length too


def test_failing_repo_tests_downgrade_the_site_and_drop_its_patch():
    breaking = migrate("tinyapp/shout.py", extra=[BREAK_SHOUT])
    llm = patcher_llm(migrate("tinyapp/greet.py"), breaking, breaking)  # shout: first try + repair round

    result = run(llm, results=[True, True, True, False])  # untouched, + client, greet, shout (then a repeat)

    shout = verdict(result, "cs_03")
    assert shout.decision == "MANUAL_REVIEW"
    assert "repo checks failed" in shout.reason and "FAILED tests/test_tinyapp.py::x" in shout.reason
    assert "tinyapp/shout.py" not in result.files
    assert result.adapters["cs_03"].patch is None and "cs_03" not in result.diffs
    assert verdict(result, "cs_01").decision == "MIGRATE"


@pytest.mark.parametrize(
    ("results", "step"),
    [([False], "the untouched repo"), ([True, False], "after adding Offramp's client module")],
)
def test_a_red_baseline_downgrades_every_migration_names_the_step_and_changes_nothing(results, step):
    result = run(FakeLLM(), results=results)

    assert [v.decision for v in result.verdicts] == ["MANUAL_REVIEW", "KEEP_CLOSED", "MANUAL_REVIEW"]
    assert step in verdict(result, "cs_01").reason
    assert result.files == {}


def test_an_empty_or_malformed_reply_is_asked_again_without_another_check():
    boxes = []
    malformed = "FILE: tests/test_tinyapp.py\n<<<<<<< SEARCH\nx\n=======\ny\n======="  # no REPLACE marker
    llm = patcher_llm(PatchEdits(edits=[]), malformed, PatchEdits(edits=[]), migrate("tinyapp/shout.py"))

    result = run(llm, sandboxes=boxes)

    reason = verdict(result, "cs_01").reason
    assert reason.startswith("the reply holds no complete edit block; last failure: repo checks failed")
    assert reason.endswith("FAILED tests/test_tinyapp.py::x")  # the checks' own failure is kept
    assert [c["tier"] for c in llm.calls[:3]] == [Tier.SUPER, Tier.ULTRA, Tier.ULTRA]
    assert len(boxes[0].checks) == 5  # untouched, base, greet alone (red); shout alone (red), its try


def test_no_accepted_migration_means_no_client_module():
    llm = patcher_llm(*[PatchEdits(edits=[])] * 6)  # two sites, each a try and two repairs, all empty

    result = run(llm, results=[True, True, False, False])

    assert set(result.files) == {"tinyapp/motto.py"}


@pytest.mark.parametrize(
    ("edits", "problem"),
    [
        ([Edit(path=TESTS, search="no such text", replace="x")], "not found"),
        ([Edit(path="pyproject.toml", search="tinyapp", replace="x")], "not editable"),
        ([Edit(path="../escape.py", search="x", replace="y")], "not editable"),
        (
            [Edit(path=TESTS, search="def test_greet_strips_the_reply():", replace="def _skipped():")],
            "removes tests",
        ),
        ([Edit(path="tinyapp/greet.py", search="max_tokens=32,", replace="max_tokens=64,")], "read-only"),
        ([Edit(path="tinyapp/greet.py", search=".strip()", replace="")], "read-only"),
    ],
    ids=["search-missing", "file-not-allowed", "path-escape", "test-removed", "request", "module"],
)
def test_patches_that_break_the_adapter_spec_or_the_rules_are_rejected(edits, problem):
    llm = patcher_llm(PatchEdits(edits=edits), PatchEdits(edits=edits), migrate("tinyapp/shout.py"))

    result = run(llm)

    greet = verdict(result, "cs_01")
    assert greet.decision == "MANUAL_REVIEW"
    assert problem in greet.reason
    assert "tinyapp/greet.py" not in result.files
    assert verdict(result, "cs_03").decision == "MIGRATE"


def test_budget_exhaustion_downgrades_instead_of_crashing():
    llm = patcher_llm(BudgetExceeded("output token cap"), migrate("tinyapp/shout.py"))

    result = run(llm)

    assert verdict(result, "cs_01").decision == "MANUAL_REVIEW"
    assert "output token cap" in verdict(result, "cs_01").reason


def test_patcher_call_is_super_with_thinking_off_and_shows_the_exact_request():
    mark = Edit(
        path="tests/test_tinyapp.py",
        search="def test_greet_strips_the_reply():",
        replace="def test_greet_strips_the_reply():  # migrated",
    )
    llm = patcher_llm(migrate("tinyapp/greet.py", extra=[mark]), migrate("tinyapp/shout.py"))

    run(llm)

    first = llm.calls[0]
    assert (first["role"], first["tier"], first["temperature"], first["reasoning"]) == (
        "patcher",
        Tier.SUPER,
        0.0,
        "off",
    )
    assert first["response_model"] is None  # plain edit blocks, not code inside JSON strings
    prompt = first["messages"][-1]["content"]
    assert "`tinyapp/greet.py` is migrated by Offramp and read-only" in prompt
    failing = "## What fails now\nWith Offramp's changes alone, this repo's checks fail:\n- FAILED tests/"
    assert failing in prompt  # Offramp's own changes were checked first
    before, now = prompt.split("Before (what the tests pin today):")[1].split("Now:")
    assert 'model="gpt-4o-mini",' in before and MIGRATED in now and "max_tokens=32," in now
    assert 'extra_body={"chat_template_kwargs": {"enable_thinking": False}},' in now
    assert "### tinyapp/greet.py (read-only)" in prompt and "### tests/test_tinyapp.py (editable)" in prompt
    assert "### tinyapp/offramp_client.py (read-only)" in prompt
    second = llm.calls[1]["messages"][-1]["content"]  # each site is patched against the same base
    assert "### tinyapp/greet.py" not in second  # unrelated module: not shown
    assert "def test_greet_strips_the_reply():  # migrated" not in second  # order independent


def test_adapter_constants_are_written_into_the_file_before_the_call():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "description": "d " * 60,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    few = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello there"}]
    adapters = tiny_adapters(
        tools=tools,
        tool_choice="required",
        few_shot=few,
        system_prompt="You write one-line greetings. Never use emoji.",
    )
    llm = patcher_llm(*[PatchEdits(edits=[])] * 6)

    run(llm, adapters=adapters)

    prompt = llm.calls[0]["messages"][-1]["content"]
    source = prompt.split("### tinyapp/greet.py (read-only)\n```python\n")[1].split("```")[0]
    module = ast.parse(source)
    consts = {n.targets[0].id: ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)}
    assert consts["NEMOTRON_TOOLS"] == tools  # a schema: a module constant, passed by name
    assert 'tool_choice="auto",' in source  # Lightning: build_request's swap, written out
    assert consts["NEMOTRON_FEW_SHOT"] == few
    assert (
        consts["GREET_SYSTEM_PROMPT"] == "You write one-line greetings. Never use emoji."
    )  # replaced in place
    assert source.index("NEMOTRON_TOOLS =") < source.index("def greet")
    assert "*NEMOTRON_FEW_SHOT," in source and "tools=NEMOTRON_TOOLS," in source and MIGRATED in source
    assert max(len(line) for line in source.splitlines()) <= 100


# ---------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    "value",
    [
        {
            "type": "json_schema",
            "json_schema": {
                "name": "X",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "ü " + "long words here " * 20,
                        }
                    },
                    "required": ["a"],
                    "additionalProperties": False,
                },
            },
        },
        ['say "hi"', "it's", "back\\slash", "tab\tnew\nline", None, 1.5, -2, True],
        "x" * 300,
    ],
)
def test_py_literal_round_trips_and_respects_line_length(value):
    text = py_literal(value, indent=4, width=100)

    assert ast.literal_eval(text) == value
    assert max(len(line) for line in ("    " + text).splitlines()) <= 100


COMPLETION = {
    "id": "x",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.mark.parametrize("fallback", [False, True])
def test_call_params_are_exactly_what_tfclient_sends(tmp_path, fallback):
    sent = []

    def capture(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=COMPLETION)

    client = TFClient(
        api_key="test-key",
        cache_dir=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(capture)),
    )
    request = {
        "role": "eval_target",
        "tier": Tier.SUPER,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2032,
        "temperature": 0.0,
        "json_mode_fallback": fallback,
        "reasoning": "high",
        "json_schema": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
        "tools": None,
        "tool_choice": None,
        "extra_body": {"top_k": 5},
    }

    client.chat(**request)
    params = call_params(request)

    body = sent[0]
    body.pop("messages")
    assert body == {**{k: v for k, v in params.items() if k != "extra_body"}, **params["extra_body"]}
    assert params["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_call_parameters_come_from_build_request():
    def build(spec, case, tier):  # e.g. Task 6 adds reasoning room and swaps tool_choice per tier
        return {**BUILD(spec, case, tier), "max_tokens": 2032, "tool_choice": "auto"}

    adapters = tiny_adapters(
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}], tool_choice="required"
    )
    llm = patcher_llm(*[PatchEdits(edits=[])] * 6)

    run(llm, adapters=adapters, build=build)

    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "max_tokens=2032," in prompt and 'tool_choice="auto",' in prompt


def test_keep_comment_format():
    v = Verdict(site_id="cs_05", decision="KEEP_CLOSED", parity_pct=71.43, reason="x")
    reasoning = CallSite(
        id="cs_05",
        file="a.py",
        lines=[1, 2],
        sdk="openai",
        call="c",
        source_model="o3",
        prompt_source="literal",
        contract={"task_type": "reasoning", "output": "text"},
    )

    assert keep_comment(reasoning, v, CONFIG) == "# offramp: kept on o3 — parity 71.4% < 90%"


# ---------------------------------------------------------------- demo repo on cassettes


DEMO_OUTCOME = ({"cs_04": True, "cs_05": True, "cs_06": False}, ["cs_04", "cs_05"])  # live, 10c r2
CS05_SUFFIX = (
    "\n\nReply with the final refund amount in the customer's currency alone on the last line as "
    "`ANSWER: <number>` (digits and a decimal point only: no currency symbol, no thousands separator)."
)


def demo_inputs():
    """The demo repo's sites with the verdicts and evaluated adapter specs of Task 12's end-to-end run
    (cassettes in its tests/fixtures/cassettes/pipeline): cs_04, cs_05 and cs_06 MIGRATE on Lightning,
    cs_01..cs_03 KEEP_CLOSED. FIXTURE verdict numbers are that run's, labelled in its review."""
    data = json.loads((FIXTURES / "pr" / "demo_run.json").read_text())["context"]
    sites = [CallSite.model_validate(s) for s in data["sites"]]
    lightning = {"decision": "MIGRATE", "tier": Tier.LIGHTNING, "model": LIGHTNING, "reason": "parity"}
    verdicts = [
        Verdict(
            site_id="cs_01", decision="KEEP_CLOSED", best_tier=Tier.LIGHTNING, parity_pct=85.7, reason="k"
        ),
        Verdict(
            site_id="cs_02", decision="KEEP_CLOSED", best_tier=Tier.LIGHTNING, parity_pct=50.0, reason="k"
        ),
        Verdict(
            site_id="cs_03", decision="KEEP_CLOSED", best_tier=Tier.LIGHTNING, parity_pct=71.4, reason="k"
        ),
        Verdict(site_id="cs_04", parity_pct=100.0, **lightning),
        Verdict(site_id="cs_05", parity_pct=100.0, **lightning),
        Verdict(site_id="cs_06", parity_pct=100.0, **lightning),
    ]
    specs = {
        "cs_04": AdapterSpec(max_tokens=400, temperature=0.2),
        "cs_05": AdapterSpec(max_tokens=16000, temperature=0.0, reasoning="medium", user_suffix=CS05_SUFFIX),
        "cs_06": AdapterSpec(max_tokens=400, temperature=0.0),
    }
    adapters = {
        sid: Adapter(site_id=sid, version=0, tier=Tier.LIGHTNING, spec=spec) for sid, spec in specs.items()
    }
    return sites, verdicts, adapters, RunConfig.model_validate(data["config"])


def run_demo(live: bool):
    if not docker_available():  # the recorded cassette keys name the docker sandbox's output
        pytest.skip("the demo replay needs docker")
    if not live and not any(CASSETTES.glob("*.json")):
        pytest.skip("demo patcher cassettes not recorded for the current patcher prompt: pytest --live")
    repo = demo_repo()
    if repo is None:
        pytest.skip("offramp-demo-app checkout not found (set OFFRAMP_DEMO_REPO_DIR)")
    sites, verdicts, adapters, config = demo_inputs()
    llm = TFClient(cache_dir=CASSETTES, live=live)
    return compose_patches(
        repo, sites, verdicts, adapters, llm, work_dir("demo"), config=config, build_request=BUILD
    )


@pytest.mark.live
def test_record_demo_patches():
    result = run_demo(live=True)
    print({v.site_id: (v.decision, v.reason) for v in result.verdicts})


def test_demo_patches_keep_the_demo_repo_green():
    """Task 12's MIGRATE set, recorded live (Super). Each site is checked alone against the same base, then
    all green ones together; the outcome below is what the recording produced (see the task-10 report)."""
    result = run_demo(live=False)

    assert result.client_path == (None if not result.combined else "helphub/offramp_client.py")
    assert result.warnings == []
    assert set(result.alone) == {"cs_04", "cs_05", "cs_06"}
    assert set(result.combined) <= {sid for sid, ok in result.alone.items() if ok}
    for v in result.verdicts:
        if v.site_id in result.combined:
            assert v.decision == "MIGRATE" and result.adapters[v.site_id].patch
        elif v.site_id in result.alone:
            assert v.decision == "MANUAL_REVIEW"
    for path, text in result.files.items():  # E501 never reaches the PR
        assert max(map(len, text.splitlines()), default=0) <= 100, path
    assert DEMO_OUTCOME == ({s: result.alone[s] for s in sorted(result.alone)}, result.combined)


def test_edit_blocks_round_trip_with_python_escapes_and_empty_replacements():
    patch = PatchEdits(
        edits=[
            Edit(path="a.py", search='x = "one\\ntwo"\n', replace='x = "one\\n"\n\n'),
            Edit(path="b.py", search="del me", replace=""),
        ],
        notes="done",
    )

    assert parse_edits(edits_text(patch)) == patch
    assert parse_edits("Sure!\n```\n" + edits_text(patch) + "\n```") == patch  # fences and chatter ignored


def test_a_failed_try_gets_one_repair_round_that_sees_the_failure():
    breaking = migrate("tinyapp/greet.py", extra=[BREAK_GREET])
    llm = patcher_llm(breaking, migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))

    result = run(llm, results=[True, True, False, True, True, True])

    assert verdict(result, "cs_01").decision == "MIGRATE"
    first, repair = llm.calls[0]["messages"], llm.calls[1]["messages"]
    assert repair[: len(first)] == first
    assert repair[-2] == {"role": "assistant", "content": edits_text(breaking)}
    assert "FAILED tests/test_tinyapp.py::x" in repair[-1]["content"]
    assert len(llm.calls) == 3


def test_a_block_overlapping_an_earlier_one_is_explained():
    first = Edit(path=TESTS, search='== "Hi Ana!"', replace='== "Hi Ana!"  # a')
    overlap = Edit(path=TESTS, search='"Ana") == "Hi Ana!"\n', replace="x")
    bad = PatchEdits(edits=[first, overlap])
    llm = patcher_llm(bad, bad, migrate("tinyapp/shout.py"))

    reason = verdict(run(llm), "cs_01").reason

    assert "an earlier block already changed these lines" in reason


def test_a_repeated_edit_is_applied_once():
    edit = Edit(path=TESTS, search='== "Hi Ana!"', replace='== "Hi Ana!"  # same')
    llm = patcher_llm(
        PatchEdits(edits=[*migrate("tinyapp/greet.py").edits, edit, edit]), migrate("tinyapp/shout.py")
    )

    result = run(llm)

    assert verdict(result, "cs_01").decision == "MIGRATE"
    assert result.files[TESTS].count("# same") == 1


def test_add_imports_binds_known_names_after_the_last_import():
    source = '"""Doc."""\n\nimport os\n\n\ndef f(c):\n    return json.loads(token_factory_client()), os\n'

    fixed = add_imports(source, "helphub.offramp_client")

    assert fixed.splitlines()[2:5] == [
        "import os",
        "from helphub.offramp_client import token_factory_client",
        "import json",
    ]
    assert add_imports(fixed, "helphub.offramp_client") == fixed
    assert add_imports("import json\n\nx = json\n", "offramp.client") == "import json\n\nx = json\n"


def test_an_unresolved_adapter_spec_is_not_patched():
    adapters = tiny_adapters(tool_choice="<expr>", tools=[{"type": "function", "function": {"name": "f"}}])
    llm = patcher_llm(migrate("tinyapp/shout.py"))  # never asked for cs_01

    result = run(llm, adapters=adapters)

    assert verdict(result, "cs_01").decision == "MANUAL_REVIEW"
    assert "patch not composed" in verdict(result, "cs_01").reason


def test_a_json_object_request_is_not_patched():
    """The eval's json_object path includes TFClient's repair retry; a patch could not ship that."""
    schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    adapters = tiny_adapters(response_format=schema, json_mode_fallback=True)

    result = run(FakeLLM(), adapters=adapters)

    assert {v.decision for v in result.verdicts if v.site_id != "cs_02"} == {"MANUAL_REVIEW"}
    assert "json_object" in verdict(result, "cs_01").reason


@pytest.mark.parametrize(
    ("files", "packages", "expected"),
    [
        (
            ["tinyapp/greet.py", "tinyapp/sub/shout.py"],
            ["tinyapp", "tinyapp/sub"],
            ("tinyapp/offramp_client.py", "tinyapp.offramp_client"),
        ),
        (["src/pkg/a.py"], ["src/pkg"], ("src/pkg/offramp_client.py", "pkg.offramp_client")),
        (["app.py"], [], ("offramp/client.py", "offramp.client")),
        (["one/a.py", "two/b.py"], ["one", "two"], ("offramp/client.py", "offramp.client")),
        (["pkg/a.py"], [".", "pkg"], ("pkg/offramp_client.py", "pkg.offramp_client")),
        (["app.py"], ["."], None),
    ],
    ids=[
        "package",
        "src-layout",
        "top-level-module",
        "two-packages",
        "root-package-sites-in-pkg",
        "root-package-sites-at-root",
    ],  # fmt: skip
)
def test_client_location_is_the_package_holding_the_sites(files, packages, expected):
    repo = work_dir("location")
    for pkg in packages:
        (repo / pkg).mkdir(parents=True, exist_ok=True)
        (repo / pkg / "__init__.py").write_text("")

    assert client_location(repo, files) == expected


def test_symlinks_leaving_the_repo_are_dropped_from_the_copy():
    src = work_dir("symlinks") / "src"
    shutil.copytree(TINY, src)
    secret = src.parent / "environ"  # stands in for /proc/self/environ
    secret.write_text("NEBIUS_API_KEY=canary-secret-123")
    (src / "tinyapp" / "data.txt").symlink_to(secret)
    (src / "tinyapp" / "alias.py").symlink_to("greet.py")  # stays inside the repo: kept, as a link
    boxes = []

    def factory(repo):
        boxes.append(FakeSandbox(repo))
        return boxes[-1]

    llm = patcher_llm(migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))
    result = compose_patches(src, tiny_sites(), tiny_verdicts(), tiny_adapters(), llm, src.parent / "work",
                             config=CONFIG, sandbox_factory=factory, build_request=BUILD)  # fmt: skip

    copy = boxes[0].repo
    assert not (copy / "tinyapp" / "data.txt").exists() and not (copy / "tinyapp" / "data.txt").is_symlink()
    assert (copy / "tinyapp" / "alias.py").is_symlink()
    assert not any(
        "canary-secret-123" in f.read_text(errors="ignore") for f in copy.rglob("*") if f.is_file()
    )
    assert result.warnings == ["dropped symlink tinyapp/data.txt: it points outside the repository"]


def test_docker_runs_without_secrets_and_with_resource_limits(monkeypatch, tmp_path):
    seen = []

    def fake_run(cmd, **kw):
        seen.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    monkeypatch.setenv("NEBIUS_API_KEY", "canary-secret-123")

    Sandbox(tmp_path, docker=True).check()

    runs = [cmd for cmd, _ in seen if cmd[:2] == ["docker", "run"]]
    assert runs and all("canary-secret-123" not in " ".join(c) for c in runs)
    for cmd in runs:
        for flag in (["--network", "none"], ["--cpus", "2"], ["--memory", "2g"], ["--pids-limit", "256"],
                     ["--cap-drop", "ALL"], ["--security-opt", "no-new-privileges"]):  # fmt: skip
            assert any(cmd[i : i + 2] == flag for i in range(len(cmd))), flag
        assert "-e" not in cmd and "--env-file" not in cmd and f"{tmp_path.resolve()}:/repo:ro" in cmd
    assert all("canary-secret-123" not in str((kw.get("env") or {}).values()) for _, kw in seen)


def test_without_docker_or_the_local_opt_in_nothing_runs(monkeypatch):
    monkeypatch.delenv("OFFRAMP_ALLOW_LOCAL_SANDBOX", raising=False)
    llm = FakeLLM()  # no reply queued: the patcher must not be asked

    result = compose_patches(TINY, tiny_sites(), tiny_verdicts(), tiny_adapters(), llm, work_dir("nobox"),
                             config=CONFIG, sandbox_factory=lambda repo: Sandbox(repo, docker=False),
                             build_request=BUILD)  # fmt: skip

    assert result.sandbox_mode == "none"
    assert {verdict(result, s).reason for s in ("cs_01", "cs_03")} == {"no sandbox (Docker unavailable)"}
    assert result.files == {}


def test_a_block_without_its_own_file_line_belongs_to_the_previous_file():
    text = (
        "FILE: a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\ny = 1\n=======\ny = 2\n>>>>>>> REPLACE\n"
        "FILE: b.py\n<<<<<<< SEARCH\nz = 1\n=======\nz = 2\n>>>>>>> REPLACE\nNOTES: n"
    )

    assert [(e.path, e.search) for e in parse_edits(text).edits] == [("a.py", "x = 1"), ("a.py", "y = 1"),
                                                                      ("b.py", "z = 1")]  # fmt: skip


def test_a_block_labelled_with_the_wrong_file_applies_where_its_text_is_unique():
    text = '== "Hi Ana!"\n'  # only in the test file; the reply labels it with the module
    wrong = Edit(path="tinyapp/greet.py", search=text, replace='== "Hi Ana!"  # relabelled\n')
    llm = patcher_llm(PatchEdits(edits=[wrong]), migrate("tinyapp/shout.py"))

    result = run(llm)

    assert verdict(result, "cs_01").decision == "MIGRATE"
    assert "# relabelled" in result.files[TESTS]


def test_failure_summaries_are_stable_across_runs():
    """Repair feedback goes into the next request: run-specific text would make every replay a cache miss."""
    tail = "\nFAILED t.py::x - boom\n1 failed in {}"
    run_a = "E   assert isinstance(<ast.Constant object at 0x7f095ee535d0>, x)" + tail.format("5.58s")
    run_b = "E   assert isinstance(<ast.Constant object at 0x7f3851520a10>, x)" + tail.format("0.91s")

    assert _summary(run_a) == _summary(run_b)
    assert "FAILED t.py::x - boom" in _summary(run_a)
    cases = "".join(f"FAILED t.py::y[{i}] - boom {i}\n" for i in range(3))
    assert _summary(cases) == "FAILED t.py::y[…] (3 cases)"  # a parametrised test counts once


def test_a_symlink_loop_is_dropped_not_fatal():
    """Python 3.12's Path.resolve() raises on a loop; a hostile repo must not abort the run."""
    src = work_dir("loop") / "src"
    shutil.copytree(TINY, src)
    (src / "tinyapp" / "a").symlink_to("b")
    (src / "tinyapp" / "b").symlink_to("a")
    (src / "tinyapp" / "self").symlink_to("self")
    (src / "tinyapp" / "0x").symlink_to("missing/../a")  # dangling prefix, then the a/b loop (3.12 re-review)

    warnings = copy_repo(src, src.parent / "copy")

    copy = src.parent / "copy" / "tinyapp"
    assert not (copy / "self").is_symlink() and not (copy / "a").is_symlink()
    assert "dropped symlink tinyapp/self: it is a symlink loop" in warnings


def test_a_repo_root_package_holding_the_sites_has_no_place_for_the_client():
    repo = work_dir("rootpkg") / "src"
    shutil.copytree(TINY, repo)
    (repo / "__init__.py").write_text("")
    (repo / "greet.py").write_text((repo / "tinyapp" / "greet.py").read_text())
    site = tiny_sites()[0].model_copy(update={"file": "greet.py"})

    work = repo.parent / "w"
    result = compose_patches(repo, [site], [tiny_verdicts()[0]], tiny_adapters(), FakeLLM(), work,
                             config=CONFIG, sandbox_factory=FakeSandbox, build_request=BUILD)  # fmt: skip

    (v,) = result.verdicts
    assert v.decision == "MANUAL_REVIEW" and "repository root, which is itself a Python package" in v.reason
    assert result.files == {}


# ---------------------------------------------------------------- Task 10b: long strings, repair, composition


LONG = "Reply with the final refund amount in the customer's currency alone on the last line as ANSWER."


@pytest.mark.parametrize(
    "line",
    [
        f'    content = "{LONG} {LONG}"\n',
        f'    content = f"Plan: {{plan}}, {{days}} days\\n\\n{LONG} {{{{literal braces}}}} {LONG}"\n',
        f'    content = f"{{d["key"]}} and {{x!r:>{{w}}}} {LONG} {LONG}"\n',
        f'    """{LONG} {LONG}"""\n',
    ],
    ids=["plain", "f-string", "f-string-nested-quotes", "triple-quoted"],
)
def test_long_strings_on_patched_lines_are_wrapped_without_changing_their_values(line):
    source = f"def f(plan, days, d, x, w):\n{line}    return content\n"

    wrapped = wrap_long_strings(source, {2}, 88)

    assert ast.dump(ast.parse(wrapped)) == ast.dump(ast.parse(source))
    assert max(map(len, wrapped.splitlines())) <= 88
    assert "noqa" not in wrapped
    for piece in wrapped.splitlines():  # a piece with no field is a plain string: no ruff F541
        assert not piece.strip().startswith(("f'", 'f"')) or "{" in piece


def test_only_lines_the_patch_touched_are_wrapped():
    source = f'a = "{LONG} {LONG}"\nb = "{LONG} {LONG}"\n'

    wrapped = wrap_long_strings(source, {2}, 88)

    assert wrapped.splitlines()[0] == source.splitlines()[0]
    assert ast.dump(ast.parse(wrapped)) == ast.dump(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    [f'x = b"{LONG} {LONG}"\n', f'x = "{LONG}" "{LONG}"\n', f'x = f"""{{a}} {LONG} {LONG}"""\n'],
    ids=["bytes", "implicit-concatenation", "triple-quoted-f-string"],
)
def test_strings_that_cannot_be_split_safely_are_left_alone(source):
    assert wrap_long_strings(source, {1}, 88) == source


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"pyproject.toml": "[tool.ruff]\nline-length = 100\n"}, 100),
        ({"ruff.toml": "line-length = 79\n", "pyproject.toml": "[tool.ruff]\nline-length = 100\n"}, 79),
        ({".ruff.toml": "line-length = 120\n"}, 120),
        ({"pyproject.toml": "[project]\nname = 'x'\n"}, 88),
        ({}, 88),
    ],
    ids=["pyproject", "ruff.toml-wins", "dot-ruff.toml", "no-ruff-section", "nothing"],
)
def test_line_length_comes_from_the_targets_ruff_config(files, expected):
    repo = work_dir("ruffcfg")
    repo.mkdir(parents=True)
    for name, text in files.items():
        (repo / name).write_text(text)

    assert line_length(repo) == expected


BASE = 'import os\nCHAT = {\n    "qa": "gpt-4o",\n    "refund": "o3",\n    "sum": "haiku",\n}\n'


def test_patches_on_adjacent_lines_of_one_file_compose():
    ours = BASE.replace("import os\n", "import os\nfrom pkg.client import tf\n").replace('"gpt-4o"', '"nemo"')
    theirs = BASE.replace("import os\n", "import os\nfrom pkg.client import tf\n").replace('"o3"', '"nemo"')

    merged = merge_onto(BASE, ours, theirs)

    assert merged == ours.replace('"o3"', '"nemo"')  # both rows changed, the shared import once
    assert merge_onto(BASE, merged, theirs) == merged  # already applied: nothing to do


def test_patches_changing_the_same_line_differently_conflict():
    assert merge_onto(BASE, BASE.replace('"o3"', '"a"'), BASE.replace('"o3"', '"b"')) is None


def test_a_repair_that_repeats_the_failed_patch_stops_without_another_check():
    breaking = migrate("tinyapp/greet.py", extra=[BREAK_GREET])
    llm = patcher_llm(breaking, breaking, migrate("tinyapp/shout.py"))
    boxes = []

    result = run(llm, results=[True, True, False, True, True], sandboxes=boxes)

    assert verdict(result, "cs_01").reason.startswith(
        "repair produced no change; last failure: repo checks failed"
    )
    assert len(boxes[0].checks) == 6  # untouched, base, greet alone + one try, shout alone + its try
    repair = llm.calls[1]
    assert repair["max_tokens"] > llm.calls[0]["max_tokens"] and repair["reasoning"] == "off"
    assert "edit blocks only" in repair["messages"][-1]["content"]


def test_each_site_is_patched_alone_then_all_together():
    mark = Edit(path="tests/test_tinyapp.py", search="from types import SimpleNamespace",
                replace="from types import SimpleNamespace  # both sites")  # fmt: skip
    llm = patcher_llm(migrate("tinyapp/greet.py", extra=[mark]), migrate("tinyapp/shout.py", extra=[mark]))
    boxes = []

    result = run(llm, sandboxes=boxes)

    combined = boxes[0].checks[-1]
    assert MIGRATED in combined["tinyapp/greet.py"] and MIGRATED in combined["tinyapp/shout.py"]
    assert combined["tests/test_tinyapp.py"].count("# both sites") == 1
    assert MIGRATED not in boxes[0].checks[5]["tinyapp/greet.py"]  # shout was checked alone
    assert result.alone == {"cs_01": True, "cs_03": True} and result.combined == ["cs_01", "cs_03"]
    assert result.files["tests/test_tinyapp.py"].count("# both sites") == 1


def test_sites_green_alone_but_red_together_get_one_combined_repair_on_ultra():
    fix = PatchEdits(
        edits=[Edit(path=TESTS, search="from types import", replace="# together\nfrom types import")]
    )
    llm = patcher_llm(migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"), fix)

    # untouched, base, greet alone, shout alone, all together (red), after the combined repair (green)
    result = run(llm, results=[True, True, True, True, False, True])

    assert result.combined == ["cs_01", "cs_03"] and result.combined_repair == [TESTS]
    assert result.files[TESTS].startswith("# together\n") and result.combined_failure is None
    repair = llm.calls[2]
    assert (repair["tier"], repair["reasoning"], repair["messages"][0]["content"]) == (
        Tier.ULTRA,
        "off",
        TOGETHER,
    )
    prompt = repair["messages"][-1]["content"]
    assert "- cs_01 `tinyapp/greet.py` `greet`" in prompt and "With all of them merged, this repo's" in prompt
    assert "### tinyapp/greet.py (read-only)" in prompt and f"### {TESTS} (editable)" in prompt


def test_a_site_green_alone_but_not_together_is_dropped_from_the_combined_tree():
    module = PatchEdits(edits=[Edit(path="tinyapp/shout.py", search=".upper()", replace="")])
    llm = patcher_llm(
        migrate("tinyapp/greet.py"),
        migrate("tinyapp/shout.py"),
        module,
        *[
            PatchEdits(
                edits=[Edit(path=TESTS, search="from types import", replace=f"# try {i}\nfrom types import")]
            )
            for i in (1, 2)
        ],
    )

    # untouched, base, greet alone, shout alone, together (red), repair 2 (red: repair 1 edits a module),
    # then one at a time: greet (green), greet + shout (red)
    result = run(llm, results=[True, True, True, True, False, False, True, False])

    assert [c["tier"] for c in llm.calls[2:]] == [Tier.ULTRA, Tier.ULTRA]  # two combined repairs at most
    assert "tinyapp/shout.py is migrated by Offramp and read-only" in llm.calls[3]["messages"][-1]["content"]
    assert result.alone == {"cs_01": True, "cs_03": True} and result.combined == ["cs_01"]
    assert result.combined_failure.startswith("repo checks failed on the merged patches")
    shout = verdict(result, "cs_03")
    assert shout.decision == "MANUAL_REVIEW" and "green alone, but not together with cs_01" in shout.reason
    assert "tinyapp/shout.py" not in result.files and verdict(result, "cs_01").decision == "MIGRATE"
    assert "# try" not in result.files[TESTS]  # the failed repair is not in the tree


def test_a_long_piece_of_a_multi_line_concatenation_is_split_in_place():
    """cs_05's recorded patch: the suffix became one more 208-column piece of an existing concatenation."""
    source = (
        "def f(request):\n"
        "    return [\n"
        "        {\n"
        '            "content": (\n'
        '                f"Plan: {request.plan_name}\\n"\n'
        f'                f"How much do we refund, in {{request.currency}}?\\n\\n{LONG} {LONG}"\n'
        "            ),\n"
        "        }\n"
        "    ]\n"
    )

    wrapped = wrap_long_strings(source, {6}, 100)

    assert ast.dump(ast.parse(wrapped)) == ast.dump(ast.parse(source))
    assert max(map(len, wrapped.splitlines())) <= 100 and "((" not in wrapped


def test_a_test_using_a_pytest_fixture_it_never_requested_gets_it_as_a_parameter():
    source = (
        "import pytest\n\n\n"
        "def test_a():\n    monkeypatch.setattr('x.y', 1)\n\n\n"
        "@pytest.mark.parametrize('n', [1])\n"
        "def test_b(n, capsys):\n    monkeypatch.setenv('A', '1')\n    capsys\n\n\n"
        "def test_c(monkeypatch):\n    monkeypatch.setenv('A', '1')\n\n\n"
        "def helper():\n    monkeypatch = 1\n    return monkeypatch\n\n\n"
        "def tmp_path(**kw):\n    return kw\n\n\n"
        "def test_d():\n    assert tmp_path(a=1)\n"
    )

    fixed = add_fixtures(source)

    assert "def test_a(monkeypatch):" in fixed and "def test_b(n, capsys, monkeypatch):" in fixed
    assert "def test_c(monkeypatch):" in fixed and "def helper():" in fixed
    assert "def test_d():" in fixed  # tmp_path is the module's own helper here, not the fixture
    assert add_fixtures(fixed) == fixed


def test_fixture_parameters_land_in_the_signature_even_with_a_comment_or_in_a_class():
    source = (
        "class TestX:\n"
        "    def test_m(self):  # covers (a) and (b): done\n"
        "        with open('f'):\n"
        "            monkeypatch.chdir('.')\n"
    )

    fixed = add_fixtures(source)

    assert "    def test_m(self, monkeypatch):  # covers (a) and (b): done\n" in fixed
    assert "        with open('f'):\n" in fixed


def test_a_test_body_on_the_signature_line_does_not_crash_the_run():
    """A model reply put the body on the `):` line: locating the signature must not raise (TokenError)."""
    source = "def test_x(\n    a,\n): monkeypatch.setattr('x', 1)\n"
    assert add_fixtures(source) == source

    odd_test = "def test_motto_returns_the_reply(\n): monkeypatch.setenv('A', '1')\n\n\ndef _unused():\n"
    odd = Edit(path="tests/test_tinyapp.py", search="def test_motto_returns_the_reply():\n", replace=odd_test)
    llm = patcher_llm(migrate("tinyapp/greet.py", extra=[odd]), migrate("tinyapp/shout.py"))

    result = run(llm)

    assert (
        verdict(result, "cs_01").decision == "MIGRATE"
    )  # decided, not crashed (the fake sandbox says green)


def test_two_repair_rounds_each_see_the_last_failure():
    first = migrate("tinyapp/greet.py", extra=[BREAK_GREET])
    second = migrate("tinyapp/greet.py", extra=[BREAK_GREET_2])
    llm = patcher_llm(first, second, migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))

    result = run(llm, results=[True, True, False, False, True, True, True])

    assert verdict(result, "cs_01").decision == "MIGRATE" and result.alone["cs_01"]
    third = llm.calls[2]
    assert third["max_tokens"] == 12000 and len(third["messages"]) == 6
    assert third["messages"][-2]["content"] == edits_text(second)


def test_a_repair_that_returns_to_an_earlier_patch_stops_too():
    a = migrate("tinyapp/greet.py", extra=[BREAK_GREET])
    b = migrate("tinyapp/greet.py", extra=[BREAK_GREET_2])
    boxes = []

    result = run(patcher_llm(a, b, a, migrate("tinyapp/shout.py")), results=[True, True, False, False, True],
                 sandboxes=boxes)  # fmt: skip

    assert verdict(result, "cs_01").reason.startswith("repair produced no change; last failure:")
    assert len(boxes[0].checks) == 7  # untouched, base, greet alone, a, b, shout alone, shout: a once


@pytest.mark.parametrize(
    "module_binding",
    [
        "tmp_path: str = 'x'\n",
        "try:\n    tmp_path = 1\nexcept ImportError:\n    pass\n",
        "from helpers import *\n",
    ],
    ids=["annotated", "inside-try", "star-import"],
)
def test_a_module_name_that_might_be_the_fixture_is_never_shadowed(module_binding):
    source = f"{module_binding}\n\ndef test_x():\n    assert tmp_path\n"

    assert add_fixtures(source) == source


def test_repairs_run_on_ultra_with_thinking_off():
    breaking = migrate("tinyapp/greet.py", extra=[BREAK_GREET])
    llm = patcher_llm(breaking, migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))

    run(llm, results=[True, True, False, True, True])

    assert [(c["tier"], c["reasoning"]) for c in llm.calls] == [
        (Tier.SUPER, "off"), (Tier.ULTRA, "off"), (Tier.SUPER, "off")]  # fmt: skip


OPENAI_SITE = """def triage(client, subject):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=16,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Subject: {subject}"},
        ],
    )
    return response.choices[0].message.content
"""
ANTHROPIC_SITE = """def step(client, text):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=PROMPT,
        tools=TOOLS,
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": (f"Customer: {text}\\n" "Choose.")}],
    )
    return response.content
"""
REQUEST = {
    "model": LIGHTNING,
    "max_tokens": 16,
    "temperature": 0.0,
    "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
    "tool_choice": "required",
    "extra_body": THINK_OFF,
}


def _site(function, sdk="openai"):
    return CallSite(id="cs_09", file="m.py", lines=[2, 10], sdk=sdk, call="c", function=function,
                    prompt_source="fstring")  # fmt: skip


@pytest.mark.parametrize(("source", "function"), [(OPENAI_SITE, "triage"), (ANTHROPIC_SITE, "step")])
def test_the_call_is_rewritten_by_code_to_the_evaluated_request(source, function):
    constants = {"few_shot": "NEMOTRON_FEW_SHOT", "suffix": "NEMOTRON_USER_SUFFIX", "tools": "NEMOTRON_TOOLS"}

    out = _rewrite_call(source, _site(function), constants, "PROMPT", REQUEST, 100)

    call = next(n for n in ast.walk(ast.parse(out)) if isinstance(n, ast.Call) and n.keywords)
    assert ast.unparse(call.func).endswith("chat.completions.create") and "messages.create" not in out
    kwargs = {k.arg: k.value for k in call.keywords}
    assert list(kwargs) == [*REQUEST, "messages"]  # written out, in order; nothing else passed
    assert ast.unparse(kwargs["tools"]) == "NEMOTRON_TOOLS"  # a schema goes by its constant's name
    assert {k: ast.literal_eval(v) for k, v in kwargs.items() if k not in ("tools", "messages")} == {
        k: v for k, v in REQUEST.items() if k != "tools"
    }
    first, few, user = kwargs["messages"].elts
    assert ast.unparse(first) == "{'role': 'system', 'content': PROMPT}"
    assert ast.unparse(few) == "*NEMOTRON_FEW_SHOT"
    assert ast.unparse(user.values[1]).endswith("+ NEMOTRON_USER_SUFFIX")
    assert "system=" not in out and "gpt-4o-mini" not in out and "claude" not in out


def test_a_call_whose_messages_are_built_elsewhere_is_left_to_the_model():
    source = "def f(client, msgs):\n    return client.chat.completions.create(model='m', messages=msgs)\n"

    assert _rewrite_call(source, _site("f"), {}, None, REQUEST, 100) is None


def _rewritten(edit=None):
    out = _rewrite_call(OPENAI_SITE, _site("triage"), {"tools": "NEMOTRON_TOOLS"}, "PROMPT", REQUEST, 100)
    out = f"PROMPT = 'p'\nNEMOTRON_TOOLS = {REQUEST['tools']!r}\n\n\n{out}"
    return out if edit is None else out.replace(*edit)


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (("max_tokens=16,", "max_tokens=64,"), "`max_tokens` must be exactly"),
        (("temperature=0.0,", "temperature=0.7,"), "`temperature` must be exactly"),
        (('"enable_thinking": False', '"enable_thinking": True'), "`extra_body` must be exactly"),
        ((f'model="{LIGHTNING}",', 'model="gpt-4o-mini",'), "`model` must be exactly"),
        (("tool_choice=", "top_p=0.5,\n        tool_choice="), "must pass exactly"),
        (("max_tokens=16,", "**EXTRA,"), "no **"),
        (("'type': 'object'", "'type': 'array'"), "`tools` must be exactly"),
        (('"content": PROMPT', '"content": "Be brief."'), "system prompt constant PROMPT is not used"),
    ],
    ids=["max-tokens", "temperature", "thinking", "model", "extra-kwarg", "splat", "schema", "system-prompt"],
)
def test_a_call_that_strays_from_the_evaluated_request_is_not_conformant(edit, problem):
    assert conformance(_rewritten(), _site("triage"), REQUEST, {"tools": "NEMOTRON_TOOLS"}, "PROMPT") == []

    problems = conformance(_rewritten(edit), _site("triage"), REQUEST, {"tools": "NEMOTRON_TOOLS"}, "PROMPT")

    assert any(problem in p for p in problems), problems


TEXT_SITE = """def summarize(client: Anthropic, thread):
    response = client.chat.completions.create(model="m", messages=[])
    return "".join(block.text for block in response.content if block.type == "text").strip()
"""
FIRST_TEXT_SITE = """def summarize(client, thread):
    response = client.chat.completions.create(model="m", messages=[])
    reply = response.content[0].text
    return reply
"""
TOOL_SITE = """def step(client: Anthropic, text):
    response = client.chat.completions.create(model="m", messages=[])
    for block in response.content:
        if block.type == "tool_use" and block.name in NAMES:
            return Action(tool=block.name, args=dict(block.input), ref=block.id)
    return None
"""


@pytest.mark.parametrize(
    ("source", "function", "expected"),
    [
        (TEXT_SITE, "summarize", 'return (response.choices[0].message.content or "").strip()'),
        (FIRST_TEXT_SITE, "summarize", 'reply = response.choices[0].message.content or ""'),
        (TOOL_SITE, "step", "for block in response.choices[0].message.tool_calls or []:\n"
         '        if block.type == "function" and block.function.name in NAMES:\n'
         "            return Action(tool=block.function.name, "
         "args=dict(json.loads(block.function.arguments)), ref=block.id)"),
    ],
    ids=["joined-text", "first-text", "tool-loop"],
)  # fmt: skip
def test_an_anthropic_reply_is_read_the_openai_way_by_code(source, function, expected):
    out = _map_reply(source, _site(function, sdk="anthropic"))

    assert expected in out and "response.content" not in out
    assert "Anthropic" not in out or "client: OpenAI" in out  # the client's annotation follows
    assert _map_reply(OPENAI_SITE, _site("triage")) == OPENAI_SITE  # an OpenAI site already reads it so


@pytest.mark.parametrize(
    "source",
    [
        ANTHROPIC_SITE.replace(
            "client.messages.create", "client.chat.completions.create"
        ),  # returns it whole
        TOOL_SITE.replace("block.id", "block.text"),  # a text block read inside the tool loop
        TEXT_SITE.replace(".strip()", " + str(response.usage.input_tokens)"),
    ],
    ids=["whole-reply", "mixed-loop", "usage"],
)
def test_a_reply_used_any_other_way_is_left_to_the_model(source):
    function = "step" if "def step" in source else "summarize"

    assert _map_reply(source, _site(function, sdk="anthropic")) is None


FAKES = """import inspect
from types import SimpleNamespace

from anthropic.resources.messages import Messages
from openai.resources.chat.completions import Completions


def _check_kwargs(method, kwargs):
    assert set(kwargs) <= set(inspect.signature(method).parameters)


class FakeAnthropic:
    \"\"\"Replies with the given content blocks.\"\"\"

    def __init__(self, *blocks):
        self.blocks = list(blocks)
        self.calls = []
        self.messages = SimpleNamespace(create=self._messages)

    def _messages(self, **kwargs):
        _check_kwargs(Messages.create, kwargs)
        self.calls.append(kwargs)
        return Message(id="msg_fake", content=self.blocks, stop_reason="end_turn")
"""


def test_an_anthropic_fake_also_answers_the_openai_call_with_the_same_blocks():
    out = convert_fakes(FAKES, "helphub.offramp_client")

    ast.parse(out)
    assert "    def _chat_completion(self, **kwargs):\n" in out
    assert "        _check_kwargs(Completions.create, kwargs)\n        self.calls.append(kwargs)\n" in out
    assert "        return chat_completion_from_blocks(self.blocks)\n" in out
    assert "self.chat = SimpleNamespace(" in out and "create=self._chat_completion" in out
    assert out.index("def chat_completion_from_blocks") < out.index("class FakeAnthropic")
    assert "import json" in out and "from openai.types.chat import ChatCompletion" in out
    assert convert_fakes(out, "helphub.offramp_client") == out  # once only


def test_the_fake_helper_turns_blocks_into_a_real_chat_completion():
    space: dict = {"json": json, "ChatCompletion": ChatCompletion}
    exec(FAKE_HELPER, space)  # noqa: S102 - our own constant
    text = SimpleNamespace(type="text", text="Hello")
    tool = SimpleNamespace(type="tool_use", id="toolu_1", name="lookup_order", input={"order_id": "HH-1"})

    message = space["chat_completion_from_blocks"]([text, tool]).choices[0].message
    empty = space["chat_completion_from_blocks"]([]).choices[0].message

    assert message.content == "Hello" and message.tool_calls[0].function.name == "lookup_order"
    assert json.loads(message.tool_calls[0].function.arguments) == {"order_id": "HH-1"}
    assert empty.content is None and empty.tool_calls is None


@pytest.mark.parametrize(
    "literal", ['"a string that fits on its own line"', 'f"Subject: {s}, thirty-odd characters"']
)
def test_a_literal_that_fits_alone_moves_to_its_own_line(literal):
    """cs_01's live patch: a test assert ran to 101 columns although its string itself was short."""
    check = f"call['messages'][1]['content'] == {literal} and call['n'] == 123456"
    source = f"def test_x(s, call):\n    assert {check}\n"

    wrapped = wrap_long_strings(source, {2}, 88)

    assert ast.dump(ast.parse(wrapped)) == ast.dump(ast.parse(source))
    assert max(map(len, wrapped.splitlines())) <= 88


CALLER = """from openai import OpenAI

from helphub.inbox.triage import triage_ticket

COMMANDS = {
    "triage": lambda c: triage_ticket(OpenAI(), c["subject"], c["body"]),
    "search": lambda c: search_kb(OpenAI(), c["q"]),
}
"""


def test_callers_pass_the_token_factory_client_to_the_migrated_function_only():
    out = rewrite_callers(CALLER, "triage_ticket", "helphub.offramp_client")

    assert 'triage_ticket(token_factory_client(), c["subject"], c["body"])' in out
    assert 'search_kb(OpenAI(), c["q"])' in out  # other functions keep their client
    assert "from helphub.offramp_client import token_factory_client" in out
    assert rewrite_callers(out, "triage_ticket", "helphub.offramp_client") == out


def test_tests_that_fake_the_openai_constructor_fake_the_token_factory_client_too():
    source = (
        "def test_runs(monkeypatch, capsys):\n"
        '    monkeypatch.setattr(cli, "OpenAI", lambda: FakeOpenAI(reply))\n'
        '    monkeypatch.setattr(cli, "Anthropic", lambda: FakeAnthropic())\n'
        '    monkeypatch.setattr(other, "OpenAI", lambda: FakeOpenAI(reply))\n'
    )

    out = mirror_fakes(source, {"cli"})

    assert (
        out.splitlines()[2]
        == '    monkeypatch.setattr(cli, "token_factory_client", lambda: FakeOpenAI(reply))'
    )
    assert 'other, "token_factory_client"' not in out  # not a module whose code was changed
    assert mirror_fakes(out, {"cli"}) == out
    anthropic = mirror_fakes(source, {"cli"}, "Anthropic")  # an Anthropic site: its (converted) fake
    assert anthropic.splitlines()[3] == (
        '    monkeypatch.setattr(cli, "token_factory_client", lambda: FakeAnthropic())'
    )


BUILT_GREET = """GREET_SYSTEM_PROMPT = "You write one-line greetings."


def greet(client, name: str) -> str:
    messages = [
        {"role": "system", "content": GREET_SYSTEM_PROMPT},
        {"role": "user", "content": f"Greet {name}."},
    ]
    response = client.chat.completions.create(model="gpt-4o-mini", max_tokens=32, messages=messages)
    return (response.choices[0].message.content or "").strip()
"""


def tiny_built() -> tuple[Path, Edit]:
    """The tiny repo with greet's messages built before the call (the code cannot rewrite that call), and
    the edit a model would write for it."""
    repo = work_dir("tiny-built") / "src"
    shutil.copytree(TINY, repo)
    (repo / "tinyapp" / "greet.py").write_text(BUILT_GREET)
    old = 'create(model="gpt-4o-mini", max_tokens=32, messages=messages)'
    thinking = 'extra_body={"chat_template_kwargs": {"enable_thinking": False}}'
    args = [MIGRATED, "max_tokens=32,", "temperature=0.0,", f"{thinking},", "messages=messages,"]
    new = "create(\n" + "".join(f"        {a}\n" for a in args) + "    )"
    return repo, Edit(path="tinyapp/greet.py", search=old, replace=new)


def test_a_call_the_code_cannot_rewrite_is_left_to_the_model_and_checked():
    """greet's messages are built before the call: the model writes the call, the checks still apply."""
    repo, write = tiny_built()
    drop = Edit(path="tinyapp/greet.py", search="def greet(client, name", replace="def greet(name")
    llm = patcher_llm(
        PatchEdits(edits=[write]), PatchEdits(edits=[write, drop]), *[PatchEdits(edits=[drop])] * 2
    )

    good = run(llm, repo=repo, name="tiny-built-run", precheck=True)  # shout: green on Offramp's own changes
    bad = run(llm, repo=repo, name="tiny-built-run", precheck=True)

    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "Rewrite the call in `tinyapp/greet.py`" in prompt and "### tinyapp/greet.py (editable)" in prompt
    assert verdict(good, "cs_01").decision == "MIGRATE" and MIGRATED in good.files["tinyapp/greet.py"]
    reason = verdict(bad, "cs_01").reason  # the parameters changed; a repair without the call fails too
    assert "signature of `greet` changed" in reason and "must pass exactly" in reason


def test_a_repair_stopped_by_the_budget_keeps_the_first_failure_in_the_reason():
    breaking = migrate("tinyapp/greet.py", extra=[BREAK_GREET])
    llm = patcher_llm(
        breaking, BudgetExceeded("Ultra agent call cap reached (0)"), migrate("tinyapp/shout.py")
    )

    reason = verdict(run(llm, results=[True, True, False, True]), "cs_01").reason

    assert reason.startswith(
        "patch not composed: Ultra agent call cap reached (0); last failure: repo checks failed"
    )


def test_once_a_literal_is_split_the_shorter_one_left_of_it_stays():
    """cs_01's live patch: after its value moved to its own line, the dict key was wrapped as well."""
    value = "Subject: Order confirmation email not received and order missing from dashboard"
    lines = [
        "def test_x(call):",
        "    assert call == {",
        '        "role": "user",',
        f'        "content": "{value}",',
    ]
    source = "\n".join([*lines, "    }", ""])

    wrapped = wrap_long_strings(source, {4}, 88)

    assert '        "content": (\n' in wrapped and '("content")' not in wrapped
    assert ast.dump(ast.parse(wrapped)) == ast.dump(ast.parse(source))


SOURCES_TEST = """def test_request_contract():
    client = FakeOpenAI("Within 30 days [returns].")

    reply = answer_question(client, "refund?", [RETURNS])

    assert reply.sources == ["returns"]
    (call,) = client.calls
    assert call["model"] == "gpt-4o"
    user = call["messages"][1]["content"]
    assert "[returns]" in user
"""


def test_a_long_block_copied_with_a_slip_lands_only_its_intended_changes():
    """cs_06's Ultra repair: a 45-line SEARCH with one value mis-copied was rejected as not found. Review I1
    (probe 2): applied over the window, the slip then replaced the real assertion with its copy."""
    slip = SOURCES_TEST.rstrip("\n").replace('assert reply.sources == ["returns"]', "assert reply.sources")
    intended = slip.replace('== "gpt-4o"', f'== "{LIGHTNING}"').replace('["messages"][1]', '["messages"][-1]')
    files = {"t.py": SOURCES_TEST}

    assert _apply([Edit(path="t.py", search=slip, replace=intended)], files) == []

    assert 'assert reply.sources == ["returns"]\n' in files["t.py"]  # the real assertion is intact
    assert f'== "{LIGHTNING}"' in files["t.py"] and '["messages"][-1]' in files["t.py"]
    gone = intended.replace("    assert reply.sources\n", "")  # the slip's own line removed: a conflict
    assert "not found" in _apply([Edit(path="t.py", search=slip, replace=gone)], {"t.py": SOURCES_TEST})[0]
    short = Edit(path="t.py", search='    reply = x\n    assert reply.sources == ["returns"]', replace="x")
    assert "not found" in _apply([short], {"t.py": SOURCES_TEST})[0]  # a short block must match exactly
    twice = {"t.py": SOURCES_TEST + "\n\n" + SOURCES_TEST.replace("test_request_contract", "test_again")}
    lone = Edit(path="t.py", search=slip.replace("def test_request_contract():", ""), replace="x")
    assert "not found" in _apply([lone], twice)[0]  # more than one place: never guess


def test_a_repair_keeps_the_previous_edits_of_files_it_leaves_out():
    """Repairs that resent 'a complete new set' kept dropping blocks that had worked (cs_06, live)."""
    repo, write = tiny_built()
    fix = migrate("tinyapp/greet.py")  # names the test file only: greet.py keeps the call written in try 1
    llm = patcher_llm(PatchEdits(edits=[write, BREAK_GREET]), fix)

    result = run(llm, results=[True, True, False], repo=repo, name="tiny-built-run", precheck=True)

    assert "a file you do not name keeps your previous edits" in llm.calls[1]["messages"][-1]["content"]
    assert verdict(result, "cs_01").decision == "MIGRATE" and MIGRATED in result.files["tinyapp/greet.py"]
    assert '"Hi Ana!"' in result.files[TESTS]  # the test file is the repair's own version


DISPATCH = """COMMANDS = {"triage": lambda c: triage_ticket(token_factory_client(), c["subject"])}


def main(argv):
    if argv[0] == "search":
        return search_kb(OpenAI(), argv[1])
    elif argv[0] == "extract":
        return extract_order(token_factory_client(), argv[1])


def run_summary(thread):
    return summarize_thread(token_factory_client(), thread)
"""


def test_a_test_reaches_a_migrated_function_through_its_command_key_or_branch():
    assert reach_handles(DISPATCH, "triage_ticket") == {"triage_ticket", "triage"}  # a command table
    assert reach_handles(DISPATCH, "extract_order") == {"extract_order", "extract"}  # an argument branch
    assert reach_handles(DISPATCH, "summarize_thread") == {"summarize_thread", "run_summary"}  # a wrapper


def test_fake_twins_stay_only_in_tests_that_reach_the_migrated_function():
    twin = '    monkeypatch.setattr(cli, "token_factory_client", lambda: FakeOpenAI("x"))\n'
    other = '    monkeypatch.setattr(cli, "OpenAI", lambda: FakeOpenAI("x"))\n'
    source = (
        '@pytest.mark.parametrize("command", ["triage", "answer"])\n'
        f"def test_runs(monkeypatch, command):\n{other}{twin}    run([command])\n\n\n"
        f'def test_search(monkeypatch):\n{other}{twin}    run(["search", "q"])\n'
    )

    out = prune_twins(source, {twin}, {"triage_ticket", "triage"})

    assert out.count(twin) == 1 and out.index(twin) < out.index("def test_search")  # kept where reached
    assert out.count(other) == 2  # only Offramp's own twins are ever removed


def test_a_combined_repair_may_change_only_the_failing_tests_and_fakes():
    """Ultra's first live combined repair rewrote the fakes and tests that were passing; the next one then
    'fixed' a behavioural assertion it had broken."""
    weaken = Edit(path=TESTS, search='== "Ship it"', replace='== "Ship it!"')  # test_motto: not failing
    fix = Edit(path=TESTS, search="from types import", replace="# together\nfrom types import")
    llm = patcher_llm(migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"), PatchEdits(edits=[weaken]),
                      PatchEdits(edits=[fix]))  # fmt: skip

    result = run(llm, results=[True, True, True, True, False, True])  # FakeSandbox fails tests/...::x

    assert (
        "only the failing tests may change: tests/test_tinyapp.py::test_motto_returns_the_reply did"
        in (llm.calls[3]["messages"][-1]["content"])
    )
    assert result.combined == ["cs_01", "cs_03"] and '"Ship it!"' not in result.files[TESTS]


CLI_TEST = """@pytest.mark.parametrize(
    ("command", "openai_reply", "blocks"),
    [("triage", "billing", []), ("next-step", None, ["b"]), ("summarize", None, ["s"]), ("refund", "r", [])],
)
def test_runs(monkeypatch, command, openai_reply, blocks):
    monkeypatch.setattr(cli, "OpenAI", lambda: FakeOpenAI(openai_reply))
    monkeypatch.setattr(cli, "token_factory_client", lambda: FakeOpenAI(openai_reply))
    monkeypatch.setattr(cli, "Anthropic", lambda: FakeAnthropic(*blocks))
    monkeypatch.setattr(
        cli, "token_factory_client", lambda: FakeAnthropic(*blocks)
    )
    run([command])
"""
OPENAI_TWIN = '    monkeypatch.setattr(cli, "token_factory_client", lambda: FakeOpenAI(openai_reply))\n'
ANTHROPIC_TWIN = '    monkeypatch.setattr(cli, "token_factory_client", lambda: FakeAnthropic(*blocks))\n'


def test_twins_of_sites_from_both_sdks_become_one_fake_chosen_per_case():
    """Run 2 (round 2): merged, the Anthropic twin served every case and the OpenAI sites' cases failed."""
    sites = [({OPENAI_TWIN}, {"triage_ticket", "triage"}), ({ANTHROPIC_TWIN}, {"next_action", "next-step"}),
             ({ANTHROPIC_TWIN}, {"summarize_thread", "summarize"})]  # fmt: skip

    out = unify_twins(CLI_TEST, sites)

    assert out.count('"token_factory_client"') == 1 and 'cli, "OpenAI"' in out and 'cli, "Anthropic"' in out
    assert (
        '        (lambda: FakeAnthropic(*blocks))\n        if command in ("next-step", "summarize")\n'
        "        else (lambda: FakeOpenAI(openai_reply)),\n    )\n    run([command])\n"
    ) in out
    assert unify_twins(CLI_TEST, sites[:1]) == CLI_TEST  # one site's twin: nothing to choose between
    unnamed = CLI_TEST.replace('"next-step"', '"other"').replace('"summarize"', '"other"')
    assert unify_twins(unnamed, sites) == unnamed  # no parameter tells the sites apart: left to the repair


def test_a_path_in_the_check_output_never_reaches_a_host_file():
    """Review C1 (probe 7): the target's tests print a FAILED line naming a file outside the copy."""
    host = work_dir("host-secret") / "host_secret" / "tests" / "creds.py"
    host.parent.mkdir(parents=True)
    host.write_text("TOKEN = 'secret-value'\n")
    rel = os.path.relpath(host, ROOT / "var" / "test-sandbox" / "tiny-c1" / "repo")
    planted = f"FAILED {rel}::test_x\nFAILED tests/test_tinyapp.py::x"
    steal = PatchEdits(
        edits=[Edit(path=rel, search="TOKEN = 'secret-value'", replace="TOKEN = 'overwritten'")]
    )
    llm = patcher_llm(migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"), steal, steal)

    result = run(llm, results=[True, True, True, True, False], name="tiny-c1", output=planted)

    assert host.read_text() == "TOKEN = 'secret-value'\n"  # never written
    sent = [m["content"] for c in llm.calls for m in c["messages"] if m["role"] != "assistant"]
    assert all("secret-value" not in m and f"### {rel}" not in m for m in sent)  # never read or shown
    assert rel not in result.files and rel not in result.combined_repair
    assert f"combined repair: ignored {rel} from the check output: not a tracked test or fake file" in (
        result.warnings
    )
    assert f"{rel} is not editable for this site" in llm.calls[3]["messages"][-1]["content"]


BACKENDS = """class OpenAIBackend:
    def complete(self, client, text):
        response = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=16, messages=[{"role": "user", "content": f"Q: {text}"}]
        )
        return response.choices[0].message.content


class AnthropicBackend:
    def complete(self, client, text):
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=16,
            system="Be brief.",
            messages=[{"role": "user", "content": f"Q: {text}"}],
        )
        return response.content[0].text
"""


def test_two_methods_sharing_the_name_are_told_apart_by_the_site_lines():
    """Review I3 (probe 4): the first `complete` in the module was rewritten with the other site's request."""
    site = _site("complete", sdk="anthropic").model_copy(update={"lines": [11, 16]})
    local = site.model_copy(update={"function": site_qualname(BACKENDS, site)})

    out = _map_reply(_rewrite_call(BACKENDS, local, {}, None, REQUEST | {"tools": None}, 100), local)

    assert local.function == "AnthropicBackend.complete"
    openai_half, anthropic_half = out.split("class AnthropicBackend")
    assert 'model="gpt-4o-mini"' in openai_half and "messages.create" not in out  # the other method untouched
    assert (
        f'model="{LIGHTNING}"' in anthropic_half
        and '{"role": "system", "content": "Be brief."}' in anthropic_half
    )
    nowhere = site.model_copy(update={"lines": [19, 19]})  # inside neither: no guess
    assert site_qualname(BACKENDS + "\n\ndef complete():\n    pass\n", nowhere) is None


ANTHROPIC_SYSTEM = """SYSTEM = [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}]


def ask(client, text):
    response = client.messages.create(
        model="claude-haiku-4-5", max_tokens=64, system=SYSTEM,
        messages=[{"role": "user", "content": f"Q: {text}"}],
    )
    return "".join(b.text for b in response.content if b.type == "text")
"""


def _captured(source: str, function: str, *args) -> dict:
    """Run `function` from `source` against a fake client; the keyword arguments it sent."""
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], content=[SimpleNamespace(type="text", text="ok")]
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           messages=SimpleNamespace(create=create))  # fmt: skip
    space = {"json": json, "OpenAI": object, "Anthropic": object}
    exec(source, space)  # noqa: S102 - our own fixture source
    space[function](fake, *args)
    return sent


@pytest.mark.parametrize(
    "source", [ANTHROPIC_SYSTEM, ANTHROPIC_SYSTEM.replace("system=SYSTEM", "system=[*SYSTEM]")]
)
def test_the_rewritten_call_sends_exactly_what_build_request_evaluated(source):
    """Review probe 1 (and I4: `system=` as text blocks): run the migrated function and compare what it sends
    with build_request's request for the same case, in order."""
    site = CallSite(id="cs_09", file="m.py", lines=[4, 9], sdk="anthropic", call="c", function="ask",
                    prompt_source="fstring")  # fmt: skip
    before = _captured(source, "ask", "refund?")
    rendered = [system_message(before["system"]), *before["messages"]]  # as the case files hold it
    case = EvalCase(id="c", site_id="cs_09", input_messages=rendered, expected_kind="text", split="dev")
    few = [{"role": "user", "content": "Q: hi"}, {"role": "assistant", "content": "Hi."}]
    spec = AdapterSpec(max_tokens=64, temperature=0.0, user_suffix=" Answer in one line.", few_shot=few)
    request = BUILD(spec, case, Tier.SUPER)
    params = call_params(request)

    prepared, *_, rewritten, fixed = _prepare(source, site, spec, params, 100)

    assert rewritten and fixed
    assert json.dumps(_captured(prepared, "ask", "refund?")) == json.dumps(
        {**params, "messages": request["messages"]}
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("max_tokens=64,", "streams its reply"),  # stream=True added below
        ("max_tokens=64, top_p=0.9,", "the call passes `top_p=`"),
        ("max_tokens=64, system=build_system(),", "not provably a string"),
    ],
    ids=["stream", "unknown-kwarg", "system-not-provable"],
)
def test_a_call_the_rewrite_cannot_carry_exactly_is_refused(change, reason):
    """Review I5 (probe 5) and I4: dropped silently, these shipped a request the code no longer matches."""
    source = ANTHROPIC_SYSTEM.replace("max_tokens=64, system=SYSTEM,", change)
    source = (
        source.replace("max_tokens=64,", "max_tokens=64, stream=True,")
        if reason.startswith("streams")
        else source
    )

    with pytest.raises(Unsupported, match=re.escape(reason)):
        _rewrite_call(source, _site("ask", sdk="anthropic"), {}, None, REQUEST | {"tools": None}, 100)


def test_a_timeout_is_carried_through_as_written():
    source = OPENAI_SITE.replace("max_tokens=16,", "max_tokens=16,\n        timeout=20,")

    out = _rewrite_call(source, _site("triage"), {"tools": "NEMOTRON_TOOLS"}, "PROMPT", REQUEST, 100)

    assert "        timeout=20,\n        messages=[" in out
    conformant = f"PROMPT = 'p'\nNEMOTRON_TOOLS = {REQUEST['tools']!r}\n\n\n{out}"
    assert conformance(conformant, _site("triage"), REQUEST, {"tools": "NEMOTRON_TOOLS"}, "PROMPT") == []


def test_a_flat_layout_keeps_every_migrated_module_read_only_in_the_combined_repair():
    """Review I2 (probe 3): with test_x.py next to x.py, `from greet import greet` made greet.py a 'helper'
    of the failing test, and the combined repair shipped max_tokens=4096 for an evaluated 32."""
    repo = work_dir("tiny-flat") / "src"
    repo.mkdir(parents=True)
    for name in ("greet.py", "shout.py", "motto.py"):
        (repo / name).write_text((TINY / "tinyapp" / name).read_text())
    (repo / TESTS.split("/")[1]).write_text((TINY / TESTS).read_text().replace("from tinyapp.", "from "))
    sites = [s.model_copy(update={"file": s.file.split("/")[1]}) for s in tiny_sites()]
    mark = {"greet.py": "def test_greet_strips_the_reply():", "shout.py": "def test_shout_"}
    replies = [
        PatchEdits(edits=[Edit(path="test_tinyapp.py", search=m, replace=f"# {f}\n{m}")])
        for f, m in mark.items()
    ]
    bump = PatchEdits(edits=[Edit(path="greet.py", search="max_tokens=32,", replace="max_tokens=4096,")])
    llm = patcher_llm(*replies, bump, bump)
    box = FakeSandbox  # untouched, base, greet alone, shout alone, merged (red); then one at a time

    result = compose_patches(repo, sites, tiny_verdicts(), tiny_adapters(), llm, work_dir("tiny-flat-run"),
                             config=CONFIG, build_request=BUILD,
                             sandbox_factory=lambda r: box(r, [True, True, True, True, False], False,
                                                           "FAILED test_tinyapp.py::x"))  # fmt: skip

    prompt = llm.calls[2]["messages"][-1]["content"]
    assert "### greet.py (read-only)" in prompt and "### test_tinyapp.py (editable)" in prompt
    assert "greet.py is migrated by Offramp and read-only" in llm.calls[3]["messages"][-1]["content"]
    assert all("max_tokens=4096" not in text for text in result.files.values())
    assert "max_tokens=32," in result.files["greet.py"]


ANTHROPIC_GREET = """GREET_SYSTEM_PROMPT = "You write one-line greetings."


def greet(client, name: str):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=32,
        system=GREET_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Greet {name}."}],
    )
    return response
"""


def test_a_model_mapping_the_reply_may_not_touch_the_call_offramp_wrote():
    """Review I6: when the reply's use is not recognised the model edits the module; the request it sends
    must still be exactly the evaluated one."""
    repo = work_dir("tiny-anthropic") / "src"
    shutil.copytree(TINY, repo)
    (repo / "tinyapp" / "greet.py").write_text(ANTHROPIC_GREET)
    sites = [s.model_copy(update={"sdk": "anthropic"}) if s.id == "cs_01" else s for s in tiny_sites()]
    reword = Edit(path="tinyapp/greet.py", search='f"Greet {name}."', replace='f"Hello {name}!"')
    llm = patcher_llm(*[PatchEdits(edits=[reword])] * 2, migrate("tinyapp/shout.py"))

    work = work_dir("tiny-anthropic-run")
    result = compose_patches(repo, sites, tiny_verdicts(), tiny_adapters(), llm, work, config=CONFIG,
                             build_request=BUILD, sandbox_factory=FakeSandbox)  # fmt: skip

    assert (
        "### tinyapp/greet.py (editable)" in llm.calls[0]["messages"][-1]["content"]
    )  # the reply is its job
    reason = verdict(result, "cs_01").reason
    assert "the call Offramp wrote in `greet` must stay exactly as it is" in reason


def test_minor_review_points_m3_m6():
    """M6: a converted fake keeps its argument check (Completions is imported when missing). M3: two fakes
    claiming one case are left to the repair."""
    bare = FAKES.replace("from openai.resources.chat.completions import Completions\n", "")
    out = convert_fakes(bare, "helphub.offramp_client")
    assert "from openai.resources.chat.completions import Completions" in out
    assert "        _check_kwargs(Completions.create, kwargs)\n" in out
    sites = [({OPENAI_TWIN}, {"triage"}), ({ANTHROPIC_TWIN}, {"triage", "next-step"})]  # both claim "triage"
    assert unify_twins(CLI_TEST, sites) == CLI_TEST
