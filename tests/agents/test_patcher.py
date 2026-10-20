"""Patch Composer (PRD §6.8): contract tests with FakeLLM + a fake sandbox, and the demo repo on cassettes.

The demo-repo tests replay tests/fixtures/cassettes/patcher (recorded live with `pytest --live`) and run the
patched copy's own ruff + pytest in the real sandbox. They skip when the demo repo checkout is absent.
"""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from offramp import sandbox as sandbox_module
from offramp.agents.patcher import (
    Edit,
    PatchEdits,
    _rewrite_call,
    _summary,
    add_fixtures,
    add_imports,
    call_params,
    client_location,
    compose_patches,
    copy_repo,
    edits_text,
    keep_comment,
    line_length,
    merge_onto,
    mirror_fakes,
    parse_edits,
    py_literal,
    rewrite_callers,
    wrap_long_strings,
)
from offramp.agents.translate import build_request
from offramp.sandbox import CheckResult, Sandbox, docker_available
from offramp.schemas import Adapter, AdapterSpec, CallSite, RunConfig, Tier, Verdict
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


class FakeSandbox:
    """Stands in for offramp.sandbox.Sandbox: canned check results, records what it was asked."""

    mode = "docker"

    def __init__(self, repo: Path, results: list[bool] | None = None):
        self.repo, self.results, self.checks, self.fixed = repo, list(results or []), [], []

    def check(self) -> CheckResult:
        self.checks.append(
            {p.relative_to(self.repo).as_posix(): p.read_text() for p in self.repo.rglob("*.py")}
        )
        ok = self.results.pop(0) if self.results else True
        return CheckResult(ok=ok, mode="docker", output="" if ok else "FAILED tests/test_tinyapp.py::x")

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


SPLAT = "**NEMOTRON_REQUEST"  # the call is rewritten by code to create(**NEMOTRON_REQUEST, messages=[...])


def migrate(path: str, extra: list[Edit] = ()) -> PatchEdits:
    """A patcher reply for a tiny-repo site: the code rewrote the call already, only `extra` edits remain."""
    return PatchEdits(edits=[*extra], notes="nothing else to change")


def request_edit(path: str, search: str, replace: str) -> list[Edit]:
    return [Edit(path=path, search=search, replace=replace)]


def patcher_llm(*replies) -> FakeLLM:
    """FakeLLM answering role `patcher` in the plain edit-block format."""
    return FakeLLM().add("patcher", *(edits_text(r) if isinstance(r, PatchEdits) else r for r in replies))


def run(llm, results=None, sandboxes=None, adapters=None, verdicts=None, name="tiny", build=BUILD):
    def factory(repo):
        box = FakeSandbox(repo, results)
        if sandboxes is not None:
            sandboxes.append(box)
        return box

    return compose_patches(
        TINY,
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
    llm = patcher_llm(migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))
    boxes = []

    result = run(llm, sandboxes=boxes)

    assert [v.decision for v in result.verdicts] == ["MIGRATE", "KEEP_CLOSED", "MIGRATE"]
    assert set(result.files) == {
        "tinyapp/offramp_client.py",  # inside the package that holds the sites: it ships in the wheel
        "tinyapp/greet.py",
        "tinyapp/motto.py",
        "tinyapp/shout.py",
    }
    assert result.client_path == "tinyapp/offramp_client.py"
    assert SPLAT in result.files["tinyapp/greet.py"] and LIGHTNING in result.files["tinyapp/greet.py"]
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
    assert boxes[0].fixed == [["tinyapp/greet.py"], ["tinyapp/shout.py"]]


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
    breaking = migrate(
        "tinyapp/shout.py",
        extra=[Edit(path="tinyapp/shout.py", search=".strip().upper()", replace=".strip()")],
    )
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


def test_no_accepted_migration_means_no_client_module():
    llm = patcher_llm(*[PatchEdits(edits=[])] * 4)  # two sites, each a try and an identical repair

    result = run(llm, results=[True, True, False, False])

    assert set(result.files) == {"tinyapp/motto.py"}


@pytest.mark.parametrize(
    ("edits", "problem"),
    [
        ([Edit(path="tinyapp/greet.py", search="no such text", replace="x")], "not found"),
        ([Edit(path="pyproject.toml", search="tinyapp", replace="x")], "not editable"),
        ([Edit(path="../escape.py", search="x", replace="y")], "not editable"),
        (
            [
                Edit(
                    path="tests/test_tinyapp.py",
                    search="def test_greet_strips_the_reply():",
                    replace="def _skipped():",
                )
            ],
            "removes tests",
        ),
        (request_edit("tinyapp/greet.py", '"max_tokens": 32,', '"max_tokens": 64,'), "evaluated request"),
        (request_edit("tinyapp/greet.py", '"temperature": 0.0,', '"temperature": 0.7,'), "evaluated request"),
        (
            request_edit("tinyapp/greet.py", '"enable_thinking": False', '"enable_thinking": True'),
            "evaluated request",
        ),
        (
            request_edit("tinyapp/greet.py", f'"model": "{LIGHTNING}",', '"model": "gpt-4o-mini",'),
            "evaluated request",
        ),
        (
            request_edit(
                "tinyapp/greet.py", "**NEMOTRON_REQUEST,", "**NEMOTRON_REQUEST,\n        top_p=0.5,"
            ),
            "the call must be exactly",
        ),  # fmt: skip
        (
            request_edit("tinyapp/greet.py", "**NEMOTRON_REQUEST,", f'model="{LIGHTNING}", max_tokens=32,'),
            "the call must be exactly",
        ),  # fmt: skip
        (
            [
                Edit(
                    path="tinyapp/greet.py",
                    search='"content": GREET_SYSTEM_PROMPT',
                    replace='"content": "Be brief."',
                ),
                *migrate("tinyapp/greet.py").edits,
            ],
            "system prompt",
        ),
    ],
    ids=[
        "search-missing",
        "file-not-allowed",
        "path-escape",
        "test-removed",
        "max-tokens",
        "temperature",
        "thinking",
        "model",
        "extra-kwarg",
        "explicit-kwargs",
        "system-prompt",
    ],
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
    assert "create(**NEMOTRON_REQUEST, messages=[...])" in prompt and "already rewritten" in prompt
    assert f'"model": "{LIGHTNING}"' in prompt and '"max_tokens": 32' in prompt
    assert '"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}' in prompt
    assert "### tinyapp/greet.py (editable)" in prompt and "### tests/test_tinyapp.py (editable)" in prompt
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
    llm = patcher_llm(*[PatchEdits(edits=[])] * 4)

    run(llm, adapters=adapters)

    prompt = llm.calls[0]["messages"][-1]["content"]
    source = prompt.split("### tinyapp/greet.py (editable)\n```python\n")[1].split("```")[0]
    module = ast.parse(source)
    consts = {n.targets[0].id: ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)}
    assert consts["NEMOTRON_REQUEST"]["tools"] == tools  # one constant holds every evaluated parameter
    assert consts["NEMOTRON_REQUEST"]["tool_choice"] == "auto"  # Lightning: build_request's swap
    assert consts["NEMOTRON_FEW_SHOT"] == few
    assert (
        consts["GREET_SYSTEM_PROMPT"] == "You write one-line greetings. Never use emoji."
    )  # replaced in place
    assert source.index("NEMOTRON_REQUEST =") < source.index("def greet")
    assert "*NEMOTRON_FEW_SHOT," in source and SPLAT in source  # the call itself is rewritten by code
    assert max(len(line) for line in source.splitlines()) <= 100
    assert '"tool_choice": "auto"' in prompt


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
    llm = patcher_llm(*[PatchEdits(edits=[])] * 4)

    run(llm, adapters=adapters, build=build)

    prompt = llm.calls[0]["messages"][-1]["content"]
    assert '"max_tokens": 2032' in prompt and '"tool_choice": "auto"' in prompt


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


DEMO_OUTCOME = ({"cs_04": False, "cs_05": False, "cs_06": False}, [])  # recorded live (Task 10c), 2026-10-20
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
    breaking = migrate(
        "tinyapp/greet.py", extra=[Edit(path="tinyapp/greet.py", search=".strip()", replace="")]
    )
    llm = patcher_llm(breaking, migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))

    result = run(llm, results=[True, True, False, True, True, True])

    assert verdict(result, "cs_01").decision == "MIGRATE"
    first, repair = llm.calls[0]["messages"], llm.calls[1]["messages"]
    assert repair[: len(first)] == first
    assert repair[-2] == {"role": "assistant", "content": edits_text(breaking)}
    assert "FAILED tests/test_tinyapp.py::x" in repair[-1]["content"]
    assert len(llm.calls) == 3


def test_a_block_overlapping_an_earlier_one_is_explained():
    first = Edit(path="tinyapp/greet.py", search='"max_tokens": 32,', replace='"max_tokens": 32,  # a')
    overlap = Edit(path="tinyapp/greet.py", search='"max_tokens": 32,\n    "temperature"', replace="x")
    bad = PatchEdits(edits=[first, overlap])
    llm = patcher_llm(bad, bad, migrate("tinyapp/shout.py"))

    reason = verdict(run(llm), "cs_01").reason

    assert "an earlier block already changed these lines" in reason


def test_a_repeated_edit_is_applied_once():
    edit = Edit(path="tinyapp/greet.py", search='"max_tokens": 32,', replace='"max_tokens": 32,  # same')
    llm = patcher_llm(
        PatchEdits(edits=[*migrate("tinyapp/greet.py").edits, edit, edit]), migrate("tinyapp/shout.py")
    )

    result = run(llm)

    assert verdict(result, "cs_01").decision == "MIGRATE"
    assert result.files["tinyapp/greet.py"].count("# same") == 1


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
    text = 'or "").strip()\n'  # only in greet.py; the reply labels it with the test file
    wrong = Edit(path="tests/test_tinyapp.py", search=text, replace='or "").strip()  # relabelled\n')
    llm = patcher_llm(PatchEdits(edits=[wrong]), migrate("tinyapp/shout.py"))

    result = run(llm)

    assert verdict(result, "cs_01").decision == "MIGRATE"
    assert "# relabelled" in result.files["tinyapp/greet.py"]


def test_failure_summaries_are_stable_across_runs():
    """Repair feedback goes into the next request: run-specific text would make every replay a cache miss."""
    tail = "\nFAILED t.py::x - boom\n1 failed in {}"
    run_a = "E   assert isinstance(<ast.Constant object at 0x7f095ee535d0>, x)" + tail.format("5.58s")
    run_b = "E   assert isinstance(<ast.Constant object at 0x7f3851520a10>, x)" + tail.format("0.91s")

    assert _summary(run_a) == _summary(run_b)
    assert "FAILED t.py::x - boom" in _summary(run_a)


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
    breaking = migrate(
        "tinyapp/greet.py", extra=[Edit(path="tinyapp/greet.py", search=".strip()", replace="")]
    )
    llm = patcher_llm(breaking, breaking, migrate("tinyapp/shout.py"))
    boxes = []

    result = run(llm, results=[True, True, False, True, True], sandboxes=boxes)

    assert verdict(result, "cs_01").reason.startswith(
        "repair produced no change; last failure: repo checks failed"
    )
    assert len(boxes[0].checks) == 4  # untouched, base, greet's one try, shout (no combined: one site)
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
    assert SPLAT in combined["tinyapp/greet.py"] and SPLAT in combined["tinyapp/shout.py"]
    assert combined["tests/test_tinyapp.py"].count("# both sites") == 1
    assert SPLAT not in boxes[0].checks[3]["tinyapp/greet.py"]  # shout was checked alone
    assert result.alone == {"cs_01": True, "cs_03": True} and result.combined == ["cs_01", "cs_03"]
    assert result.files["tests/test_tinyapp.py"].count("# both sites") == 1


def test_a_site_green_alone_but_not_together_is_dropped_from_the_combined_tree():
    llm = patcher_llm(migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))

    # untouched, base, greet alone, shout alone, all together (red), greet (green), greet + shout (red)
    result = run(llm, results=[True, True, True, True, False, True, False])

    assert result.alone == {"cs_01": True, "cs_03": True} and result.combined == ["cs_01"]
    shout = verdict(result, "cs_03")
    assert shout.decision == "MANUAL_REVIEW" and "green alone" in shout.reason
    assert "tinyapp/shout.py" not in result.files and verdict(result, "cs_01").decision == "MIGRATE"


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
    first = migrate("tinyapp/greet.py", extra=[Edit(path="tinyapp/greet.py", search=".strip()", replace="")])
    second = migrate(
        "tinyapp/greet.py", extra=[Edit(path="tinyapp/greet.py", search=".strip()", replace=".lower()")]
    )
    llm = patcher_llm(first, second, migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))

    result = run(llm, results=[True, True, False, False, True, True, True])

    assert verdict(result, "cs_01").decision == "MIGRATE" and result.alone["cs_01"]
    third = llm.calls[2]
    assert third["max_tokens"] == 12000 and len(third["messages"]) == 6
    assert third["messages"][-2]["content"] == edits_text(second)


def test_a_repair_that_returns_to_an_earlier_patch_stops_too():
    a = migrate("tinyapp/greet.py", extra=[Edit(path="tinyapp/greet.py", search=".strip()", replace="")])
    b = migrate(
        "tinyapp/greet.py", extra=[Edit(path="tinyapp/greet.py", search=".strip()", replace=".lower()")]
    )
    boxes = []

    result = run(patcher_llm(a, b, a, migrate("tinyapp/shout.py")), results=[True, True, False, False, True],
                 sandboxes=boxes)  # fmt: skip

    assert verdict(result, "cs_01").reason.startswith("repair produced no change; last failure:")
    assert len(boxes[0].checks) == 5  # untouched, base, a, b, shout: the repeated a was never checked


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
    breaking = migrate(
        "tinyapp/greet.py", extra=[Edit(path="tinyapp/greet.py", search=".strip()", replace="")]
    )
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


def _site(function):
    return CallSite(id="cs_09", file="m.py", lines=[2, 10], sdk="openai", call="c", function=function,
                    prompt_source="fstring")  # fmt: skip


@pytest.mark.parametrize(("source", "function"), [(OPENAI_SITE, "triage"), (ANTHROPIC_SITE, "step")])
def test_the_call_is_rewritten_by_code_to_the_evaluated_request(source, function):
    constants = {
        "request": "NEMOTRON_REQUEST",
        "few_shot": "NEMOTRON_FEW_SHOT",
        "suffix": "NEMOTRON_USER_SUFFIX",
    }

    out = _rewrite_call(source, _site(function), constants, "PROMPT")

    call = next(n for n in ast.walk(ast.parse(out)) if isinstance(n, ast.Call) and n.keywords)
    assert ast.unparse(call.func).endswith("chat.completions.create") and "messages.create" not in out
    assert [k.arg for k in call.keywords] == [None, "messages"] and call.keywords[
        0
    ].value.id == "NEMOTRON_REQUEST"
    first, few, user = call.keywords[1].value.elts
    assert (
        ast.unparse(first) == "{'role': 'system', 'content': PROMPT}"
        and ast.unparse(few) == "*NEMOTRON_FEW_SHOT"
    )
    assert ast.unparse(user.values[1]).endswith("+ NEMOTRON_USER_SUFFIX")
    for gone in ("model=", "max_tokens=", "temperature=", "tools=", "tool_choice=", "system="):
        assert gone not in out


def test_a_call_whose_messages_are_built_elsewhere_is_left_to_the_model():
    source = "def f(client, msgs):\n    return client.chat.completions.create(model='m', messages=msgs)\n"

    assert _rewrite_call(source, _site("f"), {"request": "NEMOTRON_REQUEST"}, None) is None


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


def test_a_patch_that_changes_the_migrated_functions_parameters_is_rejected():
    """cs_01's live try dropped `client` from triage_ticket and built a client inside: callers broke."""
    drop = Edit(
        path="tinyapp/greet.py",
        search="def greet(client, name: str) -> str:",
        replace="def greet(name: str) -> str:",
    )
    llm = patcher_llm(PatchEdits(edits=[drop]), PatchEdits(edits=[drop]), migrate("tinyapp/shout.py"))

    reason = verdict(run(llm), "cs_01").reason

    assert "signature of `greet` changed" in reason


def test_a_repair_stopped_by_the_budget_keeps_the_first_failure_in_the_reason():
    breaking = migrate(
        "tinyapp/greet.py", extra=[Edit(path="tinyapp/greet.py", search=".strip()", replace="")]
    )
    llm = patcher_llm(
        breaking, BudgetExceeded("Ultra agent call cap reached (0)"), migrate("tinyapp/shout.py")
    )

    reason = verdict(run(llm, results=[True, True, False, True]), "cs_01").reason

    assert reason.startswith(
        "patch not composed: Ultra agent call cap reached (0); last failure: repo checks failed"
    )
