"""Patch Composer (PRD §6.8): contract tests with FakeLLM + a fake sandbox, and the demo repo on cassettes.

The demo-repo tests replay tests/fixtures/cassettes/patcher (recorded live with `pytest --live`) and run the
patched copy's own ruff + pytest in the real sandbox. They skip when the demo repo checkout is absent.
"""

import ast
import json
import os
import shutil
from pathlib import Path

import httpx
import pytest

from offramp.agents.patcher import (
    Edit,
    PatchEdits,
    add_imports,
    call_params,
    compose_patches,
    edits_text,
    keep_comment,
    parse_edits,
    py_literal,
)
from offramp.sandbox import CheckResult, docker_available
from offramp.schemas import Adapter, AdapterSpec, CallSite, RunConfig, Tier, Verdict
from offramp.tf import BudgetExceeded, FakeLLM, TFClient, get_tier

try:
    from offramp.agents.translate import build_request as BUILD  # Task 6
except ImportError:  # Task 6 not merged on this branch: the same contract, enough for these specs

    def BUILD(spec, case, tier):
        swap = spec.tool_choice == "required" and Tier(tier) == Tier.LIGHTNING
        return {
            "role": "eval_target",
            "tier": Tier(tier),
            "messages": case.input_messages,
            "max_tokens": (spec.max_tokens or 1024) + (2000 if spec.reasoning != "off" else 0),
            "temperature": spec.temperature,
            "json_schema": spec.response_format,
            "json_mode_fallback": spec.json_mode_fallback,
            "tools": spec.tools,
            "tool_choice": "auto" if swap else spec.tool_choice,
            "reasoning": spec.reasoning,
            "extra_body": dict(spec.extra_body),
        }


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


GOOD_CALL = (
    f'model="{LIGHTNING}",\n'
    "        max_tokens=32,\n"
    "        temperature=0.0,\n"
    '        extra_body={"chat_template_kwargs": {"enable_thinking": False}},'
)


def migrate(path: str, call: str = GOOD_CALL, extra: list[Edit] = ()) -> PatchEdits:
    return PatchEdits(
        edits=[Edit(path=path, search='model="gpt-4o-mini",\n        max_tokens=32,', replace=call), *extra],
        notes="moved to Lightning",
    )


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
        "offramp/__init__.py",
        "offramp/client.py",
        "tinyapp/greet.py",
        "tinyapp/motto.py",
        "tinyapp/shout.py",
    }
    assert f'model="{LIGHTNING}"' in result.files["tinyapp/greet.py"]
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
    assert len(boxes[0].checks) == 3  # baseline (client + comments), then one per migrated site
    assert boxes[0].fixed == [["tinyapp/greet.py"], ["tinyapp/shout.py"]]


def test_client_module_reads_the_key_and_pins_verified_tier_ids():
    llm = patcher_llm(migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))

    client = run(llm).files["offramp/client.py"]

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


def test_failing_repo_tests_downgrade_the_site_and_drop_its_patch():
    breaking = migrate(
        "tinyapp/shout.py",
        extra=[Edit(path="tinyapp/shout.py", search=".strip().upper()", replace=".strip()")],
    )
    llm = patcher_llm(migrate("tinyapp/greet.py"), breaking, breaking)  # shout: first try + repair round

    result = run(llm, results=[True, True, False, False])

    shout = verdict(result, "cs_03")
    assert shout.decision == "MANUAL_REVIEW"
    assert "repo checks failed" in shout.reason and "FAILED tests/test_tinyapp.py::x" in shout.reason
    assert "tinyapp/shout.py" not in result.files
    assert result.adapters["cs_03"].patch is None and "cs_03" not in result.diffs
    assert verdict(result, "cs_01").decision == "MIGRATE"


def test_a_red_baseline_downgrades_every_migration_and_changes_nothing():
    result = run(FakeLLM(), results=[False])

    assert [v.decision for v in result.verdicts] == ["MANUAL_REVIEW", "KEEP_CLOSED", "MANUAL_REVIEW"]
    assert "before any call-site patch" in verdict(result, "cs_01").reason
    assert result.files == {}


def test_no_accepted_migration_means_no_client_module():
    llm = patcher_llm(*[PatchEdits(edits=[])] * 4)  # two sites, each a try and a repair round

    result = run(llm)

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
        (
            migrate("tinyapp/greet.py", GOOD_CALL.replace("max_tokens=32", "max_tokens=64")).edits,
            "max_tokens",
        ),
        (
            migrate("tinyapp/greet.py", GOOD_CALL.replace("temperature=0.0", "temperature=0.7")).edits,
            "temperature",
        ),
        (migrate("tinyapp/greet.py", GOOD_CALL.replace("False", "True")).edits, "extra_body"),
        (migrate("tinyapp/greet.py", GOOD_CALL.replace(LIGHTNING, "gpt-4o-mini")).edits, "model"),
        (migrate("tinyapp/greet.py", GOOD_CALL + "\n        top_p=0.5,").edits, "top_p"),
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
    assert f'model="{LIGHTNING}"' in prompt and "max_tokens=32" in prompt
    assert 'extra_body={"chat_template_kwargs": {"enable_thinking": False}}' in prompt
    assert "### tinyapp/greet.py (editable)" in prompt and "### tests/test_tinyapp.py (editable)" in prompt
    assert "### offramp/client.py (read-only)" in prompt
    second = llm.calls[1]["messages"][-1]["content"]
    assert "### tinyapp/greet.py" not in second  # unrelated module: not shown
    assert "def test_greet_strips_the_reply():  # migrated" in second  # builds on the accepted patch


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
    assert consts["NEMOTRON_TOOLS"] == tools
    assert consts["NEMOTRON_FEW_SHOT"] == few
    assert (
        consts["GREET_SYSTEM_PROMPT"] == "You write one-line greetings. Never use emoji."
    )  # replaced in place
    assert source.index("NEMOTRON_TOOLS =") < source.index("def greet")
    assert max(len(line) for line in source.splitlines()) <= 100
    assert (
        "tools=NEMOTRON_TOOLS" in prompt and 'tool_choice="auto"' in prompt
    )  # Lightning: build_request swap


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
    assert "max_tokens=2032" in prompt and 'tool_choice="auto"' in prompt


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


def demo_inputs():
    data = json.loads((FIXTURES / "pr" / "demo_run.json").read_text())["context"]
    sites = [CallSite.model_validate(s) for s in data["sites"]]
    verdicts = [Verdict.model_validate(v) for v in data["verdicts"]]
    adapters = {k: Adapter.model_validate(a) for k, a in data["adapters"].items()}
    return sites, verdicts, adapters, RunConfig.model_validate(data["config"])


def run_demo(live: bool):
    if not live and not any(CASSETTES.glob("*.json")):
        pytest.skip("demo patcher cassettes not recorded for the current patcher prompt: pytest --live")
    repo = demo_repo()
    if repo is None:
        pytest.skip("offramp-demo-app checkout not found (set OFFRAMP_DEMO_REPO_DIR)")
    if not (docker_available() or shutil.which("uv")):
        pytest.skip("no sandbox runtime (docker or uv)")
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
    result = run_demo(live=False)

    decisions = {v.site_id: v.decision for v in result.verdicts}
    assert decisions == {
        "cs_01": "MIGRATE",
        "cs_02": "MIGRATE",
        "cs_03": "MIGRATE",
        "cs_04": "MIGRATE",
        "cs_05": "KEEP_CLOSED",
        "cs_06": "MIGRATE",
    }
    assert "# offramp: kept on o3 — parity 71.4% < 90%" in result.files["helphub/refunds/plan.py"]
    for site_id in ("cs_02", "cs_03", "cs_06"):  # the Anthropic sites now call Token Factory via OpenAI
        assert result.adapters[site_id].patch
    agent = result.files["helphub/ops/agent_step.py"]
    assert "tool_calls" in agent and "messages.create" not in agent
    assert "anthropic" not in result.files["helphub/cli.py"].lower()


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

    result = run(llm, results=[True, False, True, True])

    assert verdict(result, "cs_01").decision == "MIGRATE"
    first, repair = llm.calls[0]["messages"], llm.calls[1]["messages"]
    assert repair[: len(first)] == first
    assert repair[-2] == {"role": "assistant", "content": edits_text(breaking)}
    assert "FAILED tests/test_tinyapp.py::x" in repair[-1]["content"]
    assert len(llm.calls) == 3


def test_a_block_overlapping_an_earlier_one_is_explained():
    first = Edit(path="tinyapp/greet.py", search='model="gpt-4o-mini",', replace=f'model="{LIGHTNING}",')
    overlap = Edit(
        path="tinyapp/greet.py", search='model="gpt-4o-mini",\n        max_tokens=32,', replace="x"
    )
    bad = PatchEdits(edits=[first, overlap])
    llm = patcher_llm(bad, bad, migrate("tinyapp/shout.py"))

    reason = verdict(run(llm), "cs_01").reason

    assert "an earlier block already changed these lines" in reason


def test_a_repeated_edit_is_applied_once():
    edit = Edit(path="tinyapp/greet.py", search="max_tokens=32,", replace="max_tokens=32,  # same")
    llm = patcher_llm(
        PatchEdits(edits=[*migrate("tinyapp/greet.py").edits, edit, edit]), migrate("tinyapp/shout.py")
    )

    result = run(llm)

    assert verdict(result, "cs_01").decision == "MIGRATE"
    assert result.files["tinyapp/greet.py"].count("# same") == 1


def test_add_imports_binds_known_names_after_the_last_import():
    source = '"""Doc."""\n\nimport os\n\n\ndef f(c):\n    return json.loads(token_factory_client()), os\n'

    fixed = add_imports(source)

    assert fixed.splitlines()[2:5] == [
        "import os",
        "from offramp.client import token_factory_client",
        "import json",
    ]
    assert add_imports(fixed) == fixed
    assert add_imports("import json\n\nx = json\n") == "import json\n\nx = json\n"
