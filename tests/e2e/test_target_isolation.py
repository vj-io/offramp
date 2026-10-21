"""Where target-repo code that Offramp runs (the Pydantic schemas it imports, the oracles it executes) runs.

- With Docker: in the sandbox (offramp/sandbox.py), with the network off, no host environment and the repo
  mounted read-only.
- Without Docker: on this machine only under OFFRAMP_ALLOW_LOCAL_SANDBOX=1. That run is not isolated (same
  user, same files); its environment just holds no Offramp key, token or LangSmith setting.
- Without either: it does not run at all, and the run fails with a message that says how to proceed.
"""

import json

import pytest

from offramp import sandbox
from offramp.agents import case_gen
from offramp.schemas import CallSite, PromptTemplate

# Imported for its schema: reports, in the schema itself, what the code could see and do where it ran, and
# leaves ran.txt beside itself when it can write there.
PROBE = """import json, os, pathlib, socket

from pydantic import BaseModel, Field


def _network():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=3).close()
        return "on"
    except OSError:
        return "off"


def _write():
    try:
        pathlib.Path(__file__).with_name("ran.txt").write_text("ran")
        return "writable"
    except OSError:
        return "read-only"


SEEN = {"env": sorted(os.environ), "network": _network(), "mount": _write()}


class Order(BaseModel):
    order_id: str = Field(description=json.dumps(SEEN))
"""
ORACLE = """import json, os, pathlib

SITE_FILE = "shop/orders.py"
try:
    pathlib.Path(__file__).with_name("ran-oracle.txt").write_text("ran")
except OSError:
    pass


def answer(params):
    return json.dumps(sorted(os.environ))


def sample(seed, n, hard=False):
    return []
"""
SECRETS = {
    "NEBIUS_API_KEY": "nb-secret",
    "NEBIUS_API_KEY_FALLBACK": "nb-fallback",
    "TAVILY_API_KEY": "tvly-secret",
    "LANGSMITH_API_KEY": "ls-secret",
    "LANGSMITH_TRACING": "true",
    "TENDEM_API_KEY_FALLBACK": "td-fallback",
    "GITHUB_TOKEN": "ghp-secret",
    "SOME_SERVICE_TOKEN": "x-secret",
}
SITE = CallSite(
    id="cs_01",
    file="shop/orders.py",
    lines=[1, 2],
    sdk="openai",
    call="client.chat.completions.create",
    prompt_source="literal",
    template=PromptTemplate(user="Extract the order as <pydantic:Order>"),
)


def target(tmp_path, monkeypatch, *, docker: bool, opt_in: bool):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(sandbox, "docker_available", lambda: docker)
    if opt_in:
        monkeypatch.setenv(sandbox.LOCAL_OPT_IN, "1")
    else:
        monkeypatch.delenv(sandbox.LOCAL_OPT_IN, raising=False)
    repo = tmp_path / "target"
    (repo / "shop").mkdir(parents=True)
    (repo / "shop" / "__init__.py").write_text("")
    (repo / "shop" / "orders.py").write_text(PROBE)
    (repo / "evals" / "reference").mkdir(parents=True)
    (repo / "evals" / "reference" / "orders_oracle.py").write_text(ORACLE)
    (repo / "requirements.txt").write_text("pydantic>=2.8\n")  # what the sandbox image installs
    return repo, repo / "evals" / "reference" / "orders_oracle.py"


def seen(schemas: dict) -> dict:
    return json.loads(schemas["Order"]["properties"]["order_id"]["description"])


def no_secret(names) -> None:
    assert not set(names) & set(SECRETS)
    assert not [n for n in names if any(s in n for s in ("KEY", "TOKEN", "LANGSMITH", "FALLBACK"))]


def test_without_docker_or_the_opt_in_target_code_never_runs(tmp_path, monkeypatch):
    repo, oracle = target(tmp_path, monkeypatch, docker=False, opt_in=False)

    with pytest.raises(RuntimeError, match="OFFRAMP_ALLOW_LOCAL_SANDBOX=1"):
        case_gen.resolved_schemas(SITE, repo)
    with pytest.raises(RuntimeError, match="OFFRAMP_ALLOW_LOCAL_SANDBOX=1"):
        case_gen._oracle(repo, oracle, {"op": "answer", "params": [{}]})

    assert not (repo / "shop" / "ran.txt").exists()  # the module was never imported
    assert not oracle.with_name("ran-oracle.txt").exists()  # nor the oracle executed
    assert not (tmp_path / "data" / "envs").exists()  # nor its dependencies installed


def test_the_local_opt_in_keeps_offramps_secrets_out_of_the_target_environment(tmp_path, monkeypatch):
    """Not isolation: the opt-in is for repos you trust. It only keeps the secrets out of the child's env."""
    repo, oracle = target(tmp_path, monkeypatch, docker=False, opt_in=True)

    probe = seen(case_gen.resolved_schemas(SITE, repo))
    oracle_env = json.loads(case_gen._oracle(repo, oracle, {"op": "answer", "params": [{}]})[0])

    for names in (probe["env"], oracle_env):
        no_secret(names)
    assert {"PATH", "HOME", "PYTHONPATH"} <= set(probe["env"])
    assert probe["mount"] == "writable" and not (repo / "shop" / "ran.txt").exists()  # it ran on a copy


@pytest.mark.skipif(not sandbox.docker_available(), reason="needs docker")
def test_with_docker_target_code_runs_network_off_on_a_read_only_mount(tmp_path, monkeypatch):
    repo, oracle = target(tmp_path, monkeypatch, docker=True, opt_in=True)  # Docker wins over the opt-in

    probe = seen(case_gen.resolved_schemas(SITE, repo))
    oracle_env = json.loads(case_gen._oracle(repo, oracle, {"op": "answer", "params": [{}]})[0])

    assert probe["network"] == "off" and probe["mount"] == "read-only"
    for names in (probe["env"], oracle_env):  # the image's own variables only (it sets GPG_KEY, not a secret)
        assert not set(names) & {*SECRETS, "OFFRAMP_DATA_DIR"}  # nothing of this process's environment
    assert not (repo / "shop" / "ran.txt").exists() and not oracle.with_name("ran-oracle.txt").exists()
    assert not (tmp_path / "data" / "envs").exists()  # nothing was installed on this machine


def test_a_relative_repo_path_reaches_uv_as_an_absolute_project(tmp_path, monkeypatch):
    """`offramp prepare ../repo` handed the relative path to `uv run --project`, whose cwd is the repo itself:
    "Project directory `../offramp-demo-app` does not exist"."""
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "pyproject.toml").write_text('[project]\nname = "t"\nversion = "0"\n')
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")

    def fake_run(command, cwd, **kwargs):  # no env build in this test: `uv venv <dir>` just makes the dir
        case_gen.Path(command[-1]).mkdir(parents=True, exist_ok=True)
        return ""

    monkeypatch.setattr(case_gen, "_run", fake_run)

    command, env = case_gen._target_python(case_gen.Path("../target"))

    project = command[command.index("--project") + 1]
    assert project == str((tmp_path / "target").resolve())
    assert all(case_gen.Path(p).is_absolute() for p in env["PYTHONPATH"].split(case_gen.os.pathsep))


# Task 23 fix round 1b: target code runs on the repo's tracked files only. A local checkout's ignored files
# (here .env, .netrc and .aws/credentials) must reach neither a schema nor an oracle's answers: both flow into
# eval files and into prompts sent to Token Factory. Fix round 2: nor a git-crypt file, tracked as plaintext
# in an unlocked checkout; and the local opt-in runs on the same copy.
CANARY = "canary-secret-123"
READS = """import json, pathlib

def _read():
    found = {}
    for name in (".env", ".netrc", ".aws/credentials", "secrets/prod.env"):
        root = pathlib.Path(__file__).resolve().parents[2]
        for base in (pathlib.Path.cwd(), root, pathlib.Path.home()):
            try:
                found[str(base / name)] = (base / name).read_text()
            except OSError:
                found[str(base / name)] = "absent"
    return json.dumps(found)
"""
GREEDY_PROBE = (
    READS
    + """
from pydantic import BaseModel, Field


class Order(BaseModel):
    order_id: str = Field(description=_read())
"""
)
GREEDY_ORACLE = (
    READS
    + """
SITE_FILE = "shop/orders.py"


def answer(params):
    return _read()


def sample(seed, n, hard=False):
    return [{"read": _read()}]
"""
)


def greedy_checkout(tmp_path):
    import subprocess

    repo = tmp_path / "target"
    files = {"shop/__init__.py": "", "shop/orders.py": GREEDY_PROBE, "requirements.txt": "pydantic>=2.8\n",
             "evals/reference/orders_oracle.py": GREEDY_ORACLE, ".gitignore": ".env\n.netrc\n.aws/\n",
             ".gitattributes": "secrets/** filter=git-crypt diff=git-crypt\n",
             "secrets/prod.env": f"TOKEN={CANARY}\n"}  # fmt: skip
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    for step in (["init", "-q"], ["add", "."], ["commit", "-qm", "init"]):
        subprocess.run([*git, *step], check=True)
    for path in (".env", ".netrc", ".aws/credentials"):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(f"TOKEN={CANARY}\n")
    return repo, repo / "evals" / "reference" / "orders_oracle.py"


@pytest.mark.parametrize("docker", [True, False], ids=["docker", "local-opt-in"])
def test_target_code_never_reads_a_checkouts_ignored_or_git_crypt_secrets(tmp_path, monkeypatch, docker):
    if docker and not sandbox.docker_available():
        pytest.skip("needs docker")
    monkeypatch.delenv(sandbox.UNTRACKED_OPT_IN)
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(sandbox, "docker_available", lambda: docker)
    monkeypatch.setenv(sandbox.LOCAL_OPT_IN, "1")  # Docker wins over it
    repo, oracle = greedy_checkout(tmp_path)

    schemas = case_gen.resolved_schemas(SITE, repo)
    answers = case_gen._oracle(repo, oracle, {"op": "answer", "params": [{}]})
    samples = case_gen._oracle(repo, oracle, {"op": "sample", "seed": 1, "n": 1, "hard": False})

    reads = json.loads(schemas["Order"]["properties"]["order_id"]["description"])
    assert reads and all(v == "absent" for v in reads.values())  # it ran, and found nothing to read
    assert CANARY not in json.dumps([schemas, answers, samples])


def test_a_plain_directory_runs_no_target_code_without_the_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv(sandbox.UNTRACKED_OPT_IN)
    repo, oracle = target(tmp_path, monkeypatch, docker=True, opt_in=False)

    with pytest.raises(RuntimeError, match="is not a git checkout"):
        case_gen.resolved_schemas(SITE, repo)
    with pytest.raises(RuntimeError, match="is not a git checkout"):
        case_gen._oracle(repo, oracle, {"op": "answer", "params": [{}]})


def test_an_untracked_oracle_is_named_as_such(tmp_path, monkeypatch):
    """The copy target code runs on holds tracked files only: an oracle not yet added is not in it."""
    repo, oracle = greedy_checkout(tmp_path)
    new = oracle.with_name("new_oracle.py")
    new.write_text(GREEDY_ORACLE)

    with pytest.raises(case_gen.OracleError) as raised:
        case_gen._oracle(repo, new, {"op": "answer", "params": [{}]})

    assert (
        str(raised.value)
        == "evals/reference/new_oracle.py is not tracked by git, so the sandbox can't see it"
    )
