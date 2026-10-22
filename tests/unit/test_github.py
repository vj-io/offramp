"""GitHub REST PR builder against httpx.MockTransport: no real branches or PRs are ever created here."""

import json
import subprocess

import httpx
import pytest

from offramp import github

REPO = "vj-io/offramp-demo-app"
FILES = {"offramp/client.py": "X = 1\n", "helphub/ops/agent_step.py": "Y = 2\n"}


class FakeGitHub:
    """Minimal GitHub REST stand-in; records every request."""

    def __init__(self, branch_exists=False, pr_exists=False):
        self.requests: list[httpx.Request] = []
        self.branch_exists, self.pr_exists = branch_exists, pr_exists

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, method = request.url.path, request.method
        routes = {
            ("GET", f"/repos/{REPO}"): (200, {"default_branch": "main"}),
            ("GET", f"/repos/{REPO}/git/ref/heads/main"): (200, {"object": {"sha": "base-sha"}}),
            ("GET", f"/repos/{REPO}/git/commits/base-sha"): (200, {"tree": {"sha": "base-tree"}}),
            ("GET", f"/repos/{REPO}/git/commits/cloned-sha"): (200, {"tree": {"sha": "cloned-tree"}}),
            ("POST", f"/repos/{REPO}/git/trees"): (201, {"sha": "new-tree"}),
            ("POST", f"/repos/{REPO}/git/commits"): (201, {"sha": "new-commit"}),
            ("POST", f"/repos/{REPO}/git/refs"): (422, {"message": "Reference already exists"})
            if self.branch_exists
            else (201, {"ref": "refs/heads/offramp/run1"}),
            ("PATCH", f"/repos/{REPO}/git/refs/heads/offramp/run1"): (200, {}),
            ("POST", f"/repos/{REPO}/pulls"): (422, {"message": "A pull request already exists"})
            if self.pr_exists
            else (201, {"html_url": "https://github.com/vj-io/offramp-demo-app/pull/7"}),
            ("GET", f"/repos/{REPO}/pulls"): (
                200,
                [{"number": 3, "html_url": "https://github.com/x/pull/3"}],
            ),
            ("PATCH", f"/repos/{REPO}/pulls/3"): (200, {}),
        }
        status, body = routes.get((method, path), (404, {"message": "Not Found"}))
        return httpx.Response(status, json=body)

    def calls(self):
        return [(r.method, r.url.path) for r in self.requests]

    def body(self, method, path):
        return next(json.loads(r.content) for r in self.requests if (r.method, r.url.path) == (method, path))


def test_opens_a_pr_from_a_new_branch_with_one_commit():
    gh = FakeGitHub()

    url = github.open_pr(
        REPO, run_id="run1", files=FILES, title="T", body="B", token="tok", transport=httpx.MockTransport(gh)
    )

    assert url == "https://github.com/vj-io/offramp-demo-app/pull/7"
    assert gh.calls() == [
        ("GET", f"/repos/{REPO}"),
        ("GET", f"/repos/{REPO}/git/ref/heads/main"),
        ("GET", f"/repos/{REPO}/git/commits/base-sha"),
        ("POST", f"/repos/{REPO}/git/trees"),
        ("POST", f"/repos/{REPO}/git/commits"),
        ("POST", f"/repos/{REPO}/git/refs"),
        ("POST", f"/repos/{REPO}/pulls"),
    ]
    tree = gh.body("POST", f"/repos/{REPO}/git/trees")
    assert tree["base_tree"] == "base-tree"
    assert {(e["path"], e["content"], e["mode"], e["type"]) for e in tree["tree"]} == {
        (p, c, "100644", "blob") for p, c in FILES.items()
    }
    commit = gh.body("POST", f"/repos/{REPO}/git/commits")
    assert commit["tree"] == "new-tree" and commit["parents"] == ["base-sha"]
    assert gh.body("POST", f"/repos/{REPO}/git/refs") == {
        "ref": "refs/heads/offramp/run1",
        "sha": "new-commit",
    }
    assert gh.body("POST", f"/repos/{REPO}/pulls") == {
        "title": "T",
        "head": "offramp/run1",
        "base": "main",
        "body": "B",
    }
    auth = {r.headers["authorization"] for r in gh.requests}
    assert auth == {"Bearer tok"}


def test_the_commit_is_based_on_the_commit_the_run_checked_not_on_the_branch_head_now():
    """Review I2 (b_probe.py part 1): the patched files are whole texts patched against the cloned commit;
    committed on today's default-branch head they would silently revert anything changed upstream since."""
    gh = FakeGitHub()
    github.open_pr(
        REPO, run_id="run1", files=FILES, title="T", body="B", parent="cloned-sha", token="tok",
        transport=httpx.MockTransport(gh),
    )  # fmt: skip
    assert gh.body("POST", f"/repos/{REPO}/git/trees")["base_tree"] == "cloned-tree"
    assert gh.body("POST", f"/repos/{REPO}/git/commits")["parents"] == ["cloned-sha"]
    assert (
        gh.body("POST", f"/repos/{REPO}/pulls")["base"] == "main"
    )  # the PR still targets the default branch
    assert ("GET", f"/repos/{REPO}/git/ref/heads/main") not in gh.calls()


def test_a_cloned_commit_github_does_not_have_is_refused_with_a_clear_message():
    gh = FakeGitHub()
    with pytest.raises(ValueError, match="commit gone-sha the run checked is not on GitHub"):  # a refusal
        github.open_pr(
            REPO, run_id="run1", files=FILES, title="T", body="B", parent="gone-sha", token="tok",
            transport=httpx.MockTransport(gh),
        )  # fmt: skip
    assert not any(method == "POST" for method, _ in gh.calls())


def test_rerun_moves_the_existing_branch_and_updates_the_open_pr():
    gh = FakeGitHub(branch_exists=True, pr_exists=True)

    url = github.open_pr(
        REPO,
        run_id="run1",
        files=FILES,
        title="T2",
        body="B2",
        base="main",
        token="tok",
        transport=httpx.MockTransport(gh),
    )

    assert url == "https://github.com/x/pull/3"
    assert gh.body("PATCH", f"/repos/{REPO}/git/refs/heads/offramp/run1") == {
        "sha": "new-commit",
        "force": True,
    }
    assert gh.body("PATCH", f"/repos/{REPO}/pulls/3") == {"title": "T2", "body": "B2"}
    (lookup,) = [r for r in gh.requests if (r.method, r.url.path) == ("GET", f"/repos/{REPO}/pulls")]
    assert lookup.url.params["head"] == "vj-io:offramp/run1"
    assert ("GET", f"/repos/{REPO}") not in gh.calls()  # base given: no default-branch lookup


def test_api_errors_raise_without_leaking_the_token():
    def deny(request):
        return httpx.Response(401, json={"message": "Bad credentials"})

    with pytest.raises(httpx.HTTPStatusError) as err:
        github.open_pr(
            REPO,
            run_id="r",
            files=FILES,
            title="T",
            body="B",
            token="secret-tok",
            transport=httpx.MockTransport(deny),
        )
    assert "secret-tok" not in str(err.value)


def test_refuses_an_empty_commit():
    with pytest.raises(ValueError):
        github.open_pr(
            REPO,
            run_id="r",
            files={},
            title="T",
            body="B",
            token="tok",
            transport=httpx.MockTransport(FakeGitHub()),
        )


def test_token_from_env_else_gh_cli(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "env-tok")
    assert github.github_token() == "env-tok"

    monkeypatch.setenv("GITHUB_TOKEN", "")
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="gh-tok\n", stderr="")

    monkeypatch.setattr(github.subprocess, "run", fake_run)
    assert github.github_token() == "gh-tok"
    assert seen == [["gh", "auth", "token"]]

    monkeypatch.setattr(
        github.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not logged in"),
    )
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        github.github_token()
