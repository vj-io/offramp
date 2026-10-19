"""GitHub REST PR builder (PRD §5.1): one commit on branch offramp/<run_id> (git data API), then a PR."""

import os
import subprocess

import httpx

API = "https://api.github.com"


def github_token() -> str:
    """GITHUB_TOKEN, else the local `gh auth token` (developer machines). Never logged."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        out = None
    if out is not None and out.returncode == 0 and out.stdout.strip():
        return out.stdout.strip()
    raise RuntimeError("GITHUB_TOKEN is not set and `gh auth token` returned no token")


def open_pr(
    repo: str,
    *,
    run_id: str,
    files: dict[str, str],
    title: str,
    body: str,
    base: str | None = None,
    token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """Commit `files` (repo-relative path -> full text) on top of `base` and open a PR; returns its URL.

    Re-running the same run_id moves the branch to the new commit and updates the open PR instead of failing.
    """
    if not files:
        raise ValueError("nothing to commit: no changed files")
    branch = f"offramp/{run_id}"
    headers = {
        "Authorization": f"Bearer {token or github_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(base_url=API, headers=headers, transport=transport, timeout=30) as gh:

        def call(method: str, path: str, **kw):
            r = gh.request(method, path, **kw)
            r.raise_for_status()
            return r.json()

        base = base or call("GET", f"/repos/{repo}")["default_branch"]
        base_sha = call("GET", f"/repos/{repo}/git/ref/heads/{base}")["object"]["sha"]
        base_tree = call("GET", f"/repos/{repo}/git/commits/{base_sha}")["tree"]["sha"]
        tree = [{"path": p, "mode": "100644", "type": "blob", "content": c} for p, c in sorted(files.items())]
        tree_sha = call("POST", f"/repos/{repo}/git/trees", json={"base_tree": base_tree, "tree": tree})[
            "sha"
        ]
        commit = call(
            "POST",
            f"/repos/{repo}/git/commits",
            json={"message": f"{title}\n\nOfframp run {run_id}", "tree": tree_sha, "parents": [base_sha]},
        )["sha"]

        r = gh.post(f"/repos/{repo}/git/refs", json={"ref": f"refs/heads/{branch}", "sha": commit})
        if r.status_code == 422:  # the branch exists from an earlier attempt of this run
            call("PATCH", f"/repos/{repo}/git/refs/heads/{branch}", json={"sha": commit, "force": True})
        else:
            r.raise_for_status()

        r = gh.post(f"/repos/{repo}/pulls", json={"title": title, "head": branch, "base": base, "body": body})
        if r.status_code == 422:  # a PR for this branch is already open: refresh it
            owner = repo.split("/")[0]
            prs = call("GET", f"/repos/{repo}/pulls", params={"head": f"{owner}:{branch}", "state": "open"})
            if prs:
                call("PATCH", f"/repos/{repo}/pulls/{prs[0]['number']}", json={"title": title, "body": body})
                return prs[0]["html_url"]
        r.raise_for_status()
        return r.json()["html_url"]
