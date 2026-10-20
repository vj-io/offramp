"""Ecosystem scan (judge-audit item 18): Offramp's static scan over about 30 popular public Python repos that
import openai or anthropic.

    uv run python scripts/ecosystem_scan.py [--target 30]

Candidates come from GitHub's repository search (through `gh api`, so no extra key), most stars
first. Each tarball is downloaded and unpacked the way /api/scan does it (offramp/github_scan.py:
the same 25 MB and 5,000-file caps, regular .py files only, no symlinks) and scanned without
importing or running any of it. A repo counts once a file the scanner reads imports `openai` or
`anthropic`. Writes docs/evidence/ecosystem-scan.md and .json. Read-only: it never opens issues or
pull requests on those repos.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

from offramp.agents.scanner import _py_files, scan_repo
from offramp.github_scan import ScanError, download, extract_python
from offramp.scan_table import table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "evidence" / "ecosystem-scan"
QUERIES = [f"{sdk} language:python archived:false fork:false stars:>1000" for sdk in ("openai", "anthropic")]
EXCLUDED = {  # they define the API client rather than call it
    "openai/openai-python": "the OpenAI SDK itself",
    "anthropics/anthropic-sdk-python": "the Anthropic SDK itself",
}
WORKERS = 4  # parallel downloads
TIME_LIMIT_S = 1200.0  # per repo: /api/scan allows 20 s; on a slow link the 25 MB cap decides
IMPORTS = re.compile(r"^[ \t]*(?:from[ \t]+|import[ \t]+(?:[\w.]+[ \t]*,[ \t]*)*)(openai|anthropic)\b", re.M)
DETERMINISTIC = ("classification", "JSON", "tool call")  # shapes Offramp scores without a judge

# Sunset risk: the provider's deprecation page lists the model (or the dated snapshot its alias points to) as
# retired or deprecated with a shutdown date. Read on SUNSET_AS_OF; ids not listed there count as no risk.
SUNSET_AS_OF = "2026-10-21"
SUNSET_SOURCES = {
    "openai": "https://developers.openai.com/api/docs/deprecations",
    "anthropic": "https://platform.claude.com/docs/en/about-claude/model-deprecations",
}
SHUTDOWN: dict[str, str] = {}
for when, ids in {
    # OpenAI, upcoming
    "2027-04-01": "gpt-5.1 gpt-5.4-nano gpt-5.3-codex",
    "2027-01-14": "gpt-4o gpt-4o-2024-08-06 gpt-4o-2024-11-20 gpt-4o-mini gpt-4o-mini-2024-07-18 gpt-4.1 "
    "gpt-4.1-2025-04-14 gpt-4.1-mini gpt-4.1-mini-2025-04-14",
    "2026-12-11": "gpt-5 gpt-5-2025-08-07 gpt-5-mini gpt-5-mini-2025-08-07 gpt-5-nano gpt-5-nano-2025-08-07 "
    "gpt-5-pro gpt-5-pro-2025-10-06 o3 o3-2025-04-16 o3-pro o3-pro-2025-06-10",
    "2026-10-23": "gpt-3.5-turbo gpt-3.5-turbo-0125 gpt-4 gpt-4-0613 gpt-4-1106-preview gpt-4-turbo "
    "gpt-4-turbo-2024-04-09 gpt-4.1-nano gpt-4.1-nano-2025-04-14 gpt-4o-2024-05-13 o1 o1-2024-12-17 o1-pro "
    "o1-pro-2025-03-19 o3-mini o3-mini-2025-01-31 o4-mini o4-mini-2025-04-16",
    # OpenAI, already shut down
    "2026-09-28": "gpt-3.5-turbo-instruct gpt-3.5-turbo-1106 babbage-002 davinci-002",
    "2026-07-23": "gpt-5-chat-latest computer-use-preview gpt-4o-search-preview gpt-4o-mini-search-preview",
    "2026-05-07": "gpt-4o-realtime-preview gpt-4o-audio-preview gpt-4o-mini-realtime-preview",
    "2026-03-26": "gpt-4-0314 gpt-4-0125-preview gpt-4-turbo-preview",
    "2026-02-17": "chatgpt-4o-latest",
    "2026-02-12": "codex-mini-latest",
    "2025-10-27": "o1-mini",
    "2025-07-28": "o1-preview",
    "2025-07-14": "gpt-4.5-preview",
    "2025-06-06": "gpt-4-32k gpt-4-32k-0613 gpt-4-32k-0314",
    "2024-12-06": "gpt-4-vision-preview gpt-4-1106-vision-preview",
    "2024-09-13": "gpt-3.5-turbo-0613 gpt-3.5-turbo-16k-0613 gpt-3.5-turbo-0301",
    "2024-01-04": "text-davinci-003 text-davinci-002 text-davinci-001 text-curie-001 text-babbage-001 "
    "text-ada-001 code-davinci-002 ada babbage curie davinci",
    # Anthropic (aliases such as claude-3-5-sonnet-latest resolve to these families)
    "2026-11-30": "claude-sonnet-4-5 claude-sonnet-4-5-20250929",
    "2026-08-05": "claude-opus-4-1 claude-opus-4-1-20250805",
    "2026-06-15": "claude-opus-4-0 claude-opus-4-20250514 claude-sonnet-4-0 claude-sonnet-4-20250514",
    "2026-04-20": "claude-3-haiku claude-3-haiku-20240307",
    "2026-02-19": "claude-3-7-sonnet claude-3-7-sonnet-20250219 claude-3-5-haiku claude-3-5-haiku-20241022",
    "2026-01-05": "claude-3-opus claude-3-opus-20240229",
    "2025-10-28": "claude-3-5-sonnet claude-3-5-sonnet-20240620 claude-3-5-sonnet-20241022",
    "2025-07-21": "claude-2 claude-2.0 claude-2.1 claude-3-sonnet claude-3-sonnet-20240229",
    "2024-11-06": "claude-1 claude-1.0 claude-1.1 claude-1.2 claude-1.3 claude-instant-1 claude-instant-1.0 "
    "claude-instant-1.1 claude-instant-1.2",
}.items():
    SHUTDOWN.update(dict.fromkeys(ids.split(), when))


def sunset(model: str | None) -> tuple[str, str] | None:
    """("retired" | "deprecated", shutdown date) of a model id; "openai/"-style prefixes and
    "-latest" are ignored."""
    if not model:
        return None
    m = model.rsplit("/", 1)[-1].lower()
    when = SHUTDOWN.get(m) or SHUTDOWN.get(m.removesuffix("-latest"))
    if when is None:
        return None
    return ("retired" if when <= SUNSET_AS_OF else "deprecated"), when


def candidates() -> list[dict[str, Any]]:
    """Repos from the GitHub repository search, deduplicated, most stars first."""
    seen: dict[str, dict[str, Any]] = {}
    for q in QUERIES:
        args = ["-f", f"q={q}", "-f", "sort=stars", "-f", "per_page=100"]
        cmd = ["gh", "api", "-X", "GET", "search/repositories", *args]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for item in json.loads(out.stdout)["items"]:
            seen.setdefault(item["full_name"], {"repo": item["full_name"], "stars": item["stargazers_count"]})
    return sorted(seen.values(), key=lambda r: (-r["stars"], r["repo"]))


def fetch(repo: str, deadline: float, cache: Path | None) -> bytes:
    """The HEAD tarball, or the one an earlier run kept in `cache` (a refusal is kept too)."""
    owner, name = repo.split("/")
    if cache is None:
        return download(owner, name, "HEAD", deadline)
    tarball, refusal = cache / f"{owner}__{name}.tar.gz", cache / f"{owner}__{name}.refused.json"
    if tarball.exists():
        return tarball.read_bytes()
    if refusal.exists():
        raise ScanError(**json.loads(refusal.read_text()))
    try:
        data = download(owner, name, "HEAD", deadline)
    except ScanError as exc:
        if exc.status in (404, 413):  # not a property of this network, so worth keeping
            refusal.write_text(json.dumps({"status": exc.status, "code": exc.code, "message": exc.message}))
        raise
    tarball.write_bytes(data)
    return data


def scan_one(repo: str, cache: Path | None = None) -> dict[str, Any]:
    """The scan table of one repo, or {"skipped": reason}."""
    deadline = time.monotonic() + TIME_LIMIT_S
    try:
        data = fetch(repo, deadline, cache)
        with tempfile.TemporaryDirectory(prefix="offramp-eco-") as tmp:
            files, commit = extract_python(data, Path(tmp), deadline)
            texts = (f.read_text("utf-8", errors="replace") for f in _py_files(Path(tmp)))
            sdks = sorted({m for text in texts for m in IMPORTS.findall(text)})
            if not sdks:
                return {"skipped": "no file the scanner reads imports openai or anthropic"}
            result = scan_repo(tmp, deadline=deadline)
    except ScanError as exc:
        return {"skipped": exc.message}
    except TimeoutError:
        return {"skipped": "scan took over the time limit"}
    t = table(result)
    return {"commit": commit, "python_files": files, "imports": sdks} | {
        k: t[k] for k in ("rows", "warnings")
    }


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


def summarize(repos: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for repo in repos for r in repo["rows"]]
    chat = [r for r in rows if r["shape"] != "embedding"]
    known = [r for r in chat if r["model"]]
    risk = [s for r in known if (s := sunset(r["model"]))]
    certain = [r for r in chat if not r["shape_guessed"]]
    return {
        "repos": len(repos),
        "repos_with_sites": sum(1 for repo in repos if repo["rows"]),
        "call_sites": len(rows),
        "chat_sites": len(chat),
        "embedding_sites": len(rows) - len(chat),
        "sdk": dict(Counter(r["sdk"] for r in rows).most_common()),
        "shapes": dict(Counter(r["shape"] for r in chat).most_common()),
        "shapes_from_parameters": dict(Counter(r["shape"] for r in certain).most_common()),
        "deterministic_shapes": sum(r["shape"] in DETERMINISTIC for r in chat),
        "deterministic_from_parameters": sum(r["shape"] in DETERMINISTIC for r in certain),
        "streams": sum(r["streaming"] is True for r in chat),
        "may_stream": sum(r["streaming"] is None for r in chat),
        "model_known": len(known),
        "sunset_retired": sum(s[0] == "retired" for s in risk),
        "sunset_deprecated": sum(s[0] == "deprecated" for s in risk),
        "models": dict(Counter(r["model"] for r in known).most_common()),
    }


def markdown(report: dict[str, Any]) -> str:
    t, repos = report["totals"], report["repos"]
    chat, known = t["chat_sites"], t["model_known"]
    det, det_params = t["deterministic_shapes"], t["deterministic_from_parameters"]
    risk = t["sunset_retired"] + t["sunset_deprecated"]
    streaming = t["streams"] + t["may_stream"]
    sdks = ", ".join(f"{v} {k}" for k, v in t["sdk"].items())
    queries = " and ".join(f"`{q}`" for q in report["queries"])
    lines = [
        "# Ecosystem scan: closed-LLM call sites in popular public Python repos",
        "",
        f"Offramp's static scan (`offramp scan`, the same code as the hosted /scan page) over the "
        f"{t['repos']} most-starred public Python repos on GitHub that import `openai` or "
        f"`anthropic`, run on {report['date']} with Offramp at `{report['offramp_commit'][:7]}`. "
        "The scan reads each repo's .py files and never imports or runs them, makes no model call "
        "and needs no key. Raw data: [ecosystem-scan.json](ecosystem-scan.json). Regenerate with "
        "`uv run python scripts/ecosystem_scan.py`.",
        "",
        "## Totals",
        "",
        f"- **{t['call_sites']} call sites in {t['repos_with_sites']} of {t['repos']} repos**: "
        f"{chat} chat or completion calls and {t['embedding_sites']} embedding calls ({sdks}).",
        f"- **{pct(det, chat)} of the chat calls ({det} of {chat}) have a classification, JSON or "
        "tool-call shape**, the shapes Offramp scores deterministically. The call's own `tools` or "
        f"`response_format` settles {det_params} of them; the rest are guessed from the prompt text.",
        f"- **{pct(streaming, chat)} stream** ({t['streams']} pass `stream=True`, "
        f"{t['may_stream']} set `stream` at runtime). Offramp flags streaming calls for manual review.",
        f"- **Sunset risk: {pct(risk, known)} of the chat calls whose model is named in the code "
        f"({risk} of {known})** use a model its provider has retired ({t['sunset_retired']}) or "
        f"scheduled for shutdown ({t['sunset_deprecated']}), per the provider deprecation pages "
        f"read on {SUNSET_AS_OF}. The other {chat - known} chat calls set the model at runtime, so "
        "a static scan cannot name it.",
        "",
        "## Call shapes (chat calls)",
        "",
        "| Shape | Sites | Share | Settled by the call's parameters |",
        "|---|---:|---:|---:|",
        *(
            f"| {shape} | {n} | {pct(n, chat)} | {t['shapes_from_parameters'].get(shape, 0)} |"
            for shape, n in t["shapes"].items()
        ),
        "",
        "`unknown`: the prompt is built at runtime and the call passes no `tools` or "
        "`response_format`. Shapes not settled by the parameters are guessed from the prompt text "
        "(it asks for JSON, or to classify or pick one of a set); a full Offramp run labels each "
        "call with a model before it tests anything.",
        "",
        "## Models named in the code",
        "",
        "| Model | Chat sites | Sunset risk |",
        "|---|---:|---|",
        *(
            f"| `{m}` | {n} | {f'{s[0]}, shutdown {s[1]}' if (s := sunset(m)) else '-'} |"
            for m, n in t["models"].items()
        ),
        "",
        "## Repos",
        "",
        "| Repo | Stars | Commit scanned | .py files | Chat sites | Embedding sites |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in repos:
        n_chat = sum(row["shape"] != "embedding" for row in r["rows"])
        sha = r["commit"]
        link = f"[`{sha[:7]}`](https://github.com/{r['repo']}/tree/{sha})" if sha else "?"
        lines.append(
            f"| [{r['repo']}](https://github.com/{r['repo']}) | {r['stars']:,} | {link} | "
            f"{r['python_files']:,} | {n_chat} | {len(r['rows']) - n_chat} |"
        )
    lines += [
        "",
        "## How the repos were picked",
        "",
        f"GitHub repository search, most stars first: {queries}. Each candidate was downloaded as "
        "GitHub's HEAD tarball with /api/scan's caps (25 MB download, 5,000 .py files; "
        f"{TIME_LIMIT_S:.0f} s instead of 20 s, since this ran locally) and kept once a file the "
        f"scanner reads imports `openai` or `anthropic`, until {t['repos']} were kept. "
        "Skipped on the way:",
        "",
        *(f"- {s['repo']} ({s['stars']:,} stars): {s['reason']}" for s in report["skipped"]),
        "",
        "Limits: the scan finds direct calls through the OpenAI, Anthropic, LiteLLM and LangChain "
        "clients. It does not count a client passed in from a file that never imports its SDK, a "
        "call through an alias of the method (`completion = client.chat.completions.create`), "
        "structured-output `.parse()` calls, the pre-1.0 `openai.ChatCompletion` API, or "
        "notebooks, so these numbers are a lower bound. A repo whose calls all go through another "
        "library (LlamaIndex, its own SDK) shows 0 sites. No issues or pull requests were opened "
        "on these repos.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target", type=int, default=30, help="repos to keep (default 30)")
    p.add_argument("--cache", type=Path, help="keep downloaded tarballs here and reuse them")
    args = p.parse_args(argv)
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    found = candidates()
    skipped = [c | {"reason": EXCLUDED[c["repo"]]} for c in found if c["repo"] in EXCLUDED]
    todo = iter(c for c in found if c["repo"] not in EXCLUDED)
    kept, window = [], deque()
    # Downloads run WORKERS at a time, but results are taken in star order, so the kept set is the same as
    # a one-by-one run's: the first `target` repos that qualify.
    pool = ThreadPoolExecutor(WORKERS)
    while len(kept) < args.target:
        while len(window) < WORKERS and (c := next(todo, None)):
            window.append((c, pool.submit(scan_one, c["repo"], args.cache)))
        if not window:
            break
        c, future = window.popleft()
        out = future.result()
        print(f"{c['repo']}: {out.get('skipped') or str(len(out['rows'])) + ' sites'}", file=sys.stderr)
        if "skipped" in out:
            skipped.append(c | {"reason": out["skipped"]})
        else:
            kept.append(c | out)
    pool.shutdown(wait=False, cancel_futures=True)
    skipped.sort(key=lambda c: -c["stars"])
    git = ["git", "rev-parse", "HEAD"]
    commit = subprocess.run(git, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {
        "date": date.today().isoformat(),
        "offramp_commit": commit,
        "queries": QUERIES,
        "sunset_sources": SUNSET_SOURCES | {"as_of": SUNSET_AS_OF},
        "totals": summarize(kept),
        "repos": kept,
        "skipped": skipped,
    }
    OUT.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    OUT.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["totals"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
