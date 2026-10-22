"""`offramp` command line.

offramp scan <path> [--json]                 closed-LLM call sites (no key, no model calls)
offramp prepare <path>                       eval cases + references into <path>/evals (TF calls)
offramp run <repo> [--no-pr] [--cache-only]  the whole pipeline on a repo URL or local path
offramp run --resume <run_id> [--cache-only] an interrupted or failed run, after its last finished step
offramp serve [--host H] [--port P]          the API (offramp/api/app.py)
offramp export-replay <run_id>               both replay files, from a finished run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import offramp
from offramp import db, pipeline
from offramp.schemas import RunConfig, RunEvent, Stage

WEB_REPLAY = Path(offramp.__file__).resolve().parent.parent / "web" / "public" / "replay" / "demo-events.json"
REPLAY_LIMIT_S = 90  # Global Constraint 14: a replay finishes in under 90 s


def _client(args: argparse.Namespace):
    from offramp.tf import Budget, TFClient

    return TFClient(cache_dir=args.cache_dir, live=not args.cache_only, budget=Budget())


def _spend(client) -> str:
    s = client.budget.snapshot()
    return (
        f"Token Factory, uncached calls only: {s['tokens_in']} tokens in, {s['tokens_out']} out, "
        f"{s['ultra_agent_calls']} Ultra agent calls, ${s['tf_spend_usd']:.4f}"
    )


def scan(args: argparse.Namespace) -> int:
    from offramp import scan_table
    from offramp.agents.scanner import scan_repo

    result = scan_repo(args.path)
    print(result.model_dump_json(indent=2) if args.json else scan_table.render(result))
    return 0


def prepare(args: argparse.Namespace) -> int:
    from offramp.agents.case_gen import prepare_repo
    from offramp.agents.scanner import label_contracts, scan_repo

    client = _client(args)
    result = scan_repo(args.path)
    if args.site:  # regenerate only these sites; the others' files stay as they are
        unknown = sorted(set(args.site) - {s.id for s in result.sites})
        if unknown:
            known = ", ".join(s.id for s in result.sites)
            print(f"unknown site id(s): {', '.join(unknown)} (this repo has: {known})", file=sys.stderr)
            return 2
        result.sites = [s for s in result.sites if s.id in args.site]
    result.sites = label_contracts(result.sites, client, root=args.path)
    print(json.dumps(prepare_repo(args.path, result, client), indent=1))
    print(_spend(client), file=sys.stderr)
    print(
        f"next: python -m offramp.agents.reference_audit {args.path} --export-blind"
        + (f" --sites {','.join(args.site)}" if args.site else "")
        + "  (blind reference audit)",
        file=sys.stderr,
    )
    return 0


def _show(e: RunEvent) -> None:
    p = e.payload
    what = {
        "stage": p.get("detail", ""),
        "log": f"{p.get('level')}: {p.get('message')}",
        "tier_result": f"{p.get('tier')} v{p.get('adapter_version')} {p.get('split')}: "
        f"{'pass' if p.get('passed') else 'fail'} {p.get('score')}",
        "iteration": f"adapter v{p.get('version')} for {p.get('tier')}",
        "verdict": f"{p.get('decision')}: {p.get('reason')}",
        "pr": p.get("url"),
        "error": p.get("reason"),
    }.get(e.type, "")
    site = f" {e.call_site_id}" if e.call_site_id else ""
    print(f"{e.seq:>4} {e.stage.value:<10} {e.type}{site} {what}".rstrip(), file=sys.stderr, flush=True)


def run(args: argparse.Namespace) -> int:
    root = pipeline.data_dir()
    engine = db.init_db(db.engine(root / "offramp.db"))
    if args.resume:
        log = pipeline.RunLog(root / "runs" / args.resume, args.resume, engine=engine, on_event=_show)
        if log.config is None:
            print(f"no run {args.resume} under {root / 'runs'}", file=sys.stderr)
            return 2
    elif args.repo:
        if "@" in urlsplit(args.repo).netloc:  # the run log masks credentials, so the clone would get "***"
            print("offramp run: put credentials in GITHUB_TOKEN, not in the repo URL", file=sys.stderr)
            return 2
        config = RunConfig(repo_url=args.repo, open_pr=not args.no_pr, live=not args.cache_only)
        log = pipeline.new_run(config, root, engine=engine, on_event=_show)
    else:
        print("offramp run: give a repo, or --resume <run_id>", file=sys.stderr)
        return 2
    stage = pipeline.execute_run(log, cache_dir=args.cache_dir, cache_only=args.cache_only)
    print(f"run {log.run_id}: {stage.value}  ({log.dir})")
    return 0 if stage == Stage.DONE else 1


def serve(args: argparse.Namespace) -> int:
    import uvicorn

    from offramp.api.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def _shipped_diffs(run_dir: Path) -> dict[str, str]:
    """Each migrated site's unified diff as its PR commits it (patch.json), scrubbed like the events. Only
    MIGRATE sites: a kept site's comment and a manual-review site never reach the report's diff panel. Under
    "merged", what only merging the sites changed (Task 26: an import their rewrites left unused together).
    scrub() knows only the secret-named variables in this shell's environment (the CLI loads no .env); the
    diff is the code the public PR already commits."""
    patch = run_dir / "patch.json"
    if not patch.exists():
        return {}
    p = json.loads(patch.read_text())
    migrated = {v["site_id"] for v in p["verdicts"] if v["decision"] == "MIGRATE"}
    merged = {"merged": p["merged_diff"]} if migrated and p.get("merged_diff") else {}
    return pipeline.scrub({sid: d for sid, d in p.get("diffs", {}).items() if sid in migrated} | merged)


def export_replay(args: argparse.Namespace) -> int:
    """Both replay files (Global Constraint 14) from a finished run: the API's JSON Lines, and the web's
    JSON, {"events": [...], "diffs": {site_id: unified diff}} with each migrated site's shipped change."""
    from offramp.api.app import REPLAY_FILE, pr_body_file, replay_delays

    log = pipeline.RunLog(pipeline.data_dir() / "runs" / args.run_id, args.run_id)
    if not log.events or log.events[-1].type != "done":
        last = log.events[-1].type if log.events else "none"
        print(f"run {args.run_id} has not finished (last event: {last}); only a finished run is the demo")
        return 1
    api_out, web_out = args.api_out or REPLAY_FILE, args.web_out or WEB_REPLAY
    for path in (api_out, web_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    api_out.write_text("".join(e.model_dump_json() + "\n" for e in log.events), encoding="utf-8")
    events = [e.model_dump(mode="json") for e in log.events]
    recording = {"events": events, "diffs": _shipped_diffs(log.dir)}
    web_out.write_text(json.dumps(recording, ensure_ascii=False) + "\n", encoding="utf-8")
    pr = log.dir / "pr.json"  # the PR body travels with the API's replay, so a replay's report shows it
    body = json.loads(pr.read_text())["body"] if pr.exists() else None
    if body:
        pr_body_file(api_out).write_text(body, encoding="utf-8")
    else:
        pr_body_file(api_out).unlink(missing_ok=True)
    seconds = sum(replay_delays(log.events))
    print(f"wrote {len(log.events)} events to {api_out} and {web_out}; the replay plays in {seconds:.0f} s")
    if seconds >= REPLAY_LIMIT_S:
        print(f"warning: over the {REPLAY_LIMIT_S} s replay budget (Global Constraint 14)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="offramp", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def tf_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cache-only", action="store_true", help="replay the response cache; a miss fails")
        p.add_argument("--cache-dir", type=Path, help="response cache (default $OFFRAMP_DATA_DIR/tf-cache)")

    p = sub.add_parser("scan", help="list the closed-LLM call sites of a repo (no model calls)")
    p.add_argument("path", type=Path)
    p.add_argument("--json", action="store_true", help="the full scan result as JSON instead of the table")
    p.set_defaults(func=scan)

    p = sub.add_parser("prepare", help="generate eval cases and references into <path>/evals")
    p.add_argument("path", type=Path)
    p.add_argument("--site", action="append", help="only this call site (repeatable), e.g. --site cs_03")
    tf_options(p)
    p.set_defaults(func=prepare)

    p = sub.add_parser("run", help="run the pipeline on a repo (https URL or local path)")
    p.add_argument("repo", nargs="?")
    p.add_argument("--no-pr", action="store_true", help="render the PR body but do not open the PR")
    p.add_argument("--resume", metavar="RUN_ID", help="resume this run (its config, or --cache-only)")
    tf_options(p)
    p.set_defaults(func=run)

    p = sub.add_parser("serve", help="start the API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=serve)

    p = sub.add_parser("export-replay", help="write the demo replay files from a finished run")
    p.add_argument("run_id")
    p.add_argument("--api-out", type=Path, help="default offramp/api/replay/demo-events.jsonl")
    p.add_argument("--web-out", type=Path, help="default web/public/replay/demo-events.json")
    p.set_defaults(func=export_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
