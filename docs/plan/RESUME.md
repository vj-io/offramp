# Resume the Offramp build

Paused 2026-10-20 (third pause). All agents stopped; every branch is pushed to `origin`.
Ledger with every ruling: `.superpowers/sdd/PLAN/progress.md` (gitignored; trust it and `git log` over memory).
Credits-only rule: spend only the granted credits; no AI Cloud, no paid hosting.

## Main

`main` = `0760b09` + this note (pushed). Merged: Tasks 1–13, 17 (+ T3 integration, T4 free-text follow-up).
Last full suite: 1259 passed, 18 skipped (at c1ad401); 0760b09 adds a relative-path fix with a red/green test.

## Open work

| Item | State | Next step |
|---|---|---|
| Task 10b patcher robustness | `wave3/task-10b` @ `0df1c3e`, implementer done; review was stopped before it reported | re-dispatch the task review with `review-4634033..0df1c3e-code.diff` (brief = requirements quoted in the ledger "Task 10b" lines); merge into main after (check pipeline.py call sites) |
| Task 14 live demo | `offramp prepare` done live on the local demo checkout `../offramp-demo-app` (generated `evals/` files are UNTRACKED there, not committed): cs_01 24, cs_02 20, cs_03 16, cs_04 24 (no refs yet), cs_05 24 (oracle), cs_06 21 (no refs yet). Blind bundle exported: `var/audit/offramp-demo-app-65c49edf/` | 1) hand ONLY that folder to a Claude Sonnet subagent → `<folder>/answers.jsonl` (GC27: blind auditor + free-text closed references); 2) `python -m offramp.agents.reference_audit <abs demo path> --closed <folder>/answers.jsonl`; 3) optional Tendem audit (quote price, ask user); 4) commit `evals/` to the demo repo; 5) live `offramp run` (watch the 40 Ultra-call cap), export replay, real PR, commit `evals/results/*.json` |
| Task 15 deploy | not started | Vercel free tier (replay without backend); API documented as local |
| Task 16 video + Devpost | not started | brag-scaled:brag-slim; docs/SUBMISSION.md; flip repos public only with user OK |
| Final whole-branch review | not started | most capable model |

Always pass ABSOLUTE paths to `offramp prepare` / `reference_audit` (fixed in 0760b09, but keep the habit).
Use `set -a; . ./.env; set +a; export OFFRAMP_DATA_DIR=$PWD/var` before live commands (never print .env).

## Resume prompt

"Resume the Offramp build from docs/plan/RESUME.md: re-dispatch the paused tasks and continue the remaining waves."
