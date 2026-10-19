# Resume the Offramp build

Paused 2026-10-19 ~18:15 (second pause). All agents stopped; every branch is pushed to `origin`.
Ledger with every ruling: `.superpowers/sdd/PLAN/progress.md` (gitignored; trust it and `git log` over memory).

## Main

`main` = `2dd592b` (pushed). Merged: Tasks 1–6, 8, 11, 13, 17 (+ T3 integration, T4 free-text follow-up).
Last full suite on main before the T13 merge: 937 passed, 10 skipped (T13 is docs/CI only, no Python changed; ruff clean).

## Open tasks

| Task | Branch @ head | State | Next step |
|---|---|---|---|
| 7 cases + references | `wave2/task-7` @ `60f352f` | fix round 3 done; re-review 3 was stopped before it reported | re-dispatch scoped re-review of `a32e076..60f352f` (findings: task-7-rereview-2.md) |
| 9 iterator/verdict/route | `wave2/task-9` @ `0e177f6` (WIP) | fix round 1 interrupted mid-edit (rebased on 5eeea9b; verdict.py/route.py partly changed) | resume implementer with task-9-review.md + ledger rulings (retry once, no digits in Super prose, minors); it must check its WIP commit first |
| 10 patcher + PR | `wave2/task-10` @ `5e0dbee` (WIP) | fix round 1 interrupted (rebased on 8819652; 14 of 20 live Super calls used, 8 patcher cassettes recorded; PR-writer cassette being replaced) | resume implementer with task-10-review.md + rulings; remaining live budget 6 Super |
| 12 orchestrator/API/CLI | — | not started | dispatch after 7 + 9 merge (brief: task-12-brief.md; carries in ledger) |
| 14 live demo | — | not started | needs 7, 9, 10, 12 |
| 15 deploy | — | not started | Vercel free tier only |
| 16 video + Devpost | — | not started | brag-scaled:brag-slim |

## Decisions since the first pause (all in the ledger)

- Credits only: no AI Cloud billing, no real Serverless Job, no paid hosting. Job path ships tested with a fake CLI.
- Nebius CLI installed (`~/.nebius/bin/nebius`, profile `offramp`); AI Cloud tenant exists but unused.
- Target client module lives inside the target package (`helphub/offramp_client.py`), not root `offramp/client.py`.

## Minors to fix before the repo goes public (from T13 re-review)

FEEDBACK.md:116 wording ("no o3 model page"); VIDEO.md:21/:34 conditional line; re-run the README path check on
the merged tree (Dockerfile.eval, scripts/audit_references.py, web/public/replay/demo-events.json).

## Resume prompt

"Resume the Offramp build from docs/plan/RESUME.md: re-dispatch the paused tasks and continue the remaining waves."
