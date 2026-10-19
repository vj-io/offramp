# Resume point — paused 2026-10-19 (session limit)

Ledger with every ruling and review outcome: `.superpowers/sdd/PLAN/progress.md` (gitignored, on disk).
Briefs, reports, reviews: `.superpowers/sdd/PLAN/task-*-{brief,report,review,rereview-*}.md`.
Worktrees: `../wt/task-N` on branches `wave1/task-N` / `wave2/task-N` (also pushed to origin).

## Merged to main (reviewed clean)
Task 1 TF client/cache/tiers/budget/pricing/db · Task 2 demo app (separate repo vj-io/offramp-demo-app) ·
Task 3 scanner (+ integration follow-up) · Task 4 scorers · Task 5 doodle dashboard.

## In flight when paused
| Task | Branch / state | Next step on resume |
|---|---|---|
| 6 adapter | wave2/task-6 @ 075baa5 — fix round 1 done | dispatch scoped re-review (6b103bf..075baa5); merge if clean |
| 7 cases + references | wave2/task-7 @ 4bb8c77 = WIP commit (implementer stopped mid-work; demo repo has evals/reference/refund_oracle.py @ 8d76134) | resume implementer: read task-7-report.md + WIP diff, finish, squash WIP, report |
| 8 runner + judge | wave2/task-8 @ e306eed = WIP over fix round 1 (provider errors, missing references, p95, return_candidates) | resume implementer to finish fix round 1, then scoped re-review |
| 10 patcher + PR | wave2/task-10 @ d191191 = WIP (was recording patcher cassettes live) | resume implementer: finish, then task review |
| 17 Tavily/LangSmith/fallback keys | wave2/task-17 @ 0a89e37 — implementation done | dispatch task review (review was stopped) |

## Not started
Task 9 iterator/verdict/routing · Task 11 serverless job · Task 12 API/CLI/pipeline · Task 13 CI/docs/README/FEEDBACK ·
Task 14 live demo run (cases+references, GC27 Claude closed references, live pipeline, real PR) · Task 15 deploy (Vercel + API) ·
Task 16 video + Devpost submission materials (docs/plan/PLAN.md "Devpost submission deliverables").

## Keys (offramp/.env, all verified)
NEBIUS, TAVILY, LANGSMITH, TENDEM primary + `_FALLBACK` (teammate account). Toloka: no API on new platform (UI only).
