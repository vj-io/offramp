# Offramp dashboard

Next.js (App Router) + Tailwind, styled with the doodle UI kit (`docs/plan/doodle-ui.md`).

```bash
pnpm install
pnpm dev        # http://localhost:3000
pnpm test       # reducer tests (vitest) over fixtures/dev-events.json
pnpm build && pnpm lint
```

| Env | Default | Used for |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | `POST /api/runs`, the run's SSE stream, the report JSON |
| `OFFRAMP_DEMO_REPO` | `vj-io/offramp-demo-app` | the "Use demo repo" button (read at build time) |

Routes: `/` new run, `/runs/[id]` live run, `/runs/[id]/report` report.

Where the numbers come from (query string on the run and report pages):

- default or `?live=1`: the Offramp API (`?live=1` wins over the others).
- `?replay=demo`: `public/replay/demo-events.json`, recorded from a real run (`offramp export-replay`), played at
  4x speed with pause, 1x / 4x / 16x and skip to end (gaps capped at 1.5 s; reduced-motion viewers start paused).
- `?replay=no-ultra`: `public/replay/no-ultra-events.json`, the same repo under a team policy that caps spend
  below Ultra. Any other `?replay=` name is refused. Until a recording's file exists the page says "This
  recording isn't published yet". A recording is an event array or `{events, diffs}`; `diffs` (each migrated
  site's shipped change) feeds the report's "What changed".
- `?fixture=dev`: `fixtures/dev-events.json`, made-up numbers for building the UI, labelled on screen.
  Development builds only; it is never served from `public/`.
