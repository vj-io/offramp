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
- `?replay=demo`: `public/replay/demo-events.json`, recorded from a real run and played at 4x speed
  (gaps capped at 1.5 s). Until that file exists the page says "No recorded demo run yet".
- `?fixture=dev`: `fixtures/dev-events.json`, made-up numbers for building the UI, labelled on screen.
  Development builds only; it is never served from `public/`.
