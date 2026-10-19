# syntax=docker/dockerfile:1
# The Offramp API (FastAPI, `offramp serve`). The dashboard in web/ deploys separately (Vercel); the eval job
# image is Dockerfile.eval.
#
#   docker build -t offramp-api .
#   docker run --rm -p 8000:8000 -v offramp-data:/data --env-file .env offramp-api   # --env-file is optional
#
# Without keys the API still serves the recorded replay (`?replay=demo`); keys only matter for live runs.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# git: the pipeline clones the repo it scans.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    OFFRAMP_DATA_DIR=/data
WORKDIR /app

# Dependencies first: this layer is rebuilt only when the lockfile changes.
# README.md is here because the build backend reads it (pyproject `readme`).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra tracing --no-install-project

# Explicit copies, never `COPY . .`: a local .env must not end up in an image layer.
COPY offramp ./offramp
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra tracing

RUN useradd --system --create-home offramp && mkdir /data && chown offramp /data
USER offramp

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["offramp", "serve", "--host", "0.0.0.0", "--port", "8000"]
