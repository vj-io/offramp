.DEFAULT_GOAL := help
.PHONY: help install lint test cov web-test web-build ci api web demo live docker

API_PORT ?= 8000
WEB_PORT ?= 3000
# Read .env for the server when it exists. Replay needs no keys, so a missing .env is fine.
UV_RUN := uv run $(if $(wildcard .env),--env-file .env,)
OPEN := $(shell command -v open 2>/dev/null || command -v xdg-open 2>/dev/null)

help: ## List the targets
	@awk -F':.*## ' '/^[a-z-]+:.*## /{printf "  make %-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install Python (uv) and web (pnpm) dependencies
	uv sync
	cd web && pnpm install --frozen-lockfile

lint: ## ruff on the Python code, eslint on the web app
	uv run ruff check .
	cd web && pnpm lint

test: ## Python tests on recorded cassettes (no keys, no network)
	uv run pytest -q

cov: ## Python tests with the CI coverage gate (>= 70% on offramp/)
	uv run pytest -q --cov=offramp --cov-report=term-missing:skip-covered --cov-fail-under=70

web-test: ## Web reducer tests
	cd web && pnpm test

web-build: ## Production build of the web app
	cd web && pnpm build

ci: lint cov web-test web-build ## Everything CI runs, except gitleaks and the docker build

api: ## Start the API (default port 8000)
	OFFRAMP_WEB_ORIGIN=http://localhost:$(WEB_PORT) $(UV_RUN) offramp serve --host 127.0.0.1 --port $(API_PORT)

web: ## Start the dashboard (default port 3000)
	cd web && NEXT_PUBLIC_API_URL=http://localhost:$(API_PORT) pnpm dev --port $(WEB_PORT)

demo: install ## One command, no keys: install, start API + web, open the recorded replay
	@echo "API http://localhost:$(API_PORT)   web http://localhost:$(WEB_PORT)   (Ctrl-C stops both)"
	@trap 'kill 0' EXIT; \
	OFFRAMP_WEB_ORIGIN=http://localhost:$(WEB_PORT) $(UV_RUN) offramp serve --host 127.0.0.1 --port $(API_PORT) & \
	(cd web && NEXT_PUBLIC_API_URL=http://localhost:$(API_PORT) pnpm dev --port $(WEB_PORT)) & \
	for i in $$(seq 1 90); do curl -fs -o /dev/null http://localhost:$(WEB_PORT)/ && break; sleep 1; done; \
	URL="http://localhost:$(WEB_PORT)/runs/demo?replay=demo"; \
	if [ -n "$(OPEN)" ]; then $(OPEN) "$$URL"; else echo "Open $$URL"; fi; \
	wait

live: ## Re-record cassettes against Token Factory (needs NEBIUS_API_KEY; spends credit)
	$(UV_RUN) pytest --live -q

docker: ## Build the API image
	docker build -t offramp-api .
