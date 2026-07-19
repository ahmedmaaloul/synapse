# Synapse — developer task runner
# Copyright (c) 2026 Ahmed Maaloul · AGPL-3.0-or-later
# https://github.com/ahmedmaaloul/synapse
#
# `make` on its own prints every documented target.

.DEFAULT_GOAL := help
.PHONY: help up down logs rebuild demo demo-local backend-dev frontend-dev \
        providers test test-int eval benchmark lint fmt clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Docker stack ──────────────────────────────────────────
up: ## Build & start the full stack (neo4j + backend + frontend)
	docker compose up -d --build

down: ## Stop the stack
	docker compose down

logs: ## Tail all container logs
	docker compose logs -f

rebuild: ## Rebuild images from scratch
	docker compose build --no-cache

# ── Zero-API-key demo ─────────────────────────────────────
demo: ## Seed the demo knowledge graph — no API key (needs: make up)
	docker compose exec -T backend python -m scripts.seed_demo --clear

demo-local: ## Same seed, but from a local venv (needs: docker compose up -d neo4j)
	cd backend && NEO4J_URI=bolt://localhost:7687 python -m scripts.seed_demo --clear

# ── Local dev (outside Docker) ────────────────────────────
backend-dev: ## Run the API locally with hot reload (needs neo4j up)
	cd backend && uvicorn app.main:app --reload --port 8000

frontend-dev: ## Run the Next.js dev server
	cd frontend && npm run dev

providers: ## Install the optional AI provider SDKs (vertex, bedrock, groq, mistral, cohere)
	pip install -r backend/requirements-providers.txt

# ── Quality gates ─────────────────────────────────────────
test: ## Backend unit tests (hermetic)
	cd backend && pytest -q

test-int: ## Backend integration tests (needs: docker compose up -d neo4j)
	cd backend && SYNAPSE_IT=1 EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7687 pytest tests/integration -v

eval: ## Run the retrieval evaluation harness (needs neo4j up)
	cd backend && EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7687 python -m eval.run_eval

benchmark: ## GraphRAG vs naive vector RAG head-to-head — replaces graph data (needs neo4j up)
	cd backend && EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7687 python -m benchmarks.run_benchmark

lint: ## Lint backend (ruff) + frontend (eslint + tsc)
	cd backend && ruff check .
	cd frontend && npx eslint src && npx tsc --noEmit

fmt: ## Auto-fix backend lint issues
	cd backend && ruff check --fix . && ruff format .

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.pytest_cache backend/.ruff_cache frontend/.next
