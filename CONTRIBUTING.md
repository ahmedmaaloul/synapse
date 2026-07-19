# Contributing

Thanks for your interest! This is primarily a portfolio project, but issues and
PRs are welcome.

## Dev setup

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Frontend
cd frontend && npm install
```

Run the stack with `make up` (Docker) or `make backend-dev` / `make frontend-dev`
for local hot-reload.

## Before opening a PR

Everything CI enforces, run locally:

```bash
make lint     # ruff + eslint + tsc
make test     # backend unit tests
```

If you touched retrieval or ingestion, also run the integration tests and eval
against a local Neo4j:

```bash
docker compose up -d neo4j
make test-int
make eval
```

## Conventions

- **Backend:** typed, `ruff`-clean, small focused functions. New behavior needs a
  test (hermetic unit test preferred; integration test if it needs Neo4j).
- **Frontend:** no `any` at call sites, components stay presentational where
  possible, shared logic lives in `src/app/lib/`.
- **Commits:** conventional style (`feat:`, `fix:`, `docs:`, `chore:`).
