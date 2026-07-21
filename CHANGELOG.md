# Changelog

All notable changes to **Synapse** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Maintained by **Ahmed Maaloul** <ahmed.maaloul@proton.me> ·
Licensed under **AGPL-3.0-or-later**; a separate commercial licence is required for
closed-source, proprietary or SaaS use.

## [Unreleased]

### Added

- Community & DX scaffolding: GitHub issue forms (bug report, feature request, **new AI provider
  request**), pull-request template wired to the real `make test` / `make lint` gates, Code of
  Conduct (Contributor Covenant 2.1), `CHANGELOG.md`, `.editorconfig`, Dependabot config, and a
  one-click **GitHub Codespaces devcontainer** (Python 3.12 + Node 20, ports 3000/8000/7474/7687).

### Changed

- **Relicensed from MIT to AGPL-3.0-or-later, plus a commercial licence.** Everything up to and
  including the `91ee2f2` commit was published under MIT and remains available under MIT; from
  this release onward Synapse is AGPL-3.0-or-later. Self-hosting, study, modification,
  forking and internal use stay free — a separate [commercial licence](./COMMERCIAL-LICENSE.md),
  granted only by Ahmed Maaloul, is required for closed-source, proprietary or SaaS use. Adds
  `NOTICE`, `COMMERCIAL-LICENSE.md`, `CLA.md`, `SECURITY.md`, SPDX headers across the source tree,
  and an `/api/about` endpoint that satisfies AGPL section 13 for network users.
- **Contributor License Agreement.** Contributors keep their copyright and additionally grant the
  right to relicense — the grant that makes dual-licensing legally possible. Accepted via a
  checkbox in the pull-request template.
- `/api/graph-data` is now scoped to `:Entity`. Community detection writes `(:Community)` nodes
  joined by `[:IN_COMMUNITY]`; the previously unscoped `MATCH (n)` would have pulled those into
  the visualization as unnamed grey nodes.

### Fixed

- **PEP 639 build break.** `backend/pyproject.toml` declared both an SPDX `license` expression and
  the legacy `License :: OSI Approved :: ...` classifier; setuptools ≥77 refuses that combination
  and fails the build outright.
- **Gemini JSON mode was a silent no-op.** `model_kwargs={"response_mime_type": ...}` is never read
  by `langchain-google-genai` 2.1.x — only `generation_config` is, and only as an invoke-time
  kwarg. Extraction now binds it correctly, so structured output is actually enforced.
- **Repo-root `.env` was ignored** when the API was launched from `backend/` (pydantic resolved
  `.env` relative to the process CWD). Both locations are now checked.
- **`pytest` printed no summary.** `addopts = "-q"` combined with the `-q` already passed by the
  Makefile and CI produced `-qq`, suppressing the "N passed" line entirely.

## [0.3.0] — 2026-07-19

The release that turns the demo into a platform: genuinely pluggable AI, vector-grounded
retrieval, streaming end to end, and a test suite plus CI that keep it honest.

### Added

- **Pluggable AI providers.** Chat and embedding backends sit behind a single factory in
  `backend/app/services/llm_provider.py`, selected by the `LLM_PROVIDER` / `EMBEDDING_PROVIDER`
  environment variables — no code changes to switch vendors. Missing keys or unknown provider
  names fail fast with an actionable error instead of a stack trace at request time.
- **Local-first embeddings.** `fastembed` (`bge-small-en-v1.5`, 384-dim) is the default, so the
  full stack boots and ingests with **no API key at all**; cloud embedding providers remain a
  one-line swap, with `EMBEDDING_DIM` documented as needing to match the model.
- **Vector GraphRAG retrieval.** Entity embeddings are written to a **Neo4j vector index** and
  queried alongside a full-text index; hybrid seeds are merged with the vector ranking preserved,
  then expanded to their 1-hop neighbourhood so the model reasons over relationships rather than
  isolated chunks.
- **SSE streaming everywhere.** Ingestion publishes live job progress and chat answers stream
  token-by-token over Server-Sent Events, with the grounding entities highlighted in the graph as
  the answer arrives.
- **Grounded citations.** Answers return the entity ids that actually grounded them, clickable
  straight through to the corresponding nodes in the graph view.
- **Retrieval evaluation harness** (`backend/eval/`, `make eval`) scoring the pipeline on
  paraphrased queries that keyword search misses — Hit@1 88% · Recall@8 100% · MRR 0.92.
- **Test suite:** 55 hermetic unit tests (no database, no network, no LLM) plus integration tests
  that run against a real Neo4j, gated behind `SYNAPSE_IT=1`.
- **CI on GitHub Actions:** backend lint + unit tests, integration tests and eval against a live
  `neo4j:5-community` service container, frontend ESLint + `tsc --noEmit` + production build, and
  a job proving both Docker images build clean.
- **Developer ergonomics:** a self-documenting `Makefile` (`up`, `test`, `test-int`, `eval`,
  `lint`, `fmt`), an annotated `.env.example` where every value has a safe default, and
  `ARCHITECTURE.md` / `DEPLOYMENT.md` / `CONTRIBUTING.md`.

### Changed

- Ingestion moved to a **background job** with a progress stream, so large PDFs no longer block
  the upload request.
- Extraction now **dedupes and canonicalises** entities before writing, cutting near-duplicate
  nodes and producing a materially cleaner graph.
- PDF chunking is **sentence-aware** rather than fixed-width, which keeps entity mentions intact
  across chunk boundaries.
- Configuration consolidated into a typed settings object (`backend/app/config.py`) instead of
  ad-hoc `os.environ` reads.
- Backend linting standardised on **ruff** (`E, F, I, UP, B, C4`, line length 100) and the whole
  codebase brought clean.

### Fixed

- Host port collisions: `BACKEND_PORT` / `FRONTEND_PORT` are overridable in `.env`, with the
  build-time `NEXT_PUBLIC_API_URL` caveat documented.
- Backend container hot-reload no longer shadows the image's site-packages with the host mount.
- CORS origins are configurable via `CORS_ORIGINS` instead of hard-coded.

### Security

- No secrets in the repository: every provider key is read from the environment, `.env` is
  git-ignored, and `.env.example` ships with empty key fields.

## [0.2.0] — 2026-02-20

### Added

- Neo4j-backed knowledge graph with an interactive force-directed graph view.
- PDF upload → LLM entity/relationship extraction → graph write pipeline.
- Chat over the graph with a FastAPI backend and a Next.js frontend.
- Docker Compose stack (Neo4j + backend + frontend) as the primary way to run the project.

## [0.1.0] — 2026-02-20

### Added

- Initial prototype: FastAPI service, Next.js UI, and the first end-to-end
  document-to-graph-to-answer loop.

[Unreleased]: https://github.com/ahmedmaaloul/synapse/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ahmedmaaloul/synapse/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ahmedmaaloul/synapse/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ahmedmaaloul/synapse/releases/tag/v0.1.0
