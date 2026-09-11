# Changelog

All notable changes to **Synapse** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Maintained by **Ahmed Maaloul** <ahmed.maaloul@proton.me> ·
Core licensed under **PolyForm-Noncommercial-1.0.0** — free for noncommercial use, a separate
commercial licence is required for any commercial use; the `synapse-graphrag` client package is
**Apache-2.0**.

## [Unreleased]

### Changed

- **Relicensed from AGPL-3.0-or-later to PolyForm Noncommercial 1.0.0, plus a commercial licence;
  the client package to Apache-2.0.** The core — backend, frontend, scripts, workflows, images and
  docs — is now [`PolyForm-Noncommercial-1.0.0`](./LICENSE): free for noncommercial purposes as the
  licence defines them (personal use; use by educational, public-research, charitable and the other
  nonprofit organisations the licence names, and by government institutions), with no copyleft, so
  modifications may stay private. **Any commercial use** — by or for a business,
  internal or external, on-prem or SaaS, redistributed or not — requires a
  [commercial licence](./COMMERCIAL-LICENSE.md) granted only by Ahmed Maaloul; evaluation licences
  are available on request. The `synapse-graphrag` client package (MCP server, CLI, SDK) in
  `packages/synapse-graphrag/` is [`Apache-2.0`](./packages/synapse-graphrag/LICENSE) with its own
  `LICENSE` and `NOTICE`, so agents and products can talk to Synapse with no obligations beyond
  Apache's. Prior versions stay as published: everything up to and including `91ee2f2` remains MIT;
  versions 0.3.0 and 0.4.0 (every commit after `91ee2f2` up to and including the `v0.4.0` tag,
  `34300d6`) remain AGPL-3.0-or-later. What it means for users: individuals, researchers, students,
  educators, nonprofits and public bodies lose nothing and gain the right to keep changes private;
  companies — including for purely internal use, which the AGPL versions allowed at no cost — now
  need the commercial licence. Attribution rides on PolyForm's own `Required Notice:` mechanism
  (the first line of `LICENSE`, which the licence's "Notices" section requires to travel with every
  copy); the "About"/credits attribution is a term of the commercial licence and a request under
  the noncommercial one. Updated: `LICENSE`, `NOTICE`, `COMMERCIAL-LICENSE.md`, `CLA.md`,
  `CONTRIBUTING.md`, `SECURITY.md`, the SPDX headers across the tree (path-based: core →
  `PolyForm-Noncommercial-1.0.0`, package → `Apache-2.0`), package metadata, `/api/about` — which
  still tells network users what they are running, who wrote it, where the source is and under
  which terms, and now also reports the client package's licence — and the UI footer badge.

## [0.4.0] — 2026-09-09

The release that turns Synapse into a GraphRAG *engine* you can call from anywhere: entity
resolution, communities and multi-hop reasoning inside; a retrieval-only API, an MCP server, a CLI
and a Python client outside; and release automation that ships all of it on a tag.

### Added

- **GraphRAG "brain".** *Entity resolution* merges near-duplicates only when embedding-cosine
  **and** fuzzy-name similarity agree, and never across entity types. *Communities:* Louvain
  clustering (`networkx`, `seed=42`, reproducible) with LLM-written titles and summaries, exposed
  at `GET /api/communities` and rebuildable on an existing graph with `POST /api/communities/rebuild`
  (SSE progress). *Query routing:* corpus-level questions are answered from community summaries
  ("global" search), specific ones from the subgraph ("local" search). *Multi-hop retrieval:*
  the reasoning paths between seed entities are returned alongside the answer, bounded by
  `retrieval_max_hops`. *Source chunks:* text units are persisted as `(:Chunk)` nodes linked to
  their entities and returned as `sources`, so answers carry graph structure **and** verbatim
  evidence, with provenance shown in the UI.
- **`POST /api/retrieve`** — retrieval only, no generation. Returns the GraphRAG context with
  `citations`, `paths` and `sources`, cut to an optional `max_context_chars` budget at the
  last line boundary that keeps most of it, plus a `usage` block (`context_chars`, `context_tokens_est`,
  `truncated`, and the citation / path / source counts). Built for MCP and agent hosts that
  already have an LLM and should not pay for a second one.
- **`usage` on the chat `done` event** (`context_chars`, `context_tokens_est`, `answer_chars`).
  Additive; the frontend event type accepts it.
- **`synapse-graphrag`** — a new package in `packages/synapse-graphrag/` (PyPI name
  `synapse-graphrag`, Python ≥ 3.11, runtime deps `mcp`, `httpx` and `pydantic` only):
  - an **MCP server** (`synapse-mcp`, stdio or streamable HTTP at `/mcp`) with eight tools —
    `synapse_retrieve`, `synapse_ask`, `synapse_ingest_pdf`, `synapse_communities`,
    `synapse_find_entities`, `synapse_graph_stats`, `synapse_status`, `synapse_clear_graph` — a
    `synapse://about` resource and the `answer_with_graph` and `safety_brief` prompts; `synapse_retrieve` applies a
    default budget (`SYNAPSE_MAX_CONTEXT_CHARS`), a TTL cache (`SYNAPSE_CACHE_TTL`) and reports
    `usage.cached`;
  - a **CLI** (`synapse-graphrag status | ask | retrieve | ingest | communities | stats | mcp |
    install-config`) that prints ready-to-paste config for Claude Code, Claude Desktop, Cursor,
    VS Code and Windsurf;
  - an **async Python client** (`SynapseClient`) covering every endpoint, with an SSE parser
    for the streaming ones.
  - Shipped as a `synapse-mcp` Docker image and a `docker compose --profile mcp` service; guide in
    [`docs/mcp.md`](./docs/mcp.md).
- **`AI Safety` extraction theme** (`--theme "AI Safety"` on the CLI, *AI Safety / Evals* in the
  UI): a safety vocabulary — `MODEL`, `CAPABILITY`, `RISK`, `FAILURE_MODE`, `MITIGATION`,
  `EVALUATION`, `BENCHMARK`, `INCIDENT`, `POLICY`, `DATASET` — linked by `EXHIBITS`, `POSES`,
  `MITIGATES`, `EVALUATED_BY`, `MEASURES`, `GOVERNS` and friends, with extraction rules that label
  every risk `demonstrated:` or `hypothesised:` and forbid invented incidents; plus a
  `safety_brief` MCP prompt that writes a structured, cited brief. Guide:
  [`docs/ai-safety.md`](./docs/ai-safety.md).
- **FinOps guide** — [`docs/finops.md`](./docs/finops.md): where a GraphRAG dollar goes, the setting
  that bounds each cost, and the context-budget tooling above.
- **Research-grade benchmark reporting** (`backend/benchmarks/`, `make benchmark`): both scoring
  rules reported side by side, an effect-size floor below which a gap is "too close to call", and
  numbers generated into the benchmark README by the script and guarded by a test so they cannot
  rot. Source retrieval is now **rank-aware** — chunks are ordered by how many seed entities they
  mention, then by best seed rank, *before* the limit is applied.
- **HotpotQA / 2WikiMultihopQA harness** (`backend/benchmarks/public/`): dataset loaders with
  fingerprinted caches, a deterministic seeded sampler, a token/USD cost ledger, a `--dry-run`
  that spends nothing and a hard `--questions` cap.
- **Release automation.** Pushing a `v*` tag runs `.github/workflows/release.yml`: a version
  consistency check across the four version sources, changelog-section extraction, sdist + wheel
  build, GHCR images (`synapse-backend`, `synapse-frontend`, `synapse-mcp`), a GitHub Release
  with the changelog section as notes, and opt-in PyPI trusted publishing. Helpers in `scripts/`
  (`check_versions.py`, `changelog_section.py`), `make release-check`, and a branch-protection
  ruleset (`scripts/ruleset-main.json`, `make protect-main`).
- CI gained an `MCP package · lint, tests & build` job and builds the MCP image; Dependabot
  watches the package; the devcontainer installs it in editable mode. Documentation:
  [`docs/mcp.md`](./docs/mcp.md), and new sections in `ARCHITECTURE.md` (clients, context budget),
  `DEPLOYMENT.md` (GHCR images, PyPI) and `CONTRIBUTING.md` (cutting a release).
- Community & DX scaffolding: GitHub issue forms (bug report, feature request, **new AI provider
  request**), pull-request template wired to the real `make test` / `make lint` gates, Code of
  Conduct (Contributor Covenant 2.1), `CHANGELOG.md`, `.editorconfig`, Dependabot config, and a
  one-click **GitHub Codespaces devcontainer** (Python 3.12 + Node 20, ports 3000/8000/8765/7474/7687).

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
- Community clustering, which is CPU-bound, runs in a worker thread (`asyncio.to_thread`)
  instead of blocking the event loop.
- Dependabot no longer proposes LangChain-stack bumps: the effective `langchain-core` ceiling
  cannot be expressed as a semver rule, so those upgrades are done by hand with the suites green.
- Contact address is **ahmed.maaloul@proton.me** everywhere — SPDX headers, `LICENSE`,
  `NOTICE`, package metadata and `/api/about`.

### Fixed

- **Silent data loss in entity merging.** A null-unsafe Cypher predicate dropped relationships
  to nameless nodes, duplicates were deleted even when the canonical node was absent, and
  relationships with unsafe types were deleted rather than rewired.
- **CI built the Docker images from the wrong context.** After both images moved to a
  repo-root build context (so they can `COPY LICENSE NOTICE`), the workflow still ran
  `docker build ./backend` and failed with `"/NOTICE": not found`. It now builds with
  `-f backend/Dockerfile .` and asserts the notices are present inside the image.
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

[Unreleased]: https://github.com/ahmedmaaloul/synapse/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/ahmedmaaloul/synapse/releases/tag/v0.4.0
[0.3.0]: https://github.com/ahmedmaaloul/synapse/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ahmedmaaloul/synapse/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ahmedmaaloul/synapse/releases/tag/v0.1.0
