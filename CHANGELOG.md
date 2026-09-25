# Changelog

All notable changes to **Synapse** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Maintained by **Ahmed Maaloul** <ahmed.maaloul@proton.me> ·
Core licensed under **PolyForm-Noncommercial-1.0.0** — free for noncommercial use, a separate
commercial licence is required for any commercial use; the `synapse-graphrag` client package is
**Apache-2.0**.

## [Unreleased]

### Added

- **Synapse Lab: compare retrieval approaches on your own data, FinOps-first.** Several retrieval
  approaches ("arms") run over the same questions and the same graph, under the same token budgets,
  read by one reader with one prompt, and ranked by **$ per 100 correct answers** next to three
  evidence floors. Guide: [`docs/lab.md`](./docs/lab.md).
  - **Eight arms** (`app/lab/arms.py`), each with a cited source and **0 retrieval-time LLM
    calls**:
    - three evidence floors: `null_closed_book` (N0), `null_vocabulary` (N1: every entity name,
      the question ignored) and `null_random` (N2: seeded random passages at the same budget);
    - two passage baselines: `bm25` (Lucene BM25 over the `:Chunk` full-text index) and `dense`
      (the `:Chunk` vector index);
    - three graph arms: `synapse_d` (the shipped retrieval path, split into units),
      `synapse_lean` (PathRAG-style flow-pruned paths with a LiteRAG-style log-degree hub
      penalty; [arXiv:2502.14902](https://arxiv.org/abs/2502.14902),
      [arXiv:2609.10239](https://arxiv.org/abs/2609.10239)) and `ppr` (HippoRAG-2-style
      Personalized PageRank over entities and passages, without the LLM triple filter;
      [arXiv:2502.14802](https://arxiv.org/abs/2502.14802)).

    The "-style" arms are re-implementations over Synapse's graph, not the authors' code, and
    say so in their descriptions.
  - **One packer, one reader.** Arms return ranked evidence units. One packer (`packer.py`) fills
    every budget, counted with the reader's own tokenizer (`tiktoken`; a flagged `len/4`
    fallback), and never exceeds it; PathRAG's reliability-ascending placement is an option. One
    short-answer prompt (`reader.py`) reads every arm, and its output cap covers a reasoning
    model's hidden reasoning.
  - **Metrics** (`metrics.py`, pure):
    - EM / F1, tokens per correct, and $ per 100 correct (cost-of-pass);
    - amortized cost-of-pass at 100, 1,000 and 10,000 queries per corpus;
    - gain above N0 and above N2, and the graph premium over the better passage baseline at the
      same budget;
    - F1-vs-$ and F1-vs-tokens Pareto frontiers.

    A difference is called only when it clears the one-question effect floor (100 / n points) and
    its seeded, paired-bootstrap 95% CI (10,000 resamples) excludes 0.
  - **Three modes.**
    - `retrieve` is the default and costs **$0**. It reports context tokens, units by kind, answer
      containment and HotpotQA gold-paragraph recall.
    - `realtime` reads now, metered call by call.
    - `batch` goes through the OpenAI Batch API at half price (`batch.py`). Parts stay under the
      SDK's documented limits, resubmission is idempotent, failed requests can be resent, and an
      optional gate keeps the enqueued tokens under the organisation's limit.
  - **Spend controls.** A free estimator (`estimate.py`) gives a point estimate and an upper bound
    per phase and per arm × budget cell. It refuses when the upper bound exceeds `max_usd` or
    cannot be priced. A run is checked again on its measured contexts before any call, and realtime
    reads reserve each request's worst case against the cap before sending it.
  - **Runs** (`runner.py`) are resumable directories under `backend/lab_runs/` (gitignored). Each
    holds:
    - a manifest: git SHA, arm config hashes, packer policy, dataset sha256 and question ids,
      models, tokenizer, price dates, and estimate vs actual per phase;
    - every packed context with its sha256;
    - the requests, the answers, the scored rows, `leaderboard.json` and `report.md`.

    Datasets: the demo QA set, uploaded QA files, and HotpotQA, which is refused until its sample
    is ingested.
  - **Batch ingest** (`ingest.py`): plan, submit and apply a corpus's extraction through the Batch
    API, with the same prompt, parser and writes as the realtime pipeline. Community summaries are
    off by default. `graph_builder` gains `render_extraction_request`, `parse_extraction` and
    `build_knowledge_graph_from_extractions`; `build_knowledge_graph` behaves exactly as before.
  - **API** under `/api/lab/`: arms, models, datasets, qa-files, estimate, runs, events (SSE) and
    resume.
  - **CLI** `synapse-graphrag lab arms | models | datasets | upload | estimate | run | runs | show
    | resume`. It prints the estimate first and asks for consent before any run. A client method
    covers each endpoint.
  - A read-only **MCP tool**, `synapse_lab_runs`. No MCP tool starts or resumes a run.
  - **UI**: a **Lab** view next to *Knowledge | Procedures*, with:
    - an arm picker by family;
    - an estimate table with a refusal banner;
    - live progress and a runs list;
    - the leaderboard, with floor rows styled apart;
    - a Pareto plot with the floor band shaded.
  - `benchmarks/public/cost.py` gains the Batch price multiplier (×0.5, dated) and
    `usd(..., batch=True)`.
- **Procedural memory: Procedural Graphs.** An implementation of Lu, Chen, Wu, Arık,
  *"Procedural Graphs: Self-Evolving Execution Structures for LLM Agents"*
  ([arXiv:2609.09153](https://arxiv.org/abs/2609.09153)). The entity graph records *what* the
  corpus says; a procedural graph records *how* to navigate it: a small directed graph of `ACTION`,
  `REASONING` and `STATUS` nodes whose transitions carry a `condition`, `guidance` and `pitfalls`.
  Guide: [`docs/procedural-graphs.md`](./docs/procedural-graphs.md).
  - **Data structure** (`procedural_graph.py`, pure): the refiner's edits applied in the paper's
    order, cycle repair, structural validation, and the local and full serializers.
  - **Storage** (`procedural_store.py`): graphs live in the same Neo4j under their own labels
    (`Procedure`, `ProcedureGraph`, `ProcedureVersion`, `ProcedureRejection`,
    `ProcedureTrajectory`), with uniqueness constraints created at startup. Each save is one write
    transaction (`neo4j_driver.execute_write_batch`) and appends a version with its edits and
    diff. Rollback re-saves an old version as a new one.
  - **Expert prior:** the bundled `graphrag-navigator` (11 nodes, 14 transitions) is seeded at
    startup when absent and never overwritten.
- **Step-local guidance** — `POST /api/procedures/{name}/guidance`. The agent's last action is
  localized to a node (`start` → `exact` → `normalized` → `semantic` → `none`, which means the full
  graph), and the node's outgoing transitions up to `PROCEDURAL_HOPS=2` hops are returned. The
  default `raw` mode returns that subgraph serialized, with **zero LLM calls**. `generative` is the
  paper's mode: one LLM call per step, cached in-process. The raw default and the cascade are
  Synapse's additions, documented as hypotheses to measure.
- **GraphRAG Navigator** — `POST /api/agent/ask`. A ReAct agent that answers by walking the
  knowledge graph with six deterministic tools (`search_entities`, `neighbors`, `read_sources`,
  `search_passages`, `find_path`, `answer`), steered by a procedural graph. It returns the full
  step trace, parse failures and token usage (measured, or estimated and flagged).
- **Offline self-evolution** (the paper's Algorithm 1) — `POST /api/procedures/{name}/evolve`,
  followed over SSE.
  - Rollouts on training QA pairs feed a refiner LLM that proposes edits. A candidate is kept iff
    its validation score does not drop (ties accepted); a structurally invalid one is rejected
    without a validation rollout.
  - Rejections are fed back to the refiner and stored as `ProcedureRejection` rows.
  - A hard `max_llm_calls` budget stops the run cleanly.
  - `static` and `scratch` modes; one run per graph at a time (409).
  - Scoring comes from the new `qa_metrics` module: SQuAD normalization, EM / F1, and the official
    HotpotQA yes/no rule.
- **Procedures API:** list, get (JSON or `?format=text`), `graph-data`, `PUT` (validated; a 422
  lists every diagnostic), `DELETE`, `trajectories`, `versions`, `rollback`, `rejections`.
- **14 settings**, each documented in `.env.example`: `PROCEDURAL_ENABLED`,
  `PROCEDURAL_DEFAULT_GRAPH`, `PROCEDURAL_HOPS`, `PROCEDURAL_WINDOW`, `PROCEDURAL_GUIDANCE_MODE`,
  `PROCEDURAL_SEMANTIC_THRESHOLD`, `PROCEDURAL_GUIDANCE_CACHE_SIZE`, `AGENT_MAX_STEPS`,
  `AGENT_OBSERVATION_MAX_CHARS`, `AGENT_TEMPERATURE`, `EVOLUTION_TRAJECTORY_MAX_CHARS`,
  `EVOLUTION_MAX_LLM_CALLS`, `EVOLUTION_DEFAULT_ROUNDS`, `EVOLUTION_DEFAULT_BATCH_SIZE`.
- **`synapse-graphrag`: procedural memory from any agent host.**
  - Four MCP tools: `synapse_procedures`, `synapse_procedure_guidance` (raw by default, no LLM
    call), `synapse_record_trajectory` and `synapse_agent_ask`.
  - A `follow_procedure` prompt.
  - Deliberately **no** evolve tool: it is long-running and costly.
  - CLI commands: `procedures list | show | export | import | versions | rollback | guide`,
    `agent` (prints the step trace) and `evolve`. `evolve` prints an upper-bound LLM-call estimate
    and refuses to start without `--yes` or an interactive "y".
  - A client method for every new endpoint. `SynapseError.payload` carries a 422's
    `diagnostics`.
- **UI.**
  - The graph panel gains a **Knowledge | Procedures** switch. The procedural graph is drawn with
    directed, labelled transitions, a card showing each transition's condition, guidance and
    pitfalls, and a version list with rollback.
  - The chat panel gains a **Chat | Navigator** switch that renders the agent's numbered step trace
    and its usage.
- **Procedural benchmark harness** (`backend/benchmarks/procedural/`).
  - Systems compared: `no_pg`, `pg_raw_local`, `pg_gen_local`, `pg_gen_full` and, with
    `--evolve`, the evolved graph.
  - Questions: 30 self-authored, programmatically verified QA pairs over the zero-key demo graph
    (train 12 / val 9 / test 9), or an already-ingested HotpotQA sample.
  - Report: EM / F1, steps, LLM calls and tokens under an effect floor.
  - Cost controls: a `--dry-run` cost bound, and `--max-usd` / `--max-llm-calls` hard stops.
  - Results are gitignored, and **none are committed**: the paper's numbers have not been
    reproduced.

### Changed

- **`DELETE /api/graph` keeps procedural memory.** It now deletes every node *except* those with a
  procedural label (`graph_schema.PROCEDURAL_LABELS`), and answers "Knowledge graph cleared
  (procedural memory kept)". A strategy learned over paid evolution rounds should not vanish when a
  corpus is re-ingested. `synapse_clear_graph`'s description says so; `DELETE
  /api/procedures/{name}` removes a procedural graph explicitly.
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
