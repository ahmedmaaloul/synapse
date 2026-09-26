# Known limitations

The complete list, grouped by area. The [README](../README.md#7-known-limitations) repeats the
most important ones. Where a document and the code disagree, the code is authoritative.

## Contents

1. [Security and operation](#1-security-and-operation)
2. [Ingestion and the graph](#2-ingestion-and-the-graph)
3. [Retrieval and chat](#3-retrieval-and-chat)
4. [Procedural memory and the Navigator](#4-procedural-memory-and-the-navigator)
5. [Synapse Lab](#5-synapse-lab)
6. [Providers](#6-providers)
7. [Client package, distribution and releases](#7-client-package-distribution-and-releases)
8. [Evidence, tests and documentation](#8-evidence-tests-and-documentation)

## 1. Security and operation

- The backend and the MCP HTTP transport have no authentication and no rate limiting. Anyone who
  can reach port 8000 can upload documents, delete the graph, spend the configured API keys (a Lab
  run's `max_usd` cap is chosen by the caller, up to 100,000 USD) and store procedural graphs whose
  text is injected into agent prompts. The client package sends `SYNAPSE_API_KEY` as
  a bearer token for an authenticating reverse proxy; the backend itself ignores it. The MCP image
  binds `0.0.0.0`.
- **Clear Database** in the UI deletes the knowledge graph with one click, without confirmation.
- Ingest jobs, theme rebuilds, evolution locks, the guidance cache and Lab job state live in one
  process. Jobs are lost on restart, and the stack is not built for more than one backend replica.
- The backend image runs as root; the frontend and MCP images run as unprivileged users.
- `GET /health/ready` returns HTTP 200 even when Neo4j is down, with the status `degraded` in the
  body.
- Neo4j runs on the floating tag `neo4j:5-community`, not a pinned minor version.
- Docker Compose publishes ports 3000, 8000, 7474 and 7687 on all host interfaces. Neo4j keeps the
  password `synapse_secret` unless `NEO4J_PASSWORD` is changed in `.env` before the first start.

## 2. Ingestion and the graph

- PDF only, recognised by the `.pdf` extension. No OCR: scanned PDFs are rejected. No upload size
  limit; the whole file is read into memory.
- Text beyond `MAX_CHUNKS` (40 chunks, at most 32,200 characters) is dropped. The UI does not
  report the cut; only an INFO-level log line records it.
- Theme schemas are instructions in the extraction prompt. Types are written as the model returns
  them, and an unknown theme name uses the Generic schema. The UI defaults to "Personal CV /
  Resume"; the API, CLI and MCP default to `Generic`.
- An entity is identified by its exact name. Two same-named entities of different types become
  one node, and each write overwrites `type`, `description` and `document`.
- The "N nodes · M edges" message counts write statements, not newly created nodes and edges.
- Writes are not wrapped in one transaction per document; a crash can leave a partial document.
  Individual documents cannot be listed or deleted.
- Themes are rebuilt from scratch after every upload, with one LLM call per theme, so the cost of
  an ingest grows with the size of the corpus. The progress bar does not show this phase. No lock
  serialises concurrent ingests or rebuilds.
- `GET /api/graph-data` returns the whole entity graph without pagination. The UI polls it every
  15 s, and the MCP tools `synapse_find_entities` and `synapse_graph_stats` fetch it on every call.
  Response time grows with the size of the graph.

## 3. Retrieval and chat

- Query routing uses English regular expressions and misroutes some questions; for example the
  UI suggestion "Summarize the key entities" is routed to global search.
- Seeds have no relevance floor: vector search always returns the nearest entities. The context
  can be unrelated to the question; the prompt asks the model to say so.
- Retrieval uses only the current question. A follow-up such as "what about him?" is retrieved
  without the earlier turns.
- Citations are the seed entities placed in the prompt (their neighbours and the entities on
  reasoning paths are in the prompt without a citation), not a verification of what the answer
  used.
- `/api/chat` applies no overall context budget, and each seed brings all of its 1-hop
  relationships, so a highly connected entity enlarges the prompt. `k` is fixed at 8 for chat.
- In `/api/retrieve`, `max_context_chars` limits the context text only. The truncation marker can
  exceed the budget by about 34 characters, and citations, paths and excerpts are returned outside
  the budget.
- Token figures in chat and retrieval are estimates (characters divided by 4). There is no
  per-request cost accounting for chat or ingest.

## 4. Procedural memory and the Navigator

- The results of the Procedural Graphs paper have not been reproduced, and no procedural benchmark
  results are published. No published result shows whether raw guidance improves answers.
- Small models can exhaust the step budget. When the screenshots were taken (2026-09-26),
  gpt-5-nano at minimal reasoning effort stopped at the 8-step limit without calling `answer` in 3 of 3 runs on
  2-hop and 3-hop demo questions (an observation, not a benchmark). Raise `AGENT_MAX_STEPS`, or
  `max_steps` (1 to 20) per request, or use a stronger model; every step is one LLM call.
- The demo graph has no passages, so `read_sources` and `search_passages` return nothing there,
  and transitions that recommend them cost a step.
- One configured model acts as solver, guidance model and refiner. Temperature 0 does not make
  runs deterministic.
- The evolution gate compares mean scores on a small validation set, so its resolution is one
  question (1/9 on the demo set). An accepted round is a
  search step, not evidence of improvement. Recorded trajectories are stored but not used by
  evolution. There is no UI for evolution, rejections or trajectories.
- Semantic localisation is calibrated for the fastembed model only, and at 0.72 one of 48 tested
  paraphrases still lands on the wrong node.
- Deleting a procedural graph through the API also deletes its version history.

## 5. Synapse Lab

- The reader and Batch mode use the OpenAI API only.
- The ranking column, $ per 100 correct, counts reader dollars only. The LLM extraction that N1 and
  the graph arms depend on is charged only in the amortized cost-of-pass, and only when its
  measured cost is passed with `--ingest-usd` (CLI or API; the UI has no field for it). Without
  it, the leaderboard shows no ingest cost for any arm.
- Prices come from a hand-recorded table (checked on 2026-07-21; the gpt-5 rows and the Batch
  discount on 2026-09-24) and are never fetched.
- Answers are scored by Exact Match and F1 only, with no LLM judge; this does not suit open-ended
  questions.
- The effect floor is 100/n points (11.1 on the 9 demo test questions), and each comparison is
  tested on its own, with no correction for multiple comparisons.
- On the demo graph, `bm25`, `dense`, N2 and `ppr` retrieve nothing, because there are no
  passages.
- The estimator's ingest token constants are fixed in the code and cannot be re-derived from the
  repository; `backend/lab_runs/calibration.json` replaces them with your own measurements
  ([lab.md](./lab.md#calibration)).
- Under Docker, the run manifest records the git commit as `unknown` unless `SYNAPSE_GIT_SHA` is
  set.

## 6. Providers

- Vertex AI, Bedrock, Groq, Mistral and Cohere need the optional SDKs, which the default image
  leaves out ([configuration.md](./configuration.md#1-chat-model)). Their pins are held back by the `langchain-core` 0.3 ceiling.
- JSON mode is not requested for `claude`, `bedrock` and `mistral`; extraction relies on the
  prompt and tolerant parsing there.
- The Gemini embedding default `models/text-embedding-004` is listed by Google as shut down on
  2026-01-14. With `EMBEDDING_PROVIDER=gemini`, set `GEMINI_EMBEDDING_MODEL` to a current model and
  `EMBEDDING_DIM` to its dimension.
- Credentials are checked at the first call. An unreachable Ollama server is detected only then.
- `LLM_PROVIDER` values are case-sensitive; use `claude`, not `anthropic`.
- `fake` embeddings are hash-based and give no semantic retrieval.
- The Neo4j vector indexes are created at the first start with `EMBEDDING_DIM` dimensions and never
  resized. Switching the embedding model means `docker compose down -v`, which deletes all data
  including evolved procedural graph versions, and re-ingesting
  ([configuration.md](./configuration.md#2-embedding-model)).

## 7. Client package, distribution and releases

- `synapse-graphrag` is not on PyPI and the MCP server is not in the MCP Registry. Install it
  from the repository ([mcp.md](./mcp.md)).
- The published artefacts (GitHub Release `v0.4.0` wheel and sdist, GHCR images tagged `0.4.0`,
  `0.4` and `latest`) are built from tag `v0.4.0`. They carry AGPL-3.0-or-later, and the MCP image
  there has 8 tools and 2 prompts: no procedural tools, no `synapse_lab_runs`, no
  `follow_procedure`. The current licences, Procedural Graphs and the Lab exist only on `main`.
- All version sources on `main` still read 0.4.0.
- The `synapse_retrieve` cache is per process and is not cleared by an ingest; results can be
  stale for up to `SYNAPSE_CACHE_TTL` seconds.
- The branch ruleset lets the admin role bypass the required checks, so a push can reach `main` with
  a failing job. The release workflow runs none of the test suites, so it can still release such
  a commit. The CI badge shows the current state.
- GitHub's licence detection shows "Other" for PolyForm Noncommercial.

## 8. Evidence, tests and documentation

- The evidence for retrieval quality is small and does not favour graph retrieval; see
  [development.md](./development.md#5-evaluation-status).
- The frontend has no automated tests.
- The `lab` CLI commands are documented in full only in [lab.md](./lab.md), not in
  [mcp.md](./mcp.md) or the package README.

