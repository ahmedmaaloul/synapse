# Synapse from Claude, Cursor, VS Code & any MCP client

`synapse-graphrag` is the client side of Synapse: an **MCP server**, a **CLI** and an **async
Python client**, all thin HTTP wrappers over the backend's REST/SSE API. Nothing in the package
talks to Neo4j or to an LLM directly — the backend keeps the credentials, the schema and the
retrieval logic; the package gives every agent host the same eight tools.

| | |
| --- | --- |
| PyPI | `synapse-graphrag` (module `synapse_graphrag`) · Python ≥ 3.11 · runtime deps: `mcp`, `httpx`, `pydantic` |
| Source | [`packages/synapse-graphrag/`](../packages/synapse-graphrag/) |
| Console scripts | `synapse-graphrag` (CLI) · `synapse-mcp` (MCP server) |
| Docker image | `ghcr.io/ahmedmaaloul/synapse-mcp` — see [Docker](#docker) |
| License | Apache-2.0 — [LICENSE](../packages/synapse-graphrag/LICENSE) · the backend it talks to is PolyForm Noncommercial 1.0.0 (+ [commercial](../COMMERCIAL-LICENSE.md)) |

**Contents**

- [Prerequisites](#prerequisites)
- [Install per client](#install-per-client) — Claude Code · Claude Desktop · Cursor · VS Code · Windsurf · any client
- [Transports: stdio vs streamable HTTP](#transports-stdio-vs-streamable-http)
- [Docker](#docker)
- [Tools reference](#tools-reference)
- [Resource and prompts](#resource-and-prompts)
- [FinOps: budgeted context, not a second LLM bill](#finops-budgeted-context-not-a-second-llm-bill)
- [CLI reference](#cli-reference)
- [Python client](#python-client)
- [Environment variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Security](#security)

---

## Prerequisites

1. **A running Synapse backend.** Locally that is `make up` then `make demo` (a populated demo
   graph, no API key); or point at a deployed instance. The MCP server needs exactly one thing
   from it: its URL, `SYNAPSE_URL` (default `http://localhost:8000`). Note that the demo graph
   ships without communities (their summaries are LLM-written), so `synapse_communities` is
   empty until a document is ingested or a rebuild is run — see
   [Troubleshooting](#troubleshooting).
2. **`uv`** for the `uvx synapse-graphrag …` one-liners below —
   [install uv](https://docs.astral.sh/uv/getting-started/installation/). Without `uv`,
   `pip install synapse-graphrag` and replace `uvx synapse-graphrag mcp` with `synapse-mcp`.
   Until the first PyPI publish (it is opt-in in the release workflow), install straight from the
   repository instead — every command below works unchanged:

   ```bash
   uvx --from "git+https://github.com/ahmedmaaloul/synapse.git#subdirectory=packages/synapse-graphrag" synapse-graphrag mcp
   # or: pip install "git+https://github.com/ahmedmaaloul/synapse.git#subdirectory=packages/synapse-graphrag"
   ```

> The demo graph is enough to try every read-only tool. `synapse_ask` and `synapse_ingest_pdf`
> additionally need the backend to have an LLM key — see [Troubleshooting](#troubleshooting).

---

## Install per client

Every client below runs the same process — `uvx synapse-graphrag mcp` — over stdio, with
`SYNAPSE_URL` in its environment. The CLI prints the exact snippet for each of them:

```bash
synapse-graphrag install-config --client claude-code     # prints the `claude mcp add …` command
synapse-graphrag install-config --client cursor          # prints the JSON to paste
synapse-graphrag install-config --client vscode --url http://localhost:8000 --budget 4000
```

`--url` overrides the backend URL baked into the snippet; `--budget` sets
`SYNAPSE_MAX_CONTEXT_CHARS` for that client.

### Claude Code

```bash
claude mcp add synapse -e SYNAPSE_URL=http://localhost:8000 -- uvx synapse-graphrag mcp
```

Check with `claude mcp list`; remove with `claude mcp remove synapse`. To attach to a server
that is already running over HTTP (see [Transports](#transports-stdio-vs-streamable-http)):

```bash
claude mcp add --transport http synapse http://localhost:8765/mcp
```

### Claude Desktop

Edit the desktop config file (create it if it does not exist), then fully quit and reopen
Claude Desktop:

| OS | Path |
| --- | --- |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux (community builds) | `~/.config/Claude/claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "synapse": {
      "command": "uvx",
      "args": ["synapse-graphrag", "mcp"],
      "env": { "SYNAPSE_URL": "http://localhost:8000" }
    }
  }
}
```

> Claude Desktop starts servers with a minimal `PATH`. If it reports that `uvx` cannot be found,
> put the absolute path from `which uvx` (or `where uvx` on Windows) in `"command"`.

### Cursor

Global: `~/.cursor/mcp.json`. Per project: `<project>/.cursor/mcp.json`. Same `mcpServers` shape
as Claude Desktop:

```json
{
  "mcpServers": {
    "synapse": {
      "command": "uvx",
      "args": ["synapse-graphrag", "mcp"],
      "env": { "SYNAPSE_URL": "http://localhost:8000" }
    }
  }
}
```

For a server already running over HTTP: `{ "mcpServers": { "synapse": { "url": "http://localhost:8765/mcp" } } }`.

### VS Code

`.vscode/mcp.json` in the workspace (VS Code uses a `servers` key and an explicit `type`):

```json
{
  "servers": {
    "synapse": {
      "type": "stdio",
      "command": "uvx",
      "args": ["synapse-graphrag", "mcp"],
      "env": { "SYNAPSE_URL": "http://localhost:8000" }
    }
  }
}
```

HTTP variant: `{ "servers": { "synapse": { "type": "http", "url": "http://localhost:8765/mcp" } } }`.

### Windsurf

`~/.codeium/windsurf/mcp_config.json`, `mcpServers` shape:

```json
{
  "mcpServers": {
    "synapse": {
      "command": "uvx",
      "args": ["synapse-graphrag", "mcp"],
      "env": { "SYNAPSE_URL": "http://localhost:8000" }
    }
  }
}
```

### Any other client

- **stdio:** run `uvx synapse-graphrag mcp` (or `synapse-mcp`) with `SYNAPSE_URL` set.
- **streamable HTTP:** start the server once (`synapse-mcp --transport streamable-http`) and point
  the client at `http://<host>:8765/mcp`.

---

## Transports: stdio vs streamable HTTP

| | stdio (default) | streamable HTTP |
| --- | --- | --- |
| Start | the host spawns `synapse-mcp` per session | `synapse-mcp --transport streamable-http --host 127.0.0.1 --port 8765` |
| Endpoint | stdin / stdout of the process | `http://127.0.0.1:8765/mcp` |
| Best for | a developer's own machine, one host per server | shared servers, Docker, remote hosts, several clients at once |
| Logging | **stderr only** — stdout is the protocol channel | stderr |
| Auth | inherits the user's session | none built in — put an authenticating reverse proxy in front (see [Security](#security)) |

`synapse-mcp` and `synapse-graphrag mcp` are the same entry point. Flags:

```text
synapse-mcp [--transport {stdio,streamable-http}] [--host 127.0.0.1] [--port 8765] [--url http://localhost:8000] [--version]
```

`--url` overrides `SYNAPSE_URL`. Bind `--host 0.0.0.0` only inside a container or behind a proxy.

---

## Docker

**Compose profile** — runs the server over streamable HTTP next to the stack, already pointed at
the backend service:

```bash
docker compose --profile mcp up -d      # → http://localhost:8765/mcp
```

The service is called `mcp` (container `synapse-mcp`), sets `SYNAPSE_URL=http://backend:8000`,
waits for the backend health check, and publishes `${MCP_PORT:-8765}` — override `MCP_PORT` in
`.env` if 8765 is taken.

**Published image** (built by the release workflow for every `v*` tag — so it exists from the
first tagged release on; see [DEPLOYMENT.md](../DEPLOYMENT.md#container-images-ghcr)):

```bash
docker run --rm -p 8765:8765 \
  -e SYNAPSE_URL=http://host.docker.internal:8000 \
  ghcr.io/ahmedmaaloul/synapse-mcp:0.4.0
```

**Build it yourself** — the Dockerfile uses the repo root as build context, like every Synapse
image, and ships the package's own Apache-2.0 `LICENSE` and `NOTICE` inside the image:

```bash
docker build -f packages/synapse-graphrag/Dockerfile -t synapse-mcp .
docker run --rm synapse-mcp ls /app/LICENSE /app/NOTICE
```

> `synapse_ingest_pdf` reads the PDF from the **server's** filesystem. Inside a container that
> means mounting the file (`-v "$PWD/docs:/data:ro"`) and passing `/data/file.pdf`.

---

## Tools reference

All eight tools are namespaced `synapse_*`. Read-only tools are annotated `readOnlyHint=true`
so hosts can call them without a confirmation prompt, and every one of them except `synapse_ask`
(which bills an LLM call) is also `idempotentHint=true`; `synapse_ingest_pdf` is a
non-destructive write; `synapse_clear_graph` is marked destructive (and idempotent — clearing
twice is clearing once).

**Error contract.** Any anticipated failure — backend unreachable, HTTP error, bad path, a job
that ended without `done` — is raised as the MCP SDK's `ToolError`, which the host receives as
a tool result flagged `isError` whose text is one readable sentence, for example
``Cannot reach Synapse at http://localhost:8000: … Is it running? (`make up`)``. Tools never
return `{"error": …}` dicts and never leak a traceback. A *refusal* (`synapse_clear_graph`
without `confirm`) is a normal result that says so. Each tool call opens its own HTTP client.

### `synapse_retrieve(query, k=8, max_context_chars=None)` — the default tool

Budgeted GraphRAG context, **no LLM call on the Synapse side**. The host model reads the
context and writes the answer itself.

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `query` | `str` | — | the question |
| `k` | `int` (1–20) | `8` | top-k ranked seed entities |
| `max_context_chars` | `int \| None` (≥ 200) | `SYNAPSE_MAX_CONTEXT_CHARS` (6000) | context budget; see [FinOps](#finops-budgeted-context-not-a-second-llm-bill) |

Returns `{mode, context, citations, paths, sources, usage}`:

```json
{
  "mode": "local",
  "context": "Entities and relationships relevant to the question:\n- Analytical Engine (TOOL): …\n…",
  "citations": [
    { "name": "Charles Babbage", "type": "PERSON", "kind": "entity" },
    { "name": "Analytical Engine", "type": "TOOL", "kind": "entity" }
  ],
  "paths": [
    {
      "nodes": ["Charles Babbage", "Analytical Engine", "Ada Lovelace"],
      "rels": ["DESIGNED", "PROGRAMMED"],
      "dirs": [true, false],
      "text": "Charles Babbage -[DESIGNED]-> Analytical Engine <-[PROGRAMMED]- Ada Lovelace"
    }
  ],
  "sources": [
    { "id": "history-of-ai.pdf:3", "document": "history-of-ai.pdf", "index": 3, "text": "In 1837 Babbage described …" }
  ],
  "usage": {
    "context_chars": 3874,
    "context_tokens_est": 969,
    "truncated": false,
    "citations": 8,
    "paths": 3,
    "sources": 4,
    "cached": false
  }
}
```

`mode` is `"local"` (subgraph around the seeds) or `"global"` (community summaries, chosen by the
backend's query router for corpus-level questions — in that case citations carry
`"kind": "community"`). `paths` are the multi-hop reasoning paths the retriever found between
seeds; `dirs[i]` says whether hop `i` follows the stored edge direction. `usage.cached` is
`true` when the result came from the server-side TTL cache.

### `synapse_ask(query, history=None)`

A full answer generated by **Synapse's own LLM** (`LLM_PROVIDER` on the backend). This is a
second LLM call — the host model then reads an answer another model already paid to write — so
prefer `synapse_retrieve` unless you specifically want the backend's provider to answer.

| Argument | Type | Default |
| --- | --- | --- |
| `query` | `str` | — |
| `history` | `list[{"role": "user" \| "assistant", "content": str}] \| None` | `None` |

Returns `{text, citations, paths, sources, usage}` where `usage` is the chat `done` event's
`{context_chars, context_tokens_est, answer_chars}`. An `error` event from the backend (for
instance no LLM key configured) surfaces as a readable tool error.

### `synapse_ingest_pdf(path, theme="Generic")`

Uploads a **local** PDF and follows the ingestion job to completion. `path` must exist and end
with `.pdf`; `theme` is the free-text domain hint passed to extraction. Returns the job's `done`
payload plus the `job_id`:

```json
{
  "job_id": "5a1d…",
  "filename": "history-of-ai.pdf",
  "chunks_processed": 42,
  "nodes_created": 87,
  "relationships_created": 143,
  "communities": 6
}
```

Ingestion calls the backend's LLM for every chunk, then embeds entities and summarises
communities — this is where a GraphRAG corpus costs money (see the
[cost model](#the-cost-model-of-a-graphrag-call)).

### `synapse_communities(limit=10)`

The corpus-level themes found by Louvain clustering, each with its LLM-written title and summary
— the same summaries `mode: "global"` retrieval answers from.

```json
{
  "communities": [
    { "id": "c-3f9a…", "title": "Early Mechanical Computation", "summary": "Babbage's engines and the people around them …", "size": 7 }
  ],
  "count": 1
}
```

### `synapse_find_entities(query, limit=20)`

Case-insensitive substring match over entity **labels and types** in `/api/graph-data` — useful
for checking spelling before a retrieve, or for "what do you know about *X*" questions. Each
match carries `id`, `label`, `type`, `degree` (number of relationships) and `description` when
the entity has one:

```json
{
  "query": "babbage",
  "total_matches": 2,
  "returned": 2,
  "entities": [
    { "id": "Charles Babbage", "label": "Charles Babbage", "type": "PERSON", "degree": 6, "description": "English mathematician …" },
    { "id": "Babbage's Difference Engine", "label": "Babbage's Difference Engine", "type": "TOOL", "degree": 2 }
  ]
}
```

### `synapse_graph_stats()`

Node and edge totals plus counts **by entity type** and **by relationship type**, computed from
`/api/graph-data`. Cheap, read-only, and a good first call to see whether a graph is populated.

```json
{
  "nodes": 53,
  "edges": 106,
  "isolated_nodes": 0,
  "entity_types": { "PERSON": 21, "TOOL": 14, "CONCEPT": 12, "ORGANIZATION": 6 },
  "relationship_types": { "INFLUENCED": 31, "CREATED": 24, "WORKED_AT": 9 },
  "top_entities": [ { "label": "Alan Turing", "type": "PERSON", "degree": 11 } ]
}
```

### `synapse_status()`

`GET /health` + `GET /health/ready` + `GET /api/about` in one call: liveness, readiness (and the
Neo4j state behind it), the backend version, the active `llm_provider` / `embedding_provider`,
the URL the server is talking to and the package's own version. Call this first when anything
else fails.

```json
{
  "url": "http://localhost:8000",
  "health": "ok",
  "ready": "ready",
  "neo4j": "up",
  "version": "0.4.0",
  "llm_provider": "gemini",
  "embedding_provider": "fastembed",
  "client_version": "0.4.0"
}
```

### `synapse_clear_graph(confirm=False)`

Deletes **every** node and relationship. Refuses — with a normal result, not an error — unless
called with `confirm=true`. Annotated `destructiveHint=true`, so well-behaved hosts ask the user
first. A successful clear also empties the server's retrieve cache.

```json
{ "cleared": false, "message": "Refused: this deletes the whole knowledge graph and cannot be undone. Call again with confirm=true once the user has agreed." }
```

---

## Resource and prompts

- **Resource `synapse://about`** — the JSON of `GET /api/about` (name, version, author,
  repository, licence and commercial-licence terms, providers). It tells an agent what it is
  talking to, who wrote it and under which terms, one `resources/read` away.
- **Prompt `answer_with_graph(question)`** — a ready-made instruction: call `synapse_retrieve`
  with the question, answer **only** from the returned context, cite entity names, and say so
  plainly when the graph does not contain the answer. The server's `instructions` text tells the
  model the same thing, so a host that honours instructions behaves this way without the prompt.
- **Prompt `safety_brief(topic)`** — for corpora ingested with the `AI Safety` theme:
  retrieve evidence on a topic and write a structured brief — claims · evidence (labelled
  `demonstrated` / `hypothesised` as the entity descriptions are) · mitigations and their
  evaluation status · open questions · *not in the corpus* — with every line cited. See
  [`docs/ai-safety.md`](./ai-safety.md).

---

## FinOps: budgeted context, not a second LLM bill

### Why retrieve-only

An MCP host already has a model. A tool that *answers* the question makes that model pay to read
an answer a second model already paid to write — twice the generation cost and a round of
information loss, for nothing. Synapse's default tool therefore returns **retrieval, not
generation**: the ranked subgraph, the reasoning paths and the source excerpts, under a budget,
with an accounting block. The host does the one generation it was going to do anyway.

### The cost model of a GraphRAG call

| Phase | When it runs | What it costs | With `synapse_retrieve` |
| --- | --- | --- | --- |
| **Extraction** | once per document (`synapse_ingest_pdf`) | one LLM call per chunk + one embedding per entity + one LLM call per community summary | paid once, by the backend's provider |
| **Retrieval** | every question | one query embedding (the default `fastembed` is local and free) + Cypher against the vector / full-text indexes; no LLM | ≈ free — and served from cache on repeats |
| **Generation** | every question | one LLM call over the retrieved context | **the host's model, once** — `synapse_ask` would add the backend's call on top |

Extraction is the expensive phase and it is amortised over every later question; retrieval is
cheap; generation is the part you should only pay for once. That is the whole argument.

### Budget knobs

| Knob | Where | Effect |
| --- | --- | --- |
| `max_context_chars` | tool argument, `POST /api/retrieve` body (≥ 200) | budget for the returned `context` on that call: the text is cut within it, then a `…[context truncated to N chars]` marker (~30 chars) is appended, so `context_chars` can exceed the budget by the marker's length |
| `SYNAPSE_MAX_CONTEXT_CHARS` | env (default `6000`) | the default budget when the tool is called without one; `0` (or negative) means *no default budget*; `1`–`199` is rejected up front, since the backend's floor is 200 |
| `--budget CHARS` | `synapse-graphrag retrieve` | the same knob on the CLI, with the same env default (`0` = no budget) |
| `k` | tool argument (1–20, default 8) | fewer seeds → smaller neighbourhood → less context |

Truncation is line-aware: the context is cut at the last newline at or before the budget, provided
that keeps at least 60 % of it, otherwise hard-cut at the budget; a trailing
`…[context truncated to N chars]` line marks it and `usage.truncated` becomes `true`. Because the
context lists the highest-ranked entities first, a truncated context loses the tail, not the head.

### `usage` fields

| Field | On | Meaning |
| --- | --- | --- |
| `context_chars` | retrieve, ask | characters of context sent to (or returned for) the model — the truncation marker included, so up to ~30 above `max_context_chars` when `truncated` |
| `context_tokens_est` | retrieve, ask | `ceil(chars / 4)` — a provider-agnostic **heuristic**, not a tokenizer count |
| `truncated` | retrieve | whether the budget cut the context |
| `citations` / `paths` / `sources` | retrieve | how many of each were returned |
| `cached` | retrieve (MCP only) | served from the server-side cache, backend not called |
| `answer_chars` | ask | length of the generated answer |

Log these per call and you have per-question cost accounting for free — one of the roadmap items
is to extend it across providers.

### Cache

`synapse_retrieve` results are cached in the MCP server process for `SYNAPSE_CACHE_TTL` seconds
(default `300`), keyed by `(query, k, budget)`. A hit returns instantly with `usage.cached: true`
and never touches the backend. The cache is per process and is emptied by a successful
`synapse_clear_graph`, but it knows nothing about ingestion — after `synapse_ingest_pdf`, either
wait out the TTL, restart the server, or run with `SYNAPSE_CACHE_TTL=0` to disable caching
entirely.

---

## CLI reference

`synapse-graphrag` is a plain `argparse` CLI over the same client. Global options:
`--url URL` (overrides `SYNAPSE_URL`) and `--version`. Every command exits `1` with a one-line
message on a connection failure or backend error — never a traceback.

| Command | What it does |
| --- | --- |
| `synapse-graphrag status` | health, readiness, version and providers of the backend |
| `synapse-graphrag ask QUERY` | stream an answer generated by the backend's LLM |
| `synapse-graphrag retrieve QUERY [--k N] [--budget CHARS] [--json]` | retrieval only; `--json` prints the full `{mode, context, citations, paths, sources, usage}` object |
| `synapse-graphrag ingest FILE.pdf [--theme T]` | upload a PDF and print progress lines until the job finishes |
| `synapse-graphrag communities [--limit N]` | list community titles and summaries |
| `synapse-graphrag stats` | node/edge counts by type |
| `synapse-graphrag mcp [--transport …] [--host H] [--port P]` | run the MCP server (same as `synapse-mcp`) |
| `synapse-graphrag install-config --client {claude-code,claude-desktop,cursor,vscode,windsurf} [--url U] [--budget N]` | print the config snippet or command for that client |

```bash
synapse-graphrag status
synapse-graphrag retrieve "Who designed the Analytical Engine?" --k 6 --budget 2000 --json
synapse-graphrag ingest ./papers/graphrag.pdf --theme "Machine learning"
synapse-graphrag --url https://synapse.example.com stats
```

---

## Python client

The same package is an SDK. `SynapseClient` is async, built on `httpx.AsyncClient`, and an async
context manager; it accepts `base_url`, `api_key` and `timeout` (and a `transport` for tests).

```python
import asyncio

from synapse_graphrag import SynapseClient


async def main() -> None:
    async with SynapseClient(base_url="http://localhost:8000") as client:
        r = await client.retrieve("Who designed the Analytical Engine?", k=8, max_context_chars=4000)
        print(r.mode, r.usage)
        print(r.context)

        answer = await client.ask("Summarise the graph in two sentences.")
        print(answer.text, answer.usage)


asyncio.run(main())
```

| Method | Returns |
| --- | --- |
| `health()` · `ready()` · `about()` | the backend JSON |
| `retrieve(query, k=8, max_context_chars=None)` | `Retrieval(mode, context, citations, paths, sources, usage)` |
| `ask(query, history=None)` | `Answer(text, citations, paths, sources, usage)`; raises `SynapseError` on an `error` event |
| `ask_stream(query, history=None)` | async iterator of parsed SSE events (`citations`, `paths`, `sources`, `token`, `done`, `error`) |
| `ingest_pdf(path, theme="Generic", on_progress=None)` | `IngestResult` — the `done` payload plus `job_id`; `on_progress` receives each progress event |
| `graph()` · `communities(limit=20)` · `rebuild_communities(on_progress=None)` · `clear_graph()` | the backend JSON |

HTTP errors raise `SynapseError(status, detail)` carrying the backend's `detail`; an unreachable
backend raises its subclass `SynapseConnectionError`. A small `run(coro)` helper exists for
synchronous scripts. Exports: `SynapseClient`, `Answer`, `Retrieval`, `IngestResult`,
`SynapseError`, `__version__`.

---

## Environment variables

All optional; the CLI and the MCP server read the same ones.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SYNAPSE_URL` | `http://localhost:8000` | base URL of the Synapse backend |
| `SYNAPSE_API_KEY` | *(unset)* | if set, sent as `Authorization: Bearer …` on every request — for an authenticating proxy or gateway in front of the backend (the backend itself ignores it today) |
| `SYNAPSE_TIMEOUT` | `120` | HTTP timeout in seconds (ingestion and `ask` stream, so keep it generous) |
| `SYNAPSE_MAX_CONTEXT_CHARS` | `6000` | default `max_context_chars` for `synapse_retrieve` and `retrieve --budget`; `0` or negative disables the default budget; positive values below `200` are rejected (the backend's floor) |
| `SYNAPSE_CACHE_TTL` | `300` | retrieve-cache lifetime in seconds; `0` disables the cache |

---

## Troubleshooting

- **``Cannot reach Synapse at http://localhost:8000: … Is it running? (`make up`)``** — the backend is
  down or the URL is wrong. `curl http://localhost:8000/health` should return
  `{"status":"ok",…}`. Inside Docker the backend is `http://backend:8000` (compose network) or
  `http://host.docker.internal:8000` (backend on the host). If you changed `BACKEND_PORT` in
  `.env`, the MCP server and CLI need `SYNAPSE_URL=http://localhost:<BACKEND_PORT>` too (or
  `--url`, or `-e SYNAPSE_URL=…` to `install-config`) — every one-liner in this guide assumes
  port 8000.
- **``Synapse at … did not respond within 120s (raise SYNAPSE_TIMEOUT)``** — the backend is up but
  slow (ingestion and `synapse_ask` wait on an LLM): raise `SYNAPSE_TIMEOUT`. A stream that
  stops half-way is reported as ``closed the connection mid-stream``, never as a dead backend.
- **`synapse_ask` fails, `synapse_retrieve` works.** Expected on a keyless install: retrieval
  only needs embeddings (local `fastembed` by default), generation needs `LLM_PROVIDER`'s key on
  the backend. Either add a key to `.env` and restart, or keep using `synapse_retrieve` and let
  the host model answer — which is the recommended mode anyway. `synapse_status` shows the
  name of the configured `llm_provider` (readiness itself only covers Neo4j).
- **`synapse_ingest_pdf` fails the same way** — same cause: extraction is an LLM call.
- **`synapse_communities` returns nothing after `make demo`.** Expected: the demo seed writes
  entities and relationships but no communities (their titles and summaries are LLM-written), so
  `mode: "global"` routing has nothing to answer from. Ingest a document, or run
  `POST /api/communities/rebuild` (the UI's rebuild button) — both need an LLM key.
- **Retrieval returns nothing / obviously wrong neighbours after switching embedding providers.**
  `EMBEDDING_DIM` must match the model (fastembed 384, Gemini/Vertex 768, Titan/Cohere 1024,
  OpenAI `text-embedding-3-small` 1536) and every vector must come from the same model —
  re-ingest, or `make demo` to re-seed. See the README's embedding table.
- **The host says the server started but lists no tools.** With stdio, anything written to
  stdout breaks the protocol; `synapse-mcp` logs to stderr only, so this usually means a wrapper
  script printed something. Run `synapse-mcp --help` in a terminal to make sure the entry point
  resolves, and check the host's MCP log.
- **`uvx: command not found`** — install `uv`, or `pip install synapse-graphrag` and use
  `synapse-mcp` as the command.
- **Stale answers after ingesting** — the retrieve cache; see [Cache](#cache).
- **Port 8765 already in use** (HTTP transport) — `--port`, or `MCP_PORT` in `.env` for the
  compose profile.

---

## Security

- **The backend has no authentication.** Anyone who can reach `SYNAPSE_URL` can read the graph,
  ingest documents and — through `DELETE /api/graph` — wipe it. Keep it on `localhost`, inside a
  private network/VPN, or behind an authenticating reverse proxy. The same applies to the MCP
  server's HTTP transport: bind `127.0.0.1` unless a proxy is in front. See
  [DEPLOYMENT.md](../DEPLOYMENT.md#container-images-ghcr) for a proxy example.
- **`SYNAPSE_API_KEY`** is forwarded as `Authorization: Bearer …` on every backend request so
  that such a proxy or gateway can authenticate the MCP server. It is not checked by the backend
  itself today.
- **`synapse_clear_graph` requires `confirm=true`** and is annotated destructive; hosts that
  honour annotations will ask before calling it. `synapse_ingest_pdf` can read any `.pdf` the
  server process can read — run the server as the user whose files it should see, or in a
  container with an explicit mount.
- **Prompt-injected content.** Everything in `context`, `sources` and community summaries came
  from the ingested documents; treat it as data, not instructions, in your own agent code.
- Report vulnerabilities through [SECURITY.md](../SECURITY.md), not a public issue.
