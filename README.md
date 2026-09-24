<div align="center">

# Synapse 🧠

**Turn any document into a queryable knowledge graph — then chat with it using vector-grounded GraphRAG.**

*Ten AI providers. One env var. Zero API keys to try it.*

[![CI](https://github.com/ahmedmaaloul/synapse/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedmaaloul/synapse/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ahmedmaaloul/synapse?include_prereleases)](https://github.com/ahmedmaaloul/synapse/releases)
<!-- Enable after the first PyPI publish (vars.PYPI_PUBLISH=true on the release workflow):
[![PyPI](https://img.shields.io/pypi/v/synapse-graphrag)](https://pypi.org/project/synapse-graphrag/)
-->
[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-blue.svg)](./LICENSE)
[![Client: Apache 2.0](https://img.shields.io/badge/Client-Apache_2.0-green.svg)](./packages/synapse-graphrag/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)
[![Stars](https://img.shields.io/github/stars/ahmedmaaloul/synapse?style=flat&logo=github)](https://github.com/ahmedmaaloul/synapse/stargazers)
[![Forks](https://img.shields.io/github/forks/ahmedmaaloul/synapse?style=flat&logo=github)](https://github.com/ahmedmaaloul/synapse/network/members)

![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Next.js 16](https://img.shields.io/badge/Next.js-16-000000?logo=next.js)
![Neo4j 5](https://img.shields.io/badge/Neo4j-5-008CC1?logo=neo4j&logoColor=white)
[![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in-Codespaces-181717?logo=github)](https://codespaces.new/ahmedmaaloul/synapse)

Upload a PDF → Synapse extracts entities & relationships with an LLM, embeds them, and writes a **Neo4j property graph**. Ask a question → it runs **hybrid (vector + full-text) retrieval**, expands the subgraph, and streams a grounded answer with **clickable citations** back to the exact nodes it used.

![Synapse — GraphRAG knowledge explorer](./docs/synapse-dashboard.png)

</div>

---

## ⚡ Try it in 60 seconds — no API key

A fresh clone ships a **hand-curated demo knowledge graph** (53 entities, 106 relationships: a brief history of AI, from Babbage's Analytical Engine to GraphRAG). Seeding it skips the only step that needs an LLM — extraction — while still running the project's *real* schema, embedding and write pipeline.

**Prerequisite:** [Docker Desktop](https://www.docker.com/products/docker-desktop/). Nothing else.

```bash
git clone https://github.com/ahmedmaaloul/synapse.git
cd synapse
cp .env.example .env      # no edits needed for the demo
make up                   # neo4j + backend + frontend
make demo                 # seed the graph — zero keys, zero signups
```

Open **<http://localhost:3000>** and you have a live, explorable knowledge graph.

| Service | URL |
| --- | --- |
| Frontend UI | <http://localhost:3000> |
| Backend API (Swagger) | <http://localhost:8000/docs> |
| Neo4j Browser | <http://localhost:7474> · `neo4j` / `synapse_secret` |

> **What works with no key:** the graph view, node inspector, hybrid vector + full-text retrieval, embeddings (local `fastembed`), the procedural-graph view and raw procedural guidance, and the whole test suite.
> **What needs a key:** *generating* chat answers, ingesting your own PDFs and running the Navigator agent — all of them call an LLM. Grab a **free** [Google AI Studio](https://aistudio.google.com/apikey) or [Groq](https://console.groq.com/keys) key, drop it in `.env`, restart, and you're done. Or run fully offline with [Ollama](#-provider-matrix).

<details>
<summary>Prefer not to install Docker Desktop? Other ways to run it</summary>

```bash
make demo-local            # seed from a local venv (needs: docker compose up -d neo4j)
python -m scripts.seed_demo --validate-only   # from backend/ — checks the fixture, touches nothing
EMBEDDING_PROVIDER=fake python -m scripts.seed_demo --clear   # 100% offline, no model download
```

Or click **[Open in GitHub Codespaces](https://codespaces.new/ahmedmaaloul/synapse)** — the
[devcontainer](./.devcontainer/devcontainer.json) installs Python 3.12, Node 20 and every backend
dependency, and forwards ports 3000/8000/8765/7474/7687 for you.

</details>

---

## 🤖 Use it from Claude, Cursor & any MCP client

Synapse ships an **[MCP](https://modelcontextprotocol.io) server** — the `synapse-graphrag` package, published to PyPI by the release workflow — so the graph you just built is a tool any agent host can call. It is a thin client over the same HTTP API the UI uses: keep the stack running (`make up`, `make demo`) and register it.

**Claude Code** — one line:

```bash
claude mcp add synapse -e SYNAPSE_URL=http://localhost:8000 -- uvx synapse-graphrag mcp
```

> Until the first PyPI release is out, run it straight from the repo instead of `uvx synapse-graphrag mcp`:
> `uvx --from "git+https://github.com/ahmedmaaloul/synapse#subdirectory=packages/synapse-graphrag" synapse-graphrag mcp`
> — or `pip install ./packages/synapse-graphrag` from a checkout and use `synapse-mcp` as the command.

**Claude Desktop / Cursor** — add to `claude_desktop_config.json` or `.cursor/mcp.json`:

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

Then ask your assistant *"What does the Synapse graph say about the Analytical Engine?"* — it calls `synapse_retrieve`, reads the subgraph and answers with entity citations. `synapse-graphrag install-config --client cursor` (or `claude-code`, `claude-desktop`, `vscode`, `windsurf`) prints the exact snippet for each client.

| Tool | What it does | Writes? |
| --- | --- | :--: |
| `synapse_retrieve` | Budgeted GraphRAG context — subgraph, reasoning paths, source chunks — with **no LLM call** on the Synapse side | |
| `synapse_ask` | A full answer generated by *Synapse's* LLM (a second LLM bill — prefer `synapse_retrieve`) | |
| `synapse_ingest_pdf` | Ingest a local PDF into the graph, following the job to completion | ✅ |
| `synapse_communities` | Corpus-level themes: Louvain communities with their LLM summaries | |
| `synapse_find_entities` | Case-insensitive substring search over entity labels and types | |
| `synapse_graph_stats` | Node/edge counts, by entity type and relationship type | |
| `synapse_status` | Health, readiness, version and the active providers | |
| `synapse_clear_graph` | Wipe the knowledge graph (procedural graphs are kept) — refuses unless called with `confirm=true` | ⚠️ |
| `synapse_procedures` | The [procedural graphs](#-procedural-memory-agents-that-learn-how-to-use-the-graph) the backend keeps, with version and score | |
| `synapse_procedure_guidance` | Call before each step of a multi-step task: the procedural subgraph around your last action (default graph `mcp-host`, built on these tools) — `raw` mode makes **no LLM call** | |
| `synapse_record_trajectory` | Record a finished run (its steps and an honest score in [0, 1]) against a procedural graph | ✅ |
| `synapse_agent_ask` | The backend's GraphRAG Navigator agent answers by walking the graph step by step (backend LLM calls — one per step) | |

### 💸 FinOps: budgeted context, not a second LLM bill

A tool that *answers* the question makes the host model pay to read an answer another model already paid to write. Synapse's default tool is **retrieval-only**: `synapse_retrieve` returns the ranked subgraph, reasoning paths and source excerpts and lets the host model do the one generation it was going to do anyway. Every call is budgeted — `max_context_chars` (default `SYNAPSE_MAX_CONTEXT_CHARS=6000`) truncates at the last line boundary within the budget (a hard cut only if that boundary would waste more than 40 % of it) and marks the cut — and every response carries a `usage` block (`context_chars`, `context_tokens_est`, `truncated`, `cached`, …) so an agent, or a bill, can see exactly what was consumed. Repeated queries are served from a TTL cache (`SYNAPSE_CACHE_TTL`, 300 s by default) without touching the backend. The same knob is on the CLI: `synapse-graphrag retrieve "…" --budget 2000 --json`.

**Over HTTP instead of stdio** (shared servers, Docker, remote hosts):

```bash
docker compose --profile mcp up -d      # → http://localhost:8765/mcp  (streamable HTTP)
```

Full guide — per-client setup, transports, tool reference, cost model, troubleshooting and security: **[docs/mcp.md](./docs/mcp.md)**.

Working on AI safety? Ingest with `--theme "AI Safety"` (risks, failure modes, mitigations, evaluations, incidents, policies — each risk labelled *demonstrated* or *hypothesised*) and use the `safety_brief` prompt: **[docs/ai-safety.md](./docs/ai-safety.md)**. The full cost model behind the budget knobs: **[docs/finops.md](./docs/finops.md)**.

---

## 🧭 Procedural memory: agents that learn *how* to use the graph

The knowledge graph is *semantic* memory: what your documents say. It says nothing about *how* to
use it: which lookup comes first, when a bridge entity has been found, when the evidence is enough
to answer. Synapse adds **procedural memory**, an implementation of *Procedural Graphs* (Lu, Chen,
Wu, Arık — [arXiv:2609.09153](https://arxiv.org/abs/2609.09153)).

A procedural graph is a small directed graph of tool actions, reasoning steps and statuses. Every
transition carries a **condition**, **guidance** and **pitfalls**. It lives in the same Neo4j as the
entities it helps navigate, and every change is versioned and can be rolled back.

- **Online.** An agent's last action places it on a node, and it is shown the transitions up to two
  hops ahead. By default the guidance is that **raw subgraph: zero extra LLM calls**. The paper
  instead has a guidance LLM rewrite it at every step. Its localized generative guidance cost
  +33 % to +55 % total tokens over no graph (GDPval, ALFWorld). Raw guidance is Synapse's own
  choice, and a hypothesis to measure rather than a result.
- **Offline.** The paper's self-evolution loop refines the graph from scored question/answer
  pairs: roll out, let an LLM propose edits, and keep a candidate only if the validation score does
  not drop. It is a search, not a guaranteed gain: in the paper's HotpotQA study, evolving the
  expert graph (what `evolve --mode static` does) scored 76.34 F1 at 10,658 tokens per question,
  *below* the unevolved expert graph (76.61 at 9,046). Of the modes Synapse implements, only
  evolution from scratch (`--mode scratch`) beat it (78.79).

```mermaid
flowchart LR
    Q[💬 Question] --> S{{Navigator step<br/>Thought → Action}}
    S -->|last action| L[Localize on the<br/>procedural graph · 2 hops]
    PG[(Procedural graph<br/>conditions · guidance · pitfalls)] --> L
    L -->|guidance · raw = no LLM call| S
    S -->|deterministic tool| KG[(Knowledge graph<br/>entities · relations · passages)]
    KG -->|observation| S
    S --> A[💡 Answer + step trace]
    QA[📋 QA pairs] --> EV[Self-evolution<br/>rollouts → refiner → validation gate]
    EV -->|new version if the score holds| PG
```

The built-in consumer is the **GraphRAG Navigator**, a ReAct agent that answers by walking
Synapse's own graph with six deterministic tools, steered by the `graphrag-navigator` graph (in
the UI: *Chat | Navigator*, and *Knowledge | Procedures* to see the graph). MCP hosts get
guidance for their own work through `synapse_procedure_guidance` and the `follow_procedure`
prompt, on a second bundled graph, `mcp-host`, whose steps are the real `synapse_*` tool names:
a host's own calls place it on the graph by exact name.

```bash
synapse-graphrag procedures show graphrag-navigator    # the bundled expert strategy, transition by transition
synapse-graphrag agent "Who designed the machine that Ada Lovelace wrote a program for?"   # step trace + usage
# evolve on the bundled demo QA set: a cap under one round, (9+6+9)×8 + 1 = 193 calls, is refused; prints the bound, asks first
synapse-graphrag evolve graphrag-navigator \
    --train backend/benchmarks/procedural/demo_qa.json --train-split train \
    --val   backend/benchmarks/procedural/demo_qa.json --val-split val \
    --rounds 1 --batch-size 6 --max-llm-calls 200
```

`agent` and `evolve` need an LLM key on the backend. The `procedures …` commands make no LLM call,
except `procedures guide --mode generative`. The demo graph has no source chunks, so on it the
Navigator's `read_sources` and `search_passages` return nothing. We have
**not** reproduced the paper's numbers: a [cost-capped harness](./backend/benchmarks/procedural/README.md)
compares systems on Synapse's own Navigator (no graph vs raw vs generative guidance), which is
not a reproduction of the paper's tables. The data model, the localization cascade, Algorithm 1 as implemented, the
API / MCP / CLI reference, costs and limitations are in
**[docs/procedural-graphs.md](./docs/procedural-graphs.md)**.

---

## ✨ Why it's interesting

- **Real GraphRAG, not keyword lookup.** Retrieval seeds from a **Neo4j vector index** over entity embeddings *and* a full-text index, then expands each seed to its 1-hop neighborhood so the model reasons over *relationships*, not isolated facts.
- **Measured, not vibes.** A [retrieval eval harness](#-retrieval-evaluation) scores the pipeline — **Hit@1 88% · Recall@8 100% · MRR 0.92** on paraphrased queries that keyword search would miss.
- **Genuinely pluggable AI.** **Ten chat providers and nine embedding providers** behind one small factory, chosen by a single env var — from OpenAI and Bedrock to a laptop running Ollama. No code change, no rebuild.
- **Streamed everything.** Ingestion progress and chat answers both stream over **Server-Sent Events**; answers arrive token-by-token with the grounding entities highlighted live in the graph.
- **Callable from any agent.** An MCP server, CLI and Python client ([`synapse-graphrag`](./docs/mcp.md)) expose the graph to Claude, Cursor, VS Code and friends — and the retrieval-only `synapse_retrieve` tool returns **budgeted** context with `usage` metadata, so a host that already has an LLM never pays for a second generation.
- **Procedural memory, not just facts.** [Procedural Graphs](./docs/procedural-graphs.md) (arXiv:2609.09153) store *how* to navigate the graph next to the graph itself: step-local guidance with **zero extra LLM calls** by default, a ReAct navigator agent to use it, and the paper's self-evolution loop with versioned, rollback-able history — plus a cost-capped harness to measure whether it helps, instead of claiming it does.
- **Tested & CI'd.** 1,300+ hermetic unit tests plus integration tests against a **real Neo4j service container** in GitHub Actions, and frontend typecheck/lint/build on every push.

---

## 🧠 How it works

```mermaid
flowchart LR
    PDF[📄 PDF] --> P[Parse + chunk<br/>sentence-aware]
    P --> LLM[LLM extraction<br/>any of 10 providers]
    LLM --> D[Dedupe + canonicalize]
    D --> E[Embed entities]
    E --> W[(Neo4j<br/>+ vector index)]

    Q[💬 Question] --> H{Hybrid retrieval}
    W --> H
    H -->|vector seeds| X[Expand 1-hop<br/>neighborhood]
    H -->|full-text seeds| X
    X --> G[LLM generation<br/>streamed + cited]
    G --> A[💡 Grounded answer]
    X --> R["🔌 /api/retrieve<br/>budgeted context · no LLM"]
    R --> M[🤖 MCP host · CLI · SDK]
```

Ingestion runs as a background job that streams progress; retrieval interleaves semantic and lexical seeds, preserves the vector ranking, and returns the entities that grounded the answer as citations. The retrieval step is also exposed on its own as **`POST /api/retrieve`** — context, citations, reasoning paths and source chunks under a `max_context_chars` budget, with `usage` metadata and no LLM call — which is what the MCP server and CLI build on. A second, step-by-step path — the GraphRAG Navigator agent steered by a procedural graph (`POST /api/agent/ask`) — is described [above](#-procedural-memory-agents-that-learn-how-to-use-the-graph). See **[ARCHITECTURE.md](./ARCHITECTURE.md)** for the full design.

---

## 🔌 Provider matrix

Set **one** env var. Everything else has a working default — see [`.env.example`](./.env.example) for the per-provider details, and [`backend/app/services/llm_provider.py`](./backend/app/services/llm_provider.py) for the factory itself.

| Provider | `LLM_PROVIDER=` | Credential needed | Free tier? | Install |
| --- | --- | --- | :--: | --- |
| **Google Gemini** *(easiest)* | `gemini` | `GOOGLE_API_KEY` — [AI Studio](https://aistudio.google.com/apikey) | ✅ | included |
| **Anthropic Claude** | `claude` | `ANTHROPIC_API_KEY` — [console](https://console.anthropic.com/) | ❌ paid credits | included |
| **OpenAI** | `openai` | `OPENAI_API_KEY` — [platform](https://platform.openai.com/api-keys) | ❌ paid credits | included |
| **Azure OpenAI** | `azure_openai` | `AZURE_OPENAI_API_KEY` + `_ENDPOINT` + `_CHAT_DEPLOYMENT` | ❌ Azure subscription | included |
| **Google Vertex AI** | `vertex` | `VERTEX_PROJECT` + [ADC](https://cloud.google.com/docs/authentication/application-default-credentials) (no key) | ⚠️ GCP trial credits | `make providers` |
| **AWS Bedrock** | `bedrock` | `BEDROCK_REGION` + the standard AWS credential chain | ❌ pay per token | `make providers` |
| **Groq** *(fastest)* | `groq` | `GROQ_API_KEY` — [console](https://console.groq.com/keys) | ✅ rate-limited | `make providers` |
| **Mistral AI** | `mistral` | `MISTRAL_API_KEY` — [console](https://console.mistral.ai/api-keys/) | ⚠️ free experiment tier | `make providers` |
| **Ollama** *(local & private)* | `ollama` | none — `ollama pull mistral` on the host | ✅ free forever | included |
| **Any OpenAI-compatible API** | `openai_compatible` | `OPENAI_COMPATIBLE_BASE_URL` (+ key for hosted gateways) | depends | included |

> **`openai_compatible` is the escape hatch.** It is `ChatOpenAI` pointed at a custom `base_url`, so it already covers **OpenRouter**, **Together**, **DeepSeek**, **Fireworks**, **vLLM**, **LM Studio** and **llama.cpp's server** — with *zero* extra dependencies. If your provider speaks `/v1/chat/completions`, it works today.

`make providers` runs `pip install -r backend/requirements-providers.txt` — the heavier first-party cloud SDKs (`langchain-google-vertexai`, `langchain-aws`, `langchain-groq`, `langchain-mistralai`, `langchain-cohere`) that are kept out of the default image so it stays small.

### Embedding providers

Embeddings are chosen independently of the chat model. The default needs **no API key and no GPU**.

| `EMBEDDING_PROVIDER=` | Default model | Dims → `EMBEDDING_DIM` | Credential | Install |
| --- | --- | :--: | --- | --- |
| `fastembed` *(default)* | `BAAI/bge-small-en-v1.5` | **384** | none — runs locally | included |
| `gemini` | `models/text-embedding-004` | 768 | `GOOGLE_API_KEY` | included |
| `ollama` | `nomic-embed-text` | 768 | none — local | included |
| `openai` | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` | included |
| `azure_openai` | your deployment | match your model | `AZURE_OPENAI_*` + `_EMBEDDING_DEPLOYMENT` | included |
| `vertex` | `text-embedding-005` | 768 | `VERTEX_PROJECT` + ADC | `make providers` |
| `bedrock` | `amazon.titan-embed-text-v2:0` | 1024 | `BEDROCK_REGION` + AWS chain | `make providers` |
| `cohere` | `embed-english-v3.0` | 1024 | `COHERE_API_KEY` | `make providers` |
| `fake` | deterministic hash | `EMBEDDING_DIM` | none — offline dev & tests | included |

> ⚠️ **`EMBEDDING_DIM` must match the model** — it sizes the Neo4j vector index. Switching embedding providers means re-ingesting (or re-running `make demo`) so all vectors share one space.

---

## 📊 Retrieval evaluation

Shipping RAG without measuring retrieval is flying blind. `backend/eval/` seeds a fixture graph and scores how well retrieval surfaces the *right* entities for **paraphrased** questions (deliberately no lexical overlap, so keyword-only search fails).

```bash
docker compose up -d neo4j
make eval        # writes backend/eval/results.md
```

| Metric | Score |
| --- | --- |
| **Hit@1** | 88% |
| **Recall@8** | 100% |
| **Precision@8** | 17%* |
| **MRR** | 0.917 |

<sub>*Precision@8 is low by construction — most queries have only 1–2 relevant entities, so returning 8 candidates for graph highlighting caps precision. Hit@1 / MRR are the quality signal.</sub>

---

## 🍴 Why fork this?

Because most GraphRAG repos are notebooks. This one is a running product with the boring parts already solved.

- **Swap the whole AI layer with one env var.** Ten chat backends and nine embedding backends behind [one small factory](./backend/app/services/llm_provider.py). Benchmark Gemini vs. Groq vs. your own vLLM box without touching application code.
- **A real vector GraphRAG reference implementation.** Neo4j native vector *and* full-text indexes, interleaved seeding, 1-hop expansion, streamed citations that map back to graph nodes. Not a `similarity_search()` wrapper.
- **A test suite and CI you can build on.** 1,300+ hermetic backend tests (no network, no DB, no LLM), integration tests against a live Neo4j service container, ruff + eslint + tsc, and all three Docker image builds — all green on every push.
- **Docs that respect your time.** [ARCHITECTURE.md](./ARCHITECTURE.md) explains *why*, [DEPLOYMENT.md](./DEPLOYMENT.md) gets it online, [CONTRIBUTING.md](./CONTRIBUTING.md) walks you through your first PR, and every env var is documented in [`.env.example`](./.env.example).
- **A clean seam to extend.** Provider branches are lazily imported and validate credentials *before* touching an SDK — which is why a new provider is a self-contained ~20-line change plus a test.

### Add a provider — the best first PR

Adding an AI provider is small, self-contained, and immediately useful to someone else. There's a
**[step-by-step walkthrough with real function names](./CONTRIBUTING.md#add-a-new-ai-provider-in-20-lines)**
in CONTRIBUTING.md.

👉 **[Open a provider request](https://github.com/ahmedmaaloul/synapse/issues/new?template=provider_request.yml)** — whether you want to build it or just want it to exist. Together AI, Nvidia NIM, Hugging Face TGI, Cerebras, xAI, Perplexity, watsonx… all fair game.

Other good entry points: the [roadmap](#-roadmap) below, anything labelled [`good first issue`](https://github.com/ahmedmaaloul/synapse/labels/good%20first%20issue), and [bug reports](https://github.com/ahmedmaaloul/synapse/issues/new?template=bug_report.yml).

---

## 🧪 Testing & CI

```bash
make test        # backend unit tests — hermetic (no DB / network / LLM)
make test-int    # integration tests against a live Neo4j
make eval        # retrieval quality harness
make lint        # ruff + eslint + tsc
make fmt         # ruff --fix + ruff format
make mcp-test    # synapse-graphrag package: ruff + pytest
```

Every push runs [CI](./.github/workflows/ci.yml): backend lint + unit tests, **integration tests against a real Neo4j 5 service container**, frontend eslint/typecheck/build, lint/tests/build of the `synapse-graphrag` package, and all three Docker image builds. Pushing a `v*` tag runs the [release workflow](./.github/workflows/release.yml): version and changelog checks, GHCR images, a GitHub Release and (opt-in) PyPI publishing — see [CONTRIBUTING.md](./CONTRIBUTING.md#cutting-a-release).

---

## 🛠️ Tech stack

| Layer | Tech |
| --- | --- |
| **Frontend** | Next.js 16, React 19, TailwindCSS 4, `react-force-graph`, `react-markdown` |
| **Backend** | FastAPI, LangChain, async Neo4j driver, `pypdf`, SSE |
| **AI** | 10 pluggable chat providers · 9 pluggable embedding providers · local `fastembed` default |
| **Database** | Neo4j 5 (Bolt + APOC + native vector & full-text indexes) |
| **Agents** | MCP server (stdio + streamable HTTP), CLI and async Python client — `synapse-graphrag` · GraphRAG Navigator (ReAct) steered by Procedural Graphs |
| **Infra** | Docker Compose, GitHub Actions (CI + tag-driven releases to GHCR / PyPI), Codespaces devcontainer |

---

## 📁 Project structure

```text
synapse/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app: CORS, lifespan, health/readiness, /api/about
│   │   ├── config.py               # typed settings (every provider, one Literal)
│   │   ├── neo4j_driver.py         # async driver + connectivity check + one-transaction write batches
│   │   ├── routers/                # upload (SSE jobs) · chat (SSE) + retrieve (JSON) · graph + communities · procedures + agent
│   │   ├── data/procedural/        # bundled expert priors, seeded at startup — graphrag-navigator.json (the Navigator) · mcp-host.json (MCP hosts)
│   │   └── services/
│   │       ├── llm_provider.py     # ⭐ the pluggable chat + embeddings factory
│   │       ├── graph_builder.py    # extract → dedupe → embed → write
│   │       ├── graph_schema.py     # vector + full-text index bootstrap (+ procedural constraints)
│   │       ├── chat_engine.py      # hybrid GraphRAG retrieval, multi-hop paths, budgeting, streaming
│   │       ├── entity_resolution.py# embedding + fuzzy-name duplicate merging
│   │       ├── communities.py      # Louvain communities + LLM summaries (global search)
│   │       ├── chunk_store.py      # source chunks (text units) linked to entities
│   │       ├── procedural_graph.py # 🧭 Procedural Graphs: the pure data structure, edits, validation, serializers
│   │       ├── procedural_store.py # procedural memory in Neo4j: versions, rollback, rejections, trajectories
│   │       ├── procedural_guidance.py # step-local guidance: localization cascade, raw / generative modes
│   │       ├── graph_agent.py      # the GraphRAG Navigator: a ReAct agent over deterministic graph tools
│   │       ├── procedural_evolution.py # offline self-evolution (the paper's Algorithm 1), LLM-call budget
│   │       ├── qa_metrics.py       # SQuAD / HotpotQA EM + F1
│   │       ├── pdf_parser.py       # sentence-aware chunking
│   │       └── jobs.py             # in-memory SSE job bus
│   ├── scripts/seed_demo.py        # zero-API-key demo graph seeder (`make demo`)
│   ├── tests/                      # hermetic unit tests + integration tests
│   ├── eval/                       # retrieval eval harness + results
│   ├── benchmarks/                 # GraphRAG vs vector RAG harness · public/ HotpotQA & 2Wiki loaders · procedural/ Procedural Graphs harness
│   ├── requirements.txt            # batteries-included providers
│   └── requirements-providers.txt  # opt-in cloud SDKs (`make providers`)
├── frontend/
│   └── src/app/
│       ├── lib/                    # typed API client, types, constants
│       └── components/             # GraphPanel · ProceduralPanel · ChatPanel (Chat | Navigator) · FileUpload · Inspector · ThemesPanel
├── packages/
│   └── synapse-graphrag/           # 🤖 MCP server · CLI · async Python client (PyPI: synapse-graphrag)
├── scripts/                        # release checks (versions, changelog) + branch-protection ruleset
├── docs/mcp.md                     # MCP / CLI / SDK guide
├── docs/procedural-graphs.md       # procedural memory: guidance, the Navigator, self-evolution, limits
├── docs/finops.md                  # cost model: where a GraphRAG dollar goes, and the knob for each
├── docs/ai-safety.md               # the `AI Safety` theme and `safety_brief` prompt
├── .devcontainer/                  # one-click Codespaces environment
├── .github/
│   └── workflows/                  # ci.yml · release.yml (tag → GHCR images, GitHub Release, PyPI)
├── docker-compose.yml              # neo4j + backend + frontend (+ `--profile mcp`)
└── ARCHITECTURE.md · DEPLOYMENT.md · CONTRIBUTING.md · CHANGELOG.md
```

---

## 🤝 Contributing

PRs are genuinely welcome — and the project is structured so a first contribution is easy to land.

```bash
make providers   # optional: the extra provider SDKs
make test        # what CI runs
make lint
```

Start with **[CONTRIBUTING.md](./CONTRIBUTING.md)** (dev setup, conventions, and the
[add-a-provider walkthrough](./CONTRIBUTING.md#add-a-new-ai-provider-in-20-lines)), and read the
[Code of Conduct](./CODE_OF_CONDUCT.md). Security issues go to [SECURITY.md](./SECURITY.md).

**Zero-setup contributing:** [![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in-Codespaces-181717?logo=github)](https://codespaces.new/ahmedmaaloul/synapse) — Python 3.12, Node 20, all dependencies, ports forwarded, `.env` pre-created. Nothing to install locally.

---

## 🗺️ Roadmap

- [x] Multi-hop retrieval (2+ hop reasoning paths)
- [x] Entity resolution with embedding-similarity merging
- [x] MCP server, CLI & Python client (`synapse-graphrag`)
- [x] `AI Safety` extraction theme + `safety_brief` MCP prompt
- [x] Procedural memory — Procedural Graphs: step-local guidance, the GraphRAG Navigator agent, self-evolution with version history & rollback
- [ ] Measure procedural guidance on HotpotQA with the navigator (tokens per correct answer) — a comparison between systems, not a reproduction of the paper's tables
- [ ] Learn from recorded agent trajectories (today they are stored, but evolution learns only from its own rollouts)
- [ ] Publish to the MCP Registry & PyPI (trusted publishing)
- [ ] Per-answer cost accounting across providers (FinOps)
- [ ] Per-document management (list / delete individual sources)
- [ ] Ingest `.docx` / `.md` / raw text and URLs
- [ ] Hosted live demo
- [ ] More providers — [request one](https://github.com/ahmedmaaloul/synapse/issues/new?template=provider_request.yml)

---

## 📄 Licensing

Synapse is **source-available**: free for noncommercial use, licensed for commercial use — and the
client package is Apache-2.0.

### ✅ Free for noncommercial use under [PolyForm Noncommercial 1.0.0](./LICENSE) — no permission, no cost, no registration

Personal projects, research, study, teaching, hobby and amateur pursuits, and use by universities,
public research organisations, charities and the other nonprofits the license names (public-safety,
health and environmental organisations), and government institutions. Use it, **fork it**, study it,
modify it, self-host it and contribute back. There is no copyleft, so you may keep your changes
private. What it asks in return: keep the license and the attribution — the `Required Notice:` line
at the top of [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE) — with every copy.

### 💼 A commercial license is needed if…

…you use it **at or for a business** — in production or internally, on-prem or as SaaS — or **ship it
inside a product**. Under this license, use by or for a company is commercial even when it is purely
internal and nothing is ever redistributed: "we only run it on our own servers" still needs a license.
Attribution is required either way (see [`NOTICE`](./NOTICE)).

### 🤖 The client package is Apache-2.0

[`packages/synapse-graphrag/`](./packages/synapse-graphrag/) — the MCP server, CLI and Python SDK — is
licensed under [Apache-2.0](./packages/synapse-graphrag/LICENSE) so agents and products can talk to
Synapse with no obligations beyond Apache's. The backend it talks to is what the two paragraphs above
are about.

**Only Ahmed Maaloul can grant a commercial license.** Email **<ahmed.maaloul@proton.me>** with the
subject `[Commercial License] <your company>`. Startups and academic spin-outs: say so, pricing is
flexible, and evaluation licenses are available on request. Full details and FAQ:
**[COMMERCIAL-LICENSE.md](./COMMERCIAL-LICENSE.md)**. Versions released before this change stay as
published — MIT up to commit `91ee2f2`, AGPL-3.0-or-later for 0.3.0 and 0.4.0.

<sub>Plain-English summary, not legal advice. [`LICENSE`](./LICENSE) is the binding document.</sub>

---

<div align="center">

**Synapse** — created and maintained by **[Ahmed Maaloul](https://github.com/ahmedmaaloul)**
&lt;ahmed.maaloul@proton.me&gt;

Copyright © 2026 Ahmed Maaloul · SPDX-License-Identifier: `PolyForm-Noncommercial-1.0.0` (core) · client `Apache-2.0`
· <https://github.com/ahmedmaaloul/synapse>

If this saved you time, a ⭐ helps other people find it.

</div>
