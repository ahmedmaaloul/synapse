# Configuration

Reference for the main backend settings, the model providers and the pinned versions. The
[README](../README.md) covers the first run; every backend setting is documented in
[.env.example](../.env.example).

## Contents

1. [Chat model](#1-chat-model)
2. [Embedding model](#2-embedding-model)
3. [Lab reader](#3-lab-reader)
4. [Backend settings](#4-backend-settings)
5. [Pinned versions](#5-pinned-versions)

## 1. Chat model

The chat model serves chat answers, PDF extraction, theme summaries, the Navigator, generative
guidance and procedural evolution. Select it with `LLM_PROVIDER`; each provider then needs its own
settings. "JSON mode" states whether extraction, theme summaries and evolution request a JSON
response format; without it they rely on the prompt and tolerant parsing.

| Provider | `LLM_PROVIDER` | Required settings | Model setting and default | JSON mode | In the default image |
| --- | --- | --- | --- | --- | --- |
| Google Gemini | `gemini` | `GOOGLE_API_KEY` | `GEMINI_CHAT_MODEL=gemini-flash-latest` | yes | yes |
| Anthropic | `claude` | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL=claude-sonnet-5` | no | yes |
| OpenAI | `openai` | `OPENAI_API_KEY` | `OPENAI_CHAT_MODEL=gpt-4o-mini` | yes | yes |
| Azure OpenAI | `azure_openai` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT` | the deployment | yes | yes |
| Google Vertex AI | `vertex` | `VERTEX_PROJECT` and Application Default Credentials | `VERTEX_CHAT_MODEL=gemini-2.0-flash` | yes | no |
| AWS Bedrock | `bedrock` | `BEDROCK_REGION` (default `us-east-1`) and the AWS credential chain | `BEDROCK_CHAT_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0` | no | no |
| Groq | `groq` | `GROQ_API_KEY` | `GROQ_CHAT_MODEL=llama-3.3-70b-versatile` | yes | no |
| Mistral AI | `mistral` | `MISTRAL_API_KEY` | `MISTRAL_CHAT_MODEL=mistral-large-latest` | no | no |
| Ollama | `ollama` | a reachable Ollama server (`OLLAMA_BASE_URL`) with the model pulled | `OLLAMA_CHAT_MODEL=mistral` | yes | yes |
| OpenAI-compatible endpoint | `openai_compatible` | `OPENAI_COMPATIBLE_BASE_URL`; `OPENAI_COMPATIBLE_API_KEY` for hosted gateways | `OPENAI_COMPATIBLE_CHAT_MODEL=gpt-oss-120b` | yes | yes |

Procedure:

1. Edit `.env`. For Gemini, set `GOOGLE_API_KEY` only; `.env.example` already sets
   `LLM_PROVIDER=gemini`. The screenshots in the [README](../README.md) used:

   ```bash
   LLM_PROVIDER=openai
   OPENAI_API_KEY=<your key>
   OPENAI_CHAT_MODEL=gpt-5-nano
   OPENAI_REASONING_EFFORT=minimal
   ```

   OpenAI reasoning models (ids starting with `gpt-5` or `o1` to `o9`) receive
   `reasoning_effort` instead of a temperature. Its value is `OPENAI_REASONING_EFFORT` (default
   `minimal`); only the gpt-5 family accepts `minimal`, so o-series models receive `low` in its
   place.

2. Recreate the backend container. Compose reads `.env` when it creates a container, so
   `docker compose restart` keeps the old values.

   ```bash
   docker compose up -d --force-recreate backend
   ```

3. Check the active providers. The web UI does not display them.

   ```bash
   curl -s http://localhost:8000/api/about      # llm_provider, embedding_provider
   ```

Credentials are checked when a provider is first called, not at startup. A missing key surfaces
as an error in the UI or an HTTP 503 from the Navigator endpoint.

Provider notes:

- **Ollama.** Run `ollama pull mistral` on the host. The default `OLLAMA_BASE_URL` is
  `http://host.docker.internal:11434`; on Linux, `.env.example` recommends
  `http://172.17.0.1:11434`.
- **Vertex AI, Bedrock, Groq, Mistral, and the Cohere embedder** need the SDKs in
  [backend/requirements-providers.txt](../backend/requirements-providers.txt), which the default
  image does not install. For a local environment run `make providers`. For Docker, add one line
  to `backend/Dockerfile` directly after `COPY backend/ ./` (the file is not in the image before
  that line), then run `make up`:

  ```dockerfile
  COPY backend/ ./
  RUN pip install --no-cache-dir -r requirements-providers.txt
  ```

- **`openai_compatible`** is LangChain's `ChatOpenAI` pointed at a custom base URL.
  `.env.example` lists OpenRouter, Together, DeepSeek, Fireworks, vLLM, LM Studio and llama.cpp's
  server as targets; none of them is exercised by the test suite. The server must support
  `/v1/chat/completions` with streaming and, for ingestion, `response_format` `json_object`.
- **When no `LLM_PROVIDER` is set at all**, the code default is `ollama`. `.env.example`,
  `render.yaml` and `backend/fly.toml` set `gemini`.

After connecting a model, click the refresh icon (circular arrows, "Rebuild themes") in the Themes
panel to replace the derived theme titles with model-written ones, and upload PDFs from the
sidebar.

## 2. Embedding model

Embeddings are selected independently of the chat model, with `EMBEDDING_PROVIDER`.
`EMBEDDING_DIM` must match the model, because it sizes the Neo4j vector indexes.

| `EMBEDDING_PROVIDER` | Default model | `EMBEDDING_DIM` | Credentials | In the default image |
| --- | --- | --- | --- | --- |
| `fastembed` (default) | `BAAI/bge-small-en-v1.5` | 384 | none, runs locally | yes |
| `gemini` | `models/text-embedding-004` (see [known limitations](./known-limitations.md#6-providers)) | 768 | `GOOGLE_API_KEY` | yes |
| `ollama` | `nomic-embed-text` | 768 | none | yes |
| `openai` | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` | yes |
| `azure_openai` | the deployment's model | the deployment's model | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | yes |
| `vertex` | `text-embedding-005` | 768 | `VERTEX_PROJECT` and Application Default Credentials | no |
| `bedrock` | `amazon.titan-embed-text-v2:0` | 1024 | `BEDROCK_REGION` and the AWS credential chain | no |
| `cohere` | `embed-english-v3.0` | 1024 | `COHERE_API_KEY` | no |
| `fake` | token hash, not semantic; for tests and offline seeding | `EMBEDDING_DIM` | none | yes |

The dimensions are those documented in `.env.example`. The vector indexes are created once, with
`IF NOT EXISTS`, and never resized. To switch embedding model:

```bash
docker compose down -v      # deletes all Neo4j data, including procedural graph versions
# set EMBEDDING_PROVIDER, EMBEDDING_DIM and the credentials in .env
make up                     # the bundled procedural graphs are seeded again at startup
make demo                   # or re-ingest your documents
```

`PROCEDURAL_SEMANTIC_THRESHOLD=0.72` is calibrated for the fastembed model only. Re-measure it for
another embedder with the calibration script, which calls no LLM and no database:

```bash
docker compose exec -T backend python -m scripts.calibrate_semantic_threshold
```

## 3. Lab reader

The Synapse Lab reads contexts through the OpenAI API directly, whatever `LLM_PROVIDER` says.
Paid Lab runs (`realtime`, `batch`) need `OPENAI_API_KEY` in `.env`; `retrieve` runs and estimates
do not. The reader models on offer are `gpt-4o-mini`, `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`,
`gpt-4.1-nano`, `gpt-5-nano` (default) and `gpt-5-mini`.

## 4. Backend settings

Settings come from environment variables, which take precedence over `.env`. Docker Compose passes
the root `.env` to the backend container and sets `NEO4J_URI=bolt://neo4j:7687` for it. Every
backend setting is documented in [.env.example](../.env.example); Lab-only variables are in
[lab.md](./lab.md), client variables in [mcp.md](./mcp.md).
Provider settings are listed in [1. Chat model](#1-chat-model) and
[2. Embedding model](#2-embedding-model); Neo4j credentials and host ports in the
[README quick start](../README.md#2-quick-start).

| Setting | Default | Meaning |
| --- | --- | --- |
| `NEO4J_URI` | `bolt://neo4j:7687` in `.env.example` | Use `bolt://localhost:7687` for processes on the host |
| `MAX_CHUNKS` | 40 | Chunks kept per PDF (up to 1,000 characters each, ending on a sentence boundary where possible, with 200 characters of overlap) |
| `EXTRACTION_CONCURRENCY`, `EXTRACTION_TIMEOUT` | 5, 180 s | Parallel extraction calls; timeout per chunk |
| `EXTRACTION_TEMPERATURE`, `CHAT_TEMPERATURE` | 0.1, 0.3 | Sampling temperatures (not sent to OpenAI reasoning models) |
| `ENTITY_RESOLUTION_ENABLED` | true | Merge near-duplicate entities |
| `ENTITY_RESOLUTION_THRESHOLD`, `ENTITY_RESOLUTION_NAME_THRESHOLD` | 0.93, 0.87 | Embedding cosine and name similarity; both must pass, types must match |
| `ENTITY_RESOLUTION_CANDIDATE_K` | 25 | Vector-index neighbours checked per new entity; 0 scans up to 5,000 entities instead |
| `COMMUNITY_DETECTION_ENABLED` | true | Rebuild themes after each upload |
| `COMMUNITY_RESOLUTION`, `COMMUNITY_MIN_SIZE` | 1.0, 3 | Louvain resolution; smallest theme kept |
| `COMMUNITY_MAX_MEMBERS_IN_SUMMARY` | 30 | Members listed in each summary prompt |
| `RETRIEVAL_MAX_HOPS` | 2 (at most 4) | Neighbourhood depth for reasoning paths |
| `QUERY_ROUTING_ENABLED` | true | Route corpus-level questions to themes |
| `MAX_REASONING_PATHS` | 6 | Paths added to the context |
| `STORE_SOURCE_CHUNKS`, `CHUNK_RETRIEVAL_ENABLED` | true, true | Store passages at ingest; add excerpts to answers |
| `CHUNK_TOP_K`, `CHUNK_CONTEXT_MAX_CHARS` | 4 per channel, 4000 | Excerpts gathered; character cap on excerpts |
| `PROCEDURAL_ENABLED`, `PROCEDURAL_DEFAULT_GRAPH` | true, `graphrag-navigator` | Procedural memory; graph used by the Navigator |
| `PROCEDURAL_GUIDANCE_MODE` | `raw` | `none`, `raw` or `generative` |
| `PROCEDURAL_HOPS`, `PROCEDURAL_WINDOW` | 2, 3 | Guidance scope; recent steps shown to the guidance model |
| `PROCEDURAL_SEMANTIC_THRESHOLD` | 0.72 | Cosine floor for semantic localisation |
| `AGENT_MAX_STEPS`, `AGENT_OBSERVATION_MAX_CHARS` | 8, 1500 | Navigator step limit; tool output kept per step |
| `EVOLUTION_MAX_LLM_CALLS` | 400 | Hard call budget per evolution run |
| `EVOLUTION_DEFAULT_ROUNDS`, `EVOLUTION_DEFAULT_BATCH_SIZE` | 3, 10 | Evolution defaults |
| `CORS_ORIGINS` | `http://localhost:3000,http://frontend:3000` | Allowed browser origins |

## 5. Pinned versions

| Item | Version | Source |
| --- | --- | --- |
| Backend and MCP images | `python:3.12-slim` | `backend/Dockerfile`, `packages/synapse-graphrag/Dockerfile` |
| Frontend image | `node:20-alpine`, Next.js standalone output | `frontend/Dockerfile` |
| Neo4j | `neo4j:5-community` (floating tag) with APOC, no GDS | `docker-compose.yml` |
| Neo4j Python driver | 5.27.0 | `backend/requirements.txt` |
| FastAPI | 0.115.0 | `backend/requirements.txt` |
| LangChain | `langchain` 0.3.14, `langchain-core` 0.3.x (below 0.4) | `backend/requirements.txt` |
| networkx | 3.2 or later, below 4 | `backend/requirements.txt` |
| fastembed | 0.4 or later, below 0.8 | `backend/requirements.txt` |
| pypdf | 5.1.0 | `backend/requirements.txt` |
| Next.js, React | 16.1.6, 19.2.3 | `frontend/package.json` |

