# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Configuration

Central, typed settings loaded from environment / .env via pydantic-settings.

The AI backend is *pluggable*: the same code path runs against any major cloud
model (OpenAI, Azure OpenAI, Google Gemini / Vertex AI, Anthropic Claude, AWS
Bedrock, Groq, Mistral), any OpenAI-compatible endpoint, or a fully local model
(Ollama). Pick the provider with ``LLM_PROVIDER`` and, for embeddings,
``EMBEDDING_PROVIDER``. Every provider-specific value has a sensible default so
the app boots even with an empty .env — it will only fail (with a clear,
actionable message) when a provider that needs a key is actually invoked.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal[
    "gemini",
    "claude",
    "ollama",
    "openai",
    "azure_openai",
    "vertex",
    "bedrock",
    "groq",
    "mistral",
    "openai_compatible",
]
EmbeddingProvider = Literal[
    "gemini",
    "ollama",
    "fastembed",
    "fake",
    "openai",
    "azure_openai",
    "vertex",
    "bedrock",
    "cohere",
]
# How procedural-graph guidance reaches the agent: not at all, as the raw
# serialized subgraph (zero extra LLM calls), or rewritten by a guidance LLM.
GuidanceMode = Literal["none", "raw", "generative"]


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── App ──────────────────────────────────────────────
    app_name: str = "Synapse"
    debug: bool = True
    # Comma-separated list of allowed CORS origins.
    cors_origins: str = "http://localhost:3000,http://frontend:3000"

    # ── Neo4j ────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "synapse_secret"

    # ── LLM provider selection ───────────────────────────
    # Which chat model backend to use for extraction + chat.
    llm_provider: LLMProvider = "ollama"

    # Google Gemini (cloud). Free tier: https://aistudio.google.com/apikey
    # "gemini-flash-latest" tracks the current free-tier flash model; some pinned
    # models (e.g. gemini-2.0-flash) may have a 0 free-tier quota on a given key.
    google_api_key: str = ""
    gemini_chat_model: str = "gemini-flash-latest"

    # Anthropic Claude (cloud). https://console.anthropic.com/
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    # Ollama (local). Requires the Ollama app running on the host.
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_chat_model: str = "mistral"

    # OpenAI (cloud). https://platform.openai.com/api-keys
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"  # 1536 dims
    # Reasoning effort sent to OpenAI REASONING models — a model id matching
    # ^(gpt-5|o[1-9]), e.g. gpt-5-nano, o3-mini — on the openai, azure_openai
    # (matched on the deployment name) and openai_compatible providers. Those
    # models reject a non-default temperature, so none is sent to them, and
    # they bill hidden reasoning tokens as output: "minimal" keeps that bill
    # lowest. One of minimal | low | medium | high, passed through as-is (the
    # API validates it); "minimal" exists only on the gpt-5 family, so o-series
    # models get "low" instead. Ignored for every other model.
    openai_reasoning_effort: str = "minimal"

    # Azure OpenAI (cloud). Portal → your resource → "Keys and Endpoint".
    # NOTE: Azure addresses *deployments*, not model names — set the deployment
    # you created for chat and (if used) for embeddings.
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""  # https://<resource>.openai.azure.com/
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""

    # Google Vertex AI (cloud, GCP). Credentials come from Application Default
    # Credentials: `gcloud auth application-default login` or a service account.
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    vertex_chat_model: str = "gemini-2.0-flash"
    vertex_embedding_model: str = "text-embedding-005"  # 768 dims

    # AWS Bedrock (cloud). Credentials come from the standard AWS chain
    # (AWS_ACCESS_KEY_ID / AWS_PROFILE / EC2-ECS instance role).
    bedrock_region: str = "us-east-1"
    bedrock_chat_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_embedding_model: str = "amazon.titan-embed-text-v2:0"  # 1024 dims

    # Groq (cloud, very fast open-weights inference). https://console.groq.com/keys
    groq_api_key: str = ""
    groq_chat_model: str = "llama-3.3-70b-versatile"

    # Mistral AI (cloud). https://console.mistral.ai/api-keys/
    mistral_api_key: str = ""
    mistral_chat_model: str = "mistral-large-latest"

    # Cohere (embeddings). https://dashboard.cohere.com/api-keys
    cohere_api_key: str = ""
    cohere_embedding_model: str = "embed-english-v3.0"  # 1024 dims

    # Any OpenAI-compatible /v1 endpoint. This single provider unlocks
    # OpenRouter, Together, DeepSeek, Fireworks, vLLM, LM Studio and
    # llama.cpp's server — just point BASE_URL at them and name the model.
    # Local servers usually ignore the API key; leave it empty in that case.
    openai_compatible_base_url: str = ""  # e.g. https://openrouter.ai/api/v1
    openai_compatible_api_key: str = ""
    openai_compatible_chat_model: str = "gpt-oss-120b"

    # ── Embeddings (for vector GraphRAG retrieval) ───────
    # "fastembed" runs locally with no API key (default, great for demos).
    # The cloud options reuse the corresponding chat provider's credentials.
    # "fake" is for tests.
    embedding_provider: EmbeddingProvider = "fastembed"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"  # 384 dims
    gemini_embedding_model: str = "models/text-embedding-004"  # 768 dims
    ollama_embedding_model: str = "nomic-embed-text"  # 768 dims
    # Vector dimension MUST match the active embedding model. Defaults to
    # fastembed's bge-small (384). Change if you switch embedding providers.
    embedding_dim: int = 384

    # ── Generation tuning ────────────────────────────────
    extraction_temperature: float = 0.1
    chat_temperature: float = 0.3
    # Max PDF chunks processed per document (guards runaway ingest cost/time).
    max_chunks: int = 40
    # Concurrent LLM extraction calls.
    extraction_concurrency: int = 5
    # Per-chunk extraction timeout (seconds).
    extraction_timeout: int = 180

    # ── GraphRAG brain: entity resolution ────────────────
    # After extraction, near-duplicate entities ("Ahmed" / "Ahmed Maaloul",
    # "Postgres" / "PostgreSQL") are merged into one canonical node.
    entity_resolution_enabled: bool = True
    # Cosine similarity between entity embeddings above which two entities are
    # treated as the same. Deliberately high — a wrong merge is worse than a miss.
    entity_resolution_threshold: float = 0.93
    # Fuzzy name-similarity floor (0-1) required alongside the vector signal.
    entity_resolution_name_threshold: float = 0.87

    # ── GraphRAG brain: communities ──────────────────────
    # Louvain clustering (networkx) groups the graph into topical communities,
    # each summarized by the LLM. This is what enables corpus-level "global"
    # questions that plain vector RAG fundamentally cannot answer.
    community_detection_enabled: bool = True
    # Louvain resolution; higher => more, smaller communities.
    community_resolution: float = 1.0
    # Clusters smaller than this are ignored (noise).
    community_min_size: int = 3
    # Cap members listed in a summary prompt to bound token cost.
    community_max_members_in_summary: int = 30
    community_summary_temperature: float = 0.2

    # ── GraphRAG brain: source chunks (text units) ───────
    # Extraction distills prose into 15-word entity descriptions, which is lossy:
    # benchmarking showed plain passage RAG beating entity-only retrieval on raw
    # fact recall. Keeping the source chunks and returning them ALONGSIDE the
    # graph gives the model both the structure and the original evidence.
    store_source_chunks: bool = True
    chunk_retrieval_enabled: bool = True
    # Source excerpts included per answer.
    chunk_top_k: int = 4
    # Hard cap on excerpt characters injected into the prompt (cost guard).
    chunk_context_max_chars: int = 4000

    # ── GraphRAG brain: retrieval ────────────────────────
    # Hops to expand from each seed entity (2 enables multi-hop reasoning paths).
    retrieval_max_hops: int = 2
    # Route thematic/aggregate questions to community summaries ("global search")
    # instead of entity neighborhoods ("local search").
    query_routing_enabled: bool = True
    # Max reasoning paths surfaced alongside an answer.
    max_reasoning_paths: int = 6

    # ── GraphRAG brain: procedural memory ────────────────
    # Procedural Graphs (Lu et al., arXiv:2609.09153): a small directed graph of
    # actions / reasoning steps / statuses whose edges carry condition, guidance
    # and pitfalls. The entity graph is *what* the corpus says; this is *how*
    # to navigate it. Stored in the same Neo4j under separate labels.
    procedural_enabled: bool = True
    # Graph used when a caller does not name one (seeded from the expert prior
    # at startup if absent).
    procedural_default_graph: str = "graphrag-navigator"
    # Directed hops shown around the agent's current node (the paper uses 2).
    procedural_hops: int = 2
    # Recent trajectory steps handed to the guidance LLM (the paper's w = 3).
    procedural_window: int = 3
    # "raw" (default) = serialized local subgraph, ZERO extra LLM calls;
    # "generative" = the paper's guidance LLM (one extra call per step, cached);
    # "none" = no procedural guidance.
    procedural_guidance_mode: GuidanceMode = "raw"
    # Cosine floor for the embedding-based localization fallback (last action +
    # observation vs node descriptions) when no node name matches. Calibrated
    # for the default fastembed BAAI/bge-small-en-v1.5, whose cosines are
    # compressed high: against the bundled prior, 44 unrelated strings ("x",
    # "ls -la", "SELECT * FROM users", "done") peaked at 0.68, while 53
    # paraphrased tool calls scored 0.52-0.87 and every one at or above 0.72
    # landed on the right node. Below the floor the full graph is used, which
    # is the paper's behaviour. Re-measure when you change EMBEDDING_PROVIDER.
    procedural_semantic_threshold: float = 0.72
    # In-process LRU of generated guidance (generative mode only).
    procedural_guidance_cache_size: int = 256

    # ── GraphRAG Navigator agent ─────────────────────────
    # A ReAct agent that answers by walking the knowledge graph with
    # deterministic tools; every step is one LLM call, so these bound its cost.
    agent_max_steps: int = 8
    # Tool observations are truncated to this many characters before they
    # re-enter the prompt (cost guard; a full source passage can be long).
    agent_observation_max_chars: int = 1500
    agent_temperature: float = 0.0

    # ── Procedural graph self-evolution (offline) ────────
    # Training traces shown to the refiner are tail-truncated to this many
    # characters (the paper's Tail_Lmax: the END of the traces is kept).
    evolution_trajectory_max_chars: int = 24000
    # Hard budget across rollouts + guidance + refiner calls for one run; the
    # loop stops cleanly (never mid-save) when it would be exceeded.
    evolution_max_llm_calls: int = 400
    evolution_default_rounds: int = 3
    evolution_default_batch_size: int = 10

    model_config = SettingsConfigDict(
        # The canonical .env lives at the REPO ROOT (docker compose reads it too),
        # but the API is usually launched from backend/ — where a bare ".env"
        # would resolve to backend/.env and be silently missed. Check both;
        # later entries win, so a backend-local override still takes precedence.
        # (Real environment variables outrank both, which is what CI/Docker use.)
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
