# SPDX-License-Identifier: AGPL-3.0-or-later
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
