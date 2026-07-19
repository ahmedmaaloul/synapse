"""
Project Synapse — Configuration

Central, typed settings loaded from environment / .env via pydantic-settings.

The AI backend is *pluggable*: the same code path runs against a cloud model
(Google Gemini or Anthropic Claude) or a fully local model (Ollama). Pick the
provider with ``LLM_PROVIDER`` and, for embeddings, ``EMBEDDING_PROVIDER``.
Every provider-specific value has a sensible default so the app boots even with
an empty .env — it will only fail (with a clear message) when a provider that
needs a key is actually invoked.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["gemini", "claude", "ollama"]
EmbeddingProvider = Literal["gemini", "ollama", "fastembed", "fake"]


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── App ──────────────────────────────────────────────
    app_name: str = "Project Synapse"
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

    # ── Embeddings (for vector GraphRAG retrieval) ───────
    # "fastembed" runs locally with no API key (default, great for demos).
    # "gemini"/"ollama" reuse the corresponding provider. "fake" is for tests.
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
