"""
Project Synapse — Pluggable LLM Provider

A single factory that returns a LangChain chat model or embeddings object for the
configured backend. This is what makes the AI layer *hot-swappable* between:

  • Google Gemini  (cloud, free tier)
  • Anthropic Claude (cloud)
  • Ollama          (fully local / private)

Provider SDKs are imported lazily inside each branch so the app boots (and the
test suite runs) even when an optional provider package isn't installed. Missing
credentials raise a clear, actionable error only when that provider is invoked.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class ProviderConfigError(RuntimeError):
    """Raised when a provider is selected but not usable (missing key/dep)."""


# ── Chat models ──────────────────────────────────────────
def get_chat_llm(
    *,
    streaming: bool = False,
    json_mode: bool = False,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Return a LangChain chat model for the configured provider.

    Args:
        streaming: enable token streaming (used by the chat engine).
        json_mode: request strict JSON output where the provider supports it
            (Ollama). Other providers rely on prompt instructions + robust parsing.
        temperature: sampling temperature; defaults per-provider if None.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    temp = temperature if temperature is not None else 0.2

    if provider == "gemini":
        if not settings.google_api_key:
            raise ProviderConfigError(
                "LLM_PROVIDER=gemini but GOOGLE_API_KEY is empty. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ProviderConfigError(
                "langchain-google-genai is not installed. `pip install langchain-google-genai`"
            ) from e
        kwargs: dict = {
            "model": settings.gemini_chat_model,
            "google_api_key": settings.google_api_key,
            "temperature": temp,
        }
        if json_mode:
            # Only set model_kwargs when non-empty — ChatGoogleGenerativeAI
            # crashes on model_kwargs=None.
            kwargs["model_kwargs"] = {"response_mime_type": "application/json"}
        return ChatGoogleGenerativeAI(**kwargs)

    if provider in ("claude", "anthropic"):
        if not settings.anthropic_api_key:
            raise ProviderConfigError(
                "LLM_PROVIDER=claude but ANTHROPIC_API_KEY is empty. "
                "Get a key at https://console.anthropic.com/"
            )
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ProviderConfigError(
                "langchain-anthropic is not installed. `pip install langchain-anthropic`"
            ) from e
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=temp,
            streaming=streaming,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ProviderConfigError(
                "langchain-ollama is not installed. `pip install langchain-ollama`"
            ) from e
        kwargs: dict = {
            "model": settings.ollama_chat_model,
            "base_url": settings.ollama_base_url,
            "temperature": temp,
        }
        if json_mode:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)

    raise ProviderConfigError(
        f"Unknown LLM_PROVIDER={provider!r}. Use one of: gemini, claude, ollama."
    )


# ── Embeddings ───────────────────────────────────────────
def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return a LangChain Embeddings object for the configured provider.

    Loading is cached on the resolved primitive config (not the Settings object,
    which isn't hashable) because model init — e.g. fastembed — is expensive.
    """
    settings = settings or get_settings()
    return _load_embeddings(
        provider=settings.embedding_provider.lower(),
        fastembed_model=settings.fastembed_model,
        gemini_model=settings.gemini_embedding_model,
        ollama_model=settings.ollama_embedding_model,
        ollama_base_url=settings.ollama_base_url,
        google_api_key=settings.google_api_key,
        dim=settings.embedding_dim,
    )


@lru_cache
def _load_embeddings(
    *,
    provider: str,
    fastembed_model: str,
    gemini_model: str,
    ollama_model: str,
    ollama_base_url: str,
    google_api_key: str,
    dim: int,
) -> Embeddings:
    if provider == "fastembed":
        try:
            from langchain_community.embeddings import FastEmbedEmbeddings
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ProviderConfigError(
                "fastembed embeddings need `pip install fastembed langchain-community`"
            ) from e
        return FastEmbedEmbeddings(model_name=fastembed_model)

    if provider == "gemini":
        if not google_api_key:
            raise ProviderConfigError(
                "EMBEDDING_PROVIDER=gemini but GOOGLE_API_KEY is empty."
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=gemini_model, google_api_key=google_api_key
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=ollama_model, base_url=ollama_base_url)

    if provider == "fake":
        return FakeDeterministicEmbeddings(dim=dim)

    raise ProviderConfigError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}. "
        "Use one of: fastembed, gemini, ollama, fake."
    )


class FakeDeterministicEmbeddings:
    """A dependency-free, deterministic embedding for offline dev & tests.

    Maps text to a fixed-dimension unit vector via hashing. Not semantically
    meaningful, but stable and dependency-free so vector search and tests work
    without any model download or API key.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0 + (h % 7) / 7.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
