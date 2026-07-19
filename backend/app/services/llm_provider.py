# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Pluggable LLM Provider

A single factory that returns a LangChain chat model or embeddings object for the
configured backend. This is what makes the AI layer *hot-swappable* between:

  • Google Gemini      (cloud, free tier)      LLM_PROVIDER=gemini
  • Anthropic Claude   (cloud)                 LLM_PROVIDER=claude
  • OpenAI             (cloud)                 LLM_PROVIDER=openai
  • Azure OpenAI       (cloud, enterprise)     LLM_PROVIDER=azure_openai
  • Google Vertex AI   (cloud, GCP)            LLM_PROVIDER=vertex
  • AWS Bedrock        (cloud, AWS)            LLM_PROVIDER=bedrock
  • Groq               (cloud, fast)           LLM_PROVIDER=groq
  • Mistral AI         (cloud)                 LLM_PROVIDER=mistral
  • Ollama             (fully local/private)   LLM_PROVIDER=ollama
  • Any OpenAI-compatible endpoint             LLM_PROVIDER=openai_compatible

``openai_compatible`` is the escape hatch: it is ``ChatOpenAI`` pointed at a
custom ``base_url``, which covers OpenRouter, Together, DeepSeek, Fireworks,
vLLM, LM Studio and llama.cpp's server without any extra dependency.

Provider SDKs are imported lazily inside each branch so the app boots (and the
test suite runs) even when an optional provider package isn't installed. Missing
credentials raise a clear, actionable error only when that provider is invoked;
the heavier SDKs live in ``requirements-providers.txt``.
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
    from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)


class ProviderConfigError(RuntimeError):
    """Raised when a provider is selected but not usable (missing key/dep)."""


#: Chat providers accepted by :func:`get_chat_llm` (kept in sync with
#: ``app.config.LLMProvider``) — used to build the "unknown provider" message.
CHAT_PROVIDERS = (
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
)

#: Embedding providers accepted by :func:`get_embeddings`.
EMBEDDING_PROVIDERS = (
    "fastembed",
    "gemini",
    "ollama",
    "fake",
    "openai",
    "azure_openai",
    "vertex",
    "bedrock",
    "cohere",
)


def _missing_package(
    package: str, *, pin: str = "", optional: bool = False
) -> ProviderConfigError:
    """Build a consistent, actionable 'SDK not installed' error.

    Args:
        package: the distribution name, e.g. ``langchain-aws``.
        pin: the version range Synapse is tested against; included in the pip
            command so the install can't drag ``langchain-core`` to 1.x.
        optional: True for SDKs that live in ``requirements-providers.txt``
            rather than the batteries-included ``requirements.txt``.
    """
    spec = f"{package}{pin}" if pin else package
    msg = f"{package} is not installed. Run: pip install '{spec}'"
    if optional:
        msg += (
            " — or install every optional provider at once with "
            "`pip install -r backend/requirements-providers.txt`"
        )
    return ProviderConfigError(msg)


# ── Chat models ──────────────────────────────────────────
def get_chat_llm(
    *,
    streaming: bool = False,
    json_mode: bool = False,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> BaseChatModel | Runnable:
    """Return a LangChain chat model for the configured provider.

    Returns a ``Runnable`` rather than a bare chat model when a provider needs
    invoke-time options bound (Gemini's JSON mode) — both compose identically
    with ``prompt | llm`` and support ainvoke/astream.

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
        # NOTE: never pass model_kwargs=None — ChatGoogleGenerativeAI crashes on it.
        model = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.google_api_key,
            temperature=temp,
        )
        if json_mode:
            # `model_kwargs={"response_mime_type": ...}` is SILENTLY IGNORED by
            # langchain-google-genai 2.1.x: the field exists, but the request
            # builder never reads it (grep response_mime_type in chat_models.py
            # -> no hits). Only `generation_config` is merged, in
            # _prepare_params, and it must arrive as an invoke-time kwarg —
            # which is exactly what .bind() does.
            return model.bind(
                generation_config={"response_mime_type": "application/json"}
            )
        return model

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

    if provider == "openai":
        if not settings.openai_api_key:
            raise ProviderConfigError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty. "
                "Create a key at https://platform.openai.com/api-keys"
            )
        ChatOpenAI = _import_chat_openai()
        kwargs: dict = {
            "model": settings.openai_chat_model,
            "api_key": settings.openai_api_key,
            "temperature": temp,
            "streaming": streaming,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    if provider == "openai_compatible":
        # ChatOpenAI + a custom base_url = OpenRouter, Together, DeepSeek,
        # Fireworks, vLLM, LM Studio, llama.cpp — no extra dependency needed.
        if not settings.openai_compatible_base_url:
            raise ProviderConfigError(
                "LLM_PROVIDER=openai_compatible but OPENAI_COMPATIBLE_BASE_URL is empty. "
                "Point it at any OpenAI-compatible /v1 endpoint, e.g. "
                "https://openrouter.ai/api/v1 (OpenRouter), "
                "https://api.together.xyz/v1 (Together), "
                "https://api.deepseek.com/v1 (DeepSeek), "
                "https://api.fireworks.ai/inference/v1 (Fireworks), "
                "http://localhost:8000/v1 (vLLM), "
                "http://localhost:1234/v1 (LM Studio), "
                "http://localhost:8080/v1 (llama.cpp). "
                "Then set OPENAI_COMPATIBLE_CHAT_MODEL and, for hosted gateways, "
                "OPENAI_COMPATIBLE_API_KEY."
            )
        ChatOpenAI = _import_chat_openai()
        kwargs: dict = {
            "model": settings.openai_compatible_chat_model,
            "base_url": settings.openai_compatible_base_url,
            # Local servers (vLLM/LM Studio/llama.cpp) ignore the key, but the
            # OpenAI SDK refuses to construct without one.
            "api_key": settings.openai_compatible_api_key or "not-needed",
            "temperature": temp,
            "streaming": streaming,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    if provider == "azure_openai":
        _require_azure(settings, deployment=settings.azure_openai_chat_deployment)
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:  # pragma: no cover - env-dependent
            raise _missing_package("langchain-openai", pin=">=0.3,<0.4") from e
        kwargs: dict = {
            "azure_endpoint": settings.azure_openai_endpoint,
            "azure_deployment": settings.azure_openai_chat_deployment,
            "api_version": settings.azure_openai_api_version,
            "api_key": settings.azure_openai_api_key,
            "temperature": temp,
            "streaming": streaming,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return AzureChatOpenAI(**kwargs)

    if provider == "vertex":
        if not settings.vertex_project:
            raise ProviderConfigError(
                "LLM_PROVIDER=vertex but VERTEX_PROJECT is empty. Set it to your GCP "
                "project id, set VERTEX_LOCATION (default us-central1), enable the "
                "Vertex AI API, and authenticate with Application Default Credentials: "
                "`gcloud auth application-default login` (or set "
                "GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON key)."
            )
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as e:  # pragma: no cover - env-dependent
            raise _missing_package(
                "langchain-google-vertexai", pin="==2.0.24", optional=True
            ) from e
        kwargs: dict = {
            "model_name": settings.vertex_chat_model,
            "project": settings.vertex_project,
            "location": settings.vertex_location,
            "temperature": temp,
            "streaming": streaming,
        }
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        return ChatVertexAI(**kwargs)

    if provider == "bedrock":
        if not settings.bedrock_region:
            raise ProviderConfigError(
                "LLM_PROVIDER=bedrock but BEDROCK_REGION is empty. Set it to a region "
                "where you have requested model access (e.g. us-east-1), and supply AWS "
                "credentials the usual way — AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, "
                "AWS_PROFILE, or an EC2/ECS/EKS instance role. Request model access at "
                "https://console.aws.amazon.com/bedrock/home#/modelaccess"
            )
        try:
            # ChatBedrockConverse uses the unified Converse API, so tool-calling
            # works the same across every Bedrock model family.
            from langchain_aws import ChatBedrockConverse
        except ImportError as e:  # pragma: no cover - env-dependent
            raise _missing_package("langchain-aws", pin="==0.2.24", optional=True) from e
        return ChatBedrockConverse(
            model=settings.bedrock_chat_model,
            region_name=settings.bedrock_region,
            temperature=temp,
        )

    if provider == "groq":
        if not settings.groq_api_key:
            raise ProviderConfigError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is empty. "
                "Get a free key at https://console.groq.com/keys"
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:  # pragma: no cover - env-dependent
            raise _missing_package("langchain-groq", pin="==0.3.2", optional=True) from e
        kwargs: dict = {
            "model": settings.groq_chat_model,
            "api_key": settings.groq_api_key,
            "temperature": temp,
            "streaming": streaming,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatGroq(**kwargs)

    if provider == "mistral":
        if not settings.mistral_api_key:
            raise ProviderConfigError(
                "LLM_PROVIDER=mistral but MISTRAL_API_KEY is empty. "
                "Get a key at https://console.mistral.ai/api-keys/"
            )
        try:
            from langchain_mistralai import ChatMistralAI
        except ImportError as e:  # pragma: no cover - env-dependent
            raise _missing_package(
                "langchain-mistralai", pin="==0.2.10", optional=True
            ) from e
        return ChatMistralAI(
            model_name=settings.mistral_chat_model,
            api_key=settings.mistral_api_key,
            temperature=temp,
            streaming=streaming,
        )

    raise ProviderConfigError(
        f"Unknown LLM_PROVIDER={provider!r}. Use one of: {', '.join(CHAT_PROVIDERS)}."
    )


def _import_chat_openai():
    """Lazily import ChatOpenAI (shared by the openai / openai_compatible branches)."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:  # pragma: no cover - env-dependent
        raise _missing_package("langchain-openai", pin=">=0.3,<0.4") from e
    return ChatOpenAI


def _require_azure(settings: Settings, *, deployment: str) -> None:
    """Validate the three values Azure OpenAI always needs."""
    missing = []
    if not settings.azure_openai_api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not settings.azure_openai_endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not deployment:
        missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT (or *_EMBEDDING_DEPLOYMENT)")
    if missing:
        raise ProviderConfigError(
            "Azure OpenAI is selected but these are empty: "
            + ", ".join(missing)
            + ". Find the key and endpoint (https://<resource>.openai.azure.com/) under "
            "'Keys and Endpoint' in the Azure portal, use the *deployment name* you "
            "created in Azure AI Foundry (not the model name), and set "
            "AZURE_OPENAI_API_VERSION (default 2024-10-21)."
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
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_embedding_model,
        azure_api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        azure_api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
        vertex_project=settings.vertex_project,
        vertex_location=settings.vertex_location,
        vertex_model=settings.vertex_embedding_model,
        bedrock_region=settings.bedrock_region,
        bedrock_model=settings.bedrock_embedding_model,
        cohere_api_key=settings.cohere_api_key,
        cohere_model=settings.cohere_embedding_model,
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
    # Cloud providers — defaulted so callers (and tests) only pass what they use.
    openai_api_key: str = "",
    openai_model: str = "text-embedding-3-small",
    azure_api_key: str = "",
    azure_endpoint: str = "",
    azure_api_version: str = "2024-10-21",
    azure_deployment: str = "",
    vertex_project: str = "",
    vertex_location: str = "us-central1",
    vertex_model: str = "text-embedding-005",
    bedrock_region: str = "",
    bedrock_model: str = "amazon.titan-embed-text-v2:0",
    cohere_api_key: str = "",
    cohere_model: str = "embed-english-v3.0",
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

    if provider == "openai":
        if not openai_api_key:
            raise ProviderConfigError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is empty. "
                "Create a key at https://platform.openai.com/api-keys. "
                "Remember to match EMBEDDING_DIM to the model "
                "(text-embedding-3-small = 1536)."
            )
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:  # pragma: no cover - env-dependent
            raise _missing_package("langchain-openai", pin=">=0.3,<0.4") from e
        return OpenAIEmbeddings(model=openai_model, api_key=openai_api_key)

    if provider == "azure_openai":
        missing = [
            name
            for name, value in (
                ("AZURE_OPENAI_API_KEY", azure_api_key),
                ("AZURE_OPENAI_ENDPOINT", azure_endpoint),
                ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", azure_deployment),
            )
            if not value
        ]
        if missing:
            raise ProviderConfigError(
                "EMBEDDING_PROVIDER=azure_openai but these are empty: "
                + ", ".join(missing)
                + ". Get the key and endpoint (https://<resource>.openai.azure.com/) "
                "from 'Keys and Endpoint' in the Azure portal and use the *deployment "
                "name* you created for the embedding model."
            )
        try:
            from langchain_openai import AzureOpenAIEmbeddings
        except ImportError as e:  # pragma: no cover - env-dependent
            raise _missing_package("langchain-openai", pin=">=0.3,<0.4") from e
        return AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=azure_api_version,
            api_key=azure_api_key,
        )

    if provider == "vertex":
        if not vertex_project:
            raise ProviderConfigError(
                "EMBEDDING_PROVIDER=vertex but VERTEX_PROJECT is empty. Set your GCP "
                "project id and authenticate with Application Default Credentials: "
                "`gcloud auth application-default login`."
            )
        try:
            from langchain_google_vertexai import VertexAIEmbeddings
        except ImportError as e:  # pragma: no cover - env-dependent
            raise _missing_package(
                "langchain-google-vertexai", pin="==2.0.24", optional=True
            ) from e
        return VertexAIEmbeddings(
            model_name=vertex_model, project=vertex_project, location=vertex_location
        )

    if provider == "bedrock":
        if not bedrock_region:
            raise ProviderConfigError(
                "EMBEDDING_PROVIDER=bedrock but BEDROCK_REGION is empty. Set a region "
                "where you have model access and supply AWS credentials via "
                "AWS_ACCESS_KEY_ID / AWS_PROFILE or an instance role."
            )
        try:
            from langchain_aws import BedrockEmbeddings
        except ImportError as e:  # pragma: no cover - env-dependent
            raise _missing_package("langchain-aws", pin="==0.2.24", optional=True) from e
        return BedrockEmbeddings(model_id=bedrock_model, region_name=bedrock_region)

    if provider == "cohere":
        if not cohere_api_key:
            raise ProviderConfigError(
                "EMBEDDING_PROVIDER=cohere but COHERE_API_KEY is empty. "
                "Get a key at https://dashboard.cohere.com/api-keys "
                "(embed-english-v3.0 = 1024 dims, so set EMBEDDING_DIM=1024)."
            )
        try:
            from langchain_cohere import CohereEmbeddings
        except ImportError as e:  # pragma: no cover - env-dependent
            raise _missing_package("langchain-cohere", pin="==0.4.5", optional=True) from e
        return CohereEmbeddings(model=cohere_model, cohere_api_key=cohere_api_key)

    if provider == "fake":
        return FakeDeterministicEmbeddings(dim=dim)

    raise ProviderConfigError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}. "
        f"Use one of: {', '.join(EMBEDDING_PROVIDERS)}."
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
