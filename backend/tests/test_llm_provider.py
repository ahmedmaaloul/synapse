# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for the pluggable provider factory and offline embeddings.

Every test here is hermetic: no network, no real API calls. Provider branches
validate credentials *before* touching an SDK, so we can assert on the error
messages without any package being installed or any key being set.
"""

from types import SimpleNamespace

import pytest

from app.config import EmbeddingProvider, LLMProvider, Settings
from app.services.llm_provider import (
    CHAT_PROVIDERS,
    EMBEDDING_PROVIDERS,
    FakeDeterministicEmbeddings,
    ProviderConfigError,
    _load_embeddings,
    _missing_package,
    get_chat_llm,
    get_embeddings,
)


def _literal_values(alias) -> set[str]:
    """Extract the string options from a `Literal[...]` type alias."""
    from typing import get_args

    return set(get_args(alias))


class TestFakeEmbeddings:
    def test_deterministic(self):
        e = FakeDeterministicEmbeddings(dim=32)
        assert e.embed_query("hello world") == e.embed_query("hello world")

    def test_unit_norm_and_dim(self):
        e = FakeDeterministicEmbeddings(dim=64)
        v = e.embed_query("some text here")
        assert len(v) == 64
        assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-6

    def test_different_text_differs(self):
        e = FakeDeterministicEmbeddings(dim=64)
        assert e.embed_query("python") != e.embed_query("javascript")

    def test_embed_documents_batches(self):
        e = FakeDeterministicEmbeddings(dim=16)
        out = e.embed_documents(["a", "b", "c"])
        assert len(out) == 3 and all(len(v) == 16 for v in out)


class TestEmbeddingsFactory:
    def test_fake_provider(self):
        s = Settings(embedding_provider="fake", embedding_dim=128)
        emb = get_embeddings(s)
        assert isinstance(emb, FakeDeterministicEmbeddings)
        assert emb.dim == 128

    def test_settings_reject_bad_provider_at_config_layer(self):
        # The Literal type means a typo is caught early, at construction time.
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            Settings(embedding_provider="bogus")  # type: ignore[arg-type]

    def test_unknown_provider_raises_at_factory(self):
        # Defensive branch, reached only if config validation is bypassed.
        with pytest.raises(ProviderConfigError, match="Unknown EMBEDDING_PROVIDER"):
            _load_embeddings(
                provider="bogus",
                fastembed_model="x",
                gemini_model="x",
                ollama_model="x",
                ollama_base_url="x",
                google_api_key="",
                dim=8,
            )

    def test_unknown_provider_message_lists_every_valid_option(self):
        with pytest.raises(ProviderConfigError) as exc:
            _load_embeddings(
                provider="bogus",
                fastembed_model="x",
                gemini_model="x",
                ollama_model="x",
                ollama_base_url="x",
                google_api_key="",
                dim=8,
            )
        for name in EMBEDDING_PROVIDERS:
            assert name in str(exc.value)


class TestEmbeddingProviderCredentialErrors:
    """Every cloud embedding provider must fail loudly *and usefully*."""

    def test_gemini_without_key(self):
        s = Settings(embedding_provider="gemini", google_api_key="")
        with pytest.raises(ProviderConfigError, match="GOOGLE_API_KEY"):
            get_embeddings(s)

    def test_openai_without_key(self):
        s = Settings(embedding_provider="openai", openai_api_key="")
        with pytest.raises(ProviderConfigError) as exc:
            get_embeddings(s)
        msg = str(exc.value)
        assert "OPENAI_API_KEY" in msg
        assert "platform.openai.com" in msg

    def test_azure_openai_lists_every_missing_value(self):
        s = Settings(
            embedding_provider="azure_openai",
            azure_openai_api_key="",
            azure_openai_endpoint="",
            azure_openai_embedding_deployment="",
        )
        with pytest.raises(ProviderConfigError) as exc:
            get_embeddings(s)
        msg = str(exc.value)
        assert "AZURE_OPENAI_API_KEY" in msg
        assert "AZURE_OPENAI_ENDPOINT" in msg
        assert "AZURE_OPENAI_EMBEDDING_DEPLOYMENT" in msg

    def test_azure_openai_reports_only_what_is_missing(self):
        s = Settings(
            embedding_provider="azure_openai",
            azure_openai_api_key="k",
            azure_openai_endpoint="https://r.openai.azure.com/",
            azure_openai_embedding_deployment="",
        )
        with pytest.raises(ProviderConfigError) as exc:
            get_embeddings(s)
        msg = str(exc.value)
        assert "AZURE_OPENAI_EMBEDDING_DEPLOYMENT" in msg
        assert "AZURE_OPENAI_API_KEY" not in msg

    def test_vertex_without_project(self):
        s = Settings(embedding_provider="vertex", vertex_project="")
        with pytest.raises(ProviderConfigError) as exc:
            get_embeddings(s)
        msg = str(exc.value)
        assert "VERTEX_PROJECT" in msg
        assert "application-default login" in msg

    def test_bedrock_without_region(self):
        s = Settings(embedding_provider="bedrock", bedrock_region="")
        with pytest.raises(ProviderConfigError) as exc:
            get_embeddings(s)
        msg = str(exc.value)
        assert "BEDROCK_REGION" in msg
        assert "AWS_ACCESS_KEY_ID" in msg

    def test_cohere_without_key(self):
        s = Settings(embedding_provider="cohere", cohere_api_key="")
        with pytest.raises(ProviderConfigError) as exc:
            get_embeddings(s)
        msg = str(exc.value)
        assert "COHERE_API_KEY" in msg
        assert "dashboard.cohere.com" in msg


class TestChatFactory:
    def test_unknown_provider_raises_at_factory(self):
        # Bypass config validation to exercise the factory's defensive branch.
        fake = SimpleNamespace(llm_provider="bogus")
        with pytest.raises(ProviderConfigError, match="Unknown LLM_PROVIDER"):
            get_chat_llm(settings=fake)  # type: ignore[arg-type]

    def test_unknown_provider_message_lists_every_valid_option(self):
        fake = SimpleNamespace(llm_provider="bogus")
        with pytest.raises(ProviderConfigError) as exc:
            get_chat_llm(settings=fake)  # type: ignore[arg-type]
        for name in CHAT_PROVIDERS:
            assert name in str(exc.value)

    def test_settings_reject_bad_llm_provider_at_config_layer(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            Settings(llm_provider="bogus")  # type: ignore[arg-type]


class TestChatProviderCredentialErrors:
    """Missing credentials must name the env var and where to get a key."""

    def test_gemini_without_key_raises_actionable_error(self):
        s = Settings(llm_provider="gemini", google_api_key="")
        with pytest.raises(ProviderConfigError, match="GOOGLE_API_KEY"):
            get_chat_llm(settings=s)

    def test_claude_without_key_raises_actionable_error(self):
        s = Settings(llm_provider="claude", anthropic_api_key="")
        with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
            get_chat_llm(settings=s)

    def test_openai_without_key(self):
        s = Settings(llm_provider="openai", openai_api_key="")
        with pytest.raises(ProviderConfigError) as exc:
            get_chat_llm(settings=s)
        msg = str(exc.value)
        assert "OPENAI_API_KEY" in msg
        assert "https://platform.openai.com/api-keys" in msg

    def test_azure_openai_without_anything(self):
        s = Settings(
            llm_provider="azure_openai",
            azure_openai_api_key="",
            azure_openai_endpoint="",
            azure_openai_chat_deployment="",
        )
        with pytest.raises(ProviderConfigError) as exc:
            get_chat_llm(settings=s)
        msg = str(exc.value)
        assert "AZURE_OPENAI_API_KEY" in msg
        assert "AZURE_OPENAI_ENDPOINT" in msg
        assert "AZURE_OPENAI_CHAT_DEPLOYMENT" in msg
        assert "openai.azure.com" in msg

    def test_azure_openai_missing_only_deployment(self):
        s = Settings(
            llm_provider="azure_openai",
            azure_openai_api_key="k",
            azure_openai_endpoint="https://r.openai.azure.com/",
            azure_openai_chat_deployment="",
        )
        with pytest.raises(ProviderConfigError) as exc:
            get_chat_llm(settings=s)
        assert "AZURE_OPENAI_CHAT_DEPLOYMENT" in str(exc.value)
        assert "AZURE_OPENAI_API_KEY" not in str(exc.value)

    def test_vertex_without_project(self):
        s = Settings(llm_provider="vertex", vertex_project="")
        with pytest.raises(ProviderConfigError) as exc:
            get_chat_llm(settings=s)
        msg = str(exc.value)
        assert "VERTEX_PROJECT" in msg
        assert "gcloud auth application-default login" in msg

    def test_bedrock_without_region(self):
        s = Settings(llm_provider="bedrock", bedrock_region="")
        with pytest.raises(ProviderConfigError) as exc:
            get_chat_llm(settings=s)
        msg = str(exc.value)
        assert "BEDROCK_REGION" in msg
        assert "AWS_ACCESS_KEY_ID" in msg
        assert "modelaccess" in msg

    def test_groq_without_key(self):
        s = Settings(llm_provider="groq", groq_api_key="")
        with pytest.raises(ProviderConfigError) as exc:
            get_chat_llm(settings=s)
        msg = str(exc.value)
        assert "GROQ_API_KEY" in msg
        assert "https://console.groq.com/keys" in msg

    def test_mistral_without_key(self):
        s = Settings(llm_provider="mistral", mistral_api_key="")
        with pytest.raises(ProviderConfigError) as exc:
            get_chat_llm(settings=s)
        msg = str(exc.value)
        assert "MISTRAL_API_KEY" in msg
        assert "https://console.mistral.ai/api-keys/" in msg

    def test_openai_compatible_without_base_url(self):
        s = Settings(llm_provider="openai_compatible", openai_compatible_base_url="")
        with pytest.raises(ProviderConfigError) as exc:
            get_chat_llm(settings=s)
        msg = str(exc.value)
        assert "OPENAI_COMPATIBLE_BASE_URL" in msg
        # The error doubles as documentation for the gateways it unlocks.
        for gateway in ("OpenRouter", "Together", "DeepSeek", "Fireworks", "vLLM", "LM Studio"):
            assert gateway in msg


class TestOpenAICompatibleBranch:
    """The one branch worth constructing: it needs no credentials at all."""

    def test_local_server_needs_no_api_key(self):
        pytest.importorskip("langchain_openai")
        s = Settings(
            llm_provider="openai_compatible",
            openai_compatible_base_url="http://localhost:1234/v1",
            openai_compatible_api_key="",
            openai_compatible_chat_model="local-model",
        )
        llm = get_chat_llm(settings=s)  # constructs offline, makes no request
        assert llm.model_name == "local-model"
        assert str(llm.openai_api_base) == "http://localhost:1234/v1"

    def test_custom_base_url_and_key_are_forwarded(self):
        pytest.importorskip("langchain_openai")
        s = Settings(
            llm_provider="openai_compatible",
            openai_compatible_base_url="https://openrouter.ai/api/v1",
            openai_compatible_api_key="sk-test-not-a-real-key",
            openai_compatible_chat_model="deepseek/deepseek-chat",
        )
        llm = get_chat_llm(settings=s)
        assert str(llm.openai_api_base) == "https://openrouter.ai/api/v1"
        assert llm.openai_api_key.get_secret_value() == "sk-test-not-a-real-key"


class TestProviderRegistryStaysInSync:
    """Guard against adding a provider to one list but not the other."""

    def test_chat_literal_matches_factory_tuple(self):
        assert _literal_values(LLMProvider) == set(CHAT_PROVIDERS)

    def test_embedding_literal_matches_factory_tuple(self):
        assert _literal_values(EmbeddingProvider) == set(EMBEDDING_PROVIDERS)

    @pytest.mark.parametrize("provider", sorted(_literal_values(LLMProvider)))
    def test_every_declared_chat_provider_is_handled(self, provider):
        # With an empty .env each provider either builds (ollama) or raises a
        # credential error — but never the "unknown provider" fallback.
        s = Settings(llm_provider=provider)  # type: ignore[arg-type]
        try:
            get_chat_llm(settings=s)
        except Exception as e:  # noqa: BLE001 - SDKs may raise their own auth errors
            # Whatever fails, it must not be "this provider doesn't exist".
            assert "Unknown LLM_PROVIDER" not in str(e)

    @pytest.mark.parametrize("provider", sorted(_literal_values(EmbeddingProvider)))
    def test_every_declared_embedding_provider_is_handled(self, provider):
        if provider == "fastembed":
            pytest.skip("fastembed downloads a model on first use — not hermetic")
        s = Settings(embedding_provider=provider)  # type: ignore[arg-type]
        try:
            get_embeddings(s)
        except Exception as e:  # noqa: BLE001 - SDKs may raise their own auth errors
            assert "Unknown EMBEDDING_PROVIDER" not in str(e)


class TestMissingPackageErrors:
    """The dependency-missing path must hand the user a copy-pasteable command."""

    def test_includes_exact_pip_command_with_pin(self):
        err = _missing_package("langchain-openai", pin=">=0.3,<0.4")
        msg = str(err)
        assert "pip install 'langchain-openai>=0.3,<0.4'" in msg

    def test_optional_packages_also_point_at_the_extras_file(self):
        err = _missing_package("langchain-aws", pin="==0.2.24", optional=True)
        msg = str(err)
        assert "pip install 'langchain-aws==0.2.24'" in msg
        assert "requirements-providers.txt" in msg

    def test_is_a_provider_config_error(self):
        assert isinstance(_missing_package("langchain-groq"), ProviderConfigError)
