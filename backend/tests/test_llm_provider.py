"""Tests for the pluggable provider factory and offline embeddings."""

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.services.llm_provider import (
    FakeDeterministicEmbeddings,
    ProviderConfigError,
    _load_embeddings,
    get_chat_llm,
    get_embeddings,
)


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
        with pytest.raises(ProviderConfigError):
            _load_embeddings(
                provider="bogus",
                fastembed_model="x",
                gemini_model="x",
                ollama_model="x",
                ollama_base_url="x",
                google_api_key="",
                dim=8,
            )


class TestChatFactory:
    def test_unknown_provider_raises_at_factory(self):
        # Bypass config validation to exercise the factory's defensive branch.
        fake = SimpleNamespace(llm_provider="bogus")
        with pytest.raises(ProviderConfigError, match="Unknown LLM_PROVIDER"):
            get_chat_llm(settings=fake)  # type: ignore[arg-type]

    def test_gemini_without_key_raises_actionable_error(self):
        s = Settings(llm_provider="gemini", google_api_key="")
        with pytest.raises(ProviderConfigError, match="GOOGLE_API_KEY"):
            get_chat_llm(settings=s)

    def test_claude_without_key_raises_actionable_error(self):
        s = Settings(llm_provider="claude", anthropic_api_key="")
        with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
            get_chat_llm(settings=s)
