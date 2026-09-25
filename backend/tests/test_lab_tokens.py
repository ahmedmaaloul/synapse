# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Lab token counting: encoding choice, the flagged fallback, offline safety.

Hermetic: nothing is downloaded. Under pytest ``tokens`` only loads encodings
already in tiktoken's cache; the one real-tokenizer test is skipped otherwise.
"""

from __future__ import annotations

import pytest

from app.lab import tokens


@pytest.fixture(autouse=True)
def _fresh_encodings():
    tokens.reset_cache()
    yield
    tokens.reset_cache()


class TestEncodingChoice:
    @pytest.mark.parametrize(
        "model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini", "gpt-5-nano",
         "gpt-5", "o1", "o3-mini", "o4-mini", "chatgpt-4o-latest", "GPT-5-NANO"],
    )
    def test_o200k_family(self, model):
        assert tokens.encoding_name_for(model) == tokens.O200K

    @pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4", "mistral", "ollama", "", None])
    def test_everything_else_is_cl100k(self, model):
        assert tokens.encoding_name_for(model) == tokens.CL100K


class _FakeEncoding:
    """Counts whitespace-separated words — enough to prove the real path is used."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(self, text, **kwargs):
        self.calls.append(kwargs)
        return text.split()


class TestCounting:
    def test_uses_the_encoding_when_available(self, monkeypatch):
        fake = _FakeEncoding()
        monkeypatch.setattr(tokens, "get_encoding", lambda model: fake)
        assert tokens.count_tokens("one two three", "gpt-5-nano") == (3, False)
        # Special-token markers in user text must never raise.
        assert fake.calls[-1] == {"disallowed_special": ()}

    def test_falls_back_to_len_over_4_and_says_so(self, monkeypatch):
        monkeypatch.setattr(tokens, "get_encoding", lambda model: None)
        assert tokens.count_tokens("x" * 9, "gpt-5-nano") == (3, True)
        assert tokens.tokenizer_label("gpt-5-nano") == "estimate:len/4"

    def test_a_crashing_encoder_degrades_to_the_estimate(self, monkeypatch):
        class Broken:
            def encode(self, text, **kwargs):
                raise RuntimeError("boom")

        monkeypatch.setattr(tokens, "get_encoding", lambda model: Broken())
        assert tokens.count_tokens("abcdefgh", "gpt-4o") == (2, True)

    def test_empty_text_is_exactly_zero(self, monkeypatch):
        monkeypatch.setattr(tokens, "get_encoding", lambda model: None)
        assert tokens.count_tokens("", "gpt-5-nano") == (0, False)

    def test_missing_tiktoken_means_the_estimate(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_tiktoken(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_tiktoken)
        assert tokens.get_encoding("gpt-5-nano") is None
        assert tokens.count_tokens("abcd", "gpt-5-nano") == (1, True)


class TestOfflineSafety:
    def test_pytest_counts_as_offline(self):
        assert tokens._offline() is True

    def test_an_uncached_encoding_is_never_downloaded_offline(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))  # empty cache
        assert tokens.encoding_cached(tokens.O200K) is False
        assert tokens.get_encoding("gpt-5-nano") is None
        assert tokens.count_tokens("hello world", "gpt-5-nano")[1] is True

    def test_explicit_env_switch(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("SYNAPSE_TOKENIZER_OFFLINE", "1")
        assert tokens._offline() is True
        monkeypatch.setenv("SYNAPSE_TOKENIZER_OFFLINE", "0")
        assert tokens._offline() is False


@pytest.mark.skipif(
    not tokens.encoding_cached(tokens.O200K), reason="o200k_base not in tiktoken's cache"
)
def test_real_o200k_counts_are_exact():
    count, estimated = tokens.count_tokens("hello world", "gpt-5-nano")
    assert (count, estimated) == (2, False)
    assert tokens.tokenizer_label("gpt-4o-mini") == "tiktoken:o200k_base"
    # Special tokens are counted as text, not rejected.
    assert tokens.count_tokens("<|endoftext|>", "gpt-4o")[1] is False
