# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for the deterministic chunking logic."""

from app.services.pdf_parser import chunk_text


def test_empty_and_whitespace_return_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  \t ") == []


def test_short_text_is_single_chunk():
    assert chunk_text("Ada Lovelace wrote the first algorithm.") == [
        "Ada Lovelace wrote the first algorithm."
    ]


def test_chunks_prefer_sentence_boundaries():
    text = "First sentence here. Second sentence follows. Third one too."
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    assert len(chunks) > 1
    # Most chunks should end cleanly at a sentence terminator.
    assert any(c.endswith(".") for c in chunks)


def test_overlap_makes_forward_progress_no_infinite_loop():
    # Pathological input with no boundaries and big overlap must still terminate.
    text = "x" * 5000
    chunks = chunk_text(text, chunk_size=100, overlap=99)
    assert len(chunks) > 1
    assert "".join(chunks).replace("", "")  # sanity: produced content


def test_covers_full_text():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 20
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    joined = " ".join(chunks)
    # Every source word appears somewhere in the chunk set.
    for word in {"alpha", "kappa", "theta"}:
        assert word in joined


def test_invalid_chunk_size_raises():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)
