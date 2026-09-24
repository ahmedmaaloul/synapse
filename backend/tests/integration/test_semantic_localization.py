# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Calibration check: semantic localization with the REAL default embedder.

The unit tests use a keyword embedder, so they cannot tell whether
``PROCEDURAL_SEMANTIC_THRESHOLD`` suits the embedder Synapse ships with.
fastembed's ``BAAI/bge-small-en-v1.5`` compresses cosines into a high band:
at the old 0.5 floor, "ls -la" or "SELECT * FROM users" localized onto the
bundled prior's ``answer`` node, so the full-graph fallback almost never
fired.

The string sets are NOT defined here: they are imported from
``scripts/calibrate_semantic_threshold.py``, the script whose output
docs/procedural-graphs.md quotes. The docs, the script and this test
therefore measure the same strings. Rerun the script after changing the
embedder or the prior, and update the docs from its output:

    python -m scripts.calibrate_semantic_threshold

Two groups of tests:

* hermetic ones (no model), which run in the unit suite: the sets are valid
  for the prior, and the script's arithmetic is right;
* real-embedder ones, skipped unless ``SYNAPSE_IT=1``. They pin what the docs
  claim at the configured threshold, through ``guide()`` itself: every
  unrelated string falls back to the full graph, ``guide()`` places each
  paraphrase exactly where the script says, and placements are mostly right.
  No LLM and no database: the model runs locally (downloaded once into
  fastembed's cache if absent).

    SYNAPSE_IT=1 pytest tests/integration/test_semantic_localization.py -v
"""

from __future__ import annotations

import os

import pytest

from scripts import calibrate_semantic_threshold as calibrate
from scripts.calibrate_semantic_threshold import (
    PARAPHRASES,
    PRIOR,
    UNRELATED,
    Measurement,
    check_sets,
    measure,
    outcome_at,
    report,
    step_text,
    thresholds_with,
)

needs_real_embedder = pytest.mark.skipif(
    os.environ.get("SYNAPSE_IT") != "1",
    reason="integration test — set SYNAPSE_IT=1 to run the real fastembed model",
)

QUERY = "Who designed the Analytical Engine?"


def _prior():
    from app.services.procedural_store import load_prior

    return load_prior(PRIOR)


# ── Hermetic: the sets and the script (unit suite) ───


def test_both_sets_are_large_enough_to_calibrate_on():
    assert len(UNRELATED) >= 40
    assert len(PARAPHRASES) >= 40
    assert {node for _, _, node in PARAPHRASES} >= {
        "search_entities",
        "neighbors",
        "read_sources",
        "search_passages",
        "find_path",
        "answer",
    }


def test_every_string_exercises_the_semantic_step_alone():
    # A string that matches a node by name (exact / normalized) never reaches
    # the semantic step, so it would measure nothing; an intended node that is
    # not a semantic candidate (Start, a terminal) could never be reached.
    assert check_sets(_prior()) == []


def test_check_sets_reports_a_name_match_a_duplicate_and_an_impossible_target():
    problems = check_sets(
        _prior(),
        unrelated=[("Search_Entities(query='x')", ""), ("x", "")],
        paraphrases=[("x", "", "End")],
    )
    assert any("already localizes by name" in p and "search_entities" in p for p in problems)
    assert any("duplicate action 'x'" in p for p in problems)
    assert any("'End' is not a semantic candidate" in p for p in problems)


class _KeywordEmbedder:
    VOCAB = ("passage", "entity", "relation", "submit", "path")

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.queries: list[str] = []

    def _vec(self, text: str) -> list[float]:
        return [float(text.lower().count(word)) for word in self.VOCAB]

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return self._vec(text)


def test_measure_scores_like_guide_does():
    embedder = _KeywordEmbedder()
    unrelated, paraphrases = measure(
        _prior(),
        embedder,
        unrelated=[("zzz", "")],
        paraphrases=[("submit_final", "x" * 900, "answer")],
    )
    # Candidates are every node but Start and the terminals, as "<id> <description>".
    assert not any(d.startswith(("Start ", "End ")) for d in embedder.documents)
    assert any(d.startswith("answer Submit the final short answer") for d in embedder.documents)
    # The embedded step text clips the observation exactly as guide() does.
    assert embedder.queries[1] == step_text("submit_final", "x" * 900)
    assert len(embedder.queries[1]) == len("submit_final ") + 500
    (paraphrase,) = paraphrases
    assert (paraphrase.node, paraphrase.right_node) == ("answer", True)
    assert paraphrase.score == pytest.approx(1.0)
    (garbage,) = unrelated
    assert garbage.intended is None and garbage.right_node is False
    assert garbage.score == 0.0  # a zero vector: cosine 0 against everything


def _m(score: float, node: str = "answer", intended: str | None = "answer") -> Measurement:
    return Measurement("a", "", intended, node, score)


def test_outcome_at_splits_right_wrong_and_missed():
    unrelated = [_m(0.60, intended=None), _m(0.70, intended=None)]
    paraphrases = [_m(0.80), _m(0.75, node="neighbors"), _m(0.65)]
    at = outcome_at(0.70, unrelated, paraphrases)
    # The rule is score >= threshold: 0.70 localizes at 0.70.
    assert (at.unrelated_full, at.right, at.wrong, at.missed) == (1, 1, 1, 1)
    assert at.accuracy == pytest.approx(2 / 5)
    low = outcome_at(0.50, unrelated, paraphrases)
    assert (low.unrelated_full, low.right, low.wrong, low.missed) == (0, 2, 1, 0)


def test_the_configured_threshold_is_always_in_the_table():
    assert thresholds_with(0.72) == [0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.8, 0.85]
    assert thresholds_with(0.75) == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


def test_report_marks_the_configured_row_and_lists_wrong_placements():
    lines = report(
        [_m(0.60, intended=None)],
        [_m(0.80), _m(0.74, node="find_path", intended="neighbors")],
        configured=0.72,
        embedder="fake",
        candidates=9,
    )
    marked = [line for line in lines if "◀ configured" in line]
    assert len(marked) == 1 and marked[0].strip().startswith("0.72")
    assert any("→ find_path (meant neighbors)" in line for line in lines)


def test_main_prints_the_report_and_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(calibrate, "get_embeddings", lambda settings=None: _KeywordEmbedder())
    assert calibrate.main([]) == 0
    out = capsys.readouterr().out
    assert f"{len(UNRELATED)} unrelated · {len(PARAPHRASES)} paraphrased tool calls" in out
    assert "◀ configured" in out


def test_main_exits_two_when_the_embedder_cannot_load(monkeypatch, capsys):
    def broken(settings=None):
        raise RuntimeError("no model here")

    monkeypatch.setattr(calibrate, "get_embeddings", broken)
    assert calibrate.main([]) == 2
    assert "no model here" in capsys.readouterr().err


# ── Real embedder (SYNAPSE_IT=1) ─────────────────────


@pytest.fixture(scope="module")
def fastembed_embeddings():
    """The real default embedder, loaded once for the module."""
    pytest.importorskip("fastembed")
    from app.config import get_settings
    from app.services.llm_provider import get_embeddings

    settings = get_settings().model_copy(update={"embedding_provider": "fastembed"})
    return get_embeddings(settings)


@pytest.fixture
def real_embedder(monkeypatch, fastembed_embeddings):
    from app.services import procedural_guidance as pgd

    monkeypatch.setattr(pgd, "get_embeddings", lambda: fastembed_embeddings)
    pgd.clear_caches()
    yield fastembed_embeddings
    pgd.clear_caches()


@pytest.fixture(scope="module")
def measured(fastembed_embeddings):
    """The script's measurement of both sets: {action: paraphrase}, unrelated, paraphrases."""
    unrelated, paraphrases = measure(_prior(), fastembed_embeddings)
    return {m.action: m for m in paraphrases}, unrelated, paraphrases


async def _guide(action: str, observation: str) -> dict:
    from app.services.procedural_guidance import guide

    return await guide(
        PRIOR,
        query=QUERY,
        trajectory=[{"action": action, "observation": observation}],
        mode="raw",
        graph=_prior(),
    )


@pytest.mark.integration
@needs_real_embedder
@pytest.mark.parametrize(("action", "observation"), UNRELATED)
async def test_unrelated_text_falls_back_to_the_full_graph(real_embedder, action, observation):
    result = await _guide(action, observation)
    assert result["localization"] == "none", (
        f"{action!r} localized onto {result['active_node']} at {result['localization_score']}"
    )
    assert result["scope"] == "full"


@pytest.mark.integration
@needs_real_embedder
@pytest.mark.parametrize(("action", "observation", "node"), PARAPHRASES)
async def test_guide_places_each_paraphrase_where_the_script_says(
    real_embedder, measured, action, observation, node
):
    from app.config import get_settings

    threshold = get_settings().procedural_semantic_threshold
    expected = measured[0][action]
    assert expected.intended == node
    result = await _guide(action, observation)
    if expected.score >= threshold:
        assert (result["active_node"], result["localization"]) == (expected.node, "semantic")
        assert result["localization_score"] == pytest.approx(expected.score, abs=1e-3)
    else:
        assert (result["localization"], result["scope"]) == ("none", "full")


@pytest.mark.integration
@needs_real_embedder
def test_the_configured_threshold_keeps_placements_mostly_right(measured):
    """What docs/procedural-graphs.md claims about the configured floor.

    Every unrelated string stays below it (the full-graph fallback works), at
    least 90 % of the paraphrases it places land on their intended node, and
    it still places at least half of them (the step is not dead weight).
    """
    from app.config import get_settings

    _, unrelated, paraphrases = measured
    at = outcome_at(get_settings().procedural_semantic_threshold, unrelated, paraphrases)
    assert at.unrelated_full == len(unrelated)
    placed = at.right + at.wrong
    assert placed >= len(paraphrases) / 2
    assert at.right >= 0.9 * placed
