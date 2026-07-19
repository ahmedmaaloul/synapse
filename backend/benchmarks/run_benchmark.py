# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — GraphRAG vs. vector RAG benchmark

Answers the only question that matters for a GraphRAG project: *does the graph
actually buy you anything?* Five retrieval systems — reported as seven rows,
because two of them are also shown at a budget matched to the graph's — are
asked the same 14 multi-hop questions over the same corpus, with the same
embedding model, and are scored on the same rule.

  A. Passage vector RAG — the naive baseline
      Embed each passage of ``dataset.json`` whole (they are already
      paragraph-sized, which is the best case for chunking), embed the question,
      take the top-k passages by cosine similarity, concatenate them.

  A′. Passage vector RAG, budget-matched
      The same system at the smallest k whose mean context reaches the headline
      system's. Computed from the sweep, never chosen by hand.

  B. Entity vector RAG — a standalone graph-free system
      Embed each entity exactly as ``graph_builder._embed_entities`` does
      (``name — description``), take the top-k entities by cosine similarity,
      and format them exactly as ``chat_engine._expand_and_format`` does **minus
      the relationship lines**. Same retrieval granularity as GraphRAG, same k,
      no edges. Without this row the headline confounds two changes at once —
      graph traversal *and* finer retrieval units — and a reader is entitled to
      dismiss the result as "you compared 4 chunks against 8 smaller chunks".

  B′. GraphRAG seeds with the graph structure stripped — the *exact* ablation
      B still differs from C in two ways (structure, and cosine vs. Synapse's
      hybrid vector+full-text seeding). B′ removes the second: it takes C's own
      context and deletes, by literal text surgery, everything the edges
      produced — the per-entity ``Relationships:`` lines **and** the rendered
      ``Reasoning paths:`` section. What is left is C's own seed entities, in
      C's own order, with their names, types and descriptions. C − B′ is
      therefore the contribution of the *relationships*, and of nothing else;
      it is not the contribution of the relationship lines alone, because the
      reasoning paths are edge-derived too and go with them.

  B″. Entity vector RAG, budget-matched to C
      The edgeless entity system given as much context as C. Reported because
      the ablation is only meaningful if the edgeless side is not simply being
      starved of characters; the sweep runs all the way to k = *every entity in
      the graph*, which is the absolute ceiling of what an edgeless entity
      retriever can ever read.

  C. Synapse GraphRAG, graph only
      Load the *pre-extracted* entities and relationships from ``dataset.json``
      into Neo4j (no LLM call — the extraction ships with the dataset so the
      benchmark is deterministic), then call the production retrieval path,
      ``app.services.chat_engine.retrieve_subgraph``, untouched, with no source
      chunks stored. This is what Synapse was before source-chunk retrieval.

  D. Synapse GraphRAG + source chunks — the shipped system
      Identical to C, except the corpus passages are also stored as ``:Chunk``
      text units linked to the entities extracted from them, so retrieval
      returns the graph structure **and** the prose behind it.

      Stated plainly, because it is the first thing a skeptic should notice:
      D's semantic excerpt channel *is* system A. ``chunk_store.search_chunks``
      does a top-``chunk_top_k`` cosine search over exactly the passages A
      ranks, with the same embedder. D beating A is therefore close to
      tautological and is not evidence of anything. The informative comparisons
      for D are **D vs. A′** (same character budget) and **D vs. C** (what the
      prose adds to the graph); the report measures and prints the actual
      overlap between D's excerpts and A's top-k so the reader can size the
      tautology instead of taking a word for it.

Scoring measures RETRIEVAL, not generation: for each question the dataset lists
the ``required_facts`` — entity names that must be present for the question to
be answerable at all. A fact counts as retrieved when its name occurs in the
retrieved context. *The identical rule is applied to every system*, so nothing
hinges on how an LLM happens to phrase an answer, and the whole run is free,
offline and reproducible.

  • Fact Recall        — mean fraction of required facts retrieved
  • Full-Coverage Rate — fraction of questions where ALL required facts were
                         retrieved. This is the number that separates multi-hop
                         traversal from top-k ranking: a partial answer to a
                         multi-hop question is a wrong answer.
  • Mean context size  — characters handed to the generator, so a win cannot be
                         bought simply by returning more text.

Because more context is easier to score well on, the report never states a bare
"we win". Every sweep runs to the point where the graph-free system has read the
*entire* corpus available to it, so a budget match is always reachable and the
comparison can never hide behind a truncated sweep.

Run it (needs Neo4j; no API key, no network):

    docker compose up -d neo4j
    cd backend && EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7687 \\
        python -m benchmarks.run_benchmark

WARNING: seeding replaces the ``:Entity`` / ``:Community`` / ``:Chunk`` contents
of the target Neo4j database. Re-seed your demo afterwards with
``make demo-local``.

Exit codes: 0 success · 1 invalid dataset · 2 Neo4j unreachable · 3 the run
produced a result the harness cannot honestly report (e.g. the graph-only row
was contaminated by leftover chunks).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DATASET_PATH = Path(__file__).with_name("dataset.json")
RESULTS_PATH = Path(__file__).with_name("results.md")

#: Passages returned by the naive baseline. The corpus is 34 passages (18 core +
#: 16 distractor), so k=4 reads ~12% of it — a conventional production setting,
#: not a handicap. For the other side of the comparison: GraphRAG seeds
#: ``DEFAULT_GRAPH_K`` = 8 entities out of 79 (~10%) and additionally
#: materialises each seed's immediate neighbours, which is why it hands the
#: generator more characters. Both budgets are printed in the report, together
#: with a budget-matched baseline row and a full k sweep, so no claim rests on
#: this particular choice of k.
DEFAULT_BASELINE_K = 4
#: Seed entities for GraphRAG. 8 is the production default of
#: ``retrieve_subgraph``; the harness does not tune it for the benchmark. The
#: entity-vector ablation uses the same k so the only difference is the edges.
DEFAULT_GRAPH_K = 8
#: k values reported in the honesty sweep. Each sweep additionally ends at
#: ``k = the whole corpus that system can retrieve from`` (see :func:`sweep_ks`),
#: because a budget match that the sweep cannot reach is not a budget match: the
#: earlier version of this file stopped at k=12, where the entity system read
#: 66% of GraphRAG's context, and then compared the two as if they were equal.
SWEEP_KS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)

CLEAR_QUERIES = (
    "MATCH (n:Entity) DETACH DELETE n",
    "MATCH (c:Community) DETACH DELETE c",
    # Chunks are cleared too: a leftover :Chunk from `make demo-local` would be
    # retrieved by the production path and silently inflate every GraphRAG row.
    "MATCH (c:Chunk) DETACH DELETE c",
)

#: Section markers emitted by ``chat_engine``. Duplicated here rather than
#: imported so the pure scoring code stays importable without LangChain; a test
#: asserts they still match the engine's own constants.
PATHS_HEADING = "Reasoning paths:"
SOURCES_HEADING = "Source excerpts:"
RELATIONSHIPS_LINE = "  Relationships:"
EDGE_LINE_PREFIX = "  → "


class DatasetError(RuntimeError):
    """Raised when ``dataset.json`` is missing or structurally invalid."""


class IntegrityError(RuntimeError):
    """Raised when a run produced numbers the harness must not report."""


# ── Dataset ──────────────────────────────────────────────────────────────────
def load_dataset(path: Path = DATASET_PATH) -> dict:
    """Read and lightly validate the benchmark dataset."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise DatasetError(f"Could not read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise DatasetError(f"{path} is not valid JSON: {e}") from e

    for key in ("passages", "entities", "relationships", "questions"):
        if not isinstance(payload.get(key), list) or not payload[key]:
            raise DatasetError(f"{path}: '{key}' must be a non-empty list")
    return payload


def sweep_ks(total: int) -> tuple[int, ...]:
    """Sweep points for a system that can retrieve at most ``total`` units.

    Always ends at ``total`` — the point where the system has read *everything
    it has*. That row is what makes a budget match guaranteed to exist: whatever
    context GraphRAG returns, the report can say either "a cheaper graph-free
    setting reaches it" or "not even reading the entire corpus does".
    """
    ks = [k for k in SWEEP_KS if k < total]
    ks.append(total)
    return tuple(ks)


# ── Lexical-leak detection (fairness rule R1, enforced in tests) ─────────────
#: Function words only. Everything else counts as *distinctive* content, because
#: the words a retriever actually keys on ("machine", "network", "database") are
#: precisely the ones a hand-written description must not echo back at the
#: question it is supposed to be reachable from only by traversal.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here
    is are was were be been being am
    do does did done doing
    have has had having
    can could will would shall should may might must
    of in on at by for with without from to into onto over under about
    as it its he she they them him his her their our your my
    which who whom whose what when where why how
    not no nor so such own same too very just also
    one two both each other another any all some none
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+")

#: Shared-prefix length at which two stems are treated as the same word family.
#: Four is the shortest prefix that is still a word in its own right for the
#: cases this rule exists to catch ("dataset"/"database", "back"/"backpropagation",
#: "statistical"/"states"). At five, all three slip through; at three it starts
#: flagging genuinely unrelated words. It is deliberately over-eager: a false
#: positive costs one dataset rewrite, a false negative costs the benchmark's
#: credibility.
PREFIX_FAMILY_LEN = 4


def _stem(word: str) -> str:
    """Crude suffix stripper so 'walks'/'walk' and 'designed'/'design' collide.

    Suffix stripping alone is not enough for R1 — it cannot see that "dataset"
    and "database" are the same lexical family — so it is paired with
    :func:`same_family`, which compares stems by shared prefix.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ations", "ation", "ings", "ing", "edly", "ed", "es"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    # Drop a silent -e last, so "popularise"/"popularised" and
    # "machine"/"machines" land on the same stem from either direction.
    if len(word) > 4 and word.endswith("e"):
        return word[:-1]
    return word


def content_words(text: str) -> set[str]:
    """Stemmed, stopword-filtered content words of ``text``.

    The unit of fairness rule R1: a question and the description of any of its
    non-anchor required facts must share no word *family* (see :func:`leaks`).
    """
    return {
        _stem(token)
        for token in _WORD.findall((text or "").lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def same_family(a: str, b: str) -> bool:
    """True when two stems share a prefix long enough to be one word family.

    Suffix stripping only ever collides words that differ at the *end*
    ("train"/"training"). The leak class it misses is the one that differs in
    the middle or the tail — "dataset" vs. "database", "back" vs.
    "backpropagation" — where a subword-tokenising embedder plainly does see the
    shared stem even though an exact-match check does not.
    """
    if a == b:
        return True
    shared = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        shared += 1
    return shared >= PREFIX_FAMILY_LEN


def leaks(question: str, text: str) -> set[str]:
    """Content words of ``question`` that share a word family with ``text``.

    Returns the *question-side* words, because those are what a human has to
    rewrite when the check fires.
    """
    asked = content_words(question)
    other = content_words(text)
    return {a for a in asked for b in other if same_family(a, b)}


# ── Pure scoring primitives (unit-tested in tests/test_benchmark.py) ─────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 if either is null."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_similarity(query: list[float], documents: list[list[float]]) -> list[int]:
    """Document indices ordered by descending cosine similarity to ``query``.

    Ties break on the lower index so a run is byte-for-byte reproducible.
    """
    scores = [cosine_similarity(query, doc) for doc in documents]
    return sorted(range(len(documents)), key=lambda i: (-scores[i], i))


def facts_found(context: str, facts: list[str]) -> list[str]:
    """Which ``facts`` (entity names) occur in ``context``, case-insensitively.

    The same rule is applied to every system. The dataset tests guarantee that
    no entity name is a substring of another and that no name ever matches
    inside a longer word, so a match is never accidental.
    """
    haystack = (context or "").lower()
    return [f for f in facts if f.lower() in haystack]


def fact_recall(found: list[str], required: list[str]) -> float:
    """Fraction of the required facts that were retrieved (0.0 when none required)."""
    if not required:
        return 0.0
    wanted = {r.lower() for r in required}
    hits = {f.lower() for f in found} & wanted
    return len(hits) / len(wanted)


@dataclass
class QuestionResult:
    """One question, as answered by one system."""

    qid: str
    question: str
    required: list[str]
    found: list[str]
    context_chars: int
    detail: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def missing(self) -> list[str]:
        found = {f.lower() for f in self.found}
        return [r for r in self.required if r.lower() not in found]

    @property
    def recall(self) -> float:
        return fact_recall(self.found, self.required)

    @property
    def complete(self) -> bool:
        return not self.missing


def aggregate(results: list[QuestionResult]) -> dict:
    """Roll per-question results into the three headline metrics."""
    if not results:
        return {"fact_recall": 0.0, "full_coverage": 0.0, "mean_context_chars": 0.0, "n": 0}
    n = len(results)
    return {
        "fact_recall": sum(r.recall for r in results) / n,
        "full_coverage": sum(1.0 for r in results if r.complete) / n,
        "mean_context_chars": sum(r.context_chars for r in results) / n,
        "n": n,
    }


@dataclass
class Run:
    """One system's results, with its aggregate computed once."""

    key: str
    label: str
    results: list[QuestionResult]

    @property
    def agg(self) -> dict:
        return aggregate(self.results)


# ── System A: passage vector RAG (the naive baseline) ────────────────────────
def evaluate_baseline(
    dataset: dict,
    rankings: dict[str, list[int]],
    k: int,
) -> list[QuestionResult]:
    """Score top-k passage retrieval. Pure: rankings are supplied by the caller."""
    passages = dataset["passages"]
    results: list[QuestionResult] = []
    for question in dataset["questions"]:
        chosen = rankings[question["id"]][:k]
        context = "\n\n".join(passages[i]["text"] for i in chosen)
        results.append(
            QuestionResult(
                qid=question["id"],
                question=question["question"],
                required=list(question["required_facts"]),
                found=facts_found(context, question["required_facts"]),
                context_chars=len(context),
                detail=", ".join(passages[i]["id"] for i in chosen),
                extras={"passages": list(chosen)},
            )
        )
    return results


# ── System B: entity vector RAG (same granularity as GraphRAG, no edges) ─────
def entity_document(entity: dict) -> str:
    """The text GraphRAG itself embeds for an entity (``graph_builder._embed_entities``)."""
    return f"{entity.get('name', '')} — {entity.get('description', '')}".strip(" —")


def entity_context_block(entity: dict) -> str:
    """``chat_engine._expand_and_format``'s per-entity block, minus the edges.

    Byte-identical to what GraphRAG emits for a seed entity with no relationships,
    so the two systems differ only in whether the graph is walked.
    """
    info = f"Entity: {entity.get('name')} (Type: {entity.get('type')})"
    if entity.get("description"):
        info += f"\n  Description: {entity['description']}"
    return info


def evaluate_entity_rag(
    dataset: dict,
    rankings: dict[str, list[int]],
    k: int,
) -> list[QuestionResult]:
    """Score top-k *entity* retrieval with no graph traversal."""
    entities = dataset["entities"]
    results: list[QuestionResult] = []
    for question in dataset["questions"]:
        chosen = rankings[question["id"]][:k]
        context = "\n\n".join(entity_context_block(entities[i]) for i in chosen)
        results.append(
            QuestionResult(
                qid=question["id"],
                question=question["question"],
                required=list(question["required_facts"]),
                found=facts_found(context, question["required_facts"]),
                context_chars=len(context),
                detail=", ".join(entities[i]["name"] for i in chosen[:4]),
            )
        )
    return results


async def embed_corpus(
    dataset: dict,
) -> tuple[list[list[float]], list[list[float]], dict[str, list[float]]]:
    """Embed every passage, every entity and every question with one embedder.

    This is the *same* model GraphRAG uses to seed its entity search, so no
    system gets a better encoder than another.
    """
    from app.services.llm_provider import get_embeddings

    embeddings = get_embeddings()
    passage_texts = [p["text"] for p in dataset["passages"]]
    entity_texts = [entity_document(e) for e in dataset["entities"]]
    question_texts = [q["question"] for q in dataset["questions"]]

    passage_vectors = await asyncio.to_thread(embeddings.embed_documents, passage_texts)
    entity_vectors = await asyncio.to_thread(embeddings.embed_documents, entity_texts)
    question_vectors = await asyncio.to_thread(embeddings.embed_documents, question_texts)
    by_qid = {
        q["id"]: vec
        for q, vec in zip(dataset["questions"], question_vectors, strict=True)
    }
    return passage_vectors, entity_vectors, by_qid


# ── Systems C and D: Synapse GraphRAG over the real retrieval path ───────────
async def seed_graph(dataset: dict, *, echo: bool = True) -> dict:
    """Load the pre-extracted graph into Neo4j via the production write path."""
    from app.neo4j_driver import execute_query
    from app.services import graph_builder
    from app.services.graph_schema import ensure_schema

    def say(message: str) -> None:
        if echo:
            print(message)

    say("🧱 Ensuring schema (constraint, full-text index, vector indexes) …")
    await ensure_schema()

    say("🧨 Clearing existing :Entity / :Community / :Chunk nodes …")
    for query in CLEAR_QUERIES:
        await execute_query(query)

    entities = dataset["entities"]
    say(f"🧠 Embedding {len(entities)} entities …")
    embeddings_by_name = await graph_builder._embed_entities(entities)

    say("✍️  Writing nodes and relationships …")
    nodes = await graph_builder._write_entities(
        entities, dataset.get("document_name", "benchmark"), embeddings_by_name
    )
    edges = await graph_builder._write_relationships(dataset["relationships"])
    return {"nodes": nodes, "edges": edges, "embedded": len(embeddings_by_name)}


async def seed_chunks(dataset: dict) -> dict:
    """Store every passage as a ``:Chunk`` text unit linked to its entities.

    This is the production write path (``chunk_store.store_chunks``) fed the
    dataset's own passages, so system D retrieves the *same* prose system A
    ranks — which is exactly the point: D is asked to justify itself against a
    baseline that reads identical text.

    The chunk's ``index`` is its position in ``dataset["passages"]``, which lets
    the report measure how much of D's excerpt channel is literally A's top-k.
    """
    from app.services import chunk_store

    passages = dataset["passages"]
    return await chunk_store.store_chunks(
        dataset.get("document_name", "benchmark"),
        [p["text"] for p in passages],
        [list(p.get("entities") or []) for p in passages],
    )


async def evaluate_graphrag(dataset: dict, k: int) -> list[QuestionResult]:
    """Score Synapse's own retrieval, calling ``retrieve_subgraph`` unmodified."""
    from app.services.chat_engine import retrieve_subgraph

    results: list[QuestionResult] = []
    for question in dataset["questions"]:
        retrieval = await retrieve_subgraph(question["question"], k=k)
        context, citations = retrieval
        sources = list(getattr(retrieval, "sources", []) or [])
        results.append(
            QuestionResult(
                qid=question["id"],
                question=question["question"],
                required=list(question["required_facts"]),
                found=facts_found(context, question["required_facts"]),
                context_chars=len(context or ""),
                detail=", ".join(c["name"] for c in citations[:4]),
                extras={
                    "mode": getattr(retrieval, "mode", "local"),
                    "paths": len(getattr(retrieval, "paths", []) or []),
                    "seeds": [c["name"] for c in citations if c.get("kind") == "entity"],
                    # The raw context is kept so ``strip_edges`` can perform its
                    # surgery on the real thing rather than on a reconstruction.
                    "context": context or "",
                    "chunks": [c.get("index") for c in sources],
                },
            )
        )
    return results


def strip_graph_structure(context: str) -> str:
    """Delete every edge-derived part of a GraphRAG context, and nothing else.

    Removed:
      • each entity block's ``Relationships:`` header and its ``→`` lines, and
      • the whole ``Reasoning paths:`` section (paths are built from edges too —
        a previous version of this file claimed the relationship lines were "the
        only thing taken away", which was false).

    Kept: every entity's name, type and description, in the order GraphRAG
    ranked them. Any ``Source excerpts:`` section is dropped as well, so the
    function is total; in practice the ablation is only ever run against the
    graph-only system, which has none.
    """
    head, _sep, _tail = (context or "").partition(SOURCES_HEADING)
    blocks: list[str] = []
    for block in head.split("\n\n"):
        if block.startswith(PATHS_HEADING):
            continue
        kept = [
            line
            for line in block.split("\n")
            if line != RELATIONSHIPS_LINE and not line.startswith(EDGE_LINE_PREFIX)
        ]
        text = "\n".join(kept).rstrip()
        if text.strip():
            blocks.append(text)
    return "\n\n".join(blocks)


def contaminated(results: list[QuestionResult]) -> list[str]:
    """Question ids whose *graph-only* context already carries source excerpts.

    The graph-only row is only honest if the chunk store was empty when it ran.
    A leftover ``:Chunk`` from ``make demo-local`` would be retrieved by the
    production path and silently inflate row C, which is why the harness clears
    ``:Chunk`` before seeding and checks the result afterwards rather than
    trusting itself.
    """
    return [r.qid for r in results if SOURCES_HEADING in (r.extras.get("context") or "")]


def strip_edges(graph: list[QuestionResult]) -> list[QuestionResult]:
    """B′ — GraphRAG's own context with everything the edges produced deleted.

    System B ranks entities by plain cosine, so B vs. C differs in two things:
    the graph structure *and* the seeding (Synapse seeds with hybrid vector +
    full-text search). This variant removes the second: identical seeds,
    identical order, identical per-entity blocks. What separates B′ from C is
    the relationships — the neighbour lines and the reasoning paths they
    generate — exactly and only.
    """
    stripped: list[QuestionResult] = []
    for result in graph:
        context = strip_graph_structure(result.extras.get("context", ""))
        seeds = list(result.extras.get("seeds", []))
        stripped.append(
            QuestionResult(
                qid=result.qid,
                question=result.question,
                required=list(result.required),
                found=facts_found(context, result.required),
                context_chars=len(context),
                detail=", ".join(seeds[:4]),
            )
        )
    return stripped


# ── Reporting ────────────────────────────────────────────────────────────────
METRICS: tuple[tuple[str, str], ...] = (
    ("Fact Recall", "fact_recall"),
    ("Full-Coverage Rate", "full_coverage"),
)

_LABEL_W = 52


def _row(label: str, agg: dict) -> str:
    return (
        f"  {label:<{_LABEL_W}}{agg['fact_recall']:>10.1%}"
        f"{agg['full_coverage']:>16.1%}{agg['mean_context_chars']:>15,.0f}"
    )


def smallest_k_reaching(
    sweep: list[tuple[int, dict]], target: float, key: str
) -> tuple[int, dict] | None:
    """Smallest baseline k whose ``key`` matches or beats ``target``.

    This is the question a skeptical reader actually asks: *how cheap a plain
    vector baseline do I need before your graph stops being interesting?*
    """
    for k, agg in sweep:
        if agg[key] >= target - 1e-9:
            return k, agg
    return None


def context_matched(sweep: list[tuple[int, dict]], target_chars: float) -> tuple[int, dict] | None:
    """The smallest sweep point whose mean context is at least ``target_chars``.

    GraphRAG returns more text than a top-4 baseline, and more text is easier to
    score well on. This finds the k at which the graph-free system is given the
    *same* budget, which is the only genuinely apples-to-apples row in the
    report — so it is computed and printed whether or not it flatters GraphRAG.
    """
    for k, agg in sweep:
        if agg["mean_context_chars"] >= target_chars:
            return k, agg
    return None


def verdict(baseline: dict, graph: dict, sweep: list[tuple[int, dict]]) -> list[str]:
    """Per-metric result, each one immediately qualified by the parity k.

    A bare "GraphRAG wins" is not a claim anybody should accept: the two systems
    were handed different amounts of text. So every metric line is followed by
    the smallest baseline k that matches or beats GraphRAG on that metric, and
    what that k costs in characters relative to GraphRAG's own context.
    """
    lines: list[str] = []
    for label, key in METRICS:
        delta = graph[key] - baseline[key]
        if abs(delta) < 1e-9:
            lines.append(f"{label}: TIE at the configured k ({graph[key]:.1%} for both).")
        elif delta > 0:
            lines.append(
                f"{label}: GraphRAG ahead at the configured k — {graph[key]:.1%} vs "
                f"{baseline[key]:.1%} ({delta * 100:+.1f} pp)."
            )
        else:
            lines.append(
                f"{label}: passage vector RAG ahead at the configured k — "
                f"{baseline[key]:.1%} vs {graph[key]:.1%} ({delta * 100:+.1f} pp for GraphRAG)."
            )
        lines.append(f"  {_parity_line(key, graph, sweep)}")

    ratio = (
        graph["mean_context_chars"] / baseline["mean_context_chars"]
        if baseline["mean_context_chars"]
        else 0.0
    )
    lines.append(
        f"Context size: GraphRAG hands the generator {ratio:.2f}x the characters "
        f"the baseline does ({graph['mean_context_chars']:,.0f} vs "
        f"{baseline['mean_context_chars']:,.0f})."
    )
    return lines


def _parity_line(key: str, graph: dict, sweep: list[tuple[int, dict]]) -> str:
    """One sentence: the cheapest plain baseline that erases GraphRAG's edge."""
    match = smallest_k_reaching(sweep, graph[key], key)
    if match is None:
        best_k, best = max(sweep, key=lambda item: item[1][key]) if sweep else (0, {key: 0.0})
        return (
            f"→ No baseline k up to {sweep[-1][0] if sweep else 0} matches GraphRAG's "
            f"{graph[key]:.1%} (best is {best[key]:.1%} at top-{best_k}), so on this "
            f"metric the graph is not reproducible by reading more passages."
        )
    k, agg = match
    share = (
        agg["mean_context_chars"] / graph["mean_context_chars"]
        if graph["mean_context_chars"]
        else 0.0
    )
    verb = "matches" if abs(agg[key] - graph[key]) < 1e-9 else "beats"
    return (
        f"→ A plain passage baseline {verb} it at top-{k} ({agg[key]:.1%}), reading "
        f"{agg['mean_context_chars']:,.0f} chars = {share:.0%} of GraphRAG's context."
    )


def headline(graph: dict, sweep: list[tuple[int, dict]]) -> list[str]:
    """The one claim this benchmark is allowed to make, stated precisely.

    Computed, never typed: for each metric, GraphRAG "wins at equal-or-smaller
    context" only if no baseline k in the sweep reaches the same score without
    reading at least as many characters. Metrics that fail that test are named.
    """
    won: list[str] = []
    lost: list[tuple[str, int, dict, float]] = []
    for label, key in METRICS:
        match = smallest_k_reaching(sweep, graph[key], key)
        if match is None:
            won.append(label)
            continue
        k, agg = match
        share = (
            agg["mean_context_chars"] / graph["mean_context_chars"]
            if graph["mean_context_chars"]
            else 0.0
        )
        if share < 1.0:
            lost.append((label, k, agg, share))
        else:
            won.append(label)

    scores = ", ".join(f"{label} {graph[key]:.1%}" for label, key in METRICS)
    lines = [
        f"HEADLINE (computed, not asserted): the shipped system scores {scores} on "
        f"{graph['mean_context_chars']:,.0f} chars of retrieved context."
    ]
    if won and not lost:
        lines.append(
            "  On every metric, no baseline k in the sweep reaches that score without "
            "reading at least as much text — the advantage survives a budget match."
        )
    elif lost and not won:
        detail = "; ".join(
            f"{label} at top-{k} using {share:.0%} of the context" for label, k, _, share in lost
        )
        lines.append(
            "  A plain passage baseline matches it on EVERY metric while reading LESS "
            f"text ({detail}). On this corpus the graph buys no context saving; what it "
            "buys has to be argued from the ablation row below, not from these numbers."
        )
    else:
        detail = "; ".join(
            f"{label} at top-{k} using {share:.0%} of the context" for label, k, _, share in lost
        )
        lines.append(
            f"  GraphRAG holds up under a budget match on: {', '.join(won)}. It does NOT on: "
            f"{detail} — a plain vector baseline gets there on less text."
        )
    return lines


def ablation_note(
    entity: dict,
    stripped: dict,
    graph: dict,
    k: int,
    entity_sweep: list[tuple[int, dict]] | None = None,
) -> list[str]:
    """Isolate the graph: same seeds, same blocks, the relationships the variable.

    ``entity_sweep`` (optional) budget-matches the *edgeless* side, which is the
    thing this ablation was previously missing: B′ reads far less text than C,
    and "the graph is worth +37 pp" is not a claim anyone should accept while
    the two sides are handed different budgets.
    """
    lines = [
        f"Graph ablation — B′ is GraphRAG's own k={k} context with every edge-derived "
        "part deleted (the per-entity Relationships lines AND the rendered Reasoning "
        "paths section), so the relationships are the only difference between B′ and C:"
    ]
    for label, key in METRICS:
        delta = graph[key] - stripped[key]
        lines.append(
            f"  {label}: {stripped[key]:.1%} without the graph → {graph[key]:.1%} with it "
            f"({delta * 100:+.1f} pp)."
        )
    ratio = (
        graph["mean_context_chars"] / stripped["mean_context_chars"]
        if stripped["mean_context_chars"]
        else 0.0
    )
    lines.append(
        f"  Cost of the graph: {ratio:.2f}x the context "
        f"({graph['mean_context_chars']:,.0f} vs {stripped['mean_context_chars']:,.0f} chars)."
    )
    lines.append(
        f"  For reference, a standalone entity vector RAG at top-{k} (its own cosine "
        f"ranking instead of Synapse's hybrid seeding) scores {entity['fact_recall']:.1%} / "
        f"{entity['full_coverage']:.1%} on {entity['mean_context_chars']:,.0f} chars."
    )
    if entity_sweep:
        lines += [f"  {line}" for line in edgeless_budget_lines(entity_sweep, graph, k)]
    return lines


def edgeless_budget_lines(
    entity_sweep: list[tuple[int, dict]], graph: dict, graph_k: int = DEFAULT_GRAPH_K
) -> list[str]:
    """Budget-match the *edgeless* side of the ablation, and say what happens.

    Without this the ablation is the one comparison in the report that is never
    budget-matched — which is exactly the objection an auditor raised, and it was
    correct: the sweep used to stop at k=12, where the entity system read 66% of
    GraphRAG's context, and the two were compared as though that were equal.
    """
    if not entity_sweep:
        return []
    ceiling_k, ceiling = entity_sweep[-1]
    target = graph["mean_context_chars"] or 1.0
    match = context_matched(entity_sweep, graph["mean_context_chars"])
    if match is None:
        lines = [
            f"Budget check (B″): the edgeless entity system never reaches C's context, "
            f"even reading ALL {ceiling_k} entities in the graph "
            f"({ceiling['mean_context_chars']:,.0f} vs {graph['mean_context_chars']:,.0f} "
            f"chars) — where it scores {ceiling['fact_recall']:.1%} / "
            f"{ceiling['full_coverage']:.1%}. The residual context gap is stated rather "
            "than hidden."
        ]
    else:
        k, agg = match
        lines = [
            f"Budget check (B″): top-{k} is the first edgeless entity setting that reads "
            f"at least as much as C ({agg['mean_context_chars']:,.0f} vs "
            f"{graph['mean_context_chars']:,.0f} chars); at that budget it scores "
            f"{agg['fact_recall']:.1%} / {agg['full_coverage']:.1%}."
        ]
        if (
            agg["fact_recall"] >= graph["fact_recall"]
            and agg["full_coverage"] >= graph["full_coverage"]
        ):
            lines.append(
                "  → Given C's own budget the edgeless system already matches it, so the "
                "ablation gap is about ranking efficiency, not about information the "
                "graph reaches and top-k cannot."
            )
        else:
            lines.append(
                "  → At equal budget the edgeless system is still behind on both metrics."
            )

    # How far the edgeless side has to widen before it reaches C at all. Stated
    # per metric, because "it never catches up" would be false: at k = every
    # entity in the graph it trivially scores 100%, and a report that implied
    # otherwise would be exactly the kind of thing this benchmark exists to avoid.
    for label, key in METRICS:
        reached = smallest_k_reaching(entity_sweep, graph[key], key)
        if reached is None:
            best_k, best = max(entity_sweep, key=lambda item: item[1][key])
            lines.append(
                f"  → {label}: no k reaches C's {graph[key]:.1%}; the edgeless best is "
                f"{best[key]:.1%} at top-{best_k}."
            )
            continue
        wide_k, wide = reached
        lines.append(
            f"  → {label}: the edgeless system needs top-{wide_k} — {wide_k} of the "
            f"{ceiling_k} entities in the graph, {wide['mean_context_chars']:,.0f} chars "
            f"= {wide['mean_context_chars'] / target:.1f}x C's context — before it "
            f"reaches C's {graph[key]:.1%}."
        )
    lines.append(
        f"  → Read as a claim: the relationships buy retrieval *width*. C gets there from "
        f"{graph_k} ranked seeds; the same entity descriptions without edges have to be "
        f"read several times wider to get there — and at k={ceiling_k} the edgeless "
        f"system has read every description in the graph "
        f"({ceiling['mean_context_chars']:,.0f} chars), which is the trivial 100% any "
        "system reaches by retrieving everything."
    )
    return lines


def scale_note(dataset: dict, chunked: dict, matched: tuple[int, dict] | None) -> list[str]:
    """State how much of the corpus each side is reading. It is a lot.

    A 34-passage corpus is small enough that a "budget-matched" baseline is
    reading most of the text that exists, and the shipped system is not far
    behind. Neither number flatters anybody, so both are printed.
    """
    total = sum(len(p["text"]) for p in dataset["passages"]) or 1
    lines = [
        f"Scale caveat (computed): the entire corpus is {total:,} characters. The "
        f"shipped system's mean context is {chunked['mean_context_chars'] / total:.0%} "
        "of it."
    ]
    if matched:
        k, agg = matched
        lines.append(
            f"  The budget-matched baseline A′ reads {agg['mean_context_chars'] / total:.0%} "
            f"of the corpus ({k} of {len(dataset['passages'])} passages), which is why it "
            "scores what it scores. On a corpus this size 'budget-matched' and 'reads "
            "nearly everything' are the same row; treat A′ as an upper bound on what "
            "top-k can do here, not as a deployable configuration."
        )
    return lines


def chunk_note(graph_only: dict, with_chunks: dict, overlap: dict) -> list[str]:
    """What the source chunks add on top of the graph — and what it costs.

    Also states, with a measured number, how much of the chunk channel is
    literally the passage baseline, because a reader who does not notice that
    would over-read D's score.
    """
    lines = ["Source chunks — what the prose adds to the graph (C → D):"]
    for label, key in METRICS:
        delta = with_chunks[key] - graph_only[key]
        lines.append(
            f"  {label}: {graph_only[key]:.1%} graph-only → {with_chunks[key]:.1%} with "
            f"source chunks ({delta * 100:+.1f} pp)."
        )
    ratio = (
        with_chunks["mean_context_chars"] / graph_only["mean_context_chars"]
        if graph_only["mean_context_chars"]
        else 0.0
    )
    lines.append(
        f"  Cost of the excerpts: {ratio:.2f}x the context "
        f"({with_chunks['mean_context_chars']:,.0f} vs "
        f"{graph_only['mean_context_chars']:,.0f} chars)."
    )
    lines.append(
        f"  Overlap with the baseline, measured: {overlap['containment']:.0%} of the "
        f"passages system A retrieves at top-{overlap['baseline_k']} also appear among "
        f"D's excerpts, and D returns {overlap['mean_excerpts']:.1f} excerpts per "
        f"question of which {overlap['mean_beyond']:.1f} are passages A did not "
        "retrieve. D's semantic excerpt channel IS the passage baseline, run inside "
        "Synapse; only the structural channel (chunks reached through the entities the "
        "graph selected) is something plain vector RAG cannot do."
    )
    return lines


def excerpt_overlap(baseline: list[QuestionResult], with_chunks: list[QuestionResult]) -> dict:
    """How much of D's excerpt channel is literally A's top-k, per question.

    ``containment`` is the mean fraction of A's retrieved passages that also
    show up among D's excerpts — 1.0 would mean D read everything A read.
    """
    by_qid = {r.qid: r for r in with_chunks}
    containments: list[float] = []
    counts: list[int] = []
    beyond: list[int] = []
    baseline_k = 0
    for base in baseline:
        chosen = set(base.extras.get("passages", []))
        baseline_k = max(baseline_k, len(chosen))
        excerpts = {i for i in by_qid[base.qid].extras.get("chunks", []) if i is not None}
        counts.append(len(excerpts))
        beyond.append(len(excerpts - chosen))
        if chosen:
            containments.append(len(chosen & excerpts) / len(chosen))
    n = len(baseline) or 1
    return {
        "baseline_k": baseline_k,
        "containment": sum(containments) / len(containments) if containments else 0.0,
        "mean_excerpts": sum(counts) / n,
        "mean_beyond": sum(beyond) / n,
    }


def misses_note(label: str, results: list[QuestionResult]) -> list[str]:
    """Name — from the data, never by hand — the questions a system got wrong.

    The README used to assert which questions failed and why. It named three;
    the generated table showed five. Prose that contradicts the table it sits
    next to is worse than no prose, so the sentence is generated here and the
    README points at it.
    """
    missed = [r for r in results if r.missing]
    total = len(results)
    if not missed:
        return [f"{label} retrieved every required fact on all {total} questions."]
    ids = ", ".join(r.qid for r in missed)
    facts = ", ".join(sorted({f for r in missed for f in r.missing}))
    return [
        f"{label} missed at least one required fact on {len(missed)} of {total} "
        f"questions ({ids}). The facts it never retrieved: {facts}."
    ]


def context_matched_note(sweep: list[tuple[int, dict]], graph: dict) -> list[str]:
    """Narrate the equal-context comparison, in whichever direction it falls."""
    match = context_matched(sweep, graph["mean_context_chars"])
    if match is None:
        return [
            "Equal-context check: even at the largest k in the sweep the baseline "
            "never reaches GraphRAG's context size, so GraphRAG's advantage is not "
            "explained by returning more text."
        ]
    k, agg = match
    lines = [
        f"Equal-context check: top-{k} is the first baseline setting that reads at "
        f"least as much as GraphRAG ({agg['mean_context_chars']:,.0f} vs "
        f"{graph['mean_context_chars']:,.0f} chars); it scores "
        f"{agg['fact_recall']:.1%} recall / {agg['full_coverage']:.1%} coverage."
    ]
    if (
        agg["fact_recall"] > graph["fact_recall"]
        or agg["full_coverage"] > graph["full_coverage"]
    ):
        lines.append(
            "  → At that budget the baseline matches or beats GraphRAG. Caveat in both "
            f"directions: {k} passages is a large share of a corpus this "
            "small, so top-k saturates here in a way it would not on a real one — but "
            "that is a limitation of the benchmark, not a point in GraphRAG's favour."
        )
    else:
        lines.append(
            "  → GraphRAG still wins at equal context budget, so the advantage is "
            "structural rather than a matter of returning more text."
        )
    return lines


@dataclass
class Report:
    """Everything one run produced, ready to be printed or written as markdown."""

    dataset: dict
    baseline: Run
    entity: Run
    stripped: Run
    graph: Run
    chunked: Run
    passage_sweep: list[tuple[int, dict]]
    entity_sweep: list[tuple[int, dict]]
    baseline_k: int
    graph_k: int
    #: The settings that produced these numbers, echoed into results.md so a
    #: reader can tell which configuration the table describes.
    settings_line: str = ""

    @property
    def matched(self) -> tuple[int, dict] | None:
        """A′ — the passage baseline at the headline system's own budget."""
        return context_matched(self.passage_sweep, self.chunked.agg["mean_context_chars"])

    @property
    def entity_matched(self) -> tuple[int, dict] | None:
        """B″ — the edgeless entity system at C's own budget."""
        return context_matched(self.entity_sweep, self.graph.agg["mean_context_chars"])

    @property
    def overlap(self) -> dict:
        return excerpt_overlap(self.baseline.results, self.chunked.results)

    def rows(self) -> list[tuple[str, dict]]:
        """(label, aggregate) for every system, in reading order."""
        rows: list[tuple[str, dict]] = [
            (f"A. Passage vector RAG (top-{self.baseline_k})", self.baseline.agg)
        ]
        if self.matched:
            rows.append(
                (
                    f"A′. Passage vector RAG (top-{self.matched[0]}, budget-matched to D)",
                    self.matched[1],
                )
            )
        rows.append(
            (f"B. Entity vector RAG, no edges (top-{self.graph_k})", self.entity.agg)
        )
        if self.entity_matched:
            rows.append(
                (
                    f"B″. Entity vector RAG (top-{self.entity_matched[0]}, "
                    "budget-matched to C)",
                    self.entity_matched[1],
                )
            )
        rows += [
            (f"B′. GraphRAG seeds, graph stripped (k={self.graph_k})", self.stripped.agg),
            (f"C. Synapse GraphRAG, graph only (k={self.graph_k})", self.graph.agg),
            (f"D. Synapse GraphRAG + source chunks (k={self.graph_k})", self.chunked.agg),
        ]
        return rows

    def analysis(self) -> list[str]:
        """The whole argument, computed from the run."""
        return (
            headline(self.chunked.agg, self.passage_sweep)
            + [""]
            + verdict(self.baseline.agg, self.chunked.agg, self.passage_sweep)
            + context_matched_note(self.passage_sweep, self.chunked.agg)
            + scale_note(self.dataset, self.chunked.agg, self.matched)
            + [""]
            + chunk_note(self.graph.agg, self.chunked.agg, self.overlap)
            + [""]
            + ablation_note(
                self.entity.agg,
                self.stripped.agg,
                self.graph.agg,
                self.graph_k,
                self.entity_sweep,
            )
            + [""]
            + misses_note("D (GraphRAG + source chunks)", self.chunked.results)
            + misses_note("C (graph only)", self.graph.results)
            + misses_note(f"A (passage baseline, top-{self.baseline_k})", self.baseline.results)
        )


def print_report(report: Report) -> None:
    """Human-readable version of everything in :class:`Report`."""
    dataset = report.dataset
    width = _LABEL_W + 43
    print("\n" + "═" * width)
    print(f"  {dataset.get('name', 'Benchmark')} — {report.baseline.agg['n']} multi-hop "
          f"questions, {len(dataset['passages'])} passages, {len(dataset['entities'])} entities")
    print("═" * width)
    print(f"  {'System':<{_LABEL_W}}{'Fact Recall':>10}{'Full Coverage':>16}{'Mean context':>15}")
    print("  " + "─" * (width - 4))
    for label, agg in report.rows():
        print(_row(label, agg))
    print("  " + "─" * (width - 4))
    for line in report.analysis():
        print(f"  {line}" if line else "")

    print("\n  Per-question fact recall")
    print("  " + "─" * (width - 4))
    print(f"  {'':<6}{'A':>7}{'B':>7}{'B′':>7}{'C':>7}{'D':>7}   question")
    for base, ent, strip, gr, ch in zip(
        report.baseline.results,
        report.entity.results,
        report.stripped.results,
        report.graph.results,
        report.chunked.results,
        strict=True,
    ):
        flag = "✅" if ch.recall > base.recall else ("➖" if ch.recall == base.recall else "❌")
        print(
            f"  {ch.qid:<6}{base.recall:>7.0%}{ent.recall:>7.0%}{strip.recall:>7.0%}"
            f"{gr.recall:>7.0%}{ch.recall:>7.0%} {flag} {ch.question[:44]}"
        )
        if ch.missing:
            print(f"  {'':<6}{'':>35}   D missed: {', '.join(ch.missing)}")

    print("\n  k sweep — passage baseline (A), to the whole corpus")
    print("  " + "─" * (width - 4))
    for k, agg in report.passage_sweep:
        print(_row(f"A. Passage vector RAG (top-{k})", agg))
    print("\n  k sweep — edgeless entity system (B), to every entity in the graph")
    print("  " + "─" * (width - 4))
    for k, agg in report.entity_sweep:
        print(_row(f"B. Entity vector RAG, no edges (top-{k})", agg))
    print("═" * width + "\n")


def _markdown_lines(lines: list[str]) -> list[str]:
    """Terminal lines → markdown bullets (indented continuations become sub-bullets)."""
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append("")
        elif line.startswith("  "):
            out.append(f"  - {line.strip()}")
        else:
            out.append(f"- {line}")
    return out


def _sweep_table(sweep: list[tuple[int, dict]], unit: str) -> list[str]:
    lines = [
        f"| top-k | Fact recall | Full coverage | Mean chars | Share of {unit} |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    total = sweep[-1][0] if sweep else 1
    for k, agg in sweep:
        lines.append(
            f"| {k} | {agg['fact_recall']:.1%} | {agg['full_coverage']:.1%} | "
            f"{agg['mean_context_chars']:,.0f} | {k / total:.0%} |"
        )
    return lines


def _results_markdown(report: Report) -> str:
    dataset = report.dataset
    lines = [
        "# GraphRAG vs. vector RAG — benchmark results",
        "",
        f"`{dataset.get('name', 'benchmark')}` — {len(dataset['passages'])} passages, "
        f"{len(dataset['entities'])} entities, {len(dataset['relationships'])} relationships, "
        f"{report.baseline.agg['n']} multi-hop questions.",
        "",
        "Every system below uses the same embedding model and is scored with the same "
        "rule (a required fact counts as retrieved when its entity name appears in the "
        "retrieved context). No LLM is involved — this measures **retrieval**, which is "
        "the part a knowledge graph actually changes. Regenerate with `make benchmark`; "
        "this file is written by the harness, never edited by hand.",
        "",
        "**D is the shipped system**: the graph *and* the source chunks the entities "
        "were extracted from. **C is what Synapse was before source chunks** — graph "
        "only. **B′ is the ablation**: C's own context with every edge-derived part "
        "deleted (the per-entity `Relationships:` lines and the rendered "
        "`Reasoning paths:` section), so C − B′ is what the relationships contribute "
        "with the seed set held fixed. **B″ and A′** are the same graph-free systems "
        "given the same character budget as C and D respectively, because a comparison "
        "between unequal budgets is not a comparison.",
        "",
        "One thing to notice before reading D's row: **D's semantic excerpt channel is "
        "system A**. It runs a top-k cosine search over exactly the passages A ranks, "
        "with the same embedder, inside `chunk_store.search_chunks`. D scoring above A "
        "is therefore close to arithmetic, not evidence. The measured overlap is "
        "reported below so the effect can be sized.",
        "",
    ]
    if report.settings_line:
        lines += [f"Run configuration: {report.settings_line}", ""]
    lines += [
        "| System | Fact Recall | Full-Coverage Rate | Mean context (chars) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, agg in report.rows():
        emphasis = "**" if label.startswith("D.") else ""
        lines.append(
            f"| {emphasis}{label}{emphasis} | {emphasis}{agg['fact_recall']:.1%}{emphasis} | "
            f"{emphasis}{agg['full_coverage']:.1%}{emphasis} | "
            f"{agg['mean_context_chars']:,.0f} |"
        )

    lines += ["", "## What the numbers support", ""]
    lines += _markdown_lines(headline(report.chunked.agg, report.passage_sweep))
    lines += ["", "**Per-metric detail (D vs. the passage baseline)**", ""]
    lines += _markdown_lines(
        verdict(report.baseline.agg, report.chunked.agg, report.passage_sweep)
        + context_matched_note(report.passage_sweep, report.chunked.agg)
        + scale_note(report.dataset, report.chunked.agg, report.matched)
    )
    lines += ["", "**What the source chunks add (C → D)**", ""]
    lines += _markdown_lines(chunk_note(report.graph.agg, report.chunked.agg, report.overlap))
    lines += ["", "**Isolating the graph (B′ vs. C, budget-matched by B″)**", ""]
    lines += _markdown_lines(
        ablation_note(
            report.entity.agg,
            report.stripped.agg,
            report.graph.agg,
            report.graph_k,
            report.entity_sweep,
        )
    )
    lines += ["", "**Where retrieval still fails**", ""]
    lines += _markdown_lines(
        misses_note("D (GraphRAG + source chunks)", report.chunked.results)
        + misses_note("C (graph only)", report.graph.results)
        + misses_note(
            f"A (passage baseline, top-{report.baseline_k})", report.baseline.results
        )
    )
    lines += [
        "",
        "## k sweeps",
        "",
        "How the two graph-free systems behave as they are allowed to retrieve more. "
        "Both sweeps run to the point where the system has read **everything it has** — "
        "all "
        f"{len(dataset['passages'])} passages, all {len(dataset['entities'])} entity "
        "records — so a budget match is always reachable and no comparison in this "
        "report is left un-matched. A corpus this small saturates, and the "
        "`Share` column is printed so that is visible rather than implied: once a row "
        "is reading a large fraction of everything that exists, its score says more "
        "about the corpus than about the retriever.",
        "",
        "### A. Passage vector RAG",
        "",
    ]
    lines += _sweep_table(report.passage_sweep, "corpus")
    lines += ["", "### B. Entity vector RAG (no edges)", ""]
    lines += _sweep_table(report.entity_sweep, "entities")
    lines += [
        "",
        "## Per-question breakdown",
        "",
        "| # | Question | Required facts | A. passage | B. entity | B′. no graph | "
        "C. graph | D. graph+chunks | D missed |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for base, ent, strip, gr, ch in zip(
        report.baseline.results,
        report.entity.results,
        report.stripped.results,
        report.graph.results,
        report.chunked.results,
        strict=True,
    ):
        lines.append(
            f"| {ch.qid} | {ch.question} | {', '.join(ch.required)} | "
            f"{base.recall:.0%} | {ent.recall:.0%} | {strip.recall:.0%} | "
            f"{gr.recall:.0%} | {ch.recall:.0%} | {', '.join(ch.missing) or '—'} |"
        )
    lines += [
        "",
        "---",
        "",
        "Generated by `backend/benchmarks/run_benchmark.py` — Synapse, "
        "© 2026 Ahmed Maaloul, AGPL-3.0-or-later.",
        "",
    ]
    return "\n".join(lines)


# ── Entry point ──────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run_benchmark",
        description="Head-to-head: Synapse GraphRAG vs. passage- and entity-level vector RAG.",
    )
    parser.add_argument(
        "--baseline-k", type=int, default=DEFAULT_BASELINE_K,
        help=f"passages the naive baseline retrieves (default: {DEFAULT_BASELINE_K})",
    )
    parser.add_argument(
        "--graph-k", type=int, default=DEFAULT_GRAPH_K,
        help=(
            f"seed entities for GraphRAG and for the edgeless entity baseline "
            f"(default: {DEFAULT_GRAPH_K}, the app default)"
        ),
    )
    parser.add_argument(
        "--dataset", type=Path, default=DATASET_PATH, help="path to a benchmark dataset JSON"
    )
    parser.add_argument(
        "--no-write", action="store_true", help="skip writing benchmarks/results.md"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="check the dataset and exit without touching Neo4j",
    )
    return parser


async def _run_systems(dataset: dict, args) -> Report:
    """Run all six systems. Raises :class:`IntegrityError` on a tainted result."""
    seeded = await seed_graph(dataset)
    print(f"   ✅ {seeded['nodes']} nodes, {seeded['edges']} edges, "
          f"{seeded['embedded']} embedded\n")

    print("🔎 Embedding corpus for the two graph-free systems …")
    passage_vectors, entity_vectors, question_vectors = await embed_corpus(dataset)
    passage_rankings = {
        qid: rank_by_similarity(vec, passage_vectors) for qid, vec in question_vectors.items()
    }
    entity_rankings = {
        qid: rank_by_similarity(vec, entity_vectors) for qid, vec in question_vectors.items()
    }
    baseline = evaluate_baseline(dataset, passage_rankings, args.baseline_k)
    entity = evaluate_entity_rag(dataset, entity_rankings, args.graph_k)
    passage_sweep = [
        (k, aggregate(evaluate_baseline(dataset, passage_rankings, k)))
        for k in sweep_ks(len(dataset["passages"]))
    ]
    entity_sweep = [
        (k, aggregate(evaluate_entity_rag(dataset, entity_rankings, k)))
        for k in sweep_ks(len(dataset["entities"]))
    ]

    # C first, while the store holds no chunks at all: that is what makes the
    # graph-only row a real measurement instead of an assertion.
    print("🔎 Running Synapse GraphRAG — graph only (no source chunks) …")
    graph = await evaluate_graphrag(dataset, args.graph_k)
    tainted = contaminated(graph)
    if tainted:
        raise IntegrityError(
            "the graph-only run returned source excerpts "
            f"({', '.join(tainted)}) — the :Chunk store was not empty, so row C would "
            "be a lie. Clear the database and re-run."
        )
    stripped = strip_edges(graph)

    print("🧾 Storing the corpus as source chunks (text units) …")
    stored = await seed_chunks(dataset)
    print(f"   ✅ {stored['chunks']} chunks, {stored['links']} entity links\n")

    print("🔎 Running Synapse GraphRAG — graph + source chunks …")
    chunked = await evaluate_graphrag(dataset, args.graph_k)
    with_excerpts = sum(1 for r in chunked if r.extras.get("chunks"))
    if not with_excerpts:
        raise IntegrityError(
            "no question retrieved a single source excerpt — chunk retrieval is off "
            "or the chunk vector index is missing, so row D would be identical to C "
            "and labelled as if it were not."
        )
    print(f"   ✅ excerpts attached on {with_excerpts}/{len(chunked)} questions\n")

    return Report(
        dataset=dataset,
        baseline=Run("A", "Passage vector RAG", baseline),
        entity=Run("B", "Entity vector RAG", entity),
        stripped=Run("B′", "GraphRAG seeds, graph stripped", stripped),
        graph=Run("C", "Synapse GraphRAG, graph only", graph),
        chunked=Run("D", "Synapse GraphRAG + source chunks", chunked),
        passage_sweep=passage_sweep,
        entity_sweep=entity_sweep,
        baseline_k=args.baseline_k,
        graph_k=args.graph_k,
    )


async def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    if args.validate_only:
        print(
            f"✅ {args.dataset} is loadable: {len(dataset['passages'])} passages, "
            f"{len(dataset['entities'])} entities, {len(dataset['questions'])} questions"
        )
        return 0

    from app import neo4j_driver
    from app.config import get_settings

    settings = get_settings()
    if not await neo4j_driver.verify_connectivity():
        print(
            f"❌ Cannot reach Neo4j at {settings.neo4j_uri}.\n"
            "   Start it first:  docker compose up -d neo4j\n"
            "   Then re-run:     make benchmark\n"
            "   (override with NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)",
            file=sys.stderr,
        )
        return 2

    settings_line = (
        f"embeddings `{settings.embedding_provider}` · seeds k={args.graph_k} · "
        f"`retrieval_max_hops={settings.retrieval_max_hops}` · "
        f"`chunk_retrieval_enabled={settings.chunk_retrieval_enabled}` · "
        f"`chunk_top_k={settings.chunk_top_k}` · "
        f"`chunk_context_max_chars={settings.chunk_context_max_chars}`"
    )

    try:
        print(f"📚 {dataset.get('name')}")
        print(f"   embeddings: {settings.embedding_provider} · neo4j: {settings.neo4j_uri}")
        print("⚠️  Seeding replaces :Entity/:Community/:Chunk data in this database "
              "(restore with `make demo-local`).\n")
        report = await _run_systems(dataset, args)
        report.settings_line = settings_line
    except IntegrityError as e:
        print(f"❌ Refusing to report this run: {e}", file=sys.stderr)
        return 3
    finally:
        await neo4j_driver.close_driver()

    print_report(report)

    if not args.no_write:
        RESULTS_PATH.write_text(_results_markdown(report), encoding="utf-8")
        print(f"📄 Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
