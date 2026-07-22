# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — GraphRAG vs. vector RAG on HotpotQA (a PUBLIC benchmark)

WHY THIS EXISTS. ``benchmarks/run_benchmark.py`` is honest about its own fatal
limitation: 34 passages and 14 questions, written by the same process that built
the graph they are scored on. A reviewer is right to discount it. This harness
runs the same argument on **HotpotQA** (Yang et al., EMNLP 2018) — a recognized
multi-hop QA benchmark whose corpus, questions and gold labels all come from
outside this project, and whose retrieval metrics have published baselines.

WHAT IS MEASURED. Retrieval, not generation — exactly as in the internal
benchmark, and for the same reason: the graph changes *what reaches the model*,
and scoring an LLM's phrasing would add noise to the only variable under test.
The metrics are HotpotQA's own paragraph-level retrieval metrics, so the numbers
are comparable to published work:

  • Gold-paragraph recall — of the 2 gold paragraphs, how many were retrieved.
  • Both-gold rate — fraction of questions where BOTH were retrieved. This is
    the multi-hop discriminator: a top-k ranker finds the paragraph that looks
    like the question, and a bridge question is unanswerable without the second.
  • Supporting-fact recall (sentence level) — fraction of the gold supporting
    sentences whose text is actually present in the retrieved prose. Derived
    from retrieved text, so it is only defined for systems that return prose.
  • Mean retrieved context characters — the budget, because a win bought by
    returning more text is not a win.

Everything is broken down by **bridge** vs. **comparison**, because they are not
the same task: a bridge question hides its second entity behind the first (where
a graph edge should help), while a comparison question names both entities up
front and is often answerable by two independent top-k lookups.

THE SYSTEMS, all at matched retrieval budgets, all over the same paragraphs.

  A.  Passage vector RAG over the pooled corpus — the standard baseline.
      Every sampled paragraph from *every* sampled question is one document;
      embed them all with Synapse's own embedder, embed the question, take the
      top-k by cosine. This is the fair comparison, because it is the pool
      Synapse's graph is built over.

  A′. The same system at the smallest k whose mean context reaches D's, read off
      the sweep — never chosen by hand.

  A_d. Passage vector RAG restricted to each question's own 10-paragraph
      distractor set — the **published HotpotQA setting**. Reported separately
      and never mixed with A: it searches 10 candidates where A searches the
      whole pool, so it is an upper bound and the only row directly comparable
      to numbers in the literature.

  C.  Synapse GraphRAG, graph only — the production retrieval path
      (``chat_engine.retrieve_subgraph``) with source-chunk retrieval switched
      off. This is what Synapse was before source chunks.

  D.  Synapse GraphRAG + source chunks — the shipped system.

The graph is built by the REAL pipeline: ``graph_builder.build_knowledge_graph``
per paragraph, with entity resolution and chunk storage exactly as a user's
ingest would run them, one document per HotpotQA paragraph so every excerpt maps
back to a labelled paragraph.

TWO SCORING RULES, BOTH REPORTED — inherited from the internal benchmark, for
the same reason: applying one rule to every system is not the same as applying a
*neutral* rule.

  PERMISSIVE (provenance)  A gold paragraph counts as retrieved when the context
      contains evidence traceable to it: an excerpt from it, or an entity that
      was extracted from it. The graph rows have this second channel; the
      passage rows do not.
  STRICT (prose)  A gold paragraph counts only when its own verbatim text came
      back as a source excerpt. Row C scores 0.0 here **by construction** — it
      returns a compressed representation of the corpus, never the corpus. That
      is the finding, not a failure, and it is stated in those words.

Every credited paragraph is attributed to the channel that credited it —
``prose`` (an excerpt), ``entity`` (a materialised entity block) or ``edge`` (a
neighbour line or reasoning path only) — with precedence prose → entity → edge,
so attribution is conservative *against* the graph. The split is printed.

Read permissive as an upper bound and strict as a lower bound.

EFFECT-SIZE FLOOR. Same rule as the internal benchmark: with N questions, a gap
smaller than one question (1/N) is reported as TOO CLOSE TO CALL, never as a
lead. It applies to the bridge/comparison subgroups with *their own* N, which is
smaller — so most subgroup gaps will, correctly, refuse to be called. No
statistical significance is claimed anywhere; the sample is a sample of dozens.

COST. This harness spends real money: one LLM extraction call per paragraph.
  • ``--dry-run`` estimates the bill and exits WITHOUT calling any model. It is
    genuinely free — no LLM, no embeddings, no Neo4j.
  • ``--max-usd`` aborts ingestion the moment the metered spend passes the cap.
  • ``--reuse-graph`` skips ingestion entirely and evaluates the graph already in
    Neo4j, which makes every re-run free. The harness verifies that the graph
    really does contain the sampled paragraphs and refuses to report if not.
  • The downloaded dataset is cached in ``benchmarks/public/data/`` — gitignored,
    so no HotpotQA bytes enter the repository — and the sample is a pure function
    of ``(--seed, --questions)``, so re-running never re-downloads and always
    scores the same questions.
  • Actual token usage is metered from the provider's own usage metadata and
    priced from ``cost.PRICES_USD_PER_1M_TOKENS`` — a hand-recorded table that
    must be re-verified. Every USD figure printed anywhere is an ESTIMATE.

RUN IT (the pilot phase, five questions, measures cost per question):

    docker compose up -d neo4j
    cd backend && EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7687 \\
        python -m benchmarks.public.run_hotpotqa --questions 5 --dry-run
    cd backend && EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7687 \\
        python -m benchmarks.public.run_hotpotqa --questions 5

Then re-score for free as often as you like with ``--reuse-graph``.

WARNING: ingestion REPLACES the ``:Entity`` / ``:Community`` / ``:Chunk``
contents of the target Neo4j database (restore the demo with ``make demo-local``).

Exit codes: 0 success · 1 dataset unusable · 2 Neo4j unreachable · 3 the run
produced a result the harness cannot honestly report · 4 ``--reuse-graph`` but
the graph does not hold this sample · 5 the cost cap was hit mid-ingestion.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from benchmarks.public import cost, hotpotqa
from benchmarks.run_benchmark import (
    EDGE,
    ENTITY,
    PROSE,
    SOURCES_HEADING,
    Channels,
    DatasetError,
    IntegrityError,
    context_matched,
    rank_by_similarity,
    smallest_k_reaching,
    sweep_ks,
)
from benchmarks.run_benchmark import (
    _markdown_lines as markdown_lines,
)

logger = logging.getLogger(__name__)

# ── Cost discipline: the constants that bound what a run can spend ───────────
#: Questions sampled when the caller does not say — the dataset module's own
#: default, so the whole wave agrees on what "the committed configuration" is.
#: Deliberately modest: 20 questions is ~200 paragraphs is ~200 extraction calls.
#: The effect-size floor at N=20 is 5.0 pp, which is stated in every verdict.
DEFAULT_QUESTIONS = hotpotqa.DEFAULT_N
#: Above this the report SHOUTS. Nothing here is expensive by the standards of a
#: research lab, but it is the author's own card, and a harness that can quietly
#: 10x its own bill is a harness nobody should run.
LOUD_QUESTION_THRESHOLD = 50
#: Hard cap on estimated spend. Ingestion aborts the moment the *metered* spend
#: passes it, and the pre-flight estimate refuses to start a run that is already
#: projected past it.
DEFAULT_MAX_USD = 2.00
#: Completion tokens assumed per extraction call by ``--dry-run``. A guess, and
#: labelled as one — the real run measures it and the report prints the measured
#: mean so this constant can be recalibrated from data instead of intuition.
DRY_RUN_COMPLETION_TOKENS = 400

# ── Dataset ──────────────────────────────────────────────────────────────────
# Acquisition, parsing and sampling live in ``benchmarks.public.hotpotqa`` — this
# harness never touches the network or the raw file itself. That module owns the
# download (with its provenance record), the strict record parser and the
# deterministic seeded sample; everything below consumes its output.
#: HotpotQA is CC BY-SA 4.0. Nothing from it enters the repository: the cache
#: lives in ``benchmarks/public/data/``, which is gitignored.
HOTPOTQA_LICENSE = "CC BY-SA 4.0 (Yang et al., EMNLP 2018) — hotpotqa.github.io"

#: The dev distractor split gives each question 10 candidate paragraphs, of which
#: 2 are normally gold. "Normally" — a handful of questions have both supporting
#: facts inside one paragraph, so the both-gold metric is really *all*-gold and
#: the count of exceptions is reported rather than assumed away.
GOLD_PARAGRAPHS_PER_QUESTION = 2
QUESTION_TYPES = hotpotqa.HOP_TYPES

#: Passages the pooled baseline retrieves by default. Two is the number of gold
#: paragraphs, i.e. the strongest-precision setting a top-k system can be given
#: on this task; the full sweep and a budget-matched row are reported too, so no
#: claim rests on it.
DEFAULT_BASELINE_K = 2
#: Seed entities for GraphRAG — the production default of ``retrieve_subgraph``,
#: untuned for this benchmark.
DEFAULT_GRAPH_K = 8
#: Ingestion theme. HotpotQA is Wikipedia prose about people, places, works and
#: organisations, so the generic schema is the honest choice; picking a
#: domain-specific one would be tuning the pipeline to the benchmark.
DEFAULT_THEME = "Generic"

#: Seed for the question sample — the dataset module's own pinned default, shared
#: so every phase of this wave samples the *same* questions. ``--seed`` changes
#: it, and the report prints whichever was used.
DEFAULT_SEED = hotpotqa.DEFAULT_SEED

#: An entity name shorter than this is never matched against a context. HotpotQA
#: entity names are produced by an LLM over Wikipedia, not curated like
#: ``dataset.json``'s, so "US" or "BBC" would otherwise credit paragraphs by
#: accident. Matching is word-bounded for the same reason — stricter than the
#: internal benchmark's plain substring rule, deliberately.
MIN_ENTITY_NAME_CHARS = 3

RESULTS_PATH = Path(__file__).with_name("results_hotpotqa.md")


class ReuseError(RuntimeError):
    """Raised when ``--reuse-graph`` is asked to score a graph that isn't this sample."""


class BudgetExceeded(RuntimeError):
    """Raised when metered spend passes ``--max-usd`` mid-ingestion."""


# ── Corpus model — a thin adapter over benchmarks.public.hotpotqa ───────────
def paragraph_text(paragraph: hotpotqa.Paragraph) -> str:
    """What Synapse ingests and what the passage baseline embeds.

    Title first, then the paragraph verbatim. The title is included because it is
    the document name the paragraph is ingested under *and* the HotpotQA gold
    label, and because leaving the subject out of a Wikipedia paragraph would
    handicap the baseline as much as the graph. The body is byte-identical to the
    dataset's sentences, which is what lets sentence-level supporting-fact recall
    be a substring test rather than a fuzzy one.
    """
    return f"{paragraph.title}\n{paragraph.text.strip()}"


def gold_titles(question: hotpotqa.HotpotQuestion) -> tuple[str, ...]:
    """Titles of the question's gold paragraphs, deduplicated, in context order."""
    return tuple(dict.fromkeys(p.title for p in question.gold_paragraphs))


def pool_titles(question: hotpotqa.HotpotQuestion) -> tuple[str, ...]:
    """Titles of the question's own 10 candidates — the published distractor pool."""
    return tuple(dict.fromkeys(p.title for p in question.paragraphs))


@dataclass
class Corpus:
    """The sampled questions and the deduplicated paragraphs they span.

    Paragraphs are keyed by title because that is simultaneously HotpotQA's gold
    label, the document name each paragraph is ingested under, and therefore the
    provenance key a retrieved excerpt comes back with. One key, no mapping table,
    nothing to get out of sync.
    """

    questions: list[hotpotqa.HotpotQuestion]
    paragraphs: dict[str, hotpotqa.Paragraph]
    #: Counted, never hidden: everything the loader had to drop or reconcile.
    notes: dict = field(default_factory=dict)

    @property
    def titles(self) -> list[str]:
        """Pooled paragraph titles in a stable order (the baseline's document ids)."""
        return sorted(self.paragraphs)

    def text(self, title: str) -> str:
        return paragraph_text(self.paragraphs[title])

    @property
    def total_chars(self) -> int:
        return sum(len(paragraph_text(p)) for p in self.paragraphs.values())

    def counts_by_type(self) -> dict[str, int]:
        counts = dict.fromkeys(QUESTION_TYPES, 0)
        for question in self.questions:
            counts[question.hop_type] = counts.get(question.hop_type, 0) + 1
        return counts


def build_corpus(
    questions: list[hotpotqa.HotpotQuestion],
    *,
    sample: hotpotqa.SampleResult | None = None,
) -> Corpus:
    """Pool the sampled questions' paragraphs into one corpus, disclosing the seams.

    The parser in ``hotpotqa.py`` has already refused malformed records and
    dropped supporting facts that point past the end of their paragraph. What is
    left to reconcile here is what *pooling* introduces:

      * the same title appearing in two questions — kept once, because the graph
        holds one document per name, with any text disagreement counted;
      * a question whose supporting facts all live in one paragraph, so
        "both-gold" is really "all-gold" for it — counted, never assumed away.
    """
    if not questions:
        raise DatasetError(
            "the sample is empty — nothing to score. Check --questions and the "
            "dataset cache."
        )

    paragraphs: dict[str, hotpotqa.Paragraph] = {}
    conflicts: list[str] = []
    single_gold: list[str] = []

    for question in questions:
        if len(gold_titles(question)) != GOLD_PARAGRAPHS_PER_QUESTION:
            single_gold.append(question.id)
        for paragraph in question.paragraphs:
            existing = paragraphs.get(paragraph.title)
            if existing is None:
                paragraphs[paragraph.title] = paragraph
            elif existing.sentences != paragraph.sentences:
                conflicts.append(paragraph.title)

    return Corpus(
        questions=list(questions),
        paragraphs=paragraphs,
        notes={
            "paragraph_text_conflicts": sorted(set(conflicts)),
            "questions_without_two_gold": single_gold,
            "dangling_supporting_facts": sum(
                len(q.dangling_supporting_facts) for q in questions
            ),
            "skipped_records": len(sample.skipped) if sample else 0,
            "pool_size": sample.pool_size if sample else len(questions),
            "source": sample.source if sample else "provided",
        },
    )


def load_corpus(
    *,
    questions: int,
    seed: int,
    data_path: Path | None = None,
    allow_download: bool = True,
) -> Corpus:
    """Load the capped, seeded sample. Free: network at worst, never a model call.

    Delegates every byte of acquisition, parsing and sampling to
    ``benchmarks.public.hotpotqa`` — which caches the download in
    ``benchmarks/public/data/`` next to a provenance record, so a re-run never
    re-fetches and every run can say exactly which bytes it scored.
    """
    if questions <= 0:
        raise DatasetError("--questions must be >= 1; an unbounded run is never allowed")
    try:
        sample = hotpotqa.load_sample(
            questions, seed=seed, path=data_path, allow_download=allow_download
        )
    except (OSError, ValueError, RuntimeError) as e:
        raise DatasetError(f"could not load HotpotQA: {e}") from e

    if len(sample.questions) < questions:
        print(
            f"⚠️  Asked for {questions} questions, the pool yielded "
            f"{len(sample.questions)} (pool size {sample.pool_size})."
        )
    return build_corpus(sample.questions, sample=sample)


# ── Pure scoring primitives (unit-tested in tests/test_hotpotqa_harness.py) ──
@cache
def _name_pattern(name: str) -> re.Pattern[str] | None:
    """Word-bounded, case-insensitive matcher for one entity name.

    Cached because every question is scored against *every* entity in the graph —
    a few thousand patterns on a 20-question run, which overflows ``re``'s own
    internal cache and would otherwise be recompiled per question.
    """
    cleaned = (name or "").strip()
    if len(cleaned) < MIN_ENTITY_NAME_CHARS:
        return None
    return re.compile(rf"(?<!\w){re.escape(cleaned)}(?!\w)", re.IGNORECASE)


def names_present(text: str, names: list[str]) -> list[str]:
    """Which ``names`` occur in ``text`` as whole words (case-insensitively)."""
    haystack = text or ""
    if not haystack:
        return []
    found: list[str] = []
    for name in names:
        pattern = _name_pattern(name)
        if pattern is not None and pattern.search(haystack):
            found.append(name)
    return found


def credit_paragraphs(
    context: str,
    sources: list[dict],
    provenance: dict[str, list[str]],
) -> dict[str, str]:
    """Map every paragraph the context carries evidence for to its best channel.

    ``provenance`` maps an entity name to the paragraph titles it was extracted
    from — the graph's own answer to "where did this come from", read back out of
    Neo4j rather than reconstructed.

    Precedence is ``prose`` → ``entity`` → ``edge``, the same conservative order
    the internal benchmark uses: a paragraph whose text actually came back is
    credited to the prose, never to the graph, so the graph can only ever be
    credited for paragraphs it reached *without* the excerpt channel.
    """
    channels = Channels.of_graph(context)
    names = list(provenance)

    by_channel: dict[str, set[str]] = {
        PROSE: {str(s.get("document") or "") for s in sources or [] if s.get("document")},
        ENTITY: set(),
        EDGE: set(),
    }
    for channel, text in ((ENTITY, channels.entities), (EDGE, channels.edges)):
        for name in names_present(text, names):
            by_channel[channel].update(provenance.get(name) or [])

    credited: dict[str, str] = {}
    for channel in (PROSE, ENTITY, EDGE):
        for title in sorted(by_channel[channel]):
            credited.setdefault(title, channel)
    return credited


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def supporting_found(
    prose: str,
    supporting: tuple[tuple[str, int], ...],
    paragraphs: dict[str, hotpotqa.Paragraph],
) -> int:
    """How many gold supporting sentences appear verbatim in ``prose``.

    Whitespace-normalised substring matching, which is as cheap as the task
    description promised and slightly *generous* to every system equally: a
    sentence counts wherever it appears, even inside a paragraph retrieved for
    another reason. Empty prose scores 0, which is the correct answer for a
    system that returned no text.
    """
    haystack = _normalize_space(prose)
    if not haystack:
        return 0
    hits = 0
    for title, index in supporting:
        paragraph = paragraphs.get(title)
        if paragraph is None or index >= len(paragraph.sentences):
            continue
        sentence = _normalize_space(paragraph.sentences[index])
        if sentence and sentence in haystack:
            hits += 1
    return hits


@dataclass
class QuestionResult:
    """One question, as retrieved by one system, under both scoring rules."""

    qid: str
    question: str
    qtype: str
    gold: tuple[str, ...]
    #: paragraph title → channel that credited it (prose / entity / edge).
    credited: dict[str, str]
    context_chars: int
    prose_chars: int
    supporting_hits: int
    supporting_total: int
    detail: str = ""
    extras: dict = field(default_factory=dict)

    # -- permissive rule ----------------------------------------------------
    @property
    def gold_found(self) -> list[str]:
        return [t for t in self.gold if t in self.credited]

    @property
    def gold_missing(self) -> list[str]:
        return [t for t in self.gold if t not in self.credited]

    @property
    def gold_recall(self) -> float:
        return len(self.gold_found) / len(self.gold) if self.gold else 0.0

    @property
    def both_gold(self) -> bool:
        return bool(self.gold) and not self.gold_missing

    # -- strict rule (retrieved prose only) ---------------------------------
    @property
    def gold_found_strict(self) -> list[str]:
        return [t for t in self.gold if self.credited.get(t) == PROSE]

    @property
    def strict_gold_recall(self) -> float:
        return len(self.gold_found_strict) / len(self.gold) if self.gold else 0.0

    @property
    def strict_both_gold(self) -> bool:
        return bool(self.gold) and len(self.gold_found_strict) == len(self.gold)

    # -- sentence level (prose only by definition) --------------------------
    @property
    def support_recall(self) -> float:
        return self.supporting_hits / self.supporting_total if self.supporting_total else 0.0


def aggregate(results: list[QuestionResult]) -> dict:
    """Roll per-question results into the headline metrics, under BOTH rules."""
    empty = {
        "gold_recall": 0.0,
        "both_gold": 0.0,
        "strict_gold_recall": 0.0,
        "strict_both_gold": 0.0,
        "support_recall": 0.0,
        "mean_context_chars": 0.0,
        "mean_prose_chars": 0.0,
        "n": 0,
    }
    if not results:
        return empty
    n = len(results)
    return {
        "gold_recall": sum(r.gold_recall for r in results) / n,
        "both_gold": sum(1.0 for r in results if r.both_gold) / n,
        "strict_gold_recall": sum(r.strict_gold_recall for r in results) / n,
        "strict_both_gold": sum(1.0 for r in results if r.strict_both_gold) / n,
        "support_recall": sum(r.support_recall for r in results) / n,
        "mean_context_chars": sum(r.context_chars for r in results) / n,
        "mean_prose_chars": sum(r.prose_chars for r in results) / n,
        "n": n,
    }


def split_by_type(results: list[QuestionResult]) -> dict[str, list[QuestionResult]]:
    """Partition results into bridge / comparison (and anything else, named).

    Bridge questions hide the second entity behind the first — the case a graph
    edge is supposed to solve. Comparison questions name both entities in the
    question text and often need no traversal at all. Reporting one number over
    both is how a GraphRAG project accidentally claims credit for the easy half.
    """
    buckets: dict[str, list[QuestionResult]] = {}
    for result in results:
        buckets.setdefault(result.qtype, []).append(result)
    ordered = {t: buckets[t] for t in QUESTION_TYPES if t in buckets}
    ordered.update({t: rs for t, rs in sorted(buckets.items()) if t not in QUESTION_TYPES})
    return ordered


def channel_split(results: list[QuestionResult]) -> dict:
    """How the credited GOLD paragraphs split across prose / entity / edge."""
    counts = dict.fromkeys((PROSE, ENTITY, EDGE), 0)
    for result in results:
        for title in result.gold_found:
            counts[result.credited[title]] += 1
    total = sum(counts.values())
    return {
        "counts": counts,
        "total": total,
        "shares": {c: (counts[c] / total if total else 0.0) for c in (PROSE, ENTITY, EDGE)},
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


# ── System A / A_d: passage vector RAG ───────────────────────────────────────
def evaluate_passage_rag(
    corpus: Corpus,
    rankings: dict[str, list[str]],
    k: int,
) -> list[QuestionResult]:
    """Score top-k passage retrieval. Pure — the caller supplies the rankings.

    Every character this system returns is corpus prose, so the permissive and
    the strict rule score it identically. That asymmetry is the whole reason the
    strict rule exists.
    """
    results: list[QuestionResult] = []
    for question in corpus.questions:
        chosen = list(rankings.get(question.id, []))[:k]
        context = "\n\n".join(corpus.text(t) for t in chosen if t in corpus.paragraphs)
        results.append(
            QuestionResult(
                qid=question.id,
                question=question.question,
                qtype=question.hop_type,
                gold=gold_titles(question),
                credited=dict.fromkeys(chosen, PROSE),
                context_chars=len(context),
                prose_chars=len(context),
                supporting_hits=supporting_found(
                    context, question.supporting_facts, corpus.paragraphs
                ),
                supporting_total=len(question.supporting_facts),
                detail=", ".join(chosen[:3]),
                extras={"titles": chosen},
            )
        )
    return results


async def embed_corpus(corpus: Corpus) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Embed every pooled paragraph and every question with Synapse's own embedder.

    The *same* model GraphRAG seeds its entity search with, so no system gets a
    better encoder than another.
    """
    from app.services.llm_provider import get_embeddings

    embeddings = get_embeddings()
    titles = corpus.titles
    paragraph_texts = [corpus.text(t) for t in titles]
    question_texts = [q.question for q in corpus.questions]

    paragraph_vectors = await asyncio.to_thread(embeddings.embed_documents, paragraph_texts)
    question_vectors = await asyncio.to_thread(embeddings.embed_documents, question_texts)
    return (
        dict(zip(titles, paragraph_vectors, strict=True)),
        {q.id: vec for q, vec in zip(corpus.questions, question_vectors, strict=True)},
    )


def rank_titles(
    corpus: Corpus,
    paragraph_vectors: dict[str, list[float]],
    question_vectors: dict[str, list[float]],
    *,
    pooled: bool,
) -> dict[str, list[str]]:
    """Rank paragraph titles per question by cosine similarity.

    ``pooled=True`` ranks over every sampled paragraph — the pool Synapse's graph
    was built from, and therefore the fair comparison. ``pooled=False`` ranks
    over each question's own 10 candidates, which is the *published* HotpotQA
    distractor setting and a much easier retrieval problem. Both are reported,
    labelled, and never averaged together.
    """
    pooled_titles = corpus.titles
    rankings: dict[str, list[str]] = {}
    for question in corpus.questions:
        titles = (
            pooled_titles
            if pooled
            else [t for t in pool_titles(question) if t in paragraph_vectors]
        )
        vectors = [paragraph_vectors[t] for t in titles]
        order = rank_by_similarity(question_vectors[question.id], vectors)
        rankings[question.id] = [titles[i] for i in order]
    return rankings


# ── Systems C and D: Synapse GraphRAG over the real retrieval path ───────────
@contextmanager
def chunk_retrieval(enabled: bool):
    """Temporarily flip ``chunk_retrieval_enabled`` on the live settings object.

    Row C is "the shipped system with the excerpt channel switched off". The
    internal benchmark got the same row by running before any ``:Chunk`` existed;
    here the chunks were written by the ingestion we just paid for, and deleting
    them to re-create them would mean paying twice. Flipping the flag the
    production code already reads is the cheap, honest equivalent — and the
    harness verifies afterwards that row C really did come back without a single
    ``Source excerpts:`` section.
    """
    from app.config import get_settings

    settings = get_settings()
    previous = settings.chunk_retrieval_enabled
    settings.chunk_retrieval_enabled = enabled
    try:
        yield settings
    finally:
        settings.chunk_retrieval_enabled = previous


PROVENANCE_QUERY = """
MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
RETURN e.name AS name, collect(DISTINCT c.document) AS documents
"""

ORPHAN_ENTITY_QUERY = """
MATCH (e:Entity)
WHERE NOT (e)-[:MENTIONED_IN]->(:Chunk) AND e.document IS NOT NULL
RETURN e.name AS name, e.document AS document
"""

CHUNK_DOCUMENTS_QUERY = "MATCH (c:Chunk) RETURN DISTINCT c.document AS document"

GRAPH_SIZE_QUERY = """
MATCH (e:Entity)
RETURN count(e) AS entities,
       sum(size(coalesce(e.name, '')) + size(coalesce(e.description, ''))) AS chars
"""


async def fetch_provenance() -> dict[str, list[str]]:
    """Entity name → the paragraph titles it was extracted from.

    Read from ``(:Entity)-[:MENTIONED_IN]->(:Chunk)``, which is the link the
    ingestion pipeline itself maintains through entity resolution — so a merged
    entity keeps the provenance of *every* paragraph it absorbed, exactly as the
    product behaves. Entities with no chunk link fall back to ``e.document``,
    which the writer sets but overwrites on re-ingest; that is why it is only a
    fallback.
    """
    from app.neo4j_driver import execute_query

    provenance: dict[str, set[str]] = {}
    for row in await execute_query(PROVENANCE_QUERY) or []:
        name = str(row.get("name") or "")
        if not name:
            continue
        docs = {str(d) for d in (row.get("documents") or []) if d}
        provenance.setdefault(name, set()).update(docs)
    for row in await execute_query(ORPHAN_ENTITY_QUERY) or []:
        name = str(row.get("name") or "")
        document = str(row.get("document") or "")
        if name and document:
            provenance.setdefault(name, set()).add(document)
    return {name: sorted(docs) for name, docs in provenance.items() if docs}


async def graph_documents() -> set[str]:
    """Distinct document names currently represented by ``:Chunk`` nodes."""
    from app.neo4j_driver import execute_query

    rows = await execute_query(CHUNK_DOCUMENTS_QUERY) or []
    return {str(r.get("document") or "") for r in rows if r.get("document")}


async def evaluate_graphrag(
    corpus: Corpus,
    provenance: dict[str, list[str]],
    k: int,
) -> list[QuestionResult]:
    """Score Synapse's own retrieval, calling ``retrieve_subgraph`` unmodified."""
    from app.services.chat_engine import retrieve_subgraph

    results: list[QuestionResult] = []
    for question in corpus.questions:
        retrieval = await retrieve_subgraph(question.question, k=k)
        context = retrieval[0] or ""
        citations = retrieval[1] or []
        sources = list(getattr(retrieval, "sources", []) or [])
        prose = Channels.of_graph(context).prose
        results.append(
            QuestionResult(
                qid=question.id,
                question=question.question,
                qtype=question.hop_type,
                gold=gold_titles(question),
                credited=credit_paragraphs(context, sources, provenance),
                context_chars=len(context),
                prose_chars=len(prose),
                supporting_hits=supporting_found(
                    prose, question.supporting_facts, corpus.paragraphs
                ),
                supporting_total=len(question.supporting_facts),
                detail=", ".join(c.get("name", "") for c in citations[:3]),
                extras={
                    "mode": getattr(retrieval, "mode", "local"),
                    "paths": len(getattr(retrieval, "paths", []) or []),
                    "seeds": [c["name"] for c in citations if c.get("kind") == "entity"],
                    "excerpt_documents": [s.get("document") for s in sources],
                    "context": context,
                },
            )
        )
    return results


def contaminated(results: list[QuestionResult]) -> list[str]:
    """Question ids whose *graph-only* context already carries source excerpts.

    Row C is only honest if the excerpt channel really was off. Checked rather
    than trusted, because the flag it depends on lives on a cached settings
    object that any other code could have re-read.
    """
    return [r.qid for r in results if SOURCES_HEADING in (r.extras.get("context") or "")]


# ── Ingestion (the part that spends money) ───────────────────────────────────
_CHAT_MODEL_ATTR = {
    "openai": "openai_chat_model",
    "openai_compatible": "openai_compatible_chat_model",
    "azure_openai": "azure_openai_chat_deployment",
    "gemini": "gemini_chat_model",
    "claude": "anthropic_model",
    "vertex": "vertex_chat_model",
    "bedrock": "bedrock_chat_model",
    "groq": "groq_chat_model",
    "mistral": "mistral_chat_model",
    "ollama": "ollama_chat_model",
}


def chat_model_name(settings) -> str:
    """The model id the configured provider will actually bill for."""
    provider = str(settings.llm_provider).lower()
    attribute = _CHAT_MODEL_ATTR.get(provider)
    if not attribute:
        return provider
    return str(getattr(settings, attribute, "") or provider)


def extraction_overhead_chars(theme: str = DEFAULT_THEME) -> int:
    """Characters of prompt template wrapped around every paragraph.

    Measured from the real prompt rather than guessed: on a 700-character
    Wikipedia paragraph the extraction system prompt is the majority of the
    input, so a cost estimate that ignores it is wrong by a factor of two.
    """
    try:
        from app.services.graph_builder import get_extraction_prompt

        messages = get_extraction_prompt(theme).format_messages(
            text="", theme=theme, document_name=""
        )
        return sum(len(str(getattr(m, "content", "") or "")) for m in messages)
    except Exception as e:  # noqa: BLE001 - estimation must never be the thing that fails
        logger.info("Could not measure the extraction prompt (%s); assuming 2000 chars", e)
        return 2000


def estimate_ingestion(corpus: Corpus, *, theme: str = DEFAULT_THEME) -> cost.Usage:
    """Estimated token usage of ingesting every pooled paragraph. No LLM involved.

    One extraction call per paragraph — the pipeline's own unit of work, since
    each paragraph is ingested as its own single-chunk document.
    """
    overhead = extraction_overhead_chars(theme)
    return cost.estimate_batch(
        [len(corpus.text(t)) for t in corpus.titles],
        overhead_chars=overhead,
        completion_tokens=DRY_RUN_COMPLETION_TOKENS,
    )


@contextmanager
def metering(ledger: cost.CostLedger):
    """Meter every extraction call ``build_knowledge_graph`` makes.

    The production ingestion path is called untouched; only the *factory* it
    resolves its chat model through is swapped, for the duration of the block,
    with one that wraps the real model in a recording Runnable. Nothing about
    what the pipeline does changes — which is the point, because the graph being
    measured has to be the graph a user would get.
    """
    from app.services import graph_builder

    original = graph_builder.get_chat_llm

    def factory(**kwargs):
        return cost.metered(original(**kwargs), ledger)

    graph_builder.get_chat_llm = factory
    try:
        yield ledger
    finally:
        graph_builder.get_chat_llm = original


CLEAR_QUERIES = (
    "MATCH (n:Entity) DETACH DELETE n",
    "MATCH (c:Community) DETACH DELETE c",
    "MATCH (c:Chunk) DETACH DELETE c",
)


async def clear_graph() -> None:
    """Empty the graph before ingesting, so leftovers cannot inflate any row."""
    from app.neo4j_driver import execute_query

    for query in CLEAR_QUERIES:
        await execute_query(query)


async def ingest_corpus(
    corpus: Corpus,
    ledger: cost.CostLedger,
    *,
    theme: str = DEFAULT_THEME,
    max_usd: float | None = None,
    concurrency: int = 1,
    echo: bool = True,
) -> dict:
    """Build the graph through the REAL pipeline, one document per paragraph.

    Each paragraph is its own document (named by its Wikipedia title) so that a
    retrieved excerpt maps back to exactly one HotpotQA gold label. That is not
    an evaluation shim: it is how a user ingesting a folder of short documents
    would have their graph built, entity resolution across documents included.

    Aborts with :class:`BudgetExceeded` the moment the *metered* spend passes
    ``max_usd``. The graph is then partial, and ``--reuse-graph`` will refuse it.
    """
    from app.services.graph_builder import build_knowledge_graph

    titles = corpus.titles
    total = len(titles)
    stats = {"documents": 0, "nodes": 0, "edges": 0, "chunks": 0, "merged": 0}
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    done = 0
    lock = asyncio.Lock()

    async def ingest(title: str) -> None:
        nonlocal done
        async with semaphore:
            result = await build_knowledge_graph([corpus.text(title)], title, theme)
        async with lock:
            done += 1
            stats["documents"] += 1
            stats["nodes"] += int(result.get("nodes_created", 0))
            stats["edges"] += int(result.get("relationships_created", 0))
            stats["chunks"] += int(result.get("chunks_stored", 0))
            stats["merged"] += int(result.get("entities_merged", 0))
            spent = ledger.usd()
            if echo and (done % 10 == 0 or done == total):
                print(
                    f"   … {done}/{total} paragraphs · {ledger.usage.total_tokens:,} tokens · "
                    f"{cost.format_usd(spent)} (estimated)"
                )
            if max_usd is not None and spent is not None and spent > max_usd:
                raise BudgetExceeded(
                    f"metered spend {cost.format_usd(spent)} passed the --max-usd cap of "
                    f"{cost.format_usd(max_usd)} after {done}/{total} paragraphs. The graph "
                    "is now PARTIAL: re-run with a higher cap or fewer --questions "
                    "(the graph is cleared and rebuilt, so nothing is lost but the spend)."
                )

    with metering(ledger):
        if concurrency <= 1:
            for title in titles:
                await ingest(title)
        else:
            await asyncio.gather(*(ingest(t) for t in titles))
    return stats


async def embedding_estimate(corpus: Corpus) -> cost.Usage:
    """Estimated embedding tokens for a whole run. Always an estimate, always said.

    LangChain's ``Embeddings`` interface returns bare vectors — no usage metadata
    exists to read — so this is characters/4 over everything the run embeds: the
    chunk texts, every entity record now in the graph, and each question (twice:
    entity seeding and chunk search).
    """
    from app.neo4j_driver import execute_query

    chars = corpus.total_chars
    chars += sum(len(q.question) for q in corpus.questions) * 2
    try:
        rows = await execute_query(GRAPH_SIZE_QUERY) or []
        chars += int((rows[0].get("chars") if rows else 0) or 0)
    except Exception as e:  # noqa: BLE001 - the estimate is a nicety, never a gate
        logger.info("Could not size the graph for the embedding estimate: %s", e)
    return cost.Usage(
        prompt_tokens=cost.tokens_from_chars(chars), calls=0, estimated_calls=1
    )


# ── Reporting ────────────────────────────────────────────────────────────────
METRICS: tuple[tuple[str, str], ...] = (
    ("Gold-paragraph recall", "gold_recall"),
    ("Both-gold rate", "both_gold"),
)
STRICT_METRICS: tuple[tuple[str, str], ...] = (
    ("Gold-paragraph recall", "strict_gold_recall"),
    ("Both-gold rate", "strict_both_gold"),
)
PERMISSIVE_RULE = (
    "permissive rule: a gold paragraph counts when the context carries evidence "
    "traceable to it (an excerpt, or an entity extracted from it)"
)
STRICT_RULE = "strict rule: a gold paragraph counts only when its own text came back as prose"

_LABEL_W = 52


def effect_floor(n_questions: int) -> float:
    """One question, as a fraction — the smallest gap this harness will call.

    Identical to the internal benchmark's rule. Zero when N < 2, because a floor
    of 100 pp would silently suppress every verdict, which is worse than
    reporting a small gap.
    """
    n = int(n_questions or 0)
    return 1.0 / n if n >= 2 else 0.0


def metric_verdict(
    label: str,
    key: str,
    baseline: dict,
    system: dict,
    *,
    baseline_label: str,
    system_label: str,
) -> str:
    """One metric, compared, with the effect-size floor applied.

    A gap smaller than one question out of N is a wash, not a lead: on a sample
    this size it is indistinguishable from re-drawing a single item. Declaring
    a sub-question "win" is exactly what a referee flags, so the harness will not
    do it — the floor is enforced here, not left to the prose.
    """
    n = int(system.get("n", 0) or baseline.get("n", 0) or 0)
    floor = effect_floor(n)
    delta = system[key] - baseline[key]
    if abs(delta) < 1e-9:
        return f"{label}: TIE ({system[key]:.1%} for both, N={n})."
    if n < 2:
        return (
            f"{label}: NOT SCOREABLE — {system_label} {system[key]:.1%} vs "
            f"{baseline_label} {baseline[key]:.1%} on N={n}. One question is not a "
            "measurement and no effect-size floor can be applied to it."
        )
    if abs(delta) < floor:
        leader = system_label if delta > 0 else baseline_label
        return (
            f"{label}: TOO CLOSE TO CALL — {system_label} {system[key]:.1%} vs "
            f"{baseline_label} {baseline[key]:.1%} ({delta * 100:+.1f} pp, nominally "
            f"{leader}). The gap is smaller than one question out of {n} "
            f"({floor * 100:.1f} pp), so it is not a meaningful lead. No significance "
            "is claimed."
        )
    ahead, behind = (
        (system_label, baseline_label) if delta > 0 else (baseline_label, system_label)
    )
    return (
        f"{label}: {ahead} ahead of {behind} — {system[key]:.1%} vs {baseline[key]:.1%} "
        f"({delta * 100:+.1f} pp for {system_label}, > one question = {floor * 100:.1f} pp, "
        f"N={n})."
    )


def parity_line(key: str, system: dict, sweep: list[tuple[int, dict]]) -> str:
    """The cheapest plain passage baseline that erases the graph's edge, if any."""
    match = smallest_k_reaching(sweep, system[key], key)
    if match is None:
        best_k, best = max(sweep, key=lambda item: item[1][key]) if sweep else (0, {key: 0.0})
        return (
            f"→ No pooled baseline k up to {sweep[-1][0] if sweep else 0} matches "
            f"{system[key]:.1%} (best is {best[key]:.1%} at top-{best_k}), so on this "
            "metric the graph is not reproducible by reading more passages."
        )
    k, agg = match
    share = (
        agg["mean_context_chars"] / system["mean_context_chars"]
        if system["mean_context_chars"]
        else 0.0
    )
    verb = "matches" if abs(agg[key] - system[key]) < 1e-9 else "beats"
    return (
        f"→ A plain passage baseline {verb} it at top-{k} ({agg[key]:.1%}), reading "
        f"{agg['mean_context_chars']:,.0f} chars = {share:.0%} of the graph's context."
    )


def headline(
    system: dict,
    sweep: list[tuple[int, dict]],
    metrics: tuple[tuple[str, str], ...] = METRICS,
    rule: str = PERMISSIVE_RULE,
) -> list[str]:
    """The one claim this benchmark is allowed to make, computed and never typed.

    For each metric the shipped system "holds up" only if no k in the pooled
    passage sweep reaches the same score while reading *less* text. Metrics that
    fail that test are named, in the same sentence.
    """
    won: list[str] = []
    lost: list[tuple[str, int, float]] = []
    for label, key in metrics:
        match = smallest_k_reaching(sweep, system[key], key)
        if match is None:
            won.append(label)
            continue
        k, agg = match
        share = (
            agg["mean_context_chars"] / system["mean_context_chars"]
            if system["mean_context_chars"]
            else 0.0
        )
        if share >= 1.0:
            won.append(label)
        else:
            lost.append((label, k, share))

    scores = ", ".join(f"{label} {system[key]:.1%}" for label, key in metrics)
    lines = [
        f"HEADLINE (computed, not asserted) — {rule}: the shipped system scores {scores} "
        f"on {system['mean_context_chars']:,.0f} chars of retrieved context over "
        f"{system['n']} HotpotQA questions."
    ]
    if won and not lost:
        lines.append(
            "  On every metric, no pooled passage baseline in the sweep reaches that score "
            "while reading less text — the advantage survives a budget match."
        )
    elif lost and not won:
        detail = "; ".join(f"{label} at top-{k} using {share:.0%} of the context" for label, k, share in lost)
        lines.append(
            f"  A plain passage baseline matches it on EVERY metric while reading LESS text "
            f"({detail}). On this sample the graph buys no context saving, and no claim to "
            "the contrary is made here."
        )
    else:
        detail = "; ".join(f"{label} at top-{k} using {share:.0%} of the context" for label, k, share in lost)
        lines.append(
            f"  It holds up under a budget match on: {', '.join(won)}. It does NOT on: "
            f"{detail} — a plain vector baseline gets there on less text."
        )
    return lines


def type_breakdown_lines(
    baseline: Run,
    system: Run,
    *,
    baseline_label: str,
    system_label: str,
) -> list[str]:
    """Bridge vs. comparison, each with its own (smaller) effect-size floor.

    Split out because they are different tasks and a single average hides which
    one the graph actually helps on. The subgroups are small by construction —
    a 20-question sample holds roughly 16 bridge and 4 comparison items — so
    most subgroup gaps will fall under their own floor and be refused. That is
    the correct outcome, not a bug, and it is why the counts are printed.
    """
    by_type_base = split_by_type(baseline.results)
    by_type_system = split_by_type(system.results)
    lines = [
        "Bridge vs. comparison — a bridge question hides its second entity behind the "
        "first (where an edge should help); a comparison question names both up front and "
        "is often two independent lookups. Each subgroup is judged against its OWN "
        "effect-size floor, which is larger because the subgroup is smaller:"
    ]
    for qtype in by_type_system:
        system_results = by_type_system[qtype]
        baseline_results = by_type_base.get(qtype, [])
        system_agg = aggregate(system_results)
        baseline_agg = aggregate(baseline_results)
        n = system_agg["n"]
        lines.append(f"  {qtype} (N={n}, floor = {effect_floor(n) * 100:.1f} pp):")
        for label, key in METRICS:
            lines.append(
                "    "
                + metric_verdict(
                    label,
                    key,
                    baseline_agg,
                    system_agg,
                    baseline_label=baseline_label,
                    system_label=system_label,
                )
            )
        if n < 5:
            lines.append(
                f"    ⚠️  UNDERPOWERED: {n} question{'' if n == 1 else 's'}. Reported for "
                "completeness; one item moves this subgroup by "
                f"{effect_floor(n) * 100:.1f} pp."
            )
    return lines


def chunk_lines(graph: dict, chunked: dict) -> list[str]:
    """What the source chunks add on top of the graph (C → D), and what they cost."""
    lines = ["Source chunks — what the prose adds to the graph (C → D):"]
    for label, key in METRICS:
        delta = chunked[key] - graph[key]
        lines.append(
            f"  {label}: {graph[key]:.1%} graph-only → {chunked[key]:.1%} with source "
            f"chunks ({delta * 100:+.1f} pp)."
        )
    ratio = (
        chunked["mean_context_chars"] / graph["mean_context_chars"]
        if graph["mean_context_chars"]
        else 0.0
    )
    lines.append(
        f"  Cost of the excerpts: {ratio:.2f}x the context "
        f"({chunked['mean_context_chars']:,.0f} vs {graph['mean_context_chars']:,.0f} chars)."
    )
    lines.append(
        f"  Under the strict rule C scores {graph['strict_gold_recall']:.1%} / "
        f"{graph['strict_both_gold']:.1%} — zero by construction, because it returns no "
        f"prose — and D scores {chunked['strict_gold_recall']:.1%} / "
        f"{chunked['strict_both_gold']:.1%}. Every gold paragraph D makes readable, it "
        "makes readable through the excerpt channel."
    )
    lines.append(
        f"  Supporting-fact recall (sentence level, prose only): C "
        f"{graph['support_recall']:.1%} → D {chunked['support_recall']:.1%}. C cannot "
        "score here at all: sentences only exist in prose, and the graph row returns none."
    )
    return lines


def channel_lines(rows: list[tuple[str, list[QuestionResult]]]) -> list[str]:
    """Which channel credited each gold paragraph — the permissive rule, sized.

    "The same rule was applied to every system" defends the rule's *uniformity*,
    not its *neutrality*. This is the measurement that sizes the difference.
    """
    lines = [
        "Where the credited gold paragraphs come from. Attribution is conservative "
        "against the graph: a paragraph whose text came back is credited to the prose, "
        "never to the entity or edge that also mentioned it."
    ]
    for label, results in rows:
        stats = channel_split(results)
        if not stats["total"]:
            lines.append(f"  {label}: no gold paragraph credited.")
            continue
        shares = stats["shares"]
        lines.append(
            f"  {label}: {stats['total']} credited — {shares[PROSE]:.0%} from retrieved "
            f"prose, {shares[ENTITY]:.0%} from an entity block, {shares[EDGE]:.0%} from an "
            "edge or reasoning path alone."
        )
    return lines


def scale_lines(corpus: Corpus, system: dict, matched: tuple[int, dict] | None) -> list[str]:
    """How much of the pooled corpus each side is reading. Both numbers, always."""
    total = corpus.total_chars or 1
    lines = [
        f"Scale (computed): the pooled corpus is {len(corpus.paragraphs)} paragraphs / "
        f"{total:,} characters. The shipped system's mean context is "
        f"{system['mean_context_chars'] / total:.1%} of it."
    ]
    if matched:
        k, agg = matched
        lines.append(
            f"  The budget-matched baseline A′ reads {agg['mean_context_chars'] / total:.1%} "
            f"of the corpus ({k} of {len(corpus.paragraphs)} paragraphs). HotpotQA's own "
            "distractor setting gives a system 10 candidates per question; row A_d is that "
            "setting and is the only row comparable to published numbers."
        )
    return lines


def misses_lines(label: str, results: list[QuestionResult], *, strict: bool = False) -> list[str]:
    """Name the questions a system failed on — from the data, never by hand."""
    if strict:
        missed = [r for r in results if not r.strict_both_gold]
    else:
        missed = [r for r in results if not r.both_gold]
    total = len(results)
    if not missed:
        return [f"{label} retrieved both gold paragraphs on all {total} questions."]
    ids = ", ".join(r.qid[:8] for r in missed[:12]) + ("…" if len(missed) > 12 else "")
    return [
        f"{label} missed at least one gold paragraph on {len(missed)} of {total} questions "
        f"({ids})."
    ]


def cost_lines(ledgers: list[cost.CostLedger], *, questions: int, reused: bool) -> list[str]:
    """The bill, per phase, always labelled an estimate."""
    lines = ["Cost (ESTIMATED — see cost.py; prices are hand-recorded, not live):"]
    if reused:
        lines.append(
            "  Ingestion: SKIPPED (--reuse-graph). This scoring run made no extraction "
            "calls and cost $0.0000 in LLM tokens."
        )
    total = 0.0
    priced = True
    for ledger in ledgers:
        lines += [f"  {line}" for line in ledger.lines()]
        amount = ledger.usd()
        if amount is None:
            priced = False
        else:
            total += amount
    if priced and ledgers:
        per_question = total / questions if questions else 0.0
        lines.append(
            f"  TOTAL: {cost.format_usd(total)} for {questions} questions = "
            f"{cost.format_usd(per_question)} per question (estimated). Prices checked "
            f"{cost.PRICES_CHECKED_ON}; re-verify at {cost.PRICING_URL}."
        )
    elif ledgers:
        lines.append(
            "  TOTAL: not priced — at least one model has no entry in "
            "cost.PRICES_USD_PER_1M_TOKENS. Token counts above are still exact."
        )
    return lines


def threats_lines(report: Report) -> list[str]:
    """Threats to validity. Generated, so the numbers in it cannot drift."""
    corpus = report.corpus
    n = report.chunked.agg["n"]
    notes = corpus.notes
    counts = corpus.counts_by_type()
    items = [
        f"SAMPLE SIZE. {n} questions ({', '.join(f'{k}={v}' for k, v in counts.items())}) "
        f"out of HotpotQA's 7,405-question dev set. One question is worth "
        f"{effect_floor(n) * 100:.1f} pp, which is why gaps below that are refused rather "
        "than reported. No statistical test is run and no significance is claimed: this "
        "is a sample of dozens, chosen to bound cost, not to support inference.",
        "RETRIEVAL ONLY. Nothing here measures answer quality. HotpotQA's headline "
        "numbers are answer EM/F1; these are its *retrieval* metrics. The rows are "
        "comparable to published retrieval numbers, not to leaderboard EM/F1, and no "
        "leaderboard comparison is implied.",
        f"POOL SIZE. Row A searches all {len(corpus.paragraphs)} sampled paragraphs, "
        "because that is the pool Synapse's graph was built over; the published distractor "
        "setting searches 10 per question. Row A_d is that setting, reported separately. "
        "Comparing A_d to A measures the pool, not the retriever.",
        "NON-DETERMINISTIC GRAPH. The graph is built by an LLM at temperature "
        f"{report.settings_summary.get('extraction_temperature', 'n/a')}; a re-ingest of "
        "the same paragraphs can produce a different graph, and therefore different C/D "
        "rows. The passage rows are deterministic. Re-running with --reuse-graph scores "
        "the same graph twice; re-ingesting does not.",
        "PROVENANCE IS WEAKER EVIDENCE THAN PROSE. Under the permissive rule a gold "
        "paragraph is credited to the graph rows when an entity extracted from it appears "
        "in the context — which is not the same as the model being able to read that "
        "paragraph. That is exactly why the strict rule is reported alongside, why C "
        "scores 0.0 under it by construction, and why the channel split is printed.",
        "PRICES ARE HAND-RECORDED. Every USD figure is an estimate from a constant "
        f"table checked on {cost.PRICES_CHECKED_ON}, not from an invoice.",
        "SENTENCE MATCHING IS LEXICAL. Supporting-fact recall is whitespace-"
        "normalised substring matching over retrieved prose. It cannot see a paraphrase, "
        "and it credits a sentence that arrived inside a paragraph retrieved for some "
        "other reason. Same rule for every system.",
        "ENTITY RESOLUTION CROSSES PARAGRAPHS. Merging 'Ed Wood' from a gold "
        "paragraph with 'Ed Wood' from a distractor is the product behaving correctly, but "
        "it means one entity can carry provenance for several paragraphs. It is measured "
        "nowhere here and can help or hurt the graph rows.",
        f"DATASET PROVENANCE. The bytes scored came from {notes.get('source', 'unknown')}; "
        f"the sample is {n} of a {notes.get('pool_size', 'n/a')}-question pool, taken by "
        "sorting on id and shuffling with the printed seed.",
    ]
    if n > LOUD_QUESTION_THRESHOLD:
        # The terminal banner promises "the report will say so". This is the report
        # saying so, in the generated section, where it cannot be scrolled past.
        items.insert(
            0,
            f"OVERSIZED RUN. {n} questions is above the {LOUD_QUESTION_THRESHOLD} this "
            "harness treats as its ceiling — one extraction call per paragraph, on the "
            "author's own card. The committed default is "
            f"{DEFAULT_QUESTIONS}; this run was asked for explicitly.",
        )
    if notes.get("skipped_records"):
        items.append(
            f"SKIPPED RECORDS. {notes['skipped_records']} record(s) in the dataset file "
            "were malformed (see benchmarks/public/hotpotqa.py) and never entered the "
            f"sampling pool of {notes.get('pool_size', 'n/a')}."
        )
    if notes.get("dangling_supporting_facts"):
        items.append(
            f"DROPPED SUPPORTING FACTS. {notes['dangling_supporting_facts']} supporting "
            "fact(s) pointed past the end of their paragraph — a known upstream quirk — and "
            "were removed from the denominator, because no system could ever retrieve them."
        )
    if notes.get("questions_without_two_gold"):
        items.append(
            f"NOT ALWAYS TWO GOLD. {len(notes['questions_without_two_gold'])} question(s) "
            "have all their supporting facts inside a single paragraph, so for those the "
            "'both-gold' rate is really an 'all-gold' rate. Counted, not assumed away."
        )
    if notes.get("paragraph_text_conflicts"):
        items.append(
            f"TITLE COLLISIONS. {len(notes['paragraph_text_conflicts'])} title(s) appeared "
            "in two questions with different text; the first was kept, since the graph "
            "holds one document per name."
        )
    if report.reused_graph:
        items.append(
            "REUSED GRAPH. This run did not ingest; it scored the graph already in Neo4j "
            "after verifying it holds every sampled paragraph. It cannot verify that the "
            "graph was built from this code at this configuration."
        )
    # Numbered here, once, so an optional item can never leave a gap in the list.
    return ["Threats to validity — the reasons not to over-read anything above:"] + [
        f"  {i}. {item}" for i, item in enumerate(items, start=1)
    ]


@dataclass
class Report:
    """Everything one run produced, ready to be printed or written as markdown."""

    corpus: Corpus
    baseline: Run
    distractor: Run
    graph: Run
    chunked: Run
    sweep: list[tuple[int, dict]]
    baseline_k: int
    graph_k: int
    ledgers: list[cost.CostLedger] = field(default_factory=list)
    settings_summary: dict = field(default_factory=dict)
    settings_line: str = ""
    reused_graph: bool = False
    ingest_stats: dict = field(default_factory=dict)

    @property
    def matched(self) -> tuple[int, dict] | None:
        """A′ — the pooled passage baseline at D's own character budget."""
        return context_matched(self.sweep, self.chunked.agg["mean_context_chars"])

    def rows(self) -> list[tuple[str, dict]]:
        rows: list[tuple[str, dict]] = [
            (f"A. Passage vector RAG, pooled (top-{self.baseline_k})", self.baseline.agg)
        ]
        if self.matched:
            rows.append(
                (
                    f"A′. Passage vector RAG, pooled (top-{self.matched[0]}, budget-matched)",
                    self.matched[1],
                )
            )
        rows += [
            (
                f"A_d. Passage vector RAG, 10-para distractor pool (top-{self.baseline_k})",
                self.distractor.agg,
            ),
            (f"C. Synapse GraphRAG, graph only (k={self.graph_k})", self.graph.agg),
            (f"D. Synapse GraphRAG + source chunks (k={self.graph_k})", self.chunked.agg),
        ]
        return rows

    def measured_rows(self) -> list[tuple[str, list[QuestionResult]]]:
        return [
            (f"A. Passage vector RAG, pooled (top-{self.baseline_k})", self.baseline.results),
            (f"A_d. Passage vector RAG, distractor pool (top-{self.baseline_k})", self.distractor.results),
            (f"C. Synapse GraphRAG, graph only (k={self.graph_k})", self.graph.results),
            (f"D. Synapse GraphRAG + source chunks (k={self.graph_k})", self.chunked.results),
        ]

    def headlines(self) -> list[str]:
        return (
            headline(self.chunked.agg, self.sweep, METRICS, PERMISSIVE_RULE)
            + [""]
            + headline(self.chunked.agg, self.sweep, STRICT_METRICS, STRICT_RULE)
        )

    def analysis(self) -> list[str]:
        base_agg = self.baseline.agg
        d_agg = self.chunked.agg
        lines = self.headlines() + [""]
        lines.append(
            f"D vs. A — the shipped system against the pooled passage baseline at "
            f"top-{self.baseline_k} (permissive rule):"
        )
        for label, key in METRICS:
            lines.append(
                "  "
                + metric_verdict(
                    label, key, base_agg, d_agg, baseline_label="A", system_label="D"
                )
            )
            lines.append(f"    {parity_line(key, d_agg, self.sweep)}")
        lines.append("")
        lines.append("D vs. A — the same comparison under the strict (prose) rule:")
        for label, key in STRICT_METRICS:
            lines.append(
                "  "
                + metric_verdict(
                    label, key, base_agg, d_agg, baseline_label="A", system_label="D"
                )
            )
        if self.matched:
            k, agg = self.matched
            lines.append("")
            lines.append(
                f"Equal-budget check (A′): top-{k} is the first pooled baseline setting that "
                f"reads at least as much as D ({agg['mean_context_chars']:,.0f} vs "
                f"{d_agg['mean_context_chars']:,.0f} chars)."
            )
            for label, key in METRICS:
                lines.append(
                    "  "
                    + metric_verdict(
                        label, key, agg, d_agg, baseline_label="A′", system_label="D"
                    )
                )
        else:
            lines += [
                "",
                "Equal-budget check (A′): no k in the sweep — up to reading every paragraph "
                "in the pool — reaches D's context size, so D's score cannot be explained by "
                "returning more text.",
            ]
        lines += [""]
        lines.append(
            f"Reference: A_d, the published 10-paragraph distractor setting at "
            f"top-{self.baseline_k}, scores "
            + ", ".join(f"{label} {self.distractor.agg[key]:.1%}" for label, key in METRICS)
            + ". It searches 10 candidates per question where every other row searches "
            f"{len(self.corpus.paragraphs)}, so it is an upper bound on top-k here and the "
            "only row to compare against the literature."
        )
        lines += [""]
        lines += type_breakdown_lines(
            self.baseline, self.chunked, baseline_label="A", system_label="D"
        )
        lines += [""]
        lines += chunk_lines(self.graph.agg, d_agg)
        lines += [""]
        lines += channel_lines(self.measured_rows())
        lines += [""]
        lines += scale_lines(self.corpus, d_agg, self.matched)
        lines += [""]
        lines += (
            misses_lines("D (GraphRAG + source chunks)", self.chunked.results)
            + misses_lines("C (graph only)", self.graph.results)
            + misses_lines(f"A (pooled passage baseline, top-{self.baseline_k})", self.baseline.results)
            + misses_lines("D under the strict rule", self.chunked.results, strict=True)
        )
        lines += [""]
        lines += cost_lines(
            self.ledgers, questions=d_agg["n"] or 1, reused=self.reused_graph
        )
        lines += [""]
        lines += threats_lines(self)
        return lines


def _row(label: str, agg: dict) -> str:
    return (
        f"  {label:<{_LABEL_W}}{agg['gold_recall']:>8.1%}{agg['both_gold']:>8.1%}"
        f"{agg['strict_gold_recall']:>9.1%}{agg['strict_both_gold']:>8.1%}"
        f"{agg['support_recall']:>9.1%}{agg['mean_context_chars']:>11,.0f}"
    )


def _header_row() -> str:
    return (
        f"  {'System':<{_LABEL_W}}{'recall':>8}{'both':>8}{'recall':>9}{'both':>8}"
        f"{'support':>9}{'chars':>11}"
    )


def print_report(report: Report) -> None:
    """Human-readable version of everything in :class:`Report`."""
    width = _LABEL_W + 53
    corpus = report.corpus
    print("\n" + "═" * width)
    print(
        f"  HotpotQA (dev, distractor) — {report.chunked.agg['n']} questions, "
        f"{len(corpus.paragraphs)} paragraphs, "
        + ", ".join(f"{k} {v}" for k, v in corpus.counts_by_type().items())
    )
    print("═" * width)
    print(f"  {'':<{_LABEL_W}}{'─ permissive ─':>16}{'─── strict ───':>17}{'prose':>9}")
    print(_header_row())
    print("  " + "─" * (width - 4))
    for label, agg in report.rows():
        print(_row(label, agg))
    print("  " + "─" * (width - 4))
    for line in report.analysis():
        print(f"  {line}" if line else "")

    print("\n  Per-question gold-paragraph recall")
    print("  " + "─" * (width - 4))
    print(f"  {'':<10}{'A':>6}{'A_d':>6}{'C':>6}{'D':>6}  type        question")
    for base, dist, graph, chunked in zip(
        report.baseline.results,
        report.distractor.results,
        report.graph.results,
        report.chunked.results,
        strict=True,
    ):
        flag = "✅" if chunked.gold_recall > base.gold_recall else (
            "➖" if chunked.gold_recall == base.gold_recall else "❌"
        )
        print(
            f"  {chunked.qid[:8]:<10}{base.gold_recall:>6.0%}{dist.gold_recall:>6.0%}"
            f"{graph.gold_recall:>6.0%}{chunked.gold_recall:>6.0%} {flag} "
            f"{chunked.qtype:<11}{chunked.question[:40]}"
        )

    print("\n  k sweep — pooled passage baseline (A), to the whole pooled corpus")
    print("  " + "─" * (width - 4))
    for k, agg in report.sweep:
        print(_row(f"A. Passage vector RAG, pooled (top-{k})", agg))
    print("═" * width + "\n")


def _sweep_table(sweep: list[tuple[int, dict]], total_paragraphs: int) -> list[str]:
    lines = [
        "| top-k | Gold-paragraph recall | Both-gold rate | Supporting-fact recall | "
        "Mean chars | Share of pool |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k, agg in sweep:
        lines.append(
            f"| {k} | {agg['gold_recall']:.1%} | {agg['both_gold']:.1%} | "
            f"{agg['support_recall']:.1%} | {agg['mean_context_chars']:,.0f} | "
            f"{k / max(1, total_paragraphs):.0%} |"
        )
    return lines


def summary_block(report: Report) -> list[str]:
    """The generated numeric core — table plus both computed headlines.

    Nothing in it is ever typed by a human. The same discipline as the internal
    benchmark, for the same reason: an audit found hand-copied numbers in the
    README that a fresh run did not reproduce.
    """
    lines: list[str] = []
    if report.settings_line:
        lines += [f"**Run configuration:** {report.settings_line}", ""]
    lines += [
        "| System | Gold-para recall | Both-gold | Gold-para recall (strict) | "
        "Both-gold (strict) | Supporting-fact recall | Mean context (chars) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, agg in report.rows():
        mark = "**" if label.startswith("D.") else ""
        cells = [label] + [
            f"{agg[key]:.1%}" for _, key in (*METRICS, *STRICT_METRICS)
        ] + [f"{agg['support_recall']:.1%}"]
        rendered = " | ".join(f"{mark}{cell}{mark}" for cell in cells)
        lines.append(f"| {rendered} | {agg['mean_context_chars']:,.0f} |")
    lines += [
        "",
        "*Permissive* credits a gold paragraph when the retrieved context carries evidence "
        "traceable to it (an excerpt from it, or an entity extracted from it); *strict* "
        "credits it only when the paragraph's own text came back as prose. Row C scores "
        "0.0 under the strict rule **by construction** — it returns a compressed "
        "representation of the corpus, never the corpus. Supporting-fact recall is "
        "sentence-level and prose-only for the same reason. Treat permissive as an upper "
        "bound and strict as a lower bound.",
        "",
    ]
    lines += markdown_lines(report.headlines())
    return lines


def results_markdown(report: Report) -> str:
    corpus = report.corpus
    counts = corpus.counts_by_type()
    lines = [
        "# Synapse on HotpotQA — a public multi-hop benchmark",
        "",
        f"`hotpot_dev_distractor_v1` — {report.chunked.agg['n']} sampled questions "
        f"({', '.join(f'{k} {v}' for k, v in counts.items())}), "
        f"{len(corpus.paragraphs)} paragraphs, {corpus.total_chars:,} characters. "
        f"Dataset: {HOTPOTQA_LICENSE}.",
        "",
        "The internal benchmark in `backend/benchmarks/` is honest but small, and its own "
        "threats-to-validity section concedes the problem: 34 passages and 14 questions, "
        "written by the same process that built the graph. This file answers the obvious "
        "objection by running the same argument on a corpus and a question set that come "
        "from outside the project, scored with **HotpotQA's own retrieval metrics** so the "
        "numbers can be read next to published work.",
        "",
        "This measures **retrieval**, not answer generation. Regenerate with the command in "
        "the run configuration below; this file is written by the harness, never edited by "
        "hand.",
        "",
    ]
    lines += summary_block(report)
    lines += ["", "## What the numbers support", ""]
    lines += markdown_lines(report.analysis()[len(report.headlines()) + 1 :])
    lines += [
        "",
        "## k sweep — pooled passage baseline (A)",
        "",
        "How the graph-free baseline behaves as it is allowed to read more. The sweep runs "
        "to the point where it has read **every paragraph in the pool**, so a budget match "
        "is always reachable and no comparison here is left unmatched.",
        "",
    ]
    lines += _sweep_table(report.sweep, len(corpus.paragraphs))
    lines += [
        "",
        "## Per-question breakdown",
        "",
        "| # | Type | Question | Gold paragraphs | A | A_d | C | D | D strict | D missed |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for base, dist, graph, chunked in zip(
        report.baseline.results,
        report.distractor.results,
        report.graph.results,
        report.chunked.results,
        strict=True,
    ):
        lines.append(
            f"| {chunked.qid[:8]} | {chunked.qtype} | {chunked.question} | "
            f"{', '.join(chunked.gold)} | {base.gold_recall:.0%} | {dist.gold_recall:.0%} | "
            f"{graph.gold_recall:.0%} | {chunked.gold_recall:.0%} | "
            f"{chunked.strict_gold_recall:.0%} | {', '.join(chunked.gold_missing) or '—'} |"
        )
    lines += [
        "",
        "---",
        "",
        "Generated by `backend/benchmarks/public/run_hotpotqa.py` — Synapse, "
        "© 2026 Ahmed Maaloul, AGPL-3.0-or-later. HotpotQA is © its authors, "
        f"{HOTPOTQA_LICENSE}.",
        "",
    ]
    return "\n".join(lines)


# ── Dry run ──────────────────────────────────────────────────────────────────
def dry_run_lines(corpus: Corpus, *, model: str, theme: str, max_usd: float) -> list[str]:
    """Estimate the bill for a run that has NOT happened. Zero model calls.

    Every number here is an estimate twice over: the token counts come from
    characters/4 and the prices come from a hand-recorded table. It exists to
    stop a run *before* it spends, not to be quoted.
    """
    usage = estimate_ingestion(corpus, theme=theme)
    price = cost.resolve_price(model)
    amount = cost.usd(usage, price)
    overhead = extraction_overhead_chars(theme)
    n = len(corpus.questions)
    lines = [
        "DRY RUN — no model was called, no token was spent, Neo4j was not touched.",
        f"  Plan: {n} questions → {len(corpus.paragraphs)} unique paragraphs "
        f"({corpus.total_chars:,} chars) → {usage.calls} extraction calls, one per "
        "paragraph, each ingested as its own document.",
        f"  Prompt overhead measured from the real extraction template: {overhead:,} chars "
        f"(~{cost.tokens_from_chars(overhead):,} tokens) on top of every paragraph.",
        f"  Estimated tokens: {usage.prompt_tokens:,} prompt + {usage.completion_tokens:,} "
        f"completion = {usage.total_tokens:,} total (completion assumed at "
        f"{DRY_RUN_COMPLETION_TOKENS}/call — a guess the real run replaces with a "
        "measurement).",
    ]
    if amount is None:
        lines.append(
            f"  Estimated cost: UNKNOWN — no price on file for {model!r}. Add it to "
            f"cost.PRICES_USD_PER_1M_TOKENS to get a figure."
        )
    else:
        lines += [
            f"  Estimated cost: {cost.format_usd(amount)} on {model} "
            f"({cost.format_usd(amount / n)} per question), from prices hand-recorded on "
            f"{cost.PRICES_CHECKED_ON} — an ESTIMATE, not a quote ({cost.PRICING_URL}).",
        ]
        if amount > max_usd:
            lines.append(
                f"  ⛔ This exceeds --max-usd ({cost.format_usd(max_usd)}). The real run "
                "would refuse to start. Lower --questions or raise --max-usd deliberately."
            )
    lines.append(
        "  Embeddings are not included above: with EMBEDDING_PROVIDER=fastembed they run "
        "locally and cost nothing, which is the configuration this harness recommends."
    )
    return lines


# ── Entry point ──────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.public.run_hotpotqa",
        description=(
            "Synapse GraphRAG vs. passage vector RAG on HotpotQA. Spends real money on "
            "ingestion — always --dry-run first."
        ),
    )
    parser.add_argument(
        "--questions", type=int, default=DEFAULT_QUESTIONS,
        help=(
            f"HARD CAP on sampled questions (default: {DEFAULT_QUESTIONS}). There is no "
            "unbounded mode."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"sampling seed (default: {DEFAULT_SEED}); (seed, questions) names the cache",
    )
    parser.add_argument(
        "--reuse-graph", action="store_true",
        help="skip ingestion and score the graph already in Neo4j — makes re-runs free",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="estimate the cost and exit WITHOUT calling any model (genuinely free)",
    )
    parser.add_argument(
        "--max-usd", type=float, default=DEFAULT_MAX_USD,
        help=f"abort ingestion once metered spend passes this (default: {DEFAULT_MAX_USD})",
    )
    parser.add_argument(
        "--baseline-k", type=int, default=DEFAULT_BASELINE_K,
        help=f"passages the vector baseline retrieves (default: {DEFAULT_BASELINE_K}, the "
             "number of gold paragraphs)",
    )
    parser.add_argument(
        "--graph-k", type=int, default=DEFAULT_GRAPH_K,
        help=f"seed entities for GraphRAG (default: {DEFAULT_GRAPH_K}, the app default)",
    )
    parser.add_argument(
        "--theme", default=DEFAULT_THEME, help=f"extraction theme (default: {DEFAULT_THEME})"
    )
    parser.add_argument(
        "--ingest-concurrency", type=int, default=1,
        help="documents ingested in parallel (default: 1 — deterministic; higher is faster "
             "but lets entity-resolution passes interleave)",
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="path to hotpot_dev_distractor_v1.json (default: the cache, downloading once)",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="never reach the network; fail if the dataset is not already cached",
    )
    parser.add_argument(
        "--model", default="",
        help="price the run as this model instead of the configured one (estimates only)",
    )
    parser.add_argument(
        "--out", type=Path, default=RESULTS_PATH, help=f"markdown report path (default: {RESULTS_PATH})"
    )
    parser.add_argument("--no-write", action="store_true", help="do not write the markdown report")
    return parser


def loud_banner(questions: int) -> list[str]:
    """Shout when a run is bigger than the modest committed default."""
    if questions <= LOUD_QUESTION_THRESHOLD:
        return []
    return [
        "",
        "🚨 " + "=" * 76,
        f"🚨 THIS RUN USES {questions} QUESTIONS — more than the {LOUD_QUESTION_THRESHOLD} "
        "this harness treats as its ceiling.",
        f"🚨 That is roughly {questions * 10} paragraphs and {questions * 10} LLM extraction "
        "calls, and it is the author's own card.",
        "🚨 The report will say so. --dry-run first if you have not already.",
        "🚨 " + "=" * 76,
        "",
    ]


async def _run_systems(corpus: Corpus, args, ledgers: list[cost.CostLedger]) -> Report:
    """Ingest (or verify), then score every system. Raises on a tainted result."""
    from app.config import get_settings
    from app.services.graph_schema import ensure_schema

    settings = get_settings()
    model = args.model or chat_model_name(settings)

    print("🧱 Ensuring schema (constraints, full-text and vector indexes) …")
    await ensure_schema()

    ingest_stats: dict = {}
    if args.reuse_graph:
        present = await graph_documents()
        missing = [t for t in corpus.titles if t not in present]
        if missing:
            raise ReuseError(
                f"--reuse-graph, but {len(missing)} of {len(corpus.titles)} sampled "
                f"paragraphs are not in Neo4j (e.g. {missing[:3]}). This graph is not this "
                "sample — scoring it would report numbers about a different corpus. Re-run "
                "without --reuse-graph (which ingests, and costs money), or with the "
                "--seed/--questions the graph was built from."
            )
        print(f"♻️  Reusing the graph in Neo4j: all {len(corpus.titles)} paragraphs present.")
    else:
        print(
            f"🧨 Clearing :Entity / :Community / :Chunk, then ingesting "
            f"{len(corpus.titles)} paragraphs through the real pipeline …"
        )
        await clear_graph()
        ledger = cost.CostLedger(model, label="Ingestion (LLM extraction)")
        ledgers.append(ledger)
        with ledger:
            ingest_stats = await ingest_corpus(
                corpus,
                ledger,
                theme=args.theme,
                max_usd=args.max_usd,
                concurrency=args.ingest_concurrency,
            )
        print(
            f"   ✅ {ingest_stats['documents']} documents · {ingest_stats['nodes']} nodes · "
            f"{ingest_stats['edges']} edges · {ingest_stats['chunks']} chunks · "
            f"{ingest_stats['merged']} entities merged"
        )
        print("   " + "\n   ".join(ledger.lines()))

    print("🔎 Embedding the pooled corpus for the passage baselines …")
    paragraph_vectors, question_vectors = await embed_corpus(corpus)
    pooled = rank_titles(corpus, paragraph_vectors, question_vectors, pooled=True)
    per_question = rank_titles(corpus, paragraph_vectors, question_vectors, pooled=False)
    baseline = evaluate_passage_rag(corpus, pooled, args.baseline_k)
    distractor = evaluate_passage_rag(corpus, per_question, args.baseline_k)
    sweep = [
        (k, aggregate(evaluate_passage_rag(corpus, pooled, k)))
        for k in sweep_ks(len(corpus.paragraphs))
    ]

    provenance = await fetch_provenance()
    if not provenance:
        raise IntegrityError(
            "no (:Entity)-[:MENTIONED_IN]->(:Chunk) provenance in the graph, so no gold "
            "paragraph could ever be credited to rows C or D. Either ingestion stored no "
            "chunks (check STORE_SOURCE_CHUNKS) or the graph is not this corpus."
        )
    print(f"🧭 Provenance: {len(provenance)} entities map back to a HotpotQA paragraph.")

    print("🔎 Running Synapse GraphRAG — graph only (excerpt channel off) …")
    with chunk_retrieval(False):
        graph = await evaluate_graphrag(corpus, provenance, args.graph_k)
    tainted = contaminated(graph)
    if tainted:
        raise IntegrityError(
            f"the graph-only run returned source excerpts ({', '.join(tainted[:3])}) — the "
            "excerpt channel was not actually off, so row C would be a lie."
        )

    print("🔎 Running Synapse GraphRAG — graph + source chunks (the shipped system) …")
    with chunk_retrieval(True):
        chunked = await evaluate_graphrag(corpus, provenance, args.graph_k)
    with_excerpts = sum(1 for r in chunked if r.extras.get("excerpt_documents"))
    if not with_excerpts:
        raise IntegrityError(
            "no question retrieved a single source excerpt — chunk retrieval is off or the "
            "chunk vector index is missing, so row D would be identical to row C and "
            "labelled as if it were not."
        )
    print(f"   ✅ excerpts attached on {with_excerpts}/{len(chunked)} questions")

    embedding_usage = await embedding_estimate(corpus)
    if str(settings.embedding_provider).lower() == "openai":
        embedding_ledger = cost.CostLedger(
            settings.openai_embedding_model, label="Embeddings (estimated from characters)"
        )
        embedding_ledger.record(embedding_usage)
        ledgers.append(embedding_ledger)

    return Report(
        corpus=corpus,
        baseline=Run("A", "Passage vector RAG (pooled)", baseline),
        distractor=Run("A_d", "Passage vector RAG (distractor pool)", distractor),
        graph=Run("C", "Synapse GraphRAG, graph only", graph),
        chunked=Run("D", "Synapse GraphRAG + source chunks", chunked),
        sweep=sweep,
        baseline_k=args.baseline_k,
        graph_k=args.graph_k,
        ledgers=ledgers,
        reused_graph=bool(args.reuse_graph),
        ingest_stats=ingest_stats,
        settings_summary={
            "extraction_temperature": settings.extraction_temperature,
            "embedding_provider": settings.embedding_provider,
            "llm_provider": settings.llm_provider,
            "model": model,
        },
    )


async def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    for line in loud_banner(args.questions):
        print(line)

    try:
        corpus = load_corpus(
            questions=args.questions,
            seed=args.seed,
            data_path=args.data,
            allow_download=not args.no_download,
        )
    except DatasetError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    counts = corpus.counts_by_type()
    print(
        f"📚 HotpotQA dev (distractor): {len(corpus.questions)} questions "
        f"({', '.join(f'{k}={v}' for k, v in counts.items())}), "
        f"{len(corpus.paragraphs)} unique paragraphs, seed {args.seed}"
    )

    # ── The free path. Nothing below this point may run before it returns. ──
    if args.dry_run:
        from app.config import get_settings

        model = args.model or chat_model_name(get_settings())
        for line in dry_run_lines(
            corpus, model=model, theme=args.theme, max_usd=args.max_usd
        ):
            print(line)
        print(
            "\n  To run it for real:\n"
            "    cd backend && EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7687 \\\n"
            f"        python -m benchmarks.public.run_hotpotqa --questions {args.questions} "
            f"--seed {args.seed}"
        )
        return 0

    from app import neo4j_driver
    from app.config import get_settings

    settings = get_settings()
    if not await neo4j_driver.verify_connectivity():
        print(
            f"❌ Cannot reach Neo4j at {settings.neo4j_uri}.\n"
            "   Start it first:  docker compose up -d neo4j\n"
            "   (override with NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)",
            file=sys.stderr,
        )
        return 2

    if not args.reuse_graph:
        # Pre-flight: refuse to start a run that is already projected over the cap.
        projected = cost.usd(
            estimate_ingestion(corpus, theme=args.theme),
            cost.resolve_price(args.model or chat_model_name(settings)),
        )
        if projected is not None and projected > args.max_usd:
            print(
                f"❌ Estimated ingestion cost {cost.format_usd(projected)} exceeds --max-usd "
                f"{cost.format_usd(args.max_usd)}. Lower --questions, or raise the cap "
                "deliberately.",
                file=sys.stderr,
            )
            await neo4j_driver.close_driver()
            return 5
        print(
            f"⚠️  Ingestion REPLACES the :Entity/:Community/:Chunk data in "
            f"{settings.neo4j_uri} (restore the demo with `make demo-local`). Estimated "
            f"cost {cost.format_usd(projected)}, cap {cost.format_usd(args.max_usd)}."
        )

    ledgers: list[cost.CostLedger] = []
    try:
        report = await _run_systems(corpus, args, ledgers)
    except ReuseError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 4
    except BudgetExceeded as e:
        print(f"⛔ {e}", file=sys.stderr)
        return 5
    except IntegrityError as e:
        print(f"❌ Refusing to report this run: {e}", file=sys.stderr)
        return 3
    finally:
        await neo4j_driver.close_driver()

    report.settings_line = (
        f"`--questions {args.questions} --seed {args.seed}` · "
        f"{len(report.corpus.paragraphs)} paragraphs · embeddings "
        f"`{settings.embedding_provider}` · extraction `{report.settings_summary['model']}` "
        f"(`{settings.llm_provider}`, temperature {settings.extraction_temperature}) · "
        f"graph seeds k={args.graph_k} · baseline top-{args.baseline_k} · "
        f"`chunk_top_k={settings.chunk_top_k}` · "
        f"`chunk_context_max_chars={settings.chunk_context_max_chars}` · "
        f"`retrieval_max_hops={settings.retrieval_max_hops}`. Reproduce: "
        f"`python -m benchmarks.public.run_hotpotqa --questions {args.questions} "
        f"--seed {args.seed} --reuse-graph`"
    )

    print_report(report)

    if not args.no_write:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(results_markdown(report), encoding="utf-8")
        except OSError as e:
            print(f"❌ Could not write {args.out}: {e}", file=sys.stderr)
            return 3
        print(f"📄 Wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
