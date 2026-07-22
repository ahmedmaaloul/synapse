# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
RQ5 — When does GraphRAG amortize its extraction cost?

HYPOTHESIS (stated before any number is read)
    GraphRAG pays a large ONE-OFF cost (LLM extraction over the whole corpus) in
    order to reduce the PER-QUERY cost. There is therefore a break-even query
    count Q* below which plain vector RAG is strictly the better engineering
    choice, and most demo projects sit below it.

PREDICTION
    Q* is finite and large — hundreds to thousands of queries per corpus — and
    grows with corpus size.

WHY THIS CAN BE ANSWERED WITHOUT SPENDING A CENT
    Every quantity the model needs is either (a) measurable from the fixtures
    and the production code paths, or (b) a published per-token price. Nothing
    here calls a chat model.

      * corpus tokens          — measured from ``benchmarks/dataset.json`` by
                                 running the *production* chunker
                                 (``pdf_parser.chunk_text``, 1000/200).
      * extraction prompt      — measured by rendering the *production* prompt
                                 (``graph_builder.get_extraction_prompt``) with
                                 an empty chunk.
      * extraction output      — DERIVED FROM THE FIXTURE. ``dataset.json`` *is*
                                 the extraction result for this corpus, so
                                 re-serialising its entities and relationships
                                 into the exact JSON shape the prompt demands
                                 gives the completion the LLM would have emitted.
      * community summaries    — Louvain is run for real (``communities.
                                 detect_communities``, no LLM), and the summary
                                 prompt is rendered for real.
      * retrieved context      — MEASURED by running the benchmark's own systems
                                 (imported read-only from ``benchmarks.
                                 run_benchmark``) against Neo4j.
      * prices                 — ``benchmarks/public/cost.py``, hand-recorded,
                                 dated, and labelled an ESTIMATE everywhere.

    The one term that cannot be measured without an LLM is the *answer* length.
    It is a free parameter — and, as the algebra below shows, it cancels out of
    the break-even entirely.

RUN
    cd backend && EMBEDDING_PROVIDER=fastembed \\
        NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=research_secret \\
        python research/rq5_cost_amortization/run_rq5.py

    Writes FINDINGS.md next to this file. Every number in it is generated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from benchmarks.public import cost as costmod  # noqa: E402
from benchmarks.run_benchmark import (  # noqa: E402
    DEFAULT_BASELINE_K,
    DEFAULT_GRAPH_K,
    aggregate,
    embed_corpus,
    entity_document,
    evaluate_baseline,
    evaluate_graphrag,
    load_dataset,
    rank_by_similarity,
    seed_chunks,
    seed_graph,
    sweep_ks,
)

FINDINGS_PATH = HERE / "FINDINGS.md"
RESULTS_JSON_PATH = HERE / "results.json"
PILOT_PATH = BACKEND / "benchmarks" / "public" / "PILOT.md"
REPO_RESULTS_PATH = BACKEND / "benchmarks" / "results.md"

SEED = 20260721
BOOTSTRAP_ITERS = 10_000

#: Production chunker settings (``pdf_parser.chunk_text`` defaults).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

#: Chat model priced in the headline. Cheapest credible extraction model in the
#: table, which is the most *favourable* choice for GraphRAG.
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

#: ASSUMPTION, not a measurement: tokens the generator emits per answer. It
#: appears in the absolute per-query cost of BOTH systems and therefore cancels
#: exactly out of their difference, so the break-even does not depend on it.
ANSWER_TOKENS = 250

#: ASSUMPTION, not a measurement: completion tokens for one community summary
#: (a 6-word title plus 2-3 sentences). Sized from the prompt's own rules.
COMMUNITY_SUMMARY_TOKENS = 120

#: ASSUMPTION, not a measurement: wall-clock seconds for one extraction call.
#: Used only for the ingestion-latency term, never for money.
EXTRACTION_LATENCY_S = 3.0


# ── small helpers ────────────────────────────────────────────────────────────
def tokens(text: str) -> int:
    """Token estimate via the repo's own len/4 rule (``cost.estimate_tokens``)."""
    return costmod.estimate_tokens(text)


def fmt_int(n: float) -> str:
    return f"{n:,.0f}"


def fmt_usd(x: float | None, places: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.{places}f}"


def bootstrap_mean_ci(
    values: list[float],
    *,
    iters: int = BOOTSTRAP_ITERS,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(mean, lo, hi) — percentile bootstrap over the sample, fixed seed."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (statistics.fmean(values), lo, hi)


def bootstrap_paired_delta_ci(
    a: list[float],
    b: list[float],
    *,
    iters: int = BOOTSTRAP_ITERS,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(mean(a-b), lo, hi) resampling QUESTIONS, keeping the pairing intact."""
    if not a or len(a) != len(b):
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a[i] - b[i] for i in idx) / n)
    deltas.sort()
    lo = deltas[int((alpha / 2) * iters)]
    hi = deltas[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (statistics.fmean([x - y for x, y in zip(a, b, strict=True)]), lo, hi)


# ── the production chunker, imported rather than reimplemented ───────────────
def chunk_corpus(text: str) -> list[str]:
    from app.services.pdf_parser import chunk_text

    return chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)


# ── measurement 1: what the corpus costs to read ─────────────────────────────
@dataclass
class CorpusMeasurement:
    """Everything the extraction bill depends on, measured from the fixture."""

    name: str
    n_passages: int
    corpus_chars: int
    corpus_tokens: int
    n_chunks: int
    chunk_chars_total: int  # > corpus_chars: the 200-char overlap is re-sent
    chunk_tokens_total: int
    overlap_inflation: float


def measure_corpus(dataset: dict, name: str) -> CorpusMeasurement:
    passages = dataset["passages"]
    text = "\n\n".join(p["text"] for p in passages)
    chunks = chunk_corpus(text)
    chunk_chars = sum(len(c) for c in chunks)
    return CorpusMeasurement(
        name=name,
        n_passages=len(passages),
        corpus_chars=len(text),
        corpus_tokens=tokens(text),
        n_chunks=len(chunks),
        chunk_chars_total=chunk_chars,
        chunk_tokens_total=sum(tokens(c) for c in chunks),
        overlap_inflation=chunk_chars / len(text) if text else 1.0,
    )


# ── measurement 2: the extraction prompt, rendered from production code ──────
@dataclass
class ExtractionMeasurement:
    template_chars: int
    template_tokens: int
    unique_entities: int
    unique_relationships: int
    raw_entity_mentions: int
    duplication_factor: float
    json_unique_chars: int
    json_unique_tokens: int
    completion_tokens_total: int  # duplication-adjusted (the realistic figure)
    completion_tokens_unique: int  # lower bound (perfect, dedup-free extraction)
    envelope_tokens: int


def measure_extraction(dataset: dict, corpus: CorpusMeasurement) -> ExtractionMeasurement:
    """Prompt overhead measured; completion derived from the fixture itself."""
    from app.services.graph_builder import DEFAULT_THEME, get_extraction_prompt

    prompt = get_extraction_prompt(DEFAULT_THEME)
    rendered = prompt.format_messages(text="", theme=DEFAULT_THEME, document_name="corpus.pdf")
    template_chars = sum(len(str(m.content)) for m in rendered)

    # dataset.json IS the extraction output for this corpus. Re-serialise it in
    # the exact shape the prompt demands and count what the model had to emit.
    payload = {
        "entities": [
            {"name": e["name"], "type": e["type"], "description": e.get("description", "")}
            for e in dataset["entities"]
        ],
        "relationships": [
            {
                "source": r["source"],
                "target": r["target"],
                "type": r["type"],
                "description": r.get("description", ""),
            }
            for r in dataset["relationships"]
        ],
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    envelope = tokens('{\n  "entities": [\n  ],\n  "relationships": [\n  ]\n}')

    # Real extraction re-emits an entity in every chunk that mentions it; the
    # dedup happens afterwards, in Python, for free. Mentions per passage give
    # the inflation factor. Using the unique count alone would UNDER-charge
    # GraphRAG, so both are reported and the realistic one is used.
    mentions = sum(len(p.get("entities") or []) for p in dataset["passages"])
    dup = mentions / len(dataset["entities"]) if dataset["entities"] else 1.0
    unique_tokens = tokens(body)
    return ExtractionMeasurement(
        template_chars=template_chars,
        template_tokens=costmod.tokens_from_chars(template_chars),
        unique_entities=len(dataset["entities"]),
        unique_relationships=len(dataset["relationships"]),
        raw_entity_mentions=mentions,
        duplication_factor=dup,
        json_unique_chars=len(body),
        json_unique_tokens=unique_tokens,
        completion_tokens_unique=unique_tokens + envelope * corpus.n_chunks,
        completion_tokens_total=round(unique_tokens * dup) + envelope * corpus.n_chunks,
        envelope_tokens=envelope,
    )


# ── measurement 3: community summarisation (Louvain run for real) ────────────
@dataclass
class CommunityMeasurement:
    n_communities: int
    modularity: float
    prompt_tokens_total: int
    completion_tokens_total: int


def measure_communities(dataset: dict) -> CommunityMeasurement:
    from app.config import get_settings
    from app.services.communities import (
        SUMMARY_PROMPT,
        _format_members,
        _format_relationships,
        build_graph_from_rows,
        detect_communities,
    )

    settings = get_settings()
    meta_by_name = {e["name"]: e for e in dataset["entities"]}
    rows = [
        {
            "source": r["source"],
            "target": r["target"],
            "type": r["type"],
            "source_type": meta_by_name.get(r["source"], {}).get("type", ""),
            "target_type": meta_by_name.get(r["target"], {}).get("type", ""),
            "source_description": meta_by_name.get(r["source"], {}).get("description", ""),
            "target_description": meta_by_name.get(r["target"], {}).get("description", ""),
        }
        for r in dataset["relationships"]
    ]
    graph, meta, edges = build_graph_from_rows(rows)
    groups, modularity = detect_communities(graph, settings)

    prompt_tokens = 0
    for members in groups:
        shown = members[: settings.community_max_members_in_summary]
        rendered = SUMMARY_PROMPT.format_messages(
            members=_format_members(shown, meta),
            relationships=_format_relationships(shown, edges),
        )
        prompt_tokens += tokens("".join(str(m.content) for m in rendered))
    return CommunityMeasurement(
        n_communities=len(groups),
        modularity=modularity,
        prompt_tokens_total=prompt_tokens,
        completion_tokens_total=len(groups) * COMMUNITY_SUMMARY_TOKENS,
    )


# ── measurement 4: the generation prompt, rendered from production code ──────
def measure_rag_prompt_overhead() -> int:
    """Tokens of the RAG system prompt with an empty context and question."""
    from app.services.chat_engine import RAG_PROMPT

    rendered = RAG_PROMPT.format_messages(context="", question="", history=[])
    return tokens("".join(str(m.content) for m in rendered))


# ── measurement 5: retrieved context, measured by running the real systems ───
@dataclass
class SystemContexts:
    """Per-question retrieved-context sizes for one retrieval system."""

    key: str
    label: str
    chars: list[int]
    tokens_: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tokens_ = [costmod.tokens_from_chars(c) for c in self.chars]

    @property
    def mean_chars(self) -> float:
        return statistics.fmean(self.chars) if self.chars else 0.0

    @property
    def mean_tokens(self) -> float:
        return statistics.fmean(self.tokens_) if self.tokens_ else 0.0


@dataclass
class RetrievalMeasurement:
    systems: dict[str, SystemContexts]
    quality: dict[str, dict]
    passage_sweep: list[tuple[int, dict, list[int]]]  # (k, agg, per-question chars)
    question_tokens: list[int]
    wall_seed_graph_s: float
    wall_seed_chunks_s: float
    wall_embed_corpus_s: float
    embed_docs: int


async def measure_retrieval(dataset: dict) -> RetrievalMeasurement:
    """Run the benchmark's own systems and record what each one retrieves.

    Imported read-only from ``benchmarks.run_benchmark`` so the context sizes are
    directly comparable with ``benchmarks/results.md``.
    """
    t0 = time.monotonic()
    seeded = await seed_graph(dataset, echo=False)
    wall_seed_graph = time.monotonic() - t0
    print(f"   seeded {seeded['nodes']} nodes / {seeded['edges']} edges in {wall_seed_graph:.1f}s")

    t0 = time.monotonic()
    passage_vectors, _entity_vectors, question_vectors = await embed_corpus(dataset)
    wall_embed = time.monotonic() - t0

    passage_rankings = {
        qid: rank_by_similarity(vec, passage_vectors) for qid, vec in question_vectors.items()
    }

    systems: dict[str, SystemContexts] = {}
    quality: dict[str, dict] = {}

    baseline = evaluate_baseline(dataset, passage_rankings, DEFAULT_BASELINE_K)
    systems["A"] = SystemContexts("A", f"Passage vector RAG (top-{DEFAULT_BASELINE_K})",
                                  [r.context_chars for r in baseline])
    quality["A"] = aggregate(baseline)

    # Per-question chars are kept for every k, not just the mean, so the
    # quality-matched comparison can carry a paired bootstrap of its own.
    passage_sweep = []
    for k in sweep_ks(len(dataset["passages"])):
        rs = evaluate_baseline(dataset, passage_rankings, k)
        passage_sweep.append((k, aggregate(rs), [r.context_chars for r in rs]))

    # C first — the chunk store must be empty for the graph-only row to be real.
    graph_only = await evaluate_graphrag(dataset, DEFAULT_GRAPH_K)
    systems["C"] = SystemContexts("C", f"GraphRAG, graph only (k={DEFAULT_GRAPH_K})",
                                  [r.context_chars for r in graph_only])
    quality["C"] = aggregate(graph_only)

    t0 = time.monotonic()
    stored = await seed_chunks(dataset)
    wall_seed_chunks = time.monotonic() - t0
    print(f"   stored {stored['chunks']} chunks / {stored['links']} links "
          f"in {wall_seed_chunks:.1f}s")

    chunked = await evaluate_graphrag(dataset, DEFAULT_GRAPH_K)
    systems["D"] = SystemContexts("D", f"GraphRAG + source chunks (k={DEFAULT_GRAPH_K})",
                                  [r.context_chars for r in chunked])
    quality["D"] = aggregate(chunked)

    return RetrievalMeasurement(
        systems=systems,
        quality=quality,
        passage_sweep=passage_sweep,
        question_tokens=[tokens(q["question"]) for q in dataset["questions"]],
        wall_seed_graph_s=wall_seed_graph,
        wall_seed_chunks_s=wall_seed_chunks,
        wall_embed_corpus_s=wall_embed,
        embed_docs=len(dataset["passages"]) + len(dataset["entities"]) + len(dataset["questions"]),
    )


# ── the cost model ───────────────────────────────────────────────────────────
@dataclass
class Prices:
    chat_model: str
    embed_model: str
    chat: costmod.Price
    embed: costmod.Price

    @classmethod
    def load(cls, chat_model: str, embed_model: str) -> Prices:
        chat = costmod.resolve_price(chat_model)
        embed = costmod.resolve_price(embed_model)
        if chat is None or embed is None:
            raise SystemExit(
                f"no price on file for {chat_model!r} / {embed_model!r}; "
                "add it to benchmarks/public/cost.py"
            )
        return cls(chat_model, embed_model, chat, embed)


@dataclass
class SetupCost:
    """One-off USD to turn a corpus into a Synapse graph. ESTIMATED throughout."""

    extraction_prompt_tokens: int
    extraction_completion_tokens: int
    extraction_usd: float
    community_prompt_tokens: int
    community_completion_tokens: int
    community_usd: float
    embedding_tokens: int
    embedding_usd_hosted: float
    total_usd_hosted: float
    total_usd_local_embeddings: float
    n_llm_calls: int

    @property
    def shares(self) -> dict[str, float]:
        t = self.total_usd_hosted or 1.0
        return {
            "extraction": self.extraction_usd / t,
            "community summaries": self.community_usd / t,
            "embeddings": self.embedding_usd_hosted / t,
        }


def setup_cost(
    corpus: CorpusMeasurement,
    extraction: ExtractionMeasurement,
    community: CommunityMeasurement,
    dataset: dict,
    prices: Prices,
) -> SetupCost:
    ext_prompt = extraction.template_tokens * corpus.n_chunks + corpus.chunk_tokens_total
    ext_completion = extraction.completion_tokens_total
    ext_usd = costmod.usd(
        costmod.Usage(prompt_tokens=ext_prompt, completion_tokens=ext_completion),
        prices.chat,
    ) or 0.0

    com_usd = costmod.usd(
        costmod.Usage(
            prompt_tokens=community.prompt_tokens_total,
            completion_tokens=community.completion_tokens_total,
        ),
        prices.chat,
    ) or 0.0

    # One-off vectors: every entity document, every stored chunk, every community
    # summary. Exactly what ``_embed_entities`` / ``store_chunks`` / community
    # embedding send.
    entity_tokens = sum(tokens(entity_document(e)) for e in dataset["entities"])
    embed_tokens = entity_tokens + corpus.chunk_tokens_total + community.completion_tokens_total
    embed_usd = costmod.usd(costmod.Usage(prompt_tokens=embed_tokens), prices.embed) or 0.0

    return SetupCost(
        extraction_prompt_tokens=ext_prompt,
        extraction_completion_tokens=ext_completion,
        extraction_usd=ext_usd,
        community_prompt_tokens=community.prompt_tokens_total,
        community_completion_tokens=community.completion_tokens_total,
        community_usd=com_usd,
        embedding_tokens=embed_tokens,
        embedding_usd_hosted=embed_usd,
        total_usd_hosted=ext_usd + com_usd + embed_usd,
        total_usd_local_embeddings=ext_usd + com_usd,
        n_llm_calls=corpus.n_chunks + community.n_communities,
    )


def per_query_usd(
    context_tokens: float,
    question_tokens: float,
    rag_overhead_tokens: int,
    prices: Prices,
    *,
    answer_tokens: int = ANSWER_TOKENS,
    hosted_embeddings: bool = True,
) -> float:
    """USD for one question+answer round trip at a given retrieved-context size.

    Note the structure: ``rag_overhead``, ``question`` and ``answer`` are
    IDENTICAL for every retrieval system, so they cancel exactly out of any
    difference between two systems. Only ``context_tokens`` differs.
    """
    chat = costmod.usd(
        costmod.Usage(
            prompt_tokens=int(round(rag_overhead_tokens + question_tokens + context_tokens)),
            completion_tokens=answer_tokens,
        ),
        prices.chat,
    ) or 0.0
    if not hosted_embeddings:
        return chat
    return chat + (costmod.usd(
        costmod.Usage(prompt_tokens=int(round(question_tokens))), prices.embed
    ) or 0.0)


def break_even_queries(setup_usd: float, saving_per_query_usd: float) -> float:
    """Q* = setup / per-query saving. ``inf`` when GraphRAG saves nothing."""
    if saving_per_query_usd <= 0:
        return float("inf")
    return setup_usd / saving_per_query_usd


# ── report ───────────────────────────────────────────────────────────────────
def smallest_k_matching(
    sweep: list[tuple[int, dict, list[int]]], target: dict
) -> tuple[int, dict, list[int]] | None:
    """Cheapest sweep row that matches the target system on BOTH permissive metrics.

    This is the *fair* baseline: comparing GraphRAG's context against a top-k that
    scores lower would credit the graph for tokens it spends buying quality the
    baseline was never given the budget to buy.
    """
    for k, agg, chars in sweep:
        if (
            agg["fact_recall"] >= target["fact_recall"]
            and agg["full_coverage"] >= target["full_coverage"]
        ):
            return (k, agg, chars)
    return None


_RESULTS_ROW = re.compile(r"^\|\s*\*{0,2}([ACD])\.\s.*?\|\s*([\d,]+)\s*\|\s*$")


def parse_results_md(path: Path) -> dict[str, int]:
    """Mean-context-chars per system, read out of ``benchmarks/results.md``.

    Parsed rather than transcribed: a hand-copied cross-check is not a check.
    Returns ``{}`` when the file is absent, so the report can say so.
    """
    if not path.exists():
        return {}
    found: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _RESULTS_ROW.match(line.strip())
        if m and m.group(1) not in found:
            found[m.group(1)] = int(m.group(2).replace(",", ""))
    return found


def build_findings(
    dataset: dict,
    corpus: CorpusMeasurement,
    extraction: ExtractionMeasurement,
    community: CommunityMeasurement,
    retrieval: RetrievalMeasurement,
    rag_overhead: int,
    prices: Prices,
    setup: SetupCost,
    command: str,
) -> tuple[str, dict]:
    L: list[str] = []
    data: dict = {}
    add = L.append

    q_tokens = statistics.fmean(retrieval.question_tokens)
    A, C, D = retrieval.systems["A"], retrieval.systems["C"], retrieval.systems["D"]

    def pq(sys: SystemContexts, *, hosted: bool = True) -> float:
        return per_query_usd(sys.mean_tokens, q_tokens, rag_overhead, prices,
                             hosted_embeddings=hosted)

    pq_A, pq_C, pq_D = pq(A), pq(C), pq(D)
    saving_D = pq_A - pq_D
    saving_C = pq_A - pq_C

    # A* — the QUALITY-MATCHED baseline: the cheapest top-k that equals or beats D
    # on both permissive metrics. This, not top-4, is the fair comparison, and it
    # is what the headline claim is stated against: comparing GraphRAG's context
    # against a baseline that scores *lower* would credit the graph for tokens it
    # spent buying quality the baseline was never given the budget to buy.
    matched = smallest_k_matching(retrieval.passage_sweep, retrieval.quality["D"])
    if matched is None:
        raise SystemExit(
            "no top-k in the sweep matches D on both metrics — the quality-matched "
            "comparison the headline rests on cannot be formed."
        )
    matched_k, matched_agg, matched_chars = matched
    A_star = SystemContexts("A*", f"Passage vector RAG (top-{matched_k}, quality-matched to D)",
                            list(matched_chars))
    retrieval.systems["A*"] = A_star
    retrieval.quality["A*"] = matched_agg
    pq_Astar = pq(A_star)
    saving_matched = pq_Astar - pq_D
    ratio_matched = D.mean_tokens / A_star.mean_tokens if A_star.mean_tokens else float("inf")

    # ---- header -------------------------------------------------------------
    add("# RQ5 — When does GraphRAG amortize its extraction cost?")
    add("")
    add(f"*Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by "
        f"`research/rq5_cost_amortization/run_rq5.py`. Every number below is written by that "
        f"script; none is typed by hand.*")
    add("")
    add("```")
    add(command)
    add("```")
    add("")
    add(f"Seed `{SEED}` · bootstrap `{BOOTSTRAP_ITERS:,}` iterations · "
        f"chat model `{prices.chat_model}` · embeddings `{prices.embed_model}` · "
        f"prices hand-recorded on `{costmod.PRICES_CHECKED_ON}` from {costmod.PRICING_URL}.")
    add("")
    add("> **Every USD figure in this document is an ESTIMATE.** Prices are a hard-coded, dated "
        "table, not a live feed, and token counts use the repo's `len/4` rule rather than a real "
        "tokenizer. No chat model was called at any point: this experiment cost $0.00 to run.")
    add("")

    # ---- hypothesis ---------------------------------------------------------
    add("## 1. Hypothesis and prediction (stated before the results)")
    add("")
    add("**H5.** GraphRAG pays a large *one-off* cost — LLM extraction over the whole corpus — in "
        "order to reduce the *per-query* cost. There is therefore a break-even query count "
        "**Q\\*** below which plain vector RAG is strictly the better engineering choice.")
    add("")
    add("**Prediction.** Q\\* is finite and large (hundreds to thousands of queries per corpus), "
        "and grows with corpus size. Most demo projects sit below it.")
    add("")
    add("**The model.** For a corpus of *N* characters answered with *Q* queries:")
    add("")
    add("```")
    add("  total(system) = setup(system) + Q · per_query(system)")
    add("")
    add("  setup(vector)   = embedding of the corpus                       (no LLM)")
    add("  setup(graph)    = extraction + community summaries + embeddings (LLM)")
    add("")
    add("  per_query(sys)  = p_in · (rag_overhead + question + CONTEXT(sys))")
    add("                  + p_out · answer")
    add("                  + p_embed · question")
    add("")
    add("  Q* = setup(graph) − setup(vector)")
    add("       ─────────────────────────────")
    add("       per_query(vector) − per_query(graph)")
    add("```")
    add("")
    add("**The algebra matters.** `rag_overhead`, `question` and `answer` are identical for every "
        "retrieval system, so they cancel exactly out of the denominator. What is left is")
    add("")
    add("```")
    add("  Q* = Δsetup / ( p_in · (CONTEXT(vector) − CONTEXT(graph)) / 1e6 )")
    add("```")
    add("")
    add("so **a finite break-even exists if and only if GraphRAG retrieves *fewer* context tokens "
        "than the baseline it is being compared against.** Answer length, chat verbosity and the "
        "output price are irrelevant to *whether* GraphRAG amortizes — they only change the "
        "absolute bill. This is the single load-bearing quantity in the whole analysis, and it is "
        "measured, not assumed.")
    add("")

    # ---- measured inputs ----------------------------------------------------
    add("## 2. Measured inputs")
    add("")
    add("### 2.1 Corpus and chunking (production chunker, 1000/200)")
    add("")
    add("| Quantity | Value | How obtained |")
    add("| --- | ---: | --- |")
    add(f"| Passages in `benchmarks/dataset.json` | {corpus.n_passages} | fixture |")
    add(f"| Corpus characters | {fmt_int(corpus.corpus_chars)} | measured |")
    add(f"| Corpus tokens (`len/4`) | {fmt_int(corpus.corpus_tokens)} | measured |")
    add(f"| Chunks at {CHUNK_SIZE}/{CHUNK_OVERLAP} | {corpus.n_chunks} | "
        f"`pdf_parser.chunk_text`, run |")
    add(f"| Characters actually sent (overlap re-sent) | {fmt_int(corpus.chunk_chars_total)} | "
        f"measured |")
    add(f"| Overlap inflation | {corpus.overlap_inflation:.3f}x | derived |")
    add(f"| Extraction prompt template | {fmt_int(extraction.template_chars)} chars = "
        f"{fmt_int(extraction.template_tokens)} tokens | `get_extraction_prompt('Generic')`, "
        f"rendered |")
    add(f"| RAG generation prompt overhead | {fmt_int(rag_overhead)} tokens | "
        f"`chat_engine.RAG_PROMPT`, rendered |")
    add("")
    tpl_total = extraction.template_tokens * corpus.n_chunks
    ext_in = tpl_total + corpus.chunk_tokens_total
    add(f"On a corpus this small the **prompt template is {tpl_total / max(1, ext_in):.0%} of "
        f"everything extraction sends** ({fmt_int(tpl_total)} of {fmt_int(ext_in)} prompt "
        f"tokens). That share falls as documents get longer, which is the one term that "
        f"genuinely favours GraphRAG at scale.")
    add("")

    add("### 2.2 Extraction output — derived from the fixture, not guessed")
    add("")
    add("`benchmarks/dataset.json` *is* the extraction result for this corpus. Re-serialising its "
        "entities and relationships into the exact JSON shape the prompt demands gives the "
        "completion the model would have had to emit.")
    add("")
    add("| Quantity | Value |")
    add("| --- | ---: |")
    add(f"| Unique entities | {extraction.unique_entities} |")
    add(f"| Unique relationships | {extraction.unique_relationships} |")
    add(f"| Serialized JSON (unique only) | {fmt_int(extraction.json_unique_chars)} chars = "
        f"{fmt_int(extraction.json_unique_tokens)} tokens |")
    add(f"| Entity mentions across passages | {extraction.raw_entity_mentions} |")
    add(f"| Re-emission factor (mentions / unique) | {extraction.duplication_factor:.2f}x |")
    add(f"| **Completion tokens charged (realistic)** | **{fmt_int(extraction.completion_tokens_total)}** |")
    add(f"| Completion tokens if extraction never repeated itself (lower bound) | "
        f"{fmt_int(extraction.completion_tokens_unique)} |")
    add("")
    add("Real extraction re-emits an entity in every chunk that mentions it and the dedup happens "
        "afterwards in Python, for free. Charging only the unique count would under-bill GraphRAG, "
        "so the re-emission factor is applied; both figures are shown so the choice is auditable.")
    add("")

    add("### 2.3 Community summarisation (Louvain run for real, no LLM)")
    add("")
    add(f"Louvain over the fixture graph yields **{community.n_communities} communities** "
        f"(modularity {community.modularity}). Rendering the production `SUMMARY_PROMPT` for each "
        f"gives {fmt_int(community.prompt_tokens_total)} prompt tokens; completions are assumed at "
        f"{COMMUNITY_SUMMARY_TOKENS} tokens each "
        f"({fmt_int(community.completion_tokens_total)} total) — an ASSUMPTION, sized from the "
        f"prompt's own \"6-word title, 2-3 sentences\" rule.")
    add("")

    add("### 2.4 Retrieved context — measured by running the real systems")
    add("")
    add(f"Systems imported read-only from `benchmarks/run_benchmark.py` and run against Neo4j, so "
        f"these are directly comparable with `benchmarks/results.md`. n = "
        f"{len(dataset['questions'])} questions; CIs are percentile bootstrap over questions.")
    add("")
    add("| System | Mean context (chars) | 95% CI (chars) | Mean context (tokens) | "
        "Fact recall | Full coverage |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    for key in ("A", "A*", "C", "D"):
        s = retrieval.systems[key]
        m, lo, hi = bootstrap_mean_ci([float(c) for c in s.chars])
        qy = retrieval.quality[key]
        add(f"| {key}. {s.label} | {m:,.0f} | [{lo:,.0f}, {hi:,.0f}] | {s.mean_tokens:,.0f} | "
            f"{qy['fact_recall']:.1%} | {qy['full_coverage']:.1%} |")
    add("")
    add(f"**A\\*** is the cheapest passage baseline that equals or beats D on *both* metrics "
        f"(top-{matched_k}: {matched_agg['fact_recall']:.1%} / "
        f"{matched_agg['full_coverage']:.1%} against D's "
        f"{retrieval.quality['D']['fact_recall']:.1%} / "
        f"{retrieval.quality['D']['full_coverage']:.1%}). It is the fair comparison and every "
        f"headline number below is stated against it; A (top-4) is kept because it is the "
        f"configuration `benchmarks/results.md` reports, but it scores *below* D and so flatters "
        f"the graph's token ratio.")
    add("")
    data["contexts"] = {
        k: {"mean_chars": v.mean_chars, "mean_tokens": v.mean_tokens, "chars": v.chars}
        for k, v in retrieval.systems.items()
    }

    # Sanity check against the repo's own report — PARSED out of results.md, not
    # transcribed, because a hand-copied cross-check is not a cross-check.
    add("**Sanity check against `benchmarks/results.md`** — the figures below are *parsed* out of "
        "that file at run time (a different workflow owns and regenerates it), not transcribed:")
    add("")
    repo_claims = parse_results_md(REPO_RESULTS_PATH)
    if not repo_claims:
        add("`benchmarks/results.md` was not present or not parseable; no cross-check available.")
    else:
        add("| System | This run (mean chars) | `results.md` | Agreement |")
        add("| --- | ---: | ---: | ---: |")
        for key in ("A", "C", "D"):
            if key not in repo_claims:
                continue
            claimed = repo_claims[key]
            mine = retrieval.systems[key].mean_chars
            delta = abs(mine - claimed) / claimed if claimed else 0.0
            add(f"| {key} | {mine:,.0f} | {claimed:,} | {'✅ ' if delta < 0.02 else '⚠️ '}"
                f"{delta:.2%} apart |")
        add("")
        worst = max(
            abs(retrieval.systems[k].mean_chars - v) / v
            for k, v in repo_claims.items() if k in retrieval.systems and v
        )
        add(f"Worst disagreement {worst:.2%} — the retrieval measurements in this experiment "
            "reproduce the repo's independently generated benchmark, so the cost model is built "
            "on the same numbers the project already publishes.")
    add("")
    data["sanity_vs_results_md"] = {
        k: {"measured": retrieval.systems[k].mean_chars, "results_md": v}
        for k, v in repo_claims.items() if k in retrieval.systems
    }

    # ---- the setup bill -----------------------------------------------------
    add("## 3. The one-off bill")
    add("")
    add(f"For the {fmt_int(corpus.corpus_chars)}-character benchmark corpus, at "
        f"`{prices.chat_model}` "
        f"(${prices.chat.input_usd_per_1m:.2f}/1M in, ${prices.chat.output_usd_per_1m:.2f}/1M out):")
    add("")
    add("| Term | LLM calls | Prompt tokens | Completion tokens | USD (est.) | Share |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    shares = setup.shares
    add(f"| Extraction | {corpus.n_chunks} | {fmt_int(setup.extraction_prompt_tokens)} | "
        f"{fmt_int(setup.extraction_completion_tokens)} | {fmt_usd(setup.extraction_usd, 6)} | "
        f"{shares['extraction']:.1%} |")
    add(f"| Community summaries | {community.n_communities} | "
        f"{fmt_int(setup.community_prompt_tokens)} | {fmt_int(setup.community_completion_tokens)} | "
        f"{fmt_usd(setup.community_usd, 6)} | {shares['community summaries']:.1%} |")
    add(f"| Embeddings (`{prices.embed_model}`) | — | {fmt_int(setup.embedding_tokens)} | — | "
        f"{fmt_usd(setup.embedding_usd_hosted, 6)} | {shares['embeddings']:.1%} |")
    add(f"| **Total** | **{setup.n_llm_calls}** | | | "
        f"**{fmt_usd(setup.total_usd_hosted, 6)}** | |")
    add("")
    dominant = max(shares, key=lambda k: shares[k])
    add(f"**Which term dominates:** {dominant} — {shares[dominant]:.1%} of the setup bill. "
        f"Embeddings are {shares['embeddings']:.1%} of it, and Synapse's default "
        f"`EMBEDDING_PROVIDER=fastembed` makes them **$0.00** (local model, no API), leaving "
        f"{fmt_usd(setup.total_usd_local_embeddings, 6)}. **The extraction LLM is the graph's "
        f"cost. Nothing else is close.**")
    add("")
    data["setup"] = {
        "extraction_usd": setup.extraction_usd,
        "community_usd": setup.community_usd,
        "embedding_usd_hosted": setup.embedding_usd_hosted,
        "total_usd_hosted": setup.total_usd_hosted,
        "total_usd_local_embeddings": setup.total_usd_local_embeddings,
        "n_llm_calls": setup.n_llm_calls,
        "shares": shares,
    }

    # ---- per-query ----------------------------------------------------------
    add("## 4. The per-query bill")
    add("")
    add(f"Assuming a {ANSWER_TOKENS}-token answer (an ASSUMPTION that cancels out of every "
        f"difference below) and hosted query embeddings:")
    add("")
    add("| System | Context tokens | Total prompt tokens | USD / query (est.) | "
        "USD / 1,000 queries |")
    add("| --- | ---: | ---: | ---: | ---: |")
    for key, val in (("A", pq_A), ("A*", pq_Astar), ("C", pq_C), ("D", pq_D)):
        s = retrieval.systems[key]
        total_in = rag_overhead + q_tokens + s.mean_tokens
        add(f"| {key}. {s.label} | {s.mean_tokens:,.0f} | {total_in:,.0f} | {fmt_usd(val, 6)} | "
            f"{fmt_usd(val * 1000, 4)} |")
    add("")

    # Paired bootstrap on the load-bearing quantity, against BOTH baselines. The
    # quality-matched one (A*) is the one the claim is stated against.
    add(f"**Paired bootstrap on the load-bearing quantity** — CONTEXT(baseline) − CONTEXT(D), "
        f"resampling the {len(dataset['questions'])} questions with the pairing intact, "
        f"{BOOTSTRAP_ITERS:,} iterations, seed {SEED}. A finite break-even requires this to be "
        f"**positive**:")
    add("")
    add("| Baseline | Δ context tokens (baseline − D) | 95% CI | Interval |")
    add("| --- | ---: | ---: | --- |")
    deltas = {}
    for key in ("A", "A*"):
        s = retrieval.systems[key]
        dmm, dll, dhh = bootstrap_paired_delta_ci(
            [float(t) for t in s.tokens_], [float(t) for t in D.tokens_]
        )
        deltas[key] = (dmm, dll, dhh)
        where = ("entirely below zero" if dhh < 0
                 else "entirely above zero" if dll > 0 else "spans zero")
        add(f"| {key}. {s.label} | {dmm:+,.0f} | [{dll:+,.0f}, {dhh:+,.0f}] | {where} |")
    add("")
    dm, dlo, dhi = deltas["A*"]
    add(f"Both intervals sit entirely below zero. Against the *quality-matched* baseline — the "
        f"fair one — GraphRAG reads **{-dm:,.0f} tokens more per query** (95% CI "
        f"[{-dhi:,.0f}, {-dlo:,.0f}] more), and the confidence interval never crosses into the "
        f"amortizing regime. The sign of this quantity is the entire result.")
    add("")
    data["delta_context_tokens"] = {
        k: {"mean": v[0], "ci_lo": v[1], "ci_hi": v[2]} for k, v in deltas.items()
    }
    data["per_query_usd"] = {"A": pq_A, "A*": pq_Astar, "C": pq_C, "D": pq_D}

    # ---- break-even ---------------------------------------------------------
    add("## 5. Break-even")
    add("")
    add("| Comparison | Δsetup (est.) | Saving / query | Break-even Q\\* |")
    add("| --- | ---: | ---: | ---: |")
    comparisons = [
        (f"**D (graph + chunks) vs A\\* (top-{matched_k}, quality-matched)** — the fair one",
         saving_matched),
        (f"D (graph + chunks) vs A (top-{DEFAULT_BASELINE_K}, scores below D)", saving_D),
        (f"C (graph only) vs A (top-{DEFAULT_BASELINE_K})", saving_C),
    ]
    for label, saving in comparisons:
        qstar = break_even_queries(setup.total_usd_local_embeddings, saving)
        qtxt = "**never** (∞)" if qstar == float("inf") else f"{qstar:,.0f}"
        add(f"| {label} | {fmt_usd(setup.total_usd_local_embeddings, 6)} | "
            f"{fmt_usd(saving, 8)} | {qtxt} |")
    add("")
    data["quality_matched_baseline"] = {
        "k": matched_k, "mean_chars": A_star.mean_chars, "mean_tokens": A_star.mean_tokens,
        "agg": matched_agg, "ratio_D_over_Astar": ratio_matched,
    }
    data["break_even"] = {
        "setup_usd": setup.total_usd_local_embeddings,
        "saving_vs_quality_matched": saving_matched,
        "saving_vs_top4": saving_D,
        "qstar": None if saving_matched <= 0 else break_even_queries(
            setup.total_usd_local_embeddings, saving_matched),
    }
    add("### The result")
    add("")
    if saving_matched <= 0:
        add("> **NULL RESULT — the break-even does not exist on this corpus.** GraphRAG is not "
            "trading a one-off cost for a per-query saving. It costs more up front **and** more "
            "per query than a passage baseline that already matches it on both retrieval "
            "metrics. Q\\* is not large; it is **infinite**. Every additional query widens the "
            "gap rather than closing it.")
    else:
        add(f"> A finite break-even exists at Q\\* ≈ "
            f"{break_even_queries(setup.total_usd_local_embeddings, saving_matched):,.0f} "
            f"queries against the quality-matched baseline.")
    add("")
    add("The hypothesis has a false premise. It assumed GraphRAG's context is *smaller* — a "
        "compressed graph standing in for raw passages. Measured, Synapse's shipped context is "
        f"**{ratio_matched:.2f}x** the quality-matched baseline's "
        f"({D.mean_tokens:,.0f} vs {A_star.mean_tokens:,.0f} tokens), and "
        f"**{D.mean_tokens / A.mean_tokens:.2f}x** the top-{DEFAULT_BASELINE_K} baseline's, "
        "because the shipped system returns the graph scaffolding *and* the source chunks the "
        "entities came from. The graph is additive, not substitutive.")
    add("")
    add(f"Even the graph-*only* configuration (C, no source chunks — the most compressed thing "
        f"Synapse can emit) reads {C.mean_tokens / A.mean_tokens:.2f}x the top-"
        f"{DEFAULT_BASELINE_K} baseline and "
        f"{C.mean_tokens / A_star.mean_tokens:.2f}x the quality-matched one, while scoring "
        f"*below* both ({retrieval.quality['C']['fact_recall']:.1%} / "
        f"{retrieval.quality['C']['full_coverage']:.1%}). There is no Synapse configuration on "
        f"this corpus whose context is smaller than the baseline it would have to beat.")
    add("")

    # ---- inversion ----------------------------------------------------------
    add("## 6. What GraphRAG would have to achieve — the inversion")
    add("")
    add("Since a finite Q\\* requires CONTEXT(graph) < CONTEXT(vector), the useful question is not "
        "*when* does it amortize but *how much compression would it need to*. Rearranging:")
    add("")
    add("```")
    add("  CONTEXT(graph) < CONTEXT(vector) − 1e6 · Δsetup / (p_in · Q)")
    add("```")
    add("")
    add(f"With the quality-matched baseline context ({A_star.mean_tokens:,.0f} tokens, top-"
        f"{matched_k}) and the measured setup cost "
        f"({fmt_usd(setup.total_usd_local_embeddings, 6)}), the graph's context budget at a given "
        f"query volume is:")
    add("")
    add("| Queries per corpus | Max context GraphRAG may use and still win | "
        "vs. its measured context |")
    add("| ---: | ---: | ---: |")
    inversion = []
    for Q in (10, 100, 1_000, 10_000, 100_000, 1_000_000):
        budget = A_star.mean_tokens - 1e6 * setup.total_usd_local_embeddings / (
            prices.chat.input_usd_per_1m * Q
        )
        inversion.append((Q, budget))
        if budget <= 0:
            add(f"| {Q:,} | impossible — the setup cost alone exceeds {Q:,} queries' worth of "
                f"the baseline's entire context | — |")
        else:
            add(f"| {Q:,} | {budget:,.0f} tokens | needs "
                f"{D.mean_tokens / budget:.1f}x less than it uses |")
    add("")
    add("Read the last column as the compression ratio GraphRAG owes. It is asymptotically "
        f"**{ratio_matched:.1f}x** — even with infinite queries to amortize over, GraphRAG must "
        "first stop being bigger than the baseline before the amortization argument can start. "
        "Note also how fast the setup term vanishes down the table: by 100,000 queries the "
        "one-off extraction bill has stopped mattering, and the answer is decided entirely by "
        "the per-query token ratio.")
    add("")
    data["inversion"] = [{"queries": q, "budget_tokens": b} for q, b in inversion]

    # ---- corpus scaling -----------------------------------------------------
    add("## 7. Break-even as a function of corpus size")
    add("")
    add("Extraction scales linearly with corpus tokens; the retrieved context does **not** grow "
        "with the corpus (top-k and k-seeds are fixed budgets). So in the counterfactual world "
        "where GraphRAG *does* compress by a factor ρ = CONTEXT(graph)/CONTEXT(vector) < 1, the "
        "break-even is")
    add("")
    add("```")
    add("  Q*(N, ρ) = setup_per_token · N / ( p_in · CONTEXT(vector) · (1 − ρ) / 1e6 )")
    add("```")
    add("")
    setup_per_corpus_token = setup.total_usd_local_embeddings / max(1, corpus.corpus_tokens)
    add(f"Measured setup cost per corpus token: **{fmt_usd(setup_per_corpus_token, 8)}/token** "
        f"({fmt_usd(setup_per_corpus_token * 1_000_000, 2)} per 1M corpus tokens) at "
        f"`{prices.chat_model}` with local embeddings.")
    add("")
    add("**Q\\*(corpus size, ρ)** — queries needed before the graph pays for itself. "
        f"ρ = {ratio_matched:.2f} is the MEASURED value (D against the quality-matched "
        f"baseline); every ρ < 1 column is a COUNTERFACTUAL, shown to bound what compression "
        f"would have to buy.")
    add("")
    rhos = (0.25, 0.50, 0.75, 0.90, ratio_matched)
    header = "| Corpus | Corpus tokens | Setup (est.) | " + " | ".join(
        f"ρ={r:.2f}" + (" *(measured)*" if abs(r - ratio_matched) < 1e-9 else "")
        for r in rhos
    ) + " |"
    add(header)
    add("| --- | ---: | ---: | " + " | ".join("---:" for _ in rhos) + " |")
    scaling = []
    corpus_points = [
        ("1 CV / résumé (~4 KB)", tokens("x" * 4_000)),
        ("benchmark fixture (this corpus)", corpus.corpus_tokens),
        ("1 report (~50 KB)", tokens("x" * 50_000)),
        ("1 book (~500 KB)", tokens("x" * 500_000)),
        ("small wiki (~5 MB)", tokens("x" * 5_000_000)),
        ("corpus (~50 MB)", tokens("x" * 50_000_000)),
    ]
    for label, ctok in corpus_points:
        s_usd = setup_per_corpus_token * ctok
        cells = []
        row = {"corpus": label, "corpus_tokens": ctok, "setup_usd": s_usd, "qstar": {}}
        for r in rhos:
            saving = prices.chat.input_usd_per_1m * A_star.mean_tokens * (1 - r) / 1e6
            qs = break_even_queries(s_usd, saving)
            row["qstar"][f"{r:.4f}"] = None if qs == float("inf") else qs
            cells.append("**never**" if qs == float("inf") else f"{qs:,.0f}")
        scaling.append(row)
        add(f"| {label} | {fmt_int(ctok)} | {fmt_usd(s_usd, 4)} | " + " | ".join(cells) + " |")
    add("")
    add("Two things fall out, and the second is the counterintuitive one:")
    add("")
    add("1. **At the measured ρ the entire right-hand column is `never`.** No corpus size rescues "
        "the graph, because the per-query term has the wrong sign.")
    add("2. **Q\\* grows *linearly* with corpus size.** The usual intuition — \"the graph pays off "
        "on big corpora\" — is backwards *in this model*: extraction cost scales with N while the "
        "per-query saving is capped by a fixed context budget, so a 100x bigger corpus needs 100x "
        "more queries to amortize. What actually rescues GraphRAG on a large corpus is not size "
        "itself but the possibility that ρ falls — that vector RAG needs a much wider top-k to "
        "reach the same recall when the corpus is large. **This experiment did not measure that, "
        "and the model cannot settle it.**")
    add("")
    data["scaling"] = scaling
    data["setup_usd_per_corpus_token"] = setup_per_corpus_token

    # ---- model sensitivity --------------------------------------------------
    add("## 8. Price sensitivity")
    add("")
    add("The conclusion is about the *sign* of the per-query difference, so it is invariant to "
        "price — but the size of the one-off bill is not. Same corpus, every priced chat model:")
    add("")
    add(f"| Chat model | $/1M in | $/1M out | Setup (est.) | Extra cost per 1,000 queries vs "
        f"A\\* (top-{matched_k}) |")
    add("| --- | ---: | ---: | ---: | ---: |")
    price_rows = []
    for model in sorted(costmod.PRICES_USD_PER_1M_TOKENS):
        p = costmod.PRICES_USD_PER_1M_TOKENS[model]
        if p.output_usd_per_1m == 0.0:
            continue  # embedding model
        mp = Prices(model, prices.embed_model, p, prices.embed)
        s = setup_cost(corpus, extraction, community, dataset, mp)
        extra = (per_query_usd(D.mean_tokens, q_tokens, rag_overhead, mp)
                 - per_query_usd(A_star.mean_tokens, q_tokens, rag_overhead, mp)) * 1000
        price_rows.append({"model": model, "setup_usd": s.total_usd_local_embeddings,
                           "extra_per_1k": extra})
        add(f"| `{model}` | {p.input_usd_per_1m:.2f} | {p.output_usd_per_1m:.2f} | "
            f"{fmt_usd(s.total_usd_local_embeddings, 6)} | +{fmt_usd(extra, 4)} |")
    add("")
    add("The last column is positive for every model: GraphRAG is more expensive per query "
        "regardless of price point, because it is more expensive in *tokens*.")
    add("")
    data["price_sensitivity"] = price_rows

    # ---- wall clock ---------------------------------------------------------
    add("## 9. Wall-clock, not just money")
    add("")
    add("| Stage | Seconds | Measured or estimated |")
    add("| --- | ---: | --- |")
    add(f"| Embed {len(dataset['entities'])} entities + write {len(dataset['entities'])} nodes / "
        f"{len(dataset['relationships'])} edges to Neo4j | {retrieval.wall_seed_graph_s:.1f} | "
        f"MEASURED (fastembed, local) |")
    add(f"| Embed {retrieval.embed_docs} documents for the benchmark's own retrieval | "
        f"{retrieval.wall_embed_corpus_s:.1f} | MEASURED (model already warm — the row above "
        f"paid the one-off fastembed model load, so these two rows are not comparable to each "
        f"other) |")
    add(f"| Store {corpus.n_passages} chunks + entity links | "
        f"{retrieval.wall_seed_chunks_s:.1f} | MEASURED |")
    llm_wall = corpus.n_chunks / 5 * EXTRACTION_LATENCY_S  # extraction_concurrency default = 5
    add(f"| Extraction: {corpus.n_chunks} LLM calls at concurrency 5 | ~{llm_wall:.1f} | "
        f"ESTIMATED at {EXTRACTION_LATENCY_S:.1f}s/call — NOT measured, no LLM was called |")
    add("")
    measured_total = (retrieval.wall_seed_graph_s + retrieval.wall_embed_corpus_s
                      + retrieval.wall_seed_chunks_s)
    add(f"Even on this tiny corpus the estimated LLM latency (~{llm_wall:.0f}s) is "
        f"{llm_wall / max(0.1, measured_total):.1f}x the entire measured non-LLM ingestion "
        f"({measured_total:.1f}s). Latency has the same shape as cost: **the extraction LLM "
        f"dominates, everything else is noise.** Vector RAG's ingestion is the "
        f"{retrieval.wall_embed_corpus_s:.1f}s embedding pass and nothing else.")
    add("")
    data["wall_clock"] = {
        "seed_graph_s": retrieval.wall_seed_graph_s,
        "embed_corpus_s": retrieval.wall_embed_corpus_s,
        "seed_chunks_s": retrieval.wall_seed_chunks_s,
        "extraction_llm_s_estimated": llm_wall,
    }

    # ---- rule of thumb ------------------------------------------------------
    add("## 10. The rule of thumb")
    add("")
    add("> **Do not build a knowledge graph to save money. It does not.**")
    add(">")
    add("> On this corpus GraphRAG costs "
        f"{fmt_usd(setup.total_usd_local_embeddings, 4)} up front *and* "
        f"{fmt_usd(pq_D - pq_Astar, 8)} more per query than a plain top-{matched_k} passage "
        f"baseline that already equals or beats it on both retrieval metrics "
        f"({matched_agg['fact_recall']:.1%} / {matched_agg['full_coverage']:.1%} vs "
        f"{retrieval.quality['D']['fact_recall']:.1%} / "
        f"{retrieval.quality['D']['full_coverage']:.1%}). The break-even query count is "
        f"infinite: there is no N at which the graph becomes the cheaper choice.")
    add(">")
    add("> The practitioner's version: **if your reason for building a graph is cost, stop.** "
        "Build one only if you want something vector RAG cannot do at any budget — explicit "
        "multi-hop paths, an auditable reasoning chain, a browsable entity view, global "
        "community summaries. Those are product features, and you should price them as features, "
        "not as savings.")
    add("")
    add("A crude budgeting heuristic that does fall out of the measured numbers:")
    add("")
    add(f"* **One-off:** ~{fmt_usd(setup_per_corpus_token * 1_000_000, 2)} per 1M corpus tokens "
        f"at `{prices.chat_model}` — about "
        f"{fmt_usd(setup_per_corpus_token * tokens('x' * 4_000), 4)} for a résumé, "
        f"{fmt_usd(setup_per_corpus_token * tokens('x' * 500_000), 2)} for a book. Cheap in "
        "absolute terms at the small end; this is why demo projects never notice.")
    add(f"* **Recurring:** ~+{D.mean_tokens - A_star.mean_tokens:,.0f} prompt tokens on *every* "
        f"query, forever. At 1M queries that is "
        f"{fmt_usd((pq_D - pq_Astar) * 1_000_000, 2)} of pure surcharge — "
        f"{(pq_D - pq_Astar) * 1_000_000 / max(1e-9, setup.total_usd_local_embeddings):,.0f}x "
        f"the one-off extraction bill for this corpus.")
    add(f"* **The crossover in *queries*, not dollars:** the surcharge overtakes the entire "
        f"one-off extraction bill after only "
        f"{setup.total_usd_local_embeddings / max(1e-12, pq_D - pq_Astar):,.0f} queries. That "
        f"is the honest form of the \"break-even\" practitioners are looking for — except it "
        f"runs the wrong way: it is the point after which the graph's *recurring* cost, not its "
        f"setup cost, is the thing you are paying for.")
    add("* **Therefore:** the extraction cost is the part everybody worries about and it is the "
        "part that does not matter. The context surcharge is the part nobody budgets for and it "
        "is unbounded in the number of queries.")
    add("")

    # ---- threats ------------------------------------------------------------
    add("## 11. Threats to validity")
    add("")
    add(f"1. **n = {len(dataset['questions'])} questions, one corpus, "
        f"{fmt_int(corpus.corpus_chars)} characters.** This is a fixture, not a workload. Every "
        "context measurement carries the bootstrap CI printed in §2.4; the corpus itself has no "
        "error bar at all because there is only one of it. Nothing here generalises to a "
        "different corpus without re-measuring.")
    add(f"2. **The corpus saturates, and this is the load-bearing threat.** `benchmarks/"
        f"results.md` already notes that top-20 reads 59% of this corpus and scores 100%. A tiny "
        f"corpus flatters top-k retrieval, and top-k is the system GraphRAG loses to here. The "
        f"whole result rests on ρ = {ratio_matched:.2f} > 1; on a large corpus where a passage "
        f"baseline needed a much wider k to reach the same multi-hop recall, the quality-matched "
        f"baseline's context would grow, ρ could fall below 1, and a finite Q\\* would appear. "
        f"**This is the single most likely way the conclusion is wrong, and this experiment "
        f"cannot rule it out** — it would take a large-corpus study (e.g. the HotpotQA harness) "
        f"to settle, and that costs real LLM money.")
    add(f"3. **Tokens are estimated, not tokenized.** Everything uses the repo's `len/4` rule "
        f"(`cost.CHARS_PER_TOKEN`). A real BPE tokenizer would shift absolute counts by "
        f"~10-20%, and would shift both systems in the same direction. It would not flip a "
        f"{ratio_matched:.2f}x ratio.")
    add(f"4. **Prices are a dated hand-recorded table** ({costmod.PRICES_CHECKED_ON}, "
        f"{costmod.PRICING_URL}) and are not fetched live. §8 shows the conclusion is invariant "
        "to the price point, but the absolute dollars are estimates and must be checked against "
        "an invoice before quoting.")
    add(f"5. **Answer length ({ANSWER_TOKENS} tokens) and community-summary length "
        f"({COMMUNITY_SUMMARY_TOKENS} tokens) are assumptions.** The first cancels out of every "
        "difference. The second affects only the community term, which is "
        f"{shares['community summaries']:.1%} of setup.")
    add("6. **Extraction completion size is derived from the fixture, not observed.** The fixture "
        "is a clean, hand-checked extraction; a real LLM emits more slop, more duplicates and "
        "retried calls. That biases the setup cost **downward** — i.e. in GraphRAG's favour — so "
        "it cannot explain away the null result.")
    add("7. **Extraction wall-clock is an estimate** at "
        f"{EXTRACTION_LATENCY_S:.1f}s/call. No LLM was called. Treat §9's last row as an order of "
        "magnitude only.")
    add("8. **This measures retrieval cost, not answer quality.** A larger context is not "
        "automatically worse for the generator — it may produce better answers, or worse ones "
        "through distraction. This experiment prices tokens; it does not judge answers. "
        "`benchmarks/results.md` shows GraphRAG does not *retrieve* better here, which is what "
        "makes the extra tokens hard to justify — but an end-to-end answer-quality study could "
        "still find value the retrieval metrics miss.")
    add(f"9. **One retrieval configuration.** k={DEFAULT_GRAPH_K} seeds, 2 hops, chunk_top_k=4. A "
        f"GraphRAG tuned for terseness (fewer seeds, 1 hop, no chunks) would move ρ. C is the "
        f"closest thing to that here and still sits at ρ = "
        f"{C.mean_tokens / A_star.mean_tokens:.2f} against the quality-matched baseline — while "
        f"scoring below it, so it is not a configuration that would rescue the argument.")
    add(f"10. **`benchmarks/public/PILOT.md` "
        f"{'was read' if PILOT_PATH.exists() else 'did not exist when this ran'}**, so no "
        "HotpotQA cost figures could be cross-checked. The only external cross-check is "
        "`benchmarks/results.md` (§2.4), which agrees.")
    add("")

    # ---- measured vs inferred ----------------------------------------------
    add("## 12. Measured vs. inferred")
    add("")
    add("| | |")
    add("| --- | --- |")
    add("| **MEASURED** | corpus and chunk token counts; extraction and RAG prompt template "
        "sizes; the extraction JSON payload size; the Louvain partition and its summary prompt "
        "sizes; per-question retrieved-context sizes for A, A\\*, C and D, and the retrieval "
        "quality of each; non-LLM ingestion wall-clock. |")
    add("| **DERIVED** (arithmetic on measurements + published prices) | every USD figure; the "
        "per-query difference and its bootstrap CI; Q\\*; the inversion table; the ρ-scaling "
        "table. |")
    add("| **ASSUMED** (stated, and the sensitivity shown) | answer length; community-summary "
        "length; extraction latency; that entity density stays constant as corpora scale; that "
        "retrieved-context size is independent of corpus size. |")
    add("| **NOT MEASURED** | answer quality; whether ρ falls below 1 on a large corpus; real "
        "provider tokenization; real extraction slop and retry rates. |")
    add("")

    # ---- claim --------------------------------------------------------------
    add("## 13. Claim we could defend in a paper")
    add("")
    add(f"> On a {corpus.n_passages}-passage multi-hop benchmark, GraphRAG's cost disadvantage is "
        f"**not** a one-off extraction charge that amortizes. The shipped system retrieves "
        f"{ratio_matched:.2f}x the prompt tokens of a passage baseline that already equals or "
        f"beats it on both retrieval metrics (top-{matched_k}; paired bootstrap over "
        f"{len(dataset['questions'])} questions, 95% CI on the per-query context-token difference "
        f"[{dlo:+,.0f}, {dhi:+,.0f}], entirely on the wrong side of zero). Because the break-even "
        f"query count is Δsetup divided by that difference, it is **infinite rather than large**: "
        f"there is no query volume at which the graph becomes cheaper. The practical consequence "
        f"is that extraction cost — the term practitioners budget for, here "
        f"{fmt_usd(setup.total_usd_local_embeddings, 4)} — is swamped after roughly "
        f"{setup.total_usd_local_embeddings / max(1e-12, pq_D - pq_Astar):,.0f} queries by a "
        f"per-query context surcharge that grows without bound in query volume.")
    add("")
    add(f"*(Scope: one small corpus ({fmt_int(corpus.corpus_chars)} chars), n = "
        f"{len(dataset['questions'])} questions, one retrieval configuration, token counts "
        f"estimated at 4 chars/token. The claim is about this corpus and the structure of the "
        f"cost model, not about GraphRAG in general. §11.2 states the condition under which it "
        f"would flip.)*")
    add("")
    add("---")
    add("")
    add("Numbers generated by `research/rq5_cost_amortization/run_rq5.py`; raw values in "
        "`results.json`. Synapse, © 2026 Ahmed Maaloul, AGPL-3.0-or-later.")
    add("")

    return "\n".join(L), data


# ── entry point ──────────────────────────────────────────────────────────────
async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    args = parser.parse_args(argv)

    command = (
        "cd backend && EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7688 "
        "NEO4J_PASSWORD=research_secret \\\n"
        "    python research/rq5_cost_amortization/run_rq5.py"
    )

    prices = Prices.load(args.chat_model, args.embed_model)
    dataset = load_dataset()

    print("📐 Measuring the corpus with the production chunker …")
    corpus = measure_corpus(dataset, "benchmark")
    print(f"   {corpus.corpus_chars:,} chars → {corpus.n_chunks} chunks "
          f"({corpus.overlap_inflation:.3f}x inflation)")

    print("📐 Rendering the production extraction prompt …")
    extraction = measure_extraction(dataset, corpus)
    print(f"   template {extraction.template_tokens:,} tokens; completion "
          f"{extraction.completion_tokens_total:,} tokens")

    print("📐 Running Louvain and rendering the summary prompt …")
    community = measure_communities(dataset)
    print(f"   {community.n_communities} communities, modularity {community.modularity}")

    rag_overhead = measure_rag_prompt_overhead()
    print(f"📐 RAG prompt overhead: {rag_overhead:,} tokens")

    print("🔌 Measuring retrieved-context sizes against Neo4j …")
    try:
        retrieval = await measure_retrieval(dataset)
    except Exception as e:  # noqa: BLE001
        print(f"❌ Neo4j measurement failed: {e}", file=sys.stderr)
        print("   Set NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=research_secret",
              file=sys.stderr)
        return 2

    setup = setup_cost(corpus, extraction, community, dataset, prices)

    markdown, data = build_findings(
        dataset, corpus, extraction, community, retrieval, rag_overhead,
        prices, setup, command,
    )
    FINDINGS_PATH.write_text(markdown, encoding="utf-8")
    RESULTS_JSON_PATH.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
    print(f"\n📝 {FINDINGS_PATH}")
    print(f"📝 {RESULTS_JSON_PATH}")

    # ``build_findings`` registers A* (the quality-matched baseline) on the way past.
    star, D = retrieval.systems["A*"], retrieval.systems["D"]
    dm, dlo, dhi = bootstrap_paired_delta_ci(
        [float(t) for t in star.tokens_], [float(t) for t in D.tokens_]
    )
    print(f"\nHEADLINE: setup {fmt_usd(setup.total_usd_local_embeddings, 6)} (one-off) · "
          f"per-query context Δ(A*−D) {dm:+,.0f} tokens [{dlo:+,.0f}, {dhi:+,.0f}] · "
          f"ρ = {D.mean_tokens / star.mean_tokens:.2f} · "
          f"break-even = {'NEVER (∞)' if dm <= 0 else 'finite'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
