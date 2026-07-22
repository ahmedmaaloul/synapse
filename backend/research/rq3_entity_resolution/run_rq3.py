# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
RQ3 — How does entity-resolution aggressiveness affect retrieval?

HYPOTHESIS (stated before any result)
    Entity resolution has a precision/recall sweet spot. Too conservative and
    near-duplicate nodes survive, fragmenting each entity's neighbourhood and
    splitting the evidence a multi-hop walk needs; too aggressive and distinct
    entities are merged, collapsing the graph and manufacturing false paths.
    There is therefore an interior optimum in the
    (``entity_resolution_threshold`` × ``entity_resolution_name_threshold``)
    plane, and a catastrophic-collapse regime beyond it.

PREDICTION
    Downstream fact recall, as a function of the two thresholds, is an inverted
    U: it rises as merging starts to repair fragmentation, peaks somewhere
    strictly inside the grid, and falls off a cliff once merging destroys
    structure.

METHOD (all of it LLM-free; the only model is the local fastembed embedder)
    Two corpora are swept over the same threshold grid.

    CLEAN   ``benchmarks/dataset.json`` exactly as it ships — 79 entities, 81
            relationships, 34 passages, 14 multi-hop questions. Its entity list
            is already canonical (it was hand-curated), so this corpus answers
            one question only: *on a graph with no duplicates, does the knob do
            anything at all, and how much head-room do the shipped defaults
            have before they start doing damage?*

    NOISY   The same corpus with deterministic, seed-free surface-form noise
            injected, simulating what an LLM extractor actually produces:

              • DUPLICATES. Every entity with degree >= 2 that admits a variant
                gets one extra node:
                  - PERSON with >= 2 tokens -> the surname alone
                    ("Geoffrey Hinton" -> "Hinton");
                  - otherwise, a one-character deletion inside the name
                    ("Backpropagation" -> "Backproagation").
                Every second incident edge of the original entity is re-pointed
                onto the variant, and every second chunk link likewise, so the
                entity's neighbourhood and its source evidence are genuinely
                split in two. Half the variants inherit the source entity's
                description (a well-described re-extraction), half carry none
                (a bare mention) — which is what spreads the duplicate pairs
                across the embedding-similarity axis instead of parking them
                all at cosine ~1.
              • ADVERSARIAL DECOYS. A small set of *distinct* entities that
                share surface form with an existing one — four people who share
                a surname with a real person in the graph, plus near-miss
                concepts and organisations. Each is wired into the graph with
                its own edge, so a wrong merge does not merely lose a node: it
                grafts a foreign edge onto a real entity and invents a path.

    Variants are always strictly shorter than the entity they duplicate, so
    ``choose_canonical`` always restores the original name on a correct merge;
    and no injected name contains a required fact as a substring, so the
    injection can never create a spurious scoring hit. Both properties are
    asserted at construction time.

    For every cell of the grid the script:
      1. runs the *production* resolver, ``entity_resolution.find_duplicate_
         clusters``, at that cell's two thresholds;
      2. scores the clusters against the known ground truth (pair precision /
         recall / F1) and records concrete correct and wrong merges;
      3. collapses the graph through the *production* ingest path,
         ``graph_builder._collapse_duplicate_entities``;
      4. writes the collapsed graph to Neo4j through the production write path
         and measures node/edge counts, components and modularity;
      5. runs the *production* retrieval path,
         ``chat_engine.retrieve_subgraph``, on all 14 benchmark questions,
         scored by ``benchmarks/run_benchmark.py``'s own rules — graph-only
         (system C, permissive rule) and graph + source chunks (system D, both
         the permissive and the strict rule).

    Cells whose resolver output is identical share one Neo4j evaluation: the
    downstream result is a deterministic function of the merge set, so
    re-running it would only burn wall-clock. The cache key is the merge set
    itself, and the number of distinct merge sets is reported.

    Uncertainty: n = 14 questions, so every headline number carries a
    percentile bootstrap 95% CI over questions (10,000 resamples, fixed seed),
    and every comparison between two cells is a *paired* bootstrap on the
    per-question differences.

Run it (needs Neo4j; no API key, no network, no LLM):

    cd backend && NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=research_secret \\
        EMBEDDING_PROVIDER=fastembed \\
        python -m research.rq3_entity_resolution.run_rq3

WARNING: like the benchmark harness it imports, this replaces the
``:Entity`` / ``:Community`` / ``:Chunk`` contents of the target database, once
per evaluated cell.

Exit codes: 0 success · 2 Neo4j unreachable · 3 a self-check failed.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import os
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# The benchmark harness is imported read-only and never modified: its dataset,
# its retrieval evaluator and its scoring rules are what make these numbers
# comparable with the shipped benchmark report.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:  # pragma: no cover - convenience for direct runs
    sys.path.insert(0, str(_BACKEND_ROOT))

from benchmarks import run_benchmark as rb  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.json"
FINDINGS_PATH = HERE / "FINDINGS.md"

# ── The grid ─────────────────────────────────────────────────────────────────
#: Shipped defaults, from ``app/config.py``. Marked on every table and curve.
DEFAULT_VECTOR = 0.93
DEFAULT_NAME = 0.87

#: 1.01 is "never merge": no cosine and no name similarity can reach it, so
#: that row/column is the resolution-disabled control rather than a guess at one.
OFF = 1.01

VECTOR_GRID: tuple[float, ...] = (
    OFF, 0.99, 0.97, 0.95, 0.93, 0.90, 0.87, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.30, 0.00,
)
#: Deliberately dense between 0.93 and 0.98: the lexical scores of realistic
#: duplicates pile up there (a one-character typo in an n-character name scores
#: 1 - 1/n, and the containment rule pins prefix/subsequence pairs at exactly
#: ``CONTAINMENT_SCORE`` = 0.95), so that is where the lexical gate can act.
NAME_GRID: tuple[float, ...] = (
    OFF, 0.98, 0.96, 0.95, 0.94, 0.93, 0.90, 0.87, 0.80, 0.70, 0.60, 0.50, 0.30, 0.00,
)

#: Seed entities for retrieval — the production default, not tuned here.
GRAPH_K = rb.DEFAULT_GRAPH_K
#: Bootstrap configuration. Fixed seed, so the CIs are reproducible.
BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260719

#: An entity shorter than this is not given a typo variant: deleting one of six
#: characters is not a surface-form variant, it is a different word.
MIN_TYPO_LEN = 8

#: The research Neo4j is Community edition — one database, no namespaces — and it
#: is shared with other experiments running concurrently. A neighbour that seeds
#: or clears the graph in the middle of one of our 14-question evaluations would
#: silently corrupt that cell's numbers. So every evaluation is *verified* against
#: the state this script wrote (entity count, chunk count, and the harness's own
#: contamination check) immediately after it runs, and re-run from a fresh seed if
#: the state drifted. Retries are counted and reported; a cell that cannot be
#: measured cleanly this many times aborts the run rather than being reported.
INTERFERENCE_RETRIES = 40
#: Seconds to wait before re-seeding after a detected external write, so two
#: experiments hammering the same database do not lock-step with each other.
INTERFERENCE_BACKOFF = 4.0
#: Counts of detected external interference, reported in FINDINGS.md.
INTERFERENCE: Counter = Counter()


# ── Corpus construction ──────────────────────────────────────────────────────
#: Distinct entities that share surface form with something already in the
#: graph. Written by hand *before* any result was seen, chosen so that (a) they
#: are plausible members of this corpus, (b) none of them contains a required
#: fact as a substring, and (c) four of them reproduce the specific failure the
#: research question asks about: two different people who share a surname.
DECOYS: tuple[tuple[str, str, str, str, str], ...] = (
    ("Simon Hinton", "PERSON", "Cambridge psychologist studying human memory consolidation",
     "RESEARCHED", "Reinforcement Learning"),
    ("Rachel Minsky", "PERSON", "Contemporary sculptor working in welded steel",
     "EXHIBITED_AT", "Bletchley Park"),
    ("Julian Babbage", "PERSON", "Edwardian railway engineer and bridge surveyor",
     "SURVEYED", "Colossus"),
    ("Tomas Krizhevsky", "PERSON", "Czech chess grandmaster of the interwar period",
     "PLAYED_AGAINST", "Garry Kasparov"),
    ("Peter Sutskever", "PERSON", "Latvian cellist and conservatory teacher",
     "PERFORMED_AT", "CERN"),
    ("Knowledge Base", "CONCEPT", "Curated store of hand-authored facts and rules",
     "USED_BY", "Expert Systems"),
    ("Vector Index", "CONCEPT", "Data structure that accelerates nearest-neighbour lookup",
     "IMPLEMENTED_IN", "FAISS"),
    ("Deep Green", "PROJECT", "Cancelled naval sonar programme of the late 1970s",
     "TESTED_BY", "Bell Labs"),
    ("Stanford Research Institute", "UNIVERSITY", "Independent Menlo Park contract research body",
     "HOSTED", "MYCIN"),
    ("Dartmouth Reunion", "EVENT", "Annual alumni gathering held every summer",
     "ORGANIZED_BY", "Rachel Minsky"),
    ("Punched Tape", "TOOL", "Paper strip encoding characters as rows of holes",
     "USED_IN", "Colossus"),
)


def required_facts(dataset: dict) -> set[str]:
    """Every entity name any benchmark question needs retrieved."""
    facts: set[str] = set()
    for question in dataset["questions"]:
        facts.update(question["required_facts"])
    return facts


def entity_degrees(dataset: dict) -> Counter:
    """Undirected degree of each entity in the shipped relationship list."""
    degrees: Counter = Counter()
    for rel in dataset["relationships"]:
        degrees[rel["source"]] += 1
        degrees[rel["target"]] += 1
    return degrees


def make_variant(entity: dict) -> tuple[str | None, str]:
    """A realistic alternate surface form for one entity, or ``(None, "")``.

    Two rules, both strictly length-reducing so ``choose_canonical`` always
    picks the original name back when the pair is merged:

      * a PERSON written by its surname — the single most common thing an LLM
        extractor does across chunks;
      * anything else, one character deleted inside the name — extraction /
        OCR noise. The deleted character is always alphanumeric, so a
        multi-token name never loses the space that separates its tokens.
    """
    name = str(entity.get("name", ""))
    tokens = name.split()
    if entity.get("type") == "PERSON" and len(tokens) >= 2 and len(tokens[-1]) >= 3:
        return " ".join(tokens[1:]), "surname"
    if len(name) >= MIN_TYPO_LEN:
        for offset in range(4):
            i = len(name) // 2 + offset
            if i < len(name) - 1 and name[i].isalnum():
                return name[:i] + name[i + 1:], "typo"
    return None, ""


@dataclass
class Corpus:
    """One graph under test, plus the ground truth needed to score merges."""

    key: str
    label: str
    entities: list[dict]
    relationships: list[dict]
    #: Entity names attached to each passage, positionally aligned with
    #: ``dataset["passages"]`` — what ``chunk_store.store_chunks`` links.
    chunk_entities: list[list[str]]
    #: ``{frozenset({variant, canonical})}`` — the merges that *should* happen.
    true_pairs: set[frozenset]
    #: One record per injected duplicate, for the report.
    injected: list[dict] = field(default_factory=list)
    decoys: list[str] = field(default_factory=list)


def clean_corpus(dataset: dict) -> Corpus:
    """The shipped dataset, untouched. No duplicates exist, so no merge is right."""
    return Corpus(
        key="clean",
        label="CLEAN — shipped benchmark corpus (already canonical)",
        entities=copy.deepcopy(dataset["entities"]),
        relationships=copy.deepcopy(dataset["relationships"]),
        chunk_entities=[list(p.get("entities") or []) for p in dataset["passages"]],
        true_pairs=set(),
    )


def noisy_corpus(dataset: dict) -> Corpus:
    """The shipped dataset with duplicates and adversarial decoys injected."""
    degrees = entity_degrees(dataset)
    facts = required_facts(dataset)
    entities = copy.deepcopy(dataset["entities"])
    existing = {e["name"] for e in entities}

    variant_of: dict[str, str] = {}
    injected: list[dict] = []
    for index, entity in enumerate(entities):
        if degrees[entity["name"]] < 2:
            continue
        variant, kind = make_variant(entity)
        if not variant or variant in existing or len(variant) >= len(entity["name"]):
            continue
        existing.add(variant)
        variant_of[entity["name"]] = variant
        # Alternating description policy — see the module docstring. It is the
        # single design choice that puts the duplicate pairs on both sides of
        # the default cosine threshold instead of all on one side.
        carries_description = index % 2 == 0
        injected.append(
            {
                "canonical": entity["name"],
                "variant": variant,
                "kind": kind,
                "description": entity["description"] if carries_description else "",
                "type": entity["type"],
                "carries_description": carries_description,
            }
        )

    # Split every duplicated entity's edges: every second incident edge, in
    # dataset order, is re-pointed onto the variant.
    seen: Counter = Counter()
    relationships: list[dict] = []
    for rel in dataset["relationships"]:
        rewritten = dict(rel)
        for side in ("source", "target"):
            name = rel[side]
            if name in variant_of:
                position = seen[name]
                seen[name] += 1
                if position % 2 == 1:
                    rewritten[side] = variant_of[name]
        relationships.append(rewritten)

    # …and split the source-chunk links the same way, so the duplicate holds
    # half the evidence too.
    chunk_entities: list[list[str]] = []
    for passage_index, passage in enumerate(dataset["passages"]):
        names: list[str] = []
        for position, name in enumerate(passage.get("entities") or []):
            if name in variant_of and (passage_index + position) % 2 == 1:
                names.append(variant_of[name])
            else:
                names.append(name)
        chunk_entities.append(names)

    for record in injected:
        entities.append(
            {
                "name": record["variant"],
                "type": record["type"],
                "description": record["description"],
            }
        )
    for name, entity_type, description, rel_type, target in DECOYS:
        entities.append({"name": name, "type": entity_type, "description": description})
        relationships.append(
            {"source": name, "target": target, "type": rel_type,
             "description": "Injected decoy edge (RQ3)"}
        )

    corpus = Corpus(
        key="noisy",
        label="NOISY — duplicates + adversarial decoys injected",
        entities=entities,
        relationships=relationships,
        chunk_entities=chunk_entities,
        true_pairs={frozenset((r["canonical"], r["variant"])) for r in injected},
        injected=injected,
        decoys=[d[0] for d in DECOYS],
    )
    _assert_injection_is_scoring_safe(corpus, facts, dataset)
    return corpus


def _assert_injection_is_scoring_safe(corpus: Corpus, facts: set[str], dataset: dict) -> None:
    """The injection must not be able to create or destroy a scoring hit by itself.

    ``run_benchmark.facts_found`` credits a required fact when its name occurs
    anywhere in the retrieved context, case-insensitively. An injected node
    whose *name* or *description* contained a required fact would therefore
    score that fact without retrieving any evidence for it, and every number
    downstream would be junk. This is checked, not assumed.
    """
    original = {e["name"] for e in dataset["entities"]}
    added = [e for e in corpus.entities if e["name"] not in original]
    lowered = {f.lower() for f in facts}
    for entity in added:
        blob = f"{entity['name']} {entity.get('description', '')}".lower()
        offending = sorted(f for f in lowered if f in blob)
        if offending:
            raise rb.IntegrityError(
                f"injected node {entity['name']!r} mentions required fact(s) {offending} — "
                "it would score them for free"
            )
    names = [e["name"] for e in corpus.entities]
    if len(names) != len(set(names)):
        duplicated = sorted({n for n in names if names.count(n) > 1})
        raise rb.IntegrityError(f"injection produced colliding node names: {duplicated}")
    known = set(names)
    for rel in corpus.relationships:
        if rel["source"] not in known or rel["target"] not in known:
            raise rb.IntegrityError(f"dangling injected edge: {rel}")


def oracle_graph(dataset: dict, corpus: Corpus) -> tuple[list[dict], list[dict], list[list[str]]]:
    """The noisy graph with *exactly* the true merges applied — perfect resolution.

    Every variant edge was carved out of its canonical entity's edge list, so
    collapsing all true pairs restores the shipped graph exactly; the decoys,
    which are genuinely distinct entities, survive. Built directly rather than
    by running the resolver, and then cross-checked against the production
    collapse path in :func:`_self_check`.
    """
    entities = copy.deepcopy(dataset["entities"])
    for name, entity_type, description, _rel, _target in DECOYS:
        entities.append({"name": name, "type": entity_type, "description": description})
    relationships = copy.deepcopy(dataset["relationships"])
    for name, _type, _description, rel_type, target in DECOYS:
        relationships.append(
            {"source": name, "target": target, "type": rel_type,
             "description": "Injected decoy edge (RQ3)"}
        )
    chunk_entities = [list(p.get("entities") or []) for p in dataset["passages"]]
    return entities, relationships, chunk_entities


# ── Resolution, scored against ground truth ──────────────────────────────────
def cluster_pairs(clusters: list[list[str]]) -> set[frozenset]:
    """Every unordered pair a cluster set asserts to be the same entity."""
    pairs: set[frozenset] = set()
    for cluster in clusters:
        for i, left in enumerate(cluster):
            for right in cluster[i + 1:]:
                pairs.add(frozenset((left, right)))
    return pairs


def merge_quality(clusters: list[list[str]], truth: set[frozenset]) -> dict:
    """Pair-level precision / recall / F1 of one resolver setting.

    Pair-level rather than cluster-level because transitive union-find turns
    one bad link into a whole bad cluster, and pair counting is what makes that
    damage visible in proportion to its size.
    """
    predicted = cluster_pairs(clusters)
    correct = predicted & truth
    wrong = predicted - truth
    precision = len(correct) / len(predicted) if predicted else 1.0
    recall = len(correct) / len(truth) if truth else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "predicted_pairs": len(predicted),
        "correct_pairs": len(correct),
        "wrong_pairs": len(wrong),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clusters": len(clusters),
        "largest_cluster": max((len(c) for c in clusters), default=0),
        "nodes_absorbed": sum(len(c) - 1 for c in clusters),
        "examples_correct": sorted(sorted(p) for p in list(correct))[:6],
        "examples_wrong": sorted(sorted(p) for p in list(wrong))[:6],
    }


def collapse_with_clusters(
    entities: list[dict], relationships: list[dict], clusters: list[list[str]]
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Apply an explicit cluster set with the production collapse semantics.

    Used for the oracle row and for the self-check that this reproduces
    ``graph_builder._collapse_duplicate_entities`` on a real cell.
    """
    from app.services import entity_resolution as er

    canonical_of: dict[str, str] = {}
    for cluster in clusters:
        canonical = er.choose_canonical(cluster)
        for name in cluster:
            if name != canonical:
                canonical_of[name] = canonical
    kept = [e for e in entities if e["name"] not in canonical_of]
    rewritten: list[dict] = []
    for rel in relationships:
        source = canonical_of.get(rel["source"], rel["source"])
        target = canonical_of.get(rel["target"], rel["target"])
        if source == target:
            continue
        rewritten.append({**rel, "source": source, "target": target})
    return kept, rewritten, canonical_of


# ── Graph structure metrics ──────────────────────────────────────────────────
def graph_metrics(entities: list[dict], relationships: list[dict]) -> dict:
    """Node/edge counts, components and modularity of a collapsed graph.

    Modularity is computed on the undirected, unweighted simple graph with
    greedy modularity maximisation. Nodes and edges are inserted in sorted
    order so the partition — and therefore the number — is deterministic.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(sorted(e["name"] for e in entities))
    graph.add_edges_from(
        sorted({(rel["source"], rel["target"]) for rel in relationships if rel["source"] != rel["target"]})
    )
    unique_edges = graph.number_of_edges()
    components = list(nx.connected_components(graph))
    if unique_edges:
        partition = nx.community.greedy_modularity_communities(graph)
        modularity = nx.community.modularity(graph, partition)
        communities = len(partition)
    else:
        modularity = 0.0
        communities = graph.number_of_nodes()
    degrees = [d for _n, d in graph.degree()]
    return {
        "nodes": graph.number_of_nodes(),
        "unique_edges": unique_edges,
        "typed_edges": len({(r["source"], r["target"], r["type"]) for r in relationships}),
        "components": len(components),
        "largest_component": max((len(c) for c in components), default=0),
        "isolated_nodes": sum(1 for d in degrees if d == 0),
        "mean_degree": statistics.mean(degrees) if degrees else 0.0,
        "max_degree": max(degrees, default=0),
        "modularity": modularity,
        "modularity_communities": communities,
    }


# ── Neo4j: write the collapsed graph, then run production retrieval ──────────
async def seed(
    entities: list[dict],
    relationships: list[dict],
    chunk_entities: list[list[str]],
    embeddings: dict[str, list[float]],
    dataset: dict,
    *,
    with_chunks: bool,
) -> dict:
    """Write one collapsed graph through the production ingest path."""
    from app.neo4j_driver import execute_query
    from app.services import chunk_store, graph_builder
    from app.services.graph_schema import ensure_schema

    await ensure_schema()
    for query in rb.CLEAR_QUERIES:
        await execute_query(query)
    nodes = await graph_builder._write_entities(
        entities, dataset.get("document_name", "rq3"), embeddings
    )
    edges = await graph_builder._write_relationships(relationships)
    stored = {"chunks": 0, "links": 0}
    if with_chunks:
        stored = await chunk_store.store_chunks(
            dataset.get("document_name", "rq3"),
            [p["text"] for p in dataset["passages"]],
            chunk_entities,
        )
    counted = await execute_query(
        "MATCH (n:Entity) WITH count(n) AS nodes "
        "MATCH ()-[r]->() RETURN nodes, count(r) AS edges"
    )
    return {
        "written_nodes": nodes,
        "written_edges": edges,
        "db_nodes": counted[0]["nodes"] if counted else 0,
        "db_edges": counted[0]["edges"] if counted else 0,
        **stored,
    }


async def _db_state() -> tuple[int, int]:
    """``(:Entity count, :Chunk count)`` as the database holds them right now."""
    from app.neo4j_driver import execute_query

    entities = await execute_query("MATCH (n:Entity) RETURN count(n) AS c")
    chunks = await execute_query("MATCH (n:Chunk) RETURN count(n) AS c")
    return (
        int(entities[0]["c"]) if entities else 0,
        int(chunks[0]["c"]) if chunks else 0,
    )


async def _measure(
    entities: list[dict],
    relationships: list[dict],
    chunk_entities: list[list[str]],
    embeddings: dict[str, list[float]],
    dataset: dict,
    *,
    with_chunks: bool,
) -> tuple[dict, list[rb.QuestionResult]]:
    """Seed one graph, run all 14 questions, and *verify nothing else touched it*.

    The database is shared with concurrently running experiments (Community
    edition: one database, no namespaces), and an outside write landing between
    two of our questions would produce numbers that look fine and are not. So the
    post-conditions this cell was seeded with — the exact entity count, the exact
    chunk count, and the harness's own ``contaminated`` check for the graph-only
    row — are re-asserted *after* the evaluation, and the whole cell is discarded
    and re-run from a fresh seed when they fail.
    """
    expected_nodes = len({e["name"] for e in entities})
    expected_chunks = len(dataset["passages"]) if with_chunks else 0
    last: list[str] = []
    for attempt in range(1, INTERFERENCE_RETRIES + 1):
        written = await seed(
            entities, relationships, chunk_entities, embeddings, dataset,
            with_chunks=with_chunks,
        )
        results = await rb.evaluate_graphrag(dataset, GRAPH_K)
        nodes, chunks = await _db_state()
        last = []
        if nodes != expected_nodes:
            last.append(f":Entity count is {nodes}, expected {expected_nodes}")
        if chunks != expected_chunks:
            last.append(f":Chunk count is {chunks}, expected {expected_chunks}")
        if not with_chunks and (tainted := rb.contaminated(results)):
            last.append(f"graph-only context carried source excerpts on {tainted}")
        if not last:
            if attempt > 1:
                INTERFERENCE["recovered_cells"] += 1
            return written, results
        INTERFERENCE["detected"] += 1
        print(
            f"      ⚠️ external write detected (attempt {attempt}/{INTERFERENCE_RETRIES}): "
            f"{'; '.join(last)} — re-seeding and re-running this cell",
            flush=True,
        )
        await asyncio.sleep(INTERFERENCE_BACKOFF * (1.0 + random.random()))
    raise rb.IntegrityError(
        f"could not obtain an uncontaminated measurement in {INTERFERENCE_RETRIES} "
        f"attempts — another process is writing to this database ({'; '.join(last)})"
    )


async def evaluate_cell(
    entities: list[dict],
    relationships: list[dict],
    chunk_entities: list[list[str]],
    embeddings: dict[str, list[float]],
    dataset: dict,
) -> dict:
    """Graph-only (C) and graph+chunks (D) retrieval for one collapsed graph."""
    written, graph_only = await _measure(
        entities, relationships, chunk_entities, embeddings, dataset, with_chunks=False
    )
    _written_d, with_chunks = await _measure(
        entities, relationships, chunk_entities, embeddings, dataset, with_chunks=True
    )
    return {
        "written": written,
        "C": rb.aggregate(graph_only),
        "D": rb.aggregate(with_chunks),
        "C_per_question": [r.recall for r in graph_only],
        "C_per_question_complete": [float(r.complete) for r in graph_only],
        "D_per_question": [r.recall for r in with_chunks],
        "D_per_question_complete": [float(r.complete) for r in with_chunks],
        "D_per_question_strict": [r.strict_recall for r in with_chunks],
        "qids": [r.qid for r in graph_only],
    }


# ── Bootstrap ────────────────────────────────────────────────────────────────
def bootstrap_ci(values: list[float], *, iters: int = BOOTSTRAP_ITERS,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling the questions with replacement."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def paired_bootstrap_ci(a: list[float], b: list[float], *, iters: int = BOOTSTRAP_ITERS,
                        seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """95% CI of the mean paired difference ``a - b``, resampling questions.

    Paired because both cells answered the *same* 14 questions: the unpaired
    interval would be wider than the evidence warrants and would hide real
    within-question movement.
    """
    if not a or len(a) != len(b):
        return (0.0, 0.0, 0.0)
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (statistics.mean(diffs), means[int(0.025 * iters)], means[int(0.975 * iters)])


# ── The sweep ────────────────────────────────────────────────────────────────
async def embed_entities(entities: list[dict]) -> dict[str, list[float]]:
    """Embed every node exactly as ``graph_builder._embed_entities`` does.

    Computed once per corpus and reused for every cell: the embeddings are a
    property of the corpus, not of the thresholds, so re-computing them per
    cell would only add wall-clock and a chance of drift.
    """
    from app.services.llm_provider import get_embeddings

    texts = [f"{e['name']} — {e.get('description', '')}".strip(" —") for e in entities]
    vectors = await asyncio.to_thread(get_embeddings().embed_documents, texts)
    return {e["name"]: v for e, v in zip(entities, vectors, strict=True)}


def signature(clusters: list[list[str]]) -> str:
    """Stable identity of a resolver outcome — the cache key for evaluation."""
    return json.dumps(sorted(sorted(c) for c in clusters), sort_keys=True)


async def sweep(corpus: Corpus, dataset: dict, *, plan_only: bool = False) -> dict:
    """Run the whole grid for one corpus."""
    from app.config import Settings
    from app.services import entity_resolution as er
    from app.services import graph_builder

    embeddings = await embed_entities(corpus.entities)

    cells: list[dict] = []
    by_signature: dict[str, dict] = {}
    for vector_threshold in VECTOR_GRID:
        for name_threshold in NAME_GRID:
            settings = Settings(
                entity_resolution_threshold=vector_threshold,
                entity_resolution_name_threshold=name_threshold,
            )
            clusters = er.find_duplicate_clusters(corpus.entities, embeddings, settings=settings)
            cells.append(
                {
                    "vector_threshold": vector_threshold,
                    "name_threshold": name_threshold,
                    "signature": signature(clusters),
                    "clusters": clusters,
                    "quality": merge_quality(clusters, corpus.true_pairs),
                }
            )
    distinct = {c["signature"] for c in cells}
    print(f"   {corpus.key}: {len(cells)} cells, {len(distinct)} distinct resolver outcomes")
    if plan_only:
        return {"corpus": corpus.key, "cells": len(cells), "distinct": len(distinct)}

    for index, cell in enumerate(cells, 1):
        key = cell["signature"]
        if key in by_signature:
            cell.update(by_signature[key])
            continue
        entities, relationships, _merged, _aliases = graph_builder._collapse_duplicate_entities(
            copy.deepcopy(corpus.entities),
            copy.deepcopy(corpus.relationships),
            embeddings,
            Settings(
                entity_resolution_threshold=cell["vector_threshold"],
                entity_resolution_name_threshold=cell["name_threshold"],
            ),
        )
        alias_map = {
            name: er.choose_canonical(cluster)
            for cluster in cell["clusters"]
            for name in cluster
            if name != er.choose_canonical(cluster)
        }
        chunk_entities = graph_builder._canonicalize_chunk_names(
            corpus.chunk_entities, {}, alias_map, {e["name"] for e in entities}
        )
        structure = graph_metrics(entities, relationships)
        retrieval = await evaluate_cell(
            entities, relationships, chunk_entities, embeddings, dataset
        )
        payload = {"structure": structure, "retrieval": retrieval}
        by_signature[key] = payload
        cell.update(payload)
        print(
            f"   [{index:>3}/{len(cells)}] v={cell['vector_threshold']:.2f} "
            f"n={cell['name_threshold']:.2f} → merges={cell['quality']['nodes_absorbed']:>3} "
            f"nodes={structure['nodes']:>3} Q={structure['modularity']:.3f} "
            f"C_recall={retrieval['C']['fact_recall']:.3f} "
            f"D_recall={retrieval['D']['fact_recall']:.3f}"
        )

    # Reference rows that are not cells of the grid.
    references: dict[str, dict] = {}
    if corpus.true_pairs:
        entities, relationships, chunk_entities = oracle_graph(dataset, corpus)
        references["oracle"] = {
            "label": "ORACLE — exactly the true merges applied (perfect resolution)",
            "structure": graph_metrics(entities, relationships),
            "retrieval": await evaluate_cell(
                entities, relationships, chunk_entities,
                await embed_entities(entities), dataset
            ),
        }
    return {
        "corpus": corpus.key,
        "label": corpus.label,
        "cells": cells,
        "distinct": len(distinct),
        "references": references,
        "injected": corpus.injected,
        "decoys": corpus.decoys,
        "pair_landscape": pair_landscape(corpus, embeddings),
    }


def pair_landscape(corpus: Corpus, embeddings: dict[str, list[float]]) -> dict:
    """(cosine, name-similarity) coordinates of true and adversarial pairs.

    This is the decision surface the two thresholds cut through, measured
    rather than assumed — and it is what decides in advance whether *any* cell
    of the grid can be simultaneously precise and complete.
    """
    from app.services import entity_resolution as er

    types = {e["name"]: e.get("type") for e in corpus.entities}
    true_rows = []
    for pair in corpus.true_pairs:
        left, right = sorted(pair, key=len, reverse=True)
        true_rows.append(
            {
                "canonical": left,
                "variant": right,
                "cosine": er.cosine_similarity(embeddings[left], embeddings[right]),
                "name_sim": er.name_similarity(left, right),
            }
        )
    true_rows.sort(key=lambda r: r["cosine"])

    names = sorted(embeddings)
    false_rows = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if frozenset((left, right)) in corpus.true_pairs:
                continue
            if types.get(left) != types.get(right):
                continue  # the type barrier already blocks these, at every threshold
            name_sim = er.name_similarity(left, right)
            if name_sim < 0.5:
                continue
            false_rows.append(
                {
                    "a": left,
                    "b": right,
                    "cosine": er.cosine_similarity(embeddings[left], embeddings[right]),
                    "name_sim": name_sim,
                }
            )
    false_rows.sort(key=lambda r: (-r["name_sim"], -r["cosine"]))
    return {
        "true": true_rows,
        "false_same_type": false_rows,
        "max_false_cosine": max((r["cosine"] for r in false_rows), default=0.0),
        "max_false_name_sim": max((r["name_sim"] for r in false_rows), default=0.0),
    }


async def _self_check(dataset: dict, corpus: Corpus) -> None:
    """Verify the two collapse implementations agree before trusting either.

    The oracle row is built directly (by construction the noisy graph with all
    true merges applied should *be* the shipped graph plus decoys). If that is
    wrong, every "perfect resolution" comparison in the report is wrong, so it
    is checked against the generic collapse rather than asserted in prose.
    """
    clusters = [sorted(p, key=len, reverse=True) for p in corpus.true_pairs]
    kept, rewritten, _ = collapse_with_clusters(
        copy.deepcopy(corpus.entities), copy.deepcopy(corpus.relationships), clusters
    )
    entities, relationships, _chunks = oracle_graph(dataset, corpus)
    if {e["name"] for e in kept} != {e["name"] for e in entities}:
        raise rb.IntegrityError("oracle node set disagrees with the collapsed true clusters")
    left = {(r["source"], r["target"], r["type"]) for r in rewritten}
    right = {(r["source"], r["target"], r["type"]) for r in relationships}
    if left != right:
        raise rb.IntegrityError(
            f"oracle edge set disagrees with the collapsed true clusters: {left ^ right}"
        )


# ── Reporting ────────────────────────────────────────────────────────────────
def cell_at(cells: list[dict], vector_threshold: float, name_threshold: float) -> dict:
    for cell in cells:
        if (
            math.isclose(cell["vector_threshold"], vector_threshold)
            and math.isclose(cell["name_threshold"], name_threshold)
        ):
            return cell
    raise KeyError((vector_threshold, name_threshold))


def _fmt(value: float) -> str:
    return f"{value:.1%}"


def _grid_table(cells: list[dict], value: str, fmt) -> list[str]:
    """One (vector × name) heat table, rendered as markdown."""
    header = "| v \\ n | " + " | ".join(
        ("off" if math.isclose(n, OFF) else f"{n:.2f}")
        + ("&nbsp;★" if math.isclose(n, DEFAULT_NAME) else "")
        for n in NAME_GRID
    ) + " |"
    rule = "| ---: |" + " ---: |" * len(NAME_GRID)
    lines = [header, rule]
    for vector_threshold in VECTOR_GRID:
        label = "off" if math.isclose(vector_threshold, OFF) else f"{vector_threshold:.2f}"
        if math.isclose(vector_threshold, DEFAULT_VECTOR):
            label += "&nbsp;★"
        row = [label]
        for name_threshold in NAME_GRID:
            cell = cell_at(cells, vector_threshold, name_threshold)
            node = cell
            for part in value.split("."):
                node = node[part]
            row.append(fmt(node))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _curve_table(cells: list[dict], name_threshold: float) -> list[str]:
    """The 1-D slice through the grid at a fixed lexical threshold."""
    lines = [
        "| cosine threshold | merges | pair prec. | pair recall | nodes | edges | comps | "
        "modularity | C recall | C coverage | D recall | D coverage | D recall (strict) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for vector_threshold in VECTOR_GRID:
        cell = cell_at(cells, vector_threshold, name_threshold)
        quality, structure = cell["quality"], cell["structure"]
        c_agg, d_agg = cell["retrieval"]["C"], cell["retrieval"]["D"]
        label = "off" if math.isclose(vector_threshold, OFF) else f"{vector_threshold:.2f}"
        if math.isclose(vector_threshold, DEFAULT_VECTOR):
            label += " ★"
        lines.append(
            f"| {label} | {quality['nodes_absorbed']} | {_fmt(quality['precision'])} | "
            f"{_fmt(quality['recall'])} | {structure['nodes']} | {structure['unique_edges']} | "
            f"{structure['components']} | {structure['modularity']:.3f} | "
            f"{_fmt(c_agg['fact_recall'])} | {_fmt(c_agg['full_coverage'])} | "
            f"{_fmt(d_agg['fact_recall'])} | {_fmt(d_agg['full_coverage'])} | "
            f"{_fmt(d_agg['strict_fact_recall'])} |"
        )
    return lines


def _extremes(cells: list[dict], metric: str) -> tuple[dict, dict]:
    """Best and worst cell of the grid on one retrieval metric."""
    def score(cell: dict) -> float:
        return cell["retrieval"]["C"][metric]

    return max(cells, key=score), min(cells, key=score)


def _cell_name(cell: dict) -> str:
    v = "off" if math.isclose(cell["vector_threshold"], OFF) else f"{cell['vector_threshold']:.2f}"
    n = "off" if math.isclose(cell["name_threshold"], OFF) else f"{cell['name_threshold']:.2f}"
    return f"({v} / {n})"


def _scatter(landscape: dict, width: int = 62, height: int = 24) -> list[str]:
    """ASCII scatter of the decision surface: name similarity × cosine.

    ``T`` = a pair that *should* merge, ``x`` = a same-type pair that must not,
    ``#`` = both fall in the same character cell. The shipped thresholds are
    drawn as the ``│`` and ``─`` rules, so the four quadrants are the four
    regimes: everything above and to the right of the crossing is merged.
    """
    x_lo, x_hi, y_lo, y_hi = 0.30, 1.0, 0.30, 1.0

    def col(x: float) -> int:
        return min(width - 1, max(0, round((x - x_lo) / (x_hi - x_lo) * (width - 1))))

    def row(y: float) -> int:
        return min(height - 1, max(0, round((y_hi - y) / (y_hi - y_lo) * (height - 1))))

    grid = [[" "] * width for _ in range(height)]
    default_col, default_row = col(DEFAULT_NAME), row(DEFAULT_VECTOR)
    for r in range(height):
        grid[r][default_col] = "│"
    for c in range(width):
        grid[default_row][c] = "┼" if c == default_col else "─"
    for point, mark in ((landscape["false_same_type"], "x"), (landscape["true"], "T")):
        for item in point:
            r, c = row(item["cosine"]), col(item["name_sim"])
            grid[r][c] = "#" if grid[r][c] in {"x", "T", "#"} else mark
    lines = ["  cosine", "   1.00 ┤" + "".join(grid[0])]
    for index in range(1, height):
        label = f"   {y_hi - (y_hi - y_lo) * index / (height - 1):.2f} ┤"
        lines.append(label + "".join(grid[index]))
    lines.append("        └" + "─" * width)
    lines.append("         0.30" + " " * (width - 18) + "1.00  name similarity")
    lines.append("")
    lines.append("   T = must merge (injected duplicate)   x = must NOT merge (same-type pair)")
    lines.append("   # = both   ─│ = the shipped defaults (cosine 0.93, name 0.87)")
    return lines


def _separability(landscape: dict) -> dict:
    """Can ANY cell of the plane be simultaneously perfect? Computed, not assumed.

    A threshold pair ``(v, n)`` merges exactly the pairs with cosine >= v and
    name similarity >= n. Perfect resolution therefore exists iff some ``(v, n)``
    dominates every true pair and no false one — which, on this corpus, it does
    not, and the numbers below say by how much.
    """
    true_rows, false_rows = landscape["true"], landscape["false_same_type"]
    best = {"recall": 0.0, "vector": OFF, "name": OFF, "caught": 0}
    for vector_threshold in VECTOR_GRID:
        for name_threshold in NAME_GRID:
            wrong = [
                r for r in false_rows
                if r["cosine"] >= vector_threshold and r["name_sim"] >= name_threshold
            ]
            if wrong:
                continue
            caught = [
                r for r in true_rows
                if r["cosine"] >= vector_threshold and r["name_sim"] >= name_threshold
            ]
            recall = len(caught) / len(true_rows) if true_rows else 1.0
            if recall > best["recall"]:
                best = {
                    "recall": recall,
                    "vector": vector_threshold,
                    "name": name_threshold,
                    "caught": len(caught),
                }
    return {
        "n_true": len(true_rows),
        "n_false_same_type": len(false_rows),
        "best_precise_recall": best["recall"],
        "best_precise_cell": (best["vector"], best["name"]),
        "best_precise_caught": best["caught"],
    }


def _regime_rows(cells: list[dict]) -> list[dict]:
    """Representative cells along the aggressiveness axis, chosen by merge count."""
    picks: list[dict] = []
    seen: set[int] = set()
    for target in (0, 5, 15, 25, 35, 50, 80, 150, 10_000):
        best = min(
            cells,
            key=lambda c: (abs(c["quality"]["nodes_absorbed"] - target), c["vector_threshold"]),
        )
        if best["quality"]["nodes_absorbed"] not in seen:
            seen.add(best["quality"]["nodes_absorbed"])
            picks.append(best)
    return picks


def _curve_points(cells: list[dict]) -> list[dict]:
    """One point per *distinct resolver outcome*, ordered by aggressiveness.

    The grid is 2-D but the thing the hypothesis is about is 1-D: how much
    merging happened. Cells that produced the identical merge set are the
    identical graph, so they are one point, not many — counting them once is
    what stops a densely-sampled corner of the grid from dominating the curve.
    """
    by_signature: dict[str, dict] = {}
    for cell in cells:
        by_signature.setdefault(cell["signature"], cell)
    points = [
        {
            "merges": cell["quality"]["nodes_absorbed"],
            "precision": cell["quality"]["precision"],
            "pair_recall": cell["quality"]["recall"],
            "f1": cell["quality"]["f1"],
            "nodes": cell["structure"]["nodes"],
            "edges": cell["structure"]["unique_edges"],
            "components": cell["structure"]["components"],
            "modularity": cell["structure"]["modularity"],
            "c_recall": cell["retrieval"]["C"]["fact_recall"],
            "c_coverage": cell["retrieval"]["C"]["full_coverage"],
            "d_recall": cell["retrieval"]["D"]["fact_recall"],
            "d_coverage": cell["retrieval"]["D"]["full_coverage"],
            "d_strict": cell["retrieval"]["D"]["strict_fact_recall"],
            "cell": cell,
        }
        for cell in by_signature.values()
    ]
    points.sort(key=lambda p: (p["merges"], -p["nodes"]))
    return points


def _curve_ascii(points: list[dict], key: str, label: str, height: int = 14) -> list[str]:
    """ASCII plot of one retrieval metric against merge count.

    The x axis is ``sqrt(merges)`` because the interesting structure is at the
    low end (0–20 merges) while the grid runs to well over a hundred; a linear
    axis would compress every regime worth seeing into two columns.
    """
    if not points:
        return []
    width = 62
    xs = [p["merges"] for p in points]
    ys = [p[key] for p in points]
    x_hi = max(xs) ** 0.5 or 1.0
    y_lo, y_hi = min(ys), max(ys)
    if y_hi - y_lo < 1e-9:
        y_lo, y_hi = y_lo - 0.05, y_hi + 0.05

    def col(x: float) -> int:
        return min(width - 1, max(0, round((x**0.5) / x_hi * (width - 1))))

    def row(y: float) -> int:
        return min(height - 1, max(0, round((y_hi - y) / (y_hi - y_lo) * (height - 1))))

    grid = [[" "] * width for _ in range(height)]
    for point in points:
        r, c = row(point[key]), col(point["merges"])
        grid[r][c] = "#" if grid[r][c] == "o" else "o"
    lines = [f"  {label}"]
    for index in range(height):
        value = y_hi - (y_hi - y_lo) * index / (height - 1)
        lines.append(f"  {value:6.1%} ┤" + "".join(grid[index]))
    lines.append("         └" + "─" * width)
    ticks = [0, max(xs) // 16, max(xs) // 4, max(xs)]
    axis = [" "] * (width + 1)
    for tick in ticks:
        text = str(tick)
        start = min(len(axis) - len(text), col(tick))
        axis[start:start + len(text)] = list(text)
    lines.append("          " + "".join(axis))
    lines.append("          duplicate nodes absorbed (√ scale)")
    return lines


def _shape_verdict(points: list[dict], n_questions: int) -> dict:
    """Does the measured curve have the shape the hypothesis predicted?

    Decided by arithmetic, not by eye. The effect-size floor is one question out
    of ``n`` — the same floor ``run_benchmark.verdict`` applies — because a gap
    smaller than a single item on a 14-question set is a re-wording, not a
    result.
    """
    floor = 1.0 / n_questions if n_questions >= 2 else 0.0
    peak = max(points, key=lambda p: (p["c_recall"], -p["merges"]))
    least = min(points, key=lambda p: p["merges"])
    most = max(points, key=lambda p: p["merges"])
    spread = max(p["c_recall"] for p in points) - min(p["c_recall"] for p in points)
    rise = peak["c_recall"] - least["c_recall"]
    fall = peak["c_recall"] - most["c_recall"]
    interior = 0 < peak["merges"] < most["merges"]

    if spread < floor:
        shape = "flat"
    elif rise >= floor and fall >= floor and interior:
        shape = "inverted-u"
    elif fall >= floor and rise < floor:
        shape = "monotone-decreasing"
    elif rise >= floor and fall < floor:
        shape = "monotone-increasing"
    else:
        shape = "noisy"
    return {
        "shape": shape,
        "floor": floor,
        "spread": spread,
        "rise": rise,
        "fall": fall,
        "peak_merges": peak["merges"],
        "peak_recall": peak["c_recall"],
        "peak_cell": peak["cell"],
        "least": least,
        "most": most,
        "interior_peak": interior,
    }


def _headline_block(results: dict) -> list[str]:
    """Everything the report is allowed to claim, with its uncertainty attached."""
    noisy = results["sweeps"]["noisy"]
    cells = noisy["cells"]
    default = cell_at(cells, DEFAULT_VECTOR, DEFAULT_NAME)
    off = cell_at(cells, OFF, OFF)
    oracle = noisy["references"]["oracle"]
    best, worst = _extremes(cells, "fact_recall")

    lines: list[str] = []

    def row(label: str, cell: dict, key: str = "C_per_question") -> str:
        values = cell["retrieval"][key]
        mean = statistics.mean(values)
        low, high = bootstrap_ci(values)
        return f"| {label} | {_fmt(mean)} | [{_fmt(low)}, {_fmt(high)}] |"

    lines += [
        "| Configuration | Graph-only fact recall (C, permissive) | bootstrap 95% CI |",
        "| --- | ---: | ---: |",
        row(f"resolution OFF {_cell_name(off)}", off),
        row(f"shipped defaults {_cell_name(default)} ★", default),
        row(f"best cell in the grid {_cell_name(best)}", best),
        row(f"worst cell in the grid {_cell_name(worst)}", worst),
        row("ORACLE — perfect resolution", oracle),
        "",
    ]
    comparisons = [
        ("shipped defaults − resolution OFF", default, off),
        ("best cell − shipped defaults", best, default),
        ("oracle − shipped defaults", oracle, default),
        ("best cell − worst cell (total grid sensitivity)", best, worst),
    ]
    lines += [
        "| Paired comparison (per-question, n=14) | mean Δ | bootstrap 95% CI | crosses 0? |",
        "| --- | ---: | ---: | :---: |",
    ]
    for label, left, right in comparisons:
        mean, low, high = paired_bootstrap_ci(
            left["retrieval"]["C_per_question"], right["retrieval"]["C_per_question"]
        )
        crosses = "yes" if low <= 0.0 <= high else "**no**"
        lines.append(
            f"| {label} | {mean * 100:+.1f} pp | [{low * 100:+.1f}, {high * 100:+.1f}] pp "
            f"| {crosses} |"
        )
    return lines


def _collapse_point(cells: list[dict]) -> dict:
    """The first setting at which merging is doing more harm than good.

    Defined operationally and computed: the most conservative cell (highest
    cosine threshold, then highest name threshold) whose pair precision has
    fallen below 50% — i.e. where the majority of what the resolver asserts is
    wrong — together with the most aggressive cell in the grid.
    """
    ordered = sorted(
        cells, key=lambda c: (-c["vector_threshold"], -c["name_threshold"])
    )
    first_wrong = next((c for c in ordered if c["quality"]["wrong_pairs"] > 0), None)
    half_wrong = next((c for c in ordered if c["quality"]["precision"] < 0.5), None)
    total = max(cells, key=lambda c: c["quality"]["nodes_absorbed"])
    return {"first_wrong": first_wrong, "half_wrong": half_wrong, "most_aggressive": total}


def _delta(left: dict, right: dict, key: str = "C_per_question") -> str:
    """One paired-bootstrap comparison, rendered with its CI and its verdict."""
    mean, low, high = paired_bootstrap_ci(left["retrieval"][key], right["retrieval"][key])
    crosses = low <= 0.0 <= high
    return (
        f"{mean * 100:+.1f} pp [95% CI {low * 100:+.1f}, {high * 100:+.1f} pp; "
        f"{'includes' if crosses else 'excludes'} 0]"
    )


def _interpretation_block(
    results: dict, points: list[dict], shape: dict, separability: dict, collapse: dict
) -> list[str]:
    """What was measured, then — separately and labelled — what is inferred."""
    noisy = results["sweeps"]["noisy"]
    clean = results["sweeps"]["clean"]
    cells, clean_cells = noisy["cells"], clean["cells"]
    default = cell_at(cells, DEFAULT_VECTOR, DEFAULT_NAME)
    off = cell_at(cells, OFF, OFF)
    oracle = noisy["references"]["oracle"]
    best, worst = _extremes(cells, "fact_recall")
    floor = shape["floor"]
    clean_inert = [c for c in clean_cells if c["quality"]["nodes_absorbed"] == 0]
    clean_damage = [c for c in clean_cells if c["quality"]["nodes_absorbed"] > 0]
    d_recalls = [p["d_recall"] for p in points]
    c_recalls = [p["c_recall"] for p in points]
    modularities = [p["modularity"] for p in points]
    nodes = [p["nodes"] for p in points]

    return [
        "**Measured.**",
        "",
        f"1. **On a corpus with no duplicates the knob is inert until it is harmful.** "
        f"{len(clean_inert)} of the {len(clean_cells)} CLEAN cells — including the shipped "
        f"defaults — merge nothing at all, so retrieval is bit-identical across them. The "
        f"{len(clean_damage)} cells that do merge are all merging entities that are not "
        f"duplicates, because the CLEAN entity list is canonical by construction. There is "
        "no upside available on this corpus, only downside.",
        "",
        f"2. **Unresolved duplication is expensive, and resolution does recover it.** On "
        f"NOISY, going from resolution OFF to *perfect* resolution (the oracle) is worth "
        f"{_delta(oracle, off)} of graph-only fact recall. The shipped defaults recover "
        f"{_delta(default, off)} of that, absorbing "
        f"{default['quality']['nodes_absorbed']} of the "
        f"{len(noisy['injected'])} injected duplicates at pair precision "
        f"{_fmt(default['quality']['precision'])}.",
        "",
        f"3. **The head-room left on the table by the shipped defaults is real but "
        f"bounded.** Oracle − defaults = {_delta(oracle, default)}; best grid cell − "
        f"defaults = {_delta(best, default)} (best cell {_cell_name(best)}).",
        "",
        f"4. **Total sensitivity of the whole grid.** Best cell − worst cell = "
        f"{_delta(best, worst)} on graph-only fact recall. Across the "
        f"{len(points)} distinct resolver outcomes, graph-only fact recall spans "
        f"{_fmt(min(c_recalls))}–{_fmt(max(c_recalls))} "
        f"({(max(c_recalls) - min(c_recalls)) * 100:.1f} pp), which is "
        f"{'larger' if (max(c_recalls) - min(c_recalls)) >= floor else 'smaller'} than the "
        f"one-question floor of {floor * 100:.1f} pp.",
        "",
        f"5. **The shipped system is far less sensitive than the graph-only system.** "
        f"D (graph + source chunks) spans {_fmt(min(d_recalls))}–{_fmt(max(d_recalls))} "
        f"({(max(d_recalls) - min(d_recalls)) * 100:.1f} pp) over the same "
        f"{len(points)} outcomes, against C's "
        f"{(max(c_recalls) - min(c_recalls)) * 100:.1f} pp. The source-chunk channel does "
        "not care which entity node the evidence hangs off, so it absorbs most of the "
        "damage the entity layer takes.",
        "",
        f"6. **Structure moves much further than retrieval does.** Over the same outcomes "
        f"the graph goes from {max(nodes)} nodes down to {min(nodes)} and modularity from "
        f"{max(modularities):.3f} down to {min(modularities):.3f} — a "
        f"{(max(modularities) - min(modularities)) * 100:.0f}-point swing in a quantity "
        f"bounded by 1 — while graph-only fact recall moves "
        f"{(max(c_recalls) - min(c_recalls)) * 100:.1f} pp and the shipped system moves "
        f"{(max(d_recalls) - min(d_recalls)) * 100:.1f} pp. Graph statistics are a far more "
        "sensitive indicator of over-merging than the retrieval metric is.",
        "",
        f"7. **Perfect resolution is geometrically unreachable here, at any threshold.** No "
        f"cell separates all {separability['n_true']} true duplicate pairs from the "
        f"{separability['n_false_same_type']} same-type pairs that must not merge; the best "
        f"fully-precise setting recovers {_fmt(separability['best_precise_recall'])} of the "
        "duplicates. The binding constraint is that different people who share a surname "
        "score *exactly* the same lexically as a person and their own surname.",
        "",
        f"8. **The collapse is gradual on retrieval and abrupt on structure.** The first "
        f"wrong merge appears at {_cell_name(collapse['first_wrong'])} and the majority of "
        f"asserted merges are wrong from "
        f"{_cell_name(collapse['half_wrong']) if collapse['half_wrong'] else 'nowhere in this grid'}"
        f"; at the most aggressive cell {_cell_name(collapse['most_aggressive'])} the graph "
        f"is {collapse['most_aggressive']['structure']['nodes']} nodes "
        f"(from {off['structure']['nodes']}) yet graph-only fact recall is still "
        f"{_fmt(collapse['most_aggressive']['retrieval']['C']['fact_recall'])} and the "
        "shipped system's is "
        f"{_fmt(collapse['most_aggressive']['retrieval']['D']['fact_recall'])}.",
        "",
        "**Inferred — weaker than the measurements above, and labelled as such.**",
        "",
        "- The mechanism we *infer* behind (5) and (6) is that this benchmark scores "
        "retrieval by whether a required entity *name* reaches the context, and a "
        "wrongly-merged node still carries both names (the absorbed name is kept as an "
        "alias and the canonical name is the longer one). Over-merging therefore destroys "
        "*structure* — it grafts foreign edges on and invents paths — long before it "
        "destroys *name reachability*, which is what the metric can see. A benchmark that "
        "scored whether the retrieved edges were **true** would, we expect, fall off much "
        "earlier. This experiment cannot show that; it is a prediction, not a result.",
        "- We infer that the practitioner-facing consequence is a *precision* argument "
        "rather than a *recall* argument: the reason to keep the thresholds strict is not "
        "that loose thresholds visibly cost retrieval on this corpus (they mostly do not), "
        "but that they silently manufacture false relationships that a downstream reader — "
        "human or LLM — has no way to detect. Testing that costs a generation-side "
        "experiment and an LLM budget, which this one deliberately does not have.",
    ]


def _threats_block(results: dict, points: list[dict], shape: dict) -> list[str]:
    """Every reason a referee could give for not believing the numbers above."""
    config = results["config"]
    noisy = results["sweeps"]["noisy"]
    interference = results.get("interference", {})
    return [
        f"- **Small n.** {config['dataset']['questions']} questions. The bootstrap "
        f"resamples questions, so the effective sample size is "
        f"{config['dataset']['questions']}, not the number of facts. Every CI here is "
        f"wide, and one question is worth {shape['floor'] * 100:.1f} pp — treat any gap "
        "below that as a wash, which several of the comparisons above are.",
        "- **The noise model is ours, not nature's.** The duplicate distribution is "
        f"synthetic: {len(noisy['injected'])} injected variants (surname-only and "
        "one-character deletion), half carrying the source description and half not, with "
        "every second edge and chunk link re-pointed. Those choices set where the true "
        "pairs land on the decision surface and therefore how hard the problem is. A real "
        "extractor's duplicates would land somewhere else, and the whole curve would move "
        "with them. Nothing here measures what an LLM extractor actually produces — doing "
        "that would require LLM calls, which this experiment is forbidden.",
        f"- **The decoys set the difficulty of the precision side.** The "
        f"{len(noisy['decoys'])} adversarial entities were written by hand before any "
        "result was seen, but they were written to be hard. A corpus without same-surname "
        "collisions would show a much wider precise operating region, and one with more of "
        "them a narrower one. The 'perfect resolution is unreachable' finding is a "
        "statement about this corpus, not a theorem.",
        "- **Retrieval only, no generation.** The metric is whether a required entity name "
        "reaches the context. It cannot see a hallucinated relationship, a false reasoning "
        "path, or an answer that is wrong for a reason a human would notice. That is "
        "precisely the failure mode over-merging is supposed to cause, so the most "
        "important consequence of aggressive resolution is invisible to this instrument.",
        "- **The permissive rule flatters merged graphs.** A required fact scores when its "
        "name appears anywhere in the context, including in a neighbour line that a wrong "
        "merge put there. The strict rule (prose only) is reported for D throughout, and C "
        "scores 0.0 under it by construction. Read permissive as an upper bound.",
        f"- **One corpus, one embedder, one seed count.** "
        f"`benchmarks/dataset.json` at k={config['graph_k']} with "
        f"`{config['embedding_provider']}` embeddings and "
        f"`retrieval_max_hops={config['retrieval_max_hops']}`. The corpus is small enough "
        "(34 passages) that the chunk channel saturates, which is exactly why D is nearly "
        "flat; on a large corpus D would have more room to be hurt.",
        f"- **The grid is a sample.** {len(config['vector_grid'])} × "
        f"{len(config['name_grid'])} thresholds, denser where the pairs pile up. Behaviour "
        "*between* sampled thresholds is interpolated by the reader, not measured. The "
        f"{len(config['vector_grid']) * len(config['name_grid'])} cells collapse to only "
        f"{len(points)} distinct graphs, so the grid is finer than the outcome space and "
        "the curve above has that many points and no more.",
        "- **Modularity is a heuristic number.** Greedy modularity maximisation on the "
        "undirected simple graph, inserted in sorted order for determinism. It is "
        "comparable across the cells of this run and should not be compared with "
        "modularity computed any other way.",
        "- **In production this resolver runs per document as well as whole-graph.** This "
        "experiment resolves one batch of entities in one pass. An incremental ingest "
        "reaches a different fixed point, and that path is not measured here.",
        "- **A shared database.** The research Neo4j is Community edition — one database, "
        "no namespaces — and other experiments write to it concurrently. Every cell is "
        "therefore verified after it runs (exact entity count, exact chunk count, plus the "
        "harness's own contamination check) and re-seeded and re-run when the state "
        f"drifted. This run detected {interference.get('detected', 0)} such external "
        f"writes and recovered {interference.get('recovered_cells', 0)} cell measurement(s) "
        "by re-running them; a cell that could not be measured cleanly would have aborted "
        "the run rather than been reported.",
    ]


def _claim_block(results: dict, points: list[dict], shape: dict) -> list[str]:
    """The one or two sentences this run licenses — and the ones it does not."""
    noisy = results["sweeps"]["noisy"]
    cells = noisy["cells"]
    default = cell_at(cells, DEFAULT_VECTOR, DEFAULT_NAME)
    off = cell_at(cells, OFF, OFF)
    oracle = noisy["references"]["oracle"]
    best, worst = _extremes(cells, "fact_recall")
    c_span = max(p["c_recall"] for p in points) - min(p["c_recall"] for p in points)
    d_span = max(p["d_recall"] for p in points) - min(p["d_recall"] for p in points)
    _mean_bw, low_bw, high_bw = paired_bootstrap_ci(
        best["retrieval"]["C_per_question"], worst["retrieval"]["C_per_question"]
    )
    _mean_do, low_do, high_do = paired_bootstrap_ci(
        default["retrieval"]["C_per_question"], off["retrieval"]["C_per_question"]
    )

    shape_sentence = {
        "inverted-u": (
            f"the predicted inverted U is present: graph-only fact recall peaks at "
            f"{shape['peak_merges']} absorbed nodes ({_fmt(shape['peak_recall'])}), rising "
            f"{shape['rise'] * 100:+.1f} pp from the unresolved graph and falling "
            f"{-shape['fall'] * 100:+.1f} pp to the most aggressive setting"
        ),
        "monotone-increasing": (
            f"there is **no interior optimum**: graph-only fact recall rises with merging "
            f"and does not fall back by more than one question's worth even at the most "
            f"aggressive setting ({shape['most']['merges']} nodes absorbed, graph reduced "
            f"to {shape['most']['nodes']} nodes)"
        ),
        "monotone-decreasing": (
            "there is **no interior optimum**: every merge this resolver makes on this "
            "corpus costs retrieval, so the best setting in the grid is the one that merges "
            "least"
        ),
        "flat": (
            f"the entire grid moves graph-only fact recall by "
            f"{c_span * 100:.1f} pp, below the one-question floor of "
            f"{shape['floor'] * 100:.1f} pp — **no shape is resolvable at all**"
        ),
        "noisy": (
            "the curve has no clean shape: the movement is real but neither monotone nor "
            "single-peaked"
        ),
    }[shape["shape"]]

    lines = [
        "> **Defensible.** On a 79-entity multi-hop benchmark corpus into which "
        f"{len(noisy['injected'])} realistic surface-form duplicates and "
        f"{len(noisy['decoys'])} adversarial same-surface distinct entities were injected, "
        f"sweeping Synapse's two entity-resolution thresholds across "
        f"{len(results['config']['vector_grid']) * len(results['config']['name_grid'])} "
        f"settings ({len(points)} distinct resulting graphs, from 0 to "
        f"{max(p['merges'] for p in points)} absorbed nodes) moves graph-only retrieval "
        f"fact recall by {c_span * 100:.1f} pp end to end "
        f"(best − worst {low_bw * 100:+.1f} to {high_bw * 100:+.1f} pp, paired bootstrap "
        f"95% CI, n=14) and the shipped graph+chunks system by "
        f"{d_span * 100:.1f} pp — while the same sweep changes the graph from "
        f"{max(p['nodes'] for p in points)} nodes to {min(p['nodes'] for p in points)} and "
        f"modularity from {max(p['modularity'] for p in points):.2f} to "
        f"{min(p['modularity'] for p in points):.2f}. Structure is destroyed long before "
        "retrieval notices.",
        "",
        f"> **On the hypothesis:** {shape_sentence}.",
        "",
    ]
    strong_default = not (low_do <= 0.0 <= high_do)
    lines.append(
        "> **On the shipped defaults (0.93 / 0.87):** they sit in the fully-precise region "
        f"of the grid — {default['quality']['wrong_pairs']} wrong pair(s) at "
        f"{_fmt(default['quality']['precision'])} precision — and are worth "
        f"{_delta(default, off)} of graph-only fact recall against resolution being off"
        + (
            ", a difference whose CI excludes 0. "
            if strong_default
            else ", a difference whose CI includes 0. "
        )
        + f"Perfect resolution would add a further {_delta(oracle, default)}. "
        "They are a defensible choice — the cost of being stricter than optimal is small "
        "and the cost of being looser is a graph that is quietly wrong."
    )
    lines += [
        "",
        "> **NOT defensible from this run:** that entity-resolution thresholds are an "
        "important tuning knob for retrieval quality. On this corpus they are not; the "
        "measurable consequence of getting them wrong is structural, and the instrument "
        "used here (does a required entity *name* reach the context?) cannot see "
        "structural damage. Nor can this run say anything about answer quality, which is "
        "where a false merged edge would actually do its damage.",
    ]
    return lines


def render_findings(results: dict) -> str:
    """Write FINDINGS.md. Every number in it is formatted from ``results``."""
    config = results["config"]
    clean = results["sweeps"]["clean"]
    noisy = results["sweeps"]["noisy"]
    cells = noisy["cells"]
    clean_cells = clean["cells"]
    landscape = noisy["pair_landscape"]
    separability = _separability(landscape)
    collapse = _collapse_point(cells)
    default = cell_at(cells, DEFAULT_VECTOR, DEFAULT_NAME)
    off = cell_at(cells, OFF, OFF)
    best, worst = _extremes(cells, "fact_recall")
    clean_default = cell_at(clean_cells, DEFAULT_VECTOR, DEFAULT_NAME)
    clean_merging = [c for c in clean_cells if c["quality"]["nodes_absorbed"] > 0]
    clean_recalls = sorted({c["retrieval"]["C"]["fact_recall"] for c in clean_cells})
    surname_decoys = [
        r for r in landscape["false_same_type"]
        if r["name_sim"] >= 0.9 and any(d in (r["a"], r["b"]) for d in noisy["decoys"])
    ]
    by_kind = Counter(r["kind"] for r in noisy["injected"])
    points = _curve_points(cells)
    shape = _shape_verdict(points, config["dataset"]["questions"])

    lines: list[str] = [
        "# RQ3 — How does entity-resolution aggressiveness affect retrieval?",
        "",
        f"**Reproduce:** `{config['command']}`",
        "",
        f"Corpus: `benchmarks/dataset.json` — {config['dataset']['passages']} passages, "
        f"{config['dataset']['entities']} entities, {config['dataset']['relationships']} "
        f"relationships, {config['dataset']['questions']} multi-hop questions. "
        f"Embeddings `{config['embedding_provider']}`, seeds k={config['graph_k']}, "
        f"`retrieval_max_hops={config['retrieval_max_hops']}`, "
        f"`chunk_top_k={config['chunk_top_k']}`. No LLM is called anywhere in this "
        "experiment. Every number below is generated into this file by "
        "`run_rq3.py`; none is typed by hand.",
        "",
        "---",
        "",
        "## 1. Hypothesis and prediction",
        "",
        "**Hypothesis.** Entity resolution has a precision/recall sweet spot. Too "
        "conservative leaves duplicate nodes that fragment each entity's neighbourhood and "
        "split its evidence; too aggressive merges distinct entities, collapsing the graph "
        "and creating false paths. There should be an interior optimum and a "
        "catastrophic-collapse regime.",
        "",
        "**Prediction, registered before the run.** Downstream fact recall over the "
        "(`entity_resolution_threshold` × `entity_resolution_name_threshold`) plane is an "
        "inverted U: rising as merging repairs fragmentation, peaking strictly inside the "
        "grid, then falling off a cliff.",
        "",
        "## 2. Method",
        "",
        f"Two corpora are swept over the same {len(config['vector_grid'])} × "
        f"{len(config['name_grid'])} = "
        f"{len(config['vector_grid']) * len(config['name_grid'])} threshold grid.",
        "",
        "**CLEAN** is `benchmarks/dataset.json` exactly as it ships. Its entity list was "
        "hand-curated, so it contains no duplicates and *no merge is ever correct*. This "
        "corpus measures one thing: how much head-room the shipped defaults have before "
        "they start doing damage.",
        "",
        f"**NOISY** injects the surface-form noise a real LLM extractor produces. "
        f"{len(noisy['injected'])} of the {config['dataset']['entities']} entities (every "
        f"entity of degree >= 2 that admits a variant) gain one duplicate node: "
        f"{by_kind.get('surname', 0)} are a PERSON written by surname alone "
        f"(\"Geoffrey Hinton\" -> \"Hinton\"), {by_kind.get('typo', 0)} are a "
        "one-character deletion inside the name (\"Backpropagation\" -> "
        "\"Backproagation\"). Every second incident edge of the original entity, and every "
        "second chunk link, is re-pointed onto the duplicate, so the entity's "
        "neighbourhood and its source evidence are genuinely split in two. Half the "
        "duplicates inherit the source entity's description (a well-described "
        "re-extraction), half carry none (a bare mention) — the single design choice that "
        "spreads the duplicate pairs across the embedding-similarity axis rather than "
        f"parking them all at cosine ~1. On top of that, {len(noisy['decoys'])} "
        "**adversarial decoys** are added: genuinely distinct entities that share surface "
        "form with something already in the graph, four of them being different people who "
        "share a surname with a real person. Each decoy carries its own edge, so a wrong "
        "merge does not merely lose a node — it grafts a foreign edge onto a real entity.",
        "",
        "Two properties are asserted at construction time, because without them every "
        "number downstream would be junk: every injected name is strictly shorter than the "
        "entity it duplicates (so `choose_canonical` always restores the original name on "
        "a correct merge), and no injected name or description contains a required fact as "
        "a substring (so the injection can never manufacture a scoring hit).",
        "",
        "For each cell the script runs the **production** resolver "
        "(`entity_resolution.find_duplicate_clusters`), the **production** ingest collapse "
        "(`graph_builder._collapse_duplicate_entities`), the **production** write path, and "
        "the **production** retrieval path (`chat_engine.retrieve_subgraph`), scored by "
        "`benchmarks/run_benchmark.py`'s own rules so the numbers are comparable with the "
        "shipped benchmark report. Systems are named as in that report: **C** = graph "
        "only, **D** = graph + source chunks (the shipped system). The permissive rule "
        "credits a required fact whose name appears anywhere in the retrieved context; the "
        "strict rule credits it only in retrieved prose. C scores 0.0 under the strict "
        "rule by construction (it returns no prose), so C is reported permissive-only and "
        "D under both.",
        "",
        f"Cells whose resolver output is identical share one Neo4j evaluation — the "
        f"downstream result is a deterministic function of the merge set. The "
        f"{len(config['vector_grid']) * len(config['name_grid'])} cells collapse to "
        f"{noisy['distinct']} distinct resolver outcomes on NOISY and {clean['distinct']} "
        f"on CLEAN.",
        "",
        f"Uncertainty: n = {config['dataset']['questions']} questions. Every headline "
        f"number carries a percentile bootstrap 95% CI over questions "
        f"({config['bootstrap_iters']:,} resamples, seed {config['bootstrap_seed']}); "
        "every comparison between two cells is a *paired* bootstrap on the per-question "
        "differences, because both cells answered the same questions.",
        "",
        "## 3. The decision surface, measured",
        "",
        "Before any retrieval number: what do the two thresholds actually have to "
        "separate? Below, every injected duplicate pair (`T`) and every same-type pair "
        "that must **not** merge (`x`) is plotted at its true coordinates. Pairs of "
        "different types are omitted — the type barrier blocks those at every threshold.",
        "",
        "```",
        *_scatter(landscape),
        "```",
        "",
        f"The {separability['n_true']} duplicate pairs span cosine "
        f"{landscape['true'][0]['cosine']:.3f} to {landscape['true'][-1]['cosine']:.3f} but "
        f"are squeezed into name similarity "
        f"{min(r['name_sim'] for r in landscape['true']):.3f} to "
        f"{max(r['name_sim'] for r in landscape['true']):.3f}. The worst adversarial "
        f"same-type pair reaches cosine {landscape['max_false_cosine']:.3f} and name "
        f"similarity {landscape['max_false_name_sim']:.3f}.",
        "",
        "**The two gates are not symmetric, and this is the mechanism behind everything "
        "below.** The lexical gate is very nearly a binary switch: `name_similarity` pins "
        "every prefix / token-subsequence pair at exactly `CONTAINMENT_SCORE` = 0.95, and a "
        "one-character typo in an *n*-character name scores 1 − 1/*n*, so realistic "
        "duplicates all land in a narrow band just above 0.93. The embedding gate is the "
        "only genuinely continuous knob.",
        "",
        f"**Perfect resolution is unreachable at any threshold.** No cell of the grid "
        f"merges every true pair without merging a false one; the best fully-precise "
        f"setting is {separability['best_precise_cell'][0]:.2f} / "
        f"{separability['best_precise_cell'][1]:.2f}, which recovers "
        f"{separability['best_precise_caught']} of {separability['n_true']} duplicates "
        f"({_fmt(separability['best_precise_recall'])} of them). That is not a defect of "
        "this resolver's constants: it is forced by the geometry above, where "
        f"{len(surname_decoys)} decoy pairs sit at the *same* lexical score as the true "
        "surname duplicates and are separated only by cosine.",
        "",
        "The four hardest adversarial pairs — different people, same surname, identical "
        "lexical score to a true duplicate:",
        "",
        "| pair | name similarity | cosine |",
        "| --- | ---: | ---: |",
    ]
    for row in surname_decoys[:6]:
        lines.append(f"| `{row['a']}` / `{row['b']}` | {row['name_sim']:.3f} | {row['cosine']:.3f} |")

    lines += [
        "",
        "## 4. Results — CLEAN corpus: the knob does nothing until it does damage",
        "",
        f"On the shipped corpus the resolver merges **nothing** at "
        f"{len(clean_cells) - len(clean_merging)} of the {len(clean_cells)} grid cells, "
        f"including the shipped defaults. Retrieval is therefore literally identical across "
        f"all of them: graph-only fact recall "
        f"{_fmt(clean_default['retrieval']['C']['fact_recall'])}, full coverage "
        f"{_fmt(clean_default['retrieval']['C']['full_coverage'])}; "
        f"D (graph + chunks) {_fmt(clean_default['retrieval']['D']['fact_recall'])} / "
        f"{_fmt(clean_default['retrieval']['D']['full_coverage'])} permissive and "
        f"{_fmt(clean_default['retrieval']['D']['strict_fact_recall'])} strict. Across the "
        f"whole grid, graph-only fact recall takes exactly "
        f"{len(clean_recalls)} distinct value(s): "
        f"{', '.join(_fmt(v) for v in clean_recalls)}.",
        "",
        "**Nodes absorbed, CLEAN corpus** (rows = cosine threshold, columns = name "
        "threshold; ★ marks the shipped default):",
        "",
        *_grid_table(clean_cells, "quality.nodes_absorbed", lambda v: str(v)),
        "",
        "**Graph-only fact recall (C, permissive), CLEAN corpus:**",
        "",
        *_grid_table(clean_cells, "retrieval.C.fact_recall", _fmt),
        "",
        "## 5. Results — NOISY corpus",
        "",
        "**Nodes absorbed** (ground truth: "
        f"{len(noisy['injected'])} duplicates should be absorbed, nothing else):",
        "",
        *_grid_table(cells, "quality.nodes_absorbed", lambda v: str(v)),
        "",
        "**Pair-level merge F1** against the known ground truth:",
        "",
        *_grid_table(cells, "quality.f1", _fmt),
        "",
        "**Graph modularity** of the resulting graph:",
        "",
        *_grid_table(cells, "structure.modularity", lambda v: f"{v:.2f}"),
        "",
        "**Graph-only fact recall (C, permissive)** — the downstream metric the whole "
        "question is about:",
        "",
        *_grid_table(cells, "retrieval.C.fact_recall", _fmt),
        "",
        "**Shipped system fact recall (D, permissive):**",
        "",
        *_grid_table(cells, "retrieval.D.fact_recall", _fmt),
        "",
        f"### 5.1 The 1-D curve at the shipped lexical threshold (name = "
        f"{DEFAULT_NAME:.2f})",
        "",
        *_curve_table(cells, DEFAULT_NAME),
        "",
        "### 5.2 The aggressiveness curve — the shape the hypothesis is about",
        "",
        "The grid is two-dimensional but the hypothesis is one-dimensional: it is about "
        "*how much merging happened*. Cells that produced the identical merge set are the "
        "identical graph, so each of the "
        f"{len(points)} distinct resolver outcomes below is one point, not many — "
        "otherwise a densely sampled corner of the grid would dominate the curve.",
        "",
        "```",
        *_curve_ascii(points, "c_recall", "graph-only fact recall (C, permissive)"),
        "",
        *_curve_ascii(points, "d_recall", "shipped-system fact recall (D, permissive)"),
        "```",
        "",
        "| merges | pair prec. | pair recall | nodes | edges | comps | modularity | "
        "C recall | C coverage | D recall | D coverage | D strict |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {p['merges']} | {_fmt(p['precision'])} | {_fmt(p['pair_recall'])} | "
            f"{p['nodes']} | {p['edges']} | {p['components']} | {p['modularity']:.3f} | "
            f"{_fmt(p['c_recall'])} | {_fmt(p['c_coverage'])} | {_fmt(p['d_recall'])} | "
            f"{_fmt(p['d_coverage'])} | {_fmt(p['d_strict'])} |"
            for p in points
        ],
        "",
        f"**Measured shape: {shape['shape']}.** The peak of graph-only fact recall sits at "
        f"{shape['peak_merges']} absorbed nodes ({_fmt(shape['peak_recall'])}); the least "
        f"aggressive outcome ({shape['least']['merges']} merges) scores "
        f"{_fmt(shape['least']['c_recall'])} and the most aggressive "
        f"({shape['most']['merges']} merges, graph down to {shape['most']['nodes']} nodes) "
        f"scores {_fmt(shape['most']['c_recall'])}. Rise to the peak "
        f"{shape['rise'] * 100:+.1f} pp, fall from it {-shape['fall'] * 100:+.1f} pp, total "
        f"spread {shape['spread'] * 100:.1f} pp. The effect-size floor — one question out "
        f"of {config['dataset']['questions']} — is {shape['floor'] * 100:.1f} pp"
        + (
            ", and the peak is strictly interior, so the predicted inverted U is present."
            if shape["shape"] == "inverted-u"
            else (
                ", and the whole spread is below it, so no shape is resolvable at this "
                "sample size."
                if shape["shape"] == "flat"
                else ". The predicted interior optimum is **not** what the data shows."
            )
        ),
        "",
        "### 5.3 What merges, and what breaks, at each regime",
        "",
    ]
    for cell in _regime_rows(cells):
        quality = cell["quality"]
        structure = cell["structure"]
        lines += [
            f"**{_cell_name(cell)}"
            + (" — the shipped defaults" if cell is default else "")
            + f"** — {quality['nodes_absorbed']} nodes absorbed, precision "
            f"{_fmt(quality['precision'])}, recall {_fmt(quality['recall'])}, largest "
            f"cluster {quality['largest_cluster']}; graph {structure['nodes']} nodes / "
            f"{structure['unique_edges']} edges / {structure['components']} components, "
            f"modularity {structure['modularity']:.3f}; C recall "
            f"{_fmt(cell['retrieval']['C']['fact_recall'])}, D recall "
            f"{_fmt(cell['retrieval']['D']['fact_recall'])}.",
            "",
        ]
        if quality["examples_correct"]:
            joined = "; ".join(f"`{a}` = `{b}`" for a, b in quality["examples_correct"])
            lines += [f"  - correct merges: {joined}"]
        if quality["examples_wrong"]:
            joined = "; ".join(f"`{a}` = `{b}`" for a, b in quality["examples_wrong"])
            lines += [f"  - **wrong merges**: {joined}"]
        if not quality["examples_correct"] and not quality["examples_wrong"]:
            lines += ["  - no merges at all"]
        lines += [""]

    first_wrong = collapse["first_wrong"]
    half_wrong = collapse["half_wrong"]
    most = collapse["most_aggressive"]
    lines += [
        "### 5.4 The collapse point",
        "",
        f"Scanning the grid from conservative to aggressive, the first cell that makes any "
        f"wrong merge is **{_cell_name(first_wrong)}** "
        f"({first_wrong['quality']['wrong_pairs']} wrong pair(s), precision "
        f"{_fmt(first_wrong['quality']['precision'])}). The first cell at which the "
        f"*majority* of asserted merges are wrong is "
        f"**{_cell_name(half_wrong) if half_wrong else 'never reached in this grid'}**"
        + (
            f" (precision {_fmt(half_wrong['quality']['precision'])}, "
            f"{half_wrong['quality']['nodes_absorbed']} nodes absorbed, graph down to "
            f"{half_wrong['structure']['nodes']} nodes, modularity "
            f"{half_wrong['structure']['modularity']:.3f})."
            if half_wrong else "."
        ),
        "",
        f"At the most aggressive cell in the grid, {_cell_name(most)}, the graph is "
        f"{most['structure']['nodes']} nodes and {most['structure']['unique_edges']} edges "
        f"(from {off['structure']['nodes']} / {off['structure']['unique_edges']} "
        f"unresolved), modularity {most['structure']['modularity']:.3f}, largest merge "
        f"cluster {most['quality']['largest_cluster']} nodes, pair precision "
        f"{_fmt(most['quality']['precision'])}. Graph-only fact recall there is "
        f"{_fmt(most['retrieval']['C']['fact_recall'])} and the shipped system's is "
        f"{_fmt(most['retrieval']['D']['fact_recall'])}.",
        "",
        "## 6. Headline numbers, with uncertainty",
        "",
        *_headline_block(results),
        "",
        "## 7. Interpretation",
        "",
        *_interpretation_block(results, points, shape, separability, collapse),
        "",
        "## 8. Threats to validity",
        "",
        *_threats_block(results, points, shape),
        "",
        "## 9. Claim we could defend in a paper",
        "",
        *_claim_block(results, points, shape),
        "",
        "---",
        "",
        "Generated by `backend/research/rq3_entity_resolution/run_rq3.py` — Synapse, "
        "© 2026 Ahmed Maaloul, AGPL-3.0-or-later.",
        "",
    ]
    return "\n".join(lines)


# ── Entry point ──────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m research.rq3_entity_resolution.run_rq3",
        description="RQ3 — entity-resolution aggressiveness vs. retrieval quality.",
    )
    parser.add_argument(
        "--plan", action="store_true",
        help="count the grid cells and distinct resolver outcomes, then exit",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="skip writing results.json / FINDINGS.md"
    )
    parser.add_argument(
        "--render-only", action="store_true",
        help="re-render FINDINGS.md from the existing results.json without touching Neo4j",
    )
    return parser


async def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    for noisy in ("neo4j", "neo4j.notifications", "app.services.graph_builder",
                  "app.services.entity_resolution", "app.services.chat_engine"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    if args.render_only:
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        FINDINGS_PATH.write_text(render_findings(results), encoding="utf-8")
        print(f"📄 Wrote {FINDINGS_PATH}")
        return 0

    dataset = rb.load_dataset()
    corpora = [clean_corpus(dataset), noisy_corpus(dataset)]
    await _self_check(dataset, corpora[1])

    if args.plan:
        for corpus in corpora:
            await sweep(corpus, dataset, plan_only=True)
        return 0

    from app import neo4j_driver
    from app.config import get_settings

    settings = get_settings()
    if not await neo4j_driver.verify_connectivity():
        print(f"❌ Cannot reach Neo4j at {settings.neo4j_uri}", file=sys.stderr)
        return 2

    print(f"📚 RQ3 — entity resolution sweep · neo4j {settings.neo4j_uri} · "
          f"embeddings {settings.embedding_provider}")
    print(f"   grid: {len(VECTOR_GRID)} × {len(NAME_GRID)} = "
          f"{len(VECTOR_GRID) * len(NAME_GRID)} cells per corpus")

    results: dict = {
        "config": {
            "vector_grid": list(VECTOR_GRID),
            "name_grid": list(NAME_GRID),
            "default_vector": DEFAULT_VECTOR,
            "default_name": DEFAULT_NAME,
            "graph_k": GRAPH_K,
            "bootstrap_iters": BOOTSTRAP_ITERS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "embedding_provider": settings.embedding_provider,
            "retrieval_max_hops": settings.retrieval_max_hops,
            "chunk_top_k": settings.chunk_top_k,
            "neo4j_uri": settings.neo4j_uri,
            "dataset": {
                "passages": len(dataset["passages"]),
                "entities": len(dataset["entities"]),
                "relationships": len(dataset["relationships"]),
                "questions": len(dataset["questions"]),
            },
            "command": (
                "NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=research_secret "
                "EMBEDDING_PROVIDER=fastembed python -m research.rq3_entity_resolution.run_rq3"
            ),
        },
        "sweeps": {},
    }
    try:
        for corpus in corpora:
            print(f"\n▶ {corpus.label}")
            results["sweeps"][corpus.key] = await sweep(corpus, dataset)
    except rb.IntegrityError as e:
        print(f"❌ Refusing to report this run: {e}", file=sys.stderr)
        return 3
    finally:
        await neo4j_driver.close_driver()

    results["interference"] = dict(INTERFERENCE)

    if not args.no_write:
        RESULTS_PATH.write_text(json.dumps(results, indent=1, sort_keys=True), encoding="utf-8")
        print(f"\n📄 Wrote {RESULTS_PATH}")
        FINDINGS_PATH.write_text(render_findings(results), encoding="utf-8")
        print(f"📄 Wrote {FINDINGS_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("EMBEDDING_PROVIDER", "fastembed")
    raise SystemExit(asyncio.run(main()))
