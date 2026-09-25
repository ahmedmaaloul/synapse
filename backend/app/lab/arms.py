# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse Lab — the arms: retrieval approaches compared side by side.

Every arm answers ONE question the same way: ``await arm.retrieve(question,
k=..., seed=...)`` returns an :class:`~app.lab.evidence.Evidence` of RANKED
units. No arm renders a prompt, and no arm cuts to a token budget — the shared
packer does both, identically for all of them. The only truncation an arm does
is its own ``k`` (documented per arm as ``k_role``: how many seeds, or how many
passages, it keeps). All arms are read-only against Neo4j.

FAMILIES (the UI groups by these):

  Evidence floors (``null``) — the controls every other number is read against.
    • ``null_closed_book`` (N0) — no evidence; the reader answers from memory.
    • ``null_vocabulary``  (N1) — every entity name in the graph, alphabetical,
      identical for every question. It "contains" nearly every answer entity,
      which is why containment-style metrics rate it highly while it carries no
      question-specific evidence at all.
    • ``null_random``      (N2) — seeded random passages from the same corpus,
      packed to the same budget.

Passage-ranking arms (``k_role = "passages"``) return ``k`` passages, and the
RUNNER chooses ``k``: by default as many as the run's largest capped budget
holds (``runner.LabRun.passage_policy``), so no floor or baseline is compared
at a smaller context than the arms it is read against.

  Passage baselines (``passage``)
    • ``bm25``  — Lucene BM25 over Neo4j's ``:Chunk`` full-text index.
    • ``dense`` — cosine top-k over the ``:Chunk`` vector index.

  Graph arms (``graph``)
    • ``synapse_d``    — the shipped path (``chat_engine.retrieve_subgraph``
      with source chunks), exploded into units.
    • ``synapse_lean`` — ours: PathRAG-style flow-pruned paths between the
      hybrid seeds, with a LiteRAG-style log-degree hub penalty.
    • ``ppr``          — HippoRAG-2-style Personalized PageRank over entities
      and passages, WITHOUT the LLM triple filter.

The "-style" arms are re-implementations over Synapse's own graph, not the
authors' code; their descriptions say so.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import networkx as nx

from app.lab.evidence import Evidence, EvidenceUnit
from app.neo4j_driver import execute_query
from app.services import chat_engine, chunk_store
from app.services.graph_schema import CHUNK_FULLTEXT_INDEX

logger = logging.getLogger(__name__)

FAMILY_TITLES: dict[str, str] = {
    "null": "Evidence floors",
    "passage": "Passage baselines",
    "graph": "Graph arms",
}

REPO_URL = "https://github.com/ahmedmaaloul/synapse"


# ── The protocol ─────────────────────────────────────────────────────────────
@runtime_checkable
class Arm(Protocol):
    """What the runner, the API and the UI need from an arm."""

    name: str
    family: str
    title: str
    description: str
    source: dict[str, str]
    retrieval_llm_calls: int
    needs_graph: bool

    async def retrieve(self, question: str, *, k: int, seed: int) -> Evidence: ...


class BaseArm:
    """Shared metadata plumbing. Subclasses set the class attributes."""

    name: str = ""
    family: str = "graph"
    title: str = ""
    description: str = ""
    source: dict[str, str] = {}
    retrieval_llm_calls: int = 0
    needs_graph: bool = False
    #: What ``k`` bounds: "seeds" (entities), "passages" (chunks) or "none".
    k_role: str = "passages"
    #: Bumped whenever an arm's behaviour changes, so config hashes change too.
    version: str = "1"

    @property
    def is_null(self) -> bool:
        return self.family == "null"

    def config(self) -> dict[str, Any]:
        """Every knob that shapes this arm's output (hashed into the manifest)."""
        return {"name": self.name, "version": self.version, "k_role": self.k_role}

    def config_hash(self) -> str:
        blob = json.dumps(self.config(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "family_title": FAMILY_TITLES.get(self.family, self.family),
            "title": self.title,
            "description": self.description,
            "source": dict(self.source),
            "retrieval_llm_calls": self.retrieval_llm_calls,
            "needs_graph": self.needs_graph,
            "is_null": self.is_null,
            "k_role": self.k_role,
        }

    async def retrieve(self, question: str, *, k: int, seed: int) -> Evidence:  # pragma: no cover
        raise NotImplementedError


# ── Shared graph reads (all read-only) ───────────────────────────────────────
#: Four counts Neo4j answers from its count store (no scan): nodes by label,
#: relationships leaving an :Entity, and MENTIONED_IN relationships. Procedural
#: memory (other labels) never moves them, so its writes never invalidate a cache.
#: Four plain statements rather than one with ``CALL { }`` subqueries: that form
#: is deprecated on recent 5.x and its replacement does not parse on older 5.x.
FINGERPRINT_QUERIES: dict[str, str] = {
    "entities": "MATCH (e:Entity) RETURN count(e) AS n",
    "chunks": "MATCH (c:Chunk) RETURN count(c) AS n",
    "relations": "MATCH (:Entity)-[r]->() RETURN count(r) AS n",
    "mentions": "MATCH ()-[m:MENTIONED_IN]->() RETURN count(m) AS n",
}

ENTITY_NAMES_QUERY = "MATCH (e:Entity) WHERE e.name IS NOT NULL RETURN e.name AS name"

CHUNK_IDS_QUERY = "MATCH (c:Chunk) WHERE c.id IS NOT NULL RETURN c.id AS id"

CHUNKS_BY_ID_QUERY = """
MATCH (c:Chunk) WHERE c.id IN $ids
RETURN c.id AS id, c.text AS text, c.document AS document, c.index AS index
"""

BM25_QUERY = """
CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
RETURN node.id AS id, node.text AS text, node.document AS document,
       node.index AS index, score
ORDER BY score DESC, id ASC
LIMIT $k
"""

ENTITY_INFO_QUERY = """
MATCH (n:Entity) WHERE n.name IN $names
OPTIONAL MATCH (n)-[r]-(:Entity)
RETURN n.name AS name, n.type AS type, n.description AS description, count(r) AS degree
"""

PPR_RELATIONS_QUERY = """
MATCH (a:Entity)-[r]->(b:Entity)
WHERE a.name IS NOT NULL AND b.name IS NOT NULL AND a.name <> b.name
RETURN a.name AS source, b.name AS target
"""

PPR_MENTIONS_QUERY = """
MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE e.name IS NOT NULL AND c.id IS NOT NULL
RETURN e.name AS entity, c.id AS chunk
"""

PPR_CHUNKS_QUERY = """
MATCH (c:Chunk) WHERE c.id IS NOT NULL
RETURN c.id AS id, c.text AS text, c.document AS document, c.index AS index
"""


def _fingerprint_key(fp: dict[str, Any]) -> str:
    return "|".join(str(int(fp.get(k) or 0)) for k in FINGERPRINT_QUERIES)


async def graph_fingerprint() -> dict[str, int]:
    """Entity / chunk / relation / mention counts — the cache key for whole-graph reads.

    Cheap (count queries) and read-only. Two graphs with the same four counts
    would share a cache entry; within one run the graph does not change, which
    is the only guarantee the caches need.
    """
    counts: dict[str, int] = {}
    for key, query in FINGERPRINT_QUERIES.items():
        rows = await execute_query(query) or []
        counts[key] = int((rows[0].get("n") if rows else 0) or 0)
    return counts


#: kind -> (fingerprint key, value). One entry per kind: a run reads one graph.
_CACHE: dict[str, tuple[str, Any]] = {}


def clear_caches() -> None:
    """Drop every whole-graph cache (tests, or after the graph changed)."""
    _CACHE.clear()


async def _cached(kind: str, build) -> Any:
    key = _fingerprint_key(await graph_fingerprint())
    hit = _CACHE.get(kind)
    if hit is not None and hit[0] == key:
        return hit[1]
    value = await build()
    _CACHE[kind] = (key, value)
    return value


async def _seed_names(question: str, k: int) -> list[str]:
    """The shipped hybrid ranker (vector + full-text, interleaved), capped at ``k``."""
    return list(await chat_engine._rank_seed_names(question, k))[: max(0, int(k))]


def _prose(chunk: dict, score: float) -> EvidenceUnit:
    """One stored chunk as a ``prose`` unit — the same rendering for every arm."""
    return EvidenceUnit(
        text=str(chunk.get("text") or "").strip(),
        kind="prose",
        source_id=str(chunk.get("document") or "") or None,
        score=float(score),
    )


def _entity_block(name: str, etype: Any, description: Any) -> str:
    """An entity block in ``chat_engine``'s own format, without relationship lines."""
    text = f"Entity: {name} (Type: {etype})"
    if description:
        text += f"\n  Description: {description}"
    return text


def question_seed(seed: int, question: str) -> int:
    """A stable integer from ``(seed, question)`` — N2's per-question RNG seed."""
    digest = hashlib.sha256(f"{int(seed)}\x1f{question}".encode()).hexdigest()
    return int(digest[:16], 16)


# ── Evidence floors ──────────────────────────────────────────────────────────
NULL_SOURCE = {
    "citation": "Synapse null controls (Maaloul, 2026): closed-book, vocabulary and "
    "random-context floors",
    "url": REPO_URL,
}


class NullClosedBook(BaseArm):
    name = "null_closed_book"
    family = "null"
    title = "N0 · Closed-book"
    description = (
        "No evidence at all: the reader answers from its own knowledge. The floor every "
        "retrieval arm has to clear."
    )
    source = NULL_SOURCE
    needs_graph = False
    k_role = "none"

    async def retrieve(self, question: str, *, k: int, seed: int) -> Evidence:
        return Evidence.empty(self.name)


class NullVocabulary(BaseArm):
    name = "null_vocabulary"
    family = "null"
    title = "N1 · Vocabulary null"
    description = (
        "Every entity name in the graph, alphabetical, the SAME for every question (the "
        "question is ignored). Containment-style recall rates it near 100% although it "
        "carries no question-specific evidence; under a budget an alphabetical prefix is kept."
    )
    source = NULL_SOURCE
    needs_graph = True
    k_role = "none"

    async def retrieve(self, question: str, *, k: int, seed: int) -> Evidence:
        async def build() -> list[str]:
            rows = await execute_query(ENTITY_NAMES_QUERY) or []
            return sorted({str(r["name"]) for r in rows if r.get("name")})

        names = await _cached("entity_names", build)
        if not names:
            return Evidence.empty(self.name, names=0)
        unit = EvidenceUnit(text="\n".join(names), kind="name_list", source_id=None, score=1.0)
        return Evidence([unit], self.name, {"names": len(names)})


class NullRandom(BaseArm):
    name = "null_random"
    family = "null"
    title = "N2 · Random context"
    description = (
        "Passages drawn at random from the same corpus (seeded by run seed + question), as "
        "many as the budget holds: what 'some context' is worth without retrieval."
    )
    source = NULL_SOURCE
    needs_graph = False
    k_role = "passages"

    async def retrieve(self, question: str, *, k: int, seed: int) -> Evidence:
        async def build() -> list[str]:
            rows = await execute_query(CHUNK_IDS_QUERY) or []
            return sorted({str(r["id"]) for r in rows if r.get("id")})

        ids = await _cached("chunk_ids", build)
        take = min(max(0, int(k)), len(ids))
        if not take:
            return Evidence.empty(self.name, pool=len(ids))
        rng = random.Random(question_seed(seed, question))
        picked = rng.sample(ids, take)
        rows = await execute_query(CHUNKS_BY_ID_QUERY, {"ids": picked}) or []
        by_id = {str(r.get("id")): r for r in rows}
        units = [
            _prose(by_id[cid], float(take - i)) for i, cid in enumerate(picked) if cid in by_id
        ]
        return Evidence(units, self.name, {"pool": len(ids), "picked": picked})


# ── Passage baselines ────────────────────────────────────────────────────────
class BM25(BaseArm):
    name = "bm25"
    family = "passage"
    title = "BM25 (Lucene)"
    description = (
        "The top source passages by Lucene BM25 over Neo4j's :Chunk full-text index, as many "
        "as the budget holds, with the same query escaping as the chat engine. No embeddings, "
        "no LLM."
    )
    source = {
        "citation": "Robertson & Zaragoza, 'The Probabilistic Relevance Framework: BM25 and "
        "Beyond', FnTIR 3(4), 2009",
        "url": "https://doi.org/10.1561/1500000019",
    }
    needs_graph = False
    k_role = "passages"

    def config(self) -> dict[str, Any]:
        return {**super().config(), "index": CHUNK_FULLTEXT_INDEX}

    async def retrieve(self, question: str, *, k: int, seed: int) -> Evidence:
        query = chat_engine._keyword_query(question)
        if not query or k <= 0:
            return Evidence.empty(self.name, query=query)
        rows = await execute_query(
            BM25_QUERY, {"index": CHUNK_FULLTEXT_INDEX, "q": query, "k": int(k)}
        )
        rows = sorted(rows or [], key=lambda r: (-float(r.get("score") or 0.0), str(r.get("id"))))
        units = [_prose(r, float(r.get("score") or 0.0)) for r in rows[: int(k)]]
        return Evidence([u for u in units if u.text], self.name, {"query": query})


class Dense(BaseArm):
    name = "dense"
    family = "passage"
    title = "Dense passages"
    description = (
        "The top source passages by cosine similarity with the configured embedder over the "
        ":Chunk vector index, as many as the budget holds — plain vector RAG. No LLM."
    )
    source = {
        "citation": "Lewis et al., 'Retrieval-Augmented Generation for Knowledge-Intensive NLP "
        "Tasks', NeurIPS 2020, arXiv:2005.11401",
        "url": "https://arxiv.org/abs/2005.11401",
    }
    needs_graph = False
    k_role = "passages"

    async def retrieve(self, question: str, *, k: int, seed: int) -> Evidence:
        if k <= 0:
            return Evidence.empty(self.name)
        rows = list(await chunk_store.search_chunks(question, int(k)) or [])
        rows.sort(key=lambda r: (-float(r.get("score") or 0.0), str(r.get("id"))))
        units = [_prose(r, float(r.get("score") or 0.0)) for r in rows[: int(k)]]
        return Evidence([u for u in units if u.text], self.name)


# ── Graph arms ───────────────────────────────────────────────────────────────
def explode_retrieval(retrieval: Any) -> tuple[list[EvidenceUnit], dict[str, Any]]:
    """Turn a shipped ``Retrieval`` into ranked units, faithfully.

    The shipped context is ``entity blocks [+ "Reasoning paths:" lines]
    [+ "Source excerpts:" blocks]`` (local) or ``community blocks [+ excerpts]``
    (global). Entity and community blocks are kept verbatim (relationship lines
    included), each path line becomes a ``path`` unit, and each excerpt the
    engine actually included (``retrieval.sources``) becomes a ``prose`` unit
    carrying the chunk text verbatim — the ``[Sn] doc (chunk i)`` provenance
    header is dropped, as for every other arm's prose.

    Scores fall with position, so packing by score keeps the engine's own
    priority (seeds, then paths, then excerpts): at an unbounded budget the
    units are exactly the shipped context's content.
    """
    context = str(retrieval[0] or "") if retrieval else ""
    citations = list(retrieval[1] or []) if retrieval else []
    mode = getattr(retrieval, "mode", "local")
    sources = list(getattr(retrieval, "sources", []) or [])

    head, _sep, _tail = context.partition(chat_engine.SOURCES_HEADING)
    head, _psep, paths_part = head.partition(chat_engine.PATHS_HEADING)

    ordered: list[tuple[str, str, str | None]] = []  # (kind, text, source_id)
    names = [c.get("name") for c in citations if c.get("kind") == "entity"]
    community_ids = [c.get("id") for c in citations if c.get("kind") == "community"]
    entity_i = community_i = 0
    for block in head.split("\n\n"):
        block = block.strip("\n")
        if block.startswith("Entity: "):
            sid = names[entity_i] if entity_i < len(names) else None
            entity_i += 1
            ordered.append(("entity", block, sid))
        elif block.startswith("Community: "):
            cid = community_ids[community_i] if community_i < len(community_ids) else None
            community_i += 1
            ordered.append(("community", block, str(cid) if cid is not None else None))
    for line in paths_part.split("\n"):
        if line.startswith("  - "):
            ordered.append(("path", line[4:].strip(), None))
    for chunk in sources:
        text = str(chunk.get("text") or "").strip()
        if text:
            ordered.append(("prose", text, str(chunk.get("document") or "") or None))

    total = len(ordered)
    units = [
        EvidenceUnit(text=text, kind=kind, source_id=sid, score=(total - i) / total)
        for i, (kind, text, sid) in enumerate(ordered)
    ]
    meta = {
        "mode": mode,
        "seeds": names,
        "paths": sum(1 for kind, *_ in ordered if kind == "path"),
        "excerpts": sum(1 for kind, *_ in ordered if kind == "prose"),
        "shipped_context_chars": len(context),
    }
    return units, meta


class SynapseD(BaseArm):
    name = "synapse_d"
    family = "graph"
    title = "Synapse GraphRAG (shipped)"
    description = (
        "The shipped retrieval path, unmodified: routed local/global search, hybrid seeds, "
        "2-hop reasoning paths and source excerpts (under its own excerpt character cap), "
        "exploded into units in the engine's own priority order."
    )
    source = {
        "citation": "Synapse (this repository); local search after Edge et al., 'From Local "
        "to Global: A Graph RAG Approach', arXiv:2404.16130",
        "url": "https://arxiv.org/abs/2404.16130",
    }
    needs_graph = True
    k_role = "seeds"

    def config(self) -> dict[str, Any]:
        from app.config import get_settings

        s = get_settings()
        return {
            **super().config(),
            "retrieval_max_hops": s.retrieval_max_hops,
            "max_reasoning_paths": s.max_reasoning_paths,
            "chunk_top_k": s.chunk_top_k,
            "chunk_context_max_chars": s.chunk_context_max_chars,
            "chunk_retrieval_enabled": s.chunk_retrieval_enabled,
            "query_routing_enabled": s.query_routing_enabled,
        }

    async def retrieve(self, question: str, *, k: int, seed: int) -> Evidence:
        retrieval = await chat_engine.retrieve_subgraph(question, k=max(1, int(k)))
        units, meta = explode_retrieval(retrieval)
        return Evidence(units, self.name, meta)


# -- Synapse-Lean: flow-pruned paths + hub penalty ---------------------------
LEAN_ALPHA = 0.7  # PathRAG's decay rate
LEAN_THETA = 0.02  # prune a path once its propagated resource falls below this
LEAN_TOP_PATHS = 10  # K
LEAN_HOPS = 2  # neighbourhood radius fetched around the seeds
LEAN_MAX_PATH_EDGES = 2 * LEAN_HOPS  # a seed-to-seed path lies inside both radii
LEAN_MAX_CANDIDATES = 20_000  # hard stop on enumeration, whatever the graph
LEAN_EXCERPT_WEIGHT = 0.9  # an excerpt ranks just below the entity that pulled it in


@dataclass(frozen=True, slots=True)
class ScoredPath:
    nodes: tuple[str, ...]
    rels: tuple[str, ...]
    dirs: tuple[bool, ...]
    #: PathRAG reliability: mean propagated resource over the path's nodes after the start.
    reliability: float
    #: Geometric mean of 1 / (1 + ln(1 + degree)) over intermediate nodes (1.0 if none).
    hub: float
    score: float
    text: str


def hub_factor(degree: int) -> float:
    """LiteRAG-style log-degree down-weighting: ``1 / (1 + ln(1 + degree))``."""
    return 1.0 / (1.0 + math.log1p(max(0, int(degree))))


def _unique_adjacency(edges: list[dict]) -> dict[str, list[tuple[str, str, bool]]]:
    """Undirected adjacency with ONE edge per neighbour pair, deterministically chosen.

    Parallel relations between the same two entities collapse to the
    lexicographically first ``(rel, direction)`` so a pair is never counted as
    several paths (and never divides a node's resource several times).
    """
    options: dict[tuple[str, str], list[tuple[str, bool]]] = {}
    for u, pairs in chat_engine._adjacency(edges).items():
        for v, rel, outgoing in pairs:
            options.setdefault((u, v), []).append((str(rel), outgoing))
    adj: dict[str, list[tuple[str, str, bool]]] = {}
    for (u, v), rels in options.items():
        # Smallest relation name; on a tie the forward direction (``not True`` sorts first).
        rel, outgoing = min(rels, key=lambda r: (r[0], not r[1]))
        adj.setdefault(u, []).append((v, rel, outgoing))
    for u in adj:
        adj[u].sort()
    return adj


def score_paths(
    seeds: list[str],
    edges: list[dict],
    degrees: dict[str, int] | None = None,
    *,
    alpha: float | None = None,
    theta: float | None = None,
    max_edges: int | None = None,
    top_k: int | None = None,
) -> list[ScoredPath]:
    """PathRAG flow-based pruning between seed pairs, with a hub penalty. Pure.

    From each seed ``s`` (best-ranked first) a resource of 1.0 flows along
    simple paths: stepping from ``u`` to ``v`` passes ``alpha · S(u) / |N(u)|``
    (PathRAG's decay-and-split). A branch whose resource drops below ``theta``
    is pruned, which is what bounds the enumeration on hub-heavy graphs. A path
    that reaches a WORSE-ranked seed is a candidate (each unordered seed pair is
    therefore scored once, from its better endpoint).

    ``score = reliability · hub``: reliability is the mean resource over the
    nodes after the start (PathRAG), hub is the geometric mean of
    :func:`hub_factor` over intermediate nodes using ``degrees`` (full-graph
    degree; the subgraph degree when absent). Top ``top_k`` by
    ``(-score, length, text)`` — fully deterministic.

    Unset knobs read the module constants AT CALL TIME (``LEAN_ALPHA`` …), the
    same values ``SynapseLean.config()`` hashes into the manifest.
    """
    alpha = LEAN_ALPHA if alpha is None else alpha
    theta = LEAN_THETA if theta is None else theta
    max_edges = LEAN_MAX_PATH_EDGES if max_edges is None else max_edges
    top_k = LEAN_TOP_PATHS if top_k is None else top_k
    ranked = list(dict.fromkeys(s for s in seeds if s))
    adj = _unique_adjacency(edges)
    rank = {name: i for i, name in enumerate(ranked)}
    degrees = degrees or {}
    candidates: dict[tuple[str, ...], ScoredPath] = {}
    budget = LEAN_MAX_CANDIDATES

    def degree(node: str) -> int:
        return int(degrees.get(node, len(adj.get(node, ()))))

    for source in ranked:
        if source not in adj:
            continue
        stack: list[tuple[list[str], list[str], list[bool], list[float]]] = [
            ([source], [], [], [1.0])
        ]
        while stack and budget > 0:
            nodes, rels, dirs, flows = stack.pop()
            here = nodes[-1]
            neighbours = adj.get(here, [])
            share = alpha * flows[-1] / max(1, len(neighbours))
            if share < theta:
                continue
            for neighbour, rel, outgoing in reversed(neighbours):
                if neighbour in nodes:
                    continue
                path = (nodes + [neighbour], rels + [rel], dirs + [outgoing], flows + [share])
                if neighbour in rank and rank[neighbour] > rank[source]:
                    budget -= 1
                    p_nodes, p_rels, p_dirs, p_flows = path
                    key = tuple(p_nodes)
                    if key not in candidates:
                        inner = p_nodes[1:-1]
                        logs = [math.log(hub_factor(degree(n))) for n in inner]
                        hub = math.exp(sum(logs) / len(logs)) if logs else 1.0
                        reliability = sum(p_flows[1:]) / (len(p_flows) - 1)
                        candidates[key] = ScoredPath(
                            nodes=key,
                            rels=tuple(p_rels),
                            dirs=tuple(p_dirs),
                            reliability=reliability,
                            hub=hub,
                            score=reliability * hub,
                            text=chat_engine._render_path(p_nodes, p_rels, p_dirs),
                        )
                if len(rels) + 1 < max_edges:
                    stack.append(path)

    ordered = sorted(candidates.values(), key=lambda p: (-p.score, len(p.rels), p.text))
    return ordered[: max(0, int(top_k))]


def path_scores(seeds: list[str], paths: list[ScoredPath]) -> list[float]:
    """Each kept path's unit score: ``(score / best) · max(seed weight of its ends)``."""
    weight = {name: 1.0 / (1.0 + i) for i, name in enumerate(seeds)}
    best = max((p.score for p in paths), default=0.0)
    return [
        (p.score / best if best > 0 else 0.0)
        * max(weight.get(p.nodes[0], 0.0), weight.get(p.nodes[-1], 0.0))
        for p in paths
    ]


def entity_mass(seeds: list[str], paths: list[ScoredPath]) -> dict[str, float]:
    """An entity's relevance: its seed weight, or the best kept path through it."""
    mass = {name: 1.0 / (1.0 + i) for i, name in enumerate(seeds)}
    for path, score in zip(paths, path_scores(seeds, paths), strict=True):
        for node in path.nodes:
            mass[node] = max(mass.get(node, 0.0), score)
    return mass


def excerpt_anchors(seeds: list[str], paths: list[ScoredPath]) -> list[str]:
    """Names whose source excerpts Lean pulls, heaviest first (ties by name)."""
    mass = entity_mass(seeds, paths)
    return sorted(mass, key=lambda name: (-mass[name], name))


def lean_units(
    seeds: list[str],
    info: dict[str, dict],
    paths: list[ScoredPath],
    chunks: list[dict],
    chunk_names: list[str],
) -> list[EvidenceUnit]:
    """Assemble Synapse-Lean's ranked units from its parts. Pure.

    Scores share one scale so the packer can interleave kinds sensibly:
      • seed ``i`` (0-based rank): ``w = 1 / (1 + i)`` — its description unit;
      • a kept path: ``(score / best path score) · max(w of its endpoints)``;
      • an entity's mass: its seed weight, or the best kept path through it;
      • an excerpt: ``0.9 ·`` the mass of the best-ranked name it mentions
        (``chunks_for_entities``' ``best_seed_rank`` indexes ``chunk_names``).
    Units are emitted entity → path → excerpt so equal scores keep that order.
    """
    weight = {name: 1.0 / (1.0 + i) for i, name in enumerate(seeds)}
    units: list[EvidenceUnit] = []
    for name in seeds:
        row = info.get(name)
        if row is None:
            continue
        units.append(
            EvidenceUnit(
                text=_entity_block(name, row.get("type"), row.get("description")),
                kind="entity",
                source_id=name,
                score=weight[name],
            )
        )
    for path, score in zip(paths, path_scores(seeds, paths), strict=True):
        units.append(EvidenceUnit(text=path.text, kind="path", source_id=None, score=score))
    mass = entity_mass(seeds, paths)
    for chunk in chunks:
        rank = int(chunk.get("best_seed_rank") or 0)
        anchor = chunk_names[rank] if 0 <= rank < len(chunk_names) else None
        score = LEAN_EXCERPT_WEIGHT * mass.get(anchor, 0.0) if anchor else 0.0
        unit = _prose(chunk, score)
        if unit.text:
            units.append(unit)
    return units


class SynapseLean(BaseArm):
    name = "synapse_lean"
    family = "graph"
    title = "Synapse-Lean (ours)"
    description = (
        "Ours, PathRAG-style + LiteRAG-style: hybrid seeds; relational paths between seeds "
        "inside their 2-hop neighbourhood scored by flow propagation (decay 0.7, pruning "
        "threshold) and down-weighted at hub nodes by log-degree; top-10 paths + the seeds' "
        "descriptions + the source excerpts of the entities on kept paths. 0 LLM calls."
    )
    source = {
        "citation": "After PathRAG (Chen et al., arXiv:2502.14902) and LiteRAG (Coll Tejeda "
        "et al., arXiv:2609.10239) — a re-implementation, not the authors' code",
        "url": "https://arxiv.org/abs/2502.14902",
    }
    needs_graph = True
    k_role = "seeds"

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "alpha": LEAN_ALPHA,
            "theta": LEAN_THETA,
            "top_paths": LEAN_TOP_PATHS,
            "hops": LEAN_HOPS,
            "max_path_edges": LEAN_MAX_PATH_EDGES,
            "excerpt_weight": LEAN_EXCERPT_WEIGHT,
            "hub": "1/(1+ln(1+degree)), geometric mean over intermediate nodes",
        }

    async def retrieve(self, question: str, *, k: int, seed: int) -> Evidence:
        seeds = await _seed_names(question, max(1, int(k)))
        if not seeds:
            return Evidence.empty(self.name, seeds=[])
        edges = await chat_engine._neighborhood_edges(seeds, LEAN_HOPS) if len(seeds) > 1 else []
        endpoints = [e["source"] for e in edges] + [e["target"] for e in edges]
        nodes = list(dict.fromkeys([*seeds, *endpoints]))
        rows = await execute_query(ENTITY_INFO_QUERY, {"names": nodes}) or []
        info = {str(r["name"]): r for r in rows if r.get("name")}
        seeds = [s for s in seeds if s in info]
        degrees = {name: int(r.get("degree") or 0) for name, r in info.items()}
        paths = score_paths(seeds, edges, degrees)

        # Excerpts of the entities on kept paths and of the seeds, heaviest first
        # (the seeds alone when no path survived, so Lean never loses its prose).
        chunk_names = excerpt_anchors(seeds, paths)
        chunks = list(await chunk_store.chunks_for_entities(chunk_names, max(1, int(k))) or [])
        units = lean_units(seeds, info, paths, chunks, chunk_names)
        meta = {"seeds": seeds, "edges": len(edges), "paths": len(paths), "excerpts": len(chunks)}
        return Evidence(units, self.name, meta)


# -- HippoRAG-2-style PPR ------------------------------------------------------
PPR_DAMPING = 0.5  # networkx ``alpha`` = probability of following an edge


@dataclass
class PPRGraph:
    graph: nx.Graph
    chunks: dict[str, dict]
    #: node -> index of its connected component (PPR mass never leaves it).
    component: dict[str, int]
    members: list[list[str]]


def entity_node(name: str) -> str:
    return f"e:{name}"


def chunk_node(chunk_id: str) -> str:
    return f"c:{chunk_id}"


def build_ppr_graph(
    relations: list[dict], mentions: list[dict], chunks: list[dict]
) -> PPRGraph:
    """Undirected entity + passage graph. Pure and order-independent.

    Nodes: every entity on a relation or a mention, every chunk. Edges:
    entity–entity relations (parallel relations add weight, as in
    ``communities.build_graph_from_rows``) and entity–chunk ``MENTIONED_IN``.
    """
    graph = nx.Graph()
    chunk_meta: dict[str, dict] = {}
    for row in sorted(chunks, key=lambda r: str(r.get("id"))):
        cid = str(row.get("id") or "")
        if not cid:
            continue
        chunk_meta[cid] = {
            "id": cid,
            "text": str(row.get("text") or ""),
            "document": str(row.get("document") or ""),
            "index": int(row.get("index") or 0),
        }
        graph.add_node(chunk_node(cid))
    pairs = sorted(
        (str(r.get("source")), str(r.get("target")))
        for r in relations
        if r.get("source") and r.get("target") and r.get("source") != r.get("target")
    )
    for a, b in pairs:
        u, v = entity_node(a), entity_node(b)
        if graph.has_edge(u, v):
            graph[u][v]["weight"] += 1.0
        else:
            graph.add_edge(u, v, weight=1.0)
    links = sorted(
        (str(r.get("entity")), str(r.get("chunk")))
        for r in mentions
        if r.get("entity") and r.get("chunk")
    )
    for name, cid in links:
        graph.add_edge(entity_node(name), chunk_node(cid), weight=1.0)
    members = [sorted(c) for c in nx.connected_components(graph)]
    members.sort(key=lambda m: m[0])
    component = {node: i for i, m in enumerate(members) for node in m}
    return PPRGraph(graph=graph, chunks=chunk_meta, component=component, members=members)


def personalized_pagerank(
    graph: nx.Graph, personalization: dict[str, float], *, alpha: float | None = None
) -> dict[str, float]:
    """``nx.pagerank`` with a pure-Python fallback when SciPy is not installed.

    networkx 3's ``pagerank`` needs SciPy, which is not in ``requirements.txt``;
    the fallback is networkx's own power iteration, same maths.
    """
    alpha = PPR_DAMPING if alpha is None else alpha
    try:
        return nx.pagerank(graph, alpha=alpha, personalization=personalization, weight="weight")
    except (ImportError, ModuleNotFoundError):
        from networkx.algorithms.link_analysis import pagerank_alg

        return pagerank_alg._pagerank_python(
            graph, alpha=alpha, personalization=personalization, weight="weight"
        )


def ppr_rank_chunks(
    ppr: PPRGraph, seeds: list[str], k: int, *, alpha: float | None = None
) -> list[tuple[str, float]]:
    """``[(chunk_id, mass)]`` best first for seeds weighted ``1/(1+rank)``. Pure.

    PageRank runs on the union of the seeds' connected components: with the
    reset mass only on seeds, nothing outside them can receive any, so this is
    the same ranking at a fraction of the cost (and does not depend on the size
    of unrelated parts of the graph).
    """
    weights: dict[str, float] = {}
    for i, name in enumerate(dict.fromkeys(seeds)):
        node = entity_node(name)
        if node in ppr.component:
            weights[node] = 1.0 / (1.0 + i)
    if not weights or k <= 0:
        return []
    nodes: list[str] = []
    for comp in sorted({ppr.component[n] for n in weights}):
        nodes.extend(ppr.members[comp])
    sub = ppr.graph.subgraph(nodes)
    mass = personalized_pagerank(sub, weights, alpha=alpha)
    ranked = sorted(
        ((node[2:], float(m)) for node, m in mass.items() if node.startswith("c:") and m > 0),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return ranked[: int(k)]


class PPR(BaseArm):
    name = "ppr"
    family = "graph"
    title = "HippoRAG-2-style PPR"
    description = (
        "HippoRAG-2-style PPR without the LLM filter: Personalized PageRank (damping 0.5) over "
        "an entity + passage graph, reset on the question's hybrid seed entities weighted by "
        "rank; passages ranked by PPR mass. 0 retrieval LLM calls (the paper's triple filter "
        "is not implemented)."
    )
    source = {
        "citation": "After HippoRAG 2 (Jiménez Gutiérrez et al., 'From RAG to Memory', ICML "
        "2025, arXiv:2502.14802) — a re-implementation without the LLM triple filter",
        "url": "https://arxiv.org/abs/2502.14802",
    }
    needs_graph = True
    k_role = "passages"

    #: Seed entities in the reset distribution — the shipped default seed count;
    #: ``k`` bounds the PASSAGES returned (``k_role = "passages"``).
    seed_k: int = 8

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "damping": PPR_DAMPING,
            "seed_weights": "1/(1+rank)",
            "seed_k": self.seed_k,
            "llm_filter": False,
            "restricted_to_seed_components": True,
        }

    async def retrieve(self, question: str, *, k: int, seed: int) -> Evidence:
        async def build() -> PPRGraph:
            relations = await execute_query(PPR_RELATIONS_QUERY) or []
            mentions = await execute_query(PPR_MENTIONS_QUERY) or []
            chunks = await execute_query(PPR_CHUNKS_QUERY) or []
            return build_ppr_graph(relations, mentions, chunks)

        ppr = await _cached("ppr_graph", build)
        seeds = await _seed_names(question, self.seed_k)
        ranked = ppr_rank_chunks(ppr, seeds, int(k))
        units = [_prose(ppr.chunks[cid], mass) for cid, mass in ranked if cid in ppr.chunks]
        return Evidence(
            [u for u in units if u.text],
            self.name,
            {"seeds": seeds, "nodes": ppr.graph.number_of_nodes()},
        )


# ── Registry ─────────────────────────────────────────────────────────────────
ARMS: dict[str, BaseArm] = {
    arm.name: arm
    for arm in (
        NullClosedBook(),
        NullVocabulary(),
        NullRandom(),
        BM25(),
        Dense(),
        SynapseD(),
        SynapseLean(),
        PPR(),
    )
}
NULL_ARMS: tuple[str, ...] = tuple(n for n, a in ARMS.items() if a.is_null)


def get_arm(name: str) -> BaseArm:
    try:
        return ARMS[name]
    except KeyError:
        raise KeyError(f"unknown arm {name!r} (known: {', '.join(ARMS)})") from None


def arm_catalog() -> list[dict[str, Any]]:
    """Every arm's public metadata, grouped floors → passages → graph."""
    order = list(FAMILY_TITLES)
    arms = sorted(ARMS.values(), key=lambda a: order.index(a.family))
    return [a.catalog_entry() for a in arms]
