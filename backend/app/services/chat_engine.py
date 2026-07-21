# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Chat Engine (GraphRAG)

Answers questions grounded in the Neo4j knowledge graph. Unlike plain vector
RAG, the engine *routes* the question and retrieves accordingly:

  0. ROUTE — :func:`classify_query` decides, deterministically and without an
     LLM call, whether the question is about specific entities ("local") or
     about the corpus as a whole ("global").
  1a. LOCAL SEARCH — hybrid retrieval finds the most relevant seed entities:
        • semantic (vector index over entity embeddings), and
        • lexical (full-text index over name + description),
      seeds are expanded up to ``retrieval_max_hops`` and the *paths* that
      actually connect the top seeds are extracted, e.g.
        Ahmed -[WORKED_AT]-> Crédit Mutuel -[USES_TOOL]-> Argo Workflows
      Those reasoning paths are what let the model answer multi-hop questions
      that no single node's neighborhood contains.
  1b. GLOBAL SEARCH — thematic questions ("what are the main themes?") are
      answered from Louvain community summaries instead of entity
      neighborhoods; citations then point at communities, not nodes.
  1c. SOURCE EVIDENCE — entity ``description`` fields are ~15-word distillations
      of the prose they came from, and benchmarking showed that losing that
      prose is what made entity-only GraphRAG lose to plain passage RAG on raw
      fact recall. So retrieval also returns the *text units* behind the answer,
      gathered from both directions and merged:
        • semantically  — vector search over the stored chunks, and
        • structurally  — the passages the selected entities were extracted from
      The structural half is the one plain vector RAG cannot do: the graph picks
      the entities (including multi-hop), and the chunks bring back their prose.
  2. GENERATE — the configured LLM streams an answer grounded in that context,
     with conversation history for follow-ups.

The engine yields *typed events* ({"type": "citations"|"paths"|"sources"
|"token"|"done"|"error"}) so the frontend can render streamed tokens, highlight
the exact entities that grounded the answer, draw the reasoning chain, and show
the source excerpts the answer was written from.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import AsyncGenerator
from itertools import zip_longest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import get_settings
from app.neo4j_driver import execute_query
from app.services import chunk_store, communities
from app.services.graph_schema import ENTITY_FULLTEXT_INDEX, ENTITY_VECTOR_INDEX
from app.services.llm_provider import get_chat_llm, get_embeddings

logger = logging.getLogger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Synapse, an assistant that answers questions using ONLY the \
provided context. That context has two complementary halves: a knowledge graph \
(entities, the relationships between them, optional community summaries and \
explicit reasoning paths) and, under "Source excerpts:", verbatim passages from \
the user's own documents.

Knowledge Graph Context:
{context}

Rules:
- Ground every claim in the context above; never invent facts.
- Prefer the source excerpts for specific details (names, numbers, dates,
  quotes, wording) — they are the original text, while entity descriptions are
  compressed summaries of it.
- Use the entities, relationships and reasoning paths for how things connect,
  following any "Reasoning paths" chain step by step.
- Reference specific entities by name when relevant.
- If the context lacks the answer, say so plainly.
- Be concise and well-structured (use short paragraphs or bullets).""",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

PATHS_HEADING = "Reasoning paths:"
COMMUNITY_HEADING = "Corpus-level themes (community summaries):"
SOURCES_HEADING = "Source excerpts:"

# Retrieval guard-rails — bound the work done per question.
EDGE_FETCH_LIMIT = 500  # edges pulled per hop
MAX_EXPANDED_NODES = 400  # nodes kept while walking the neighborhood
MAX_SEEDS_FOR_PATHS = 6  # top seeds paired up for path finding
MAX_HOPS_CAP = 4  # hard ceiling on settings.retrieval_max_hops
MAX_MEMBERS_IN_CITATION = 12  # community members echoed to the UI

# Source-evidence guard-rails.
SOURCE_SNIPPET_CHARS = 400  # per-excerpt text sent to the browser
MAX_COMMUNITY_MEMBERS_FOR_CHUNKS = 20  # members used to pull global excerpts

_LUCENE_SPECIAL = re.compile(r'[+\-!(){}\[\]^"~*?:\\/]|&&|\|\|')
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "who", "what", "when", "where",
    "why", "how", "of", "to", "in", "on", "for", "and", "or", "about", "tell",
    "me", "his", "her", "their", "this", "that", "with", "does", "do", "did",
}


# ── Query routing ────────────────────────────────────────────────────────────
# A question is "global" when it asks about the corpus as a whole rather than
# about named entities. This is intentionally rule-based: routing must be
# deterministic, instant and unit-testable — spending an LLM call (and its
# latency + non-determinism) on a two-way decision is a bad trade.
_GLOBAL_CUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        # Explicit aggregation verbs / nouns.
        r"\bsummar(?:y|ies|ise|ize|ising|izing|isation|ization)\b",
        r"\b(?:overall|overview|synopsis|gist|tl;?dr)\b",
        r"\bbig[\s-]picture\b",
        r"\bhigh[\s-]level\b",
        r"\bbird'?s[\s-]eye\b",
        r"\bat a glance\b",
        r"\bas a whole\b",
        r"\bin general\b",
        r"\bgenerally\b",
        r"\bbroadly\b",
        # Corpus-wide scoping.
        r"\bacross (?:all|the|these|every|my)\b",
        r"\b(?:throughout|among) (?:all|the|these|every|my)\b",
        r"\b(?:whole|entire) (?:corpus|graph|dataset|collection|knowledge base)\b",
        r"\bevery (?:document|doc|file|paper)s?\b",
        r"\ball (?:the )?(?:documents|docs|files|papers)\b",
        # Thematic vocabulary — the classic "global search" trigger.
        r"\b(?:themes?|topics?)\b",
        r"\b(?:main|key|central|core|common|major|recurring|overarching|dominant)\s+"
        r"(?:ideas?|concepts?|subjects?|points?|takeaways?|findings?|insights?|"
        r"patterns?|trends?)\b",
        r"\bwhat (?:is|are) (?:this|these|the) (?:document|documents|corpus|data|"
        r"graph|knowledge base|papers?|files?)\b",
    )
)

# Words that merely express "give me the global view" — stripping them tells us
# whether anything *specific* is left to search for.
_CUE_WORDS = {
    "main", "key", "central", "core", "common", "major", "recurring", "dominant",
    "overarching", "theme", "themes", "topic", "topics", "idea", "ideas",
    "concept", "concepts", "subject", "subjects", "point", "points", "takeaway",
    "takeaways", "finding", "findings", "insight", "insights", "pattern",
    "patterns", "trend", "trends", "overall", "overview", "synopsis", "gist",
    "summary", "summaries", "summarize", "summarise", "summarizing",
    "summarising", "general", "generally", "broadly", "big", "picture", "high",
    "level", "glance", "whole", "entire", "everything", "across", "throughout",
    "among", "all", "every", "document", "documents", "doc", "docs", "file",
    "files", "paper", "papers", "corpus", "dataset", "data", "collection",
    "knowledge", "base", "graph", "covered", "cover", "covers", "discussed",
    "discuss", "contain", "contains", "content", "contents", "list", "give",
    "provide", "show", "brief", "briefly", "please", "important", "most",
    "these", "there",
}


def classify_query(question: str) -> str:
    """Route a question to ``"local"`` (entity search) or ``"global"`` (themes).

    Deterministic and LLM-free. Returns ``"local"`` whenever routing is disabled
    via ``settings.query_routing_enabled`` — local search is the safe default
    because it always has entities to fall back on.
    """
    if not get_settings().query_routing_enabled:
        return "local"
    text = (question or "").lower()
    if any(pattern.search(text) for pattern in _GLOBAL_CUE_PATTERNS):
        return "global"
    return "local"


def _is_broad_question(question: str) -> bool:
    """True when a global question carries no specific subject of its own.

    "What are the main themes?" is broad (list the biggest communities);
    "Summarize what Ahmed worked on" is focused (vector-search the summaries).
    """
    terms = _keyword_query(question).split(" OR ")
    return not [t for t in terms if t and t not in _CUE_WORDS]


# ── Seeding (hybrid vector + full-text) ──────────────────────────────────────
def _keyword_query(question: str) -> str:
    """Turn a natural-language question into a safe Lucene OR-query."""
    cleaned = _LUCENE_SPECIAL.sub(" ", question.lower())
    terms = [t for t in cleaned.split() if len(t) > 2 and t not in _STOPWORDS]
    return " OR ".join(dict.fromkeys(terms))  # dedupe, preserve order


async def _seeds_by_vector(question: str, k: int) -> list[dict]:
    """Semantic seed entities via the vector index (best-effort)."""
    try:
        vector = await _embed_query(question)
    except Exception as e:  # noqa: BLE001
        logger.warning("Vector seed skipped (embedding failed): %s", e)
        return []
    query = """
    CALL db.index.vector.queryNodes($index, $k, $vec)
    YIELD node, score
    RETURN node.name AS name, node.type AS type,
           node.description AS description, score
    """
    try:
        return await execute_query(
            query, {"index": ENTITY_VECTOR_INDEX, "k": k, "vec": vector}
        )
    except Exception as e:  # noqa: BLE001 - no vector index / no embeddings yet
        logger.info("Vector search unavailable (%s); relying on keyword search", e)
        return []


async def _seeds_by_keyword(question: str, k: int) -> list[dict]:
    """Lexical seed entities via the full-text index (best-effort)."""
    q = _keyword_query(question)
    if not q:
        return []
    query = """
    CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
    RETURN node.name AS name, node.type AS type,
           node.description AS description, score
    LIMIT $k
    """
    try:
        return await execute_query(
            query, {"index": ENTITY_FULLTEXT_INDEX, "q": q, "k": k}
        )
    except Exception as e:  # noqa: BLE001 - index may not exist yet
        logger.info("Full-text search unavailable (%s); trying CONTAINS", e)
        # Last-resort substring match on individual keywords.
        terms = q.split(" OR ")
        contains = """
        MATCH (n:Entity)
        WHERE any(t IN $terms WHERE toLower(n.name) CONTAINS t
                              OR toLower(n.description) CONTAINS t)
        RETURN n.name AS name, n.type AS type, n.description AS description, 0.0 AS score
        LIMIT $k
        """
        try:
            return await execute_query(contains, {"terms": terms, "k": k})
        except Exception:  # noqa: BLE001
            return []


async def _embed_query(question: str) -> list[float]:
    import asyncio

    embeddings = get_embeddings()
    return await asyncio.to_thread(embeddings.embed_query, question)


async def _rank_seed_names(question: str, k: int) -> list[str]:
    """Merge vector + keyword seeds into one ranked, deduped list of names."""
    vector_seeds = await _seeds_by_vector(question, k)
    keyword_seeds = await _seeds_by_keyword(question, k)

    # Interleave vector (semantic) and keyword (lexical) seeds so a strong lexical
    # match isn't buried behind many weaker semantic ones, then dedupe by name.
    ordered_names: list[str] = []
    for a, b in zip_longest(vector_seeds, keyword_seeds):
        for seed in (a, b):
            name = seed.get("name") if seed else None
            if name and name not in ordered_names:
                ordered_names.append(name)

    if not ordered_names:
        # Nothing matched — surface the most connected entities as a general view.
        fallback = """
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        RETURN n.name AS name ORDER BY degree DESC LIMIT $k
        """
        rows = await execute_query(fallback, {"k": k})
        ordered_names = [r["name"] for r in rows]
    return ordered_names


async def _expand_and_format(seed_names: list[str]) -> tuple[str, list[dict]]:
    """Expand seeds to their neighborhood and format context + citations."""
    if not seed_names:
        return "No relevant information found in the knowledge graph.", []

    query = """
    MATCH (n:Entity) WHERE n.name IN $names
    OPTIONAL MATCH (n)-[r]-(m:Entity)
    RETURN n.name AS entity, n.type AS type, n.description AS description,
           collect(DISTINCT {
               related_entity: m.name,
               related_type: m.type,
               relationship: coalesce(r.type, type(r))
           }) AS connections
    """
    results = await execute_query(query, {"names": seed_names})
    by_name = {r["entity"]: r for r in results}

    # Preserve the *ranked* seed order (vector similarity first) — Neo4j's
    # `WHERE n.name IN $names` returns rows in arbitrary order, which would
    # otherwise destroy the ranking that makes the top citation the best one.
    context_parts: list[str] = []
    citations: list[dict] = []
    for name in seed_names:
        record = by_name.get(name)
        if record is None:
            continue
        citations.append(
            {"name": record["entity"], "type": record.get("type"), "kind": "entity"}
        )
        info = f"Entity: {record['entity']} (Type: {record.get('type')})"
        if record.get("description"):
            info += f"\n  Description: {record['description']}"
        lines = [
            f"  → {c['relationship']} → {c['related_entity']} ({c['related_type']})"
            for c in record.get("connections", [])
            if c.get("related_entity")
        ]
        if lines:
            info += "\n  Relationships:\n" + "\n".join(lines)
        context_parts.append(info)

    context = "\n\n".join(context_parts)
    return context, citations


# ── Multi-hop expansion + reasoning paths ────────────────────────────────────
NEIGHBOR_EDGES_QUERY = """
MATCH (n:Entity)-[r]-(m:Entity)
WHERE n.name IN $names
RETURN startNode(r).name AS source, endNode(r).name AS target,
       coalesce(r.type, type(r)) AS rel
LIMIT $limit
"""


async def _neighborhood_edges(seed_names: list[str], hops: int) -> list[dict]:
    """Breadth-first expansion of the seeds, returning the edges discovered.

    One query per hop (the frontier is the previous hop's *new* nodes), so a
    2-hop expansion costs two round-trips regardless of graph size. Failures are
    non-fatal: reasoning paths are an enrichment, never a prerequisite.
    """
    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()
    visited: set[str] = set(seed_names)
    frontier: list[str] = list(seed_names)

    for _ in range(max(1, hops)):
        if not frontier:
            break
        try:
            rows = await execute_query(
                NEIGHBOR_EDGES_QUERY, {"names": frontier, "limit": EDGE_FETCH_LIMIT}
            )
        except Exception as e:  # noqa: BLE001 - graph may be empty / unreachable
            logger.info("Neighborhood expansion unavailable (%s)", e)
            break

        next_frontier: list[str] = []
        for row in rows or []:
            source, target, rel = row.get("source"), row.get("target"), row.get("rel")
            if not source or not target:
                continue
            key = (source, target, rel or "RELATED_TO")
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({"source": source, "target": target, "rel": key[2]})
            for node in (source, target):
                if node not in visited and len(visited) < MAX_EXPANDED_NODES:
                    visited.add(node)
                    next_frontier.append(node)
        frontier = next_frontier
    return edges


def _adjacency(edges: list[dict]) -> dict[str, list[tuple[str, str, bool]]]:
    """Undirected adjacency ``node -> [(neighbor, rel, is_outgoing)]``."""
    adj: dict[str, list[tuple[str, str, bool]]] = {}
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        rel = edge.get("rel") or "RELATED_TO"
        if not source or not target or source == target:
            continue
        adj.setdefault(source, []).append((target, rel, True))
        adj.setdefault(target, []).append((source, rel, False))
    return adj


def _shortest_path(
    adj: dict[str, list[tuple[str, str, bool]]],
    start: str,
    goal: str,
    max_len: int,
) -> tuple[list[str], list[str], list[bool]] | None:
    """BFS shortest path from ``start`` to ``goal``.

    Returns ``(nodes, rels, directions)`` or ``None``. BFS (not Dijkstra) is the
    right tool here: every edge costs one hop, and shorter chains are both more
    likely to be true and easier for the model to verbalize.
    """
    if start == goal or start not in adj or goal not in adj:
        return None
    # (node, nodes_so_far, rels_so_far, dirs_so_far)
    queue: deque[tuple[str, list[str], list[str], list[bool]]] = deque(
        [(start, [start], [], [])]
    )
    seen = {start}
    while queue:
        node, nodes, rels, dirs = queue.popleft()
        if len(rels) >= max_len:
            continue
        for neighbor, rel, outgoing in adj.get(node, []):
            if neighbor in seen:
                continue
            path = (nodes + [neighbor], rels + [rel], dirs + [outgoing])
            if neighbor == goal:
                return path
            seen.add(neighbor)
            queue.append((neighbor, *path))
    return None


def _render_path(nodes: list[str], rels: list[str], dirs: list[bool]) -> str:
    """Render a path as ``A -[REL]-> B <-[REL]- C``."""
    text = nodes[0]
    for i, rel in enumerate(rels):
        arrow = f" -[{rel}]-> " if dirs[i] else f" <-[{rel}]- "
        text += f"{arrow}{nodes[i + 1]}"
    return text


def build_reasoning_paths(
    seed_names: list[str],
    edges: list[dict],
    settings=None,
) -> list[dict]:
    """Extract the chains that connect the question's top seed entities.

    Pure and synchronous, so it is trivially testable. Candidates are the
    shortest path between every pair of ranked seeds; they are then ordered by
    (1) length — shorter chains are stronger evidence — and (2) the combined
    rank of the two endpoints, so paths joining the two *best* seeds win. At
    most ``settings.max_reasoning_paths`` are returned.
    """
    settings = settings or get_settings()
    limit = max(0, int(settings.max_reasoning_paths))
    hops = max(1, min(int(settings.retrieval_max_hops), MAX_HOPS_CAP))
    max_len = max(2, hops * 2)

    unique_seeds = list(dict.fromkeys(n for n in seed_names if n))
    if limit == 0 or len(unique_seeds) < 2 or not edges:
        return []

    adj = _adjacency(edges)
    ranked = [name for name in unique_seeds if name in adj][:MAX_SEEDS_FOR_PATHS]

    scored: list[tuple[int, int, str, dict]] = []
    for i, source in enumerate(ranked):
        for j, target in enumerate(ranked[i + 1 :], start=i + 1):
            found = _shortest_path(adj, source, target, max_len)
            if found is None:
                continue
            nodes, rels, dirs = found
            text = _render_path(nodes, rels, dirs)
            # ``dirs`` travels with the path: the adjacency is undirected, so a
            # hop may be traversed against the edge's direction. Without it the
            # UI would draw every hop as a forward arrow and misstate the graph.
            scored.append(
                (
                    len(rels),
                    i + j,
                    text,
                    {"nodes": nodes, "rels": rels, "dirs": dirs, "text": text},
                )
            )

    # (hops, rank-sum, text) — the text keeps ties deterministic.
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return [path for *_rest, path in scored[:limit]]


# ── Source evidence (text units) ─────────────────────────────────────────────
async def _semantic_excerpts(question: str, k: int) -> list[dict]:
    """Chunks that match the question itself (best-effort)."""
    try:
        return list(await chunk_store.search_chunks(question, k) or [])
    except Exception as e:  # noqa: BLE001 - evidence is an enrichment, never a gate
        logger.info("Semantic excerpt lookup failed (%s); continuing graph-only", e)
        return []


async def _structural_excerpts(names: list[str], limit: int) -> list[dict]:
    """Chunks the given entities were extracted from (best-effort)."""
    if not names:
        return []
    try:
        return list(await chunk_store.chunks_for_entities(names, limit) or [])
    except Exception as e:  # noqa: BLE001
        logger.info("Structural excerpt lookup failed (%s); continuing graph-only", e)
        return []


async def _gather_excerpts(
    question: str,
    entity_names: list[str],
    settings,
) -> list[dict]:
    """Merge the semantic and structural excerpt channels, deduped by chunk id.

    The two channels answer different questions — "what text looks like the
    query?" and "what text produced the entities the graph chose?" — so they are
    *interleaved* rather than concatenated: the character budget applied later
    must not starve one channel of the other, and the structural half leads
    because it is the one plain vector RAG cannot reproduce.
    """
    if not settings.chunk_retrieval_enabled:
        return []
    k = max(0, int(settings.chunk_top_k))
    if k <= 0:
        return []

    structural = await _structural_excerpts(list(entity_names or []), k)
    semantic = await _semantic_excerpts(question, k)

    merged: list[dict] = []
    seen: set[str] = set()
    for a, b in zip_longest(structural, semantic):
        for chunk in (a, b):
            if not chunk:
                continue
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            # Fall back to the text itself when a row carries no id, so a
            # missing id can never smuggle the same passage in twice.
            key = str(chunk.get("id") or "") or text
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "id": str(chunk.get("id") or ""),
                    "text": text,
                    "document": str(chunk.get("document") or ""),
                    "index": chunk.get("index", 0),
                }
            )
    return merged


def _format_excerpts(chunks: list[dict], max_chars: int) -> tuple[str, list[dict]]:
    """Render excerpts under a delimited heading within a character budget.

    Returns ``(section_text, excerpts_actually_included)``. Truncation happens at
    a **whole-excerpt boundary** — half a passage is exactly the kind of thing a
    model reads confidently and wrongly — and the number dropped is stated so
    the model knows the evidence in front of it is partial.

    The budget covers the excerpt blocks themselves; the heading and the
    "omitted" note are a fixed, negligible overhead on top.
    """
    budget = max(0, int(max_chars))
    blocks: list[str] = []
    kept: list[dict] = []
    used = 0

    for position, chunk in enumerate(chunks, start=1):
        source = chunk.get("document") or "unknown source"
        block = f"[S{position}] {source} (chunk {chunk.get('index', 0)})\n{chunk['text']}"
        if used + len(block) > budget:
            break  # ordered by priority — stop, don't cherry-pick smaller ones
        used += len(block)
        blocks.append(block)
        kept.append(chunk)

    if not blocks:
        return "", []

    section = f"{SOURCES_HEADING}\n\n" + "\n\n".join(blocks)
    omitted = len(chunks) - len(kept)
    if omitted > 0:
        plural = "s" if omitted != 1 else ""
        section += f"\n\n({omitted} further excerpt{plural} omitted for length.)"
    return section, kept


async def _attach_source_evidence(
    context: str,
    question: str,
    entity_names: list[str],
    settings,
) -> tuple[str, list[dict]]:
    """Append the source-excerpt section to ``context``; never raise.

    A chunk failure must degrade to exactly today's answer, so every failure
    mode — disabled flag, missing index, cold database, dead embedder — returns
    the context untouched instead of breaking the chat.
    """
    try:
        chunks = await _gather_excerpts(question, entity_names, settings)
        if not chunks:
            return context, []
        section, kept = _format_excerpts(chunks, settings.chunk_context_max_chars)
    except Exception as e:  # noqa: BLE001
        logger.info("Source excerpts unavailable (%s); answering from the graph alone", e)
        return context, []

    if not section:
        return context, []
    return f"{context}\n\n{section}", kept


def _sources_payload(chunks: list[dict]) -> list[dict]:
    """Shrink excerpts for transport — the UI shows provenance, not the corpus."""
    payload: list[dict] = []
    for chunk in chunks:
        text = chunk.get("text") or ""
        if len(text) > SOURCE_SNIPPET_CHARS:
            text = text[: SOURCE_SNIPPET_CHARS - 1].rstrip() + "…"
        payload.append(
            {
                "id": chunk.get("id", ""),
                "document": chunk.get("document", ""),
                "index": chunk.get("index", 0),
                "text": text,
            }
        )
    return payload


# ── Retrieval result ─────────────────────────────────────────────────────────
class Retrieval(tuple):
    """Retrieval result: ``(context, citations)`` plus GraphRAG extras.

    It is a genuine 2-tuple so every existing caller (`context, citations =
    await retrieve_subgraph(...)`, the eval harness, the integration tests)
    keeps working unchanged, while the streaming layer can also read
    ``.paths``, ``.sources`` and ``.mode``.
    """

    def __new__(
        cls,
        context: str,
        citations: list[dict],
        paths: list[dict] | None = None,
        mode: str = "local",
        sources: list[dict] | None = None,
    ) -> Retrieval:
        self = super().__new__(cls, (context, citations))
        self.paths = list(paths or [])
        self.mode = mode
        self.sources = list(sources or [])
        return self

    @property
    def context(self) -> str:
        return self[0]

    @property
    def citations(self) -> list[dict]:
        return self[1]


# ── Global search (community summaries) ──────────────────────────────────────
def _format_communities(summaries: list[dict]) -> tuple[str, list[dict]]:
    """Format community summaries as LLM context + community citations."""
    context_parts: list[str] = []
    citations: list[dict] = []
    for community in summaries:
        title = community.get("title") or community.get("id") or "Untitled community"
        members = [m for m in (community.get("members") or []) if m]
        size = community.get("size") or len(members)
        citations.append(
            {
                "name": title,
                "type": "Community",
                "kind": "community",
                "id": community.get("id"),
                "size": size,
                "members": members[:MAX_MEMBERS_IN_CITATION],
            }
        )
        info = f"Community: {title} ({size} entities)"
        if community.get("summary"):
            info += f"\n  Summary: {community['summary']}"
        if members:
            shown = ", ".join(members[:MAX_MEMBERS_IN_CITATION])
            more = len(members) - MAX_MEMBERS_IN_CITATION
            info += f"\n  Key members: {shown}" + (f" (+{more} more)" if more > 0 else "")
        context_parts.append(info)

    context = f"{COMMUNITY_HEADING}\n\n" + "\n\n".join(context_parts)
    return context, citations


def _community_member_names(summaries: list[dict]) -> list[str]:
    """Member entities of the top communities, deduped and capped.

    Summaries arrive ranked, so walking them in order keeps the excerpt lookup
    anchored on the communities that actually drive the answer.
    """
    names: list[str] = []
    for community in summaries:
        for member in community.get("members") or []:
            if member and member not in names:
                names.append(member)
        if len(names) >= MAX_COMMUNITY_MEMBERS_FOR_CHUNKS:
            break
    return names[:MAX_COMMUNITY_MEMBERS_FOR_CHUNKS]


async def _global_search(question: str, k: int, settings=None) -> Retrieval | None:
    """Answer corpus-level questions from community summaries.

    Returns ``None`` when no communities exist (nothing has been clustered yet),
    which makes the caller fall back to local search rather than answer blind.
    """
    settings = settings or get_settings()
    try:
        if _is_broad_question(question):
            summaries = await communities.get_community_summaries(limit=k)
        else:
            summaries = await communities.search_communities(question, k=k)
    except Exception as e:  # noqa: BLE001 - communities are optional
        logger.info("Global search unavailable (%s); falling back to local", e)
        return None

    if not summaries:
        return None
    context, citations = _format_communities(summaries)
    # A community summary is a summary of summaries — the furthest the pipeline
    # ever gets from the prose. Attaching the members' source text is what keeps
    # a thematic answer able to quote a real sentence.
    context, sources = await _attach_source_evidence(
        context, question, _community_member_names(summaries), settings
    )
    return Retrieval(context, citations, [], "global", sources)


# ── Retrieval entry point ────────────────────────────────────────────────────
async def _local_search(question: str, k: int, settings) -> Retrieval:
    """Hybrid seeding + multi-hop expansion + reasoning-path extraction."""
    seeds = (await _rank_seed_names(question, k))[:k]
    context, citations = await _expand_and_format(seeds)

    # Paths are anchored on the entities that actually exist in the graph.
    anchors = [c["name"] for c in citations] or seeds
    hops = max(1, min(int(settings.retrieval_max_hops), MAX_HOPS_CAP))
    paths: list[dict] = []
    if len(anchors) > 1:
        edges = await _neighborhood_edges(anchors, hops)
        paths = build_reasoning_paths(anchors, edges, settings)
    if paths:
        rendered = "\n".join(f"  - {p['text']}" for p in paths)
        context = f"{context}\n\n{PATHS_HEADING}\n{rendered}"

    # Source evidence comes last: graph facts frame the answer, the verbatim
    # prose backs it up, and the model reads the two as distinct sections.
    context, sources = await _attach_source_evidence(context, question, anchors, settings)
    return Retrieval(context, citations, paths, "local", sources)


async def retrieve_subgraph(question: str, k: int = 8) -> Retrieval:
    """Route the question, then retrieve the context that answers it.

    Returns ``(context_text, citations)`` — citations being the seed entities
    (or communities, for a global question) that grounded the answer, in ranked
    order, used to highlight nodes in the UI. The returned object additionally
    exposes ``.paths`` (reasoning chains), ``.sources`` (the source excerpts
    injected into the context) and ``.mode`` ("local" | "global").
    """
    settings = get_settings()
    if classify_query(question) == "global":
        result = await _global_search(question, k, settings)
        if result is not None:
            return result
        logger.info("No communities available; answering '%s' locally", question[:60])
    return await _local_search(question, k, settings)


# ── Generation ───────────────────────────────────────────────────────────────
def _to_lc_history(history: list[dict] | None, max_turns: int = 6):
    """Convert [{role, content}] into LangChain messages.

    Empty turns are dropped first, then the most recent ``max_turns`` are kept so
    the model always gets whole, recent turns rather than a padded window.
    """
    messages = []
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages[-max_turns:]


async def generate_rag_response(
    question: str,
    history: list[dict] | None = None,
) -> AsyncGenerator[dict, None]:
    """GraphRAG pipeline yielding typed events for streaming to the client."""
    settings = get_settings()

    try:
        retrieval = await retrieve_subgraph(question)
        context, citations = retrieval
        paths = list(getattr(retrieval, "paths", []) or [])
        sources = list(getattr(retrieval, "sources", []) or [])
    except Exception as e:  # noqa: BLE001
        logger.exception("Retrieval failed: %s", e)
        yield {"type": "error", "data": "Failed to query the knowledge graph."}
        return

    # Emit citations up-front so the UI can highlight grounding nodes immediately,
    # then the reasoning chain, then the source excerpts that back the answer,
    # then the answer itself.
    yield {"type": "citations", "data": citations}
    if paths:
        yield {"type": "paths", "data": paths}
    if sources:
        yield {"type": "sources", "data": _sources_payload(sources)}

    try:
        llm = get_chat_llm(streaming=True, temperature=settings.chat_temperature)
        chain = RAG_PROMPT | llm
        async for chunk in chain.astream(
            {
                "context": context or "No relevant information found.",
                "question": question,
                "history": _to_lc_history(history),
            }
        ):
            token = getattr(chunk, "content", "")
            if token:
                yield {"type": "token", "data": token}
    except Exception as e:  # noqa: BLE001
        logger.exception("Generation failed: %s", e)
        yield {"type": "error", "data": f"Generation failed: {e}"}
        return

    yield {"type": "done"}
