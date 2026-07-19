"""
Project Synapse — Chat Engine (GraphRAG)

Answers questions grounded in the Neo4j knowledge graph:

  1. RETRIEVE — hybrid search finds the most relevant entities:
       • semantic (vector index over entity embeddings), and
       • lexical (full-text index over name + description),
     then expands each seed to its 1-hop neighborhood so the model sees
     relationships, not just isolated nodes.
  2. GENERATE — the configured LLM streams an answer grounded in that subgraph,
     with conversation history for follow-ups.

The engine yields *typed events* ({"type": "citations"|"token"|"done"|"error"})
so the frontend can render streamed tokens and highlight the exact entities that
grounded the answer.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator
from itertools import zip_longest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import get_settings
from app.neo4j_driver import execute_query
from app.services.graph_schema import ENTITY_FULLTEXT_INDEX, ENTITY_VECTOR_INDEX
from app.services.llm_provider import get_chat_llm, get_embeddings

logger = logging.getLogger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Synapse, an assistant that answers questions using ONLY the \
provided knowledge-graph context. The context is a set of entities and the \
relationships between them, extracted from the user's documents.

Knowledge Graph Context:
{context}

Rules:
- Ground every claim in the entities and relationships above.
- Reference specific entities by name when relevant.
- If the context lacks the answer, say so plainly — do not invent facts.
- Be concise and well-structured (use short paragraphs or bullets).""",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

_LUCENE_SPECIAL = re.compile(r'[+\-!(){}\[\]^"~*?:\\/]|&&|\|\|')
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "who", "what", "when", "where",
    "why", "how", "of", "to", "in", "on", "for", "and", "or", "about", "tell",
    "me", "his", "her", "their", "this", "that", "with", "does", "do", "did",
}


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
        citations.append({"name": record["entity"], "type": record.get("type")})
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


async def retrieve_subgraph(question: str, k: int = 8) -> tuple[str, list[dict]]:
    """Hybrid retrieval: merge vector + keyword seeds, expand, and format.

    Returns ``(context_text, citations)`` where citations is the list of seed
    entities that grounded the answer (used to highlight nodes in the UI).
    """
    # Fetch a few extra candidates per source, then keep the top-k after merge.
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

    # Citations = the top-k ranked seeds (their neighborhoods enrich the context).
    return await _expand_and_format(ordered_names[:k])


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
        context, citations = await retrieve_subgraph(question)
    except Exception as e:  # noqa: BLE001
        logger.exception("Retrieval failed: %s", e)
        yield {"type": "error", "data": "Failed to query the knowledge graph."}
        return

    # Emit citations up-front so the UI can highlight grounding nodes immediately.
    yield {"type": "citations", "data": citations}

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
