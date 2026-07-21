# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Graph Builder Service

Turns raw text chunks into a knowledge graph:

  1. Extract entities + relationships from each chunk with the configured LLM
     (Gemini / Claude / Ollama — see ``llm_provider``).
  2. Deduplicate entities by canonical name.
  3. Embed each entity (name + description) for vector retrieval.
  4. Resolve near-duplicate entities ("Ahmed" / "Ahmed Maaloul") onto a single
     canonical node and re-point the extracted edges at it — see
     ``entity_resolution``.
  5. Write nodes (with embeddings) and typed relationships to Neo4j.
  6. Persist the source chunks themselves as ``:Chunk`` text units, linked to
     the entities extracted from them — see ``chunk_store``. Step 1 compresses
     prose into 15-word descriptions; keeping the original passages is what lets
     retrieval return the graph *and* the evidence it was built from.
  7. Resolve the freshly written entities against those earlier documents left
     in the graph, so duplicates cannot accumulate across ingests.

The extraction schema is theme-aware so a CV, a research paper, and a contract
each get domain-appropriate entity/relationship types.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable

from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings, get_settings
from app.neo4j_driver import execute_query
from app.services import chunk_store, entity_resolution
from app.services.llm_provider import get_chat_llm, get_embeddings

logger = logging.getLogger(__name__)

# ── Theme Schemas ────────────────────────────────────
THEMES: dict[str, dict[str, list[str]]] = {
    "Personal CV / Resume": {
        "entities": ["PERSON", "COMPANY", "UNIVERSITY", "ROLE", "PROJECT", "SKILL", "TOOL", "LANGUAGE", "CERTIFICATION", "LOCATION"],
        "relationships": ["WORKED_AT", "STUDIED_AT", "HELD_ROLE", "WORKED_ON", "HAS_SKILL", "USES_TOOL", "MASTER_OF_LANGUAGE"],
    },
    "Technology, Tools & Docs": {
        "entities": ["PERSON", "ORGANIZATION", "ROLE", "PROJECT", "SKILL", "TOOL", "FRAMEWORK", "DATABASE", "CERTIFICATION", "LOCATION", "EDUCATION"],
        "relationships": ["WORKED_AT", "HELD_ROLE", "WORKED_ON", "HAS_SKILL", "USES_TOOL", "STUDIED_AT", "EARNED_CERTIFICATION"],
    },
    "Medical/Scientific": {
        "entities": ["DISEASE", "SYMPTOM", "DRUG", "TREATMENT", "ANATOMY", "GENE", "RESEARCH_STUDY", "PERSON", "ORGANIZATION"],
        "relationships": ["CAUSES", "TREATS", "IS_SYMPTOM_OF", "PREVENTS", "INTERACTS_WITH", "STUDIED_BY"],
    },
    "Business/Legal": {
        "entities": ["COMPANY", "PERSON", "CONTRACT", "LAW", "FINANCIAL_METRIC", "PRODUCT", "LOCATION"],
        "relationships": ["OWNS", "PARTNERS_WITH", "REGULATES", "SUED_BY", "SELLS", "EMPLOYS"],
    },
    "Generic": {
        "entities": ["PERSON", "ORGANIZATION", "CONCEPT", "EVENT", "LOCATION", "THING"],
        "relationships": ["RELATED_TO", "PART_OF", "CAUSED", "PARTICIPATED_IN"],
    },
}

DEFAULT_THEME = "Generic"


# ── Dynamic Prompt Builder ───────────────────────────
def get_extraction_prompt(theme: str) -> ChatPromptTemplate:
    schema = THEMES.get(theme, THEMES[DEFAULT_THEME])
    entity_types = ", ".join(schema["entities"])
    rel_types = ", ".join(schema["relationships"])

    extra_instructions = ""
    if theme == "Personal CV / Resume":
        extra_instructions = """
8. CV SPECIFIC: Extract highly granular skills and technologies. Instead of generic terms like 'Data Science', extract "Python", "Pandas", "React", "Docker", "PyTorch" as individual SKILL or TOOL entities.
9. CV SPECIFIC: Always ensure the main PERSON is linked directly to their skills using HAS_SKILL or USES_TOOL.
10. CV SPECIFIC: Capture all universities and companies as UNIVERSITY and COMPANY, linked to the PERSON via STUDIED_AT and WORKED_AT.
"""

    system_prompt = f"""You are an elite knowledge graph extraction expert.
Extract all meaningful entities and relationships from this chunk representing a {{theme}} document (source file: {{document_name}}).

Return ONLY valid JSON with this exact structure (no markdown fences):
{{{{
  "entities": [
    {{{{"name": "Entity Name", "type": "ENTITY_TYPE", "description": "Brief description"}}}}
  ],
  "relationships": [
    {{{{"source": "Entity A", "target": "Entity B", "type": "RELATIONSHIP_TYPE", "description": "Brief description"}}}}
  ]
}}}}

CRITICAL RULES:
1. ONLY use these Entity Types: {entity_types}
2. ONLY use these Relationship Types: {rel_types}
3. If extracting a person from pronouns ('He', 'She', 'The candidate', 'I'), infer their full name from the document context if known.
4. CANONICAL NAMES: Standardize technology/concept names (e.g. use "AWS" instead of "Amazon Web Services", "React" instead of "React.js", "PostgreSQL" instead of "Postgres").
5. Merge duplicate entities by using strictly consistent naming.
6. Extract at least 5-8 distinct, meaningful entities per chunk if available.
7. Keep descriptions concise (under 15 words) and factual.{extra_instructions}
"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Extract entities and relationships from this text:\n\n{text}"),
        ]
    )


def _parse_llm_json(text: str) -> dict:
    """Parse JSON from an LLM response, tolerating markdown fences and prose.

    Strategy: strip code fences, then try a direct parse; if that fails, extract
    the outermost ``{...}`` block and parse that. Always returns the expected
    shape so callers never crash on malformed model output.
    """
    empty = {"entities": [], "relationships": []}
    if not text:
        return empty

    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the outermost brace-delimited object.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            logger.warning("Failed to parse LLM JSON: %s", text[:200])
            return empty
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON (brace fallback): %s", text[:200])
            return empty

    if not isinstance(parsed, dict):
        return empty
    return {
        "entities": parsed.get("entities", []) or [],
        "relationships": parsed.get("relationships", []) or [],
    }


async def _extract_with_timeout(
    chain,
    chunk: str,
    chunk_idx: int,
    timeout: int,
    theme: str,
    document_name: str,
) -> dict:
    """Extract entities from a single chunk with a hard timeout."""
    try:
        response = await asyncio.wait_for(
            chain.ainvoke(
                {"text": chunk, "theme": theme, "document_name": document_name}
            ),
            timeout=timeout,
        )
        parsed = _parse_llm_json(getattr(response, "content", "") or "")
        logger.info(
            "   Chunk %s: %s entities, %s relationships",
            chunk_idx,
            len(parsed["entities"]),
            len(parsed["relationships"]),
        )
        return parsed
    except TimeoutError:
        logger.warning("   Chunk %s: TIMEOUT after %ss, skipping", chunk_idx, timeout)
        return {"entities": [], "relationships": []}
    except Exception as e:  # noqa: BLE001 - one bad chunk must not fail the doc
        logger.warning("   Chunk %s: extraction failed: %s", chunk_idx, e)
        return {"entities": [], "relationships": []}


def _dedupe_entities(entities: list[dict]) -> list[dict]:
    """Collapse entities to one per canonical (lowercased, trimmed) name."""
    seen: dict[str, dict] = {}
    for entity in entities:
        name = str(entity.get("name", "")).strip()
        if name:
            seen[name.lower()] = entity
    return list(seen.values())


async def _embed_entities(entities: list[dict]) -> dict[str, list[float]]:
    """Embed each entity as ``name — description`` for vector retrieval.

    Returns a map of canonical name -> embedding vector. Best-effort: on any
    embedding error we log and return an empty map so ingestion still succeeds
    (retrieval then falls back to keyword search).
    """
    if not entities:
        return {}
    texts = [
        f"{e.get('name', '')} — {e.get('description', '')}".strip(" —")
        for e in entities
    ]
    try:
        embeddings = get_embeddings()
        # LangChain embeddings are sync; run off the event loop.
        vectors = await asyncio.to_thread(embeddings.embed_documents, texts)
    except Exception as e:  # noqa: BLE001
        logger.warning("⚠️ Embedding step skipped (%s); using keyword retrieval only", e)
        return {}
    return {e["name"]: vec for e, vec in zip(entities, vectors, strict=False) if e.get("name")}


def _collapse_duplicate_entities(
    entities: list[dict],
    relationships: list[dict],
    embeddings_by_name: dict[str, list[float]],
    settings: Settings,
) -> tuple[list[dict], list[dict], int, dict[str, str]]:
    """Merge near-duplicates *before* anything touches Neo4j.

    Collapsing in memory means the duplicate node is never written in the first
    place, and it lets us rewrite the extracted relationship endpoints onto the
    canonical names so every edge follows its node.

    Pure CPU, no I/O — deliberately kept synchronous and called through
    :func:`asyncio.to_thread` so its blocking/union-find sweep never stalls the
    event loop of the API server.

    Returns ``(entities, relationships, merged_count, alias_map)``; the canonical
    entity inherits a description if it had none, plus an ``aliases`` list.
    ``alias_map`` maps every absorbed name to its canonical one — relationships
    are rewritten here, but the caller needs the same map to re-point the
    per-chunk entity name lists (see :func:`_canonicalize_chunk_names`).
    """
    clusters = entity_resolution.find_duplicate_clusters(
        entities, embeddings_by_name, settings=settings
    )
    if not clusters:
        return entities, relationships, 0, {}

    by_name = {str(e.get("name", "")): e for e in entities}
    canonical_of: dict[str, str] = {}
    for cluster in clusters:
        canonical = entity_resolution.choose_canonical(cluster)
        target = by_name.get(canonical)
        if target is None:
            continue
        aliases = sorted({n for n in cluster if n != canonical})
        for dup in aliases:
            canonical_of[dup] = canonical
            inherited = str(by_name.get(dup, {}).get("description", "")).strip()
            if inherited and not str(target.get("description", "")).strip():
                target["description"] = inherited
        target["aliases"] = sorted(set(target.get("aliases") or []) | set(aliases))
        logger.info("🧬 Resolved %s -> '%s'", aliases, canonical)

    kept = [e for e in entities if str(e.get("name", "")) not in canonical_of]

    rewritten: list[dict] = []
    for rel in relationships:
        source = str(rel.get("source", "")).strip()
        target_name = str(rel.get("target", "")).strip()
        new_source = canonical_of.get(source, source)
        new_target = canonical_of.get(target_name, target_name)
        if new_source and new_source == new_target:
            continue  # the merge turned this edge into a self-loop
        rewritten.append({**rel, "source": new_source, "target": new_target})

    return kept, rewritten, len(canonical_of), canonical_of


def _canonicalize_chunk_names(
    names_per_chunk: list[list[str]],
    survivor_by_lower: dict[str, str],
    alias_map: dict[str, str],
    kept_names: set[str],
) -> list[list[str]]:
    """Re-point per-chunk entity names onto the nodes that were actually written.

    A name travels through two collapses between extraction and Neo4j:
    :func:`_dedupe_entities` folds case/whitespace variants together, then entity
    resolution merges near-duplicates onto a canonical node. A chunk link built
    from the *raw extracted* name would therefore point at a node that no longer
    exists ("Postgres" after it was merged into "PostgreSQL"), and the excerpt
    would be silently unreachable from the surviving entity — the exact evidence
    loss this whole feature exists to prevent.

    Names that survive neither collapse are dropped rather than linked blindly,
    so ``store_chunks`` never asks for an edge to a node we did not write.
    """
    resolved_per_chunk: list[list[str]] = []
    for names in names_per_chunk:
        resolved: list[str] = []
        for raw in names:
            name = survivor_by_lower.get(raw.lower(), raw)
            name = alias_map.get(name, name)
            if name in kept_names and name not in resolved:
                resolved.append(name)
        resolved_per_chunk.append(resolved)
    return resolved_per_chunk


async def _store_source_chunks(
    chunks: list[str],
    names_per_chunk: list[list[str]],
    document_name: str,
    report: Callable[[dict], Awaitable[None]],
) -> int:
    """Persist the source text units, streaming a progress event around the write."""
    total = len(chunks)
    await report(
        {"type": "progress", "stage": "storing_chunks", "processed": 0, "total": total}
    )
    result = await chunk_store.store_chunks(document_name, chunks, names_per_chunk)
    stored = int(result.get("chunks", 0))
    await report(
        {"type": "progress", "stage": "storing_chunks", "processed": stored, "total": total}
    )
    logger.info("🧾 Stored %s source chunks (%s entity links)", stored, result.get("links", 0))
    return stored


async def _resolve_against_graph(
    fresh_entities: list[dict],
    embeddings_by_name: dict[str, list[float]],
    settings: Settings,
) -> int:
    """Merge the just-written entities into equivalents from earlier documents.

    The in-memory pass only sees this document; this pass compares the fresh
    entities against everything already in Neo4j, so "Postgres" from document 1
    and "PostgreSQL" from document 2 still end up as one node. Only clusters that
    involve at least one fresh entity are merged — the rest of the graph is left
    exactly as previous ingests decided.
    """
    graph_entities, graph_embeddings = await entity_resolution.fetch_graph_entities()
    if not graph_entities:
        return 0

    fresh_names = {str(e.get("name", "")) for e in fresh_entities if str(e.get("name", "")).strip()}
    # Fresh embeddings win: the node's stored vector may be missing when the
    # Neo4j build lacks the vector procedures.
    combined_embeddings = {**graph_embeddings, **embeddings_by_name}
    combined_entities = {str(e.get("name", "")): e for e in graph_entities}
    for entity in fresh_entities:
        name = str(entity.get("name", ""))
        if name.strip():
            combined_entities[name] = entity

    # Clustering is CPU-bound (blocking + union-find over every candidate pair)
    # and scales with the whole graph, not just this document. Running it inline
    # would freeze the event loop — and with it every other request the API is
    # serving — so it is offloaded to a worker thread.
    clusters = await asyncio.to_thread(
        entity_resolution.find_duplicate_clusters,
        list(combined_entities.values()),
        combined_embeddings,
        settings=settings,
    )
    clusters = [c for c in clusters if fresh_names.intersection(c)]
    if not clusters:
        return 0

    result = await entity_resolution.merge_entity_clusters(clusters)
    return int(result.get("merged", 0))


ProgressCallback = Callable[[dict], Awaitable[None]]


async def build_knowledge_graph(
    chunks: list[str],
    document_name: str,
    theme: str = DEFAULT_THEME,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Extract a knowledge graph from text chunks and persist it to Neo4j.

    If ``on_progress`` is provided it's awaited with structured progress events
    ({"type": "progress", "stage": ..., "processed": i, "total": n}) so callers
    can stream live status to the UI.
    """
    settings = get_settings()

    async def report(event: dict) -> None:
        if on_progress is not None:
            await on_progress(event)

    llm = get_chat_llm(json_mode=True, temperature=settings.extraction_temperature)
    chain = get_extraction_prompt(theme) | llm

    total = len(chunks)
    semaphore = asyncio.Semaphore(settings.extraction_concurrency)
    completed = 0
    completed_lock = asyncio.Lock()

    async def process_chunk(idx: int, c: str) -> dict:
        nonlocal completed
        async with semaphore:
            logger.info("🔍 Processing chunk %s/%s (%s chars)...", idx, total, len(c))
            parsed = await _extract_with_timeout(
                chain,
                c,
                idx,
                timeout=settings.extraction_timeout,
                theme=theme,
                document_name=document_name,
            )
        async with completed_lock:
            completed += 1
            await report(
                {
                    "type": "progress",
                    "stage": "extracting",
                    "processed": completed,
                    "total": total,
                    "entities_so_far": len(parsed["entities"]),
                }
            )
        return parsed

    results = await asyncio.gather(
        *(process_chunk(i + 1, chunk) for i, chunk in enumerate(chunks))
    )

    all_entities: list[dict] = []
    all_relationships: list[dict] = []
    # ``asyncio.gather`` preserves argument order, so ``results[i]`` is chunk i —
    # which is what lets us remember *which chunk each entity came from* and
    # later wire (:Entity)-[:MENTIONED_IN]->(:Chunk).
    names_per_chunk: list[list[str]] = []
    for parsed in results:
        all_entities.extend(parsed["entities"])
        all_relationships.extend(parsed["relationships"])
        names_per_chunk.append(
            [
                str(e.get("name", "")).strip()
                for e in parsed["entities"]
                if str(e.get("name", "")).strip()
            ]
        )

    logger.info(
        "📊 Extraction complete: %s entities, %s rels",
        len(all_entities),
        len(all_relationships),
    )

    unique_entities = _dedupe_entities(all_entities)
    logger.info("🧹 Deduplicated to %s unique entities", len(unique_entities))

    # Snapshot the case-folding collapse *before* resolution rebinds the list, so
    # a chunk that said "python" can still find the "Python" node that survived.
    survivor_by_lower = {
        str(e.get("name", "")).strip().lower(): str(e.get("name", "")).strip()
        for e in unique_entities
    }

    await report({"type": "progress", "stage": "embedding", "total": len(unique_entities)})
    embeddings_by_name = await _embed_entities(unique_entities)

    entities_merged = 0
    alias_map: dict[str, str] = {}
    if settings.entity_resolution_enabled:
        await report(
            {
                "type": "progress",
                "stage": "resolving_entities",
                "processed": 0,
                "total": len(unique_entities),
                "merged": 0,
            }
        )
        # CPU-bound and synchronous by design — see _collapse_duplicate_entities.
        unique_entities, all_relationships, entities_merged, alias_map = await asyncio.to_thread(
            _collapse_duplicate_entities,
            unique_entities,
            all_relationships,
            embeddings_by_name,
            settings,
        )
        logger.info("🧬 Resolved %s duplicate entities within this document", entities_merged)

    await report({"type": "progress", "stage": "writing_nodes", "total": len(unique_entities)})
    nodes_created = await _write_entities(unique_entities, document_name, embeddings_by_name)
    logger.info("✅ Wrote %s nodes to Neo4j", nodes_created)

    await report({"type": "progress", "stage": "writing_edges", "total": len(all_relationships)})
    rels_created = await _write_relationships(all_relationships)
    logger.info("✅ Wrote %s relationships to Neo4j", rels_created)

    # Chunks are stored here — after the entity nodes exist (so MENTIONED_IN has
    # something to attach to) but *before* the whole-graph resolution pass below.
    # That pass merges nodes in the database and re-points every relationship a
    # duplicate held, MENTIONED_IN included; storing first therefore lets the
    # merge carry the chunk links along. Storing afterwards would look up names
    # the merge had already deleted and silently drop those excerpts.
    chunks_stored = 0
    if settings.store_source_chunks:
        chunks_stored = await _store_source_chunks(
            chunks,
            _canonicalize_chunk_names(
                names_per_chunk,
                survivor_by_lower,
                alias_map,
                {str(e.get("name", "")).strip() for e in unique_entities},
            ),
            document_name,
            report,
        )

    if settings.entity_resolution_enabled:
        # Edges are written first so their endpoints still exist; this pass then
        # re-points them onto entities from earlier documents.
        entities_merged += await _resolve_against_graph(
            unique_entities, embeddings_by_name, settings
        )
        await report(
            {
                "type": "progress",
                "stage": "resolving_entities",
                "processed": len(unique_entities),
                "total": len(unique_entities),
                "merged": entities_merged,
            }
        )
        logger.info("🧬 Entity resolution merged %s duplicates in total", entities_merged)

    return {
        "nodes_created": nodes_created,
        "relationships_created": rels_created,
        "entities_extracted": len(all_entities),
        "unique_entities": len(unique_entities),
        "entities_merged": entities_merged,
        "chunks_stored": chunks_stored,
    }


async def _write_entities(
    entities: list[dict],
    document_name: str,
    embeddings_by_name: dict[str, list[float]],
) -> int:
    """MERGE each entity node, storing type, description, source doc, embedding.

    ``aliases`` (the names entity resolution folded into this one) are appended
    rather than replaced, so a node keeps every alias it has ever absorbed.
    """
    query = """
    MERGE (n:Entity {name: $name})
    SET n.type = $type,
        n.description = $description,
        n.document = $document,
        n.aliases = [a IN coalesce(n.aliases, []) WHERE NOT a IN $aliases] + $aliases
    WITH n
    CALL {
        WITH n
        WITH n WHERE $embedding IS NOT NULL
        CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
    }
    RETURN n.name AS name
    """
    fallback_query = """
    MERGE (n:Entity {name: $name})
    SET n.type = $type,
        n.description = $description,
        n.document = $document,
        n.aliases = [a IN coalesce(n.aliases, []) WHERE NOT a IN $aliases] + $aliases
    """
    created = 0
    for entity in entities:
        # MUST match the normalization used everywhere else (_dedupe_entities and
        # _canonicalize_chunk_names both strip). Writing " Ada " here while the
        # chunk linker looks up "Ada" creates a node no MENTIONED_IN edge can
        # ever match, silently losing that entity's source excerpts.
        name = str(entity.get("name", "Unknown")).strip() or "Unknown"
        params = {
            "name": name,
            "type": entity.get("type", "UNKNOWN"),
            "description": entity.get("description", ""),
            "document": document_name,
            "embedding": embeddings_by_name.get(name),
            "aliases": sorted(entity.get("aliases") or []),
        }
        try:
            await execute_query(query, params)
            created += 1
        except Exception:  # noqa: BLE001 - older Neo4j lacks vector procs
            try:
                await execute_query(fallback_query, params)
                created += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("⚠️ Node write failed for %s: %s", name, e)
    return created


async def _write_relationships(relationships: list[dict]) -> int:
    """Write typed relationships, falling back to :RELATED_TO if APOC is absent."""
    created = 0
    for rel in relationships:
        source = str(rel.get("source", "")).strip()
        target = str(rel.get("target", "")).strip()
        rel_type = str(rel.get("type", "RELATED_TO")).replace(" ", "_").upper()
        description = rel.get("description", "")
        if not source or not target:
            continue

        apoc_query = """
        MATCH (a:Entity {name: $source})
        MATCH (b:Entity {name: $target})
        CALL apoc.merge.relationship(a, $rel_type, {description: $description}, {}, b, {})
        YIELD rel
        RETURN rel
        """
        try:
            await execute_query(
                apoc_query,
                {"source": source, "target": target, "rel_type": rel_type, "description": description},
            )
            created += 1
        except Exception:  # noqa: BLE001 - APOC not installed / node missing
            fallback_query = """
            MATCH (a:Entity {name: $source})
            MATCH (b:Entity {name: $target})
            MERGE (a)-[r:RELATED_TO]->(b)
            SET r.type = $rel_type, r.description = $description
            """
            try:
                await execute_query(
                    fallback_query,
                    {"source": source, "target": target, "rel_type": rel_type, "description": description},
                )
                created += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("⚠️ Relationship write failed (%s -> %s): %s", source, target, e)
    return created
