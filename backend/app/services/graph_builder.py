"""
Project Synapse — Graph Builder Service

Turns raw text chunks into a knowledge graph:

  1. Extract entities + relationships from each chunk with the configured LLM
     (Gemini / Claude / Ollama — see ``llm_provider``).
  2. Deduplicate entities by canonical name.
  3. Embed each entity (name + description) for vector retrieval.
  4. Write nodes (with embeddings) and typed relationships to Neo4j.

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

from app.config import get_settings
from app.neo4j_driver import execute_query
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
    for parsed in results:
        all_entities.extend(parsed["entities"])
        all_relationships.extend(parsed["relationships"])

    logger.info(
        "📊 Extraction complete: %s entities, %s rels",
        len(all_entities),
        len(all_relationships),
    )

    unique_entities = _dedupe_entities(all_entities)
    logger.info("🧹 Deduplicated to %s unique entities", len(unique_entities))

    await report({"type": "progress", "stage": "embedding", "total": len(unique_entities)})
    embeddings_by_name = await _embed_entities(unique_entities)

    await report({"type": "progress", "stage": "writing_nodes", "total": len(unique_entities)})
    nodes_created = await _write_entities(unique_entities, document_name, embeddings_by_name)
    logger.info("✅ Wrote %s nodes to Neo4j", nodes_created)

    await report({"type": "progress", "stage": "writing_edges", "total": len(all_relationships)})
    rels_created = await _write_relationships(all_relationships)
    logger.info("✅ Wrote %s relationships to Neo4j", rels_created)

    return {
        "nodes_created": nodes_created,
        "relationships_created": rels_created,
        "entities_extracted": len(all_entities),
        "unique_entities": len(unique_entities),
    }


async def _write_entities(
    entities: list[dict],
    document_name: str,
    embeddings_by_name: dict[str, list[float]],
) -> int:
    """MERGE each entity node, storing type, description, source doc, embedding."""
    query = """
    MERGE (n:Entity {name: $name})
    SET n.type = $type,
        n.description = $description,
        n.document = $document
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
        n.document = $document
    """
    created = 0
    for entity in entities:
        name = entity.get("name", "Unknown")
        params = {
            "name": name,
            "type": entity.get("type", "UNKNOWN"),
            "description": entity.get("description", ""),
            "document": document_name,
            "embedding": embeddings_by_name.get(name),
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
