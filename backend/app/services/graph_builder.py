"""
Project Synapse — Graph Builder Service
Uses LangChain + Gemini to extract entities/relationships,
then writes them to Neo4j via Cypher.
"""

import asyncio
import json
import logging
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.config import get_settings
from app.neo4j_driver import execute_query

logger = logging.getLogger(__name__)

# ── Theme Schemas ────────────────────────────────────
THEMES = {
    "Personal CV / Resume": {
        "entities": ["PERSON", "COMPANY", "UNIVERSITY", "ROLE", "PROJECT", "SKILL", "TOOL", "LANGUAGE", "CERTIFICATION", "LOCATION"],
        "relationships": ["WORKED_AT", "STUDIED_AT", "HELD_ROLE", "WORKED_ON", "HAS_SKILL", "USES_TOOL", "MASTER_OF_LANGUAGE"]
    },
    "Technology, Tools & Docs": {
        "entities": ["PERSON", "ORGANIZATION", "ROLE", "PROJECT", "SKILL", "TOOL", "FRAMEWORK", "DATABASE", "CERTIFICATION", "LOCATION", "EDUCATION"],
        "relationships": ["WORKED_AT", "HELD_ROLE", "WORKED_ON", "HAS_SKILL", "USES_TOOL", "STUDIED_AT", "EARNED_CERTIFICATION"]
    },
    "Medical/Scientific": {
        "entities": ["DISEASE", "SYMPTOM", "DRUG", "TREATMENT", "ANATOMY", "GENE", "RESEARCH_STUDY", "PERSON", "ORGANIZATION"],
        "relationships": ["CAUSES", "TREATS", "IS_SYMPTOM_OF", "PREVENTS", "INTERACTS_WITH", "STUDIED_BY"]
    },
    "Business/Legal": {
        "entities": ["COMPANY", "PERSON", "CONTRACT", "LAW", "FINANCIAL_METRIC", "PRODUCT", "LOCATION"],
        "relationships": ["OWNS", "PARTNERS_WITH", "REGULATES", "SUED_BY", "SELLS", "EMPLOYS"]
    },
    "Generic": {
        "entities": ["PERSON", "ORGANIZATION", "CONCEPT", "EVENT", "LOCATION", "THING"],
        "relationships": ["RELATED_TO", "PART_OF", "CAUSED", "PARTICIPATED_IN"]
    }
}

# ── Dynamic Prompt Builder ───────────────────────────
def get_extraction_prompt(theme: str) -> ChatPromptTemplate:
    schema = THEMES.get(theme, THEMES["Generic"])
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
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extract entities and relationships from this text:\n\n{text}")
    ])


def _parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM output, handling markdown fences."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM JSON: {text[:200]}")
        return {"entities": [], "relationships": []}


async def _extract_with_timeout(chain, chunk: str, chunk_idx: int, timeout: int = 60, theme: str = "Generic", document_name: str = "") -> dict:
    """Extract entities from a single chunk with a timeout."""
    try:
        response = await asyncio.wait_for(
            chain.ainvoke({
                "text": chunk, 
                "theme": theme, 
                "document_name": document_name
            }),
            timeout=timeout,
        )
        parsed = _parse_llm_json(response.content)
        entities = parsed.get("entities", [])
        rels = parsed.get("relationships", [])
        logger.info(f"   Chunk {chunk_idx}: {len(entities)} entities, {len(rels)} relationships")
        return parsed
    except asyncio.TimeoutError:
        logger.warning(f"   Chunk {chunk_idx}: TIMEOUT after {timeout}s, skipping")
        return {"entities": [], "relationships": []}
    except Exception as e:
        logger.warning(f"   Chunk {chunk_idx}: extraction failed: {e}")
        return {"entities": [], "relationships": []}


async def build_knowledge_graph(chunks: list[str], document_name: str, theme: str = "Generic") -> dict:
    """
    Process text chunks through Gemini for entity extraction,
    then write the resulting nodes and edges to Neo4j.
    """
    settings = get_settings()

    llm = ChatOllama(
        model="gemini-3-flash-preview",
        base_url="http://host.docker.internal:11434",
        temperature=0.1,
        format="json",
    )

    chain = get_extraction_prompt(theme) | llm

    all_entities: list[dict] = []
    all_relationships: list[dict] = []

    # Extract from each chunk using concurrency to prevent massive total turnaround times
    total = len(chunks)
    semaphore = asyncio.Semaphore(5)  # Allow up to 5 concurrent Ollama connections

    async def process_chunk(idx: int, c: str):
        async with semaphore:
            logger.info(f"🔍 Processing chunk {idx}/{total} ({len(c)} chars)...")
            return await _extract_with_timeout(
                chain, 
                c, 
                idx, 
                timeout=180,  # Increased to 3 minutes per chunk for local models 
                theme=theme, 
                document_name=document_name
            )

    tasks = [process_chunk(i + 1, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    for parsed in results:
        all_entities.extend(parsed.get("entities", []))
        all_relationships.extend(parsed.get("relationships", []))

    logger.info(f"📊 Extraction complete: {len(all_entities)} total entities, {len(all_relationships)} total rels")

    # Deduplicate entities by name
    seen_entities: dict[str, dict] = {}
    for entity in all_entities:
        name = entity.get("name", "").strip()
        if name:
            seen_entities[name.lower()] = entity
    unique_entities = list(seen_entities.values())
    logger.info(f"🧹 Deduplicated to {len(unique_entities)} unique entities")

    # Write entities to Neo4j
    nodes_created = 0
    for entity in unique_entities:
        query = """
        MERGE (n:Entity {name: $name})
        SET n.type = $type,
            n.description = $description,
            n.document = $document
        """
        try:
            await execute_query(query, {
                "name": entity.get("name", "Unknown"),
                "type": entity.get("type", "UNKNOWN"),
                "description": entity.get("description", ""),
                "document": document_name,
            })
            nodes_created += 1
        except Exception as e:
            logger.warning(f"⚠️ Node write failed for {entity.get('name')}: {e}")

    logger.info(f"✅ Wrote {nodes_created} nodes to Neo4j")

    # Write relationships to Neo4j (using MERGE with RELATED_TO fallback)
    rels_created = 0
    for rel in all_relationships:
        source = rel.get("source", "").strip()
        target = rel.get("target", "").strip()
        rel_type = rel.get("type", "RELATED_TO").replace(" ", "_").upper()
        description = rel.get("description", "")

        if not source or not target:
            continue

        # Try APOC first, fall back to static RELATED_TO if APOC not available
        try:
            query = """
            MATCH (a:Entity {name: $source})
            MATCH (b:Entity {name: $target})
            CALL apoc.merge.relationship(a, $rel_type, {description: $description}, {}, b, {})
            YIELD rel
            RETURN rel
            """
            await execute_query(query, {
                "source": source,
                "target": target,
                "rel_type": rel_type,
                "description": description,
            })
            rels_created += 1
        except Exception as apoc_err:
            # Fallback: use static RELATED_TO relationship type
            try:
                fallback_query = """
                MATCH (a:Entity {name: $source})
                MATCH (b:Entity {name: $target})
                MERGE (a)-[r:RELATED_TO]->(b)
                SET r.type = $rel_type, r.description = $description
                """
                await execute_query(fallback_query, {
                    "source": source,
                    "target": target,
                    "rel_type": rel_type,
                    "description": description,
                })
                rels_created += 1
            except Exception as e:
                logger.warning(f"⚠️ Relationship write failed ({source} -> {target}): {e}")

    logger.info(f"✅ Wrote {rels_created} relationships to Neo4j")

    return {
        "nodes_created": nodes_created,
        "relationships_created": rels_created,
    }
