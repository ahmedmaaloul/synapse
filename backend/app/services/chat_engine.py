"""
Project Synapse — Chat Engine Service
GraphRAG: retrieves relevant subgraph context, then streams a
Gemini-powered answer grounded in the knowledge graph.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.config import get_settings
from app.neo4j_driver import execute_query
from typing import AsyncGenerator

# ── RAG Prompt ───────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant for Project Synapse, a Knowledge Graph Explorer.
You answer questions based ONLY on the provided knowledge graph context.
If the context doesn't contain enough information, say so honestly.

Knowledge Graph Context:
{context}

Rules:
- Ground your answer in the provided entities and relationships
- Reference specific entities when relevant
- Be concise but thorough
- If information is not in the context, clearly state that""",
    ),
    ("human", "{question}"),
])


async def _retrieve_subgraph(query: str, document_name: str | None = None) -> str:
    """
    Search Neo4j for entities matching the query and retrieve
    their neighborhood (connected nodes + relationships).
    """
    # Full-text search for matching entities
    search_query = """
    MATCH (n:Entity)
    WHERE toLower(n.name) CONTAINS toLower($query)
       OR toLower(n.description) CONTAINS toLower($query)
    WITH n
    LIMIT 10
    OPTIONAL MATCH (n)-[r]-(m:Entity)
    RETURN
        n.name AS entity,
        n.type AS type,
        n.description AS description,
        collect(DISTINCT {
            related_entity: m.name,
            related_type: m.type,
            relationship: type(r),
            related_description: m.description
        }) AS connections
    """
    params: dict = {"query": query}

    results = await execute_query(search_query, params)

    if not results:
        # Fallback: return all entities if no match found
        fallback_query = """
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]-(m:Entity)
        RETURN
            n.name AS entity,
            n.type AS type,
            n.description AS description,
            collect(DISTINCT {
                related_entity: m.name,
                related_type: m.type,
                relationship: type(r),
                related_description: m.description
            }) AS connections
        LIMIT 20
        """
        results = await execute_query(fallback_query)

    # Format into readable context
    context_parts = []
    for record in results:
        entity_info = f"Entity: {record['entity']} (Type: {record['type']})"
        if record.get("description"):
            entity_info += f"\n  Description: {record['description']}"

        connections = record.get("connections", [])
        if connections:
            conn_lines = []
            for conn in connections:
                if conn.get("related_entity"):
                    conn_lines.append(
                        f"  → {conn['relationship']} → {conn['related_entity']} ({conn['related_type']})"
                    )
            if conn_lines:
                entity_info += "\n  Relationships:\n" + "\n".join(conn_lines)

        context_parts.append(entity_info)

    return "\n\n".join(context_parts) if context_parts else "No relevant information found in the knowledge graph."


async def generate_rag_response(
    question: str, document_name: str | None = None
) -> AsyncGenerator[str, None]:
    """
    GraphRAG pipeline:
    1. Retrieve relevant subgraph context from Neo4j
    2. Stream a Gemini-generated answer grounded in the context
    """
    settings = get_settings()

    # 1. Retrieve context
    context = await _retrieve_subgraph(question, document_name)

    # 2. Generate streaming response
    llm = ChatOllama(
        model="mistral",
        base_url="http://host.docker.internal:11434",
        temperature=0.3,
    )

    chain = RAG_PROMPT | llm

    async for chunk in chain.astream({"context": context, "question": question}):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content
