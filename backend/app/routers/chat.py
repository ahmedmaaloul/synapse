# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Chat Router

GraphRAG conversational endpoint. Streams Server-Sent Events so the client
receives, in order: a ``citations`` event (entities that ground the answer),
optional ``paths`` and ``sources`` events, many ``token`` events, then ``done``
— carrying ``usage`` — or ``error``.

``POST /retrieve`` is the same retrieval with the generation step left out. It
exists for MCP servers and agent hosts that already have a model in the loop:
they get the routed, ranked context cut to a character budget plus the numbers
to budget against, and are not billed for a second generation call made here
on their behalf.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.chat_engine import (
    _sources_payload,
    estimate_tokens,
    generate_rag_response,
    retrieve_subgraph,
    truncate_context,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    # Seed entities (or communities) to retrieve; the graph walk fans out from
    # these, so the cap bounds the work done per call.
    k: int = Field(8, ge=1, le=20)
    # Character budget for ``context``; ``None`` returns everything retrieved.
    # Below ~200 chars there is no room for even one entity block.
    max_context_chars: int | None = Field(None, ge=200)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@router.post("/chat")
async def chat(request: ChatRequest):
    """Answer a question with a GraphRAG-grounded, streamed response."""
    history = [turn.model_dump() for turn in request.history]

    async def stream():
        async for event in generate_rag_response(request.query, history):
            yield _sse(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
        },
    )


@router.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """Retrieve the GraphRAG context for a question — without answering it.

    Built for MCP servers and agent hosts that already run their own LLM. They
    receive exactly what ``/chat`` would have prompted — routed ``mode``,
    ranked entities or communities, reasoning ``paths`` and source excerpts —
    cut to ``max_context_chars`` on a line boundary, plus a ``usage`` block
    (sizes, a token estimate, whether the cut happened) to budget against.
    Skipping the generation step here means one LLM bill per answer, not two.
    """
    try:
        retrieval = await retrieve_subgraph(request.query, k=request.k)
        context, citations = retrieval
        paths = list(getattr(retrieval, "paths", []) or [])
        sources = list(getattr(retrieval, "sources", []) or [])
        mode = getattr(retrieval, "mode", "local")
    except Exception as e:  # noqa: BLE001
        logger.exception("Retrieval failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to query the knowledge graph."
        ) from e

    context, truncated = truncate_context(context, request.max_context_chars)
    return {
        "mode": mode,
        "context": context,
        "citations": citations,
        "paths": paths,
        "sources": _sources_payload(sources),
        "usage": {
            "context_chars": len(context),
            "context_tokens_est": estimate_tokens(context),
            "truncated": truncated,
            "citations": len(citations),
            "paths": len(paths),
            "sources": len(sources),
        },
    }
