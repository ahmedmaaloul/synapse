"""
Project Synapse — Chat Router

GraphRAG conversational endpoint. Streams Server-Sent Events so the client
receives, in order: a ``citations`` event (entities that ground the answer),
many ``token`` events, then ``done`` (or ``error``).
"""

from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.chat_engine import generate_rag_response

router = APIRouter()


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)


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
