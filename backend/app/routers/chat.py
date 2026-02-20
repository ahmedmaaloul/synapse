"""
Project Synapse — Chat Router
GraphRAG-powered conversational endpoint.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.chat_engine import generate_rag_response

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    document_name: str | None = None


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Send a question and receive a GraphRAG-grounded answer.
    Streams the response token-by-token.
    """

    async def stream():
        async for chunk in generate_rag_response(request.query, request.document_name):
            yield chunk

    return StreamingResponse(stream(), media_type="text/plain")
