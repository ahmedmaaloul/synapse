"""
Project Synapse — Upload Router
Handles PDF file upload, parsing, and knowledge graph construction.
Uses BackgroundTasks so the server stays responsive during processing.
"""

import logging
import asyncio
import fastapi
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.services.pdf_parser import extract_text_from_pdf
from app.services.graph_builder import build_knowledge_graph

logger = logging.getLogger(__name__)
router = APIRouter()

# Max chunks to process (prevents extremely long processing)
MAX_CHUNKS = 15

# Simple in-memory status tracker
_upload_status: dict[str, dict] = {}


async def _process_document(file_id: str, contents: bytes, filename: str):
    """Background task: parse PDF, extract entities, write to Neo4j."""
    try:
        _upload_status[file_id] = {"status": "parsing", "filename": filename}

        text_chunks = extract_text_from_pdf(contents)
        logger.info(f"✂️  Extracted {len(text_chunks)} chunks from {filename}")

        if not text_chunks:
            _upload_status[file_id] = {
                "status": "error",
                "filename": filename,
                "detail": "Could not extract text from PDF.",
            }
            return

        if len(text_chunks) > MAX_CHUNKS:
            logger.info(f"⚠️  Limiting to {MAX_CHUNKS} chunks (was {len(text_chunks)})")
            text_chunks = text_chunks[:MAX_CHUNKS]

        _upload_status[file_id] = {
            "status": "extracting",
            "filename": filename,
            "total_chunks": len(text_chunks),
        }

        logger.info(f"🧠 Starting entity extraction for {filename}...")
        result = await build_knowledge_graph(text_chunks, filename)

        _upload_status[file_id] = {
            "status": "done",
            "filename": filename,
            "chunks_processed": len(text_chunks),
            "nodes_created": result["nodes_created"],
            "relationships_created": result["relationships_created"],
        }
        logger.info(
            f"✅ Done: {result['nodes_created']} nodes, "
            f"{result['relationships_created']} rels for {filename}"
        )

    except Exception as e:
        logger.exception(f"❌ Background processing failed for {filename}: {e}")
        _upload_status[file_id] = {
            "status": "error",
            "filename": filename,
            "detail": str(e),
        }


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    theme: str = fastapi.Form("Generic")
):
    """
    Upload a PDF document. Returns immediately with a job ID.
    Processing happens in the background.
    """
    logger.info(f"📥 Upload request received: {file.filename} (theme={theme})")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        contents = await file.read()
        logger.info(f"📄 Read {len(contents)} bytes from {file.filename}")

        try:
            text_chunks = extract_text_from_pdf(contents)
            logger.info(f"✂️  Extracted {len(text_chunks)} chunks")
        except Exception as parse_err:
            logger.exception(f"❌ PDF parsing failed: {parse_err}")
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(parse_err)}")

        if not text_chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # Limit chunks to prevent timeout
        if len(text_chunks) > MAX_CHUNKS:
            logger.info(f"⚠️  Limiting to {MAX_CHUNKS} chunks (was {len(text_chunks)})")
            text_chunks = text_chunks[:MAX_CHUNKS]

        # Process synchronously but with per-chunk timeouts so it finishes
        logger.info(f"🧠 Starting entity extraction with Gemini (Theme: {theme})...")
        result = await build_knowledge_graph(text_chunks, file.filename, theme=theme)
        logger.info(f"✅ Done: {result['nodes_created']} nodes, {result['relationships_created']} rels")

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_processed": len(text_chunks),
            "nodes_created": result["nodes_created"],
            "relationships_created": result["relationships_created"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
