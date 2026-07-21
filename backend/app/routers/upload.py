# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Upload Router

Ingestion is a background job: ``POST /api/upload`` validates + parses the PDF,
starts extraction in the background, and returns a ``job_id`` immediately. The
client then opens ``GET /api/upload/{job_id}/events`` (SSE) to watch progress in
real time — parsing → extracting (n/N chunks) → writing → clustering → done.

New entities change the shape of the graph, so communities are re-detected as
the final step of the *same* job (when ``community_detection_enabled``): the
themes the UI shows are then always in sync with the graph the user just built.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.services.communities import detect_and_summarize
from app.services.graph_builder import build_knowledge_graph
from app.services.jobs import job_bus
from app.services.pdf_parser import extract_text_from_pdf

logger = logging.getLogger(__name__)
router = APIRouter()

ProgressCallback = Callable[[dict], Awaitable[None]]


async def _cluster_communities(on_progress: ProgressCallback) -> dict:
    """Re-detect + re-summarize communities as the last phase of an ingest.

    Progress is republished under ``community_progress`` rather than ``progress``
    so the ingest progress bar (which maps *extraction* stages onto 0-100%) is
    not rewound by a stage it does not know about.

    Returns the fields merged into the job's ``done`` payload — zeros when the
    feature is disabled, and zeros again if clustering fails, because a failed
    rebuild must never lose the user the document they just ingested.
    """
    empty = {"communities": 0, "communities_summarized": 0, "modularity": 0.0}
    if not get_settings().community_detection_enabled:
        return empty

    async def on_community_progress(event: dict) -> None:
        await on_progress({**event, "type": "community_progress"})

    try:
        result = await detect_and_summarize(on_progress=on_community_progress)
    except Exception as e:  # noqa: BLE001 - the graph is already written; keep it
        logger.warning("⚠️ Community detection skipped after ingest: %s", e)
        return empty

    return {
        "communities": result["communities"],
        "communities_summarized": result["summarized"],
        "modularity": result["modularity"],
    }


async def _process_document(job_id: str, chunks: list[str], filename: str, theme: str) -> None:
    """Background task: extract entities from parsed chunks and write to Neo4j."""
    try:
        async def on_progress(event: dict) -> None:
            await job_bus.publish(job_id, event)

        await job_bus.publish(
            job_id,
            {"type": "progress", "stage": "extracting", "processed": 0, "total": len(chunks)},
        )

        result = await build_knowledge_graph(
            chunks, filename, theme=theme, on_progress=on_progress
        )

        community_result = await _cluster_communities(on_progress)

        await job_bus.publish(
            job_id,
            {
                "type": "done",
                "data": {
                    "filename": filename,
                    "chunks_processed": len(chunks),
                    **result,
                    **community_result,
                },
            },
        )
        logger.info(
            "✅ Done: %s nodes, %s rels for %s",
            result["nodes_created"],
            result["relationships_created"],
            filename,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("❌ Background processing failed for %s", filename)
        await job_bus.publish(job_id, {"type": "error", "data": str(e)})


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    theme: str = Form("Generic"),
):
    """Validate + parse a PDF, then kick off background extraction.

    Returns a ``job_id`` the client subscribes to for live progress.
    """
    settings = get_settings()
    logger.info("📥 Upload request: %s (theme=%s)", file.filename, theme)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    logger.info("📄 Read %s bytes from %s", len(contents), file.filename)

    try:
        chunks = extract_text_from_pdf(contents)
    except Exception as e:
        logger.exception("❌ PDF parsing failed")
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}") from e

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from PDF (is it a scanned image?).",
        )

    if len(chunks) > settings.max_chunks:
        logger.info("⚠️ Limiting to %s chunks (was %s)", settings.max_chunks, len(chunks))
        chunks = chunks[: settings.max_chunks]

    job_id = job_bus.create()
    background_tasks.add_task(_process_document, job_id, chunks, file.filename, theme)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "total_chunks": len(chunks),
        "status": "processing",
    }


@router.get("/upload/{job_id}/events")
async def upload_events(job_id: str):
    """SSE stream of progress events for an ingestion job."""

    async def stream():
        async for event in job_bus.subscribe(job_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
