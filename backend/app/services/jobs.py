# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — In-memory Job Bus

A tiny pub/sub so a long-running ingestion (background task) can stream progress
events to a Server-Sent-Events endpoint the browser subscribes to.

Flow: the client POSTs a document, gets a ``job_id``, then opens an SSE stream to
watch that job. Events published before the stream connects are retained in the
job's queue (``asyncio.Queue`` is unbounded), so nothing is lost in the gap.

Single-process only (fine for a single backend replica). For horizontal scaling
this would move to Redis pub/sub — the interface is deliberately small so that
swap is mechanical.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

# A terminal event type ends the SSE stream.
TERMINAL_TYPES = {"done", "error"}


@dataclass
class Job:
    id: str
    queue: asyncio.Queue[dict] = field(default_factory=asyncio.Queue)
    finished: bool = False
    last_event: dict | None = None


class JobBus:
    """Registry of in-flight jobs and their event queues (one consumer each)."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._counter = 0

    def create(self) -> str:
        self._counter += 1
        job_id = f"job_{self._counter:06d}"
        self._jobs[job_id] = Job(id=job_id)
        return job_id

    async def publish(self, job_id: str, event: dict) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.last_event = event
        if event.get("type") in TERMINAL_TYPES:
            job.finished = True
        await job.queue.put(event)

    async def subscribe(self, job_id: str) -> AsyncGenerator[dict, None]:
        """Yield a job's events until a terminal (``done``/``error``) event."""
        job = self._jobs.get(job_id)
        if job is None:
            yield {"type": "error", "data": "Unknown job id."}
            return

        while True:
            event = await job.queue.get()
            yield event
            if event.get("type") in TERMINAL_TYPES:
                # Free the job shortly after completion.
                self._jobs.pop(job_id, None)
                return

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


# Module-level singleton.
job_bus = JobBus()
