# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for the in-memory job bus (progress pub/sub)."""

from app.services.jobs import JobBus


class TestJobBus:
    def test_create_unique_ids(self):
        bus = JobBus()
        assert bus.create() != bus.create()

    async def test_publish_then_subscribe_delivers_until_terminal(self):
        bus = JobBus()
        job_id = bus.create()
        await bus.publish(job_id, {"type": "progress", "processed": 1})
        await bus.publish(job_id, {"type": "progress", "processed": 2})
        await bus.publish(job_id, {"type": "done", "data": {"nodes_created": 3}})

        events = [e async for e in bus.subscribe(job_id)]
        assert [e["type"] for e in events] == ["progress", "progress", "done"]

    async def test_terminal_error_ends_stream(self):
        bus = JobBus()
        job_id = bus.create()
        await bus.publish(job_id, {"type": "error", "data": "boom"})
        events = [e async for e in bus.subscribe(job_id)]
        assert events == [{"type": "error", "data": "boom"}]

    async def test_unknown_job_yields_error(self):
        bus = JobBus()
        events = [e async for e in bus.subscribe("job_999999")]
        assert events[0]["type"] == "error"

    async def test_job_freed_after_completion(self):
        bus = JobBus()
        job_id = bus.create()
        await bus.publish(job_id, {"type": "done", "data": {}})
        _ = [e async for e in bus.subscribe(job_id)]
        assert bus.get(job_id) is None
