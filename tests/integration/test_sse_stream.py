"""Contract tests for the SSE wire format and bus-to-stream delivery (HLD 10).

:func:`app.publish.sse.format_sse` must render an event as the SSE
``event:``/``data:`` frame the Digital Twin / dashboards consume, and
:func:`app.publish.sse.sse_stream` must forward events published on the in-memory
:class:`~app.events.event_bus.InMemoryEventBus` to a subscriber as those frames.

The async tests run under ``asyncio_mode = "auto"`` (see ``pyproject.toml``) and
use the real bus and stream — no mocks of the unit under test — so they assert
the observable wire output rather than internal queue mechanics.
"""

from __future__ import annotations

import json

from app.events.event_bus import InMemoryEventBus
from app.events.events import CrossingEvent, OccupancyUpdate
from app.publish.sse import format_sse, sse_stream


def test_format_sse_renders_event_name_and_json_data() -> None:
    event = OccupancyUpdate(ts=1000.0, zone_id=7, camera_id=1, count=3)

    frame = format_sse(event)

    lines = frame.split("\n")
    assert lines[0] == "event: occupancy_update"
    assert lines[1].startswith("data: ")
    assert json.loads(lines[1][len("data: ") :]) == event.payload()


def test_format_sse_frame_ends_with_blank_line() -> None:
    event = OccupancyUpdate(ts=1000.0, zone_id=1, camera_id=1, count=1)

    frame = format_sse(event)

    assert frame.endswith("\n\n")


async def test_sse_stream_first_chunk_is_connected_comment() -> None:
    bus = InMemoryEventBus()
    subscription = bus.subscribe()

    stream = sse_stream(subscription, keepalive_interval=15.0)
    first_chunk = await stream.__anext__()
    await stream.aclose()

    assert first_chunk == ": connected\n\n"


async def test_sse_stream_delivers_published_event_as_frame() -> None:
    bus = InMemoryEventBus()
    subscription = bus.subscribe()
    stream = sse_stream(subscription, keepalive_interval=15.0)
    # Drain the initial ": connected" comment so the next chunk is the event.
    await stream.__anext__()

    event = CrossingEvent(ts=1000.0, line_id=1, area_id="area-1", track_ref=4)
    bus.publish(event)
    delivered = await stream.__anext__()
    await stream.aclose()

    assert delivered == format_sse(event)
