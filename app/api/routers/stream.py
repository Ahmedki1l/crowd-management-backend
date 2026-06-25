"""Server-Sent Events stream router (HLD 8.4).

Exposes the single live analytics stream. A client subscribes to the in-process
event bus and receives every :class:`~app.events.events.Event` rendered as an SSE
frame by :mod:`app.publish.sse`. Optional ``area_id`` / ``zone_id`` query
parameters narrow the subscription to the matching coarse topic
(``"area:<id>"`` / ``"zone:<id>"``); with neither, the client follows all events.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.api.deps import AuthDep, event_bus
from app.events.event_bus import InMemoryEventBus
from app.publish.sse import sse_stream

router = APIRouter(tags=["stream"])

#: SSE responses must not be cached by clients or intervening proxies.
_SSE_HEADERS = {"Cache-Control": "no-cache"}


def _topics_for(area_id: str | None, zone_id: int | None) -> set[str] | None:
    """Derive the bus subscription topics from optional filter parameters.

    The event bus uses coarse routing keys (``"area:<id>"`` / ``"zone:<id>"``)
    that match :attr:`app.events.events.Event.topic`. When both filters are
    absent the subscription is unfiltered (``None``), so the client receives
    every event.

    Args:
        area_id: Restrict the stream to one entry/exit area, if given.
        zone_id: Restrict the stream to one zone, if given.

    Returns:
        The set of topics to subscribe to, or ``None`` to receive all events.
    """
    topics: set[str] = set()
    if area_id is not None:
        topics.add(f"area:{area_id}")
    if zone_id is not None:
        topics.add(f"zone:{zone_id}")
    return topics or None


@router.get("/stream", dependencies=[AuthDep])
async def stream(
    area_id: str | None = Query(default=None, description="Filter to one area's events."),
    zone_id: int | None = Query(default=None, description="Filter to one zone's events."),
    bus: InMemoryEventBus = Depends(event_bus),
) -> StreamingResponse:
    """Stream live analytics events to the caller as Server-Sent Events.

    Subscribes to the event bus for the topics derived from ``area_id`` /
    ``zone_id`` (all events when neither is supplied) and streams them as
    ``text/event-stream``. The subscription is created on the request event loop
    and is closed by :func:`app.publish.sse.sse_stream` when the client
    disconnects or the generator is cancelled.

    Args:
        area_id: Optional area filter.
        zone_id: Optional zone filter.
        bus: The injected in-process event bus.

    Returns:
        A streaming ``text/event-stream`` response carrying SSE frames.
    """
    subscription = bus.subscribe(_topics_for(area_id, zone_id))
    return StreamingResponse(
        sse_stream(subscription),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
