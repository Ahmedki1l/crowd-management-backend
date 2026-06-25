"""Server-Sent Events (SSE) wire formatting and streaming (HLD 6.6 / 10).

One SSE endpoint streams analytics events to the Digital Twin / dashboards. Each
:class:`~app.events.events.Event` is rendered on the wire as::

    event: <event.sse_event>
    data: <json of event.payload()>

(blank line terminator). A subscription is a per-client async stream produced by
:meth:`app.events.event_bus.EventBus.subscribe`; this module adapts it to the SSE
text protocol, emitting periodic keepalive comments so idle connections and
intervening proxies do not time the connection out.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import anyio

from app.events.events import Event, Subscription
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: First line sent on a fresh connection so the client knows the stream is live.
_CONNECTED_COMMENT = ": connected\n\n"
#: Sent whenever the stream is idle longer than ``keepalive_interval`` seconds.
_KEEPALIVE_COMMENT = ": keepalive\n\n"


def format_sse(event: Event) -> str:
    """Render one event as an SSE ``event:``/``data:`` frame.

    The event name is ``event.sse_event`` and the data line is the compact JSON
    encoding of ``event.payload()``. The frame is terminated by a blank line, as
    required by the SSE specification.

    Args:
        event: The analytics event to serialise.

    Returns:
        The complete SSE frame, including its trailing blank line.
    """
    data = json.dumps(event.payload())
    return f"event: {event.sse_event}\ndata: {data}\n\n"


async def sse_stream(
    subscription: Subscription,
    keepalive_interval: float = 15.0,
) -> AsyncIterator[str]:
    """Yield SSE-formatted text for every event from ``subscription``.

    Emits ``: connected`` immediately, then forwards each event via
    :func:`format_sse`. When no event arrives within ``keepalive_interval``
    seconds a ``: keepalive`` comment is emitted instead, keeping the connection
    and any intermediary proxies alive. The subscription is always closed via
    ``aclose()`` when the generator terminates (client disconnect, cancellation,
    or end of stream).

    Args:
        subscription: Async event stream for one client (see
            :class:`app.events.events.Subscription`).
        keepalive_interval: Maximum idle time, in seconds, before a keepalive
            comment is sent.

    Yields:
        SSE wire-protocol text chunks ready to write to the HTTP response.
    """
    yield _CONNECTED_COMMENT
    iterator = subscription.__aiter__()
    try:
        while True:
            event: Event | None = None
            with anyio.move_on_after(keepalive_interval):
                try:
                    event = await iterator.__anext__()
                except StopAsyncIteration:
                    return
            if event is None:
                # Idle past the keepalive window: keep the connection warm.
                yield _KEEPALIVE_COMMENT
                continue
            yield format_sse(event)
    finally:
        await subscription.aclose()
