"""Shared async event subscription (one per SSE client).

Both bus implementations (:class:`~app.events.event_bus.InMemoryEventBus` and
:class:`~app.events.redis_bus.RedisEventBus`) hand SSE clients the *same* async
stream: a drop-oldest ``asyncio.Queue`` bound to the API event loop, fed
thread-safely from a producer thread. Defining it once here keeps the two buses
from drifting in how they buffer, filter, and close client streams.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

from app.events.events import Event

# Marker pushed into a subscription's queue to end its async iteration cleanly.
SENTINEL = object()


class AsyncSubscription:
    """Async, topic-filtered, drop-oldest event stream for one SSE client.

    The owning bus passes ``on_close`` so the subscription can deregister itself
    when the client disconnects, without depending on a concrete bus type.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        topics: set[str] | None,
        on_close: Callable[[AsyncSubscription], None],
        maxsize: int = 1000,
    ) -> None:
        self._loop = loop
        self._topics = topics
        self._on_close = on_close
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    def _matches(self, event: Event) -> bool:
        if self._topics is None:
            return True
        return event.topic in self._topics or event.topic == "all"

    def offer(self, event: Event) -> None:
        """Thread-safe: enqueue onto the API loop from any producer thread."""
        if self._closed or not self._matches(event):
            return

        def _put() -> None:
            if self._queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()  # drop oldest
            self._queue.put_nowait(event)

        # Loop already closed (shutdown race) — nothing to deliver.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(_put)

    def __aiter__(self) -> AsyncSubscription:
        return self

    async def __anext__(self) -> Event:
        event = await self._queue.get()
        if event is SENTINEL:
            raise StopAsyncIteration
        return event

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_close(self)
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(SENTINEL)
