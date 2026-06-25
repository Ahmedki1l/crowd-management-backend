"""In-process event bus (HLD 5 event bus / 6.6).

Two subscriber kinds, because producers and consumers live on different sides:

* **sync handlers** — called inline inside ``publish()`` on the *worker* thread.
  Used by the state projector, persistence projector, and DT-push enqueue. They
  must be fast and non-blocking (offload slow work to their own queue/thread).
* **async subscriptions** — one per SSE client, backed by an ``asyncio.Queue``
  bound to the API event loop. ``publish()`` hands events across the thread
  boundary with ``loop.call_soon_threadsafe``; a slow client drops oldest.

When workers are scaled out into separate processes, this class is swapped for a
Redis pub/sub implementation behind the same ``EventBus`` Protocol.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

from app.events.events import Event, EventBus
from app.events.subscription import AsyncSubscription

logger = logging.getLogger("app.events.bus")


class InMemoryEventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: list[AsyncSubscription] = []
        self._sync: list[Callable[[Event], None]] = []

    def publish(self, event: Event) -> None:
        with self._lock:
            sync = list(self._sync)
            subs = list(self._subs)
        for handler in sync:
            try:
                handler(event)
            except Exception:  # one bad handler must not stop fan-out
                logger.exception("sync event handler failed for %s", event.type)
        for sub in subs:
            sub.offer(event)

    def subscribe(self, topics: set[str] | None = None) -> AsyncSubscription:
        loop = asyncio.get_running_loop()
        sub = AsyncSubscription(loop, topics, on_close=self._remove)
        with self._lock:
            self._subs.append(sub)
        return sub

    def subscribe_sync(self, handler: Callable[[Event], None]) -> None:
        with self._lock:
            self._sync.append(handler)

    def unsubscribe_sync(self, handler: Callable[[Event], None]) -> None:
        with self._lock:
            if handler in self._sync:
                self._sync.remove(handler)

    def _remove(self, sub: AsyncSubscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)


_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the process-wide event bus singleton.

    Uses the Redis-backed bus when ``cache.url`` is configured (scaled-out
    topology, HLD 6.5 / 13.1), else the in-process bus. The Redis connection is
    deferred, so selecting it does not require a live server at import time.
    """
    global _bus
    if _bus is None:
        from app.config.settings import get_settings

        cache_url = get_settings().cache.url
        if cache_url:
            from app.events.redis_bus import RedisEventBus

            _bus = RedisEventBus(cache_url)
        else:
            _bus = InMemoryEventBus()
    return _bus


def reset_event_bus() -> None:
    global _bus
    _bus = None
