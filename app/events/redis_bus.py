"""Redis pub/sub event bus for the scaled-out topology (HLD 6.5 / 13.1).

The in-process :class:`~app.events.event_bus.InMemoryEventBus` only fans events
out within a single process. When workers and the API run as separate processes,
this implementation carries events between them over a Redis pub/sub channel
while preserving the exact same surface, so single-process behaviour is
unchanged:

* **sync handlers** still run inline inside :meth:`publish` on the calling
  thread (state/persistence projectors, DT-push enqueue), AND a single
  background listener thread dispatches events *published by other processes*
  to the same sync handlers (so an API-only process receives worker events).
* **async subscriptions** mirror ``_AsyncSubscription``: a drop-oldest
  ``asyncio.Queue`` bound to the API loop, fed from the listener thread.

Self-published events are tagged with a per-process id and skipped on receive so
a sync handler is never invoked twice for the same publish.

Resilience: if Redis is unreachable, :meth:`publish` logs and does NOT raise —
analytics must keep running — and the listener thread reconnects in the
background.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.events.events import Event
from app.events.serialization import deserialize, serialize
from app.events.subscription import AsyncSubscription
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from redis import Redis

logger = get_logger("events.redis_bus")

# Envelope keys for the wire form: the serialized event plus the origin process
# id used to suppress double-dispatch of self-published events.
_ORIGIN_KEY = "__origin__"
_EVENT_KEY = "__event__"

# How long the listener blocks waiting for a message before looping (lets it
# observe the stop flag and reconnect promptly).
_LISTEN_TIMEOUT_S = 1.0
_RECONNECT_BACKOFF_S = 1.0


class RedisEventBus:
    """Redis pub/sub event bus matching the :class:`EventBus` surface.

    Single-process behaviour is identical to ``InMemoryEventBus``: sync handlers
    fire inline in :meth:`publish` and async subscriptions stream the same
    in-process events. Cross-process delivery is layered on top via a single
    background listener thread that deserializes incoming messages and dispatches
    them to the local sync handlers and async subscriptions.
    """

    def __init__(self, redis_url: str, channel: str = "camera-analytics:events") -> None:
        """Build a Redis-backed bus.

        The Redis connection is created lazily on first publish/listen, so the
        bus is importable and constructible without a live Redis server.

        :param redis_url: Redis connection URL (``redis://host:port/db``).
        :param channel: pub/sub channel events are published to / read from.
        """
        self._redis_url = redis_url
        self._channel = channel
        self._origin = f"{os.getpid()}:{uuid.uuid4().hex}"

        self._lock = threading.Lock()
        self._subs: list[AsyncSubscription] = []
        self._sync: list[Callable[[Event], None]] = []

        # Lazily created Redis clients (publisher + subscriber are separate so a
        # blocking pubsub read never blocks publishes).
        self._publisher: Redis | None = None
        self._listener_thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Set once the listener thread has an active channel subscription, so a
        # caller can wait until cross-process events will actually be received
        # (Redis pub/sub has no backlog: events published before the subscribe
        # completes are not delivered).
        self._listening = threading.Event()

    # -- Connection management ---------------------------------------------
    def _make_client(self) -> Redis:
        """Create a Redis client from the URL. Lazy-imports ``redis``."""
        from redis import Redis

        return Redis.from_url(self._redis_url)

    def _get_publisher(self) -> Redis:
        """Return the (lazily created) publisher client."""
        if self._publisher is None:
            self._publisher = self._make_client()
        return self._publisher

    def _encode(self, event: Event) -> str:
        """Wrap a serialized event with its origin id for the wire."""
        return json.dumps({_ORIGIN_KEY: self._origin, _EVENT_KEY: serialize(event)})

    def _decode(self, raw: str | bytes) -> tuple[str, Event]:
        """Reverse :meth:`_encode`, returning ``(origin, event)``."""
        envelope: dict[str, Any] = json.loads(raw)
        return envelope[_ORIGIN_KEY], deserialize(envelope[_EVENT_KEY])

    # -- Publish ------------------------------------------------------------
    def publish(self, event: Event) -> None:
        """Dispatch locally and publish to Redis for other processes.

        Local sync handlers and async subscriptions are served inline (exactly
        like the in-process bus). The event is additionally published to Redis so
        other processes receive it. A Redis outage is logged and swallowed here —
        analytics must not crash when the cache is down.
        """
        self._dispatch(event)
        try:
            self._get_publisher().publish(self._channel, self._encode(event))
        except Exception:
            # Never let a Redis outage break the local analytics pipeline.
            logger.exception(
                "redis publish failed for %s; event delivered locally only", event.type.value
            )
            # Drop the dead client so the next publish reconnects.
            self._publisher = None

    def _dispatch(self, event: Event) -> None:
        """Run local sync handlers and offer to async subscriptions.

        Used both for self-published events (from :meth:`publish`) and for events
        received from other processes via the listener thread.
        """
        with self._lock:
            sync = list(self._sync)
            subs = list(self._subs)
        for handler in sync:
            try:
                handler(event)
            except Exception:  # one bad handler must not stop fan-out
                logger.exception("sync event handler failed for %s", event.type.value)
        for sub in subs:
            sub.offer(event)

    # -- Subscriptions ------------------------------------------------------
    def subscribe(self, topics: set[str] | None = None) -> AsyncSubscription:
        """Return an async :class:`Subscription` bound to the running loop.

        Ensures the background listener thread is running so events from other
        processes reach this subscription.
        """
        loop = asyncio.get_running_loop()
        sub = AsyncSubscription(loop, topics, on_close=self._remove)
        with self._lock:
            self._subs.append(sub)
        self.start()
        return sub

    def subscribe_sync(self, handler: Callable[[Event], None]) -> None:
        """Register an inline sync handler and ensure the listener is running."""
        with self._lock:
            self._sync.append(handler)
        self.start()

    def unsubscribe_sync(self, handler: Callable[[Event], None]) -> None:
        """Remove a previously registered sync handler."""
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

    # -- Listener thread ----------------------------------------------------
    def start(self) -> None:
        """Start the background pubsub listener thread if not already running."""
        with self._lock:
            if self._listener_thread is not None and self._listener_thread.is_alive():
                return
            self._stop.clear()
            self._listening.clear()
            thread = threading.Thread(
                target=self._listen_loop, name="redis-event-bus-listener", daemon=True
            )
            self._listener_thread = thread
        thread.start()

    def wait_until_listening(self, timeout: float | None = None) -> bool:
        """Block until the listener has an active subscription (or ``timeout``).

        Returns ``True`` if the subscription is live. Useful before publishing
        when the publisher and subscriber share a process and the test/caller
        needs delivery to be guaranteed (Redis pub/sub drops pre-subscribe
        messages).
        """
        return self._listening.wait(timeout)

    def _listen_loop(self) -> None:
        """Subscribe to the channel and dispatch incoming events until stopped.

        Reconnects with a fixed backoff on connection errors so a transient Redis
        outage does not permanently stop cross-process delivery.
        """
        while not self._stop.is_set():
            try:
                client = self._make_client()
                pubsub = client.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(self._channel)
                self._listening.set()
                self._consume(pubsub)
                pubsub.close()
                client.close()
            except Exception:
                if self._stop.is_set():
                    break
                logger.exception(
                    "redis listener error; reconnecting in %.1fs", _RECONNECT_BACKOFF_S
                )
                time.sleep(_RECONNECT_BACKOFF_S)
            finally:
                self._listening.clear()

    def _consume(self, pubsub: Any) -> None:
        """Pull messages off ``pubsub`` and dispatch non-self events."""
        while not self._stop.is_set():
            message = pubsub.get_message(timeout=_LISTEN_TIMEOUT_S)
            if message is None or message.get("type") != "message":
                continue
            self._handle_message(message["data"])

    def _handle_message(self, raw: str | bytes) -> None:
        """Decode one pubsub payload and dispatch it unless it is self-published."""
        try:
            origin, event = self._decode(raw)
        except (ValueError, KeyError, TypeError):
            logger.exception("dropping undecodable event from redis bus")
            return
        if origin == self._origin:
            # Already dispatched locally in publish(); skip to avoid double-fire.
            return
        self._dispatch(event)

    def close(self) -> None:
        """Stop the listener thread and close the publisher connection."""
        self._stop.set()
        thread = self._listener_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_LISTEN_TIMEOUT_S + _RECONNECT_BACKOFF_S + 1.0)
        with self._lock:
            self._listener_thread = None
        if self._publisher is not None:
            try:
                self._publisher.close()
            except Exception:
                logger.exception("error closing redis publisher")
            finally:
                self._publisher = None
