"""Outbound Digital Twin push publisher (HLD 6.6 / 10).

Bridges the synchronous, hot-path event bus to the Digital Twin's HTTP ingest
endpoint without blocking the worker that produced the event. ``handle`` is wired
as a *sync* bus handler (see ``app.api.app``); it only enqueues, returning
immediately. A single background thread drains the queue and performs the
blocking HTTP POST with bounded exponential-backoff retries. Transport failures
are logged and never propagated — a flaky Digital Twin must not stall analytics.
"""

from __future__ import annotations

import queue
import threading

import anyio
import httpx

from app.config.schema import DigitalTwinConfig
from app.events.events import Event
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Hard cap on the outbound queue. Beyond this the Digital Twin is treated as a
#: slow consumer and the newest event is dropped rather than growing memory
#: unboundedly on the producer side.
_MAX_QUEUE_SIZE = 10_000
#: Base delay (seconds) for exponential backoff between POST retries.
_BACKOFF_BASE_SECONDS = 0.5
#: Upper bound (seconds) on a single backoff sleep.
_BACKOFF_MAX_SECONDS = 30.0
#: Sentinel enqueued by ``aclose`` to unblock and stop the worker thread.
_STOP = object()

_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _get_client() -> httpx.Client:
    """Return the shared module-level HTTP client, creating it if needed.

    The client is shared across publisher instances. ``aclose`` may close it; a
    subsequent call here transparently recreates a fresh client so a restarted
    publisher keeps working.
    """
    global _client
    with _client_lock:
        if _client is None or _client.is_closed:
            _client = httpx.Client()
        return _client


def _close_client() -> None:
    """Close and discard the shared HTTP client if one is open."""
    global _client
    with _client_lock:
        if _client is not None and not _client.is_closed:
            _client.close()
        _client = None


class DigitalTwinPublisher:
    """Pushes analytics events to the Digital Twin over HTTP off the hot path.

    Args:
        dt_cfg: Digital Twin connection settings (URL, auth header, timeout,
            retry budget).
    """

    def __init__(self, dt_cfg: DigitalTwinConfig) -> None:
        self._cfg = dt_cfg
        self._queue: queue.Queue[object] = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        """Spawn the background worker thread.

        Idempotent: a second call while the worker is alive is a no-op.
        """
        if self._worker is not None and self._worker.is_alive():
            logger.debug("DigitalTwinPublisher worker already running")
            return
        self._stopping.clear()
        self._worker = threading.Thread(
            target=self._run,
            name="dt-publisher",
            daemon=True,
        )
        self._worker.start()
        logger.info("DigitalTwinPublisher started (push_url=%s)", self._cfg.push_url)

    def handle(self, event: Event) -> None:
        """Enqueue ``event`` for asynchronous delivery (sync bus handler).

        Non-blocking: if the queue is full the event is dropped and logged so the
        producing worker thread is never stalled by a slow Digital Twin.

        Args:
            event: The analytics event to push.
        """
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning(
                "Digital Twin push queue full (size=%d); dropping %s event",
                _MAX_QUEUE_SIZE,
                event.type.value,
            )

    def _run(self) -> None:
        """Worker loop: drain the queue and POST each event until stopped."""
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                if isinstance(item, Event):
                    self._deliver(item)
            finally:
                self._queue.task_done()

    def _deliver(self, event: Event) -> None:
        """POST a single event's payload with bounded backoff retries.

        Never raises: exhausted retries and transport errors are logged. The
        retry budget is ``dt_cfg.max_retries`` *additional* attempts after the
        first try.

        Args:
            event: The event whose ``payload()`` is sent as the JSON body.
        """
        if not self._cfg.push_url:
            logger.error("Digital Twin push_url not configured; dropping event")
            return

        headers: dict[str, str] = {}
        if self._cfg.auth_header:
            headers["Authorization"] = self._cfg.auth_header

        payload = event.payload()
        attempts = max(self._cfg.max_retries, 0) + 1
        client = _get_client()

        for attempt in range(1, attempts + 1):
            if self._stopping.is_set():
                return
            try:
                response = client.post(
                    self._cfg.push_url,
                    json=payload,
                    headers=headers,
                    timeout=self._cfg.timeout_seconds,
                )
                response.raise_for_status()
                return
            except httpx.HTTPError as exc:
                if attempt >= attempts:
                    logger.error(
                        "Digital Twin push failed after %d attempts for %s event: %s",
                        attempts,
                        event.type.value,
                        exc,
                    )
                    return
                backoff = min(
                    _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                    _BACKOFF_MAX_SECONDS,
                )
                logger.warning(
                    "Digital Twin push attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    attempts,
                    exc,
                    backoff,
                )
                if self._stopping.wait(backoff):
                    return

    async def aclose(self) -> None:
        """Stop the worker, join its thread, and close the shared HTTP client.

        Signals the worker to stop, then joins it off the event loop so the
        caller's loop is not blocked. Safe to call without a prior ``start``.
        """
        self._stopping.set()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            # Queue saturated: the blocking get in the worker will still return
            # an item and observe the stopping flag on its next backoff/iteration.
            logger.debug("Digital Twin queue full while signalling stop")

        worker = self._worker
        if worker is not None and worker.is_alive():
            await anyio.to_thread.run_sync(worker.join)
        self._worker = None
        _close_client()
        logger.info("DigitalTwinPublisher stopped")
