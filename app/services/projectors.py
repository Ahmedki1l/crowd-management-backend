"""Event projectors — fan analytics events into the read model and storage (HLD 6.6).

Every analytics result is published once on the internal event bus
(:mod:`app.events.event_bus`). Two projectors consume that stream for orthogonal
purposes:

* :class:`StateProjector` updates the in-memory current-state cache
  (:class:`~app.services.state_store.StateStore`) **synchronously** on the
  publishing thread. It must stay fast — no I/O, no DB — so it never slows the
  pipeline. It is the write side of the query model the API reads.
* :class:`PersistenceProjector` durably stores the historical time series. Its
  :meth:`~PersistenceProjector.handle` only enqueues onto an internal queue and
  returns immediately; a single background worker thread drains the queue and
  writes through ``session_scope()`` + repositories. This keeps slow DB writes
  off the pipeline thread and batches them behind one transaction per drained
  event.

Alerts are persisted by the engine at the point they are produced (they carry an
evidence snapshot and so need the live frame), not here, so this projector ignores
that event type.
"""

from __future__ import annotations

import queue
import threading
from datetime import datetime

from app.db.repositories.crossing_repo import CrossingRepository
from app.db.repositories.dwell_repo import DwellRepository
from app.db.repositories.occupancy_repo import OccupancyRepository
from app.db.session import session_scope
from app.events.events import (
    CameraHealth,
    CountUpdate,
    CrossingEvent,
    DwellClosed,
    Event,
    OccupancyUpdate,
    WaitingUpdate,
)
from app.services.state_store import CameraHealthState, StateStore
from app.utils.logging import get_logger
from app.utils.timeutil import to_datetime

logger = get_logger(__name__)

# Default minimum spacing between persisted occupancy samples per zone. Live
# occupancy updates can arrive every frame; persisting each one would flood the
# time-series table without adding analytical value, so we throttle to one
# sample per zone per interval. History queries bucket-average anyway (HLD 9).
_DEFAULT_PERSIST_INTERVAL_S = 5.0

# How long the worker waits for the next event before re-checking the stop flag.
_QUEUE_POLL_TIMEOUT_S = 0.5


class StateProjector:
    """Update the in-memory current-state cache from analytics events.

    Synchronous and side-effect-free beyond the store write, so it can run on the
    publishing thread without adding latency. Unknown event types are ignored —
    a projector only owns the projections it knows about.
    """

    def __init__(self, store: StateStore) -> None:
        """Bind the projector to the current-state ``store`` it writes into."""
        self._store = store

    def handle(self, event: Event) -> None:
        """Route one event to the matching current-state write.

        Args:
            event: An analytics event from the bus. Types this projector does not
                project (alerts, crossings, dwell) are ignored.
        """
        if isinstance(event, OccupancyUpdate):
            self._store.set_occupancy(
                event.zone_id, event.count, event.dt_space_id, event.ts
            )
        elif isinstance(event, CountUpdate):
            self._store.set_counts(
                area_id=event.area_id,
                in_count=event.in_count,
                out_count=event.out_count,
                net=event.net,
                ts=event.ts,
                lines=self._lines_from_count(event),
            )
        elif isinstance(event, WaitingUpdate):
            self._store.set_waiting(
                event.zone_id,
                event.current_waits,
                event.avg_dwell_s,
                event.dt_space_id,
                event.ts,
            )
        elif isinstance(event, CameraHealth):
            self._store.set_camera_health(
                CameraHealthState(
                    camera_id=event.camera_id,
                    fps=event.fps,
                    last_frame_age_s=event.last_frame_age_s,
                    queue_depth=event.queue_depth,
                    healthy=event.healthy,
                    ts=event.ts,
                )
            )

    @staticmethod
    def _lines_from_count(event: CountUpdate) -> dict[int, tuple[int, int]] | None:
        """Rebuild the per-line ``{line_id: (in, out)}`` map for the area state.

        A :class:`CountUpdate` carries the area totals plus the single ``line_id``
        whose crossing triggered it; the per-line breakdown attributes the area's
        running totals to that line. When no line is identified (area-level
        aggregate) there is nothing to attribute, so ``None`` is returned and the
        store keeps the area totals without a line breakdown.
        """
        if event.line_id is None:
            return None
        return {event.line_id: (event.in_count, event.out_count)}


class PersistenceProjector:
    """Durably store the historical time series off the pipeline thread.

    :meth:`handle` enqueues events without blocking; a background worker drains
    the queue and writes each event through a fresh ``session_scope()``
    transaction. Per-event failures are logged and skipped so one bad write never
    stalls the stream. Occupancy samples are throttled per zone (see
    ``persist_interval``) to bound write volume.
    """

    def __init__(self, persist_interval: float = _DEFAULT_PERSIST_INTERVAL_S) -> None:
        """Configure the projector (does not start the worker thread).

        Args:
            persist_interval: Minimum seconds between persisted occupancy samples
                for a given zone. Must be non-negative; ``0`` disables throttling.
        """
        if persist_interval < 0:
            raise ValueError("persist_interval must be non-negative")
        self._persist_interval = persist_interval
        self._queue: queue.Queue[Event] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        # Per-zone epoch-seconds of the last persisted occupancy sample. Read and
        # written only on the single worker thread, so it needs no lock.
        self._last_occupancy_write: dict[int, float] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the background worker thread. Idempotent."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._run, name="persistence-projector", daemon=True
        )
        self._worker.start()
        logger.info("persistence projector started", extra={"event": "projector_started"})

    def stop(self) -> None:
        """Stop the worker, draining queued events first, then join it.

        Blocks until the queue is empty and the worker thread has exited so no
        buffered event is lost on shutdown.
        """
        worker = self._worker
        if worker is None:
            return
        # Drain remaining work before signalling stop so queued events are not
        # dropped, then let the worker observe the stop flag on an empty queue.
        self._queue.join()
        self._stop_event.set()
        worker.join()
        self._worker = None
        logger.info("persistence projector stopped", extra={"event": "projector_stopped"})

    # ------------------------------------------------------------------ #
    # Producer side (pipeline thread)
    # ------------------------------------------------------------------ #
    def handle(self, event: Event) -> None:
        """Enqueue an event for durable storage without blocking.

        Only event types this projector persists (occupancy samples, crossings,
        closed dwells) are queued; others are dropped here so the worker never
        wakes for work it would ignore.

        Args:
            event: The analytics event to persist asynchronously.
        """
        if isinstance(event, (OccupancyUpdate, CrossingEvent, DwellClosed)):
            self._queue.put_nowait(event)

    # ------------------------------------------------------------------ #
    # Consumer side (worker thread)
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        """Drain the queue until stopped, persisting each event in its own txn."""
        while not (self._stop_event.is_set() and self._queue.empty()):
            try:
                event = self._queue.get(timeout=_QUEUE_POLL_TIMEOUT_S)
            except queue.Empty:
                continue
            try:
                self._persist(event)
            except Exception:  # noqa: BLE001
                # A persistence failure for one event (DB hiccup, constraint,
                # serialization) must not kill the worker or stall the stream.
                # Log with the stack and move on; the in-memory read model is
                # unaffected and the next event still gets its own transaction.
                logger.exception(
                    "failed to persist event",
                    extra={"event": "persist_failed"},
                )
            finally:
                self._queue.task_done()

    def _persist(self, event: Event) -> None:
        """Write one event to its time-series table in a fresh transaction."""
        if isinstance(event, OccupancyUpdate):
            self._persist_occupancy(event)
        elif isinstance(event, CrossingEvent):
            self._persist_crossing(event)
        elif isinstance(event, DwellClosed):
            self._persist_dwell(event)

    def _persist_occupancy(self, event: OccupancyUpdate) -> None:
        """Append an occupancy sample, throttled to one per zone per interval."""
        if not self._should_persist_occupancy(event.zone_id, event.ts):
            return
        ts = to_datetime(event.ts)
        with session_scope() as session:
            OccupancyRepository(session).add_sample(event.zone_id, ts, event.count)
        # Record the write only after the transaction committed, so a failed
        # write does not advance the throttle and silently drop the next sample.
        self._last_occupancy_write[event.zone_id] = event.ts

    def _persist_crossing(self, event: CrossingEvent) -> None:
        """Persist one directional line-crossing event."""
        ts = to_datetime(event.ts)
        with session_scope() as session:
            CrossingRepository(session).add(
                line_id=event.line_id,
                area_id=event.area_id,
                ts=ts,
                direction=event.direction.value,
                track_ref=event.track_ref,
            )

    def _persist_dwell(self, event: DwellClosed) -> None:
        """Persist one completed dwell session."""
        enter_ts: datetime = to_datetime(event.enter_ts)
        leave_ts: datetime = to_datetime(event.leave_ts)
        with session_scope() as session:
            DwellRepository(session).add_closed(
                zone_id=event.zone_id,
                track_ref=event.track_ref,
                enter_ts=enter_ts,
                leave_ts=leave_ts,
                dwell_s=event.dwell_s,
            )

    def _should_persist_occupancy(self, zone_id: int, ts: float) -> bool:
        """Return whether enough time has elapsed to persist this zone's sample."""
        if self._persist_interval <= 0:
            return True
        last = self._last_occupancy_write.get(zone_id)
        return last is None or (ts - last) >= self._persist_interval
