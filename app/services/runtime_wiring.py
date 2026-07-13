"""Wire the analytics event bus and the history writers to their consumers (HLD 6.6).

A single place that attaches the read-model projector, the persistence projector, and the
occupancy-history writers. Used by **both** entry paths so the wiring never drifts or gets
duplicated:

* the FastAPI lifespan (``app.api.app``) in ``--api`` deployments;
* the worker entry points (``app.main`` ``--workers`` / ``--worker``), which have
  no FastAPI lifespan and would otherwise publish events that nothing consumes.

The two history writers are *pollers*, not bus subscribers: :class:`OccupancySampler`
reads the live read model at a fixed rate (which is what makes a plain mean a correct
time-weighted mean — see its module docstring), and :class:`HistoryWorker` aggregates
the minutes it writes and prunes what has aged out. Both upsert on ``(space_id, bucket_ts)``, so running them in more
than one process cannot duplicate a bucket.

``aclose`` is for async callers (the lifespan); ``close_sync`` is for the plain
worker processes.
"""

from __future__ import annotations

from app.events.event_bus import InMemoryEventBus
from app.services.history_worker import HistoryWorker
from app.services.occupancy_sampler import OccupancySampler
from app.services.projectors import PersistenceProjector, StateProjector
from app.services.state_store import StateStore
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RuntimeWiring:
    """Attach the bus consumers and history writers for a process, and tear them down."""

    def __init__(self, bus: InMemoryEventBus, store: StateStore) -> None:
        self._bus = bus
        self.state_projector = StateProjector(store)
        self.persistence_projector = PersistenceProjector()
        self.occupancy_sampler = OccupancySampler(store)
        self.history_worker = HistoryWorker()

    def start(self) -> None:
        """Start background workers and subscribe every consumer to the bus."""
        self.persistence_projector.start()
        self._bus.subscribe_sync(self.state_projector.handle)
        self._bus.subscribe_sync(self.persistence_projector.handle)
        # Started after the state projector: the sampler reads the read model the
        # projector fills, so there is nothing to sample until that is subscribed.
        self.occupancy_sampler.start()
        self.history_worker.start()
        logger.info("runtime wiring started")

    def _unsubscribe(self) -> None:
        self._bus.unsubscribe_sync(self.state_projector.handle)
        self._bus.unsubscribe_sync(self.persistence_projector.handle)

    async def aclose(self) -> None:
        """Async teardown for the FastAPI lifespan."""
        self.close_sync()

    def close_sync(self) -> None:
        """Unsubscribe every consumer and stop the background writers."""
        self._unsubscribe()
        self.persistence_projector.stop()
        # Stopping the sampler flushes its open minute, so a clean shutdown does not
        # discard the bucket in progress.
        self.occupancy_sampler.stop()
        self.history_worker.stop()
        logger.info("runtime wiring stopped")
