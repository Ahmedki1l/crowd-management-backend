"""Wire the analytics event bus to its consumers (HLD 6.6).

A single place that attaches the read-model projector and the persistence projector
to the event bus. Used by **both** entry paths so the wiring never drifts or gets
duplicated:

* the FastAPI lifespan (``app.api.app``) in ``--api`` deployments;
* the worker entry points (``app.main`` ``--workers`` / ``--worker``), which have
  no FastAPI lifespan and would otherwise publish events that nothing consumes.

``aclose`` is for async callers (the lifespan); ``close_sync`` is for the plain
worker processes.
"""

from __future__ import annotations

from app.events.event_bus import InMemoryEventBus
from app.services.projectors import PersistenceProjector, StateProjector
from app.services.state_store import StateStore
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RuntimeWiring:
    """Attach the bus consumers for a process and tear them down cleanly."""

    def __init__(self, bus: InMemoryEventBus, store: StateStore) -> None:
        self._bus = bus
        self.state_projector = StateProjector(store)
        self.persistence_projector = PersistenceProjector()

    def start(self) -> None:
        """Start background workers and subscribe every consumer to the bus."""
        self.persistence_projector.start()
        self._bus.subscribe_sync(self.state_projector.handle)
        self._bus.subscribe_sync(self.persistence_projector.handle)
        logger.info("runtime wiring started")

    def _unsubscribe(self) -> None:
        self._bus.unsubscribe_sync(self.state_projector.handle)
        self._bus.unsubscribe_sync(self.persistence_projector.handle)

    async def aclose(self) -> None:
        """Async teardown for the FastAPI lifespan."""
        self.close_sync()

    def close_sync(self) -> None:
        """Unsubscribe every consumer and stop the persistence writer."""
        self._unsubscribe()
        self.persistence_projector.stop()
        logger.info("runtime wiring stopped")
