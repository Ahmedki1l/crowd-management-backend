"""Wire the analytics event bus to its consumers (HLD 6.6).

A single place that attaches the read-model projector, the persistence projector,
and (when configured) the Digital Twin push publisher to the event bus. Used by
**both** entry paths so the wiring never drifts or gets duplicated:

* the FastAPI lifespan (``app.api.app``) in ``--api`` deployments;
* the worker entry points (``app.main`` ``--workers`` / ``--worker``), which have
  no FastAPI lifespan and would otherwise publish events that nothing consumes.

``aclose`` is for async callers (the lifespan); ``close_sync`` is for the plain
worker processes.
"""

from __future__ import annotations

from app.config.schema import AppConfig
from app.events.event_bus import InMemoryEventBus
from app.publish.dt_client import DigitalTwinPublisher
from app.services.projectors import PersistenceProjector, StateProjector
from app.services.state_store import StateStore
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RuntimeWiring:
    """Attach the bus consumers for a process and tear them down cleanly."""

    def __init__(self, bus: InMemoryEventBus, store: StateStore, settings: AppConfig) -> None:
        self._bus = bus
        self.state_projector = StateProjector(store)
        self.persistence_projector = PersistenceProjector()
        self.dt_publisher: DigitalTwinPublisher | None = None
        if settings.digital_twin.push_enabled and settings.digital_twin.push_url:
            self.dt_publisher = DigitalTwinPublisher(settings.digital_twin)

    def start(self) -> None:
        """Start background workers and subscribe every consumer to the bus."""
        self.persistence_projector.start()
        self._bus.subscribe_sync(self.state_projector.handle)
        self._bus.subscribe_sync(self.persistence_projector.handle)
        if self.dt_publisher is not None:
            self.dt_publisher.start()
            self._bus.subscribe_sync(self.dt_publisher.handle)
        logger.info(
            "runtime wiring started (dt_push=%s)", self.dt_publisher is not None
        )

    def _unsubscribe(self) -> None:
        self._bus.unsubscribe_sync(self.state_projector.handle)
        self._bus.unsubscribe_sync(self.persistence_projector.handle)
        if self.dt_publisher is not None:
            self._bus.unsubscribe_sync(self.dt_publisher.handle)

    async def aclose(self) -> None:
        """Async teardown for the FastAPI lifespan."""
        self._unsubscribe()
        self.persistence_projector.stop()
        if self.dt_publisher is not None:
            await self.dt_publisher.aclose()
        logger.info("runtime wiring stopped")

    def close_sync(self) -> None:
        """Synchronous teardown for plain worker processes (no event loop)."""
        import asyncio

        self._unsubscribe()
        self.persistence_projector.stop()
        if self.dt_publisher is not None:
            asyncio.run(self.dt_publisher.aclose())
        logger.info("runtime wiring stopped")
