"""Dashboard summary service backing the stats router (HLD 8.2).

Builds the single :class:`~app.api.schemas.metrics.StatsOut` tile shown on the
operator dashboard. Static counts (registered cameras, configured zones) come
from the DB; live counts (healthy cameras, total people, active alerts) come
from the in-memory :class:`~app.services.state_store.StateStore` and the alert
table. Combining both sources is the point of this service.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.api.schemas.metrics import StatsOut
from app.db.repositories.camera_repo import CameraRepository
from app.db.repositories.zone_repo import ZoneRepository
from app.services.state_store import StateStore
from app.utils.clock import system_clock

ALERT_STATUS_ACTIVE = "active"


class StatsService:
    """Builds the consolidated dashboard summary."""

    def __init__(self, session: Session, store: StateStore) -> None:
        """Bind the service to a DB session and the live state store.

        Args:
            session: An open SQLAlchemy session used for the static counts.
            store: The current-state cache feeding the live counts.
        """
        self._session = session
        self._store = store
        self._camera_repo = CameraRepository(session)
        self._zone_repo = ZoneRepository(session)

    def build(self) -> StatsOut:
        """Assemble the current dashboard summary.

        Returns:
            A :class:`StatsOut` combining DB-registered totals with live health,
            occupancy counts, stamped with the wall clock.
        """
        cameras_total = len(self._camera_repo.list())
        zones_total = len(self._zone_repo.list())
        cameras_healthy = sum(
            1 for health in self._store.all_camera_health() if health.healthy
        )
        total_occupancy = self._store.total_occupancy()

        return StatsOut(
            cameras_total=cameras_total,
            cameras_healthy=cameras_healthy,
            zones_total=zones_total,
            total_occupancy=total_occupancy,
            ts=system_clock().now(),
        )
