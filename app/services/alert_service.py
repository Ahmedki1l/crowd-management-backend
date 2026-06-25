"""Alert read/write service backing the alert router and engine (HLD 8.2, 8.3).

Two callers share this service:

* the analytics engine raises alerts here (:meth:`AlertService.raise_alert`),
  which persists the row and hands the ORM object back so the engine can attach
  the generated id to its outbound :class:`~app.events.events.AlertRaised` event;
* the API reads and transitions alerts, receiving the pure
  :class:`~app.api.schemas.metrics.AlertOut` DTO.

The ``active -> acknowledged -> resolved`` lifecycle and ``acknowledged_at``
stamping live here; the repository only performs the storage transition.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.api.schemas.metrics import AlertOut
from app.db.models.alerts import Alert
from app.db.repositories.alert_repo import AlertRepository
from app.events.events import AlertRaised
from app.utils.logging import get_logger
from app.utils.timeutil import to_datetime

logger = get_logger(__name__)

STATUS_ACKNOWLEDGED = "acknowledged"
STATUS_RESOLVED = "resolved"


class AlertService:
    """Read/write service for safety alerts."""

    def __init__(self, session: Session) -> None:
        """Bind the service to a DB session.

        Args:
            session: An open SQLAlchemy session owning the alert transaction.
        """
        self._session = session
        self._alert_repo = AlertRepository(session)

    def raise_alert(self, event: AlertRaised, snapshot_url: str | None) -> Alert:
        """Persist an alert from an engine event and return the stored row.

        The engine emits float epoch ``event.ts``; it is converted to the
        timezone-aware DB timestamp at this persistence boundary. The returned
        row carries the generated id the engine echoes back to subscribers.

        Args:
            event: The raised-alert event produced by the analytics engine.
            snapshot_url: URL of the captured evidence frame, if any. Falls back
                to the URL already carried on the event when not supplied.

        Returns:
            The persisted :class:`~app.db.models.alerts.Alert` row, with id.
        """
        url = snapshot_url if snapshot_url is not None else event.snapshot_url
        alert = self._alert_repo.create(
            type=event.alert_type.value,
            zone_id=event.zone_id,
            camera_id=event.camera_id,
            ts=to_datetime(event.ts),
            detail=event.detail,
            snapshot_url=url,
        )
        logger.info(
            "alert raised",
            extra={
                "zone_id": event.zone_id,
                "camera_id": event.camera_id,
                "event": event.alert_type.value,
            },
        )
        return alert

    def list_alerts(
        self,
        status: str | None = None,
        type: str | None = None,
        zone_id: int | None = None,
    ) -> list[AlertOut]:
        """List alerts (newest first) filtered by any supplied criteria.

        Args:
            status: Restrict to this lifecycle status when given.
            type: Restrict to this alert type when given.
            zone_id: Restrict to this zone when given.

        Returns:
            The matching alerts as response DTOs.
        """
        rows = self._alert_repo.list(status=status, type=type, zone_id=zone_id)
        return [AlertOut.model_validate(row) for row in rows]

    def ack_alert(
        self,
        id: int,
        resolve: bool,
        now: datetime,
    ) -> AlertOut | None:
        """Acknowledge or resolve an alert and stamp its acknowledgement time.

        Args:
            id: The alert id to transition.
            resolve: When ``True`` move straight to ``resolved``; otherwise mark
                it ``acknowledged``.
            now: The acknowledgement timestamp to record (timezone-aware UTC).

        Returns:
            The updated alert as a response DTO, or ``None`` if no alert with
            ``id`` exists.
        """
        new_status = STATUS_RESOLVED if resolve else STATUS_ACKNOWLEDGED
        row = self._alert_repo.set_status(id, new_status, acknowledged_at=now)
        if row is None:
            return None
        return AlertOut.model_validate(row)

    def history(self, zone_id: int) -> list[AlertOut]:
        """Return every alert for ``zone_id``, newest first.

        Args:
            zone_id: The zone whose alert history to return.

        Returns:
            The zone's alerts as response DTOs.
        """
        rows = self._alert_repo.history(zone_id)
        return [AlertOut.model_validate(row) for row in rows]
