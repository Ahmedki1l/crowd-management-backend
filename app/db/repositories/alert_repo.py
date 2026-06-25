"""Safety-alert data access (HLD 6.5, 9).

CRUD plus status transitions over the :class:`Alert` ORM row. ``type`` and
``status`` are the wire strings (matching
:class:`~app.domain.models.AlertType` and the ``active|acknowledged|resolved``
lifecycle). Repositories flush but never commit.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.alerts import Alert


class AlertRepository:
    """Access to the ``alerts`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        type: str,
        zone_id: int | None,
        camera_id: int | None,
        ts: datetime,
        detail: str,
        snapshot_url: str | None,
        status: str = "active",
    ) -> Alert:
        """Insert a new alert and return the row."""
        alert = Alert(
            type=type,
            zone_id=zone_id,
            camera_id=camera_id,
            ts=ts,
            detail=detail,
            snapshot_url=snapshot_url,
            status=status,
        )
        self._session.add(alert)
        self._session.flush()
        return alert

    def get(self, id: int) -> Alert | None:
        """Return the alert with ``id`` or ``None`` if it does not exist."""
        return self._session.get(Alert, id)

    def list(
        self,
        status: str | None = None,
        type: str | None = None,
        zone_id: int | None = None,
    ) -> list[Alert]:
        """List alerts, newest first, filtered by any supplied criteria.

        A ``None`` filter is not applied, so ``list()`` returns every alert.
        """
        stmt = select(Alert)
        if status is not None:
            stmt = stmt.where(Alert.status == status)
        if type is not None:
            stmt = stmt.where(Alert.type == type)
        if zone_id is not None:
            stmt = stmt.where(Alert.zone_id == zone_id)
        stmt = stmt.order_by(Alert.ts.desc(), Alert.id.desc())
        return list(self._session.scalars(stmt))

    def set_status(
        self,
        id: int,
        status: str,
        acknowledged_at: datetime | None,
    ) -> Alert | None:
        """Transition an alert's ``status`` and return it, or ``None`` if missing.

        ``acknowledged_at`` is stored as given so the caller controls whether a
        transition stamps the acknowledgement time.
        """
        alert = self.get(id)
        if alert is None:
            return None
        alert.status = status
        alert.acknowledged_at = acknowledged_at
        self._session.flush()
        return alert

    def history(self, zone_id: int) -> list[Alert]:
        """Return all alerts for ``zone_id``, newest first."""
        return list(
            self._session.scalars(
                select(Alert)
                .where(Alert.zone_id == zone_id)
                .order_by(Alert.ts.desc(), Alert.id.desc())
            )
        )
