"""Zone geometry data access (HLD 6.5, 9).

CRUD over the :class:`Zone` ORM row. Polygons are stored as JSON
``[[x, y], ...]``; decoding to domain :class:`~app.domain.models.ZoneSpec`
happens in :mod:`app.db.repositories.mappers`, not here. Repositories flush but
never commit.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.geometry import Zone


class ZoneRepository:
    """CRUD access to the ``zones`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, data: dict[str, Any]) -> Zone:
        """Insert a new zone from a column->value mapping and return it."""
        zone = Zone(**data)
        self._session.add(zone)
        self._session.flush()
        return zone

    def get(self, id: int) -> Zone | None:
        """Return the zone with ``id`` or ``None`` if it does not exist."""
        return self._session.get(Zone, id)

    def list(self) -> list[Zone]:
        """Return all zones ordered by id."""
        return list(self._session.scalars(select(Zone).order_by(Zone.id)))

    def list_by_camera(self, camera_id: int) -> list[Zone]:
        """Return all zones belonging to ``camera_id`` ordered by id."""
        return list(
            self._session.scalars(
                select(Zone).where(Zone.camera_id == camera_id).order_by(Zone.id)
            )
        )

    def update(self, id: int, data: dict[str, Any]) -> Zone | None:
        """Apply ``data`` (column->value) to the zone and return it, or ``None``."""
        zone = self.get(id)
        if zone is None:
            return None
        for key, value in data.items():
            setattr(zone, key, value)
        self._session.flush()
        return zone

    def delete(self, id: int) -> bool:
        """Delete the zone. Return ``True`` if a row was removed."""
        zone = self.get(id)
        if zone is None:
            return False
        self._session.delete(zone)
        self._session.flush()
        return True
