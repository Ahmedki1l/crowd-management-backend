"""Counting-line geometry data access (HLD 6.5, 9).

CRUD over the :class:`Line` ORM row. Endpoints and the IN-direction normal are
stored as JSON; decoding to domain :class:`~app.domain.models.LineSpec` happens
in :mod:`app.db.repositories.mappers`, not here. Repositories flush but never
commit.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.geometry import Line


class LineRepository:
    """CRUD access to the ``lines`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, data: dict[str, Any]) -> Line:
        """Insert a new line from a column->value mapping and return it."""
        line = Line(**data)
        self._session.add(line)
        self._session.flush()
        return line

    def get(self, id: int) -> Line | None:
        """Return the line with ``id`` or ``None`` if it does not exist."""
        return self._session.get(Line, id)

    def list(self) -> list[Line]:
        """Return all lines ordered by id."""
        return list(self._session.scalars(select(Line).order_by(Line.id)))

    def list_by_camera(self, camera_id: int) -> list[Line]:
        """Return all lines belonging to ``camera_id`` ordered by id."""
        return list(
            self._session.scalars(
                select(Line).where(Line.camera_id == camera_id).order_by(Line.id)
            )
        )

    def update(self, id: int, data: dict[str, Any]) -> Line | None:
        """Apply ``data`` (column->value) to the line and return it, or ``None``."""
        line = self.get(id)
        if line is None:
            return None
        for key, value in data.items():
            setattr(line, key, value)
        self._session.flush()
        return line

    def delete(self, id: int) -> bool:
        """Delete the line. Return ``True`` if a row was removed."""
        line = self.get(id)
        if line is None:
            return False
        self._session.delete(line)
        self._session.flush()
        return True
