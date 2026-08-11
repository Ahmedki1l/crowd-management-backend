"""Camera registry data access (HLD 6.5, 9).

Plain CRUD over the :class:`Camera` ORM row. The encrypted RTSP password
(``password_encrypted``) is owned by the camera service, which encrypts the
plaintext before handing the row to this repository — it is never set here.
Repositories flush but never commit; ``session_scope`` / the request lifecycle
owns the transaction.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.camera import Camera


class CameraRepository:
    """CRUD access to the ``cameras`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, data: dict[str, Any]) -> Camera:
        """Insert a new camera from a column->value mapping and return it.

        ``data`` must not contain ``password_encrypted``; the service layer sets
        that separately after encrypting the plaintext credential.
        """
        camera = Camera(**data)
        self._session.add(camera)
        self._session.flush()
        return camera

    def get(self, id: int) -> Camera | None:
        """Return the camera with ``id`` or ``None`` if it does not exist."""
        return self._session.get(Camera, id)

    def get_by_name(self, name: str) -> Camera | None:
        """Return the camera with the unique ``name`` or ``None``."""
        return self._session.scalars(
            select(Camera).where(Camera.name == name)
        ).first()

    def list_by_ip(self, ip: str) -> list[Camera]:
        """Return all cameras with ``ip``, ordered by id.

        The current schema does not enforce IP uniqueness, so callers that
        require exactly one match must reject an ambiguous result explicitly.
        """
        return list(
            self._session.scalars(
                select(Camera).where(Camera.ip == ip).order_by(Camera.id)
            )
        )

    def list(self) -> list[Camera]:
        """Return all cameras ordered by id."""
        return list(self._session.scalars(select(Camera).order_by(Camera.id)))

    def list_enabled(self) -> list[Camera]:
        """Return all enabled cameras ordered by id."""
        return list(
            self._session.scalars(
                select(Camera).where(Camera.enabled.is_(True)).order_by(Camera.id)
            )
        )

    def list_enabled_by_ip(self, ip: str) -> list[Camera]:
        """Return enabled cameras with the exact IP, ordered by id."""
        return list(
            self._session.scalars(
                select(Camera)
                .where(Camera.enabled.is_(True), Camera.ip == ip)
                .order_by(Camera.id)
            )
        )

    def update(self, id: int, data: dict[str, Any]) -> Camera | None:
        """Apply ``data`` (column->value) to the camera and return it, or ``None``.

        Only keys present in ``data`` are touched, so partial updates are safe.
        """
        camera = self.get(id)
        if camera is None:
            return None
        for key, value in data.items():
            setattr(camera, key, value)
        self._session.flush()
        return camera

    def delete(self, id: int) -> bool:
        """Delete the camera (cascading to its zones/lines). Return ``True`` if removed."""
        camera = self.get(id)
        if camera is None:
            return False
        self._session.delete(camera)
        self._session.flush()
        return True
