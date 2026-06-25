"""Evidence-snapshot data access (HLD 6.5, 9).

Records the storage path of a captured snapshot frame for a camera. Repositories
flush but never commit.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models.alerts import Snapshot


class SnapshotRepository:
    """Access to the ``snapshots`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, camera_id: int, ts: datetime, path: str) -> Snapshot:
        """Insert a snapshot record and return the row."""
        snapshot = Snapshot(camera_id=camera_id, ts=ts, path=path)
        self._session.add(snapshot)
        self._session.flush()
        return snapshot
