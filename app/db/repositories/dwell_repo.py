"""Dwell-session data access (HLD 6.5, 9).

Stores closed dwell sessions (a track's enter/leave span inside a zone) and
answers the average-dwell history query. Bucketed by ``leave_ts`` — a session
contributes to the window in which it completed. Repositories flush but never
commit.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.timeseries import DwellSession
from app.db.repositories._bucketing import group_by_bucket


class DwellRepository:
    """Access to the ``dwell_sessions`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_closed(
        self,
        zone_id: int,
        track_ref: int,
        enter_ts: datetime,
        leave_ts: datetime,
        dwell_s: float,
    ) -> DwellSession:
        """Persist one completed dwell session and return the row."""
        session_row = DwellSession(
            zone_id=zone_id,
            track_ref=track_ref,
            enter_ts=enter_ts,
            leave_ts=leave_ts,
            dwell_s=dwell_s,
        )
        self._session.add(session_row)
        self._session.flush()
        return session_row

    def query_series(
        self,
        zone_id: int,
        frm: datetime,
        to: datetime,
        bucket_seconds: int,
    ) -> list[tuple[datetime, float]]:
        """Return ``(bucket_start, avg_dwell_s)`` pairs over ``[frm, to)``.

        Sessions are bucketed by ``leave_ts`` (completion time). The value is the
        mean ``dwell_s`` of sessions in the bucket. Buckets with no completed
        sessions are omitted.
        """
        rows = self._session.scalars(
            select(DwellSession)
            .where(
                DwellSession.zone_id == zone_id,
                DwellSession.leave_ts.is_not(None),
                DwellSession.leave_ts >= frm,
                DwellSession.leave_ts < to,
            )
            .order_by(DwellSession.leave_ts)
        ).all()

        series: list[tuple[datetime, float]] = []
        for bucket_start, bucket_rows in group_by_bucket(
            rows, lambda r: r.leave_ts, bucket_seconds
        ):
            durations = [r.dwell_s for r in bucket_rows if r.dwell_s is not None]
            if not durations:
                continue
            series.append((bucket_start, sum(durations) / len(durations)))
        return series
