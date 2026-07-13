"""Line-crossing event data access (HLD 6.5, 9).

Stores directional crossing events per counting line and answers per-area
entry/exit history and running totals. ``direction`` is the wire string ``"in"``
or ``"out"`` (matching :class:`~app.domain.models.CrossingDirection`).
Repositories flush but never commit.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models.timeseries import CrossingEvent
from app.db.repositories._bucketing import group_by_bucket
from app.domain.models import CrossingDirection


class CrossingRepository:
    """Access to the ``crossing_events`` time-series table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        line_id: int,
        area_id: str,
        ts: datetime,
        direction: str,
        track_ref: int,
    ) -> CrossingEvent:
        """Append one crossing event and return the row."""
        event = CrossingEvent(
            line_id=line_id,
            area_id=area_id,
            ts=ts,
            direction=direction,
            track_ref=track_ref,
        )
        self._session.add(event)
        self._session.flush()
        return event

    def query_series(
        self,
        area_id: str,
        frm: datetime,
        to: datetime,
        bucket_seconds: int,
    ) -> list[tuple[datetime, dict[str, int]]]:
        """Return ``(bucket_start, {"in", "out", "net"})`` pairs over ``[frm, to)``.

        ``net`` is ``in - out``. Buckets with no events are omitted.
        """
        rows = self._session.scalars(
            select(CrossingEvent)
            .where(
                CrossingEvent.area_id == area_id,
                CrossingEvent.ts >= frm,
                CrossingEvent.ts < to,
            )
            .order_by(CrossingEvent.ts)
        ).all()

        series: list[tuple[datetime, dict[str, int]]] = []
        for bucket_start, bucket_rows in group_by_bucket(
            rows, lambda r: r.ts, bucket_seconds
        ):
            in_count = sum(1 for r in bucket_rows if r.direction == CrossingDirection.IN.value)
            out_count = sum(1 for r in bucket_rows if r.direction == CrossingDirection.OUT.value)
            series.append(
                (bucket_start, {"in": in_count, "out": out_count, "net": in_count - out_count})
            )
        return series

    def totals(
        self,
        area_id: str,
        frm: datetime | None = None,
        to: datetime | None = None,
    ) -> dict[str, int]:
        """Return ``{"in", "out", "net"}`` counts for ``area_id``.

        With no bounds the counts are all-time; passing ``frm`` (inclusive) and/or
        ``to`` (exclusive) restricts them to a ``[frm, to)`` window — e.g. a single
        calendar day. ``net`` is ``in - out``.
        """
        stmt = select(CrossingEvent.direction, func.count()).where(
            CrossingEvent.area_id == area_id
        )
        if frm is not None:
            stmt = stmt.where(CrossingEvent.ts >= frm)
        if to is not None:
            stmt = stmt.where(CrossingEvent.ts < to)
        counts = dict(
            self._session.execute(stmt.group_by(CrossingEvent.direction)).all()
        )
        in_count = int(counts.get(CrossingDirection.IN.value, 0))
        out_count = int(counts.get(CrossingDirection.OUT.value, 0))
        return {"in": in_count, "out": out_count, "net": in_count - out_count}

    def delete_before(self, cutoff: datetime) -> int:
        """Delete crossings recorded before ``cutoff``. Returns the number removed."""
        rows = self._session.scalars(
            select(CrossingEvent).where(CrossingEvent.ts < cutoff)
        ).all()
        for row in rows:
            self._session.delete(row)
        self._session.flush()
        return len(rows)
