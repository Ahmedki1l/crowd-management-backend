"""Occupancy-sample data access (HLD 6.5, 9).

Stores per-zone instantaneous people counts and answers the live "latest count"
and bucketed-history queries. Bucketing is done in Python (see
:mod:`app.db.repositories._bucketing`) so the query stays dialect-agnostic.
Repositories flush but never commit.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.timeseries import OccupancySample
from app.db.repositories._bucketing import group_by_bucket


class OccupancyRepository:
    """Access to the ``occupancy_samples`` time-series table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_sample(self, zone_id: int, ts: datetime, count: int) -> OccupancySample:
        """Append one occupancy sample for ``zone_id`` and return the row."""
        sample = OccupancySample(zone_id=zone_id, ts=ts, count=count)
        self._session.add(sample)
        self._session.flush()
        return sample

    def latest(self, zone_id: int) -> OccupancySample | None:
        """Return the most recent sample for ``zone_id`` or ``None``."""
        return self._session.scalars(
            select(OccupancySample)
            .where(OccupancySample.zone_id == zone_id)
            .order_by(OccupancySample.ts.desc())
            .limit(1)
        ).first()

    def query_series(
        self,
        zone_id: int,
        frm: datetime,
        to: datetime,
        bucket_seconds: int,
    ) -> list[tuple[datetime, float]]:
        """Return ``(bucket_start, avg_count)`` pairs over ``[frm, to)``.

        The value per bucket is the mean of the sample counts that fall in it.
        Buckets with no samples are omitted.
        """
        rows = self._session.scalars(
            select(OccupancySample)
            .where(
                OccupancySample.zone_id == zone_id,
                OccupancySample.ts >= frm,
                OccupancySample.ts < to,
            )
            .order_by(OccupancySample.ts)
        ).all()

        series: list[tuple[datetime, float]] = []
        for bucket_start, bucket_rows in group_by_bucket(
            rows, lambda r: r.ts, bucket_seconds
        ):
            avg = sum(r.count for r in bucket_rows) / len(bucket_rows)
            series.append((bucket_start, avg))
        return series
