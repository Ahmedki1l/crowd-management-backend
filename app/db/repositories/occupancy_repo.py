"""Occupancy-rollup data access (HLD 6.5, 9).

Reads and writes the two pre-aggregated tables (``occupancy_minute``,
``occupancy_hour``). There is no raw-sample table: history is aggregated at the source
by :mod:`app.services.occupancy_sampler`, so a query never scans raw rows and never
buckets in Python — it reads the grain it was asked for.

Every write is an **upsert on (space_id, bucket_ts)**. That is what makes the sampler and
the rollup worker safe to re-run: a restart, a catch-up over an outage, or a second
process cannot duplicate a bucket. Repositories flush but never commit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.timeseries import (
    OccupancyHour,
    OccupancyMinute,
    OccupancyRollupBase,
)


class Grain(str, Enum):
    """The two aggregation grains that are actually stored.

    A typed grain rather than a bare string: the repository dispatches on it, and a typo
    would otherwise be a runtime KeyError deep in a background thread. ``str, Enum``
    matches the project's enum convention (see CLAUDE.md).
    """

    MINUTE = "minute"
    HOUR = "hour"


# The row type each grain maps to. Typed as the union rather than left to inference:
# mypy would otherwise widen it to their common base and lose every column.
RollupModel = type[OccupancyMinute] | type[OccupancyHour]
_MODELS: dict[Grain, RollupModel] = {
    Grain.MINUTE: OccupancyMinute,
    Grain.HOUR: OccupancyHour,
}


@dataclass(frozen=True, slots=True)
class RollupRow:
    """One aggregated bucket, independent of which table it came from."""

    space_id: str
    floor: str | None
    bucket_ts: datetime
    avg: float
    peak: int
    min: int
    samples: int
    cameras_healthy: int
    cameras_total: int


class OccupancyRepository:
    """Access to the occupancy rollup tables."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def upsert(self, grain: Grain, row: RollupRow) -> None:
        """Insert ``row``, or overwrite the bucket if one is already there.

        Overwrite rather than skip is deliberate: a re-run means the bucket was
        recomputed from more complete data, so the newer value is the better one.

        Args:
            grain: Which rollup table to write.
            row: The aggregated bucket to store.
        """
        model = _MODELS[grain]
        existing = cast(
            "OccupancyRollupBase | None",
            self._session.scalars(
                select(model).where(
                    model.space_id == row.space_id, model.bucket_ts == row.bucket_ts
                )
            ).first(),
        )

        if existing is None:
            self._session.add(
                model(
                    space_id=row.space_id,
                    floor=row.floor,
                    bucket_ts=row.bucket_ts,
                    avg=row.avg,
                    peak=row.peak,
                    min=row.min,
                    samples=row.samples,
                    cameras_healthy=row.cameras_healthy,
                    cameras_total=row.cameras_total,
                )
            )
        else:
            existing.floor = row.floor
            existing.avg = row.avg
            existing.peak = row.peak
            existing.min = row.min
            existing.samples = row.samples
            existing.cameras_healthy = row.cameras_healthy
            existing.cameras_total = row.cameras_total
        self._session.flush()

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def query(
        self,
        grain: Grain,
        frm: datetime,
        to: datetime,
        space_ids: list[str] | None = None,
        floors: list[str] | None = None,
        limit: int | None = None,
    ) -> list[RollupRow]:
        """Return buckets in ``[frm, to)``, oldest first.

        Args:
            grain: Which rollup table to read.
            frm: Inclusive window start (timezone-aware UTC).
            to: Exclusive window end (timezone-aware UTC).
            space_ids: Restrict to these spaces; ``None`` means every space.
            floors: Restrict to these floors; ``None`` means every floor.
            limit: Hard cap on rows returned, so an absurd window cannot exhaust memory.
        """
        model = _MODELS[grain]
        stmt = select(model).where(model.bucket_ts >= frm, model.bucket_ts < to)
        if space_ids:
            stmt = stmt.where(model.space_id.in_(space_ids))
        if floors:
            stmt = stmt.where(model.floor.in_(floors))
        stmt = stmt.order_by(model.bucket_ts, model.space_id)
        if limit is not None:
            stmt = stmt.limit(limit)

        rows = cast("Sequence[OccupancyRollupBase]", self._session.scalars(stmt).all())
        return [
            RollupRow(
                space_id=r.space_id,
                floor=r.floor,
                bucket_ts=r.bucket_ts,
                avg=r.avg,
                peak=r.peak,
                min=r.min,
                samples=r.samples,
                cameras_healthy=r.cameras_healthy,
                cameras_total=r.cameras_total,
            )
            for r in rows
        ]

    def latest_bucket(self, grain: Grain) -> datetime | None:
        """Return the newest bucket present, or ``None`` when the table is empty.

        This is the rollup worker's watermark. Deriving it from the data rather than
        keeping a separate cursor means there is no state to get out of sync: after an
        outage the worker simply resumes from the last bucket it actually wrote.
        """
        model = _MODELS[grain]
        return self._session.scalars(
            select(model.bucket_ts).order_by(model.bucket_ts.desc()).limit(1)
        ).first()

    def delete_before(self, grain: Grain, cutoff: datetime) -> int:
        """Delete buckets starting before ``cutoff``. Returns the number removed."""
        model = _MODELS[grain]
        rows = cast(
            "Sequence[OccupancyRollupBase]",
            self._session.scalars(select(model).where(model.bucket_ts < cutoff)).all(),
        )
        for row in rows:
            self._session.delete(row)
        self._session.flush()
        return len(rows)
