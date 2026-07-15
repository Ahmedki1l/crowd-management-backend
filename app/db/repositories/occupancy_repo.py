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
from datetime import UTC, datetime
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


class ResultTooLargeError(Exception):
    """A history query would return more rows than the caller's cap allows.

    Raised instead of silently truncating: a truncated time series is worse than an
    error, because the missing tail reads as "the data ends here". The router turns this
    into a 422 telling the caller to narrow the window or coarsen the bucket.
    """

    def __init__(self, limit: int) -> None:
        super().__init__(f"result exceeds {limit} rows")
        self.limit = limit


def _as_utc(ts: datetime) -> datetime:
    """Re-attach UTC to a bucket timestamp read back from the DB.

    Every ``bucket_ts`` is written as timezone-aware UTC (the sampler stamps
    ``datetime.now(tz=UTC)``), but SQLite has no native timezone type and returns the
    value **naive**. A naive datetime serializes to ISO-8601 with no offset, which a
    client parses as *local* time — so a 10:00 UTC bucket would render at 10:00 in the
    viewer's zone, hours off. Normalising here, at the DB boundary, guarantees every
    consumer (the API and the rollup watermark) sees UTC.
    """
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


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
    def upsert(self, grain: Grain, row: RollupRow, *, merge: bool = False) -> None:
        """Insert ``row``, or combine it with an existing bucket for the same key.

        Two callers need opposite semantics for an existing row:

        * The hourly rollup **overwrites** (``merge=False``): it recomputes an hour from
          all of its minutes, so its value is authoritative and replaces whatever partial
          value a previous pass wrote.
        * The sampler **merges** (``merge=True``): a graceful restart mid-minute flushes a
          partial minute, and the fresh process then accumulates only the rest of that
          minute. Overwriting would discard the pre-restart samples; merging folds the two
          partials into the whole minute (samples add, mean re-weights, peak/min/coverage
          compose).

        Args:
            grain: Which rollup table to write.
            row: The aggregated bucket to store.
            merge: Combine with an existing bucket instead of replacing it.
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
        elif merge:
            combined = existing.samples + row.samples
            existing.avg = (
                (existing.avg * existing.samples + row.avg * row.samples) / combined
                if combined
                else row.avg
            )
            existing.peak = max(existing.peak, row.peak)
            existing.min = min(existing.min, row.min)
            existing.samples = combined
            existing.cameras_healthy = min(existing.cameras_healthy, row.cameras_healthy)
            existing.cameras_total = max(existing.cameras_total, row.cameras_total)
            existing.floor = row.floor
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
            space_ids: Restrict to these spaces; ``None`` or empty means every space.
            floors: Restrict to these floors; ``None`` or empty means every floor.
            limit: Cap on rows; exceeding it raises rather than truncating (see below).

        Raises:
            ResultTooLargeError: If more than ``limit`` rows match. Truncating a time
                series silently would drop its newest buckets and read as "ends here".
        """
        model = _MODELS[grain]
        stmt = select(model).where(model.bucket_ts >= frm, model.bucket_ts < to)
        # Drop blank values: ``?space_id=`` arrives as ``['']``, and ``.in_([''])`` would
        # match nothing — i.e. an empty filter would return NO rows instead of all.
        spaces = [s for s in space_ids if s] if space_ids else None
        floor_names = [f for f in floors if f] if floors else None
        if spaces:
            stmt = stmt.where(model.space_id.in_(spaces))
        if floor_names:
            stmt = stmt.where(model.floor.in_(floor_names))
        stmt = stmt.order_by(model.bucket_ts, model.space_id)
        if limit is not None:
            # Over-fetch one so an exact-limit result is distinguishable from an
            # over-limit one; the extra row means the window is genuinely too large.
            stmt = stmt.limit(limit + 1)

        rows = cast("Sequence[OccupancyRollupBase]", self._session.scalars(stmt).all())
        if limit is not None and len(rows) > limit:
            raise ResultTooLargeError(limit)
        return [
            RollupRow(
                space_id=r.space_id,
                floor=r.floor,
                bucket_ts=_as_utc(r.bucket_ts),
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
        latest = self._session.scalars(
            select(model.bucket_ts).order_by(model.bucket_ts.desc()).limit(1)
        ).first()
        return _as_utc(latest) if latest is not None else None

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
