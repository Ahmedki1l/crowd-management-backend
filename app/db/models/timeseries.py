"""Time-series tables (HLD 9). The ORM definition stays dialect-agnostic.

Occupancy history is stored as **pre-aggregated rollups keyed by logical space**, not
as raw per-zone samples. Two decisions drove that, both learned the hard way:

* **Space, not zone.** Zone rows are volatile — polygons get redrawn, which deletes the
  zone and (with SQLite's ``foreign_keys`` pragma off) silently orphans its history.
  77% of the old ``occupancy_samples`` table pointed at zone ids that no longer existed.
  ``dt_space_id`` ("b1-waiting-area") survives a redraw; a zone id does not.
* **Dimensions are denormalised onto the row.** ``floor`` is copied in at write time
  rather than joined through ``zone -> camera`` at read time, so re-assigning a camera's
  floor cannot rewrite the past. A history row must be self-describing.

``samples`` is the coverage metric: the sampler ticks at 1 Hz, so a full minute is 60 and
a full hour is 3600. Fewer means the pipeline was dropping frames or a camera was down —
which is exactly what lets a reader tell "the space was empty" from "we could not see it".
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class OccupancyRollupBase:
    """Columns shared by the minute and hour rollups (identical shape, different grain).

    Public because the repository annotates against it: the two tables are selected
    dynamically by grain, and this is the type that actually carries the columns.
    """

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # The logical Digital-Twin space this row describes. Not a foreign key: it must
    # outlive the zones that fed it.
    space_id: Mapped[str] = mapped_column(String(128))
    # Denormalised so a floor re-assignment cannot rewrite history. NULL when the
    # space's cameras carry no floor.
    floor: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Start of the bucket, UTC, aligned to the grain (minute or hour).
    bucket_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Mean occupancy over the bucket. Correct as a plain sum/count because the sampler
    # takes evenly-spaced 1 Hz samples — see app.services.occupancy_sampler.
    avg: Mapped[float] = mapped_column(Float)
    # The safety-relevant number: an average actively hides spikes.
    peak: Mapped[int] = mapped_column(Integer)
    min: Mapped[int] = mapped_column(Integer)
    # Samples that fed this bucket. 60 = a fully-covered minute, 3600 = a full hour.
    samples: Mapped[int] = mapped_column(Integer)
    # How many of the space's cameras were healthy while it was measured. A number
    # summed from fewer cameras than usual is not the same number.
    cameras_healthy: Mapped[int] = mapped_column(Integer, default=0)
    cameras_total: Mapped[int] = mapped_column(Integer, default=0)


class OccupancyMinute(OccupancyRollupBase, Base):
    """One row per space per minute. The durable record; the hour rollup derives from it."""

    __tablename__ = "occupancy_minute"

    __table_args__ = (
        # Idempotency: the sampler and the rollup worker upsert on this, so a restart,
        # a catch-up, or a second process cannot duplicate a bucket.
        UniqueConstraint("space_id", "bucket_ts", name="uq_occupancy_minute_space_bucket"),
        Index("ix_occupancy_minute_space_bucket", "space_id", "bucket_ts"),
        Index("ix_occupancy_minute_bucket", "bucket_ts"),
    )


class OccupancyHour(OccupancyRollupBase, Base):
    """One row per space per hour. This is what the Digital Twin reads."""

    __tablename__ = "occupancy_hour"

    __table_args__ = (
        UniqueConstraint("space_id", "bucket_ts", name="uq_occupancy_hour_space_bucket"),
        Index("ix_occupancy_hour_space_bucket", "space_id", "bucket_ts"),
        Index("ix_occupancy_hour_floor_bucket", "floor", "bucket_ts"),
        Index("ix_occupancy_hour_bucket", "bucket_ts"),
    )


class CrossingEvent(Base):
    __tablename__ = "crossing_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    line_id: Mapped[int] = mapped_column(ForeignKey("lines.id", ondelete="CASCADE"))
    area_id: Mapped[str] = mapped_column(String(128), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    direction: Mapped[str] = mapped_column(String(8))  # in | out
    track_ref: Mapped[int] = mapped_column(Integer)

    __table_args__ = (Index("ix_crossing_events_area_ts", "area_id", "ts"),)
