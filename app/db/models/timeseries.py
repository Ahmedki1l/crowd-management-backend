"""Time-series tables (HLD 9). On SQL Server these are partitioned by ``ts``
(see migrations); the ORM definition stays dialect-agnostic."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class OccupancySample(Base):
    __tablename__ = "occupancy_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id", ondelete="CASCADE"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    count: Mapped[int] = mapped_column(Integer)

    __table_args__ = (Index("ix_occupancy_samples_zone_ts", "zone_id", "ts"),)


class CrossingEvent(Base):
    __tablename__ = "crossing_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    line_id: Mapped[int] = mapped_column(ForeignKey("lines.id", ondelete="CASCADE"))
    area_id: Mapped[str] = mapped_column(String(128), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    direction: Mapped[str] = mapped_column(String(8))  # in | out
    track_ref: Mapped[int] = mapped_column(Integer)

    __table_args__ = (Index("ix_crossing_events_area_ts", "area_id", "ts"),)
