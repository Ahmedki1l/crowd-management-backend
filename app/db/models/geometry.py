"""Zones and counting lines drawn in image space (HLD 9)."""

from __future__ import annotations

from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Zone(Base, TimestampMixin):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128))
    type: Mapped[str] = mapped_column(String(32))  # occupancy | waiting | restricted
    polygon: Mapped[list[list[float]]] = mapped_column(JSON)  # [[x,y], ...]
    safe_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dt_space_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    camera: Mapped[Camera] = relationship(back_populates="zones")  # noqa: F821


class Line(Base, TimestampMixin):
    __tablename__ = "lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128))
    points: Mapped[list[list[float]]] = mapped_column(JSON)  # [[x1,y1],[x2,y2]]
    in_direction: Mapped[list[float]] = mapped_column(JSON)  # [dx,dy] reference IN normal
    area_id: Mapped[str] = mapped_column(String(128), index=True)
    dt_space_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    camera: Mapped[Camera] = relationship(back_populates="lines")  # noqa: F821
