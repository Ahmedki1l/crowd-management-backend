"""Camera registry (HLD 9). Credentials are encrypted at rest (HLD 5.8 / 14)."""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Camera(Base, TimestampMixin):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    area: Mapped[str] = mapped_column(String(128), index=True)
    ip: Mapped[str] = mapped_column(String(64))
    port: Mapped[int] = mapped_column(Integer, default=554)
    username: Mapped[str] = mapped_column(String(128))
    # AES-256-GCM ciphertext of the RTSP password; key held outside the DB.
    password_encrypted: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    roles: Mapped[list[str]] = mapped_column(JSON, default=list)
    stream_channel_sub: Mapped[int] = mapped_column(Integer, default=102)
    stream_channel_main: Mapped[int] = mapped_column(Integer, default=101)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    zones: Mapped[list[Zone]] = relationship(  # noqa: F821
        back_populates="camera", cascade="all, delete-orphan"
    )
    lines: Mapped[list[Line]] = relationship(  # noqa: F821
        back_populates="camera", cascade="all, delete-orphan"
    )
