"""add cameras.imgsz for per-camera detector input resolution (HLD 6.2)

Lets each camera override the global ``detector.imgsz`` (e.g. 1280 for a gate or
a lounge with distant people, 640 for close-range cameras to save CPU). Nullable:
unset means "use the global detector.imgsz". Requires a dynamic-shape detector
model so one model serves any imgsz.

Revision ID: 0003_camera_imgsz
Revises: 0002_camera_floor
Create Date: 2026-07-04
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_camera_imgsz"
down_revision: str | None = "0002_camera_floor"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("cameras", sa.Column("imgsz", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("cameras", "imgsz")
