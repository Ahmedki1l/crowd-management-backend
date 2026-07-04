"""add cameras.floor for per-floor occupancy rollup (HLD 9)

Adds a nullable ``floor`` column to ``cameras`` so zones/spaces can be grouped by
physical floor (``GET /api/v1/occupancy/floors``). Nullable so existing rows
remain valid until a floor is assigned.

Revision ID: 0002_camera_floor
Revises: 0001_initial
Create Date: 2026-07-04
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_camera_floor"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("cameras", sa.Column("floor", sa.String(length=128), nullable=True))
    op.create_index("ix_cameras_floor", "cameras", ["floor"])


def downgrade() -> None:
    op.drop_index("ix_cameras_floor", table_name="cameras")
    op.drop_column("cameras", "floor")
