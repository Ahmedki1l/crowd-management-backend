"""drop heatmap_grid — the heat-map metric is out of scope

The delivered scope is occupancy (with entry/exit paused); heat maps were part of
the original five-metric design and are removed (see
``docs/OCCUPANCY_ONLY_REMOVAL_PLAN.md``).

The table was also the single largest consumer of storage while being the least
useful: 4.1 MB of an 8.0 MB SQLite file, of which 76% of rows held an empty grid
(``{}``) — one row per camera per minute regardless of whether anything was
detected — and nothing ever pruned them, because ``RetentionConfig`` is read by no
code.

``downgrade`` recreates the table but cannot restore its rows. That is intentional
and safe: the data has no remaining reader.

Revision ID: 0004_drop_heatmap_grid
Revises: 0003_camera_imgsz
Create Date: 2026-07-13
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_drop_heatmap_grid"
down_revision: str | None = "0003_camera_imgsz"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_heatmap_grid_camera_bucket", table_name="heatmap_grid")
    op.drop_table("heatmap_grid")


def downgrade() -> None:
    op.create_table(
        "heatmap_grid",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=False),
        sa.Column("ts_bucket", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cols", sa.Integer(), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("grid", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_heatmap_grid_camera_bucket", "heatmap_grid", ["camera_id", "ts_bucket"]
    )
