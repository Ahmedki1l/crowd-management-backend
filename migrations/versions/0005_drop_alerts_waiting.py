"""drop alerts, snapshots and dwell_sessions — safety and waiting are out of scope

The delivered scope is occupancy, with entry/exit paused. Safety alerts and waiting
times were part of the original five-metric design and are removed (see
``docs/OCCUPANCY_ONLY_REMOVAL_PLAN.md``).

``snapshots`` goes with ``alerts`` because it only ever held alert evidence: rows were
written solely by the pipeline's alert path, and the two snapshot endpoints served
*stored* files. With no alerts, nothing would have written a row and both endpoints
would have returned 503 forever.

All three tables are empty in the deployment this was written against (0 alerts,
0 snapshots, 0 dwell sessions), so no data is lost. ``downgrade`` recreates the
schema but not the rows.

Revision ID: 0005_drop_alerts_waiting
Revises: 0004_drop_heatmap_grid
Create Date: 2026-07-13
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_drop_alerts_waiting"
down_revision: str | None = "0004_drop_heatmap_grid"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_dwell_sessions_zone_enter", table_name="dwell_sessions")
    op.drop_table("dwell_sessions")

    op.drop_table("snapshots")

    op.drop_index("ix_alerts_status", table_name="alerts")
    op.drop_index("ix_alerts_zone_ts", table_name="alerts")
    op.drop_table("alerts")


def downgrade() -> None:
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("zone_id", sa.Integer(), nullable=True),
        sa.Column("camera_id", sa.Integer(), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("snapshot_url", sa.String(length=512), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_alerts_zone_ts", "alerts", ["zone_id", "ts"])
    op.create_index("ix_alerts_status", "alerts", ["status"])

    op.create_table(
        "snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "dwell_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("zone_id", sa.Integer(), nullable=False),
        sa.Column("track_ref", sa.Integer(), nullable=False),
        sa.Column("enter_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("leave_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dwell_s", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_dwell_sessions_zone_enter", "dwell_sessions", ["zone_id", "enter_ts"]
    )
