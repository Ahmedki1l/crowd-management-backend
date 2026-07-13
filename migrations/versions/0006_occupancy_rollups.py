"""replace occupancy_samples with space-keyed minute/hour rollups

Occupancy history moves from raw per-zone samples to pre-aggregated rollups keyed by
logical space (``dt_space_id``). Two reasons, both learned from the data:

* **Zone ids are not a stable key.** Redrawing a zone polygon deletes the zone row, and
  with SQLite's ``foreign_keys`` pragma off the ``ON DELETE CASCADE`` never fired — 36,549
  of the 47,215 rows in ``occupancy_samples`` (77%) pointed at zone ids that no longer
  existed. ``dt_space_id`` survives a redraw.
* **The stored numbers were not trustworthy anyway.** Samples were written only when a
  count changed, and a change arriving within the 5 s persistence throttle was *dropped*,
  so an occupancy level could go unrecorded entirely. The reader then took a plain mean of
  those unevenly-spaced samples, which is not a time-weighted average.

The rollups are written by a 1 Hz sampler, so their evenly-spaced ``sum/count`` mean *is*
the time-weighted mean. ``samples`` doubles as a coverage metric (60 per minute, 3600 per
hour at full coverage).

``occupancy_samples`` is dropped, not migrated: 77% of it is unresolvable and the rest is
distorted by the throttle, so backfilling from it would launder bad data into a table that
looks authoritative.

Revision ID: 0006_occupancy_rollups
Revises: 0005_drop_alerts_waiting
Create Date: 2026-07-13
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_occupancy_rollups"
down_revision: str | None = "0005_drop_alerts_waiting"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _rollup_table(name: str) -> None:
    """Create one rollup table.

    The unique constraint is declared **inline**, not added afterwards with
    ``create_unique_constraint``: SQLite has no ``ALTER TABLE ADD CONSTRAINT``, so the
    separate form raises ``NotImplementedError`` there. Inline works on both dialects.
    """
    op.create_table(
        name,
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        # Deliberately NOT a foreign key: this must outlive the zones that fed it.
        sa.Column("space_id", sa.String(length=128), nullable=False),
        # Denormalised so re-assigning a camera's floor cannot rewrite the past.
        sa.Column("floor", sa.String(length=32), nullable=True),
        sa.Column("bucket_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("avg", sa.Float(), nullable=False),
        sa.Column("peak", sa.Integer(), nullable=False),
        sa.Column("min", sa.Integer(), nullable=False),
        sa.Column("samples", sa.Integer(), nullable=False),
        sa.Column("cameras_healthy", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cameras_total", sa.Integer(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "space_id", "bucket_ts", name=f"uq_{name}_space_bucket"
        ),
    )
    op.create_index(f"ix_{name}_space_bucket", name, ["space_id", "bucket_ts"])
    op.create_index(f"ix_{name}_bucket", name, ["bucket_ts"])


def upgrade() -> None:
    _rollup_table("occupancy_minute")
    _rollup_table("occupancy_hour")
    # Only the hour grain is filtered by floor (the Digital Twin's floor view).
    op.create_index(
        "ix_occupancy_hour_floor_bucket", "occupancy_hour", ["floor", "bucket_ts"]
    )

    op.drop_index("ix_occupancy_samples_zone_ts", table_name="occupancy_samples")
    op.drop_table("occupancy_samples")


def downgrade() -> None:
    op.create_table(
        "occupancy_samples",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("zone_id", sa.Integer(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_occupancy_samples_zone_ts", "occupancy_samples", ["zone_id", "ts"]
    )

    op.drop_table("occupancy_hour")
    op.drop_table("occupancy_minute")
