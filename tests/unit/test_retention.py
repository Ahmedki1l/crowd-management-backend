"""Tests for history retention (NFR-04).

``RetentionConfig`` existed from the first commit and was read by no code, while the README
and the traceability matrix both asserted a retention policy was in force. These pin that
it now actually deletes — and, just as importantly, that a window of ``0`` means *keep
forever* rather than *delete everything*, which is the failure that would be unrecoverable.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.config.schema import RetentionConfig
from app.db.repositories.occupancy_repo import Grain, OccupancyRepository, RollupRow
from app.services.retention import prune

_NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


def _bucket(session: Session, grain: Grain, age_days: int) -> None:
    OccupancyRepository(session).upsert(
        grain,
        RollupRow(
            space_id="b1-waiting-area",
            floor="B1",
            bucket_ts=_NOW - timedelta(days=age_days),
            avg=1.0,
            peak=1,
            min=0,
            samples=60,
            cameras_healthy=1,
            cameras_total=1,
        ),
    )


def _count(session: Session, grain: Grain) -> int:
    return len(
        OccupancyRepository(session).query(
            grain, _NOW - timedelta(days=10_000), _NOW + timedelta(days=1)
        )
    )


def test_prune_deletes_minute_buckets_past_the_window(session: Session) -> None:
    _bucket(session, Grain.MINUTE, age_days=40)  # older than the 30-day window
    _bucket(session, Grain.MINUTE, age_days=5)
    session.commit()

    removed = prune(RetentionConfig(occupancy_minute_days=30), _NOW)

    assert removed["occupancy_minute"] == 1
    assert _count(session, Grain.MINUTE) == 1  # the recent one survives


def test_prune_keeps_hour_buckets_far_longer_than_minutes(session: Session) -> None:
    """The hour grain is small enough to keep for years; the minute grain is not."""
    _bucket(session, Grain.MINUTE, age_days=40)
    _bucket(session, Grain.HOUR, age_days=40)
    session.commit()

    prune(RetentionConfig(occupancy_minute_days=30, occupancy_hour_days=730), _NOW)

    assert _count(session, Grain.MINUTE) == 0
    assert _count(session, Grain.HOUR) == 1  # 40 days is nothing at the hour grain


def test_zero_means_keep_forever_not_delete_everything(session: Session) -> None:
    """A misread config that silently emptied a table would be unrecoverable."""
    _bucket(session, Grain.MINUTE, age_days=9_000)
    session.commit()

    removed = prune(RetentionConfig(occupancy_minute_days=0), _NOW)

    assert "occupancy_minute" not in removed
    assert _count(session, Grain.MINUTE) == 1
