"""Tests for the minute -> hour rollup (HLD 9).

The arithmetic that matters here is the **weighting**. Averaging averages is only correct
when every minute carries the same weight, and they do not: a minute during which a camera
was down contributed fewer samples than a fully-covered one. These pin that a partially
observed minute cannot drag an hour's mean around as if it were a full one.

The worker's other contract — that it is restartable and idempotent — is pinned by driving
``run_once`` twice and re-running across a watermark.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.db.repositories.occupancy_repo import Grain, OccupancyRepository, RollupRow
from app.services.history_worker import HistoryWorker

_HOUR = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
# "Now" must be in a later hour, or the 10:00 bucket is still open and not rolled up.
_NOW = _HOUR + timedelta(hours=1, minutes=5)


def _minute(
    session: Session,
    minute: int,
    avg: float,
    samples: int = 60,
    peak: int | None = None,
    low: int = 0,
    space_id: str = "b1-waiting-area",
    floor: str | None = "B1",
    cameras_healthy: int = 2,
    cameras_total: int = 2,
) -> None:
    OccupancyRepository(session).upsert(
        Grain.MINUTE,
        RollupRow(
            space_id=space_id,
            floor=floor,
            bucket_ts=_HOUR + timedelta(minutes=minute),
            avg=avg,
            peak=int(avg) if peak is None else peak,
            min=low,
            samples=samples,
            cameras_healthy=cameras_healthy,
            cameras_total=cameras_total,
        ),
    )


def _hours(session: Session) -> list[RollupRow]:
    return OccupancyRepository(session).query(
        Grain.HOUR, _HOUR - timedelta(days=1), _NOW
    )


def test_hourly_mean_weights_each_minute_by_its_sample_count(session: Session) -> None:
    """A thinly-observed minute must not count as much as a fully-observed one.

    Ten samples of 30 people and fifty of 0 is a mean of 5.0, not the 15.0 a plain
    average of the two minute-means would give.
    """
    _minute(session, 0, avg=30.0, samples=10)  # camera mostly down; only 10 samples
    _minute(session, 1, avg=0.0, samples=50)
    session.commit()

    HistoryWorker().run_once(_NOW)

    (hour,) = _hours(session)
    assert hour.avg == pytest.approx(5.0)  # (30*10 + 0*50) / 60
    assert hour.samples == 60


def test_hourly_peak_is_the_highest_minute_peak(session: Session) -> None:
    """The average hides spikes; the peak is what a safety limit is checked against."""
    _minute(session, 0, avg=2.0, peak=3)
    _minute(session, 1, avg=2.0, peak=41)  # a spike inside an otherwise-quiet hour
    session.commit()

    HistoryWorker().run_once(_NOW)

    (hour,) = _hours(session)
    assert hour.peak == 41
    assert hour.avg == pytest.approx(2.0)  # ...which the mean gives no hint of


def test_each_space_gets_its_own_hour_bucket(session: Session) -> None:
    _minute(session, 0, avg=4.0, space_id="b1-waiting-area", floor="B1")
    _minute(session, 0, avg=1.0, space_id="gf-waiting-area", floor="GF")
    session.commit()

    HistoryWorker().run_once(_NOW)

    hours = sorted(_hours(session), key=lambda r: r.space_id)
    assert [(h.space_id, h.floor, h.avg) for h in hours] == [
        ("b1-waiting-area", "B1", 4.0),
        ("gf-waiting-area", "GF", 1.0),
    ]


def test_the_open_hour_is_not_rolled_up(session: Session) -> None:
    """Rolling up an in-progress hour would publish a number that is still changing."""
    _minute(session, 0, avg=5.0)
    session.commit()

    # "Now" is inside the same hour the minute belongs to.
    HistoryWorker().run_once(_HOUR + timedelta(minutes=30))

    assert _hours(session) == []


def test_rerunning_does_not_duplicate_or_drift(session: Session) -> None:
    """Idempotency is what makes a restart, a catch-up, or a second process harmless."""
    _minute(session, 0, avg=4.0)
    session.commit()
    worker = HistoryWorker()

    worker.run_once(_NOW)
    worker.run_once(_NOW)

    (hour,) = _hours(session)
    assert hour.avg == pytest.approx(4.0)


def test_a_late_minute_corrects_an_already_written_hour(session: Session) -> None:
    """The watermark hour is re-rolled, so a minute that arrives late is not lost.

    This is why the worker resumes *from* the watermark rather than after it.
    """
    _minute(session, 0, avg=10.0)
    session.commit()
    worker = HistoryWorker()
    worker.run_once(_NOW)

    _minute(session, 1, avg=0.0)  # arrives after the hour was first written
    session.commit()
    worker.run_once(_NOW)

    (hour,) = _hours(session)
    assert hour.avg == pytest.approx(5.0)  # recomputed from both minutes, not just one
    assert hour.samples == 120


def test_retention_is_throttled_to_hourly_not_every_tick(session: Session) -> None:
    """Retention windows are measured in days.

    The worker ticks every 60s to catch newly-complete hours; re-scanning for expired rows
    at that cadence would burn a table scan a minute to delete nothing.
    """
    worker = HistoryWorker()

    assert worker.prune_if_due(_NOW) is True  # first pass always runs
    assert worker.prune_if_due(_NOW + timedelta(minutes=1)) is False  # a tick later: skipped
    assert worker.prune_if_due(_NOW + timedelta(minutes=59)) is False
    assert worker.prune_if_due(_NOW + timedelta(hours=1)) is True  # due again


def test_hour_coverage_is_the_worst_minute_not_the_best(session: Session) -> None:
    """An hour that lost a camera for even one minute must report reduced coverage.

    Aggregating coverage as max would let a single fully-covered minute paper over an hour
    that was mostly blind, hiding that the hour's average is depressed by an outage.
    """
    _minute(session, 0, avg=10.0, cameras_healthy=2, cameras_total=2)
    _minute(session, 1, avg=10.0, cameras_healthy=1, cameras_total=2)  # a camera dropped
    session.commit()

    HistoryWorker().run_once(_NOW)

    (hour,) = _hours(session)
    assert hour.cameras_healthy == 1  # worst minute, not the best (which was 2)
    assert hour.cameras_total == 2
