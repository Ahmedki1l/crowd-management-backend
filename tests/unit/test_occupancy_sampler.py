"""Tests for the fixed-rate occupancy sampler (HLD 9).

The sampler exists because occupancy is a *step function* and the calculators emit only on
change: a plain mean over that uneven stream is not a time-weighted mean. Sampling the live
read model at a fixed rate makes the samples evenly spaced, which is what makes ``sum/count``
correct. These tests drive ``tick()`` directly with explicit timestamps rather than waiting
on the thread, so the arithmetic is deterministic.

They also pin the two things that make a stored number honest: that a space is the **sum of
its zones across cameras**, and that a **dead camera's stale count is excluded** rather than
frozen into history as though it were real.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.db.repositories.occupancy_repo import Grain, OccupancyRepository, RollupRow
from app.services.occupancy_sampler import OccupancySampler
from app.services.state_store import CameraHealthState, StateStore

_T0 = datetime(2026, 7, 13, 10, 30, 0, tzinfo=UTC)


@pytest.fixture
def store() -> StateStore:
    return StateStore()


def _minutes(session: Session) -> list[RollupRow]:
    return OccupancyRepository(session).query(
        Grain.MINUTE, _T0 - timedelta(hours=1), _T0 + timedelta(hours=1)
    )


def _healthy(store: StateStore, *camera_ids: int, healthy: bool = True) -> None:
    for camera_id in camera_ids:
        store.set_camera_health(
            CameraHealthState(camera_id=camera_id, healthy=healthy, ts=_T0.timestamp())
        )


def test_sampler_sums_a_spaces_zones_across_cameras(
    session: Session, store: StateStore, make_camera, make_zone
) -> None:
    """A space spans several cameras; no single pipeline can see it, so the sum lives here."""
    cam_a = make_camera(session, name="a", ip="10.0.0.1", floor="B1")
    cam_b = make_camera(session, name="b", ip="10.0.0.2", floor="B1")
    zone_a = make_zone(session, camera_id=cam_a.id, dt_space_id="b1-waiting-area")
    zone_b = make_zone(session, camera_id=cam_b.id, dt_space_id="b1-waiting-area")
    session.commit()
    _healthy(store, cam_a.id, cam_b.id)
    store.set_occupancy(zone_a.id, 3, "b1-waiting-area", _T0.timestamp())
    store.set_occupancy(zone_b.id, 2, "b1-waiting-area", _T0.timestamp())

    sampler = OccupancySampler(store)
    sampler.tick(_T0)
    sampler.tick(_T0 + timedelta(minutes=1))  # crossing the boundary flushes the bucket

    (row,) = _minutes(session)
    assert row.space_id == "b1-waiting-area"
    assert row.floor == "B1"
    assert row.avg == pytest.approx(5.0)  # 3 + 2, not two separate rows
    assert row.samples == 1


def test_sampler_excludes_zones_whose_camera_is_unhealthy(
    session: Session, store: StateStore, make_camera, make_zone
) -> None:
    """A dead camera's last count would otherwise sit in the read model forever.

    Counting it would freeze a stale number into history and pass it off as observed.
    """
    cam_ok = make_camera(session, name="ok", ip="10.0.0.1", floor="B1")
    cam_dead = make_camera(session, name="dead", ip="10.0.0.2", floor="B1")
    zone_ok = make_zone(session, camera_id=cam_ok.id, dt_space_id="b1-waiting-area")
    zone_dead = make_zone(session, camera_id=cam_dead.id, dt_space_id="b1-waiting-area")
    session.commit()
    _healthy(store, cam_ok.id)
    _healthy(store, cam_dead.id, healthy=False)
    store.set_occupancy(zone_ok.id, 3, "b1-waiting-area", _T0.timestamp())
    store.set_occupancy(zone_dead.id, 99, "b1-waiting-area", _T0.timestamp())  # stale

    sampler = OccupancySampler(store)
    sampler.tick(_T0)
    sampler.tick(_T0 + timedelta(minutes=1))

    (row,) = _minutes(session)
    assert row.avg == pytest.approx(3.0)  # the dead camera's 99 is not counted
    # ...and the row says so, rather than silently reporting a smaller number as fact.
    assert (row.cameras_healthy, row.cameras_total) == (1, 2)


def test_sampler_mean_is_the_plain_average_of_its_evenly_spaced_samples(
    session: Session, store: StateStore, make_camera, make_zone
) -> None:
    """Evenly-spaced sampling is what makes sum/count a correct time-weighted mean."""
    cam = make_camera(session, name="a", ip="10.0.0.1", floor="B1")
    zone = make_zone(session, camera_id=cam.id, dt_space_id="b1-waiting-area")
    session.commit()
    _healthy(store, cam.id)

    sampler = OccupancySampler(store)
    for i, count in enumerate([2, 4, 9, 1]):
        store.set_occupancy(zone.id, count, "b1-waiting-area", _T0.timestamp())
        sampler.tick(_T0 + timedelta(seconds=i))
    sampler.tick(_T0 + timedelta(minutes=1))

    (row,) = _minutes(session)
    assert row.avg == pytest.approx(4.0)  # (2 + 4 + 9 + 1) / 4
    assert row.peak == 9  # the spike the mean hides
    assert row.min == 1
    assert row.samples == 4


def test_sampler_ignores_zones_with_no_space(
    session: Session, store: StateStore, make_camera, make_zone
) -> None:
    """History is keyed by space; a zone belonging to none has nowhere to be recorded."""
    cam = make_camera(session, name="a", ip="10.0.0.1", floor="B1")
    orphan = make_zone(session, camera_id=cam.id, dt_space_id=None)
    session.commit()
    _healthy(store, cam.id)
    store.set_occupancy(orphan.id, 7, None, _T0.timestamp())

    sampler = OccupancySampler(store)
    sampler.tick(_T0)
    sampler.tick(_T0 + timedelta(minutes=1))

    assert _minutes(session) == []


def test_stopping_the_sampler_flushes_the_open_bucket(
    session: Session, store: StateStore, make_camera, make_zone
) -> None:
    """A clean shutdown must not discard the minute in progress."""
    cam = make_camera(session, name="a", ip="10.0.0.1", floor="B1")
    zone = make_zone(session, camera_id=cam.id, dt_space_id="b1-waiting-area")
    session.commit()
    _healthy(store, cam.id)
    store.set_occupancy(zone.id, 6, "b1-waiting-area", _T0.timestamp())

    sampler = OccupancySampler(store)
    sampler.tick(_T0)
    sampler.stop()  # never crosses a minute boundary

    (row,) = _minutes(session)
    assert row.avg == pytest.approx(6.0)
