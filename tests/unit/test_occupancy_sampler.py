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


def test_empty_zone_map_does_not_requery_the_db_every_tick(
    session: Session, store: StateStore, monkeypatch
) -> None:
    """With no space-assigned zones the map is legitimately empty.

    Gating the refresh on the map's truthiness would re-read the DB on every 1 Hz tick
    forever, because an empty dict is falsy. The throttle keys on the load timestamp.
    """
    import app.services.occupancy_sampler as sampler_mod

    reads = {"n": 0}
    real_list = sampler_mod.ZoneRepository.list

    def _counting_list(self):  # type: ignore[no-untyped-def]
        reads["n"] += 1
        return real_list(self)

    monkeypatch.setattr(sampler_mod.ZoneRepository, "list", _counting_list)

    sampler = OccupancySampler(store)
    for i in range(5):
        sampler.tick(_T0 + timedelta(seconds=i))

    assert reads["n"] == 1, f"reloaded the zone map {reads['n']}x over 5 ticks"


def test_a_fully_blind_space_is_a_gap_not_a_fabricated_zero(
    session: Session, store: StateStore, make_camera, make_zone
) -> None:
    """When every camera covering a space is down, we have no information.

    Recording count 0 would read as "the space emptied"; the honest record is no row —
    a gap in the series — so a reader can tell blindness from emptiness.
    """
    cam_a = make_camera(session, name="a", ip="10.0.0.1", floor="B1")
    cam_b = make_camera(session, name="b", ip="10.0.0.2", floor="B1")
    zone_a = make_zone(session, camera_id=cam_a.id, dt_space_id="b1-waiting-area")
    zone_b = make_zone(session, camera_id=cam_b.id, dt_space_id="b1-waiting-area")
    session.commit()
    _healthy(store, cam_a.id, cam_b.id, healthy=False)  # both down
    store.set_occupancy(zone_a.id, 5, "b1-waiting-area", _T0.timestamp())
    store.set_occupancy(zone_b.id, 3, "b1-waiting-area", _T0.timestamp())

    sampler = OccupancySampler(store)
    sampler.tick(_T0)
    sampler.tick(_T0 + timedelta(minutes=1))

    assert _minutes(session) == []  # a gap, not a row of 0


def test_coverage_is_the_worst_seen_in_the_bucket_not_the_best(
    session: Session, store: StateStore, make_camera, make_zone
) -> None:
    """A partly-blind minute must report reduced coverage, so its depressed average is
    attributable to an outage rather than mistaken for a drop in people."""
    cam_a = make_camera(session, name="a", ip="10.0.0.1", floor="B1")
    cam_b = make_camera(session, name="b", ip="10.0.0.2", floor="B1")
    zone_a = make_zone(session, camera_id=cam_a.id, dt_space_id="b1-waiting-area")
    zone_b = make_zone(session, camera_id=cam_b.id, dt_space_id="b1-waiting-area")
    session.commit()
    _healthy(store, cam_a.id, cam_b.id)
    store.set_occupancy(zone_a.id, 4, "b1-waiting-area", _T0.timestamp())
    store.set_occupancy(zone_b.id, 2, "b1-waiting-area", _T0.timestamp())

    sampler = OccupancySampler(store)
    sampler.tick(_T0)  # both healthy -> 2/2
    _healthy(store, cam_b.id, healthy=False)
    sampler.tick(_T0 + timedelta(seconds=1))  # b down -> 1/2
    sampler.tick(_T0 + timedelta(minutes=1))

    (row,) = _minutes(session)
    # Worst coverage across the minute's two ticks, not the best (which would be 2/2).
    assert (row.cameras_healthy, row.cameras_total) == (1, 2)


def test_a_restart_mid_minute_merges_the_partial_row_rather_than_clobbering_it(
    session: Session, store: StateStore, make_camera, make_zone
) -> None:
    """A graceful restart flushes a partial minute; the fresh process must fold into it.

    Overwriting would discard the pre-restart samples and skew the minute's mean and its
    weight in the hour rollup.
    """
    cam = make_camera(session, name="a", ip="10.0.0.1", floor="B1")
    zone = make_zone(session, camera_id=cam.id, dt_space_id="b1-waiting-area")
    session.commit()
    _healthy(store, cam.id)

    store.set_occupancy(zone.id, 10, "b1-waiting-area", _T0.timestamp())
    before = OccupancySampler(store)
    for sec in range(3):
        before.tick(_T0 + timedelta(seconds=sec))
    before.stop()  # flush a partial minute: 3 samples of 10

    store.set_occupancy(zone.id, 4, "b1-waiting-area", _T0.timestamp())
    after = OccupancySampler(store)  # a fresh process, same minute
    for sec in range(3, 5):
        after.tick(_T0 + timedelta(seconds=sec))
    after.stop()  # 2 samples of 4 -> must MERGE with the partial, not replace it

    (row,) = _minutes(session)
    assert row.samples == 5  # 3 + 2, not clobbered to 2
    assert row.avg == pytest.approx((10 * 3 + 4 * 2) / 5)  # sample-weighted merge
