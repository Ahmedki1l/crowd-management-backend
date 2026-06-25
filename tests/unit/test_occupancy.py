"""Unit tests for :class:`app.analytics.occupancy.OccupancyCalculator`.

The calculator turns debounced per-zone confirmed membership into
``OccupancyUpdate`` events, emitting one event per zone only when that zone's
confirmed count changes. These tests drive the real calculator with the
deterministic ``FakeClock`` and assert the returned events and their counts.
"""

from __future__ import annotations

from app.analytics.occupancy import OccupancyCalculator
from app.domain.models import ZoneType
from app.events.events import OccupancyUpdate
from tests.fixtures.specs import make_zone_spec


def test_count_equals_number_of_confirmed_members(fake_clock):
    zone = make_zone_spec(id=7, type=ZoneType.OCCUPANCY)
    calc = OccupancyCalculator(camera_id=3, zones=[zone], clock=fake_clock)

    updates = calc.process({7: {11, 12, 13}}, ts=1000.0)

    assert [u.count for u in updates] == [3]


def test_update_carries_zone_and_camera_identity(fake_clock):
    zone = make_zone_spec(id=7, type=ZoneType.OCCUPANCY, dt_space_id="space-7")
    calc = OccupancyCalculator(camera_id=3, zones=[zone], clock=fake_clock)

    (update,) = calc.process({7: {11, 12}}, ts=1000.0)

    assert isinstance(update, OccupancyUpdate)
    assert (update.zone_id, update.camera_id, update.count, update.dt_space_id) == (
        7,
        3,
        2,
        "space-7",
    )


def test_zone_absent_from_membership_counts_as_zero(fake_clock):
    zone = make_zone_spec(id=7, type=ZoneType.OCCUPANCY)
    calc = OccupancyCalculator(camera_id=3, zones=[zone], clock=fake_clock)

    (update,) = calc.process({}, ts=1000.0)

    assert update.count == 0


def test_unchanged_count_emits_no_duplicate_update(fake_clock):
    zone = make_zone_spec(id=7, type=ZoneType.OCCUPANCY)
    calc = OccupancyCalculator(camera_id=3, zones=[zone], clock=fake_clock)

    calc.process({7: {11, 12}}, ts=1000.0)
    repeat = calc.process({7: {11, 12}}, ts=1001.0)

    assert repeat == []


def test_changed_count_emits_a_new_update(fake_clock):
    zone = make_zone_spec(id=7, type=ZoneType.OCCUPANCY)
    calc = OccupancyCalculator(camera_id=3, zones=[zone], clock=fake_clock)

    calc.process({7: {11, 12}}, ts=1000.0)
    after_change = calc.process({7: {11, 12, 13}}, ts=1001.0)

    assert [u.count for u in after_change] == [3]


def test_only_changed_zone_emits_when_others_are_stable(fake_clock):
    zone_a = make_zone_spec(id=1, type=ZoneType.OCCUPANCY)
    zone_b = make_zone_spec(id=2, type=ZoneType.OCCUPANCY)
    calc = OccupancyCalculator(camera_id=3, zones=[zone_a, zone_b], clock=fake_clock)

    calc.process({1: {11}, 2: {21, 22}}, ts=1000.0)
    second = calc.process({1: {11}, 2: {21, 22, 23}}, ts=1001.0)

    assert [(u.zone_id, u.count) for u in second] == [(2, 3)]
