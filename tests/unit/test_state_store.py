"""Unit tests for the in-memory current-state read model (app.services.state_store)."""

from __future__ import annotations

from app.services.state_store import StateStore


def test_set_get_occupancy_roundtrip() -> None:
    store = StateStore()
    store.set_occupancy(zone_id=5, count=3, dt_space_id="space-a", ts=1000.0)

    state = store.get_occupancy(5)

    assert state is not None
    assert (state.zone_id, state.count, state.dt_space_id) == (5, 3, "space-a")


def test_all_occupancy_returns_all_set_zones() -> None:
    store = StateStore()
    store.set_occupancy(zone_id=1, count=2, dt_space_id=None, ts=1000.0)
    store.set_occupancy(zone_id=2, count=4, dt_space_id=None, ts=1000.0)

    zone_ids = {s.zone_id for s in store.all_occupancy()}

    assert zone_ids == {1, 2}


def test_set_counts_with_lines_records_per_line_breakdown() -> None:
    store = StateStore()
    store.set_counts(
        area_id="area-1",
        in_count=10,
        out_count=4,
        net=6,
        ts=1000.0,
        lines={7: (10, 4)},
    )

    state = store.get_counts("area-1")

    assert state is not None
    assert (state.in_count, state.out_count, state.net) == (10, 4, 6)
    assert state.lines[7].in_count == 10
    assert state.lines[7].out_count == 4


def test_set_get_waiting_roundtrip() -> None:
    store = StateStore()
    store.set_waiting(
        zone_id=9, current_waits=2, avg_dwell_s=12.5, dt_space_id="space-b", ts=1000.0
    )

    state = store.get_waiting(9)

    assert state is not None
    assert (state.current_waits, state.avg_dwell_s) == (2, 12.5)


def test_set_get_camera_health_roundtrip() -> None:
    from app.services.state_store import CameraHealthState

    store = StateStore()
    store.set_camera_health(
        CameraHealthState(camera_id=3, fps=15.0, queue_depth=1, healthy=False, ts=1000.0)
    )

    state = store.get_camera_health(3)

    assert state is not None
    assert state.healthy is False
    assert state.fps == 15.0


def test_total_occupancy_sums_zone_counts() -> None:
    store = StateStore()
    store.set_occupancy(zone_id=1, count=3, dt_space_id=None, ts=1000.0)
    store.set_occupancy(zone_id=2, count=5, dt_space_id=None, ts=1000.0)

    assert store.total_occupancy() == 8


def test_clear_resets_all_maps() -> None:
    store = StateStore()
    store.set_occupancy(zone_id=1, count=3, dt_space_id=None, ts=1000.0)
    store.set_counts(area_id="a", in_count=1, out_count=0, net=1, ts=1000.0)
    store.set_waiting(zone_id=1, current_waits=1, avg_dwell_s=1.0, dt_space_id=None, ts=1000.0)

    store.clear()

    assert store.all_occupancy() == []
    assert store.all_counts() == []
    assert store.all_waiting() == []
    assert store.total_occupancy() == 0
