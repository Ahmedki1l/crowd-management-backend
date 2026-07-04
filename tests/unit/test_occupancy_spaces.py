"""Tests for per-space occupancy aggregation (``dt_space_id`` grouping).

Covers :meth:`OccupancyService.by_space` (grouping, exclusion of unassigned
zones, area filter) and the ``GET /occupancy/spaces`` endpoint (auth + shape).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.services.occupancy_service import OccupancyService
from app.services.state_store import StateStore, get_state_store


def test_by_space_sums_counts_grouped_by_dt_space_id(session: Session) -> None:
    store = StateStore()
    # zones 1 & 2 (different cameras) form one logical space "lobby"; zone 3 is
    # "hall"; zone 4 has no space and must be excluded.
    store.set_occupancy(zone_id=1, count=3, dt_space_id="lobby", ts=100.0)
    store.set_occupancy(zone_id=2, count=4, dt_space_id="lobby", ts=120.0)
    store.set_occupancy(zone_id=3, count=2, dt_space_id="hall", ts=110.0)
    store.set_occupancy(zone_id=4, count=9, dt_space_id=None, ts=130.0)

    spaces = OccupancyService(session, store).by_space()

    assert [s.dt_space_id for s in spaces] == ["hall", "lobby"]  # sorted
    lobby = next(s for s in spaces if s.dt_space_id == "lobby")
    assert lobby.count == 7
    assert lobby.zone_ids == [1, 2]
    assert lobby.ts == 120.0  # newest member timestamp


def test_by_space_area_filter_restricts_to_matching_cameras(
    session: Session, make_camera, make_zone
) -> None:
    cam_a = make_camera(session, name="a", area="floor1")
    cam_b = make_camera(session, name="b", area="floor2")
    zone_a = make_zone(session, camera_id=cam_a.id, dt_space_id="s1")
    zone_b = make_zone(session, camera_id=cam_b.id, dt_space_id="s2")

    store = StateStore()
    store.set_occupancy(zone_id=zone_a.id, count=3, dt_space_id="s1", ts=100.0)
    store.set_occupancy(zone_id=zone_b.id, count=5, dt_space_id="s2", ts=100.0)

    spaces = OccupancyService(session, store).by_space(area_id="floor1")

    assert [(s.dt_space_id, s.count) for s in spaces] == [("s1", 3)]


def test_by_space_dt_space_id_filter_returns_only_that_space(session: Session) -> None:
    store = StateStore()
    store.set_occupancy(zone_id=1, count=3, dt_space_id="lobby", ts=100.0)
    store.set_occupancy(zone_id=2, count=4, dt_space_id="lobby", ts=110.0)
    store.set_occupancy(zone_id=3, count=2, dt_space_id="hall", ts=100.0)

    spaces = OccupancyService(session, store).by_space(dt_space_id="lobby")

    assert [(s.dt_space_id, s.count) for s in spaces] == [("lobby", 7)]


def test_occupancy_spaces_endpoint_returns_grouped(client, auth_headers) -> None:
    store = get_state_store()
    store.set_occupancy(zone_id=1, count=2, dt_space_id="lobby", ts=100.0)
    store.set_occupancy(zone_id=2, count=5, dt_space_id="lobby", ts=140.0)

    resp = client.get("/api/v1/occupancy/spaces", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == [
        {"dt_space_id": "lobby", "count": 7, "zone_ids": [1, 2], "ts": 140.0}
    ]


def test_occupancy_spaces_endpoint_requires_auth(client) -> None:
    assert client.get("/api/v1/occupancy/spaces").status_code in (401, 403)


# --- by floor --------------------------------------------------------------
def test_by_floor_groups_by_camera_floor_with_space_breakdown(
    session: Session, make_camera, make_zone
) -> None:
    cam_a = make_camera(session, name="a", floor="B1")
    cam_b = make_camera(session, name="b", floor="B1")  # 2nd camera, same space
    cam_c = make_camera(session, name="c", floor="B1")  # different space
    cam_g = make_camera(session, name="g", floor="GF")
    za = make_zone(session, camera_id=cam_a.id, dt_space_id="s1")
    zb = make_zone(session, camera_id=cam_b.id, dt_space_id="s1")
    zc = make_zone(session, camera_id=cam_c.id, dt_space_id="s2")
    zg = make_zone(session, camera_id=cam_g.id, dt_space_id="g1")

    store = StateStore()
    store.set_occupancy(za.id, 2, "s1", 100.0)
    store.set_occupancy(zb.id, 3, "s1", 120.0)
    store.set_occupancy(zc.id, 1, "s2", 110.0)
    store.set_occupancy(zg.id, 5, "g1", 130.0)

    floors = OccupancyService(session, store).by_floor()

    assert [f.floor for f in floors] == ["B1", "GF"]  # sorted
    b1 = next(f for f in floors if f.floor == "B1")
    assert b1.count == 6  # 2 + 3 + 1 across the whole floor
    assert [(s.dt_space_id, s.count, s.zone_ids) for s in b1.spaces] == [
        ("s1", 5, sorted([za.id, zb.id])),
        ("s2", 1, [zc.id]),
    ]
    assert b1.ts == 120.0  # newest member on the floor


def test_by_floor_filter_returns_single_floor(
    session: Session, make_camera, make_zone
) -> None:
    cam_a = make_camera(session, name="a", floor="B1")
    cam_g = make_camera(session, name="g", floor="GF")
    za = make_zone(session, camera_id=cam_a.id, dt_space_id="s1")
    zg = make_zone(session, camera_id=cam_g.id, dt_space_id="g1")
    store = StateStore()
    store.set_occupancy(za.id, 2, "s1", 100.0)
    store.set_occupancy(zg.id, 4, "g1", 100.0)

    floors = OccupancyService(session, store).by_floor(floor="B1")

    assert [(f.floor, f.count) for f in floors] == [("B1", 2)]


def test_by_floor_excludes_cameras_without_a_floor(
    session: Session, make_camera, make_zone
) -> None:
    cam = make_camera(session, name="a")  # floor defaults to None
    zone = make_zone(session, camera_id=cam.id, dt_space_id="s1")
    store = StateStore()
    store.set_occupancy(zone.id, 3, "s1", 100.0)

    assert OccupancyService(session, store).by_floor() == []


def test_occupancy_floors_endpoint_returns_rollup(
    client, auth_headers, session: Session, make_camera, make_zone
) -> None:
    cam = make_camera(session, name="a", floor="B1")
    zone = make_zone(session, camera_id=cam.id, dt_space_id="b1-waiting-area")
    get_state_store().set_occupancy(zone.id, 2, "b1-waiting-area", 100.0)

    resp = client.get("/api/v1/occupancy/floors", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == [
        {
            "floor": "B1",
            "count": 2,
            "spaces": [
                {"dt_space_id": "b1-waiting-area", "count": 2, "zone_ids": [zone.id]}
            ],
            "ts": 100.0,
        }
    ]
