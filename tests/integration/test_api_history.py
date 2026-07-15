"""Integration tests for the historical time-series router (HLD 8.3).

Rows are inserted directly through their repositories, then the matching
``/api/v1/history/*`` endpoints are queried with an explicit ``from``/``to`` window.

Occupancy history is pre-aggregated, so these seed rollup buckets rather than raw
samples and assert the endpoint serves the stored grain: the space series, the floor
aggregation, the filters, and that a bucket width we do not store is refused rather
than silently re-bucketed. The rollup arithmetic itself is covered in
``tests/unit/test_history_worker.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.repositories.crossing_repo import CrossingRepository
from app.db.repositories.occupancy_repo import Grain, OccupancyRepository, RollupRow

# A clean hour boundary so 10:00/10:30 share one 1h bucket and 11:00 starts the next.
_BASE = datetime(2024, 6, 12, 10, 0, tzinfo=UTC)
_WINDOW = {
    "from": str(datetime(2024, 6, 12, 9, 0, tzinfo=UTC).timestamp()),
    "to": str(datetime(2024, 6, 12, 13, 0, tzinfo=UTC).timestamp()),
    "bucket": "1h",
}


def _at(hour: int, minute: int = 0) -> datetime:
    return _BASE.replace(hour=hour, minute=minute)


@pytest.fixture
def line_id(session: Session, make_camera, make_line) -> int:
    """A real line row. Crossings reference it, and foreign keys are now enforced."""
    camera = make_camera(session, name="gate-cam", ip="10.0.0.9")
    return make_line(session, camera_id=camera.id).id


def _hour_row(space_id: str, floor: str | None, hour: int, avg: float, peak: int) -> RollupRow:
    return RollupRow(
        space_id=space_id,
        floor=floor,
        bucket_ts=_at(hour),
        avg=avg,
        peak=peak,
        min=0,
        samples=3600,
        cameras_healthy=2,
        cameras_total=2,
    )


def _seed_hours(session: Session) -> None:
    repo = OccupancyRepository(session)
    repo.upsert(Grain.HOUR, _hour_row("b1-waiting-area", "B1", 10, avg=4.0, peak=9))
    repo.upsert(Grain.HOUR, _hour_row("b1-waiting-area", "B1", 11, avg=6.0, peak=12))
    repo.upsert(Grain.HOUR, _hour_row("gf-waiting-area", "GF", 10, avg=1.0, peak=3))
    session.commit()


def test_occupancy_history_returns_one_series_per_space(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    _seed_hours(session)

    response = client.get(
        "/api/v1/history/occupancy", params=_WINDOW, headers=auth_headers
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [s["space_id"] for s in body["series"]] == [
        "b1-waiting-area",
        "gf-waiting-area",
    ]
    b1 = body["series"][0]
    assert [p["avg"] for p in b1["points"]] == [4.0, 6.0]
    assert [p["peak"] for p in b1["points"]] == [9, 12]


def test_occupancy_history_timestamps_are_utc_marked(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    """Bucket timestamps must serialize as UTC (offset/Z), not a bare naive string.

    SQLite has no timezone type and returns ``bucket_ts`` naive, which serializes with no
    offset — a client then reads a 10:00 UTC bucket as 10:00 *local*, hours off. The repo
    re-stamps UTC on read; this pins that the API output carries it. It slipped the whole
    suite because every other test asserts on ``avg``/``peak``, never the timestamp string.
    """
    _seed_hours(session)

    body = client.get(
        "/api/v1/history/occupancy", params=_WINDOW, headers=auth_headers
    ).json()

    stamps = [p["ts"] for s in body["series"] for p in s["points"]]
    assert stamps, "expected at least one bucket"
    for ts in [*stamps, body["start"], body["end"]]:
        assert ts.endswith("Z") or "+00:00" in ts, f"timestamp not UTC-marked: {ts!r}"


@pytest.mark.parametrize(
    ("filter_param", "expected_spaces"),
    [
        ({"floor": "GF"}, ["gf-waiting-area"]),
        ({"space_id": "b1-waiting-area"}, ["b1-waiting-area"]),
        ({}, ["b1-waiting-area", "gf-waiting-area"]),
    ],
    ids=["by-floor", "by-space", "unfiltered"],
)
def test_occupancy_history_filters_the_series(
    client: TestClient,
    auth_headers: dict[str, str],
    session: Session,
    filter_param: dict[str, str],
    expected_spaces: list[str],
) -> None:
    _seed_hours(session)

    response = client.get(
        "/api/v1/history/occupancy",
        params={**_WINDOW, **filter_param},
        headers=auth_headers,
    )

    assert [s["space_id"] for s in response.json()["series"]] == expected_spaces


def test_occupancy_history_rejects_a_bucket_that_is_not_stored(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """Only the stored grains exist; re-bucketing at read time is what rollups avoid."""
    response = client.get(
        "/api/v1/history/occupancy",
        params={**_WINDOW, "bucket": "15m"},
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_a_window_offset_is_converted_to_utc_not_taken_literally(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    """A ``from``/``to`` carrying an offset must select the right UTC buckets.

    bucket_ts is stored naive on SQLite, so the WHERE compares wall-clock digits. If the
    router kept a client's +02:00 offset instead of converting, ``12:00+02:00`` would
    match the 12:00-UTC bucket instead of the 10:00-UTC one it denotes.
    """
    _seed_hours(session)  # buckets at 10:00 and 11:00 UTC

    # 09:00-12:00 at +02:00  ==  07:00-10:00 UTC  -> excludes both seeded buckets.
    response = client.get(
        "/api/v1/history/occupancy",
        params={
            "from": "2024-06-12T09:00:00+02:00",
            "to": "2024-06-12T12:00:00+02:00",
            "bucket": "1h",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    # 10:00 UTC == 12:00+02:00, which is the exclusive upper bound -> no buckets.
    assert response.json()["series"] == []


def test_an_empty_space_filter_means_all_not_none(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    """``?space_id=`` arrives as ``['']``; a naive ``.in_([''])`` would match nothing."""
    _seed_hours(session)

    response = client.get(
        "/api/v1/history/occupancy",
        params={**_WINDOW, "space_id": ""},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert {s["space_id"] for s in response.json()["series"]} == {
        "b1-waiting-area",
        "gf-waiting-area",
    }


def test_a_window_too_large_for_the_cap_is_refused_not_truncated(
    client: TestClient, auth_headers: dict[str, str], session: Session, monkeypatch
) -> None:
    """Silently truncating drops the newest buckets, so the series looks like it ends early."""
    import app.api.routers.history as history_router

    monkeypatch.setattr(history_router, "_MAX_ROWS", 1)
    _seed_hours(session)  # 3 rows

    response = client.get(
        "/api/v1/history/occupancy", params=_WINDOW, headers=auth_headers
    )

    assert response.status_code == 422
    assert "narrow" in response.json()["detail"]


def test_floor_history_sums_the_spaces_on_a_floor(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    repo = OccupancyRepository(session)
    # Two spaces on one floor, same hour: the floor is their sum.
    repo.upsert(Grain.HOUR, _hour_row("b1-waiting-area", "B1", 10, avg=4.0, peak=9))
    repo.upsert(Grain.HOUR, _hour_row("b1-lobby", "B1", 10, avg=2.5, peak=5))
    session.commit()

    response = client.get(
        "/api/v1/history/occupancy/floors", params=_WINDOW, headers=auth_headers
    )

    assert response.status_code == 200, response.text
    floors = response.json()["floors"]
    assert len(floors) == 1
    assert floors[0]["floor"] == "B1"
    assert floors[0]["space_ids"] == ["b1-lobby", "b1-waiting-area"]
    point = floors[0]["points"][0]
    assert point["avg"] == 6.5  # 4.0 + 2.5
    assert point["peak"] == 14  # 9 + 5


def test_entry_exit_history_returns_net_per_bucket(
    client: TestClient, auth_headers: dict[str, str], session: Session, line_id: int
) -> None:
    repo = CrossingRepository(session)
    repo.add(line_id=line_id, area_id="lobby", ts=_at(10, 0), direction="in", track_ref=1)
    repo.add(line_id=line_id, area_id="lobby", ts=_at(10, 10), direction="in", track_ref=2)
    repo.add(line_id=line_id, area_id="lobby", ts=_at(10, 20), direction="out", track_ref=3)
    session.commit()

    response = client.get(
        "/api/v1/history/entry-exit",
        params={"area_id": "lobby", **_WINDOW},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert [p["value"] for p in response.json()["points"]] == [1.0]


# Noon UTC lands on the same calendar day in every server timezone (UTC-11..+11),
# so these daily-total tests are deterministic regardless of where CI runs.
def _noon(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, 0, tzinfo=UTC)


def test_entry_exit_daily_returns_todays_in_and_out(
    client: TestClient, auth_headers: dict[str, str], session: Session, line_id: int
) -> None:
    repo = CrossingRepository(session)
    repo.add(line_id=line_id, area_id="gate", ts=_noon(2024, 6, 12), direction="in", track_ref=1)
    repo.add(line_id=line_id, area_id="gate", ts=_noon(2024, 6, 12), direction="in", track_ref=2)
    repo.add(line_id=line_id, area_id="gate", ts=_noon(2024, 6, 12), direction="out", track_ref=3)
    session.commit()

    response = client.get(
        "/api/v1/history/entry-exit/daily",
        params={"area_id": "gate", "date": "2024-06-12"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "area_id": "gate",
        "date": "2024-06-12",
        "in_count": 2,
        "out_count": 1,
        "net": 1,
    }


def test_entry_exit_daily_excludes_other_days(
    client: TestClient, auth_headers: dict[str, str], session: Session, line_id: int
) -> None:
    repo = CrossingRepository(session)
    repo.add(line_id=line_id, area_id="gate", ts=_noon(2024, 6, 12), direction="in", track_ref=1)
    repo.add(line_id=line_id, area_id="gate", ts=_noon(2024, 6, 13), direction="in", track_ref=2)
    session.commit()

    response = client.get(
        "/api/v1/history/entry-exit/daily",
        params={"area_id": "gate", "date": "2024-06-12"},
        headers=auth_headers,
    )

    assert response.json()["in_count"] == 1  # the 13th's crossing is excluded


def test_entry_exit_daily_empty_day_is_zeros(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.get(
        "/api/v1/history/entry-exit/daily",
        params={"area_id": "gate", "date": "2024-06-12"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "area_id": "gate",
        "date": "2024-06-12",
        "in_count": 0,
        "out_count": 0,
        "net": 0,
    }


def test_entry_exit_daily_rejects_malformed_date(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.get(
        "/api/v1/history/entry-exit/daily",
        params={"area_id": "gate", "date": "12-06-2024"},
        headers=auth_headers,
    )

    assert response.status_code == 422
