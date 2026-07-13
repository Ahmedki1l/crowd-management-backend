"""Integration tests for the historical time-series router (HLD 8.3).

Raw time-series rows (occupancy samples, crossing events) are inserted directly
through their repositories, then the matching ``/api/v1/history/*`` endpoints are
queried with an explicit ``from``/``to`` window and ``bucket`` width. Assertions
cover the bucketed aggregation each endpoint performs: mean occupancy and net
crossings.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.repositories.crossing_repo import CrossingRepository
from app.db.repositories.occupancy_repo import OccupancyRepository

# A clean hour boundary so 10:00/10:30 share one 1h bucket and 11:00 starts the next.
_BASE = datetime(2024, 6, 12, 10, 0, tzinfo=UTC)
_WINDOW = {
    "from": str(datetime(2024, 6, 12, 9, 0, tzinfo=UTC).timestamp()),
    "to": str(datetime(2024, 6, 12, 13, 0, tzinfo=UTC).timestamp()),
    "bucket": "1h",
}


def _at(hour: int, minute: int = 0) -> datetime:
    return _BASE.replace(hour=hour, minute=minute)


def test_occupancy_history_returns_mean_per_bucket(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    repo = OccupancyRepository(session)
    repo.add_sample(zone_id=1, ts=_at(10, 0), count=4)
    repo.add_sample(zone_id=1, ts=_at(10, 30), count=6)
    session.commit()

    response = client.get(
        "/api/v1/history/occupancy",
        params={"zone_id": 1, **_WINDOW},
        headers=auth_headers,
    )

    assert response.status_code == 200
    points = response.json()["points"]
    assert [p["value"] for p in points] == [5.0]


def test_occupancy_history_separates_distinct_buckets(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    repo = OccupancyRepository(session)
    repo.add_sample(zone_id=1, ts=_at(10, 0), count=4)
    repo.add_sample(zone_id=1, ts=_at(11, 0), count=8)
    session.commit()

    response = client.get(
        "/api/v1/history/occupancy",
        params={"zone_id": 1, **_WINDOW},
        headers=auth_headers,
    )

    assert [p["value"] for p in response.json()["points"]] == [4.0, 8.0]


def test_entry_exit_history_returns_net_per_bucket(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    repo = CrossingRepository(session)
    repo.add(line_id=1, area_id="lobby", ts=_at(10, 0), direction="in", track_ref=1)
    repo.add(line_id=1, area_id="lobby", ts=_at(10, 10), direction="in", track_ref=2)
    repo.add(line_id=1, area_id="lobby", ts=_at(10, 20), direction="out", track_ref=3)
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
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    repo = CrossingRepository(session)
    repo.add(line_id=1, area_id="gate", ts=_noon(2024, 6, 12), direction="in", track_ref=1)
    repo.add(line_id=1, area_id="gate", ts=_noon(2024, 6, 12), direction="in", track_ref=2)
    repo.add(line_id=1, area_id="gate", ts=_noon(2024, 6, 12), direction="out", track_ref=3)
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
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    repo = CrossingRepository(session)
    repo.add(line_id=1, area_id="gate", ts=_noon(2024, 6, 12), direction="in", track_ref=1)
    repo.add(line_id=1, area_id="gate", ts=_noon(2024, 6, 13), direction="in", track_ref=2)
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
