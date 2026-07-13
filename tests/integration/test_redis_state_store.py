"""Integration tests for the Redis-backed state store (app.services.redis_state_store).

These exercise the store end to end through a Redis client, asserting the same
dataclasses from :mod:`app.services.state_store` survive the encode/decode trip
and that two stores sharing one server see each other's writes. Hermetic: a
shared ``fakeredis`` server stands in for Redis, patched in via the same
``redis.Redis.from_url`` entry point the store uses, so no live server is
required. Each test owns its data.
"""

from __future__ import annotations

from collections.abc import Iterator

import fakeredis
import pytest

from app.services.redis_state_store import RedisStateStore
from app.services.state_store import CameraHealthState


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[RedisStateStore]:
    """A RedisStateStore backed by an isolated in-memory fakeredis server."""
    server = fakeredis.FakeServer()

    def _from_url(url: str, *args: object, **kwargs: object) -> fakeredis.FakeStrictRedis:
        return fakeredis.FakeStrictRedis(server=server)

    monkeypatch.setattr("redis.Redis.from_url", _from_url)
    yield RedisStateStore("redis://localhost:6379/0")


def test_set_get_occupancy_roundtrip(store: RedisStateStore) -> None:
    store.set_occupancy(zone_id=5, count=3, dt_space_id="space-a", ts=1000.0)

    state = store.get_occupancy(5)

    assert state is not None
    assert (state.zone_id, state.count, state.dt_space_id, state.ts) == (5, 3, "space-a", 1000.0)


def test_get_occupancy_missing_returns_none(store: RedisStateStore) -> None:
    assert store.get_occupancy(404) is None


def test_all_occupancy_returns_all_set_zones(store: RedisStateStore) -> None:
    store.set_occupancy(zone_id=1, count=2, dt_space_id=None, ts=1000.0)
    store.set_occupancy(zone_id=2, count=4, dt_space_id=None, ts=1000.0)

    zone_ids = {s.zone_id for s in store.all_occupancy()}

    assert zone_ids == {1, 2}


def test_set_counts_with_lines_records_per_line_breakdown(store: RedisStateStore) -> None:
    store.set_counts(
        area_id="area-1", in_count=10, out_count=4, net=6, ts=1000.0, lines={7: (10, 4)}
    )

    state = store.get_counts("area-1")

    assert state is not None
    assert (state.in_count, state.out_count, state.net) == (10, 4, 6)
    assert state.lines[7].in_count == 10
    assert state.lines[7].out_count == 4



def test_set_get_camera_health_roundtrip(store: RedisStateStore) -> None:
    store.set_camera_health(
        CameraHealthState(camera_id=3, fps=15.0, queue_depth=1, healthy=False, ts=1000.0)
    )

    state = store.get_camera_health(3)

    assert state is not None
    assert state.healthy is False
    assert state.fps == 15.0
    assert state.queue_depth == 1


def test_total_occupancy_sums_zone_counts(store: RedisStateStore) -> None:
    store.set_occupancy(zone_id=1, count=3, dt_space_id=None, ts=1000.0)
    store.set_occupancy(zone_id=2, count=5, dt_space_id=None, ts=1000.0)

    assert store.total_occupancy() == 8


def test_clear_resets_all_maps(store: RedisStateStore) -> None:
    store.set_occupancy(zone_id=1, count=3, dt_space_id=None, ts=1000.0)
    store.set_counts(area_id="a", in_count=1, out_count=0, net=1, ts=1000.0)
    store.set_camera_health(CameraHealthState(camera_id=1, ts=1000.0))

    store.clear()

    assert store.all_occupancy() == []
    assert store.all_counts() == []
    assert store.all_camera_health() == []
    assert store.total_occupancy() == 0


def test_two_stores_share_one_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """A writer process and a reader process see the same state via Redis."""
    server = fakeredis.FakeServer()
    monkeypatch.setattr(
        "redis.Redis.from_url",
        lambda url, *a, **k: fakeredis.FakeStrictRedis(server=server),
    )
    writer = RedisStateStore("redis://localhost:6379/0")
    reader = RedisStateStore("redis://localhost:6379/0")

    writer.set_occupancy(zone_id=7, count=4, dt_space_id="space-z", ts=1000.0)

    state = reader.get_occupancy(7)
    assert state is not None
    assert (state.count, state.dt_space_id) == (4, "space-z")
