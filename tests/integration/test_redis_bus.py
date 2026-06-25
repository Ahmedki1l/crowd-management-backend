"""Integration tests for the Redis-backed event bus (app.events.redis_bus).

These exercise the bus end to end against its real serialization layer and
pub/sub listener thread, wired across two "process" instances. Hermetic: a
shared ``fakeredis`` server (with working pub/sub) stands in for Redis, patched
via the ``redis.Redis.from_url`` entry point the bus uses, so no live server is
required. Cross-process delivery is exercised by giving two bus instances the
same fakeredis server. Polling loops use a deadline (not fixed sleeps) so they
finish as soon as the event arrives.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator

import fakeredis
import pytest

from app.events.events import Event, EventBus
from app.events.redis_bus import RedisEventBus


def _occupancy(zone_id: int, count: int):
    from app.events.events import OccupancyUpdate

    return OccupancyUpdate(ts=1000.0, zone_id=zone_id, count=count)


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses (no fixed sleep)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def shared_server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def patch_redis(
    monkeypatch: pytest.MonkeyPatch, shared_server: fakeredis.FakeServer
) -> fakeredis.FakeServer:
    """Route every ``Redis.from_url`` to clients sharing one fakeredis server."""
    monkeypatch.setattr(
        "redis.Redis.from_url",
        lambda url, *a, **k: fakeredis.FakeStrictRedis(server=shared_server),
    )
    return shared_server


@pytest.fixture
def make_bus(patch_redis: fakeredis.FakeServer) -> Iterator[Callable[[], RedisEventBus]]:
    """Factory creating RedisEventBus instances, all closed on teardown."""
    created: list[RedisEventBus] = []

    def _make() -> RedisEventBus:
        bus = RedisEventBus("redis://localhost:6379/0")
        created.append(bus)
        return bus

    yield _make
    for bus in created:
        bus.close()


def test_implements_event_bus_protocol(make_bus: Callable[[], RedisEventBus]) -> None:
    assert isinstance(make_bus(), EventBus)


def test_local_sync_handler_receives_published_event(
    make_bus: Callable[[], RedisEventBus],
) -> None:
    bus = make_bus()
    received: list[Event] = []
    bus.subscribe_sync(received.append)

    event = _occupancy(zone_id=5, count=2)
    bus.publish(event)

    # Same-process sync handlers fire inline, exactly like the in-process bus.
    assert received == [event]


def test_self_published_event_dispatched_once(
    make_bus: Callable[[], RedisEventBus],
) -> None:
    bus = make_bus()
    received: list[Event] = []
    bus.subscribe_sync(received.append)

    bus.publish(_occupancy(zone_id=5, count=2))

    # Give the listener a chance to (wrongly) redeliver the self-published event.
    assert not _wait_until(lambda: len(received) > 1, timeout=0.3)
    assert len(received) == 1


def test_cross_process_sync_delivery(make_bus: Callable[[], RedisEventBus]) -> None:
    worker = make_bus()
    api = make_bus()
    received: list[Event] = []
    api.subscribe_sync(received.append)  # starts api's listener thread
    assert api.wait_until_listening(timeout=2.0)

    event = _occupancy(zone_id=8, count=3)
    worker.publish(event)

    assert _wait_until(lambda: len(received) == 1)
    delivered = received[0]
    assert delivered.zone_id == 8
    assert delivered.count == 3


async def test_async_subscription_yields_cross_process_event(
    make_bus: Callable[[], RedisEventBus],
) -> None:
    worker = make_bus()
    api = make_bus()
    subscription = api.subscribe()  # starts api's listener bound to this loop
    assert await asyncio.to_thread(api.wait_until_listening, 2.0)

    worker.publish(_occupancy(zone_id=5, count=2))
    received = await asyncio.wait_for(subscription.__anext__(), timeout=2.0)

    assert received.zone_id == 5
    assert received.count == 2
    await subscription.aclose()


async def test_async_subscription_topic_filter(
    make_bus: Callable[[], RedisEventBus],
) -> None:
    worker = make_bus()
    api = make_bus()
    subscription = api.subscribe(topics={"zone:5"})
    assert await asyncio.to_thread(api.wait_until_listening, 2.0)

    worker.publish(_occupancy(zone_id=9, count=1))  # filtered out
    worker.publish(_occupancy(zone_id=5, count=3))  # matches

    received = await asyncio.wait_for(subscription.__anext__(), timeout=2.0)
    assert received.zone_id == 5
    await subscription.aclose()


def test_publish_does_not_raise_when_redis_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Redis outage must not crash analytics; locals still receive the event."""

    class _BrokenClient:
        def publish(self, channel: str, data: str) -> int:
            raise ConnectionError("redis down")

        def pubsub(self, *a: object, **k: object) -> object:
            raise ConnectionError("redis down")

        def close(self) -> None:
            return None

    monkeypatch.setattr("redis.Redis.from_url", lambda url, *a, **k: _BrokenClient())
    bus = RedisEventBus("redis://localhost:6379/0")
    received: list[Event] = []
    bus.subscribe_sync(received.append)
    try:
        bus.publish(_occupancy(zone_id=1, count=1))  # must not raise
    finally:
        bus.close()

    assert len(received) == 1
