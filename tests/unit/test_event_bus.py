"""Unit tests for the in-process event bus (app.events.event_bus).

Async tests rely on pytest-asyncio's auto mode (configured in pyproject) so the
running loop is the same one ``subscribe()`` binds the subscription to.
"""

from __future__ import annotations

import asyncio

from app.events.event_bus import InMemoryEventBus
from app.events.events import OccupancyUpdate


def _occupancy(zone_id: int, count: int) -> OccupancyUpdate:
    return OccupancyUpdate(ts=1000.0, zone_id=zone_id, count=count)


def test_sync_handler_receives_published_event() -> None:
    bus = InMemoryEventBus()
    received: list[OccupancyUpdate] = []
    bus.subscribe_sync(received.append)

    event = _occupancy(zone_id=5, count=2)
    bus.publish(event)

    assert received == [event]


async def test_async_subscription_yields_published_event() -> None:
    bus = InMemoryEventBus()
    subscription = bus.subscribe()
    event = _occupancy(zone_id=5, count=2)

    bus.publish(event)
    received = await asyncio.wait_for(subscription.__anext__(), timeout=1.0)

    assert received is event


async def test_topic_filtered_subscription_ignores_other_zone() -> None:
    bus = InMemoryEventBus()
    subscription = bus.subscribe(topics={"zone:5"})

    bus.publish(_occupancy(zone_id=9, count=1))  # filtered out
    matching = _occupancy(zone_id=5, count=3)
    bus.publish(matching)

    # The zone:9 event must be skipped; the first item delivered is zone:5.
    received = await asyncio.wait_for(subscription.__anext__(), timeout=1.0)

    assert received is matching
