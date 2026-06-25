"""The bus/store factories select Redis vs in-memory by ``cache.url`` (HLD 6.5).

When ``cache.url`` is unset the in-memory implementations are used (default,
single-process); when it is set the Redis-backed ones are selected. The Redis
selection must not require a live server, since the connection is deferred.
"""

from __future__ import annotations

from app.config.settings import get_settings
from app.events.event_bus import InMemoryEventBus, get_event_bus, reset_event_bus
from app.events.redis_bus import RedisEventBus
from app.services.redis_state_store import RedisStateStore
from app.services.state_store import StateStore, get_state_store, reset_state_store


def test_event_bus_defaults_to_in_memory_when_cache_unset() -> None:
    # conftest's _isolated_environment loads config with cache.url unset.
    assert isinstance(get_event_bus(), InMemoryEventBus)


def test_state_store_defaults_to_in_memory_when_cache_unset() -> None:
    assert isinstance(get_state_store(), StateStore)


def test_event_bus_selects_redis_when_cache_url_set() -> None:
    get_settings().cache.url = "redis://host:6379/0"
    reset_event_bus()

    bus = get_event_bus()

    assert isinstance(bus, RedisEventBus)
    bus.close()


def test_state_store_selects_redis_when_cache_url_set() -> None:
    get_settings().cache.url = "redis://host:6379/0"
    reset_state_store()

    store = get_state_store()

    assert isinstance(store, RedisStateStore)
