"""Redis-backed current-state read model for scaled-out deployments (HLD 6.5).

Replaces the in-memory maps of :class:`~app.services.state_store.StateStore`
with Redis hashes so the read model is shared across processes (workers write,
the API reads) while exposing the exact same public methods. Each metric family
is one Redis hash:

* ``occupancy``      field = ``zone_id``   value = JSON of :class:`OccupancyState`
* ``counts``         field = ``area_id``   value = JSON of :class:`AreaCountState`
* ``waiting``        field = ``zone_id``   value = JSON of :class:`WaitingState`
* ``camera_health``  field = ``camera_id`` value = JSON of :class:`CameraHealthState`

The dataclasses from :mod:`app.services.state_store` are reused verbatim (never
re-modelled); only their fields are JSON-encoded for the wire. The Redis
connection is created lazily, so this module imports without a live server.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

from app.services.state_store import (
    AreaCountState,
    CameraHealthState,
    LineCountState,
    OccupancyState,
    WaitingState,
)
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from redis import Redis

logger = get_logger("services.redis_state_store")

# Hash names, namespaced so the read model never collides with other Redis keys.
_PREFIX = "camera-analytics:state"
_OCCUPANCY_HASH = f"{_PREFIX}:occupancy"
_COUNTS_HASH = f"{_PREFIX}:counts"
_WAITING_HASH = f"{_PREFIX}:waiting"
_HEALTH_HASH = f"{_PREFIX}:camera_health"


def _encode_occupancy(state: OccupancyState) -> str:
    return json.dumps(dataclasses.asdict(state))


def _decode_occupancy(raw: bytes | str) -> OccupancyState:
    return OccupancyState(**json.loads(raw))


def _encode_waiting(state: WaitingState) -> str:
    return json.dumps(dataclasses.asdict(state))


def _decode_waiting(raw: bytes | str) -> WaitingState:
    return WaitingState(**json.loads(raw))


def _encode_health(state: CameraHealthState) -> str:
    return json.dumps(dataclasses.asdict(state))


def _decode_health(raw: bytes | str) -> CameraHealthState:
    return CameraHealthState(**json.loads(raw))


def _encode_counts(state: AreaCountState) -> str:
    data: dict[str, Any] = dataclasses.asdict(state)
    # ``lines`` is keyed by int; JSON object keys must be strings — round-trip
    # via string keys and restore on decode.
    data["lines"] = {str(lid): line for lid, line in data["lines"].items()}
    return json.dumps(data)


def _decode_counts(raw: bytes | str) -> AreaCountState:
    data: dict[str, Any] = json.loads(raw)
    lines = {
        int(lid): LineCountState(**line) for lid, line in data.pop("lines", {}).items()
    }
    state = AreaCountState(**data)
    state.lines = lines
    return state


class RedisStateStore:
    """Redis-backed current-state cache matching :class:`StateStore`'s surface.

    Every accessor reconstructs the same dataclasses returned by the in-memory
    store, so API routes and projectors are agnostic to which store backs them.
    """

    def __init__(self, redis_url: str) -> None:
        """Build a Redis-backed store.

        The connection is created lazily on first access, so the store is
        importable and constructible without a live Redis server.

        :param redis_url: Redis connection URL (``redis://host:port/db``).
        """
        self._redis_url = redis_url
        self._client: Redis | None = None

    def _get_client(self) -> Redis:
        """Return the lazily created Redis client. Lazy-imports ``redis``."""
        if self._client is None:
            from redis import Redis

            self._client = Redis.from_url(self._redis_url)
        return self._client

    # --- Occupancy ---------------------------------------------------------
    def set_occupancy(self, zone_id: int, count: int, dt_space_id: str | None, ts: float) -> None:
        state = OccupancyState(zone_id, count, dt_space_id, ts)
        self._get_client().hset(_OCCUPANCY_HASH, str(zone_id), _encode_occupancy(state))

    def get_occupancy(self, zone_id: int) -> OccupancyState | None:
        raw = self._get_client().hget(_OCCUPANCY_HASH, str(zone_id))
        return _decode_occupancy(raw) if raw is not None else None

    def all_occupancy(self) -> list[OccupancyState]:
        return [_decode_occupancy(v) for v in self._get_client().hvals(_OCCUPANCY_HASH)]

    # --- Entry/Exit counts -------------------------------------------------
    def set_counts(
        self,
        area_id: str,
        in_count: int,
        out_count: int,
        net: int,
        ts: float,
        lines: dict[int, tuple[int, int]] | None = None,
    ) -> None:
        state = AreaCountState(area_id, in_count, out_count, net, ts)
        if lines:
            state.lines = {lid: LineCountState(lid, i, o) for lid, (i, o) in lines.items()}
        self._get_client().hset(_COUNTS_HASH, area_id, _encode_counts(state))

    def get_counts(self, area_id: str) -> AreaCountState | None:
        raw = self._get_client().hget(_COUNTS_HASH, area_id)
        return _decode_counts(raw) if raw is not None else None

    def all_counts(self) -> list[AreaCountState]:
        return [_decode_counts(v) for v in self._get_client().hvals(_COUNTS_HASH)]

    # --- Waiting -----------------------------------------------------------
    def set_waiting(
        self, zone_id: int, current_waits: int, avg_dwell_s: float, dt_space_id: str | None, ts: float
    ) -> None:
        state = WaitingState(zone_id, current_waits, avg_dwell_s, dt_space_id, ts)
        self._get_client().hset(_WAITING_HASH, str(zone_id), _encode_waiting(state))

    def get_waiting(self, zone_id: int) -> WaitingState | None:
        raw = self._get_client().hget(_WAITING_HASH, str(zone_id))
        return _decode_waiting(raw) if raw is not None else None

    def all_waiting(self) -> list[WaitingState]:
        return [_decode_waiting(v) for v in self._get_client().hvals(_WAITING_HASH)]

    # --- Camera health -----------------------------------------------------
    def set_camera_health(self, health: CameraHealthState) -> None:
        self._get_client().hset(_HEALTH_HASH, str(health.camera_id), _encode_health(health))

    def get_camera_health(self, camera_id: int) -> CameraHealthState | None:
        raw = self._get_client().hget(_HEALTH_HASH, str(camera_id))
        return _decode_health(raw) if raw is not None else None

    def all_camera_health(self) -> list[CameraHealthState]:
        return [_decode_health(v) for v in self._get_client().hvals(_HEALTH_HASH)]

    def total_occupancy(self) -> int:
        return sum(state.count for state in self.all_occupancy())

    def clear(self) -> None:
        self._get_client().delete(
            _OCCUPANCY_HASH, _COUNTS_HASH, _WAITING_HASH, _HEALTH_HASH
        )
