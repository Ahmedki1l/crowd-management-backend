"""Operational (health/readiness) API response schemas (HLD 8.4 / 15).

Kept here — with every other API DTO — rather than inline in the router, so the
single search location for response shapes stays authoritative.
"""

from __future__ import annotations

from pydantic import BaseModel


class ReadyOut(BaseModel):
    """Readiness probe response."""

    status: str
    db_ok: bool
    models_loaded: bool


class CameraHealthOut(BaseModel):
    """Per-camera health view derived from the live state store.

    Field-compatible with :class:`app.services.state_store.CameraHealthState` and
    the :class:`app.events.events.CameraHealth` event (one camera-health shape,
    projected per boundary).
    """

    camera_id: int
    fps: float
    last_frame_age_s: float
    queue_depth: int
    healthy: bool
    ts: float
