"""Live-metric, history, alert, and heat-map response schemas (HLD 8.2-8.4)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class OccupancyOut(BaseModel):
    zone_id: int
    dt_space_id: str | None = None
    count: int
    ts: float


class SpaceOccupancyOut(BaseModel):
    """Occupancy for one logical space (Digital Twin ``dt_space_id``).

    A space covered by several cameras is stored as one zone row per camera, all
    sharing a ``dt_space_id``; this is those per-camera counts summed into a
    single number so a multi-camera area reports one occupancy figure.
    """

    dt_space_id: str
    count: int
    zone_ids: list[int]
    ts: float


class FloorSpaceOccupancy(BaseModel):
    """One logical space's occupancy, nested inside :class:`FloorOccupancyOut`."""

    dt_space_id: str
    count: int
    zone_ids: list[int]


class FloorOccupancyOut(BaseModel):
    """Occupancy for one physical floor: total across all its zones, plus a
    per-space (``dt_space_id``) breakdown. A floor groups the spaces that group
    the zones — resolved via each zone's camera ``floor``."""

    floor: str
    count: int
    spaces: list[FloorSpaceOccupancy]
    ts: float


class LineCount(BaseModel):
    line_id: int
    in_count: int
    out_count: int


class EntryExitOut(BaseModel):
    area_id: str
    in_count: int
    out_count: int
    net: int
    lines: list[LineCount] = []
    ts: float


class DailyEntryExitOut(BaseModel):
    """Entry/exit totals for one area on one local calendar day.

    Read from the persisted crossing events, so it survives process restarts —
    unlike the live in-memory ``/entry-exit`` counter, which resets on restart.
    ``date`` is the local day the counts cover; ``net`` (``in - out``) is a
    convenience derived from the same day's crossings.
    """

    area_id: str
    date: str  # YYYY-MM-DD, in the server's local timezone
    in_count: int
    out_count: int
    net: int


class WaitingOut(BaseModel):
    zone_id: int
    dt_space_id: str | None = None
    current_waits: int
    avg_dwell_s: float
    ts: float


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    type: str
    zone_id: int | None = None
    camera_id: int | None = None
    ts: datetime
    detail: str
    snapshot_url: str | None = None
    status: str


class HeatmapCell(BaseModel):
    r: int
    c: int
    weight: float


class HeatmapOut(BaseModel):
    camera_id: int
    cols: int
    rows: int
    cells: list[HeatmapCell]
    overlay_url: str | None = None
    from_ts: float | None = None
    to_ts: float | None = None


class StateOut(BaseModel):
    """Consolidated snapshot for the DT initial load (GET /state)."""

    occupancy: list[OccupancyOut] = []
    entry_exit: list[EntryExitOut] = []
    waiting: list[WaitingOut] = []
    alerts: list[AlertOut] = []
    ts: float


class StatsOut(BaseModel):
    cameras_total: int
    cameras_healthy: int
    zones_total: int
    active_alerts: int
    total_occupancy: int
    ts: float


# --- History (bucketed time-series) ----------------------------------------
class TimeBucket(BaseModel):
    ts: datetime
    value: float


class HistorySeriesOut(BaseModel):
    key: str
    bucket: str
    points: list[TimeBucket]
