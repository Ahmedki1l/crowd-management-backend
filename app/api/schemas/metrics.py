"""Live-metric, history, alert, and heat-map response schemas (HLD 8.2-8.4)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class OccupancyOut(BaseModel):
    zone_id: int
    dt_space_id: str | None = None
    count: int
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
