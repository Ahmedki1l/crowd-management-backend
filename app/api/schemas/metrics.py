"""Live-metric and history response schemas (HLD 8.2-8.4)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


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






class StateOut(BaseModel):
    """Consolidated snapshot for the DT initial load (GET /state)."""

    occupancy: list[OccupancyOut] = []
    entry_exit: list[EntryExitOut] = []
    ts: float


class StatsOut(BaseModel):
    cameras_total: int
    cameras_healthy: int
    zones_total: int
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


# --- Occupancy history (pre-aggregated rollups) -----------------------------
class OccupancyBucket(BaseModel):
    """One aggregated bucket of occupancy history.

    ``avg`` is the mean occupancy over the bucket and ``peak`` the highest value seen in
    it. Chart the average, but alarm on the peak: an hourly mean of 12 can contain a
    90-person spike.
    """

    ts: datetime
    avg: float
    peak: int
    min: int
    # Samples that fed this bucket, at 1 Hz: 60 = a fully-covered minute, 3600 a full
    # hour. A low value means frames were dropped or a camera was down, so the bucket is
    # real but thinly observed.
    samples: int
    cameras_healthy: int
    cameras_total: int


class OccupancySeriesOut(BaseModel):
    """Occupancy history for one logical space."""

    space_id: str
    floor: str | None = None
    points: list[OccupancyBucket]


class OccupancyHistoryOut(BaseModel):
    """Bucketed occupancy history, one series per space."""

    bucket: str
    start: datetime
    end: datetime
    series: list[OccupancySeriesOut]


class FloorBucket(OccupancyBucket):
    """One floor bucket: its spaces summed, plus how many of them actually reported.

    A floor total is the sum of the spaces present in that bucket. A space is absent only
    when it was fully blind (all its cameras down) for the whole bucket, so a bucket with
    ``spaces_reporting < spaces_expected`` is summing fewer areas — its lower total is a
    coverage gap, not necessarily a real drop in people. These two counts make that
    explicit so a dip is never silently mistaken for people leaving.
    """

    spaces_reporting: int
    spaces_expected: int


class FloorSeriesOut(BaseModel):
    """Occupancy history for one floor: its spaces summed per bucket."""

    floor: str
    space_ids: list[str]
    points: list[FloorBucket]


class FloorHistoryOut(BaseModel):
    """Bucketed occupancy history aggregated to floors."""

    bucket: str
    start: datetime
    end: datetime
    floors: list[FloorSeriesOut]
