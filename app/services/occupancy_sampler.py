"""Sample live occupancy at a fixed rate and write per-minute rollups (HLD 9).

Why a sampler rather than a rollup over stored samples
------------------------------------------------------
Occupancy is a *step function* — a value holds until it changes. The calculators emit
only on change, so the event stream is unevenly spaced, and a plain ``sum/count`` over it
is wrong: a zone empty for 55 minutes (one event) then full for 5 (ten events) would
average as if the two states had equal weight.

Rather than reconstruct time-weights from a sparse log, this samples the live read model
at a **fixed 1 Hz cadence**. Evenly-spaced samples make the arithmetic mean exactly the
time-weighted mean, so the aggregation is a plain sum and count, and there is no raw
sample table to grow, prune, or get wrong.

Why it samples spaces, not zones
--------------------------------
A logical space (``dt_space_id``) is covered by several cameras, each contributing its own
zone. No single camera's pipeline sees the whole space, so the sum can only be taken here,
where every camera's state converges. It is also the only key stable enough to hang history
on: zone rows are deleted whenever someone redraws a polygon.

Coverage, not just a number
---------------------------
A dead camera's last count would otherwise sit in the read model forever, silently frozen.
Only zones whose camera is currently healthy are summed, and every row records how many of
the space's cameras were healthy — so a degraded measurement is visible as degraded rather
than passing for a real drop in occupancy.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from app.db.repositories.occupancy_repo import Grain, OccupancyRepository, RollupRow
from app.db.repositories.zone_repo import ZoneRepository
from app.db.session import session_scope
from app.services.state_store import StateStore
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Sampling cadence. 60 samples per minute bucket; `samples` on the row doubles as the
# coverage metric, so this constant is what "full coverage" means.
_SAMPLE_INTERVAL_S = 1.0

# How often the zone -> (space, floor, camera) map is refreshed from the DB. Zones and
# cameras are config, not hot data; re-reading them every tick would be pure waste.
_MAP_REFRESH_S = 60.0


class _SpaceAccumulator:
    """Running sum/count/min/max for one space within the current bucket."""

    __slots__ = ("total", "n", "peak", "low", "cameras_healthy", "cameras_total")

    def __init__(self) -> None:
        self.total = 0
        self.n = 0
        self.peak = 0
        self.low: int | None = None
        self.cameras_healthy = 0
        self.cameras_total = 0

    def add(self, count: int, healthy: int, total: int) -> None:
        self.total += count
        self.n += 1
        self.peak = max(self.peak, count)
        self.low = count if self.low is None else min(self.low, count)
        # Coverage is reported as the best the space achieved during the bucket: a
        # momentary health blip should not make an otherwise-good hour look untrusted.
        self.cameras_healthy = max(self.cameras_healthy, healthy)
        self.cameras_total = max(self.cameras_total, total)

    def to_row(self, space_id: str, floor: str | None, bucket_ts: datetime) -> RollupRow:
        return RollupRow(
            space_id=space_id,
            floor=floor,
            bucket_ts=bucket_ts,
            avg=self.total / self.n,
            peak=self.peak,
            min=self.low or 0,
            samples=self.n,
            cameras_healthy=self.cameras_healthy,
            cameras_total=self.cameras_total,
        )


class OccupancySampler:
    """Background thread: sample the read model at 1 Hz, flush one row per space per minute."""

    def __init__(self, store: StateStore) -> None:
        """Bind the sampler to the live-state cache it reads."""
        self._store = store
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # zone_id -> (space_id, floor, camera_id); rebuilt from the DB periodically.
        self._zone_map: dict[int, tuple[str, str | None, int]] = {}
        self._map_loaded_at: float = 0.0

        self._buckets: dict[str, _SpaceAccumulator] = {}
        self._floors: dict[str, str | None] = {}
        self._bucket_ts: datetime | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Launch the sampling thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="occupancy-sampler", daemon=True
        )
        self._thread.start()
        logger.info("occupancy sampler started", extra={"event": "sampler_started"})

    def stop(self) -> None:
        """Signal the thread to stop and wait for it, flushing the open bucket first."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=_SAMPLE_INTERVAL_S * 5)
        self._flush()
        logger.info("occupancy sampler stopped", extra={"event": "sampler_stopped"})

    # ------------------------------------------------------------------ #
    # Sampling loop
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        # `wait` doubles as the sleep, so a stop is observed immediately rather than
        # after the current tick.
        while not self._stop.wait(_SAMPLE_INTERVAL_S):
            try:
                self.tick(datetime.now(tz=UTC))
            except Exception:  # noqa: BLE001 - one bad tick must not kill the sampler
                logger.exception(
                    "occupancy sample failed", extra={"event": "sample_failed"}
                )

    def tick(self, now: datetime) -> None:
        """Take one sample, flushing the previous bucket first if the minute rolled over.

        Public so tests can drive it deterministically instead of waiting on wall time.
        """
        bucket = now.replace(second=0, microsecond=0)
        if self._bucket_ts is not None and bucket != self._bucket_ts:
            self._flush()
        self._bucket_ts = bucket

        self._refresh_zone_map(now)
        self._sample()

    def _sample(self) -> None:
        """Sum the healthy zones of each space and fold the result into its accumulator."""
        healthy = {h.camera_id: h.healthy for h in self._store.all_camera_health()}

        totals: dict[str, int] = {}
        seen_cameras: dict[str, set[int]] = {}
        live_cameras: dict[str, set[int]] = {}

        for state in self._store.all_occupancy():
            mapped = self._zone_map.get(state.zone_id)
            if mapped is None:
                continue  # zone not assigned to a space -> belongs to no history
            space_id, floor, camera_id = mapped
            self._floors[space_id] = floor
            seen_cameras.setdefault(space_id, set()).add(camera_id)

            # A camera with no health record yet has not reported a heartbeat, so it is
            # not yet trustworthy; treat it as down rather than assume the count is real.
            if not healthy.get(camera_id, False):
                continue
            live_cameras.setdefault(space_id, set()).add(camera_id)
            totals[space_id] = totals.get(space_id, 0) + state.count

        for space_id in seen_cameras:
            acc = self._buckets.setdefault(space_id, _SpaceAccumulator())
            acc.add(
                count=totals.get(space_id, 0),
                healthy=len(live_cameras.get(space_id, ())),
                total=len(seen_cameras[space_id]),
            )

    def _flush(self) -> None:
        """Write one minute row per space and reset the accumulators."""
        if not self._buckets or self._bucket_ts is None:
            self._buckets = {}
            return

        rows = [
            acc.to_row(space_id, self._floors.get(space_id), self._bucket_ts)
            for space_id, acc in self._buckets.items()
        ]
        self._buckets = {}

        with session_scope() as session:
            repo = OccupancyRepository(session)
            for row in rows:
                repo.upsert(Grain.MINUTE, row)

        logger.debug(
            "flushed occupancy minute buckets",
            extra={"event": "minute_flushed", "spaces": len(rows)},
        )

    # ------------------------------------------------------------------ #
    # Config map
    # ------------------------------------------------------------------ #
    def _refresh_zone_map(self, now: datetime) -> None:
        """Reload zone -> (space, floor, camera), at most once per ``_MAP_REFRESH_S``.

        Throttle on the load *timestamp*, not the map contents: a deployment with no
        space-assigned zones has a legitimately empty map, and gating on ``self._zone_map``
        (falsy when empty) would re-query the DB on every 1 Hz tick forever.
        ``_map_loaded_at`` starts at 0.0, so the first tick still loads.
        """
        epoch = now.timestamp()
        if self._map_loaded_at and epoch - self._map_loaded_at < _MAP_REFRESH_S:
            return

        with session_scope() as session:
            self._zone_map = {
                zone.id: (
                    zone.dt_space_id,
                    zone.camera.floor if zone.camera else None,
                    zone.camera_id,
                )
                for zone in ZoneRepository(session).list()
                if zone.dt_space_id
            }
        self._map_loaded_at = epoch
