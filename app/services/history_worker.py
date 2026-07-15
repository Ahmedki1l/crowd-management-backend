"""Periodic history maintenance (HLD 9): roll minutes up into hours, and prune what has
aged out.

Both jobs live on one thread because both are slow, periodic, and touch the same tables —
but they run on **different cadences**. The rollup checks each minute for a newly-complete
hour; retention windows are measured in *days*, so scanning for expired rows that often
would be pure overhead. Retention therefore runs hourly.

The rollup itself
-----------------

The hourly mean is a **sample-weighted** mean of the minute means, never a plain mean of
them. Averaging averages is only correct when every minute carries the same weight, and
they do not: a minute that started or ended mid-clock, or lost ticks to a starved thread,
holds fewer than 60 samples, and a plain mean would give it equal say.

    hour.avg = Σ(minute.avg × minute.samples) / Σ(minute.samples)

(The weight is time, not coverage: a camera-down minute is handled upstream — the sampler
drops fully-blind ticks and records worst-case coverage, so a degraded minute stays
visible through ``cameras_healthy`` rather than being silently down-weighted here.)

``peak`` composes as a max of maxes and ``min`` as a min of mins. **Coverage composes as
the worst** — ``cameras_healthy`` is the *fewest* healthy any contributing minute had — so
an hour that lost a camera for even one minute reports reduced coverage rather than hiding
it behind a best-case max.

The worker is **restartable and idempotent by construction**: it resumes from the newest
hour already written (the watermark), rolls up every complete hour since, and upserts on
``(space_id, bucket_ts)``. An outage, a restart, or a second process running the same
work therefore self-heals rather than duplicating or skipping buckets.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from app.config.settings import get_settings
from app.db.repositories.occupancy_repo import Grain, OccupancyRepository, RollupRow
from app.db.session import session_scope
from app.services.retention import prune
from app.utils.logging import get_logger

logger = get_logger(__name__)

# How often to look for newly-complete hours. An hour is long; checking each minute is
# ample and keeps the tick cheap.
_TICK_S = 60.0

# How often to prune. Retention windows are in days, so re-scanning for expired rows every
# tick would burn a table scan a minute to delete nothing.
_PRUNE_INTERVAL = timedelta(hours=1)

# How far back to reach when the hour table is empty (first run, or after a purge).
# Bounded so a first start cannot try to rebuild an unbounded history in one pass.
_COLD_START_LOOKBACK = timedelta(days=7)


class HistoryWorker:
    """Background thread owning the periodic history jobs: rollup, then retention."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_prune: datetime | None = None

    def start(self) -> None:
        """Launch the rollup thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="history-worker", daemon=True
        )
        self._thread.start()
        logger.info("history worker started", extra={"event": "history_worker_started"})

    def stop(self) -> None:
        """Signal the thread to stop and wait for it."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=_TICK_S)
        logger.info("history worker stopped", extra={"event": "history_worker_stopped"})

    def _run(self) -> None:
        # Roll up immediately on start so a restart does not wait a full tick to catch up.
        while True:
            now = datetime.now(tz=UTC)
            try:
                self.run_once(now)
            except Exception:  # noqa: BLE001 - a bad tick must not kill the worker
                logger.exception(
                    "hourly rollup failed", extra={"event": "rollup_failed"}
                )
            try:
                # Far less often than the rollup: a failure to prune must not stop it.
                self.prune_if_due(now)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "retention prune failed", extra={"event": "retention_failed"}
                )
            if self._stop.wait(_TICK_S):
                return

    def run_once(self, now: datetime) -> int:
        """Roll up every complete hour since the watermark. Returns rows written.

        Public so tests (and a backfill) can drive it directly rather than on a timer.
        """
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        with session_scope() as session:
            repo = OccupancyRepository(session)

            watermark = repo.latest_bucket(Grain.HOUR)
            # Re-roll the watermark hour itself: it may have been written from a
            # partially-complete set of minutes, and re-running only improves it.
            start = watermark if watermark is not None else current_hour - _COLD_START_LOOKBACK
            if start.tzinfo is None:
                start = start.replace(tzinfo=UTC)
            if start >= current_hour:
                return 0

            minutes = repo.query(Grain.MINUTE, start, current_hour)
            if not minutes:
                return 0

            rows = _aggregate_to_hours(minutes)
            for row in rows:
                repo.upsert(Grain.HOUR, row)

        if rows:
            logger.info(
                "rolled up occupancy hours",
                extra={"event": "hours_rolled_up", "rows": len(rows)},
            )
        return len(rows)


    def prune_if_due(self, now: datetime) -> bool:
        """Prune aged-out history, at most once per ``_PRUNE_INTERVAL``. Returns whether it ran.

        Public because the throttle *is* the contract: the worker ticks every minute to
        catch newly-complete hours, but retention windows are measured in days, so the
        promise is that pruning does not run at the tick cadence.
        """
        if self._last_prune is not None and now - self._last_prune < _PRUNE_INTERVAL:
            return False
        prune(get_settings().retention, now)
        self._last_prune = now
        return True


def _aggregate_to_hours(minutes: list[RollupRow]) -> list[RollupRow]:
    """Group minute rows into hour rows, weighting the mean by each minute's sample count."""
    groups: dict[tuple[str, datetime], list[RollupRow]] = defaultdict(list)
    for row in minutes:
        bucket = row.bucket_ts.replace(minute=0, second=0, microsecond=0)
        groups[(row.space_id, bucket)].append(row)

    hours: list[RollupRow] = []
    for (space_id, bucket_ts), members in groups.items():
        samples = sum(m.samples for m in members)
        if samples == 0:
            continue  # nothing was actually observed; do not invent a bucket
        hours.append(
            RollupRow(
                space_id=space_id,
                # The floor of the most recent contributing minute: if a camera was
                # re-assigned mid-hour, the newer answer is the better one.
                floor=max(members, key=lambda m: m.bucket_ts).floor,
                bucket_ts=bucket_ts,
                avg=sum(m.avg * m.samples for m in members) / samples,
                peak=max(m.peak for m in members),
                min=min(m.min for m in members),
                samples=samples,
                # Worst coverage across the hour's minutes, so a one-minute outage still
                # shows as reduced coverage rather than being hidden behind a best case.
                cameras_healthy=min(m.cameras_healthy for m in members),
                cameras_total=max(m.cameras_total for m in members),
            )
        )
    return sorted(hours, key=lambda r: (r.bucket_ts, r.space_id))
