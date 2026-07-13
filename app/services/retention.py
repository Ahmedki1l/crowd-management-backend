"""Prune history past its retention window (NFR-04).

``RetentionConfig`` has existed since the first commit and, until now, was read by no code
at all — while ``README.md`` and ``docs/TRACEABILITY.md`` both asserted a retention policy
was in force. It was not: nothing ever deleted a row. This is the reader that makes the
config true.

Runs on the same thread as the hourly rollup (both are slow, periodic, and touch the same
tables), once per pass. A day's granularity means there is nothing to gain from running it
more often than that.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.config.schema import RetentionConfig
from app.db.repositories.crossing_repo import CrossingRepository
from app.db.repositories.occupancy_repo import Grain, OccupancyRepository
from app.db.session import session_scope
from app.utils.logging import get_logger

logger = get_logger(__name__)


def prune(cfg: RetentionConfig, now: datetime) -> dict[str, int]:
    """Delete history older than its retention window. Returns rows removed per table.

    A window of ``0`` means "keep forever" — the table is skipped rather than emptied,
    because a misread config that silently deleted everything would be unrecoverable.
    """
    removed: dict[str, int] = {}

    with session_scope() as session:
        occupancy = OccupancyRepository(session)
        crossings = CrossingRepository(session)

        if cfg.occupancy_minute_days > 0:
            cutoff = now - timedelta(days=cfg.occupancy_minute_days)
            removed["occupancy_minute"] = occupancy.delete_before(Grain.MINUTE, cutoff)

        if cfg.occupancy_hour_days > 0:
            cutoff = now - timedelta(days=cfg.occupancy_hour_days)
            removed["occupancy_hour"] = occupancy.delete_before(Grain.HOUR, cutoff)

        if cfg.crossing_days > 0:
            cutoff = now - timedelta(days=cfg.crossing_days)
            removed["crossing_events"] = crossings.delete_before(cutoff)

    total = sum(removed.values())
    if total:
        logger.info(
            "pruned history past retention",
            extra={"event": "retention_pruned", "removed": removed},
        )
    return removed
