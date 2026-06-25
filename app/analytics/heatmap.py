"""Foot-traffic heatmap accumulator (HLD 7).

Accumulates ground-contact points of tracked people into a coarse spatial grid
over a flush interval, then emits a :class:`GridSnapshot` of the accumulated
cell weights. The accumulator is pure: it holds only in-memory grid state. The
engine persists each flushed snapshot and publishes the corresponding
``HeatmapFlushed`` event.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.analytics.base import BaseCalculator
from app.domain.interfaces import Clock
from app.domain.models import TrackedDetection
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GridSnapshot:
    """An immutable snapshot of accumulated heatmap cells for one interval.

    Attributes:
        camera_id: Camera the snapshot belongs to.
        ts_bucket: Timestamp (float epoch seconds) at which the interval flushed.
        cols: Number of grid columns.
        rows: Number of grid rows.
        cells: Sparse ``"row,col" -> accumulated weight`` mapping. Cells with no
            traffic are absent rather than zero-valued.
    """

    camera_id: int
    ts_bucket: float
    cols: int
    rows: int
    cells: dict[str, float]


class HeatmapAccumulator(BaseCalculator):
    """Accumulates tracked ground points into a grid and flushes periodically.

    Holds a sparse cell accumulator and the timestamp of the last flush. Each
    tracked detection's bottom-centre point is mapped to a clamped grid cell and
    its weight incremented. :meth:`maybe_flush` returns a snapshot and resets the
    accumulator once ``flush_interval_seconds`` have elapsed since the last flush.
    """

    def __init__(
        self,
        camera_id: int,
        cols: int,
        rows: int,
        flush_interval_seconds: float,
        clock: Clock,
    ) -> None:
        """Initialise the accumulator.

        Args:
            camera_id: Camera this accumulator serves.
            cols: Grid columns (clamped to a minimum of 1).
            rows: Grid rows (clamped to a minimum of 1).
            flush_interval_seconds: Seconds between snapshot flushes.
            clock: Shared pipeline time source.

        Raises:
            ValueError: If ``flush_interval_seconds`` is not positive.
        """
        super().__init__(clock)
        if flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be positive")
        self._camera_id = camera_id
        self._cols = max(1, cols)
        self._rows = max(1, rows)
        self._flush_interval_seconds = flush_interval_seconds
        self._cells: dict[str, float] = {}
        self._last_flush: float | None = None

    def accumulate(
        self,
        tracked: list[TrackedDetection],
        frame_w: int,
        frame_h: int,
        ts: float,
    ) -> None:
        """Add one frame's tracked ground points to the grid.

        Args:
            tracked: Tracked detections for the frame.
            frame_w: Frame width in pixels (must be positive).
            frame_h: Frame height in pixels (must be positive).
            ts: Frame timestamp (float epoch seconds). Used to seed the flush
                clock on the first call.

        Raises:
            ValueError: If ``frame_w`` or ``frame_h`` is not positive.
        """
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError("frame_w and frame_h must be positive")
        if self._last_flush is None:
            self._last_flush = ts

        for det in tracked:
            point = det.bbox.bottom_center
            col = self._clamp(int(point.x / frame_w * self._cols), self._cols)
            row = self._clamp(int(point.y / frame_h * self._rows), self._rows)
            key = f"{row},{col}"
            self._cells[key] = self._cells.get(key, 0.0) + 1.0

    def maybe_flush(self, ts: float) -> GridSnapshot | None:
        """Return and reset a snapshot when the flush interval has elapsed.

        Args:
            ts: Current timestamp (float epoch seconds).

        Returns:
            A :class:`GridSnapshot` of the accumulated cells when at least
            ``flush_interval_seconds`` have elapsed since the last flush, after
            which the accumulator is reset; otherwise ``None``.
        """
        if self._last_flush is None:
            self._last_flush = ts
            return None
        if ts - self._last_flush < self._flush_interval_seconds:
            return None

        snapshot = GridSnapshot(
            camera_id=self._camera_id,
            ts_bucket=ts,
            cols=self._cols,
            rows=self._rows,
            cells=dict(self._cells),
        )
        self._cells = {}
        self._last_flush = ts
        return snapshot

    @staticmethod
    def _clamp(value: int, upper_exclusive: int) -> int:
        """Clamp a grid index into ``[0, upper_exclusive - 1]``."""
        if value < 0:
            return 0
        if value >= upper_exclusive:
            return upper_exclusive - 1
        return value
