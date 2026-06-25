"""Heat-map grid data access (HLD 6.5, 7, 9).

Stores pre-aggregated sparse density grids per camera per time bucket and
answers the "summed grid over a range" query used to render overlays. Cells are
JSON ``{"r,c": weight}`` maps; summing happens in Python. Repositories flush but
never commit.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.timeseries import HeatmapGrid
from app.utils.logging import get_logger

_logger = get_logger(__name__)


class HeatmapRepository:
    """Access to the ``heatmap_grid`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_grid(
        self,
        camera_id: int,
        ts_bucket: datetime,
        cols: int,
        rows: int,
        grid: dict[str, float],
    ) -> HeatmapGrid:
        """Persist one sparse density grid for a camera/time-bucket and return it."""
        row = HeatmapGrid(
            camera_id=camera_id,
            ts_bucket=ts_bucket,
            cols=cols,
            rows=rows,
            grid=grid,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def query_range(
        self,
        camera_id: int,
        frm: datetime,
        to: datetime,
    ) -> tuple[int, int, dict[str, float]]:
        """Sum the sparse grids for ``camera_id`` over ``[frm, to)``.

        :returns: ``(cols, rows, summed_cells)``. ``cols``/``rows`` come from the
            latest grid in the range. When the range holds no grids, returns
            ``(0, 0, {})``. Grids whose dimensions disagree with the latest grid
            are still summed cell-by-cell (cell keys are dimension-independent),
            and a warning is logged because mixed grid sizes usually mean a
            camera was re-calibrated mid-range.
        """
        grids = self._session.scalars(
            select(HeatmapGrid)
            .where(
                HeatmapGrid.camera_id == camera_id,
                HeatmapGrid.ts_bucket >= frm,
                HeatmapGrid.ts_bucket < to,
            )
            .order_by(HeatmapGrid.ts_bucket)
        ).all()

        if not grids:
            return (0, 0, {})

        latest = grids[-1]
        cols, rows = latest.cols, latest.rows
        summed: dict[str, float] = {}
        mixed_dims = False
        for grid_row in grids:
            if grid_row.cols != cols or grid_row.rows != rows:
                mixed_dims = True
            for cell, weight in grid_row.grid.items():
                summed[cell] = summed.get(cell, 0.0) + weight

        if mixed_dims:
            _logger.warning(
                "heatmap grids have mixed dimensions in range; using latest grid dims",
                extra={"camera_id": camera_id},
            )
        return (cols, rows, summed)
