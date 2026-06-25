"""Heat-map read service backing the heat-map router (HLD 7, 8.3).

Reads pre-aggregated density grids for a camera over a time range from the
:class:`~app.db.repositories.heatmap_repo.HeatmapRepository`, expands the sparse
``"r,c" -> weight`` map into the list of :class:`~app.api.schemas.metrics.HeatmapCell`
the API returns, and reports the queried window as epoch seconds.

``overlay_url`` is left ``None`` by default: rendering a PNG overlay needs Pillow
(a heavy, optional dependency) and is only produced on demand via
:meth:`HeatmapService.render_overlay` when a caller opts in with
``include_overlay``. The cell list is returned regardless of the overlay.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.api.schemas.metrics import HeatmapCell, HeatmapOut
from app.db.repositories.heatmap_repo import HeatmapRepository
from app.utils.logging import get_logger
from app.utils.snapshot import safe_join
from app.utils.timeutil import to_epoch

logger = get_logger(__name__)

# Subdirectory under the snapshot base dir that overlay PNGs are written to.
_OVERLAY_SUBDIR = "overlays"


class HeatmapService:
    """Query/read service for camera density heat-maps."""

    def __init__(self, session: Session) -> None:
        """Bind the service to a DB session.

        Args:
            session: An open SQLAlchemy session used for range queries.
        """
        self._session = session
        self._heatmap_repo = HeatmapRepository(session)

    def range(
        self,
        camera_id: int,
        frm: datetime,
        to: datetime,
        resolution: int | None = None,
        include_overlay: bool = False,
        base_dir: str | None = None,
    ) -> HeatmapOut:
        """Return the summed density grid for ``camera_id`` over ``[frm, to)``.

        Args:
            camera_id: The camera whose grids to aggregate.
            frm: Inclusive start of the window (timezone-aware UTC).
            to: Exclusive end of the window (timezone-aware UTC).
            resolution: Reserved for caller-requested down-sampling; the stored
                grids are pre-bucketed at capture resolution, so it does not
                re-bin here and is accepted for forward compatibility.
            include_overlay: When ``True`` (and ``base_dir`` is provided), render
                a heat overlay PNG and populate ``overlay_url`` with its relative
                path (FR-HM-03). Defaults to ``False``, preserving the historical
                behaviour where ``overlay_url`` is always ``None``. Rendering is
                best-effort: a missing Pillow dependency yields ``None`` rather
                than failing the read.
            base_dir: Snapshot root directory the overlay PNG is written under
                (``snapshots.dir``). Required for overlay rendering; ignored when
                ``include_overlay`` is ``False``.

        Returns:
            A :class:`HeatmapOut` with one cell per non-empty grid position and
            the queried window echoed as epoch seconds. ``cells`` is populated
            regardless of whether an overlay was requested or produced.
        """
        cols, rows, cells = self._heatmap_repo.query_range(camera_id, frm, to)
        expanded = self._expand_cells(camera_id, cells)
        overlay_url = self._maybe_render_overlay(
            camera_id=camera_id,
            cols=cols,
            rows=rows,
            cells=expanded,
            include_overlay=include_overlay,
            base_dir=base_dir,
        )
        return HeatmapOut(
            camera_id=camera_id,
            cols=cols,
            rows=rows,
            cells=expanded,
            overlay_url=overlay_url,
            from_ts=to_epoch(frm),
            to_ts=to_epoch(to),
        )

    def _maybe_render_overlay(
        self,
        camera_id: int,
        cols: int,
        rows: int,
        cells: list[HeatmapCell],
        include_overlay: bool,
        base_dir: str | None,
    ) -> str | None:
        """Render an overlay when opted in, else return ``None``.

        Centralises the ``include_overlay``/``base_dir`` guard so :meth:`range`
        stays linear. A request to render without a ``base_dir`` is a caller
        wiring error worth logging, not a silent no-op.
        """
        if not include_overlay:
            return None
        if base_dir is None:
            logger.warning(
                "heatmap overlay requested without a base_dir; skipping render",
                extra={"camera_id": camera_id},
            )
            return None
        return self.render_overlay(camera_id, cols, rows, cells, base_dir)

    def render_overlay(
        self,
        camera_id: int,
        cols: int,
        rows: int,
        cells: list[HeatmapCell],
        base_dir: str,
    ) -> str | None:
        """Render a heat overlay PNG for a density grid and return its rel path.

        Builds an ``cols``x``rows`` RGBA image whose pixels run from cool/low
        opacity at low density to warm/high opacity at high density. Cell weights
        are normalised against the maximum weight in ``cells`` so the warmest
        cell always saturates, making sparse and dense grids equally legible. The
        PNG is written under ``<base_dir>/overlays/cam<camera_id>.png`` and its
        path relative to ``base_dir`` is returned (mirroring
        :func:`app.utils.snapshot.save_snapshot`).

        Pillow is imported lazily so this module stays importable on the core
        dependency set. If Pillow is unavailable the failure is logged and
        ``None`` is returned, so an overlay request never crashes a read.

        Args:
            camera_id: Source camera; selects the ``cam<id>.png`` filename.
            cols: Grid width in cells (overlay pixel width).
            rows: Grid height in cells (overlay pixel height).
            cells: Non-empty grid cells to paint; cells outside ``cols``/``rows``
                are skipped defensively.
            base_dir: Snapshot root directory the PNG is written under.

        Returns:
            The PNG path relative to ``base_dir``
            (``overlays/cam<camera_id>.png``), or ``None`` if Pillow is
            unavailable or the grid has no positive density to render.
        """
        try:
            from PIL import Image  # lazy: heavy/optional, not needed to import
        except ImportError:
            logger.warning(
                "Pillow unavailable; skipping heatmap overlay render",
                extra={"camera_id": camera_id, "event": "overlay_pillow_missing"},
            )
            return None

        import numpy as np

        rgba = self._build_overlay_array(np, camera_id, cols, rows, cells)
        if rgba is None:
            return None

        rel_path = f"{_OVERLAY_SUBDIR}/cam{camera_id}.png"
        abs_path = safe_join(base_dir, rel_path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(abs_path, format="PNG")
        return rel_path

    @staticmethod
    def _build_overlay_array(
        np_mod,
        camera_id: int,
        cols: int,
        rows: int,
        cells: list[HeatmapCell],
    ) -> object | None:
        """Build the ``rows``x``cols``x4 RGBA ``uint8`` array for the overlay.

        Returns ``None`` when there is nothing to render (empty/degenerate grid
        or no positive weight), so the caller skips writing an empty PNG. The
        ``numpy`` module is passed in to keep the heavy import lazy and at the
        call site.
        """
        if cols <= 0 or rows <= 0 or not cells:
            logger.info(
                "no heatmap density to render; skipping overlay",
                extra={"camera_id": camera_id},
            )
            return None

        weights = np_mod.zeros((rows, cols), dtype=np_mod.float64)
        for cell in cells:
            if 0 <= cell.r < rows and 0 <= cell.c < cols:
                weights[cell.r, cell.c] = cell.weight

        max_weight = float(weights.max())
        if max_weight <= 0.0:
            logger.info(
                "no positive heatmap density to render; skipping overlay",
                extra={"camera_id": camera_id},
            )
            return None

        norm = np_mod.clip(weights / max_weight, 0.0, 1.0)
        # Warm ramp: red rises with density, blue falls; alpha tracks density so
        # empty cells stay transparent and the hottest cell is fully opaque.
        red = (norm * 255.0).astype(np_mod.uint8)
        blue = ((1.0 - norm) * 255.0).astype(np_mod.uint8)
        green = np_mod.zeros((rows, cols), dtype=np_mod.uint8)
        alpha = (norm * 255.0).astype(np_mod.uint8)
        return np_mod.dstack((red, green, blue, alpha))

    def _expand_cells(
        self,
        camera_id: int,
        cells: dict[str, float],
    ) -> list[HeatmapCell]:
        """Expand a sparse ``"r,c" -> weight`` map into ``HeatmapCell`` rows.

        Malformed keys are skipped with a warning rather than aborting the whole
        response, since one corrupt cell should not mask an otherwise valid grid.
        """
        out: list[HeatmapCell] = []
        for key, weight in cells.items():
            parsed = self._parse_cell_key(key)
            if parsed is None:
                logger.warning(
                    "skipping malformed heatmap cell key",
                    extra={"camera_id": camera_id},
                )
                continue
            r, c = parsed
            out.append(HeatmapCell(r=r, c=c, weight=weight))
        return out

    @staticmethod
    def _parse_cell_key(key: str) -> tuple[int, int] | None:
        """Parse a ``"r,c"`` cell key into ``(r, c)``, or ``None`` if malformed."""
        parts = key.split(",")
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None
