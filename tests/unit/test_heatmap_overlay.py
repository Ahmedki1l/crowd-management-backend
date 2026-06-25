"""Unit tests for heat-map overlay rendering (FR-HM-03).

Covers :class:`app.services.heatmap_service.HeatmapService`:

* ``range`` leaves ``overlay_url`` ``None`` by default and always returns cells,
  so the existing read contract is unchanged.
* ``range(include_overlay=True)`` populates ``overlay_url`` when Pillow can
  render a PNG, and degrades to ``None`` when it cannot — without dropping cells.
* ``render_overlay`` returns ``None`` for an empty/zero-density grid and writes a
  PNG (returning its relative path) when there is density to paint.

Tests that need a real PNG ``importorskip`` Pillow so they pass on the core
dependency set (no Pillow) and on a full set alike. The pixel-ramp test uses
only ``numpy`` and runs unconditionally, since the colour mapping is the
load-bearing logic.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.api.schemas.metrics import HeatmapCell
from app.services.heatmap_service import HeatmapService

_FRM = datetime(2024, 1, 1, tzinfo=UTC)
_TO = datetime(2024, 1, 2, tzinfo=UTC)


def _seed_grid(session: Session, camera_id: int, cells: dict[str, float]) -> None:
    """Insert one density grid so ``range`` has something to aggregate."""
    from app.db.repositories.heatmap_repo import HeatmapRepository

    HeatmapRepository(session).add_grid(
        camera_id=camera_id,
        ts_bucket=datetime(2024, 1, 1, 12, tzinfo=UTC),
        cols=4,
        rows=4,
        grid=cells,
    )
    session.commit()


def test_range_without_overlay_leaves_url_none_and_keeps_cells(session):
    _seed_grid(session, camera_id=1, cells={"1,1": 3.0})

    out = HeatmapService(session).range(1, _FRM, _TO)

    assert out.overlay_url is None
    assert out.cells == [HeatmapCell(r=1, c=1, weight=3.0)]


def test_range_with_overlay_but_no_base_dir_skips_render(session):
    _seed_grid(session, camera_id=1, cells={"1,1": 3.0})

    out = HeatmapService(session).range(1, _FRM, _TO, include_overlay=True)

    # No base_dir provided -> nothing rendered, but cells still returned.
    assert out.overlay_url is None
    assert out.cells == [HeatmapCell(r=1, c=1, weight=3.0)]


def test_render_overlay_returns_none_for_empty_grid(session, tmp_path):
    result = HeatmapService(session).render_overlay(
        camera_id=7, cols=4, rows=4, cells=[], base_dir=str(tmp_path)
    )

    assert result is None


def test_render_overlay_returns_none_for_zero_density(session, tmp_path):
    cells = [HeatmapCell(r=0, c=0, weight=0.0)]

    result = HeatmapService(session).render_overlay(
        camera_id=7, cols=4, rows=4, cells=cells, base_dir=str(tmp_path)
    )

    assert result is None


def test_render_overlay_returns_none_when_pillow_missing(session, tmp_path, monkeypatch):
    # Force the lazy ``from PIL import Image`` to fail regardless of whether
    # Pillow is installed, proving graceful degradation rather than a crash.
    import builtins

    real_import = builtins.__import__

    def _no_pil(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("Pillow not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pil)

    result = HeatmapService(session).render_overlay(
        camera_id=7,
        cols=4,
        rows=4,
        cells=[HeatmapCell(r=1, c=1, weight=5.0)],
        base_dir=str(tmp_path),
    )

    assert result is None


def test_render_overlay_writes_png_and_returns_relative_path(session, tmp_path):
    pytest.importorskip("PIL")

    rel = HeatmapService(session).render_overlay(
        camera_id=7,
        cols=4,
        rows=4,
        cells=[HeatmapCell(r=1, c=1, weight=5.0)],
        base_dir=str(tmp_path),
    )

    assert rel == "overlays/cam7.png"
    written = tmp_path / rel
    assert written.is_file()
    # It is a real PNG of the grid's pixel dimensions.
    from PIL import Image

    with Image.open(written) as img:
        assert img.mode == "RGBA"
        assert img.size == (4, 4)  # (width=cols, height=rows)


def test_range_with_overlay_and_base_dir_populates_url(session, tmp_path):
    pytest.importorskip("PIL")
    _seed_grid(session, camera_id=3, cells={"2,1": 4.0})

    out = HeatmapService(session).range(
        3, _FRM, _TO, include_overlay=True, base_dir=str(tmp_path)
    )

    assert out.overlay_url == "overlays/cam3.png"
    assert (tmp_path / out.overlay_url).is_file()
    assert out.cells == [HeatmapCell(r=2, c=1, weight=4.0)]


def test_overlay_array_normalises_to_max_with_warm_ramp(session):
    # Two cells: the hotter one must saturate (full red/alpha, no blue); the
    # cooler one sits halfway up the ramp. This is the FR-HM-03 "warmer = higher
    # density, normalise to max" contract, testable with numpy alone.
    cells = [
        HeatmapCell(r=0, c=0, weight=10.0),  # max -> norm 1.0
        HeatmapCell(r=0, c=1, weight=5.0),  # half -> norm 0.5
    ]

    rgba = HeatmapService._build_overlay_array(
        np, camera_id=1, cols=2, rows=1, cells=cells
    )

    assert rgba is not None
    assert rgba.shape == (1, 2, 4)
    assert rgba.dtype == np.uint8

    hot = rgba[0, 0]
    warm = rgba[0, 1]
    # Hottest cell: max red, max alpha, zero blue.
    assert (hot[0], hot[1], hot[2], hot[3]) == (255, 0, 0, 255)
    # Half cell: warmer than cold but not saturated, with partial alpha.
    assert warm[0] == 127  # red rises with density
    assert warm[2] == 127  # blue falls with density
    assert warm[3] == 127  # alpha tracks density
