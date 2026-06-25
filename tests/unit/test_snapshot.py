"""Unit tests for snapshot path safety and resolution (app.utils.snapshot)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.utils.snapshot import resolve_snapshot, safe_join


def test_safe_join_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        safe_join(tmp_path, "../escape.jpg")


def test_safe_join_accepts_normal_relative_path(tmp_path: Path) -> None:
    resolved = safe_join(tmp_path, "cam1/42.jpg")

    assert resolved == (tmp_path.resolve() / "cam1" / "42.jpg")


def test_resolve_snapshot_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_snapshot(tmp_path, "cam1/missing.jpg")
