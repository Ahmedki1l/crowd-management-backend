"""Integration tests for :func:`app.engine.engine.build_engine` (degradation path).

These exercise the process-level orchestrator's *graceful-degradation* contract
(HLD constraint): the API process must still come up when there are no cameras to
run, or when the heavy perception backends (Ultralytics / OSNet runtimes) — or
their model artefacts — are absent, as they are in this core+dev environment.

The autouse ``_isolated_environment`` conftest fixture gives each test a fresh
in-memory database, cleared singletons and the example config (which has
``tracker.reid_enabled: true``). No camera, model runtime, or RTSP stream is ever
touched: ``build_engine`` reads cameras from the (empty or seeded) database and
its inference factories raise :class:`ImportError` because the heavy libraries are
not installed, which the engine catches to run zero pipelines.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.config.settings import load_config
from app.engine.engine import Engine, build_engine
from app.inference.reid import ReIDManager
from app.services.credentials import CredentialCipher


def test_build_engine_with_no_cameras_runs_zero_pipelines() -> None:
    """With no cameras in the database, the engine builds but runs no pipelines."""
    engine = build_engine()

    assert isinstance(engine, Engine)
    assert engine.pipeline_count == 0


def test_build_engine_start_stop_is_clean_with_zero_pipelines() -> None:
    """An engine with no pipelines starts and stops cleanly (idle supervisor)."""
    engine = build_engine()

    engine.start()
    try:
        # The supervisor is running but has nothing to watch; no pipeline exists.
        assert engine.pipeline_count == 0
    finally:
        engine.stop()

    # After stop the engine holds no pipelines and is safe to stop again.
    assert engine.pipeline_count == 0
    engine.stop()


def test_build_engine_degrades_to_zero_pipelines_when_backends_absent(
    session: Session, make_camera
) -> None:
    """A resolvable camera still yields zero pipelines when heavy deps are absent.

    The camera has a decryptable credential, so ``build_engine`` proceeds past
    credential resolution into the inference factories — where the missing
    Ultralytics/OSNet runtimes raise :class:`ImportError`. The engine catches it
    and degrades to zero pipelines rather than crashing the process.
    """
    camera = make_camera(session)
    camera.password_encrypted = CredentialCipher.from_env().encrypt("secret")
    session.commit()

    engine = build_engine()

    assert engine.pipeline_count == 0


def test_build_engine_wires_shared_reid_manager_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``tracker.reid_enabled`` is set, the engine gets one shared gallery."""
    cfg = load_config()
    cfg.tracker.reid_enabled = True
    monkeypatch.setattr("app.engine.engine.get_settings", lambda: cfg)

    engine = build_engine()

    assert isinstance(engine._reid_manager, ReIDManager)


def test_build_engine_omits_reid_manager_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Re-ID is disabled, no shared gallery is built (the contrast case)."""
    cfg = load_config()
    cfg.tracker.reid_enabled = False
    monkeypatch.setattr("app.engine.engine.get_settings", lambda: cfg)

    engine = build_engine()

    assert engine._reid_manager is None
