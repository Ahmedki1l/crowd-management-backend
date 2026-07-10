"""The engine restricts pipelines to the camera IP allowlist when configured.

Exercises :func:`app.engine.engine._resolve_target_ids` — the seam that decides
which cameras get a pipeline — across the empty-allowlist (all enabled),
allowlisted, still-requires-enabled, and explicit-override cases.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.engine.engine import _resolve_target_ids
from app.services.camera_service import CameraService


def test_empty_allowlist_runs_all_enabled_cameras(session: Session, make_camera) -> None:
    cam_a = make_camera(session, name="a", ip="10.0.0.1")
    make_camera(session, name="b", ip="10.0.0.2", enabled=False)
    cam_c = make_camera(session, name="c", ip="10.0.0.3")

    ids = _resolve_target_ids(CameraService(session), None, set())

    assert set(ids) == {cam_a.id, cam_c.id}  # disabled excluded; no IP filtering


def test_allowlist_restricts_to_listed_ips(session: Session, make_camera) -> None:
    cam_a = make_camera(session, name="a", ip="10.0.0.1")
    make_camera(session, name="b", ip="10.0.0.2")
    cam_c = make_camera(session, name="c", ip="10.0.0.3")

    ids = _resolve_target_ids(
        CameraService(session), None, {"10.0.0.1", "10.0.0.3"}
    )

    assert set(ids) == {cam_a.id, cam_c.id}


def test_allowlist_still_requires_enabled(session: Session, make_camera) -> None:
    make_camera(session, name="a", ip="10.0.0.1", enabled=False)
    cam_b = make_camera(session, name="b", ip="10.0.0.2")

    ids = _resolve_target_ids(
        CameraService(session), None, {"10.0.0.1", "10.0.0.2"}
    )

    assert set(ids) == {cam_b.id}  # 'a' is allowlisted but disabled


def test_explicit_camera_ids_bypass_allowlist(session: Session, make_camera) -> None:
    cam_a = make_camera(session, name="a", ip="10.0.0.9")

    ids = _resolve_target_ids(CameraService(session), [cam_a.id], {"1.1.1.1"})

    assert ids == [cam_a.id]  # explicit selection wins over the allowlist


def test_entry_exit_disabled_drops_the_gate(session: Session, make_camera) -> None:
    occ = make_camera(session, name="occ", ip="10.0.0.1", roles=["occupancy"])
    make_camera(session, name="gate", ip="10.0.0.2", roles=["entry_exit"])

    ids = _resolve_target_ids(
        CameraService(session), None, set(), entry_exit_enabled=False
    )

    assert set(ids) == {occ.id}  # the entry_exit camera gets no pipeline


def test_entry_exit_enabled_keeps_the_gate(session: Session, make_camera) -> None:
    occ = make_camera(session, name="occ", ip="10.0.0.1", roles=["occupancy"])
    gate = make_camera(session, name="gate", ip="10.0.0.2", roles=["entry_exit"])

    ids = _resolve_target_ids(
        CameraService(session), None, set(), entry_exit_enabled=True
    )

    assert set(ids) == {occ.id, gate.id}


def test_entry_exit_disable_ignored_for_explicit_camera(session: Session, make_camera) -> None:
    gate = make_camera(session, name="gate", ip="10.0.0.2", roles=["entry_exit"])

    ids = _resolve_target_ids(
        CameraService(session), [gate.id], set(), entry_exit_enabled=False
    )

    assert ids == [gate.id]  # explicit --worker --camera still runs the gate


def test_entry_exit_disable_composes_with_allowlist(session: Session, make_camera) -> None:
    occ = make_camera(session, name="occ", ip="10.0.0.1", roles=["occupancy"])
    make_camera(session, name="gate", ip="10.0.0.2", roles=["entry_exit"])
    make_camera(session, name="other", ip="10.0.0.9", roles=["occupancy"])

    ids = _resolve_target_ids(
        CameraService(session),
        None,
        {"10.0.0.1", "10.0.0.2"},
        entry_exit_enabled=False,
    )

    assert set(ids) == {occ.id}  # allowlisted AND gate-dropped
