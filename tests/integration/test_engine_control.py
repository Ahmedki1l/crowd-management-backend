"""Engine-control endpoints and role resolution (/api/v1/engine/*).

Covers the runtime "buttons" that switch the engine scope — everything,
entry/exit only, occupancy only, reset, stop — plus the role→camera resolver
that backs them. Cameras created here have no stored credential, so the built
engine runs zero pipelines (no perception backend is loaded); the assertions are
on the reported scope and on which camera ids each role resolves to.
"""

from __future__ import annotations

import dotenv
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.domain.models import CameraRole
from app.engine.manager import camera_ids_for_role


def test_camera_ids_for_role_filters_by_role(session: Session, make_camera) -> None:
    occ = make_camera(session, name="occ", ip="10.0.0.11", roles=["occupancy"])
    ee = make_camera(session, name="ee", ip="10.0.0.12", roles=["entry_exit"])
    both = make_camera(session, name="both", ip="10.0.0.13", roles=["occupancy", "entry_exit"])
    disabled = make_camera(session, name="off", ip="10.0.0.14", roles=["occupancy"], enabled=False)

    assert camera_ids_for_role(None) is None  # "all" defers to build_engine

    occ_ids = set(camera_ids_for_role(CameraRole.OCCUPANCY))
    assert occ.id in occ_ids and both.id in occ_ids
    assert ee.id not in occ_ids
    assert disabled.id not in occ_ids  # enabled filter applies

    ee_ids = set(camera_ids_for_role(CameraRole.ENTRY_EXIT))
    assert ee.id in ee_ids and both.id in ee_ids
    assert occ.id not in ee_ids


def test_engine_status_requires_auth(client: TestClient) -> None:
    assert client.get("/api/v1/engine/status").status_code in (401, 403)
    assert client.post("/api/v1/engine/occupancy").status_code in (401, 403)


@pytest.mark.parametrize(
    ("path", "expected_mode"),
    [
        ("/api/v1/engine/occupancy", "occupancy"),
        ("/api/v1/engine/entry-exit", "entry_exit"),
        ("/api/v1/engine/all", "all"),
        ("/api/v1/engine/stop", "stopped"),
    ],
)
def test_engine_switch_modes(
    client: TestClient, auth_headers: dict[str, str], session: Session, make_camera, path, expected_mode
) -> None:
    make_camera(session, roles=["occupancy"])
    status = client.get("/api/v1/engine/status", headers=auth_headers)
    assert status.status_code == 200
    assert "mode" in status.json() and "pipeline_count" in status.json()

    response = client.post(path, headers=auth_headers)
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == expected_mode


def test_engine_reset_returns_to_all(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep the test hermetic: don't let reset re-read the real .env over the
    # monkeypatched test environment (DATABASE_URL / CONFIG_PATH / secrets).
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    response = client.post("/api/v1/engine/reset", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["mode"] == "all"
