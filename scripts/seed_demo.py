#!/usr/bin/env python
"""Seed one demo camera with zones and a counting line for quick start / Digital Twin.

Registers a single ``Demo Lobby Cam`` together with an occupancy zone, a waiting
zone, a restricted zone (with a ``safe_limit``) and an entry/exit counting line,
all drawn in image-space coordinates and tagged with ``dt_space_id`` values so a
Digital Twin can bind the analytics to its spatial model. The script is
idempotent: if the demo camera already exists it changes nothing and reports the
existing ids. The created (or pre-existing) ids are printed as JSON.

The camera password is encrypted at rest via :class:`CameraService`. If
``CAMERA_CREDENTIALS_KEY`` is not set, an ephemeral AES-256 key is generated for
this run so the quick start works out of the box; persist a stable key in the
environment before seeding a database you intend to keep.

    python scripts/seed_demo.py
    python scripts/seed_demo.py --database-url sqlite:///./demo.db
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from app.api.schemas.camera import CameraCreate
from app.config.settings import reset_settings_cache
from app.db.repositories.line_repo import LineRepository
from app.db.repositories.zone_repo import ZoneRepository
from app.db.session import configure_engine, init_db, session_scope
from app.domain.models import CameraRole, ZoneType
from app.services.camera_service import CameraService
from app.services.credentials import CredentialCipher
from app.utils.logging import get_logger

logger = get_logger(__name__)

DEMO_CAMERA_NAME = "Demo Lobby Cam"

# Demo camera connection details. The password is encrypted before it reaches the
# database (CameraService owns that); this is a throwaway value for the demo.
_DEMO_CAMERA = CameraCreate(
    name=DEMO_CAMERA_NAME,
    area="lobby",
    ip="192.0.2.10",  # TEST-NET-1 (RFC 5737): never a real host
    port=554,
    username="demo",
    password="demo-password",  # noqa: S106 - demo-only credential, stored encrypted
    roles=[CameraRole.OCCUPANCY, CameraRole.ENTRY_EXIT],
    stream_channel_sub=102,
    stream_channel_main=101,
)

# Zones drawn in image space ([x, y] pixels on a 1280x720 frame). Each carries a
# dt_space_id so a Digital Twin can map the analytic region to its own geometry.
_DEMO_ZONES: tuple[dict[str, object], ...] = (
    {
        "name": "Lobby Occupancy",
        "type": ZoneType.OCCUPANCY.value,
        "polygon": [[200, 200], [1080, 200], [1080, 620], [200, 620]],
        "safe_limit": None,
        "dt_space_id": "dt:lobby:occupancy",
    },
    {
        "name": "Reception Queue",
        "type": ZoneType.WAITING.value,
        "polygon": [[300, 460], [620, 460], [620, 680], [300, 680]],
        "safe_limit": None,
        "dt_space_id": "dt:lobby:waiting",
    },
    {
        "name": "Server Closet",
        "type": ZoneType.RESTRICTED.value,
        "polygon": [[980, 80], [1240, 80], [1240, 360], [980, 360]],
        "safe_limit": 0,  # nobody is allowed inside the restricted area
        "dt_space_id": "dt:lobby:restricted",
    },
)

# Entry/exit counting line across the main doorway. ``in_direction`` is the
# reference normal pointing into the venue; ``area_id`` groups crossings.
_DEMO_LINE: dict[str, object] = {
    "name": "Main Entrance",
    "points": [[640, 700], [640, 480]],
    "in_direction": [-1.0, 0.0],
    "area_id": "lobby",
    "dt_space_id": "dt:lobby:entrance",
}


def _ensure_credentials_key() -> None:
    """Provide an ephemeral encryption key if ``CAMERA_CREDENTIALS_KEY`` is unset.

    Camera passwords are encrypted at rest, so :class:`CameraService` needs a key.
    For a frictionless quick start we generate one for this process if none is
    configured and clear the cached secrets so it is picked up. A warning is
    logged because the key is not persisted — credentials sealed with it cannot be
    decrypted on a later run.
    """
    if os.environ.get("CAMERA_CREDENTIALS_KEY"):
        return
    os.environ["CAMERA_CREDENTIALS_KEY"] = CredentialCipher.generate_key()
    reset_settings_cache()
    logger.warning(
        "CAMERA_CREDENTIALS_KEY was not set; generated an ephemeral key for this "
        "seed run. Set a stable key in the environment to keep credentials usable.",
        extra={"event": "seed_ephemeral_key"},
    )


def _seed_geometry(
    zones: ZoneRepository, lines: LineRepository, camera_id: int
) -> dict[str, object]:
    """Create the demo zones and line for ``camera_id`` and return their ids."""
    zone_ids = [zones.create({**zone, "camera_id": camera_id}).id for zone in _DEMO_ZONES]
    line = lines.create({**_DEMO_LINE, "camera_id": camera_id})
    return {"zone_ids": zone_ids, "line_id": line.id}


def seed(database_url: str | None) -> dict[str, object]:
    """Seed the demo camera, zones and line, idempotently.

    Args:
        database_url: Optional SQLAlchemy URL to bind before seeding. When ``None``
            the engine configured from settings / ``DATABASE_URL`` is used.

    Returns:
        A summary mapping with ``created`` (whether new rows were written),
        ``camera_id``, ``zone_ids`` and ``line_id``.
    """
    if database_url:
        configure_engine(database_url)
    _ensure_credentials_key()
    init_db()

    with session_scope() as session:
        service = CameraService(session)
        zones = ZoneRepository(session)
        lines = LineRepository(session)
        existing = next(
            (cam for cam in service.list() if cam.name == DEMO_CAMERA_NAME), None
        )
        if existing is not None:
            camera_lines = lines.list_by_camera(existing.id)
            camera_zones = zones.list_by_camera(existing.id)
            logger.info(
                "demo camera already present; nothing to seed",
                extra={"camera_id": existing.id, "event": "seed_skipped"},
            )
            return {
                "created": False,
                "camera_id": existing.id,
                "zone_ids": [zone.id for zone in camera_zones],
                "line_id": camera_lines[0].id if camera_lines else None,
            }

        camera = service.create_camera(_DEMO_CAMERA)
        geometry = _seed_geometry(zones, lines, camera.id)
        logger.info(
            "seeded demo camera",
            extra={"camera_id": camera.id, "event": "seed_created"},
        )
        return {"created": True, "camera_id": camera.id, **geometry}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, seed, and print the result as JSON."""
    parser = argparse.ArgumentParser(
        description="Seed a demo camera, zones and counting line for quick start."
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="SQLAlchemy URL to seed (default: env DATABASE_URL / configured engine).",
    )
    args = parser.parse_args(argv)

    summary = seed(args.database_url)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
