"""Shared validation-capture helper (dev/ops).

Fetches a live frame per enabled occupancy camera (scoped to the deployment's
camera allowlist), runs the detector, annotates the zone(s) + person boxes, and
writes the result to the validation folder — wiping the previous run first.
Returns a per-camera + per-space + total count rollup.

Used by the tools capture endpoint and by the occupancy read endpoints'
optional ``refresh_images`` side effect. Heavy deps (``cv2``, the detector) are
imported lazily so the core / test environment stays light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.domain.models import ZoneType
from app.services.camera_service import CameraService

# Annotated validation frames are written here (wiped each capture).
VALIDATION_DIR = Path("validation_snapshots/b1-waiting")

_detector: Any = None


def get_detector() -> Any:
    """Lazily build and cache a detector from config (heavy import inside)."""
    global _detector
    if _detector is None:
        from app.inference.detector import YoloDetector

        _detector = YoloDetector(get_settings().detector)
    return _detector


def fetch_frame(service: CameraService, camera_id: int):
    """Fetch one live BGR frame for a camera via the HTTP snapshot path.

    Returns ``(spec, frame)``; ``frame`` is ``None`` if the fetch/decode failed.
    """
    from app.ingestion.snapshot import SnapshotClient, decode_jpeg
    from app.ingestion.stream_url import snapshot_url

    spec = service.build_camera_spec(camera_id)
    if spec is None:
        return None, None
    password = service.resolve_password(camera_id)
    snap = get_settings().processing.snapshot_pull
    client = SnapshotClient(
        snapshot_url(
            spec, scheme=snap.scheme, port=snap.http_port,
            path_template=snap.path_template,
        ),
        spec.username,
        password,
        auth=snap.auth,
        timeout_s=snap.timeout_s,
    )
    try:
        frame = decode_jpeg(client.fetch_bytes())
    finally:
        client.close()
    return spec, frame


def capture_validation_images(session: Session) -> dict:
    """Capture every allowlisted, enabled occupancy camera, annotate zone +
    detections, and save to the validation folder (wiping the previous run).

    green box = counted (inside a zone), grey = detected but outside, orange =
    zone outline. Returns the total, a per-space rollup, and per-camera counts.
    """
    import cv2
    import numpy as np
    from shapely.geometry import Point as ShPoint
    from shapely.geometry import Polygon

    service = CameraService(session)
    detector = get_detector()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    for old in [f for f in VALIDATION_DIR.iterdir() if f.is_file()]:
        old.unlink()

    # Scope to the deployment's cameras (the allowlist), so the folder mirrors
    # what the engine actually runs. Empty allowlist = every occupancy camera.
    allowlist = set(get_settings().processing.camera_allowlist_ips)
    cameras: list[dict] = []
    spaces_total: dict[str, int] = {}
    grand_total = 0
    for camera in service.list():
        if not camera.enabled or "occupancy" not in list(camera.roles):
            continue
        if allowlist and camera.ip not in allowlist:
            continue
        spec, frame = fetch_frame(service, camera.id)
        if spec is None or frame is None:
            cameras.append({"camera_id": camera.id, "name": camera.name, "error": "no frame"})
            continue

        detections = detector.detect(frame)
        zones = [z for z in spec.zones if z.type is ZoneType.OCCUPANCY]
        zone_polys = [
            (z.dt_space_id, Polygon([(p.x, p.y) for p in z.polygon])) for z in zones
        ]
        count = 0
        for det in detections:
            ground = ShPoint(det.bbox.bottom_center.x, det.bbox.bottom_center.y)
            matched = next(((sid, p) for sid, p in zone_polys if p.covers(ground)), None)
            count += 1 if matched is not None else 0
            if matched is not None and matched[0] is not None:
                spaces_total[matched[0]] = spaces_total.get(matched[0], 0) + 1
            x1, y1, x2, y2 = (int(v) for v in det.bbox.as_xyxy())
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 200, 0) if matched is not None else (150, 150, 150), 2,
            )
        grand_total += count
        for zone in zones:
            pts = np.asarray([[int(p.x), int(p.y)] for p in zone.polygon], np.int32)
            cv2.polylines(frame, [pts], True, (0, 180, 255), 2)
        cv2.putText(
            frame, f"{camera.name}  in-zone={count}  dets={len(detections)}",
            (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2,
        )
        safe = "".join(c if c.isalnum() else "_" for c in camera.name).strip("_")
        cv2.imwrite(str(VALIDATION_DIR / f"{camera.ip}_{safe}.jpg"), frame)
        cameras.append(
            {"camera_id": camera.id, "name": camera.name, "ip": camera.ip,
             "in_zone": count, "detected": len(detections)}
        )

    return {
        "saved_to": str(VALIDATION_DIR.resolve()),
        "cleared_old": True,
        "total_in_zone": grand_total,
        "spaces": [{"dt_space_id": sid, "count": c} for sid, c in sorted(spaces_total.items())],
        "cameras": cameras,
    }
