#!/usr/bin/env python
"""Validate RTSP connectivity for a registered camera (HLD 12.1, 8.1 /test).

    python scripts/test_camera.py --camera-id 3
    python scripts/test_camera.py --url rtsp://user:pass@10.0.0.5:554/Streaming/Channels/102
"""

from __future__ import annotations

import argparse
import json
import sys


def _probe_url(url: str, timeout_s: float = 8.0) -> dict:
    """Open a stream directly and report basic properties (lazy OpenCV)."""
    import time

    import cv2  # lazy: only needed when actually probing

    start = time.time()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return {"reachable": False, "error": "could not open stream"}
        ok, frame = cap.read()
        latency_ms = (time.time() - start) * 1000.0
        if not ok or frame is None:
            return {"reachable": False, "error": "opened but no frame", "latency_ms": latency_ms}
        h, w = frame.shape[:2]
        return {
            "reachable": True,
            "resolution": f"{w}x{h}",
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) or None,
            "latency_ms": round(latency_ms, 1),
        }
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate RTSP connectivity for a camera")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera-id", type=int, help="registered camera id (uses the DB)")
    group.add_argument("--url", type=str, help="raw RTSP URL to probe directly")
    args = parser.parse_args(argv)

    if args.url:
        result = _probe_url(args.url)
    else:
        from app.db.session import session_scope
        from app.services.camera_service import CameraService

        with session_scope() as session:
            result = CameraService(session).test_camera(args.camera_id).model_dump()

    print(json.dumps(result, indent=2))
    return 0 if result.get("reachable") else 1


if __name__ == "__main__":
    sys.exit(main())
