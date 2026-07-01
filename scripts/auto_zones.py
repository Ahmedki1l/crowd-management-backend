#!/usr/bin/env python
"""Auto-generate full-scene occupancy zones for every camera.

For each camera: connects to its RTSP sub-stream, grabs one frame, and writes
a zones JSON covering the entire visible area (4-corner polygon).

Filename  : <camera_name_slug>.json   e.g. CAM-35_B1-DATA_CENTER.json
Zone name : site zone name            e.g. "Data Center"

Output files are ready to pass directly to register_zones.py.

Usage
-----
  python scripts/auto_zones.py                     # all cameras -> zones/
  python scripts/auto_zones.py --out-dir my_zones  # custom output dir
  python scripts/auto_zones.py --show              # preview each frame before saving
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Camera list — ONLY the originally requested cameras (29 total, skip 10.1.13.32)
#   name : camera label  →  used as the output filename
#   ip   : camera IP
#   zone : site zone name  →  written inside the JSON as the zone name
# ---------------------------------------------------------------------------
CAMERAS = [
    # ── Basement 1 ──────────────────────────────────────────────────────────
    {"name": "CAM-35 (B1-DATA CENTER)",   "ip": "10.1.13.54", "zone": "Data Center"},
    {"name": "CAM-16 (B1-PASSWAY)",       "ip": "10.1.13.35", "zone": "Left Hallway"},
    {"name": "CAM-32 (B1-STAIRCASE)",     "ip": "10.1.13.51", "zone": "Left Hallway"},
    {"name": "CAM-20 (B1-OFFICE)",        "ip": "10.1.13.39", "zone": "Lounge Area"},
    {"name": "CAM-25 (B1-OFFICE)",        "ip": "10.1.13.44", "zone": "Open Space"},
    {"name": "CAM-19 (B1-LIFT PASSWAY)",  "ip": "10.1.13.38", "zone": "Right Hallway"},
    # ── Ground Floor ────────────────────────────────────────────────────────
    {"name": "CAM-09 GF OFFICE LEFT",     "ip": "10.1.13.28", "zone": "Claims & Recovery Workstation"},
    {"name": "CAM-01 MAIN DOOR",          "ip": "10.1.13.21", "zone": "Entrance"},
    {"name": "CAM-21 GF BACK STAIR",      "ip": "10.1.13.80", "zone": "GF Backside"},
    {"name": "CAM-05 (GF-OFFICE L)",      "ip": "10.1.13.24", "zone": "HR"},
    {"name": "CAM-10 GF IT ROOM",         "ip": "10.1.13.29", "zone": "IT Room"},
    {"name": "CAM-06 (GF-LIFT)",          "ip": "10.1.13.25", "zone": "Lift Area"},
    {"name": "CAM-07 (GF-OFFICE R)",      "ip": "10.1.13.26", "zone": "Office Right Area"},
    {"name": "CAM-40 (GF-OFFICE-R2)",     "ip": "10.1.13.59", "zone": "Office Right Area"},
    {"name": "CAM-04 (GF-OFFICE)",        "ip": "10.1.13.23", "zone": "Open Area"},
    {"name": "CAM-08 GF OFFICE RIGHT",    "ip": "10.1.13.27", "zone": "Open Offices Area"},
    {"name": "CAM-02 (GF-WAITING)",       "ip": "10.1.13.20", "zone": "Reception"},
    {"name": "CAM-03 (GF-RECEPTION)",     "ip": "10.1.13.22", "zone": "Reception"},
    {"name": "CAM-11 (GF-STAIRCASE)",     "ip": "10.1.13.30", "zone": "Staircase"},
    # ── First Floor ─────────────────────────────────────────────────────────
    {"name": "CAM-23 (F1-BACK OUTSIDE)",  "ip": "10.1.13.82", "zone": "FF Back Roof"},
    {"name": "Camera-22 F1-BACK STAIR",   "ip": "10.1.13.81", "zone": "FF Backside"},
    {"name": "CAM-38 (FF-IT-ROOM)",       "ip": "10.1.13.55", "zone": "IT Room"},
    {"name": "CAM-12 1F-RIGHT WAY",       "ip": "10.1.13.31", "zone": "Left Hallway"},
    {"name": "CAM-17 1F-LEFT PASSWAY",    "ip": "10.1.13.36", "zone": "Right-1 Hallway"},
    {"name": "CAM-36 1F-BACK PASSWAY",    "ip": "10.1.13.56", "zone": "Right-2 Hallway"},
    {"name": "CAM-14 1F-STAIRCASE",       "ip": "10.1.13.33", "zone": "Stairs"},
    {"name": "CAM-15 1F-WAITING",         "ip": "10.1.13.34", "zone": "Waiting area"},
    # ── Roof ────────────────────────────────────────────────────────────────
    {"name": "CAM-37 (ROOF-LIFT)",        "ip": "10.1.13.57", "zone": "Lift Area"},
    {"name": "CAM-39 (ROOF-STAIRCASE)",   "ip": "10.1.13.58", "zone": "Staircase"},
]

# Credentials
_USER = "kloudspot"
_PASS = "Kloud%40321"   # @ is URL-encoded
_PORT = 554
_CHANNEL = "102"        # sub-stream

# Cameras to skip (by IP)
SKIP_IPS = {"10.1.13.32"}   # Camera 13 - 1F Left way (excluded by request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rtsp_url(ip: str) -> str:
    return f"rtsp://{_USER}:{_PASS}@{ip}:{_PORT}/Streaming/Channels/{_CHANNEL}"


def _slug(name: str) -> str:
    """Camera name -> filename slug.  Spaces and () become _, doubles collapsed."""
    s = name.replace("(", "").replace(")", "").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _grab_frame(url: str, retries: int = 3):
    import cv2

    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        finally:
            cap.release()
        print(f"    attempt {attempt}/{retries} failed")
    return None


def _full_scene_polygon(frame) -> list[list[int]]:
    h, w = frame.shape[:2]
    return [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]


def _show_frame(frame, camera_name: str, zone_name: str) -> bool:
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    canvas = frame.copy()
    pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], (0, 200, 0))
    cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)
    cv2.polylines(canvas, [pts], True, (0, 220, 0), 2)
    cv2.putText(canvas, f"{camera_name}  |  zone: {zone_name}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    cv2.putText(canvas, "ENTER = save   Q = skip this camera",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    win = "auto_zones preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1280), min(h, 720))
    cv2.imshow(win, canvas)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(win)
    return key not in (ord("q"), ord("Q"))


def _write_zone_json(out_path: Path, zone_name: str, polygon: list[list[int]]) -> None:
    payload = {
        "zones": [
            {
                "name": zone_name,
                "type": "occupancy",
                "safe_limit": None,
                "dt_space_id": None,
                "polygon": polygon,
            }
        ],
        "lines": [],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-generate full-scene occupancy zones")
    parser.add_argument("--out-dir", default="zones", help="Output directory (default: zones/)")
    parser.add_argument("--show", action="store_true",
                        help="Preview each frame before saving (ENTER=save, Q=skip)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = skipped = failed = saved = 0

    for cam in CAMERAS:
        total += 1
        ip   = cam["ip"]
        name = cam["name"]
        zone = cam["zone"]

        if ip in SKIP_IPS:
            print(f"[SKIP]  {name}  ({ip})")
            skipped += 1
            continue

        url = _rtsp_url(ip)
        print(f"[....] {name}  ({ip})", end="", flush=True)

        frame = _grab_frame(url)
        if frame is None:
            print(f"\r[FAIL] {name}  ({ip})  — could not grab frame")
            failed += 1
            continue

        h, w = frame.shape[:2]
        polygon = _full_scene_polygon(frame)

        if args.show:
            print()
            if not _show_frame(frame, name, zone):
                print(f"[SKIP]  {name}  — skipped by user")
                skipped += 1
                continue

        # Filename from camera name, zone name goes inside the JSON
        out_path = out_dir / f"{_slug(name)}.json"
        _write_zone_json(out_path, zone, polygon)
        saved += 1
        print(f"\r[ OK ] {name}  ({ip})  {w}x{h}  -> {out_path.name}")

    print()
    print(f"Done.  {saved} saved  |  {failed} failed  |  {skipped} skipped  (of {total} cameras)")
    if failed:
        print(f"\n{failed} camera(s) could not be reached — check RTSP/network.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
