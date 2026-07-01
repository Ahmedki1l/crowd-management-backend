#!/usr/bin/env python
"""Live debug visualiser — detection, tracking, zones, occupancy, and lines.

Opens an RTSP stream or video file, runs the full YOLO → ByteTrack →
zone-evaluator → state-machine → occupancy + line-crossing pipeline and shows
an annotated cv2 window in real time.

Usage
-----
  # Zones only
  python scripts/visualize.py \
      --url rtsp://admin:PASS@10.1.13.5:554/Streaming/Channels/102 \
      --zones cam01.json

  # Zones + counting lines
  python scripts/visualize.py \
      --url rtsp://admin:PASS@10.1.13.5:554/Streaming/Channels/102 \
      --zones zones/CAM-01_MAIN_DOOR.json \
      --lines lines/Cam_01_MAIN_DOOR.json

  # Lines only (entry/exit without occupancy zones)
  python scripts/visualize.py \
      --url rtsp://admin:PASS@10.1.13.5:554/Streaming/Channels/102 \
      --lines lines/Cam_01_MAIN_DOOR.json

  # Zones fetched from a running backend
  python scripts/visualize.py \
      --url rtsp://admin:PASS@10.1.13.5:554/Streaming/Channels/102 \
      --api http://localhost:8000/api/v1 --token <token> --camera-id 3

  # Local video file
  python scripts/visualize.py --url /path/to/clip.mp4 --zones cam01.json

Keys
----
  Q  quit
  P  pause / resume

Requires: pip install -e ".[inference]"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import deque
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Colours  (BGR)
# ---------------------------------------------------------------------------
_ZONE_COLORS: dict[str, tuple[int, int, int]] = {
    "occupancy":  (0,   210,  60),
    "waiting":    (200, 130,   0),
    "restricted": (0,    40, 220),
}
_BOX_DEFAULT   = (200, 200, 200)
_BOX_IN_ZONE   = (0,   255,  80)
_BOX_ENTERING  = (0,   200, 255)
_LINE_COLOR    = (0,   165, 255)   # orange
_IN_ARROW      = (0,   255,   0)   # green arrow = IN direction
_FONT           = cv2.FONT_HERSHEY_SIMPLEX
_FILL_ALPHA     = 0.18


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _fetch_zones_from_api(base: str, token: str, camera_id: int) -> list[dict]:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/zones?camera_id={camera_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    return data if isinstance(data, list) else data.get("items", [])


def _load_zones_from_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict) and "zones" in raw:
        return raw["zones"]
    if isinstance(raw, list):
        return raw
    return []


def _load_lines_from_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        return raw.get("lines", [])
    return []


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------

def _build_zone_specs(zones_raw: list[dict]) -> list[Any]:
    from app.domain.models import Point, ZoneSpec, ZoneType

    specs: list[ZoneSpec] = []
    for idx, z in enumerate(zones_raw):
        zone_id = z.get("id", idx + 1)
        name = z.get("name", f"zone-{zone_id}")
        raw_type = z.get("type", "occupancy")
        try:
            zone_type = ZoneType(raw_type)
        except ValueError:
            zone_type = ZoneType.OCCUPANCY
        raw_pts = z.get("points") or z.get("polygon") or []
        polygon = tuple(Point(float(p[0]), float(p[1])) for p in raw_pts)
        if len(polygon) < 3:
            print(f"  [warn] zone {name!r} has < 3 points — skipped", file=sys.stderr)
            continue
        specs.append(
            ZoneSpec(
                id=zone_id,
                camera_id=z.get("camera_id", 0),
                name=name,
                type=zone_type,
                polygon=polygon,
                dt_space_id=z.get("dt_space_id"),
                safe_limit=z.get("safe_limit"),
            )
        )
    return specs


def _build_line_specs(lines_raw: list[dict]) -> list[Any]:
    from app.domain.models import LineSpec, Point

    specs: list[LineSpec] = []
    for idx, ln in enumerate(lines_raw):
        line_id = ln.get("id", idx + 1)
        raw_pts = ln.get("points", [])
        if len(raw_pts) < 2:
            print(f"  [warn] line {ln.get('name', idx)!r} has < 2 points — skipped", file=sys.stderr)
            continue
        points = (Point(float(raw_pts[0][0]), float(raw_pts[0][1])),
                  Point(float(raw_pts[1][0]), float(raw_pts[1][1])))
        raw_dir = ln.get("in_direction", [1, 0])
        in_direction = Point(float(raw_dir[0]), float(raw_dir[1]))
        specs.append(
            LineSpec(
                id=line_id,
                camera_id=ln.get("camera_id", 0),
                name=ln.get("name", f"line-{line_id}"),
                points=points,
                in_direction=in_direction,
                area_id=ln.get("area_id", "area1"),
                dt_space_id=ln.get("dt_space_id"),
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _pts_array(zone_spec: Any) -> np.ndarray:
    pts = np.array([[int(p.x), int(p.y)] for p in zone_spec.polygon], dtype=np.int32)
    return pts.reshape((-1, 1, 2))


def _draw_zone(frame: np.ndarray, spec: Any, count: int) -> None:
    color = _ZONE_COLORS.get(spec.type.value, (0, 200, 200))
    pts = _pts_array(spec)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, _FILL_ALPHA, frame, 1.0 - _FILL_ALPHA, 0, frame)

    thickness = 3 if count > 0 else 2
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    ys = [int(p.y) for p in spec.polygon]
    xs = [int(p.x) for p in spec.polygon]
    lx, ly = min(xs), min(ys) - 8
    label = f"{spec.name}: {count}"
    if spec.safe_limit:
        label += f"/{spec.safe_limit}"

    (tw, th), baseline = cv2.getTextSize(label, _FONT, 0.52, 1)
    cv2.rectangle(frame, (lx - 2, ly - th - 3), (lx + tw + 2, ly + baseline), (0, 0, 0), -1)
    cv2.putText(frame, label, (lx, ly), _FONT, 0.52, color, 1, cv2.LINE_AA)


def _draw_line(frame: np.ndarray, spec: Any, in_count: int, out_count: int) -> None:
    p1 = (int(spec.points[0].x), int(spec.points[0].y))
    p2 = (int(spec.points[1].x), int(spec.points[1].y))

    # Main line
    cv2.line(frame, p1, p2, _LINE_COLOR, 2)

    # Endpoints
    cv2.circle(frame, p1, 5, _LINE_COLOR, -1)
    cv2.circle(frame, p2, 5, _LINE_COLOR, -1)

    # IN direction arrow from midpoint
    mx = (p1[0] + p2[0]) // 2
    my = (p1[1] + p2[1]) // 2
    arrow_len = 35
    dx = int(spec.in_direction.x * arrow_len)
    dy = int(spec.in_direction.y * arrow_len)
    cv2.arrowedLine(frame, (mx, my), (mx + dx, my + dy), _IN_ARROW, 2, tipLength=0.4)

    # Label  IN / OUT counts
    label = f"{spec.name}  IN:{in_count}  OUT:{out_count}"
    lx = min(p1[0], p2[0])
    ly = min(p1[1], p2[1]) - 8
    (tw, th), baseline = cv2.getTextSize(label, _FONT, 0.5, 1)
    cv2.rectangle(frame, (lx - 2, ly - th - 3), (lx + tw + 2, ly + baseline), (0, 0, 0), -1)
    cv2.putText(frame, label, (lx, ly), _FONT, 0.5, _LINE_COLOR, 1, cv2.LINE_AA)


def _draw_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    track_id: int,
    confidence: float,
    in_zone: bool,
    entering: bool,
) -> None:
    if in_zone:
        color = _BOX_IN_ZONE
    elif entering:
        color = _BOX_ENTERING
    else:
        color = _BOX_DEFAULT

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"#{track_id}  {confidence:.2f}"
    (tw, th), baseline = cv2.getTextSize(label, _FONT, 0.45, 1)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 2), (x1 + tw + 2, ty + baseline), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 1, ty), _FONT, 0.45, color, 1, cv2.LINE_AA)


def _draw_hud(
    frame: np.ndarray,
    fps: float,
    total_occ: int,
    n_tracks: int,
    total_in: int,
    total_out: int,
    paused: bool,
) -> None:
    lines = [
        f"FPS: {fps:.1f}",
        f"Tracks: {n_tracks}",
        f"Total occ: {total_occ}",
        f"IN: {total_in}  OUT: {total_out}",
    ]
    if paused:
        lines.append("** PAUSED **")

    for i, text in enumerate(lines):
        y = 20 + i * 20
        cv2.putText(frame, text, (8, y), _FONT, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (8, y), _FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Pipeline bootstrap
# ---------------------------------------------------------------------------

def _build_pipeline(args: argparse.Namespace, zone_specs: list, line_specs: list) -> tuple:
    from app.analytics.entry_exit import EntryExitCalculator
    from app.analytics.occupancy import OccupancyCalculator
    from app.config.schema import DetectorConfig, TrackerConfig
    from app.inference.detector import YoloDetector
    from app.inference.tracker import ByteTrackTracker
    from app.localisation.lines import LineCrossingDetector
    from app.localisation.state_machine import ZonePresenceTracker
    from app.localisation.zones import ZoneEvaluator
    from app.utils.clock import SystemClock

    det_cfg = DetectorConfig(
        model_path=args.model_path,
        model_weights=args.model_weights,
        runtime="openvino",
        confidence=args.conf,
        iou=0.45,
        imgsz=640,
        batch_size=1,
    )
    trk_cfg = TrackerConfig(
        reid_enabled=False,
        frame_rate=8,
    )

    print("Loading detector (this may export the model on first run) ...")
    clock = SystemClock()
    detector      = YoloDetector(det_cfg)
    tracker       = ByteTrackTracker(trk_cfg)
    evaluator     = ZoneEvaluator(zone_specs)
    presence      = ZonePresenceTracker(zone_specs, confirm_enter_frames=5, confirm_leave_frames=8)
    occ_calc      = OccupancyCalculator(camera_id=args.camera_id or 0, zones=zone_specs, clock=clock)
    line_detector = LineCrossingDetector(line_specs)
    ee_calc       = EntryExitCalculator(line_specs, clock)
    return detector, tracker, evaluator, presence, occ_calc, line_detector, ee_calc


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live detection + occupancy + entry/exit visualiser")
    parser.add_argument("--url",           required=True,          help="RTSP URL or video file path")
    parser.add_argument("--zones",         default=None,           help="Zones JSON from draw_zones.py")
    parser.add_argument("--lines",         default=None,           help="Lines JSON from draw_zones.py")
    parser.add_argument("--api",           default=None,           help="Backend API base URL (to fetch zones)")
    parser.add_argument("--token",         default=None,           help="Bearer token (for --api)")
    parser.add_argument("--camera-id",     type=int, default=None, help="Backend camera ID")
    parser.add_argument("--model-path",    default="models/detector")
    parser.add_argument("--model-weights", default="yolo11m.pt")
    parser.add_argument("--conf",          type=float, default=0.30, help="Detection confidence threshold")
    parser.add_argument("--width",         type=int,   default=0,    help="Resize output window width (0=native)")
    args = parser.parse_args(argv)

    # ---- Load zones ----------------------------------------------------------
    zones_raw: list[dict] = []
    if args.api and args.token and args.camera_id:
        print(f"Fetching zones from {args.api} for camera {args.camera_id} ...")
        try:
            zones_raw = _fetch_zones_from_api(args.api, args.token, args.camera_id)
        except Exception as exc:
            print(f"  [warn] API fetch failed: {exc} — no zones will be shown", file=sys.stderr)
    elif args.zones:
        zones_raw = _load_zones_from_file(args.zones)
    else:
        print("[warn] No --zones or --api given; running without zone overlay.", file=sys.stderr)

    # ---- Load lines ----------------------------------------------------------
    lines_raw: list[dict] = []
    if args.lines:
        lines_raw = _load_lines_from_file(args.lines)
        # Also check if the zones file contains lines
        if args.zones and not lines_raw:
            with open(args.zones, encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                lines_raw = raw.get("lines", [])
    elif args.zones:
        # If no --lines given, try to load lines from the zones file too
        with open(args.zones, encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            lines_raw = raw.get("lines", [])

    zone_specs = _build_zone_specs(zones_raw)
    line_specs = _build_line_specs(lines_raw)
    print(f"Loaded {len(zone_specs)} zone(s): {[z.name for z in zone_specs]}")
    print(f"Loaded {len(line_specs)} line(s): {[l.name for l in line_specs]}")

    # ---- Build pipeline ------------------------------------------------------
    detector, tracker, evaluator, presence, occ_calc, line_detector, ee_calc = \
        _build_pipeline(args, zone_specs, line_specs)

    # ---- Open stream ---------------------------------------------------------
    print(f"Opening stream: {args.url}")
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"ERROR: could not open {args.url!r}", file=sys.stderr)
        return 1

    zone_counts: dict[int, int] = {z.id: 0 for z in zone_specs}
    # area_id -> (in, out)
    area_counts: dict[str, tuple[int, int]] = {}

    fps_buf: deque[float] = deque(maxlen=30)
    t_prev = time.monotonic()
    paused = False

    print("Window open. Press Q to quit, P to pause.")
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Stream ended or read error.")
                break

            ts = time.time()

            # --- detection & tracking ---
            detections = detector.detect(frame)
            tracked    = tracker.update(detections, frame)

            # --- zone logic ---
            membership      = evaluator.membership(tracked)
            presence_result = presence.update(membership, ts)

            for event in occ_calc.process(presence_result.confirmed, ts):
                zone_counts[event.zone_id] = event.count

            # --- line crossing ---
            crossings = line_detector.update(tracked, ts)
            for event in ee_calc.process(crossings, ts):
                if hasattr(event, "in_count"):   # CountUpdate
                    area_counts[event.area_id] = (event.in_count, event.out_count)

            # --- confirmed / entering sets ---
            confirmed_ids: set[int] = set()
            for tids in presence_result.confirmed.values():
                confirmed_ids.update(tids)
            entering_ids: set[int] = set()
            for tids in membership.values():
                entering_ids.update(tids - confirmed_ids)

            # --- FPS ---
            t_now = time.monotonic()
            fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now
            fps = sum(fps_buf) / len(fps_buf)

            # ----------------------------------------------------------------
            # Draw
            # ----------------------------------------------------------------
            if args.width > 0:
                h, w = frame.shape[:2]
                new_h = int(h * args.width / w)
                frame = cv2.resize(frame, (args.width, new_h))

            # Zones
            for spec in zone_specs:
                _draw_zone(frame, spec, zone_counts.get(spec.id, 0))

            # Lines
            for spec in line_specs:
                in_c, out_c = area_counts.get(spec.area_id, (0, 0))
                _draw_line(frame, spec, in_c, out_c)

            # Bounding boxes
            for td in tracked:
                x1, y1, x2, y2 = (
                    int(td.bbox.x1), int(td.bbox.y1),
                    int(td.bbox.x2), int(td.bbox.y2),
                )
                _draw_box(
                    frame, x1, y1, x2, y2,
                    td.track_id, td.confidence,
                    in_zone=td.track_id in confirmed_ids,
                    entering=td.track_id in entering_ids,
                )

            # HUD
            total_in  = sum(v[0] for v in area_counts.values())
            total_out = sum(v[1] for v in area_counts.values())
            _draw_hud(
                frame,
                fps=fps,
                total_occ=sum(zone_counts.values()),
                n_tracks=len(tracked),
                total_in=total_in,
                total_out=total_out,
                paused=False,
            )

        else:
            total_in  = sum(v[0] for v in area_counts.values())
            total_out = sum(v[1] for v in area_counts.values())
            _draw_hud(frame, fps=0.0, total_occ=sum(zone_counts.values()),
                      n_tracks=0, total_in=total_in, total_out=total_out, paused=True)

        cv2.imshow("Camera Analytics — Visualiser", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
