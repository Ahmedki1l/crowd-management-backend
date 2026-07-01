#!/usr/bin/env python
"""Rescale zone polygons and line points in the DB after a resolution change.

When a camera's sub-stream changes resolution (e.g. 640x360 -> 1280x720), every
pixel coordinate you drew is now in the wrong place. This multiplies all zone
polygons and line points by the width/height ratio so they line up again.

A line's ``in_direction`` is a DIRECTION (unit normal), not a position, so it is
left unchanged -- correct for a uniform resize like 360 -> 720.

!!! IMPORTANT
  * This OVERWRITES coordinates in the database.
  * BACK UP first. For SQLite just copy the DB file:  copy camera_analytics.db camera_analytics.db.bak
  * Always run --dry-run first to preview.
  * Run it ONCE. A second run would scale a second time.

Usage (from the repo root, in the venv):
    python scripts/scale_geometry.py --from 640x360 --to 1280x720 --dry-run
    python scripts/scale_geometry.py --from 640x360 --to 1280x720
    python scripts/scale_geometry.py --from 640x360 --to 1280x720 --camera-ids 27,28,42
"""

from __future__ import annotations

import argparse


def _parse_res(text: str) -> tuple[float, float]:
    """Parse a 'WxH' string into (width, height) floats."""
    width, height = text.lower().replace(" ", "").split("x")
    return float(width), float(height)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescale zone/line coordinates after a resolution change")
    parser.add_argument("--from", dest="src", required=True, help="OLD resolution WxH (e.g. 640x360)")
    parser.add_argument("--to", dest="dst", required=True, help="NEW resolution WxH (e.g. 1280x720)")
    parser.add_argument("--camera-ids", default="", help="comma list to limit to (default: all cameras)")
    parser.add_argument("--dry-run", action="store_true", help="preview only, write nothing")
    args = parser.parse_args()

    # Imported here so --help works anywhere; run from the repo root (venv).
    import app.db.models  # noqa: F401  (registers tables on the metadata)
    from app.db.models.geometry import Line, Zone
    from app.db.session import session_scope

    from_w, from_h = _parse_res(args.src)
    to_w, to_h = _parse_res(args.dst)
    scale_x, scale_y = to_w / from_w, to_h / from_h
    only = {int(x) for x in args.camera_ids.split(",") if x.strip()} or None
    print(f"scale  x={scale_x:.4f}  y={scale_y:.4f}   cameras={'all' if only is None else sorted(only)}")

    def _scaled(points: list) -> list:
        return [[round(p[0] * scale_x), round(p[1] * scale_y)] for p in points]

    with session_scope() as session:
        zones = session.query(Zone).all()
        lines = session.query(Line).all()

        zones_done = lines_done = 0
        sample_shown = False
        for zone in zones:
            if only is not None and zone.camera_id not in only:
                continue
            new_poly = _scaled(zone.polygon)
            if not sample_shown:
                print(f"  e.g. zone '{zone.name}': {zone.polygon[:2]} -> {new_poly[:2]}")
                sample_shown = True
            zone.polygon = new_poly  # reassign so SQLAlchemy detects the JSON change
            zones_done += 1

        sample_shown = False
        for line in lines:
            if only is not None and line.camera_id not in only:
                continue
            new_points = _scaled(line.points)
            if not sample_shown:
                print(f"  e.g. line '{line.name}': {line.points} -> {new_points}")
                sample_shown = True
            line.points = new_points  # in_direction left unchanged on purpose
            lines_done += 1

        if args.dry_run:
            session.rollback()

        print(f"\n{'[dry-run] would scale' if args.dry_run else 'scaled'}: "
              f"zones={zones_done} lines={lines_done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
