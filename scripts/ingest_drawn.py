#!/usr/bin/env python
"""Bulk-ingest drawn zone/line JSON files straight into the database.

Reads every ``*.json`` in a folder (default ``Lines/``). Each file is the output
of ``draw_zones.py``::

    {"zones": [{"name","type","polygon",...}],
     "lines": [{"name","points","in_direction","area_id",...}]}

For each file it works out which camera it belongs to by matching the FILE NAME
to a camera name already in the database -- e.g. ``CAM_17_B1_BACK_LEFT.json``
matches the camera ``"CAM-17 (B1-BACK LEFT)"`` (comparison ignores spaces, dashes,
underscores and case). So you do NOT need camera_id baked into the files.

Writes zones to the ``zones`` table and lines to the ``lines`` table using the
app's own models, against whatever database the app is configured for (SQLite by
default). Re-running is safe: rows that already exist (same camera + same name)
are skipped, so you can add more files and run it again.

Run from the repo root, in the venv::

    python scripts/ingest_drawn.py --dir Lines
    python scripts/ingest_drawn.py --dir Lines --dry-run     # preview only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _norm(text: str) -> str:
    """Lowercase and strip everything except a-z/0-9 for fuzzy name matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _resolve_camera(file_stem: str, cameras: list):
    """Find the camera whose name matches the file name (normalised)."""
    target = _norm(file_stem)
    for cam in cameras:  # exact normalised match first
        if _norm(cam.name) == target:
            return cam
    for cam in cameras:  # then containment either way (handles small differences)
        normalised = _norm(cam.name)
        if normalised and (normalised in target or target in normalised):
            return cam
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk-ingest drawn zones/lines into the DB")
    parser.add_argument("--dir", default="Lines", help="folder with the draw_zones JSON files")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen, write nothing")
    args = parser.parse_args()

    # Imported here so --help works anywhere; run this from the repo root (venv).
    import app.db.models  # noqa: F401  (registers all tables on the metadata)
    from app.db.models.camera import Camera  # noqa: F401
    from app.db.models.geometry import Line, Zone
    from app.db.session import session_scope

    files = sorted(Path(args.dir).glob("*.json"))
    if not files:
        print(f"No .json files found in {args.dir!r}")
        return 1

    with session_scope() as session:
        cameras = session.query(Camera).all()
        if not cameras:
            print("No cameras in the database -- register the cameras first.")
            return 1
        seen_zones = {(z.camera_id, z.name) for z in session.query(Zone).all()}
        seen_lines = {(ln.camera_id, ln.name) for ln in session.query(Line).all()}

        zones_added = lines_added = skipped = unresolved = 0
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or ("zones" not in data and "lines" not in data):
                print(f"  [!] {path.name}: not a draw_zones zones/lines file -> SKIPPED")
                unresolved += 1
                continue
            camera = _resolve_camera(path.stem, cameras)
            if camera is None:
                print(f"  [!] {path.name}: no matching camera -> SKIPPED")
                unresolved += 1
                continue
            print(f"  {path.name}  ->  camera {camera.id} ({camera.name})")

            for zone in data.get("zones", []):
                key = (camera.id, zone["name"])
                if key in seen_zones:
                    skipped += 1
                    continue
                if not args.dry_run:
                    session.add(Zone(
                        camera_id=camera.id, name=zone["name"], type=zone["type"],
                        polygon=zone["polygon"], safe_limit=zone.get("safe_limit"),
                        dt_space_id=zone.get("dt_space_id"),
                    ))
                seen_zones.add(key)
                zones_added += 1

            for line in data.get("lines", []):
                key = (camera.id, line["name"])
                if key in seen_lines:
                    skipped += 1
                    continue
                if not args.dry_run:
                    session.add(Line(
                        camera_id=camera.id, name=line["name"], points=line["points"],
                        in_direction=line["in_direction"], area_id=line["area_id"],
                        dt_space_id=line.get("dt_space_id"),
                    ))
                seen_lines.add(key)
                lines_added += 1

        if args.dry_run:
            session.rollback()

        print(
            f"\n{'[dry-run] would add' if args.dry_run else 'added'}: "
            f"zones={zones_added} lines={lines_added}  "
            f"skipped(existing)={skipped}  unresolved(files)={unresolved}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())