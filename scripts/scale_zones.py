#!/usr/bin/env python
"""Scale zone polygon coordinates to match a new stream resolution.

Use this when the camera sub-stream resolution changes and existing zone JSON
files have coordinates in the old resolution space.

Example: 640x360 -> 1280x720  (scale_x=2.0, scale_y=2.0)

Usage
-----
  python scripts/scale_zones.py                          # default: zones/ dir, 2x scale
  python scripts/scale_zones.py --sx 2.0 --sy 2.0       # explicit scale factors
  python scripts/scale_zones.py --old 640x360 --new 1280x720  # auto-compute from resolutions
  python scripts/scale_zones.py --dry-run                # preview without saving
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _scale_polygon(polygon: list, sx: float, sy: float) -> list:
    return [[round(p[0] * sx, 2), round(p[1] * sy, 2)] for p in polygon]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scale zone polygon coordinates to a new resolution")
    parser.add_argument("--dir",     default=None, nargs="+",
                        help="One or more directories to process (default: zones zones1 lines)")
    parser.add_argument("--sx",      type=float,        help="X scale factor (e.g. 2.0)")
    parser.add_argument("--sy",      type=float,        help="Y scale factor (e.g. 2.0)")
    parser.add_argument("--old",                        help="Old resolution e.g. 640x360")
    parser.add_argument("--new",                        help="New resolution e.g. 1280x720")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without saving")
    args = parser.parse_args(argv)

    # Resolve scale factors
    if args.old and args.new:
        ow, oh = (int(v) for v in args.old.lower().split("x"))
        nw, nh = (int(v) for v in args.new.lower().split("x"))
        sx, sy = nw / ow, nh / oh
    elif args.sx is not None and args.sy is not None:
        sx, sy = args.sx, args.sy
    else:
        # Default: 640x360 -> 1280x720
        sx, sy = 2.0, 2.0

    print(f"Scale: x={sx}  y={sy}")
    if args.dry_run:
        print("(dry-run — no files will be modified)\n")

    dirs = args.dir if args.dir else ["zones", "zones1", "Lines"]
    all_files: list[Path] = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            print(f"[WARN] '{d}' does not exist — skipping", file=sys.stderr)
            continue
        all_files.extend(sorted(p.glob("*.json")))

    if not all_files:
        print("No JSON files found in any directory.")
        return 0

    ok = skipped = 0

    for path in all_files:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        changed = False

        # New format: {"zones": [...], "lines": [...]}
        if isinstance(data, dict):
            for zone in data.get("zones", []):
                if "polygon" in zone:
                    zone["polygon"] = _scale_polygon(zone["polygon"], sx, sy)
                    changed = True
            for line in data.get("lines", []):
                if "points" in line:
                    line["points"] = _scale_polygon(line["points"], sx, sy)
                    changed = True
        # Old flat list format: [{"kind": "polygon", "points": [...]}, ...]
        elif isinstance(data, list):
            for shape in data:
                if "points" in shape:
                    shape["points"] = _scale_polygon(shape["points"], sx, sy)
                    changed = True

        if not changed:
            print(f"[SKIP] {path.name}  — no polygon/points found")
            skipped += 1
            continue

        if args.dry_run:
            print(f"[DRY]  {path.name}  — would scale coordinates")
        else:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            print(f"[ OK ] {path.name}")
        ok += 1

    print()
    print(f"Done.  {ok} updated  |  {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
