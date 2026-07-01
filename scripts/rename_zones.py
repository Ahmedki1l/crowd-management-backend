#!/usr/bin/env python
"""Rename auto_zones.py output files to match the camera name in the JSON.

Each file produced by auto_zones.py contains a "name" field inside the first
zone.  This script reads that name and renames the file to:

    <camera name with spaces replaced by underscores>.json

Usage
-----
  python scripts/rename_zones.py              # default: zones/ directory
  python scripts/rename_zones.py --dir my_zones
  python scripts/rename_zones.py --dry-run    # preview without renaming
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _slug(name: str) -> str:
    """Camera name -> filename slug: spaces and () removed/replaced, doubles collapsed."""
    s = name.replace("(", "").replace(")", "").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rename zone JSON files to their camera name")
    parser.add_argument("--dir", default="zones", help="Directory containing zone JSON files (default: zones/)")
    parser.add_argument("--dry-run", action="store_true", help="Print renames without applying them")
    args = parser.parse_args(argv)

    zone_dir = Path(args.dir)
    if not zone_dir.exists():
        print(f"ERROR: directory '{zone_dir}' does not exist", file=sys.stderr)
        return 1

    files = sorted(zone_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in '{zone_dir}'")
        return 0

    renamed = skipped = errors = 0

    for path in files:
        try:
            # The filename itself IS the camera name slug — just confirm it matches
            # the slug format (spaces/parens removed, underscores).
            # Re-slug the stem to normalise any legacy formats.
            new_filename = _slug(path.stem) + ".json"
            new_path = path.parent / new_filename

            if new_path == path:
                print(f"[SAME] {path.name}  — already correct")
                skipped += 1
                continue

            if new_path.exists():
                print(f"[SKIP] {path.name}  — target '{new_filename}' already exists")
                skipped += 1
                continue

            if args.dry_run:
                print(f"[DRY]  {path.name}  ->  {new_filename}")
            else:
                path.rename(new_path)
                print(f"[OK]   {path.name}  ->  {new_filename}")
            renamed += 1

        except OSError as exc:
            print(f"[ERR]  {path.name}  — {exc}", file=sys.stderr)
            errors += 1

    print()
    if args.dry_run:
        print(f"Dry run: {renamed} would be renamed  |  {skipped} skipped  |  {errors} errors")
    else:
        print(f"Done: {renamed} renamed  |  {skipped} skipped  |  {errors} errors")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
