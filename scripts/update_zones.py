#!/usr/bin/env python
"""Update zone polygons in the backend from edited local JSON files.

Fetches all zones from GET /zones, matches each local JSON file to a zone
by (camera_id, name), then PATCHes only the polygon field with the new value.
Zone IDs and all other data stay unchanged.

Usage
-----
  python scripts/update_zones.py --token <token>
  python scripts/update_zones.py --token <token> --zones-dir zones
  python scripts/update_zones.py --token <token> --zones-dir zones zones_1
  python scripts/update_zones.py --token <token> --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Reuse the same filename -> IP -> camera_id mappings from bulk_register_zones.py
IP_TO_CAMERA_ID: dict[str, int] = {
    "10.1.13.48": 1,   "10.1.13.49": 2,   "10.1.13.37": 3,   "10.1.13.41": 4,
    "10.1.13.50": 5,   "10.1.13.52": 6,   "10.1.13.53": 7,   "10.1.13.47": 8,
    "10.1.13.76": 9,   "10.1.13.74": 10,  "10.1.13.77": 11,  "10.1.13.75": 12,
    "10.1.13.78": 13,  "10.1.13.79": 14,  "10.1.13.42": 15,  "10.1.13.43": 16,
    "10.1.13.45": 17,  "10.1.13.54": 18,  "10.1.13.35": 19,  "10.1.13.51": 20,
    "10.1.13.39": 21,  "10.1.13.40": 22,  "10.1.13.63": 23,  "10.1.13.44": 24,
    "10.1.13.38": 25,  "10.1.13.28": 26,  "10.1.13.21": 27,  "10.1.13.80": 28,
    "10.1.13.24": 29,  "10.1.13.29": 30,  "10.1.13.25": 31,  "10.1.13.26": 32,
    "10.1.13.59": 33,  "10.1.13.23": 34,  "10.1.13.27": 35,  "10.1.13.60": 36,
    "10.1.13.61": 37,  "10.1.13.20": 38,  "10.1.13.22": 39,  "10.1.13.30": 40,
    "10.1.13.82": 41,  "10.1.13.81": 42,  "10.1.13.55": 43,  "10.1.13.31": 44,
    "10.1.13.36": 45,  "10.1.13.32": 46,  "10.1.13.56": 47,  "10.1.13.33": 48,
    "10.1.13.34": 49,  "10.1.13.57": 50,  "10.1.13.58": 51,
}

FILENAME_TO_IP: dict[str, str] = {
    "CAM-35_B1-DATA_CENTER":   "10.1.13.54",
    "CAM-16_B1-PASSWAY":       "10.1.13.35",
    "CAM-32_B1-STAIRCASE":     "10.1.13.51",
    "CAM-20_B1-OFFICE":        "10.1.13.39",
    "CAM-25_B1-OFFICE":        "10.1.13.44",
    "CAM-19_B1-LIFT_PASSWAY":  "10.1.13.38",
    "CAM-09_GF_OFFICE_LEFT":   "10.1.13.28",
    "CAM-01_MAIN_DOOR":        "10.1.13.21",
    "CAM-21_GF_BACK_STAIR":    "10.1.13.80",
    "CAM-05_GF-OFFICE_L":      "10.1.13.24",
    "CAM-10_GF_IT_ROOM":       "10.1.13.29",
    "CAM-06_GF-LIFT":          "10.1.13.25",
    "CAM-07_GF-OFFICE_R":      "10.1.13.26",
    "CAM-40_GF-OFFICE-R2":     "10.1.13.59",
    "CAM-04_GF-OFFICE":        "10.1.13.23",
    "CAM-08_GF_OFFICE_RIGHT":  "10.1.13.27",
    "CAM-02_GF-WAITING":       "10.1.13.20",
    "CAM-03_GF-RECEPTION":     "10.1.13.22",
    "CAM-11_GF-STAIRCASE":     "10.1.13.30",
    "CAM-23_F1-BACK_OUTSIDE":  "10.1.13.82",
    "Camera-22_F1-BACK_STAIR": "10.1.13.81",
    "CAM-38_FF-IT-ROOM":       "10.1.13.55",
    "CAM-12_1F-RIGHT_WAY":     "10.1.13.31",
    "CAM-17_1F-LEFT_PASSWAY":  "10.1.13.36",
    "CAM-36_1F-BACK_PASSWAY":  "10.1.13.56",
    "CAM-14_1F-STAIRCASE":     "10.1.13.33",
    "CAM-15_1F-WAITING":       "10.1.13.34",
    "CAM-37_ROOF-LIFT":        "10.1.13.57",
    "CAM-39_ROOF-STAIRCASE":   "10.1.13.58",
    # Manually drawn zones
    "CAM-16_B1-BACK_DOOR":         "10.1.13.75",
    "CAM-17_B1-BACK_LEFT":         "10.1.13.76",
    "CAM-18_B1-BACK_RIGHT":        "10.1.13.77",
    "Cam-18_B2-OFFICE":            "10.1.13.37",
    "CAM-20_B1-RIGHT_EXIT_DOOR":   "10.1.13.79",
    "CAM-21_B1-OFFICE":            "10.1.13.40",
    "Cam-22_B2-OFFICE":            "10.1.13.41",
    "CAM-22_B2-OFFICE":            "10.1.13.41",   # case variant
    "CAM-23_B1-EXIT_DOOR":         "10.1.13.42",
    "CAM-24_B1-OFFICE":            "10.1.13.43",
    "CAM-26_B1-OFFICE":            "10.1.13.45",
    "Cam-28_B2-PASSWAY":           "10.1.13.47",
    "CAM-29_B2-LEFT":              "10.1.13.48",
    "Cam-30_B2_Lift_Passway":      "10.1.13.49",
    "Cam-31_B2-OFFICE":            "10.1.13.50",
    "CAM-31_B2-OFFICE":            "10.1.13.50",   # case variant
    "Cam-33_B2-OFFICE":            "10.1.13.52",
    "Cam-34_B2-OFFICE":            "10.1.13.53",
    "CAM-36_1F_Back_Passway":      "10.1.13.56",
    "CAM-27_B1-WAITING_AREA":      "10.1.13.63",   # missing entry
}


def _get(base_url: str, token: str, path: str):
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _patch(base_url: str, token: str, zone_id: int, polygon: list) -> dict | None:
    url = base_url.rstrip("/") + f"/zones/{zone_id}"
    body = json.dumps({"polygon": polygon}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="PATCH",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"    ERROR {exc.code}: {exc.read().decode()}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update zone polygons from edited local JSON files")
    parser.add_argument("--token",     required=True,                       help="Bearer token")
    parser.add_argument("--api",       default="http://localhost:8008/api/v1", help="API base URL")
    parser.add_argument("--zones-dir", default=None, nargs="+",
                        help="Zone JSON directories (default: zones zones_1)")
    parser.add_argument("--dry-run",   action="store_true",                 help="Print without patching")
    args = parser.parse_args(argv)

    dirs = args.zones_dir if args.zones_dir else ["zones", "zones_1"]

    # Fetch all zones from the backend: build (camera_id, name) -> zone_id lookup
    print(f"Fetching zones from {args.api} ...")
    try:
        all_zones = _get(args.api, args.token, "/zones")
    except Exception as exc:
        print(f"ERROR fetching zones: {exc}", file=sys.stderr)
        return 1

    # (camera_id, zone_name) -> zone_id
    db_lookup: dict[tuple[int, str], int] = {
        (z["camera_id"], z["name"]): z["id"] for z in all_zones
    }
    # camera_id -> list of zone_ids  (for fallback when name doesn't match)
    camera_zones: dict[int, list[int]] = {}
    for z in all_zones:
        camera_zones.setdefault(z["camera_id"], []).append(z["id"])
    print(f"Found {len(db_lookup)} zone(s) in database.\n")

    ok = skipped = failed = 0

    for d in dirs:
        zone_dir = Path(d)
        if not zone_dir.exists():
            print(f"[WARN] '{d}' does not exist — skipping")
            continue

        for path in sorted(zone_dir.glob("*.json")):
            stem = path.stem

            # Resolve camera_id from filename
            ip = FILENAME_TO_IP.get(stem)
            if ip is None:
                print(f"[SKIP] {path.name}  — not in filename map")
                skipped += 1
                continue

            camera_id = IP_TO_CAMERA_ID.get(ip)
            if camera_id is None:
                print(f"[SKIP] {path.name}  — IP {ip} not in camera ID map")
                skipped += 1
                continue

            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)

            zones = data.get("zones", []) if isinstance(data, dict) else []
            if not zones:
                print(f"[SKIP] {path.name}  — no zones inside")
                skipped += 1
                continue

            zone = zones[0]
            zone_name = zone.get("name", "")
            polygon   = zone.get("polygon", [])

            if not polygon:
                print(f"[SKIP] {path.name}  — polygon is empty")
                skipped += 1
                continue

            zone_id = db_lookup.get((camera_id, zone_name))
            if zone_id is None:
                # Fallback: if this camera has exactly one zone, use it regardless of name
                fallback_ids = camera_zones.get(camera_id, [])
                if len(fallback_ids) == 1:
                    zone_id = fallback_ids[0]
                    print(f"[WARN] {path.name}  — name mismatch, using only zone for camera {camera_id} (zone_id={zone_id})")
                elif len(fallback_ids) == 0:
                    print(f"[SKIP] {path.name}  — camera {camera_id} has no zones in DB (register it first)")
                    skipped += 1
                    continue
                else:
                    print(f"[SKIP] {path.name}  — no zone named {zone_name!r} for camera {camera_id} in DB")
                    skipped += 1
                    continue

            if args.dry_run:
                print(f"[DRY]  {path.name}  zone_id={zone_id}  name={zone_name!r}  points={len(polygon)}")
                ok += 1
                continue

            result = _patch(args.api, args.token, zone_id, polygon)
            if result:
                print(f"[ OK ] {path.name}  zone_id={zone_id}  name={result.get('name')!r}  points={len(polygon)}")
                ok += 1
            else:
                print(f"[FAIL] {path.name}  zone_id={zone_id}")
                failed += 1

    print()
    print(f"Done.  {ok} updated  |  {skipped} skipped  |  {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
