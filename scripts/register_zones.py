#!/usr/bin/env python
"""Register zones and counting lines from a draw_zones.py output file (HLD 8.1).

Reads the JSON produced by ``draw_zones.py --out zones.json``, prompts for the
metadata the API requires for each shape, and POSTs them to the REST API.
No extra dependencies are required beyond the standard library.

Usage
-----
  # 1. Grab a frame and draw your shapes:
  python scripts/draw_zones.py --url rtsp://user:pass@host:554/Streaming/Channels/102 --out zones.json

  # 2. Register them:
  python scripts/register_zones.py \\
      --file zones.json \\
      --camera-id 1 \\
      --api http://localhost:8008/api/v1 \\
      --token <bearer-token>
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# Friendly direction labels → [dx, dy] normal vectors for line IN crossing.
# The vector points in the direction a person is moving when they count as IN.
_IN_DIRECTIONS: dict[str, list[int]] = {
    "left_to_right": [1, 0],
    "right_to_left": [-1, 0],
    "top_to_bottom": [0, 1],
    "bottom_to_top": [0, -1],
}

_ZONE_TYPES = ("occupancy", "waiting", "restricted")


def _post(base_url: str, token: str, path: str, body: dict) -> dict | None:
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"  ERROR {exc.code}: {exc.read().decode()}", file=sys.stderr)
        return None


def _ask(prompt: str, choices: tuple[str, ...] | list[str] | None = None) -> str:
    while True:
        value = input(f"  {prompt}: ").strip()
        if choices is None or value.lower() in choices:
            return value.lower() if choices else value
        print(f"  Must be one of: {', '.join(choices)}", file=sys.stderr)


def _register_polygon(base_url: str, token: str, camera_id: int, points: list) -> None:
    name = _ask("Zone name")
    zone_type = _ask(f"Type  [{'/'.join(_ZONE_TYPES)}]", _ZONE_TYPES)
    safe_str = input("  Safe limit (press Enter to skip): ").strip()
    dt_id = input("  DT space ID (press Enter to skip): ").strip() or None

    body: dict = {
        "camera_id": camera_id,
        "name": name,
        "type": zone_type,
        "polygon": points,
    }
    if safe_str.isdigit():
        body["safe_limit"] = int(safe_str)
    if dt_id:
        body["dt_space_id"] = dt_id

    result = _post(base_url, token, "/zones", body)
    if result:
        print(f"  ✓ Zone created  id={result.get('id')}  name={result.get('name')}")


def _register_line(base_url: str, token: str, camera_id: int, points: list) -> None:
    name = _ask("Line name")
    dirs = list(_IN_DIRECTIONS)
    direction_key = _ask(f"IN direction  [{'/'.join(dirs)}]", dirs)
    area_id = _ask("Area ID")
    dt_id = input("  DT space ID (press Enter to skip): ").strip() or None

    body: dict = {
        "camera_id": camera_id,
        "name": name,
        "points": points,
        "in_direction": _IN_DIRECTIONS[direction_key],
        "area_id": area_id,
    }
    if dt_id:
        body["dt_space_id"] = dt_id

    result = _post(base_url, token, "/lines", body)
    if result:
        print(f"  ✓ Line created   id={result.get('id')}  name={result.get('name')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register zones/lines from draw_zones.py output via the REST API"
    )
    parser.add_argument("--file", required=True, help="JSON file produced by draw_zones.py")
    parser.add_argument("--camera-id", required=True, type=int, help="Camera ID to attach shapes to")
    parser.add_argument("--api", default="http://localhost:8008/api/v1", help="API base URL")
    parser.add_argument("--token", required=True, help="Bearer token for authentication")
    args = parser.parse_args(argv)

    with open(args.file, encoding="utf-8") as fh:
        raw = json.load(fh)

    # Support both formats:
    #   New (auto_zones.py / draw_zones.py):  {"zones": [...], "lines": [...]}
    #   Old (draw_zones.py legacy):           [{"kind": "polygon", "points": [...]}, ...]
    if isinstance(raw, dict):
        # New format — zones already have name/type/safe_limit/dt_space_id filled in
        zones_data = raw.get("zones", [])
        lines_data = raw.get("lines", [])
        polygons = [{"points": z["polygon"], "name": z.get("name"), "type": z.get("type", "occupancy"),
                     "safe_limit": z.get("safe_limit"), "dt_space_id": z.get("dt_space_id")}
                    for z in zones_data if z.get("polygon")]
        lines = [{"points": ln["points"], "name": ln.get("name"), "area_id": ln.get("area_id"),
                  "in_direction": ln.get("in_direction"), "dt_space_id": ln.get("dt_space_id")}
                 for ln in lines_data if ln.get("points")]
        new_format = True
    else:
        polygons = [s for s in raw if s.get("kind") == "polygon"]
        lines    = [s for s in raw if s.get("kind") == "line"]
        new_format = False

    print(f"Loaded {len(polygons)} polygon(s) and {len(lines)} line(s) from '{args.file}'.")
    print(f"Camera ID: {args.camera_id}   API: {args.api}\n")

    for i, shape in enumerate(polygons, 1):
        preview = shape["points"][:2]
        if new_format:
            print(f"--- Zone {i}/{len(polygons)}  name={shape['name']!r}  type={shape['type']}  first 2 pts: {preview} ---")
        else:
            print(f"--- Polygon {i}/{len(polygons)}  first 2 pts: {preview} ---")
        skip = input("  Register this zone? [Y/n]: ").strip().lower()
        if skip == "n":
            print("  Skipped.")
            continue
        if new_format and shape.get("name"):
            # Already has all metadata — POST directly, no prompting
            body: dict = {
                "camera_id": args.camera_id,
                "name": shape["name"],
                "type": shape["type"],
                "polygon": shape["points"],
            }
            if shape.get("safe_limit") is not None:
                body["safe_limit"] = shape["safe_limit"]
            if shape.get("dt_space_id"):
                body["dt_space_id"] = shape["dt_space_id"]
            result = _post(args.api, args.token, "/zones", body)
            if result:
                print(f"  ✓ Zone created  id={result.get('id')}  name={result.get('name')}")
        else:
            _register_polygon(args.api, args.token, args.camera_id, shape["points"])

    for i, shape in enumerate(lines, 1):
        print(f"\n--- Line {i}/{len(lines)}  points: {shape['points']} ---")
        skip = input("  Register this line? [Y/n]: ").strip().lower()
        if skip == "n":
            print("  Skipped.")
            continue
        if new_format and shape.get("name") and shape.get("in_direction"):
            body = {
                "camera_id": args.camera_id,
                "name": shape["name"],
                "points": shape["points"],
                "in_direction": shape["in_direction"],
                "area_id": shape.get("area_id", "area1"),
            }
            if shape.get("dt_space_id"):
                body["dt_space_id"] = shape["dt_space_id"]
            result = _post(args.api, args.token, "/lines", body)
            if result:
                print(f"  ✓ Line created  id={result.get('id')}  name={result.get('name')}")
        else:
            _register_line(args.api, args.token, args.camera_id, shape["points"])

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
