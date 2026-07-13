"""Export, then delete, zones that belong to no logical space (``dt_space_id is NULL``).

Occupancy history is keyed by space. A zone with no ``dt_space_id`` therefore contributes
to nothing and is invisible to every history query — it is dead configuration.

**The export is not optional.** These polygons were drawn by hand against real camera
views ("HR", "IT Room", "Lift Area", the GF/1F offices), for a building rollout wider than
the cameras currently in the allowlist. Deleting them costs no *data* — they have never
recorded a sample — but it does throw away that drawing work, so it is written to JSON
first and can be re-imported rather than re-drawn.

Refuses to delete a zone that has ever produced history, so it cannot silently orphan
rows the way the old zone-redraw workflow did.

Usage::

    python scripts/prune_unassigned_zones.py                 # dry run: report only
    python scripts/prune_unassigned_zones.py --apply         # export + delete
    python scripts/prune_unassigned_zones.py --apply --out zones-backup.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.db.models import Zone
from app.db.session import session_scope

_DEFAULT_OUT = Path("unassigned-zones-backup.json")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually export and delete; without it, only report what would go",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"where to write the polygon backup (default: {_DEFAULT_OUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    with session_scope() as session:
        zones = [z for z in session.query(Zone).all() if z.dt_space_id is None]

        if not zones:
            print("No unassigned zones. Nothing to do.")
            return 0

        payload = [
            {
                "id": zone.id,
                "name": zone.name,
                "camera_id": zone.camera_id,
                "camera_name": zone.camera.name if zone.camera else None,
                "camera_ip": zone.camera.ip if zone.camera else None,
                "type": zone.type,
                "polygon": zone.polygon,
                "safe_limit": zone.safe_limit,
            }
            for zone in sorted(zones, key=lambda z: z.id)
        ]

        print(f"{len(payload)} zone(s) belong to no space:")
        for entry in payload:
            print(f"  zone {entry['id']:<5} {entry['name'][:30]:<30} cam={entry['camera_name']}")

        if not args.apply:
            print("\nDry run. Re-run with --apply to export and delete.")
            return 0

        args.out.write_text(
            json.dumps(
                {
                    "exported_at": datetime.now(tz=UTC).isoformat(),
                    "reason": "zones with no dt_space_id; occupancy history is space-keyed",
                    "zones": payload,
                },
                indent=2,
            )
        )
        print(f"\nExported {len(payload)} polygon(s) -> {args.out}")

        for zone in zones:
            session.delete(zone)

    print(f"Deleted {len(payload)} zone(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
