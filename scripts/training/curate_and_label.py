#!/usr/bin/env python
"""Detection-driven curation + pseudo-labeling of captured frames (GPU).

Whole-frame hashing fails here: a small person in a large static scene barely
changes the image, so hash-dedup throws away exactly the people-in-new-positions
frames we need. Instead this runs a strong detector on the GPU and keeps, per
camera, frames whose *person configuration* (count + quantised box centres) is
new — maximising pose/position/occupancy variety — plus a bounded sample of empty
frames as negatives. Pseudo-labels (class 0 = person) are written in the same
pass, so a human only has to correct them afterwards.

    .venv-train/Scripts/python scripts/training/curate_and_label.py \
        --src training_data --out dataset/curated --model yolo11x.pt \
        --imgsz 1920 --conf 0.15 --stride 2 --people-cap 400 --empty-cap 100 --device 0

NOTE: pseudo-labels (and the keep/empty decision) inherit the base model's blind
spots — a frame whose only person is a missed thobe/seated case looks "empty". The
kept empties are therefore also review material; the human pass is what closes the gap.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

GRID = 12  # spatial quantisation for the "is this person layout new?" signature


def signature(boxes_xywhn) -> tuple:
    """(person_count, frozenset of quantised box centres) for a frame."""
    cells = frozenset((int(cx * GRID), int(cy * GRID)) for cx, cy, _w, _h in boxes_xywhn)
    return (len(boxes_xywhn), cells)


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU curation + person pseudo-labels")
    ap.add_argument("--src", default="training_data")
    ap.add_argument("--out", default="dataset/curated")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--stride", type=int, default=2, help="process every Nth frame (redundancy)")
    ap.add_argument("--desc", action="store_true",
                    help="process newest frames first so per-camera caps fill with recent (daytime) people")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--people-cap", type=int, default=400, help="max distinct people-frames/camera")
    ap.add_argument("--empty-cap", type=int, default=100, help="max empty frames/camera (negatives)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    out_img = Path(args.out) / "images"
    out_lbl = Path(args.out) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    grand = {"people": 0, "empty": 0}
    for cam_dir in sorted(d for d in Path(args.src).iterdir() if d.is_dir()):
        files = sorted(cam_dir.rglob("*.jpg"), reverse=args.desc)[:: args.stride]
        seen: set[tuple] = set()
        n_people = 0
        empties: list[tuple[Path, list]] = []

        def keep(path: Path, boxes) -> None:
            shutil.copy2(path, out_img / f"{cam_dir.name}__{path.name}")
            lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes]
            (out_lbl / f"{cam_dir.name}__{path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        for i in range(0, len(files), args.batch):
            batch = files[i : i + args.batch]
            results = model.predict(
                [str(p) for p in batch], imgsz=args.imgsz, conf=args.conf,
                classes=[0], device=args.device, verbose=False,
            )
            for path, res in zip(batch, results, strict=True):
                boxes = res.boxes.xywhn.cpu().numpy()
                if len(boxes) == 0:
                    empties.append((path, boxes))
                    continue
                sig = signature(boxes)
                if sig not in seen and n_people < args.people_cap:
                    seen.add(sig)
                    keep(path, boxes)
                    n_people += 1
            if n_people >= args.people_cap:
                break  # cap reached; with --desc this keeps us in the recent window

        # Even sample of empties as negatives (also human-review candidates).
        step = max(1, len(empties) // args.empty_cap) if empties else 1
        n_empty = 0
        for path, boxes in empties[::step][: args.empty_cap]:
            keep(path, boxes)
            n_empty += 1

        grand["people"] += n_people
        grand["empty"] += n_empty
        print(f"{cam_dir.name}: people={n_people} empty={n_empty} (scanned {len(files)})", flush=True)

    print(f"\nCURATED people={grand['people']} empty={grand['empty']} -> {out_img}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
