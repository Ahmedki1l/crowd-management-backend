#!/usr/bin/env python
"""Assemble a YOLO train/val dataset from curated images + labels.

Splits stratified **by camera** so both train and val cover every view (a split
that put a whole camera in val would test on an unseen scene, not unseen people).
Images with no label file become explicit negatives (empty label) so the model
also learns "no person here" on empty corridors/floors.

    python scripts/training/build_dataset.py --images dataset/curated/images \
        --labels dataset/curated/labels --out dataset/yolo --val-frac 0.2

Writes dataset/yolo/{images,labels}/{train,val} and dataset/yolo/data.yaml.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a YOLO dataset from curated data")
    ap.add_argument("--images", default="dataset/curated/images")
    ap.add_argument("--labels", default="dataset/curated/labels")
    ap.add_argument("--out", default="dataset/yolo")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    by_cam: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(Path(args.images).glob("*.jpg")):
        by_cam[p.name.split("__")[0]].append(p)

    n = {"train": 0, "val": 0}
    for _cam, paths in by_cam.items():
        rnd.shuffle(paths)
        cut = int(len(paths) * args.val_frac)
        for split, group in (("val", paths[:cut]), ("train", paths[cut:])):
            for img in group:
                shutil.copy2(img, out / "images" / split / img.name)
                src_lbl = Path(args.labels) / f"{img.stem}.txt"
                dst_lbl = out / "labels" / split / f"{img.stem}.txt"
                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    dst_lbl.write_text("", encoding="utf-8")  # explicit negative
                n[split] += 1

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: person\n",
        encoding="utf-8",
    )
    print(f"train={n['train']} val={n['val']} cameras={len(by_cam)} -> {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
