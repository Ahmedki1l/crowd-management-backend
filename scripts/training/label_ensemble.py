#!/usr/bin/env python
"""Ensemble + open-vocabulary auto-labeling of curated frames (the "teacher").

We train teacher-only (no human review), so the label quality IS the ceiling on
the student. To push recall as high as possible on the hard poses (seated,
thobe-from-behind, fisheye) we fuse several strong detectors:

  * yolo11x   — COCO, with test-time augmentation (multi-scale + flip)
  * yolo26x   — newer architecture, COCO
  * YOLO-World — open-vocabulary, text prompt "person"; catches poses the
    closed-vocabulary COCO models miss.

Per image, each model's person boxes are merged with Weighted Box Fusion (boxes
several models agree on get high scores; lone boxes get low ones). Writes YOLO
labels (class 0 = person), overwriting the rough curation labels.

    .venv-train/Scripts/python scripts/training/label_ensemble.py \
        --images dataset/curated/images --out dataset/curated/labels \
        --imgsz 1920 --keep 0.25 --device 0

Runs one image at a time (batch 1) to stay within GPU memory with TTA on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO


def build_models() -> list[dict]:
    """Load the ensemble; skip any member that fails to load."""
    specs: list[dict] = [{"model": YOLO("yolo11x.pt"), "weight": 2.0, "tta": True, "world": False}]
    for weights, w in (("yolo26x.pt", 2.0),):
        try:
            specs.append({"model": YOLO(weights), "weight": w, "tta": False, "world": False})
        except Exception as exc:  # noqa: BLE001
            print(f"skip {weights}: {exc}")
    try:
        world = YOLO("yolov8x-worldv2.pt")
        world.set_classes(["person"])
        specs.append({"model": world, "weight": 1.5, "tta": False, "world": True})
    except Exception as exc:  # noqa: BLE001
        print(f"skip yolo-world: {exc}")
    print(f"ensemble: {len(specs)} models")
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensemble/open-vocab person auto-labeling")
    ap.add_argument("--images", default="dataset/curated/images")
    ap.add_argument("--out", default="dataset/curated/labels")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.10, help="per-model conf floor into WBF")
    ap.add_argument("--iou-thr", type=float, default=0.55, help="WBF fusion IoU")
    ap.add_argument("--keep", type=float, default=0.25, help="min fused score to keep a box")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = build_models()
    weights = [s["weight"] for s in specs]
    images = sorted(Path(args.images).glob("*.jpg"))
    print(f"labeling {len(images)} images @ {args.imgsz}, keep>={args.keep}")

    for i, path in enumerate(images):
        boxes_l, scores_l, labels_l = [], [], []
        for s in specs:
            kw = dict(imgsz=args.imgsz, conf=args.conf, iou=0.6,
                      device=args.device, verbose=False, augment=s["tta"])
            if not s["world"]:
                kw["classes"] = [0]
            res = s["model"].predict(str(path), **kw)[0]
            boxes_l.append(res.boxes.xyxyn.cpu().numpy().tolist())
            scores_l.append(res.boxes.conf.cpu().numpy().tolist())
            labels_l.append([0] * len(res.boxes))

        fused_b, fused_s, _ = weighted_boxes_fusion(
            boxes_l, scores_l, labels_l, weights=weights,
            iou_thr=args.iou_thr, skip_box_thr=0.0,
        )
        lines = []
        for (x1, y1, x2, y2), score in zip(fused_b, fused_s, strict=True):
            if score < args.keep:
                continue
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (out / f"{path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(images)}", flush=True)

    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
