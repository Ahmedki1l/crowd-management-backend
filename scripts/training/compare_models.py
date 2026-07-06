#!/usr/bin/env python
"""Side-by-side detection comparison of two+ models on the same frames.

Used to eyeball the before/after of fine-tuning: run stock nano and the
fine-tuned nano on identical images and stack the annotated panels, with a
per-panel person count. Green boxes = detections.

    .venv-train/Scripts/python scripts/training/compare_models.py \
        --models yolo11n.pt,runs/detect/training_runs/nano_pilot/weights/best.pt \
        --images <img1.jpg> <img2.jpg> ... --out dataset/_compare_models --imgsz 1280 --conf 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PANEL_H = 540


def annotate(img, boxes, label):
    frame = img.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 3)
    h = int(frame.shape[0] * PANEL_H / frame.shape[1] * (frame.shape[1] / frame.shape[0]))
    panel = cv2.resize(frame, (int(frame.shape[1] * PANEL_H / frame.shape[0]), PANEL_H))
    cv2.rectangle(panel, (0, 0), (460, 40), (0, 0, 0), -1)
    cv2.putText(panel, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 0), 2)
    return panel


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare models' detections side by side")
    ap.add_argument("--models", required=True, help="comma-separated model paths")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="dataset/_compare_models")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_paths = args.models.split(",")
    models = [(Path(p).stem, YOLO(p)) for p in model_paths]

    for img_path in args.images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"skip {img_path}")
            continue
        panels = []
        counts = []
        for name, m in models:
            r = m.predict(img, imgsz=args.imgsz, conf=args.conf, classes=[0],
                          device=args.device, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            counts.append(len(boxes))
            panels.append(annotate(img, boxes, f"{name}: {len(boxes)}"))
        combo = np.hstack(panels)
        name = Path(img_path).stem[:40]
        cv2.imwrite(str(out / f"{name}.jpg"), combo)
        print(f"{name}: " + "  ".join(f"{n}={c}" for (n, _), c in zip(models, counts)), flush=True)
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
