#!/usr/bin/env python
"""Evaluate a detector's person-detection accuracy on the val split.

Runs ultralytics validation restricted to the person class and prints the key
metrics, so a baseline (stock nano) and a fine-tuned model can be compared on the
exact same held-out set (labels from the teacher ensemble).

    .venv-train/Scripts/python scripts/training/eval.py --model yolo11n.pt
    .venv-train/Scripts/python scripts/training/eval.py --model runs/.../best.pt
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval person detection on the val split")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="dataset/yolo/data.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    r = YOLO(args.model).val(
        data=args.data, imgsz=args.imgsz, classes=[0],
        workers=0, device=args.device, verbose=False, plots=False,
    )
    print(
        f"MODEL {args.model}\n"
        f"  mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  "
        f"precision={r.box.mp:.4f}  recall={r.box.mr:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
