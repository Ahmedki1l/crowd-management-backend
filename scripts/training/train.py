#!/usr/bin/env python
"""Fine-tune the person detector on the curated dataset, then export for CPU.

Transfer-learns from COCO-pretrained ``yolo11m.pt`` (the model already in prod) on
our local frames, so it keeps general person knowledge and specialises to these
cameras/poses (seated, thobe-from-behind, fisheye). Trains on the GPU, then
exports OpenVINO FP16 (dynamic) — the runtime the server uses on CPU.

    .venv-train/Scripts/python scripts/training/train.py \
        --data dataset/yolo/data.yaml --model yolo11m.pt --imgsz 1280 --epochs 100 --device 0

Swap ``--model yolo26m.pt`` to fine-tune YOLO26 instead.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune the person detector")
    ap.add_argument("--data", default="dataset/yolo/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="student to fine-tune; nano = cheap CPU deploy")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train data (pilot runs)")
    ap.add_argument("--batch", type=int, default=8,
                    help="8 is safe on this box's Windows page file; larger can hit 'bad allocation' in backward")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0,
                    help="dataloader workers; 0 avoids Windows page-file/CUDA-DLL exhaustion (WinError 1455)")
    ap.add_argument("--cache", default=None,
                    help="ram | disk — cache decoded images so a slow disk / workers=0 doesn't starve the GPU")
    ap.add_argument("--project", default="training_runs")
    ap.add_argument("--name", default="person_ft")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache or False,
        project=args.project,
        name=args.name,
        patience=20,          # early-stop if val mAP plateaus
        single_cls=True,      # one class: person
        cos_lr=True,
        # light augmentation suited to fixed indoor CCTV (no vertical flip, mild HSV)
        fliplr=0.5, flipud=0.0, degrees=0.0, mosaic=1.0, close_mosaic=10,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
    )

    if not args.no_export:
        from pathlib import Path

        best = Path(model.trainer.save_dir) / "weights" / "best.pt"  # actual run dir
        out = YOLO(str(best)).export(format="openvino", half=True, dynamic=True)
        print(f"best weights -> {best}")
        print(f"exported for CPU deployment -> {out}")
        print("point detector.model_path (config.local.yaml + .env MODEL_DETECTOR_PATH) at it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
