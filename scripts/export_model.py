#!/usr/bin/env python
"""Export / quantise the person detector to a deployable runtime (HLD 5.2, 11).

  CPU  (OpenVINO INT8):  python scripts/export_model.py --weights yolo11n.pt --format openvino --int8
  GPU  (TensorRT FP16):  python scripts/export_model.py --weights yolo11n.pt --format engine --half

The exported artifact goes under models/detector/ and is referenced by
detector.model_path in config.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the YOLO detector")
    parser.add_argument("--weights", required=True, help="path to YOLO .pt weights")
    parser.add_argument(
        "--format", choices=["openvino", "engine", "onnx"], default="openvino",
        help="openvino (CPU) | engine (TensorRT/GPU) | onnx",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true", help="INT8 quantisation (OpenVINO/CPU)")
    parser.add_argument("--half", action="store_true", help="FP16 (TensorRT/GPU)")
    parser.add_argument("--data", default=None, help="calibration dataset yaml for INT8")
    args = parser.parse_args(argv)

    from ultralytics import YOLO  # lazy: only needed at export time

    model = YOLO(args.weights)
    export_kwargs: dict = {"format": args.format, "imgsz": args.imgsz}
    if args.int8:
        export_kwargs["int8"] = True
        if args.data:
            export_kwargs["data"] = args.data
    if args.half:
        export_kwargs["half"] = True

    out = model.export(**export_kwargs)
    print(f"exported -> {out}")
    print("Move/point detector.model_path (config) at this artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
