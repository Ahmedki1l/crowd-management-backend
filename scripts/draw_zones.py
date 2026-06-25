#!/usr/bin/env python
"""Draw zone polygons and counting lines on a camera frame (HLD 12.1).

Interactive helper to produce the image-space coordinates that the API expects
(POST /zones polygon, POST /lines points). Left-click adds points; keys:
  n = finish current polygon/line   l = toggle line mode (2 points)
  u = undo last point   s = save JSON   q = quit

    python scripts/draw_zones.py --image frame.jpg --out zones.json
    python scripts/draw_zones.py --url rtsp://user:pass@host:554/Streaming/Channels/102
"""

from __future__ import annotations

import argparse
import json
import sys


def _grab_frame(url: str):
    import cv2  # lazy

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not read a frame from {url!r}")
        return frame
    finally:
        cap.release()


def run(image_path: str | None, url: str | None, out_path: str) -> int:
    import cv2  # lazy

    frame = cv2.imread(image_path) if image_path else _grab_frame(url or "")
    if frame is None:
        raise RuntimeError("no frame to annotate (bad --image path?)")

    shapes: list[dict] = []
    current: list[list[int]] = []
    line_mode = {"on": False}

    def on_mouse(event, x, y, _flags, _param):  # noqa: ANN001
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([int(x), int(y)])

    window = "draw_zones (n=finish u=undo l=line s=save q=quit)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = frame.copy()
        for shape in shapes:
            pts = shape["points"]
            color = (0, 0, 255) if shape["kind"] == "line" else (0, 255, 0)
            for px, py in pts:
                cv2.circle(canvas, (px, py), 4, color, -1)
            if len(pts) > 1:
                closed = shape["kind"] == "polygon"
                import numpy as np

                cv2.polylines(canvas, [np.array(pts)], closed, color, 2)
        for px, py in current:
            cv2.circle(canvas, (px, py), 4, (255, 0, 0), -1)
        cv2.putText(
            canvas, "LINE" if line_mode["on"] else "POLYGON", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and current:
            current.pop()
        if key == ord("l"):
            line_mode["on"] = not line_mode["on"]
        if key == ord("n") and current:
            kind = "line" if line_mode["on"] else "polygon"
            if kind == "line" and len(current) != 2:
                print("a line needs exactly 2 points", file=sys.stderr)
            else:
                shapes.append({"kind": kind, "points": list(current)})
                current.clear()
        if key == ord("s"):
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(shapes, fh, indent=2)
            print(f"saved {len(shapes)} shapes -> {out_path}")

    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw zones/lines on a camera frame")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to a still frame")
    src.add_argument("--url", help="RTSP URL to grab one frame from")
    parser.add_argument("--out", default="zones.json", help="output JSON path")
    args = parser.parse_args(argv)
    return run(args.image, args.url, args.out)


if __name__ == "__main__":
    sys.exit(main())
