# #!/usr/bin/env python
# """Draw zone polygons and counting lines on a camera frame (HLD 12.1).

# Interactive helper to produce the image-space coordinates that the API expects
# (POST /zones polygon, POST /lines points). Left-click adds points; keys:
#   n = finish current polygon/line   l = toggle line mode (2 points)
#   u = undo last point   s = save JSON   q = quit

#     python scripts/draw_zones.py --image frame.jpg --out zones.json
#     python scripts/draw_zones.py --url rtsp://user:pass@host:554/Streaming/Channels/102
# """

# from __future__ import annotations

# import argparse
# import json
# import sys


# def _grab_frame(url: str):
#     import cv2  # lazy

#     cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#     try:
#         ok, frame = cap.read()
#         if not ok or frame is None:
#             raise RuntimeError(f"could not read a frame from {url!r}")
#         return frame
#     finally:
#         cap.release()


# def run(image_path: str | None, url: str | None, out_path: str) -> int:
#     import cv2  # lazy

#     frame = cv2.imread(image_path) if image_path else _grab_frame(url or "")
#     if frame is None:
#         raise RuntimeError("no frame to annotate (bad --image path?)")

#     shapes: list[dict] = []
#     current: list[list[int]] = []
#     line_mode = {"on": False}

#     def on_mouse(event, x, y, _flags, _param):  # noqa: ANN001
#         if event == cv2.EVENT_LBUTTONDOWN:
#             current.append([int(x), int(y)])

#     window = "draw_zones (n=finish u=undo l=line s=save q=quit)"
#     cv2.namedWindow(window)
#     cv2.setMouseCallback(window, on_mouse)

#     while True:
#         canvas = frame.copy()
#         for shape in shapes:
#             pts = shape["points"]
#             color = (0, 0, 255) if shape["kind"] == "line" else (0, 255, 0)
#             for px, py in pts:
#                 cv2.circle(canvas, (px, py), 4, color, -1)
#             if len(pts) > 1:
#                 closed = shape["kind"] == "polygon"
#                 import numpy as np

#                 cv2.polylines(canvas, [np.array(pts)], closed, color, 2)
#         for px, py in current:
#             cv2.circle(canvas, (px, py), 4, (255, 0, 0), -1)
#         cv2.putText(
#             canvas, "LINE" if line_mode["on"] else "POLYGON", (10, 24),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
#         )
#         cv2.imshow(window, canvas)
#         key = cv2.waitKey(20) & 0xFF
#         if key == ord("q"):
#             break
#         if key == ord("u") and current:
#             current.pop()
#         if key == ord("l"):
#             line_mode["on"] = not line_mode["on"]
#         if key == ord("n") and current:
#             kind = "line" if line_mode["on"] else "polygon"
#             if kind == "line" and len(current) != 2:
#                 print("a line needs exactly 2 points", file=sys.stderr)
#             else:
#                 shapes.append({"kind": kind, "points": list(current)})
#                 current.clear()
#         if key == ord("s"):
#             with open(out_path, "w", encoding="utf-8") as fh:
#                 json.dump(shapes, fh, indent=2)
#             print(f"saved {len(shapes)} shapes -> {out_path}")

#     cv2.destroyAllWindows()
#     return 0


# def main(argv: list[str] | None = None) -> int:
#     parser = argparse.ArgumentParser(description="Draw zones/lines on a camera frame")
#     src = parser.add_mutually_exclusive_group(required=True)
#     src.add_argument("--image", help="path to a still frame")
#     src.add_argument("--url", help="RTSP URL to grab one frame from")
#     parser.add_argument("--out", default="zones.json", help="output JSON path")
#     args = parser.parse_args(argv)
#     return run(args.image, args.url, args.out)


# if __name__ == "__main__":
#     sys.exit(main())


#!/usr/bin/env python
"""Draw zone polygons and counting lines on a camera frame (HLD 12.1).

Interactive helper that produces the exact JSON the API expects
(``POST /zones`` and ``POST /lines``), with naming, zone types, edit/remove/
rename, and a clickable IN direction for lines.

Output (``--out zones.json``)::

    {
      "zones": [{"camera_id": 1, "name": "Lobby", "type": "occupancy",
                 "polygon": [[x,y], ...], "safe_limit": null, "dt_space_id": null}],
      "lines": [{"camera_id": 1, "name": "Main Door", "points": [[x,y],[x,y]],
                 "in_direction": [dx,dy], "area_id": "lobby", "dt_space_id": null}]
    }

Run::

    python scripts/draw_zones.py --image frame.jpg --camera-id 1 --out zones.json
    python scripts/draw_zones.py --url rtsp://user:pass@host:554/Streaming/Channels/102

Controls (shown in the window):
    Left click   - add a point  (EDIT: grab/drag a point · REMOVE/RENAME: pick a shape)
    Right click / f - finish the current ZONE (needs >= 3 points)
    LINE mode    - click the 2 ends, then 1 point on the IN side -> auto-finishes
    l = toggle LINE mode      u = undo last point
    e = EDIT mode (drag points)   r = REMOVE mode   m = RENAME/edit props
    s = save JSON             q / ESC = quit
"""

from __future__ import annotations

import argparse
import json
import math
import sys

_ZONE_TYPES = {"o": "occupancy", "w": "waiting", "r": "restricted"}
_TYPE_COLOR = {
    "occupancy": (0, 200, 0),
    "waiting": (0, 180, 230),
    "restricted": (0, 0, 230),
}
_LINE_COLOR = (255, 255, 0)


def line_in_direction(p1: list[int], p2: list[int], inside: list[int]) -> list[float]:
    """Unit normal of segment ``p1->p2`` pointing toward the ``inside`` click.

    The API's ``in_direction`` is a ``[dx, dy]`` vector that counts as IN; this
    returns the perpendicular to the line, flipped so it points to the side the
    operator clicked.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    nx, ny = -dy, dx
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    if (inside[0] - mx) * nx + (inside[1] - my) * ny < 0:
        nx, ny = -nx, -ny
    norm = math.hypot(nx, ny) or 1.0
    return [round(nx / norm, 4), round(ny / norm, 4)]


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


class ZoneLineDrawer:
    """Interactive zone/line editor with draw, edit, remove and rename modes."""

    DRAW, EDIT, REMOVE, RENAME = "DRAW", "EDIT", "REMOVE", "RENAME"

    def __init__(self, frame, camera_id: int | None) -> None:
        self.frame = frame
        self.camera_id = camera_id
        self.zones: list[dict] = []
        self.lines: list[dict] = []
        self.current: list[list[int]] = []
        self.line_mode = False
        self.mode = self.DRAW
        self._drag = (-1, -1, "")  # (shape_index, point_index, "zone"|"line")
        self._cursor = (0, 0)

    # ---- geometry helpers --------------------------------------------------
    def _scale(self) -> float:
        return max(1.0, self.frame.shape[1] / 1280.0)

    def _grab_radius(self) -> int:
        return int(round(15 * self._scale()))

    def _shapes(self):
        for i, z in enumerate(self.zones):
            yield ("zone", i, z["polygon"])
        for i, ln in enumerate(self.lines):
            yield ("line", i, ln["points"])

    def _nearest_vertex(self, x, y):
        best, found = self._grab_radius(), (-1, -1, "")
        for kind, idx, pts in self._shapes():
            for pi, (px, py) in enumerate(pts):
                d = math.hypot(px - x, py - y)
                if d < best:
                    best, found = d, (idx, pi, kind)
        return found

    def _pick_shape(self, x, y):
        import cv2
        import numpy as np

        for kind, idx, pts in self._shapes():
            if kind == "zone" and len(pts) >= 3:
                if cv2.pointPolygonTest(np.array(pts, np.int32), (x, y), False) >= 0:
                    return kind, idx
            else:  # line: pick if click is near the segment
                if len(pts) == 2 and self._point_near_segment(x, y, pts[0], pts[1]) < 12:
                    return kind, idx
        return None, -1

    @staticmethod
    def _point_near_segment(x, y, a, b) -> float:
        ax, ay, bx, by = a[0], a[1], b[0], b[1]
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.hypot(x - ax, y - ay)
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
        return math.hypot(x - (ax + t * dx), y - (ay + t * dy))

    # ---- mouse -------------------------------------------------------------
    def on_mouse(self, event, x, y, _flags, _param):
        import cv2

        self._cursor = (x, y)
        if self.mode == self.DRAW:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current.append([int(x), int(y)])
                if self.line_mode and len(self.current) == 3:
                    self._finish_line()
            elif event == cv2.EVENT_RBUTTONDOWN and not self.line_mode:
                self._finish_zone()
        elif self.mode == self.EDIT:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._drag = self._nearest_vertex(x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self._drag[0] >= 0:
                idx, pi, kind = self._drag
                pts = self.zones[idx]["polygon"] if kind == "zone" else self.lines[idx]["points"]
                pts[pi] = [int(x), int(y)]
            elif event == cv2.EVENT_LBUTTONUP:
                self._drag = (-1, -1, "")
        elif event == cv2.EVENT_LBUTTONDOWN and self.mode in (self.REMOVE, self.RENAME):
            kind, idx = self._pick_shape(x, y)
            if idx < 0:
                print("  click inside a zone / on a line")
            elif self.mode == self.REMOVE:
                gone = (self.zones if kind == "zone" else self.lines).pop(idx)
                print(f"  removed {kind} '{gone['name']}'")
            else:
                self._rename(kind, idx)

    # ---- finish / prompts --------------------------------------------------
    def _finish_zone(self):
        if len(self.current) < 3:
            print("  a zone needs at least 3 points", file=sys.stderr)
            return
        z = self._prompt_zone(default_name=f"Z{len(self.zones) + 1}")
        z["polygon"] = [list(p) for p in self.current]
        self.zones.append(z)
        print(f"  [OK] zone '{z['name']}' ({z['type']}, {len(z['polygon'])} pts)")
        self.current = []

    def _finish_line(self):
        p1, p2, inside = self.current[0], self.current[1], self.current[2]
        ln = self._prompt_line(default_name=f"L{len(self.lines) + 1}")
        ln["points"] = [list(p1), list(p2)]
        ln["in_direction"] = line_in_direction(p1, p2, inside)
        self.lines.append(ln)
        print(f"  [OK] line '{ln['name']}' IN={ln['in_direction']} area={ln['area_id']}")
        self.current = []

    def _prompt_zone(self, default_name: str) -> dict:
        name = input(f"  zone name (default {default_name}): ").strip() or default_name
        t = input("  type [o]ccupancy / [w]aiting / [r]estricted (default o): ").strip().lower()
        ztype = _ZONE_TYPES.get(t[:1], "occupancy")
        dt = input("  dt_space_id (optional): ").strip() or None
        sl = input("  safe_limit (optional, number): ").strip()
        safe_limit = int(sl) if sl.isdigit() else None
        z = {"name": name, "type": ztype, "safe_limit": safe_limit, "dt_space_id": dt}
        if self.camera_id is not None:
            z = {"camera_id": self.camera_id, **z}
        return z

    def _prompt_line(self, default_name: str) -> dict:
        name = input(f"  line name (default {default_name}): ").strip() or default_name
        area = input("  area_id: ").strip() or "area1"
        dt = input("  dt_space_id (optional): ").strip() or None
        ln = {"name": name, "area_id": area, "dt_space_id": dt}
        if self.camera_id is not None:
            ln = {"camera_id": self.camera_id, **ln}
        return ln

    def _rename(self, kind: str, idx: int):
        if kind == "zone":
            new = self._prompt_zone(default_name=self.zones[idx]["name"])
            new["polygon"] = self.zones[idx]["polygon"]
            self.zones[idx] = new
            print(f"  renamed zone -> '{new['name']}' ({new['type']})")
        else:
            old = self.lines[idx]
            new = self._prompt_line(default_name=old["name"])
            new["points"] = old["points"]
            new["in_direction"] = old["in_direction"]
            self.lines[idx] = new
            print(f"  renamed line -> '{new['name']}'")

    # ---- drawing -----------------------------------------------------------
    def _redraw(self, window):
        import cv2
        import numpy as np

        c = self.frame.copy()
        for z in self.zones:
            col = _TYPE_COLOR[z["type"]]
            pts = np.array(z["polygon"], np.int32)
            cv2.polylines(c, [pts], True, col, 2)
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            cv2.putText(c, f"{z['name']} [{z['type']}]", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            if self.mode == self.EDIT:
                for px, py in z["polygon"]:
                    cv2.circle(c, (px, py), 5, (0, 255, 255), -1)
        for ln in self.lines:
            a, b = tuple(ln["points"][0]), tuple(ln["points"][1])
            cv2.line(c, a, b, _LINE_COLOR, 2)
            mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
            dx, dy = ln["in_direction"]
            cv2.arrowedLine(c, (mx, my), (int(mx + dx * 40), int(my + dy * 40)),
                            (0, 165, 255), 2, tipLength=0.3)
            cv2.putText(c, ln["name"], a, cv2.FONT_HERSHEY_SIMPLEX, 0.5, _LINE_COLOR, 2)
            if self.mode == self.EDIT:
                for px, py in ln["points"]:
                    cv2.circle(c, (px, py), 5, (0, 255, 255), -1)
        # in-progress points
        for i, (px, py) in enumerate(self.current):
            cv2.circle(c, (px, py), 5, (255, 0, 0), -1)
            if i > 0:
                cv2.line(c, tuple(self.current[i - 1]), (px, py), (255, 0, 0), 1)
        kind = "LINE (2 ends + IN side)" if self.line_mode else "ZONE"
        cv2.putText(c, f"Mode: {self.mode}   Draw: {kind}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        h = c.shape[0]
        cv2.putText(c, "L=point  R/f=finish-zone  l=line  u=undo  e=edit  r=remove  m=rename  s=save  q=quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
        cv2.imshow(window, c)

    def _toggle(self, mode: str):
        self.mode = self.DRAW if self.mode == mode else mode
        self.current = []
        print(f"  [MODE] {self.mode}")

    def run(self, out_path: str) -> int:
        import cv2

        window = "draw_zones"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 720)
        cv2.setMouseCallback(window, self.on_mouse)
        while True:
            self._redraw(window)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("u") and self.current:
                self.current.pop()
            elif key == ord("l"):
                self.line_mode = not self.line_mode
                self.current = []
            elif key == ord("f") and self.mode == self.DRAW and not self.line_mode:
                self._finish_zone()
            elif key == ord("e"):
                self._toggle(self.EDIT)
            elif key == ord("r"):
                self._toggle(self.REMOVE)
            elif key == ord("m"):
                self._toggle(self.RENAME)
            elif key == ord("s"):
                payload = {"zones": self.zones, "lines": self.lines}
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                print(f"  saved {len(self.zones)} zone(s), {len(self.lines)} line(s) -> {out_path}")
        cv2.destroyAllWindows()
        return 0


def run(image_path: str | None, url: str | None, out_path: str, camera_id: int | None) -> int:
    import cv2  # lazy

    frame = cv2.imread(image_path) if image_path else _grab_frame(url or "")
    if frame is None:
        raise RuntimeError("no frame to annotate (bad --image path?)")
    return ZoneLineDrawer(frame, camera_id).run(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw zones/lines on a camera frame")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to a still frame")
    src.add_argument("--url", help="RTSP URL to grab one frame from")
    parser.add_argument("--camera-id", type=int, default=None, help="camera id to embed in the JSON")
    parser.add_argument("--out", default="zones.json", help="output JSON path")
    args = parser.parse_args(argv)
    return run(args.image, args.url, args.out, args.camera_id)


if __name__ == "__main__":
    sys.exit(main())