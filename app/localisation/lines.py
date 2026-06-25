"""Line-crossing detection for entry/exit counting (HLD 6.3).

A track crosses a configured counting line when the segment between its previous
and current ground points intersects the line segment. The direction (IN/OUT) is
resolved from the line's configured ``in_direction`` reference vector. The
detector keeps the last ground point per track so a crossing can only be
reported once it has two observations of a track.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.models import CrossingDirection, LineSpec, Point, TrackedDetection
from app.localisation.geometry import crossing_direction


@dataclass(frozen=True, slots=True)
class Crossing:
    """A single confirmed line crossing by one track."""

    line_id: int
    area_id: str
    direction: CrossingDirection
    track_id: int
    ts: float
    dt_space_id: str | None


class LineCrossingDetector:
    """Detects counting-line crossings for a single camera.

    The last observed ground point (bbox bottom-centre) of each track is kept
    between frames. On a track's first appearance no crossing can be reported —
    only its position is recorded for the next frame.
    """

    def __init__(self, lines: Sequence[LineSpec]) -> None:
        """Initialise the detector.

        Args:
            lines: Counting-line specifications for the camera. May be empty.
        """
        self._lines: tuple[LineSpec, ...] = tuple(lines)
        self._last_point: dict[int, Point] = {}

    def update(
        self, tracked: Sequence[TrackedDetection], ts: float
    ) -> list[Crossing]:
        """Detect crossings for the current frame and update track positions.

        For every track with a recorded previous point, each line is tested for a
        segment intersection; a crossing is emitted when one occurs. New tracks
        produce no crossing on first sight. Tracks that disappear are dropped so
        their stale positions cannot create spurious long-jump crossings later.

        Args:
            tracked: Tracked detections in the current frame.
            ts: Epoch-seconds timestamp of the frame.

        Returns:
            Crossings detected on this frame (possibly empty).
        """
        crossings: list[Crossing] = []
        seen_track_ids: set[int] = set()

        for det in tracked:
            track_id = det.track_id
            seen_track_ids.add(track_id)
            curr = det.bottom_center
            prev = self._last_point.get(track_id)
            if prev is not None and self._lines:
                crossings.extend(self._detect_for_track(track_id, prev, curr, ts))
            self._last_point[track_id] = curr

        self._prune(seen_track_ids)
        return crossings

    def _detect_for_track(
        self, track_id: int, prev: Point, curr: Point, ts: float
    ) -> list[Crossing]:
        """Test one track's movement segment against every configured line."""
        results: list[Crossing] = []
        for line in self._lines:
            line_a, line_b = line.points
            direction = crossing_direction(
                prev, curr, line_a, line_b, line.in_direction
            )
            if direction is None:
                continue
            results.append(
                Crossing(
                    line_id=line.id,
                    area_id=line.area_id,
                    direction=direction,
                    track_id=track_id,
                    ts=ts,
                    dt_space_id=line.dt_space_id,
                )
            )
        return results

    def _prune(self, seen_track_ids: set[int]) -> None:
        """Forget tracks absent from the current frame to avoid stale segments."""
        stale = [tid for tid in self._last_point if tid not in seen_track_ids]
        for track_id in stale:
            del self._last_point[track_id]
