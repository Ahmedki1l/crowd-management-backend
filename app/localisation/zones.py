"""Zone membership evaluation (HLD 6.3).

Given the tracked detections of one frame, decide which tracks occupy which
configured zones. Shapely polygons are built once from the immutable
:class:`~app.domain.models.ZoneSpec` list and reused for every frame, since
zone geometry is static for the lifetime of the evaluator.
"""

from __future__ import annotations

from collections.abc import Sequence

from shapely.geometry.polygon import Polygon

from app.domain.models import TrackedDetection, ZoneSpec, ZoneType
from app.localisation.geometry import point_in_polygon, polygon_overlap_ratio
from app.utils.logging import get_logger

logger = get_logger(__name__)

# A track whose ground point is outside the polygon still counts as a member if
# its bbox overlaps the zone by at least this fraction (HLD 6.3 — partial entry).
MIN_OVERLAP_RATIO = 0.3


class ZoneEvaluator:
    """Assigns tracked detections to zones for a single camera.

    Polygons are precomputed in :meth:`__init__` so per-frame :meth:`membership`
    calls do no polygon construction. A track is a member of a zone when its
    bbox *bottom-centre* (the ground-contact point) lies inside the polygon, or,
    failing that, when its bbox overlaps the polygon by at least
    :data:`MIN_OVERLAP_RATIO`.
    """

    def __init__(self, zones: Sequence[ZoneSpec]) -> None:
        """Precompute the shapely polygon for each zone.

        Args:
            zones: Zone specifications for one camera. May be empty.
        """
        self._zones: tuple[ZoneSpec, ...] = tuple(zones)
        self._polygons: dict[int, Polygon] = {
            zone.id: Polygon([(p.x, p.y) for p in zone.polygon])
            for zone in self._zones
        }

    @property
    def zones_by_type(self) -> dict[ZoneType, tuple[ZoneSpec, ...]]:
        """Group the configured zones by their :class:`ZoneType`."""
        grouped: dict[ZoneType, list[ZoneSpec]] = {}
        for zone in self._zones:
            grouped.setdefault(zone.type, []).append(zone)
        return {zone_type: tuple(specs) for zone_type, specs in grouped.items()}

    def membership(
        self, tracked: Sequence[TrackedDetection]
    ) -> dict[int, set[int]]:
        """Compute raw (un-debounced) zone membership for one frame.

        Args:
            tracked: Tracked detections in the current frame.

        Returns:
            Mapping of ``zone_id -> set of track_ids`` present in that zone. Every
            configured zone appears as a key, with an empty set when no track is
            inside it.
        """
        result: dict[int, set[int]] = {zone.id: set() for zone in self._zones}
        if not self._zones or not tracked:
            return result

        for zone in self._zones:
            polygon = self._polygons[zone.id]
            members = result[zone.id]
            for det in tracked:
                inside = point_in_polygon(det.bbox.bottom_center, polygon)
                overlapping = polygon_overlap_ratio(det.bbox, polygon) >= MIN_OVERLAP_RATIO
                if inside or overlapping:
                    members.add(det.track_id)
        return result
