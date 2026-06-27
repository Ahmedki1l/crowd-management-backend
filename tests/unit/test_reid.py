"""Unit tests for the pure-numpy ReIDManager gallery (app.inference.reid).

Re-ID is exercised with hand-built numpy embedding vectors so the cosine-
similarity matching, new-id allocation, TTL eviction and "no embedding" path
can each be asserted in isolation, with the deterministic FakeClock driving
eviction.
"""

from __future__ import annotations

import numpy as np

from app.domain.models import BBox, TrackedDetection
from app.inference.reid import ReIDManager
from app.utils.clock import FakeClock


def _tracked(track_id: int, embedding: np.ndarray | None) -> TrackedDetection:
    """Build a TrackedDetection carrying ``embedding`` (geometry is irrelevant)."""
    return TrackedDetection(
        track_id=track_id,
        bbox=BBox(0.0, 0.0, 10.0, 20.0),
        confidence=0.9,
        embedding=embedding,
    )


def test_near_identical_embedding_reuses_same_global_id() -> None:
    clock = FakeClock(start=1000.0)
    manager = ReIDManager(similarity_threshold=0.9, gallery_ttl_seconds=60.0, clock=clock)
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    near = np.array([1.0, 0.02, 0.0, 0.0], dtype=np.float32)  # cosine ~0.9998

    first = manager.assign_global_ids(camera_id=1, tracked=[_tracked(1, base)], ts=1000.0)
    second = manager.assign_global_ids(camera_id=2, tracked=[_tracked(7, near)], ts=1001.0)

    assert second[0].global_id == first[0].global_id


def test_dissimilar_embedding_allocates_new_global_id() -> None:
    clock = FakeClock(start=1000.0)
    manager = ReIDManager(similarity_threshold=0.9, gallery_ttl_seconds=60.0, clock=clock)
    first_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    orthogonal = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # cosine 0.0

    first = manager.assign_global_ids(camera_id=1, tracked=[_tracked(1, first_vec)], ts=1000.0)
    second = manager.assign_global_ids(camera_id=1, tracked=[_tracked(2, orthogonal)], ts=1001.0)

    assert second[0].global_id != first[0].global_id


def test_gallery_entry_past_ttl_is_evicted_before_matching() -> None:
    clock = FakeClock(start=1000.0)
    manager = ReIDManager(similarity_threshold=0.9, gallery_ttl_seconds=30.0, clock=clock)
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    first = manager.assign_global_ids(camera_id=1, tracked=[_tracked(1, vec)], ts=1000.0)
    # Advance the clock well past the TTL so the first entry expires, then feed
    # the same vector: it must not match the (evicted) entry and gets a new id.
    clock.advance(100.0)
    second = manager.assign_global_ids(camera_id=1, tracked=[_tracked(2, vec)], ts=1100.0)

    assert manager.gallery_size == 1
    assert second[0].global_id != first[0].global_id


def test_detection_without_embedding_keeps_global_id_none() -> None:
    clock = FakeClock(start=1000.0)
    manager = ReIDManager(similarity_threshold=0.9, gallery_ttl_seconds=60.0, clock=clock)

    result = manager.assign_global_ids(camera_id=1, tracked=[_tracked(1, None)], ts=1000.0)

    assert result[0].global_id is None
    assert manager.gallery_size == 0


def test_same_appearance_across_cameras_yields_same_global_id() -> None:
    # HLD 5.6: global_id is the consumed identity, so the same person walking
    # from camera 1 into camera 2 must keep one identity, not be double-counted.
    clock = FakeClock(start=1000.0)
    manager = ReIDManager(similarity_threshold=0.9, gallery_ttl_seconds=60.0, clock=clock)
    appearance = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    on_cam1 = manager.assign_global_ids(
        camera_id=1, tracked=[_tracked(1, appearance)], ts=1000.0
    )
    on_cam2 = manager.assign_global_ids(
        camera_id=2, tracked=[_tracked(99, appearance)], ts=1002.0
    )

    assert on_cam2[0].global_id == on_cam1[0].global_id
    # One identity spanning both cameras, not one per camera.
    assert manager.gallery_size == 1


def test_appearance_reappearing_within_ttl_reuses_global_id_but_not_after() -> None:
    # An identity that exits frame and returns within the TTL window must reuse
    # its global_id; once the gallery entry has expired it gets a fresh one.
    clock = FakeClock(start=1000.0)
    manager = ReIDManager(similarity_threshold=0.9, gallery_ttl_seconds=30.0, clock=clock)
    appearance = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    first = manager.assign_global_ids(
        camera_id=1, tracked=[_tracked(1, appearance)], ts=1000.0
    )
    # Absent for a stretch shorter than the TTL: entry still live -> same id.
    clock.advance(20.0)
    within_ttl = manager.assign_global_ids(
        camera_id=1, tracked=[_tracked(2, appearance)], ts=1020.0
    )

    assert within_ttl[0].global_id == first[0].global_id

    # Absent again, now long enough that last_seen (1020) falls past the TTL
    # cutoff: the entry is evicted before matching and a new id is allocated.
    clock.advance(40.0)
    after_ttl = manager.assign_global_ids(
        camera_id=1, tracked=[_tracked(3, appearance)], ts=1060.0
    )

    assert after_ttl[0].global_id != first[0].global_id
    assert manager.gallery_size == 1


def test_two_near_identical_detections_in_one_frame_get_distinct_ids() -> None:
    # CRITICAL (occupancy undercount guard): two tracks with near-identical
    # embeddings in a SINGLE assign call must not collapse onto one identity.
    # Per-frame mutual exclusion forbids the second from claiming the first's
    # gallery entry, so it must fall through to a new global_id.
    clock = FakeClock(start=1000.0)
    manager = ReIDManager(similarity_threshold=0.9, gallery_ttl_seconds=60.0, clock=clock)
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    near = np.array([1.0, 0.02, 0.0, 0.0], dtype=np.float32)  # cosine ~0.9998

    result = manager.assign_global_ids(
        camera_id=1, tracked=[_tracked(1, base), _tracked(2, near)], ts=1000.0
    )

    assert result[0].global_id != result[1].global_id
    # Two people in the frame -> two distinct identities held.
    assert manager.gallery_size == 2


def test_two_distinct_appearances_in_one_frame_get_distinct_ids() -> None:
    # Two clearly different people in the same frame must each get their own id.
    clock = FakeClock(start=1000.0)
    manager = ReIDManager(similarity_threshold=0.9, gallery_ttl_seconds=60.0, clock=clock)
    person_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    person_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # orthogonal

    result = manager.assign_global_ids(
        camera_id=1, tracked=[_tracked(1, person_a), _tracked(2, person_b)], ts=1000.0
    )

    assert result[0].global_id != result[1].global_id
    assert manager.gallery_size == 2


def test_matched_prototype_blends_toward_new_crop_not_overwrite() -> None:
    # Regression: a matched gallery entry must EMA-blend toward the new crop, NOT
    # be overwritten by it. Overwriting let a few above-threshold-but-wrong frames
    # walk the prototype onto a different person, slowly merging two identities.
    clock = FakeClock(start=1000.0)
    manager = ReIDManager(
        similarity_threshold=0.9,
        gallery_ttl_seconds=60.0,
        clock=clock,
        embedding_update_rate=0.1,
    )
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # A matching but shifted observation (cosine ~0.91, just above threshold).
    shifted = np.array([1.0, 0.45, 0.0, 0.0], dtype=np.float32)

    first = manager.assign_global_ids(camera_id=1, tracked=[_tracked(1, base)], ts=1000.0)
    second = manager.assign_global_ids(camera_id=1, tracked=[_tracked(2, shifted)], ts=1001.0)

    assert second[0].global_id == first[0].global_id  # same identity reused
    prototype = manager._gallery[0].embedding
    # The prototype stayed anchored to the original identity (a tiny nudge only),
    # instead of jumping to the shifted crop (which overwrite would have done:
    # prototype[0]~0.91, prototype[1]~0.41).
    assert prototype[0] > 0.97
    assert prototype[1] < 0.10
