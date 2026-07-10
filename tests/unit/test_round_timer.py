"""RoundTimer closes a round only when every expected camera has reported."""

from __future__ import annotations

import logging

from app.engine.round_timer import RoundTimer


def _record(timer: RoundTimer, cam: int, capture: float, finish: float, detect: float = 0.01) -> None:
    timer.record(
        camera_id=cam, capture_ts=capture, finish_ts=finish, detect_seconds=detect, detections=1
    )


def test_round_closes_only_when_every_camera_reported(caplog) -> None:
    timer = RoundTimer([1, 2, 3])
    with caplog.at_level(logging.INFO):
        _record(timer, 1, 100.0, 100.2)
        _record(timer, 2, 100.1, 100.4)
        assert not [r for r in caplog.records if getattr(r, "event", None) == "round_complete"]

        _record(timer, 3, 100.3, 100.9)

    done = [r for r in caplog.records if getattr(r, "event", None) == "round_complete"]
    assert len(done) == 1
    rec = done[0]
    # earliest capture (100.0) -> latest finish (100.9)
    assert rec.wall_seconds == 0.9
    assert rec.cameras_reported == 3
    assert rec.slowest_camera_id == 3  # latency 0.6 vs 0.2 / 0.3
    assert rec.latency_seconds_max == 0.6


def test_repeat_before_completion_closes_a_partial_round(caplog) -> None:
    """A camera that laps the others must not hold a round open forever."""
    timer = RoundTimer([1, 2])
    with caplog.at_level(logging.INFO):
        _record(timer, 1, 100.0, 100.1)
        _record(timer, 1, 103.0, 103.1)  # camera 1 again; camera 2 never reported

    partial = [r for r in caplog.records if getattr(r, "event", None) == "round_partial"]
    assert len(partial) == 1
    assert partial[0].missing_camera_ids == [2]
    assert partial[0].cameras_reported == 1


def test_latency_exposes_queue_wait_separately_from_detect(caplog) -> None:
    """capture->finish is 5s while detect is 0.02s => the frame sat in a queue."""
    timer = RoundTimer([1])
    with caplog.at_level(logging.INFO):
        timer.record(camera_id=1, capture_ts=10.0, finish_ts=15.0, detect_seconds=0.02, detections=4)

    rec = next(r for r in caplog.records if getattr(r, "event", None) == "round_complete")
    assert rec.latency_seconds_max == 5.0
    assert rec.detect_seconds_total == 0.02
    assert rec.detections_total == 4


def test_empty_expected_set_is_a_noop(caplog) -> None:
    timer = RoundTimer([])
    with caplog.at_level(logging.INFO):
        _record(timer, 1, 1.0, 2.0)
    assert not caplog.records


def test_consecutive_rounds_increment(caplog) -> None:
    timer = RoundTimer([1])
    with caplog.at_level(logging.INFO):
        _record(timer, 1, 1.0, 1.1)
        _record(timer, 1, 4.0, 4.1)
    rounds = [r.round for r in caplog.records if getattr(r, "event", None) == "round_complete"]
    assert rounds == [1, 2]
