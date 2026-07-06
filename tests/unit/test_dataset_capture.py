"""DatasetWriter: saves original snapshot bytes, folders per camera/day, throttles."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.dataset_capture import DatasetWriter


def test_saves_original_bytes_under_camera_day_folder(tmp_path: Path) -> None:
    writer = DatasetWriter(str(tmp_path))
    writer.save(52, "10.1.13.46", b"\xff\xd8-original-jpeg-bytes", ts=1000.5)

    files = list(tmp_path.rglob("*.jpg"))
    assert len(files) == 1
    # byte-for-byte what was passed in (no re-encode)
    assert files[0].read_bytes() == b"\xff\xd8-original-jpeg-bytes"
    # <base>/cam52_10.1.13.46/<date>/<file>.jpg
    assert files[0].parent.parent.name == "cam52_10.1.13.46"


def test_throttles_per_camera(tmp_path: Path) -> None:
    writer = DatasetWriter(str(tmp_path), min_interval_s=10.0)
    writer.save(1, "1.2.3.4", b"a", ts=1000.0)  # saved
    writer.save(1, "1.2.3.4", b"b", ts=1005.0)  # +5s  -> throttled
    writer.save(1, "1.2.3.4", b"c", ts=1012.0)  # +12s -> saved
    writer.save(2, "5.6.7.8", b"d", ts=1005.0)  # other camera -> saved

    assert len(list(tmp_path.rglob("*.jpg"))) == 3


def test_no_throttle_saves_every_frame(tmp_path: Path) -> None:
    writer = DatasetWriter(str(tmp_path))  # min_interval_s=0
    for i in range(5):
        writer.save(7, "9.9.9.9", b"x", ts=1000.0 + i)  # snapshot-cadence spacing
    assert len(list(tmp_path.rglob("*.jpg"))) == 5


def test_save_image_encodes_decoded_frame(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")  # RTSP path re-encodes; needs the inference extra
    import numpy as np

    writer = DatasetWriter(str(tmp_path))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    writer.save_image(1, "10.1.13.21", frame, ts=1000.0)

    files = list(tmp_path.rglob("*.jpg"))
    assert len(files) == 1
    raw = files[0].read_bytes()
    assert raw[:2] == b"\xff\xd8"  # valid JPEG magic
    assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).shape == (16, 16, 3)


def test_write_failure_is_swallowed(tmp_path: Path) -> None:
    # Point the base at a path whose parent is a file -> mkdir raises OSError,
    # which save() must catch (collecting data never breaks the pipeline).
    blocker = tmp_path / "afile"
    blocker.write_text("not a dir")
    writer = DatasetWriter(str(blocker))
    writer.save(1, "1.2.3.4", b"x", ts=1000.0)  # must not raise
