"""Per-camera orchestration pipeline (HLD 6, 13).

A :class:`CameraPipeline` wires one camera's end-to-end processing chain and runs
it on a dedicated worker thread:

    capture -> queue -> detect -> track -> [embed + Re-ID] -> localise -> analyse

It owns:

* an :class:`~app.ingestion.frame_queue.BoundedFrameQueue` decoupling capture
  from inference, fed by one :class:`~app.ingestion.capture.RtspCaptureThread`
  reading the camera's *sub* stream at the role's fps tier;
* the localisation collaborators (:class:`~app.localisation.zones.ZoneEvaluator`,
  :class:`~app.localisation.state_machine.ZonePresenceTracker`,
  :class:`~app.localisation.lines.LineCrossingDetector`);
* the analytics calculators (occupancy, entry/exit), each scoped to the zones/lines
  that matter for its role.

The pipeline is a pure orchestrator: the calculators are side-effect-free and
return events, and the pipeline publishes them. Persistence and the read-model
projection are handled by the bus projectors wired in :mod:`app.api.app`, so the
pipeline only ever ``publish``-es.

Heavy/optional dependencies are imported lazily inside the methods that need them,
so this module imports cleanly on core deps.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from collections.abc import Callable

import numpy as np

from app.analytics.entry_exit import EntryExitCalculator
from app.analytics.occupancy import OccupancyCalculator
from app.config.schema import AppConfig, SnapshotPullConfig
from app.domain.interfaces import Clock, Detector, EmbeddingExtractor, Tracker
from app.domain.models import (
    CameraSpec,
    Detection,
    FramePacket,
    TrackedDetection,
    ZoneType,
)
from app.engine.round_timer import RoundTimer
from app.events.event_bus import InMemoryEventBus
from app.events.events import CameraHealth
from app.inference.reid import ReIDManager
from app.ingestion.capture import CaptureThread, RtspCaptureThread, SnapshotCaptureThread
from app.ingestion.dataset_capture import DatasetWriter
from app.ingestion.frame_queue import BoundedFrameQueue
from app.ingestion.snapshot import SnapshotClient, decode_jpeg
from app.ingestion.stream_url import snapshot_url, sub_stream_url
from app.localisation.lines import Crossing, LineCrossingDetector
from app.localisation.state_machine import PresenceResult, ZonePresenceTracker
from app.localisation.zones import ZoneEvaluator
from app.services.state_store import CameraHealthState, StateStore, get_state_store
from app.utils.logging import get_logger

logger = get_logger(__name__)

# How often (seconds) the worker emits a CameraHealth event and writes health to
# the read model. HLD 13 calls for a ~1 Hz health heartbeat per camera.
_HEALTH_INTERVAL_S = 1.0

# Blocking timeout for a single queue read. Bounded so the worker periodically
# wakes to emit health / observe the stop flag even when no frame arrives.
_QUEUE_GET_TIMEOUT_S = 0.5


class CameraPipeline:
    """Orchestrates the full processing chain for a single camera.

    The pipeline is constructed with already-built perception backends (so the
    engine controls their lifecycle and dependency injection) and builds the
    cheap, per-camera collaborators itself in :meth:`start`. Call :meth:`start`
    to spin up the capture and worker threads and :meth:`stop` to tear them down.
    """

    def __init__(
        self,
        spec: CameraSpec,
        password: str,
        detector: Detector,
        tracker: Tracker | None,
        embedding_extractor: EmbeddingExtractor | None,
        bus: InMemoryEventBus,
        clock: Clock,
        cfg: AppConfig,
        store: StateStore | None = None,
        reid_manager: ReIDManager | None = None,
        capture_factory: Callable[[BoundedFrameQueue], CaptureThread] | None = None,
        round_timer: RoundTimer | None = None,
    ) -> None:
        """Configure the pipeline (does not start any thread).

        Args:
            spec: Immutable camera specification (zones/lines included).
            password: Resolved plaintext RTSP password for this camera.
            detector: Person detector backend (per camera).
            tracker: Multi-object tracker backend (per camera — tracker state is
                camera-local).
            embedding_extractor: Appearance-embedding extractor for Re-ID, or
                ``None`` when Re-ID is disabled.
            bus: Event bus the produced events are published to.
            clock: Pipeline time source (epoch seconds).
            cfg: Application configuration supplying all tuning parameters.
            store: Current-state cache the per-camera health is written into.
                Defaults to the process-wide store (:func:`get_state_store`),
                matching the singleton the API read side queries.
            reid_manager: Shared Re-ID gallery injected by the engine so identity
                spans cameras. When ``None`` but an extractor is present, a
                per-camera manager is built (single-camera Re-ID only).
            capture_factory: Builds the frame producer for a queue. Defaults to an
                :class:`RtspCaptureThread`; tests inject a recorded-clip source.
        """
        self._spec = spec
        self._password = password
        self._detector = detector
        self._tracker = tracker
        self._extractor = embedding_extractor
        self._bus = bus
        self._clock = clock
        self._cfg = cfg
        self._store = store if store is not None else get_state_store()
        self._injected_reid = reid_manager
        self._capture_factory = capture_factory
        # Shared across the snapshot pipelines so one line per round is logged
        # (see app.engine.round_timer). None unless PIPELINE_ROUND_TIMING is set.
        self._round_timer = round_timer
        # Opt-in per-frame diagnostic trace: set PIPELINE_TRACE_CAMERA=<id> to log
        # this camera's detections/tracks/crossings each frame (line-tuning aid).
        self._trace_enabled = os.environ.get("PIPELINE_TRACE_CAMERA") == str(spec.id)

        # Opt-in training-data capture. Snapshot (ISAPI) cameras save the original
        # JPEG bytes in the provider; RTSP cameras (e.g. the entry/exit door) save
        # the decoded frame (encoded) at the worker. One writer serves both.
        dataset_cfg = cfg.dataset_capture
        self._dataset_writer = (
            DatasetWriter(dataset_cfg.dir, dataset_cfg.min_interval_s)
            if dataset_cfg.enabled
            else None
        )
        # Both predicates come from the config object rather than being re-derived
        # here, so the engine (which decides whether to even build a tracker for this
        # camera) and this branch read the same rule.
        snap = cfg.processing.snapshot_pull
        self._is_snapshot_source = snap.is_snapshot_role(spec.fps_role.value)
        # Snapshot occupancy: count detections in-zone directly, skipping the
        # tracker/presence machine (unreliable at the snapshot cadence, only undercounts).
        self._occ_from_detections = snap.counts_from_detections_for(spec.fps_role.value)

        # Fail here rather than per-frame. The worker catches every exception so one
        # bad frame cannot kill the pipeline, which means a None tracker on the tracked
        # path would surface only as an AttributeError logged on every single frame,
        # forever, while the camera silently published nothing.
        if tracker is None and not self._occ_from_detections:
            raise ValueError(
                f"camera {spec.id} ({spec.fps_role.value}) runs the tracked path but was "
                "built without a tracker; engine and pipeline disagree on "
                "SnapshotPullConfig.counts_from_detections_for"
            )

        # Built in start() so a pipeline can be re-created cheaply on restart.
        self._queue: BoundedFrameQueue | None = None
        self._capture: CaptureThread | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Localisation collaborators (built in start()).
        self._zone_eval: ZoneEvaluator | None = None
        self._presence: ZonePresenceTracker | None = None
        self._line_detector: LineCrossingDetector | None = None
        self._reid: ReIDManager | None = None

        # Analytics calculators (built in start()).
        self._occupancy: OccupancyCalculator | None = None
        self._entry_exit: EntryExitCalculator | None = None

        # Health heartbeat throttle (worker thread only — no lock needed).
        self._last_health_emit: float | None = None

    @property
    def camera_id(self) -> int:
        """Database id of the camera this pipeline serves."""
        return self._spec.id

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Build the per-camera chain and launch the capture and worker threads.

        Idempotent: a second call while running is a no-op. After this returns
        the pipeline is consuming frames and publishing events.
        """
        if self._worker is not None and self._worker.is_alive():
            return

        self._stop_event.clear()
        self._build_collaborators()

        self._queue = BoundedFrameQueue(self._cfg.processing.queue_maxsize)
        self._capture = (
            self._capture_factory(self._queue)
            if self._capture_factory is not None
            else self._build_capture(self._queue)
        )

        self._worker = threading.Thread(
            target=self._run,
            name=f"camera-pipeline-{self.camera_id}",
            daemon=True,
        )

        self._capture.start()
        self._worker.start()
        logger.info(
            "camera pipeline started",
            extra={"camera_id": self.camera_id, "event": "pipeline_started"},
        )

    def stop(self) -> None:
        """Stop capture and the worker thread and join them.

        Blocks until both threads have exited. Idempotent.
        """
        self._stop_event.set()
        if self._capture is not None:
            self._capture.stop()
            self._capture.join(timeout=_HEALTH_INTERVAL_S * 5)
        if self._worker is not None:
            self._worker.join(timeout=_HEALTH_INTERVAL_S * 5)
        logger.info(
            "camera pipeline stopped",
            extra={"camera_id": self.camera_id, "event": "pipeline_stopped"},
        )

    def is_alive(self) -> bool:
        """Return whether the worker thread is currently running."""
        return self._worker is not None and self._worker.is_alive()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _build_capture(self, queue: BoundedFrameQueue) -> CaptureThread:
        """Build the frame producer for this camera's role (HLD 6.1).

        Cameras whose ``fps_role`` is listed in
        ``processing.snapshot_pull.roles`` pull HTTP stills (cheap on GPU-less
        hardware — decode cost scales with the pull rate, not the stream fps);
        every other camera decodes the RTSP sub stream. Default config lists no
        roles, so this is RTSP unless snapshot-pull is explicitly enabled.
        """
        snap = self._cfg.processing.snapshot_pull
        if self._spec.fps_role.value in snap.roles:
            return self._build_snapshot_capture(queue, snap)
        return self._build_rtsp_capture(queue)

    def _build_rtsp_capture(self, queue: BoundedFrameQueue) -> RtspCaptureThread:
        """Build the RTSP capture thread for the camera's sub stream."""
        url = sub_stream_url(
            self._spec, self._password, self._cfg.processing.rtsp_transport
        )
        target_fps = self._cfg.processing.fps_tiers.for_role(self._spec.fps_role.value)
        return RtspCaptureThread(
            camera_id=self.camera_id,
            url=url,
            target_fps=target_fps,
            processing_cfg=self._cfg.processing,
            clock=self._clock,
            queue=queue,
            role=self._spec.fps_role,
        )

    def _build_snapshot_capture(
        self, queue: BoundedFrameQueue, snap: SnapshotPullConfig
    ) -> SnapshotCaptureThread:
        """Build the HTTP snapshot-pull capture thread for this camera.

        Wires an :class:`~app.ingestion.snapshot.SnapshotClient` (HTTP fetch) and
        the lazy JPEG decoder into a ``frame_provider`` the thread calls each tick.
        No network happens here — the client connects lazily on first fetch.
        """
        url = snapshot_url(
            self._spec,
            scheme=snap.scheme,
            port=snap.http_port,
            path_template=snap.path_template,
        )
        client = SnapshotClient(
            url,
            self._spec.username,
            self._password,
            auth=snap.auth,
            timeout_s=snap.timeout_s,
            verify_tls=snap.verify_tls,
        )

        # Optional: persist the original ISAPI JPEG bytes for a training dataset.
        writer = self._dataset_writer

        def _provider() -> np.ndarray | None:
            raw = client.fetch_bytes()
            if raw is not None and writer is not None:
                writer.save(self.camera_id, self._spec.ip, raw, self._clock.now())
            return decode_jpeg(raw)

        return SnapshotCaptureThread(
            camera_id=self.camera_id,
            frame_provider=_provider,
            queue=queue,
            clock=self._clock,
            interval_s=snap.interval_s,
            watchdog_stale_seconds=self._cfg.processing.watchdog_stale_seconds,
            role=self._spec.fps_role,
            on_close=client.close,  # release the HTTP client when the loop exits
        )

    def _build_collaborators(self) -> None:
        """Build the localisation collaborators and analytics calculators.

        Each calculator is scoped to the zones/lines relevant to its concern so
        it does no work for cameras that lack a given role's geometry.
        """
        zones = list(self._spec.zones)
        lines = list(self._spec.lines)
        sm = self._cfg.state_machine

        occupancy_zones = [z for z in zones if z.type is ZoneType.OCCUPANCY]

        self._zone_eval = ZoneEvaluator(zones)
        self._presence = ZonePresenceTracker(
            zones, sm.confirm_enter_frames, sm.confirm_leave_frames
        )
        self._line_detector = LineCrossingDetector(lines)

        # Prefer the engine-injected shared manager (cross-camera identity); fall
        # back to a per-camera manager when only an extractor was supplied.
        if self._injected_reid is not None:
            self._reid = self._injected_reid
        elif self._extractor is not None:
            self._reid = ReIDManager(
                similarity_threshold=self._cfg.tracker.reid_similarity_threshold,
                gallery_ttl_seconds=self._cfg.tracker.reid_gallery_ttl_seconds,
                clock=self._clock,
            )

        self._occupancy = OccupancyCalculator(self.camera_id, occupancy_zones, self._clock)
        self._entry_exit = EntryExitCalculator(lines, self._clock)

    # ------------------------------------------------------------------ #
    # Worker loop
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        """Worker thread body: drain the queue and process each frame.

        Runs until :meth:`stop` is signalled. Per-frame failures are logged and
        skipped so a single bad frame never tears the pipeline down. A health
        heartbeat is emitted roughly once per second regardless of frame flow.
        """
        assert self._queue is not None  # set in start() before the worker runs
        while not self._stop_event.is_set():
            # Snapshot occupancy reports "who is here now", so a queued backlog is
            # pure latency: take the freshest frame and drop the rest. The tracked
            # path keeps FIFO — its tracker and line-crossing logic need every
            # frame it can get, in order.
            packet = (
                self._queue.get_latest(timeout=_QUEUE_GET_TIMEOUT_S)
                if self._is_snapshot_source
                else self._queue.get(timeout=_QUEUE_GET_TIMEOUT_S)
            )
            if packet is not None:
                try:
                    self._process_frame(packet)
                except Exception:  # noqa: BLE001 - one bad frame must not kill the worker
                    # Broad catch is deliberate: detector/tracker/calculator
                    # faults on a single frame are recoverable. We log with the
                    # stack and continue; we never fabricate a successful result.
                    logger.exception(
                        "frame processing failed",
                        extra={
                            "camera_id": self.camera_id,
                            "event": "frame_processing_failed",
                        },
                    )
            self._maybe_emit_health()

    def _process_frame(self, packet: FramePacket) -> None:
        """Run the full chain for one frame and publish all produced events."""
        ts = packet.ts
        image = packet.image

        # Training-data capture for RTSP sources (snapshot sources save the raw
        # JPEG in the provider instead, so they're skipped here to avoid re-encode).
        if self._dataset_writer is not None and not self._is_snapshot_source:
            self._dataset_writer.save_image(self.camera_id, self._spec.ip, image, ts)

        detect_started = time.perf_counter()
        detections = self._detector.detect(image)
        detect_seconds = time.perf_counter() - detect_started

        # Snapshot occupancy (3s cadence): count detections in-zone directly and
        # publish — no tracker, no presence debounce. ByteTrack can't reliably link
        # people across snapshots, so tracking only drops real people from the count.
        if self._occ_from_detections:
            counts = self._zone_eval.count_in_zones(detections)  # type: ignore[union-attr]
            for event in self._occupancy.process_counts(counts, ts):  # type: ignore[union-attr]
                self._bus.publish(event)
            if self._round_timer is not None:
                # packet.ts is the capture instant, so this latency spans the
                # queue wait too — a backlog shows up as latency >> detect_seconds.
                self._round_timer.record(
                    camera_id=self.camera_id,
                    capture_ts=ts,
                    finish_ts=self._clock.now(),
                    detect_seconds=detect_seconds,
                    detections=len(detections),
                )
            return

        # Non-None whenever this line is reachable: the engine builds a tracker for
        # exactly the cameras that fall through the _occ_from_detections return above.
        tracked = self._tracker.update(detections, image)  # type: ignore[union-attr]
        tracked = self._attach_identities(image, tracked, ts)

        membership = self._zone_eval.membership(tracked)  # type: ignore[union-attr]
        presence = self._presence.update(membership, ts)  # type: ignore[union-attr]
        crossings = self._line_detector.update(tracked, ts)  # type: ignore[union-attr]

        if self._trace_enabled:
            self._trace_frame(detections, tracked, crossings)

        counts = {zone_id: len(ids) for zone_id, ids in presence.confirmed.items()}

        self._publish_analytics(presence, crossings, ts)

    def _trace_frame(
        self,
        detections: list[Detection],
        tracked: list[TrackedDetection],
        crossings: list[Crossing],
    ) -> None:
        """Log one frame's raw detections, tracks and crossings for line tuning."""
        tracks = [
            f"id{t.track_id}@({int(t.bottom_center.x)},{int(t.bottom_center.y)})"
            for t in tracked
        ]
        cross = [f"id{c.track_id}:{c.direction.value}" for c in crossings]
        logger.info(
            "trace dets=%d tracks=%s crossings=%s",
            len(detections),
            " ".join(tracks) or "-",
            " ".join(cross) or "-",
            extra={"camera_id": self._spec.id, "event": "pipeline_trace"},
        )

    def _attach_identities(
        self,
        image: np.ndarray,
        tracked: list[TrackedDetection],
        ts: float,
    ) -> list[TrackedDetection]:
        """Embed tracked boxes, assign global IDs, and adopt them as the identity.

        Returns ``tracked`` unchanged when Re-ID is disabled or there are no
        tracks. Otherwise it extracts one embedding per box, resolves a stable
        cross-camera ``global_id`` via the Re-ID gallery, and **promotes that
        global_id to the effective ``track_id``** so every downstream consumer
        (zone membership, line crossing, presence state machine, dwell sessions)
        keys on the appearance-stable identity. This is what makes Re-ID actually
        count: it de-duplicates a person across overlapping cameras and preserves
        identity across a leave/re-enter that ByteTrack alone would renumber
        (HLD 5.6).
        """
        if self._extractor is None or self._reid is None or not tracked:
            return tracked

        bboxes = [det.bbox.as_xyxy() for det in tracked]
        embeddings: np.ndarray = self._extractor.extract(image, bboxes)
        embedded = [
            dataclasses.replace(det, embedding=embeddings[idx])
            for idx, det in enumerate(tracked)
        ]
        identified = self._reid.assign_global_ids(self.camera_id, embedded, ts)
        return [
            dataclasses.replace(det, track_id=det.global_id)
            if det.global_id is not None
            else det
            for det in identified
        ]

    def _publish_analytics(
        self,
        presence: PresenceResult,
        crossings: list[Crossing],
        ts: float,
    ) -> None:
        """Run the calculators and publish their events."""
        for event in self._occupancy.process(presence.confirmed, ts):  # type: ignore[union-attr]
            self._bus.publish(event)
        for event in self._entry_exit.process(crossings, ts):  # type: ignore[union-attr]
            self._bus.publish(event)

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    def _maybe_emit_health(self) -> None:
        """Publish a CameraHealth event and update the store roughly every second."""
        now = self._clock.now()
        if self._last_health_emit is not None and (
            now - self._last_health_emit < _HEALTH_INTERVAL_S
        ):
            return

        fps = self._capture.fps_estimate if self._capture is not None else 0.0
        last_frame_ts = (
            self._capture.last_frame_ts if self._capture is not None else None
        )
        last_frame_age = (now - last_frame_ts) if last_frame_ts is not None else 0.0
        queue_depth = self._queue.qsize() if self._queue is not None else 0
        healthy = self._capture.is_healthy() if self._capture is not None else False

        self._bus.publish(
            CameraHealth(
                ts=now,
                camera_id=self.camera_id,
                fps=fps,
                last_frame_age_s=last_frame_age,
                queue_depth=queue_depth,
                healthy=healthy,
            )
        )
        self._store.set_camera_health(
            CameraHealthState(
                camera_id=self.camera_id,
                fps=fps,
                last_frame_age_s=last_frame_age,
                queue_depth=queue_depth,
                healthy=healthy,
                ts=now,
            )
        )
        self._last_health_emit = now
