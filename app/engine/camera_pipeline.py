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
* the analytics calculators (occupancy, entry/exit, safety, waiting, heatmap),
  each scoped to the zones/lines that matter for its role.

The pipeline is a pure orchestrator: the calculators are side-effect-free and
return events, and the pipeline publishes them. The two flows that *do* have side
effects — persisting an alert (with an evidence snapshot) and flushing a heatmap
grid — are owned here, at the point the event is produced, exactly as the
analytics layer's contract requires. Everything else (occupancy/crossing/dwell
persistence and the read-model projection) is handled by the bus projectors wired
in :mod:`app.api.app`, so the pipeline only ever ``publish``-es those.

Heavy/optional dependencies (``cv2`` via the snapshot helper) are imported lazily
inside the methods that need them, so this module imports cleanly on core deps.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable

import numpy as np

from app.analytics.entry_exit import EntryExitCalculator
from app.analytics.heatmap import GridSnapshot, HeatmapAccumulator
from app.analytics.occupancy import OccupancyCalculator
from app.analytics.safety import SafetyCalculator
from app.analytics.waiting import WaitingCalculator
from app.config.schema import AppConfig
from app.db.repositories.heatmap_repo import HeatmapRepository
from app.db.session import session_scope
from app.domain.interfaces import Clock, Detector, EmbeddingExtractor, Tracker
from app.domain.models import (
    CameraSpec,
    FramePacket,
    TrackedDetection,
    ZoneType,
)
from app.events.event_bus import InMemoryEventBus
from app.events.events import AlertRaised, CameraHealth, HeatmapFlushed
from app.inference.reid import ReIDManager
from app.ingestion.capture import CaptureThread, RtspCaptureThread
from app.ingestion.frame_queue import BoundedFrameQueue
from app.ingestion.stream_url import sub_stream_url
from app.localisation.lines import Crossing, LineCrossingDetector
from app.localisation.state_machine import PresenceResult, ZonePresenceTracker
from app.localisation.zones import ZoneEvaluator
from app.services.alert_service import AlertService
from app.services.state_store import CameraHealthState, StateStore, get_state_store
from app.utils.logging import get_logger
from app.utils.timeutil import to_datetime

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
        tracker: Tracker,
        embedding_extractor: EmbeddingExtractor | None,
        bus: InMemoryEventBus,
        clock: Clock,
        cfg: AppConfig,
        store: StateStore | None = None,
        reid_manager: ReIDManager | None = None,
        capture_factory: Callable[[BoundedFrameQueue], CaptureThread] | None = None,
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
        self._safety: SafetyCalculator | None = None
        self._waiting: WaitingCalculator | None = None
        self._heatmap: HeatmapAccumulator | None = None

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
    def _build_capture(self, queue: BoundedFrameQueue) -> RtspCaptureThread:
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

    def _build_collaborators(self) -> None:
        """Build the localisation collaborators and analytics calculators.

        Each calculator is scoped to the zones/lines relevant to its concern so
        it does no work for cameras that lack a given role's geometry.
        """
        zones = list(self._spec.zones)
        lines = list(self._spec.lines)
        sm = self._cfg.state_machine

        occupancy_zones = [z for z in zones if z.type is ZoneType.OCCUPANCY]
        waiting_zones = [z for z in zones if z.type is ZoneType.WAITING]

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
        self._safety = SafetyCalculator(
            self.camera_id,
            zones,
            self._clock,
            sm.alert_debounce_frames,
            sm.alert_cooldown_seconds,
        )
        self._waiting = WaitingCalculator(waiting_zones, self._clock)
        self._heatmap = HeatmapAccumulator(
            self.camera_id,
            self._cfg.heatmap.grid_cols,
            self._cfg.heatmap.grid_rows,
            self._cfg.heatmap.flush_interval_seconds,
            self._clock,
        )

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
            packet = self._queue.get(timeout=_QUEUE_GET_TIMEOUT_S)
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

        detections = self._detector.detect(image)
        tracked = self._tracker.update(detections, image)
        tracked = self._attach_identities(image, tracked, ts)

        membership = self._zone_eval.membership(tracked)  # type: ignore[union-attr]
        presence = self._presence.update(membership, ts)  # type: ignore[union-attr]
        crossings = self._line_detector.update(tracked, ts)  # type: ignore[union-attr]

        counts = {zone_id: len(ids) for zone_id, ids in presence.confirmed.items()}

        self._publish_analytics(presence, crossings, ts)
        self._handle_alerts(presence.confirmed, counts, ts, image)
        self._handle_heatmap(tracked, image, ts)

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
        """Run the non-alert calculators and publish their events."""
        for event in self._occupancy.process(presence.confirmed, ts):  # type: ignore[union-attr]
            self._bus.publish(event)
        for event in self._entry_exit.process(crossings, ts):  # type: ignore[union-attr]
            self._bus.publish(event)
        for event in self._waiting.process(presence.transitions, ts):  # type: ignore[union-attr]
            self._bus.publish(event)

    def _handle_alerts(
        self,
        confirmed: dict[int, set[int]],
        counts: dict[int, int],
        ts: float,
        image: np.ndarray,
    ) -> None:
        """Evaluate safety alerts, persist each with a snapshot, then publish.

        Each raised alert is enriched with a captured evidence snapshot and the
        persisted alert id before it goes on the wire, so subscribers receive a
        complete, addressable alert (HLD 7 / 13).
        """
        alerts = self._safety.process(confirmed, counts, ts)  # type: ignore[union-attr]
        for alert in alerts:
            self._bus.publish(self._persist_alert(alert, image, ts))

    def _persist_alert(
        self, alert: AlertRaised, image: np.ndarray, ts: float
    ) -> AlertRaised:
        """Save a snapshot, persist the alert, and return an enriched event.

        On any persistence/encoding failure the original alert is still published
        (without an id/snapshot) so the alert is never silently lost; the failure
        is logged with its stack.
        """
        snapshot_url = self._capture_snapshot(image, ts)
        try:
            with session_scope() as session:
                row = AlertService(session).raise_alert(alert, snapshot_url)
                alert_id = row.id
            return dataclasses.replace(
                alert, alert_id=alert_id, snapshot_url=snapshot_url
            )
        except Exception:  # noqa: BLE001 - alert delivery must survive DB faults
            # A storage failure must not drop the alert: log with the stack and
            # publish the un-persisted alert so the condition still reaches
            # subscribers. We do not invent an id.
            logger.exception(
                "failed to persist alert",
                extra={
                    "camera_id": self.camera_id,
                    "zone_id": alert.zone_id,
                    "event": "alert_persist_failed",
                },
            )
            if snapshot_url is not None:
                return dataclasses.replace(alert, snapshot_url=snapshot_url)
            return alert

    def _capture_snapshot(self, image: np.ndarray, ts: float) -> str | None:
        """Write an evidence JPEG and return its relative path, or ``None``.

        ``save_snapshot`` imports ``cv2`` lazily. A capture failure is logged and
        swallowed (returning ``None``) so a missing snapshot never blocks the
        alert itself from being raised.
        """
        from app.utils.snapshot import save_snapshot

        try:
            return save_snapshot(
                image, self._cfg.snapshots.dir, self.camera_id, ts
            )
        except Exception:  # noqa: BLE001 - snapshot is best-effort evidence
            logger.exception(
                "failed to save alert snapshot",
                extra={
                    "camera_id": self.camera_id,
                    "event": "snapshot_save_failed",
                },
            )
            return None

    def _handle_heatmap(
        self, tracked: list[TrackedDetection], image: np.ndarray, ts: float
    ) -> None:
        """Accumulate ground points and persist/publish a grid on flush."""
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return
        self._heatmap.accumulate(tracked, width, height, ts)  # type: ignore[union-attr]
        snapshot = self._heatmap.maybe_flush(ts)  # type: ignore[union-attr]
        if snapshot is None:
            return
        self._persist_heatmap(snapshot)
        self._bus.publish(
            HeatmapFlushed(
                ts=ts, camera_id=self.camera_id, ts_bucket=snapshot.ts_bucket
            )
        )

    def _persist_heatmap(self, snapshot: GridSnapshot) -> None:
        """Persist a flushed heatmap grid snapshot.

        Failures are logged with their stack and swallowed so a DB hiccup cannot
        stall the pipeline; the next interval's grid will be persisted normally.
        """
        try:
            with session_scope() as session:
                HeatmapRepository(session).add_grid(
                    camera_id=snapshot.camera_id,
                    ts_bucket=to_datetime(snapshot.ts_bucket),
                    cols=snapshot.cols,
                    rows=snapshot.rows,
                    grid=snapshot.cells,
                )
        except Exception:  # noqa: BLE001 - heatmap persistence is non-critical
            logger.exception(
                "failed to persist heatmap grid",
                extra={
                    "camera_id": self.camera_id,
                    "event": "heatmap_persist_failed",
                },
            )

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
