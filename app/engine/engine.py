"""Engine: builds and supervises the per-camera pipelines (HLD 6, 13).

The :class:`Engine` is the process-level orchestrator. It loads the enabled
cameras (or an explicit subset), builds one :class:`~app.engine.camera_pipeline.CameraPipeline`
per camera sharing the process-wide event bus and state store, starts them, and
runs a supervisor thread that restarts any pipeline whose worker has died, with
exponential backoff.

Dependency resilience (HLD constraint): the heavy perception backends
(Ultralytics / supervision / OSNet runtimes) are imported lazily by the inference
factories. In a dev/test environment those packages — or their model artefacts —
may be absent. :func:`build_engine` therefore wraps the factory build in a guard
for :class:`ImportError`: when a heavy dependency is missing it logs a clear
warning and returns an engine with **zero** pipelines, so the API process still
comes up and serves reads. Any other error (bad config, corrupt model) is *not*
swallowed — it propagates so the failure is visible.

This module imports cleanly on core dependencies: nothing heavy is imported at
module top level, and the factory build that would pull heavy libs in only runs
inside :func:`build_engine`.
"""

from __future__ import annotations

import threading

from app.config.schema import AppConfig
from app.config.settings import get_settings
from app.db.session import session_scope
from app.domain.interfaces import Clock, Detector, EmbeddingExtractor, Tracker
from app.domain.models import CameraSpec
from app.engine.camera_pipeline import CameraPipeline
from app.events.event_bus import InMemoryEventBus, get_event_bus
from app.inference.reid import ReIDManager
from app.services.camera_service import CameraService
from app.services.state_store import StateStore, get_state_store
from app.utils.clock import system_clock
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Supervisor restart backoff bounds (seconds). A crashed pipeline is restarted
# after ``_BACKOFF_BASE_S``, doubling up to ``_BACKOFF_MAX_S`` on repeated crashes.
_BACKOFF_BASE_S = 2.0
_BACKOFF_MAX_S = 60.0

# How often the supervisor scans pipeline liveness.
_SUPERVISOR_POLL_S = 1.0


class _CameraRuntime:
    """A camera's spec/credentials and the backends needed to (re)build its pipeline.

    Holds everything required to construct a fresh :class:`CameraPipeline` so the
    supervisor can rebuild one after a crash without re-reading the database.
    """

    def __init__(
        self,
        spec: CameraSpec,
        password: str,
        detector: Detector,
        tracker: Tracker,
        extractor: EmbeddingExtractor | None,
    ) -> None:
        """Store the immutable inputs for one camera's pipeline."""
        self.spec = spec
        self.password = password
        self.detector = detector
        self.tracker = tracker
        self.extractor = extractor
        # Crash-restart bookkeeping (read/written by the supervisor thread only).
        self.backoff = _BACKOFF_BASE_S
        self.restart_at: float | None = None


class Engine:
    """Owns the lifecycle of all camera pipelines and their supervisor.

    Construct via :func:`build_engine`. The engine shares one event bus and one
    state store across every pipeline (the same singletons the API reads), so
    analytics events and health flow to the same place regardless of how many
    pipelines run.
    """

    def __init__(
        self,
        runtimes: list[_CameraRuntime],
        bus: InMemoryEventBus,
        store: StateStore,
        clock: Clock,
        cfg: AppConfig,
        reid_manager: ReIDManager | None = None,
    ) -> None:
        """Initialise the engine with already-resolved per-camera runtimes.

        Args:
            runtimes: One :class:`_CameraRuntime` per camera to run. May be empty
                (e.g. no enabled cameras, or heavy deps absent).
            bus: Shared event bus pipelines publish to.
            store: Shared current-state cache pipelines write health into.
            clock: Shared pipeline time source.
            cfg: Application configuration.
            reid_manager: One Re-ID gallery shared across every pipeline so a
                person keeps one identity across cameras. ``None`` disables it.
        """
        self._bus = bus
        self._store = store
        self._clock = clock
        self._cfg = cfg
        self._runtimes = runtimes
        self._reid_manager = reid_manager
        self._pipelines: dict[int, CameraPipeline] = {}

        self._stop_event = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def pipeline_count(self) -> int:
        """Number of camera pipelines currently managed by the engine."""
        with self._lock:
            return len(self._pipelines)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start every camera pipeline and the supervisor thread.

        Idempotent: calling :meth:`start` while already running is a no-op. With
        no runtimes the engine starts an (idle) supervisor so the API process
        still functions; the supervisor simply has nothing to watch.
        """
        if self._supervisor is not None and self._supervisor.is_alive():
            return
        self._stop_event.clear()

        with self._lock:
            for runtime in self._runtimes:
                self._start_pipeline(runtime)

        self._supervisor = threading.Thread(
            target=self._supervise, name="engine-supervisor", daemon=True
        )
        self._supervisor.start()
        logger.info(
            "engine started",
            extra={"event": "engine_started"},
        )

    def stop(self) -> None:
        """Stop the supervisor and every pipeline, blocking until all have stopped."""
        self._stop_event.set()
        if self._supervisor is not None:
            self._supervisor.join(timeout=_SUPERVISOR_POLL_S * 5)
        with self._lock:
            for pipeline in self._pipelines.values():
                pipeline.stop()
            self._pipelines.clear()
        logger.info("engine stopped", extra={"event": "engine_stopped"})

    def run(self) -> None:
        """Start the engine and block until :meth:`stop` is called.

        Intended for a foreground process that owns nothing else; the call
        returns only once the engine has been stopped (typically from a signal
        handler on another thread).
        """
        self.start()
        self._stop_event.wait()

    # ------------------------------------------------------------------ #
    # Supervision
    # ------------------------------------------------------------------ #
    def _start_pipeline(self, runtime: _CameraRuntime) -> None:
        """Build and start a fresh pipeline for ``runtime`` (caller holds the lock)."""
        pipeline = CameraPipeline(
            spec=runtime.spec,
            password=runtime.password,
            detector=runtime.detector,
            tracker=runtime.tracker,
            embedding_extractor=runtime.extractor,
            bus=self._bus,
            clock=self._clock,
            cfg=self._cfg,
            store=self._store,
            reid_manager=self._reid_manager,
        )
        pipeline.start()
        self._pipelines[runtime.spec.id] = pipeline

    def _supervise(self) -> None:
        """Restart crashed pipelines with per-camera exponential backoff.

        Scans pipeline liveness every :data:`_SUPERVISOR_POLL_S`. When a
        pipeline's worker has died, the camera is scheduled for a restart after
        its current backoff; the restart fires once the backoff has elapsed, then
        the backoff doubles (capped) so a persistently failing camera does not
        spin. A pipeline that stays alive long enough resets its backoff.
        """
        while not self._stop_event.wait(timeout=_SUPERVISOR_POLL_S):
            now = self._clock.now()
            for runtime in self._runtimes:
                self._check_runtime(runtime, now)

    def _check_runtime(self, runtime: _CameraRuntime, now: float) -> None:
        """Evaluate one camera's pipeline health and restart it if needed."""
        with self._lock:
            if self._stop_event.is_set():
                return
            pipeline = self._pipelines.get(runtime.spec.id)
            if pipeline is not None and pipeline.is_alive():
                # Healthy: reset the backoff so the next crash retries quickly.
                runtime.backoff = _BACKOFF_BASE_S
                runtime.restart_at = None
                return

            if runtime.restart_at is None:
                runtime.restart_at = now + runtime.backoff
                logger.warning(
                    "camera pipeline down; scheduling restart",
                    extra={
                        "camera_id": runtime.spec.id,
                        "event": "pipeline_restart_scheduled",
                    },
                )
                return

            if now < runtime.restart_at:
                return

            self._restart_pipeline(runtime)

    def _restart_pipeline(self, runtime: _CameraRuntime) -> None:
        """Tear down a dead pipeline and start a fresh one (caller holds the lock)."""
        old = self._pipelines.pop(runtime.spec.id, None)
        if old is not None:
            old.stop()
        logger.info(
            "restarting camera pipeline",
            extra={"camera_id": runtime.spec.id, "event": "pipeline_restarting"},
        )
        self._start_pipeline(runtime)
        runtime.restart_at = None
        runtime.backoff = min(runtime.backoff * 2.0, _BACKOFF_MAX_S)


def _load_camera_specs(
    camera_ids: list[int] | None,
) -> list[tuple[CameraSpec, str]]:
    """Load the (spec, password) pairs for the cameras the engine should run.

    Reads inside a single ``session_scope`` so the ORM relationships are loaded
    while the session is open. Cameras whose credential cannot be resolved are
    skipped with a warning rather than aborting the whole engine.

    Args:
        camera_ids: Explicit camera ids to run, or ``None`` to run every enabled
            camera.

    Returns:
        The resolved ``(CameraSpec, password)`` pairs, in id order.
    """
    pairs: list[tuple[CameraSpec, str]] = []
    with session_scope() as session:
        service = CameraService(session)
        target_ids = _resolve_target_ids(service, camera_ids)
        for camera_id in target_ids:
            spec = service.build_camera_spec(camera_id)
            if spec is None:
                logger.warning(
                    "skipping unknown camera",
                    extra={"camera_id": camera_id, "event": "camera_missing"},
                )
                continue
            try:
                password = service.resolve_password(camera_id)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "skipping camera with unresolved credential: %s",
                    exc,
                    extra={"camera_id": camera_id, "event": "credential_unresolved"},
                )
                continue
            pairs.append((spec, password))
    return pairs


def _resolve_target_ids(
    service: CameraService, camera_ids: list[int] | None
) -> list[int]:
    """Resolve the list of camera ids to run for this engine.

    Args:
        service: Camera service bound to the open session.
        camera_ids: Explicit ids, or ``None`` to select every enabled camera.

    Returns:
        Camera ids to build pipelines for.
    """
    if camera_ids is not None:
        return list(camera_ids)
    return [camera.id for camera in service.list() if camera.enabled]


def _build_runtimes(
    pairs: list[tuple[CameraSpec, str]], cfg: AppConfig
) -> list[_CameraRuntime]:
    """Build the perception backends for each camera via the inference factories.

    The heavy backends are imported lazily by the factories; this is the boundary
    where a missing optional dependency surfaces as :class:`ImportError`. That is
    caught by :func:`build_engine` (which then runs zero pipelines), so this
    helper deliberately lets ``ImportError`` propagate.

    Args:
        pairs: Resolved ``(spec, password)`` pairs.
        cfg: Application configuration carrying detector/tracker settings.

    Returns:
        One :class:`_CameraRuntime` per camera, each with its own detector,
        tracker and (optional) embedding extractor.
    """
    # Imported here (not at module top) so importing this module never pulls the
    # inference layer's lazy heavy deps; the factories themselves stay light.
    from app.inference.factory import (
        build_detector,
        build_embedding_extractor,
        build_tracker,
    )

    runtimes: list[_CameraRuntime] = []
    for spec, password in pairs:
        detector = build_detector(cfg.detector)
        tracker = build_tracker(cfg.tracker)
        extractor = build_embedding_extractor(cfg.tracker)
        runtimes.append(
            _CameraRuntime(
                spec=spec,
                password=password,
                detector=detector,
                tracker=tracker,
                extractor=extractor,
            )
        )
    return runtimes


def build_engine(camera_ids: list[int] | None = None) -> Engine:
    """Build the :class:`Engine` for the enabled cameras (or an explicit subset).

    Loads each camera's spec and credential from the database, then builds its
    perception backends via the inference factories. If a heavy perception
    dependency (or its model artefact) is missing — common in dev/test — the
    factory build raises :class:`ImportError`; this is caught, logged, and the
    engine is returned with **zero** pipelines so the API process still serves.
    Non-:class:`ImportError` errors are not swallowed.

    Args:
        camera_ids: Explicit camera ids to run, or ``None`` to run every enabled
            camera.

    Returns:
        A configured :class:`Engine`. Call :meth:`Engine.start` /
        :meth:`Engine.run` to launch it.
    """
    cfg = get_settings()
    bus = get_event_bus()
    store = get_state_store()
    clock = system_clock()

    pairs = _load_camera_specs(camera_ids)

    try:
        runtimes = _build_runtimes(pairs, cfg)
    except ImportError as exc:
        # Heavy perception deps/models are absent (dev/test). Degrade gracefully:
        # the API must still come up. We surface the cause but do not crash.
        logger.warning(
            "perception backend unavailable (%s); starting engine with no pipelines",
            exc,
            extra={"event": "engine_no_backend"},
        )
        runtimes = []

    # One shared Re-ID gallery across all pipelines gives a person a single
    # identity across overlapping cameras (HLD 5.6). Built only when Re-ID is on.
    reid_manager = (
        ReIDManager(
            similarity_threshold=cfg.tracker.reid_similarity_threshold,
            gallery_ttl_seconds=cfg.tracker.reid_gallery_ttl_seconds,
            clock=clock,
        )
        if cfg.tracker.reid_enabled
        else None
    )

    logger.info(
        "engine built",
        extra={"event": "engine_built", "camera_id": len(runtimes)},
    )
    return Engine(
        runtimes=runtimes,
        bus=bus,
        store=store,
        clock=clock,
        cfg=cfg,
        reid_manager=reid_manager,
    )
