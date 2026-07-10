"""Typed configuration schema (pydantic).

Mirrors ``config/config.example.yaml``. This is the single source of truth for
runtime tuning; every layer reads its parameters from these models rather than
re-parsing YAML. Cameras/zones/lines are NOT here — they live in the database.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///./camera_analytics.db"
    echo: bool = False
    pool_size: int = 10


class CacheConfig(BaseModel):
    url: str | None = None


class DigitalTwinConfig(BaseModel):
    push_url: str | None = None
    push_enabled: bool = False
    auth_header: str | None = None
    timeout_seconds: float = 5.0
    max_retries: int = 3


class ApiConfig(BaseModel):
    prefix: str = "/api/v1"
    auth_secret: str = "change-me"
    auth_scheme: str = "jwt"  # jwt | api_key
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class FpsTiers(BaseModel):
    entry_exit: float = 8
    waiting: float = 6
    occupancy: float = 2
    heatmap: float = 1

    def for_role(self, role: str) -> float:
        return float(getattr(self, role, self.occupancy))


class SnapshotPullConfig(BaseModel):
    """Still-image (HTTP snapshot) capture for CPU-constrained deployments.

    Cameras whose :attr:`~app.domain.models.CameraSpec.fps_role` is listed in
    ``roles`` pull a single JPEG every ``interval_s`` over HTTP instead of holding
    a continuously-decoding RTSP connection. Decode cost then scales with the pull
    rate, not the stream's native frame rate — the key lever on a GPU-less box.
    Empty ``roles`` (the default) keeps every camera on RTSP, so this is fully
    backward-compatible.
    """

    roles: list[str] = Field(default_factory=list)
    interval_s: float = 2.0
    http_port: int = 80
    scheme: str = "http"  # http | https
    # ``{channel}`` is substituted with the camera's sub-stream channel.
    path_template: str = "/ISAPI/Streaming/channels/{channel}/picture"
    auth: str = "digest"  # digest | basic
    timeout_s: float = 5.0
    verify_tls: bool = True  # only consulted when scheme is https
    # Count detections in-zone directly (no tracker/presence debounce). At the
    # snapshot cadence ByteTrack can't reliably link people across frames, so
    # tracking only drops real people from the count. Recommended for snapshot pull.
    count_from_detections: bool = False


class ProcessingConfig(BaseModel):
    fps_tiers: FpsTiers = Field(default_factory=FpsTiers)
    default_tier: str = "occupancy"
    # When non-empty, ONLY cameras whose IP is listed get a pipeline (feed pull);
    # every other camera is skipped. Empty (default) runs every enabled camera.
    camera_allowlist_ips: list[str] = Field(default_factory=list)
    # Master switch for the entry/exit (line-crossing) gate. When false, cameras
    # carrying the entry_exit role get no pipeline at all — no RTSP decode, no
    # tracker, no Re-ID — freeing the heavy tracked path on CPU-only boxes while
    # leaving occupancy untouched. Applies only to the "run every enabled camera"
    # path; an explicit --worker --camera <id> still runs the gate on demand.
    entry_exit_enabled: bool = True
    stream_channel_sub: int = 102
    stream_channel_main: int = 101
    queue_maxsize: int = 4
    rtsp_transport: str = "tcp"
    reconnect_backoff_base_s: float = 1.0
    reconnect_backoff_max_s: float = 30.0
    watchdog_stale_seconds: float = 15.0
    snapshot_pull: SnapshotPullConfig = Field(default_factory=SnapshotPullConfig)


class DetectorConfig(BaseModel):
    model_path: str = "models/detector_openvino_model"  # OpenVINO dir must end in _openvino_model
    runtime: str = "openvino"  # openvino | tensorrt
    person_class: int = 0
    confidence: float = 0.30
    iou: float = 0.45
    imgsz: int = 640
    batch_size: int = 4


class TrackerConfig(BaseModel):
    type: str = "bytetrack"
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 8
    reid_enabled: bool = True
    reid_model_path: str = "models/reid"
    reid_runtime: str = "onnx"  # onnx | torch
    reid_similarity_threshold: float = 0.6
    reid_gallery_ttl_seconds: float = 120


class StateMachineConfig(BaseModel):
    confirm_enter_frames: int = 5
    confirm_leave_frames: int = 8
    alert_debounce_frames: int = 5
    alert_cooldown_seconds: float = 30


class HeatmapConfig(BaseModel):
    grid_cols: int = 64
    grid_rows: int = 36
    flush_interval_seconds: float = 60


class SnapshotsConfig(BaseModel):
    dir: str = "data/snapshots"


class DatasetCaptureConfig(BaseModel):
    """Persist every fetched ISAPI snapshot to disk for building a training set.

    Off by default. When ``enabled``, each successful snapshot-pull fetch writes
    its *original* JPEG bytes (no re-encode) under ``dir/<camera>/<date>/``.
    ``min_interval_s`` throttles per camera (0 = save every fetched frame).
    """

    enabled: bool = False
    dir: str = "training_data"
    min_interval_s: float = 0.0


class RetentionConfig(BaseModel):
    occupancy_days: int = 90
    crossing_days: int = 90
    dwell_days: int = 90
    heatmap_days: int = 180
    alert_days: int = 365
    snapshot_days: int = 30


class AppConfig(BaseModel):
    """Root configuration object assembled by ``app.config.settings``."""

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    digital_twin: DigitalTwinConfig = Field(default_factory=DigitalTwinConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    state_machine: StateMachineConfig = Field(default_factory=StateMachineConfig)
    heatmap: HeatmapConfig = Field(default_factory=HeatmapConfig)
    snapshots: SnapshotsConfig = Field(default_factory=SnapshotsConfig)
    dataset_capture: DatasetCaptureConfig = Field(default_factory=DatasetCaptureConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
