# Occupancy-only removal plan

## Why

This backend was built for the five-metric Digital Twin module (occupancy, entry/exit,
safety alerts, waiting times, heat maps): one pipeline, five calculators. The delivered
scope is now **occupancy**, with **entry/exit paused but returning**. Safety alerts,
waiting times, and heat maps are out.

This document records what may be removed, what must not be, and in what order.

## Ground truth (verified against the deployed config, not the example config)

The live deployment loads `config/config.local.yaml` via `CONFIG_PATH`. Under it:

- `processing.entry_exit_enabled: false` → the gate camera (id 27, "Cam 01 MAIN DOOR")
  gets no pipeline at all (`app/engine/engine.py:339-351`).
- `processing.snapshot_pull.roles: ["occupancy"]` with `count_from_detections: true` →
  the 12 live cameras pull HTTP ISAPI stills and count detections in-zone **directly**.

The live path is therefore exactly:

```
SnapshotCaptureThread -> SnapshotClient.fetch_bytes -> decode_jpeg
  -> BoundedFrameQueue.get_latest -> YoloDetector.detect (OpenVINO)
  -> ZoneEvaluator.count_in_zones -> OccupancyCalculator.process_counts
  -> InMemoryEventBus -> StateProjector (live reads) + PersistenceProjector (history)
```

`app/engine/camera_pipeline.py:442` returns **before** the tracker, Re-ID, presence
machine, line crossing, and the entry/exit, safety, waiting and heatmap calculators.
The four unused metrics are already inert at runtime — the cull buys maintainability,
not speed.

### Two facts that invalidate the obvious plan

1. **`count_from_detections` is a config flag, not a structural fact.** Its schema
   default is `False` (`app/config/schema.py:71`). `OccupancyCalculator` has *two*
   entry points — `process()` (tracked) and `process_counts()` (detection-only) — so
   the tracker, `ZonePresenceTracker` and the state machine are **occupancy's default
   path**, not entry/exit-only machinery.

2. **The live path had no test coverage.** No test referenced `count_from_detections`
   or `process_counts`; the whole suite exercised the *tracked* occupancy path that
   production never runs. Phase 0 fixes this and is a prerequisite for everything else.

## Phase 0 — Safety net (prerequisite)

End-to-end test of the `count_from_detections` snapshot path, driving the real
`CameraPipeline` through the `capture_factory` seam with fake perception.

**Done when:** a test asserts the occupancy counts the snapshot path publishes and
reaches `StateStore`, and proves the tracked half of the pipeline is never invoked.

## Phase 1 — Zero-risk deletions (~375 LOC)

Nothing in the occupancy path imports these.

| Unit | LOC |
| --- | --- |
| `app/publish/dt_client.py`, `DigitalTwinConfig` (`schema.py:23-28`), the `dt_publisher` branches of `runtime_wiring.py`, `tests/integration/test_dt_push.py` | ~200 |
| `scripts/training/export_reid_onnx.py` | ~107 |
| `docker-compose.yaml` — the SQL Server + Redis + split api/worker topology | ~68 |

`httpx` **stays**: it reads as a Digital-Twin-push dependency but it is the ISAPI
snapshot client, i.e. the occupancy ingestion path.

Deleting the compose topology also removes the only cause of the time-series
double-write (see "Paired fixes").

**Done when:** `pytest` green, `ruff check` clean, the app boots on
`config/config.local.yaml`, and `/api/v1/occupancy` returns unchanged counts.

## Phase 2 — Heat maps (~500 LOC, 2 blocking untangles)

Highest-value cut: `heatmap_grid` is **4.1 MB of the 8.0 MB SQLite file**, and **76% of
its 31,290 rows are empty (`{}`)**.

Untangle first, or the API will not boot:

1. `app/api/schemas/config.py:21` declares `heatmap: HeatmapConfig` as a **required,
   no-default** field of `RuntimeConfigOut`. Dropping `HeatmapConfig` changes the
   `GET /config` contract — decide whether to drop the field or keep the config type.
2. `app/services/heatmap_service.py:23` imports `safe_join` from `app/utils/snapshot.py`,
   which is shared three ways (`safe_join` → heatmap, `save_snapshot` → alerts,
   `resolve_snapshot` → snapshots router). Lift `safe_join` somewhere neutral.

Then remove: `app/analytics/heatmap.py`, `app/services/heatmap_service.py`,
`app/db/repositories/heatmap_repo.py`, `app/api/routers/heatmap.py`, `HeatmapConfig`,
the `HeatmapFlushed` event, `camera_pipeline.py:602-642`, the `heatmap_grid` table
(+ migration), and `tests/unit/test_heatmap.py` / `test_heatmap_overlay.py`.

Then reclaim the 4.1 MB.

## Phase 3 — Safety/alerts + waiting/dwell (~1,000 LOC, 3 blocking untangles)

Genuinely entangled. Do not start before Phase 0 is green.

1. `app/db/models/__init__.py:4` re-exports `Alert, Snapshot`. Deleting the alert models
   breaks `app/services/occupancy_service.py`'s import chain — **every occupancy endpoint
   dies at boot**.
2. `app/services/state_service.py:34-37` — `GET /state` constructs `EntryExitService`,
   `WaitingService` **and** `AlertService`. Removing waiting/alerts changes the `/state`
   payload, which is occupancy-facing. Decide what `/state` returns.
3. `app/analytics/waiting.py:27` imports `Transition` from
   `app/localisation/state_machine.py`, which is on occupancy's tracked path. Remove
   waiting; **keep the state machine**.

## Phase 4 — Re-ID (~380 LOC, optional)

`app/inference/reid.py` and `camera_pipeline._attach_identities` (`:480-513`).

1. `app/config/settings.py:62-63` assigns `config.tracker.reid_model_path` whenever
   `MODEL_REID_PATH` is set — boot-breaking if the field is gone but the env var is not.
2. `app/inference/factory.py:11` imports `build_embedding_extractor` at module level, and
   the factory is the engine's only detector import surface.

**This frees zero memory.** `torch`/`torchvision` are pulled in by `ultralytics`
(`app/inference/detector.py:119`), not by Re-ID. They only leave if the detector is
rewritten against the OpenVINO runtime directly — a separate, larger job.

## Paired fixes (not removals, but they belong to this work)

- **`app/engine/engine.py:398`** — `build_tracker(cfg.tracker)` runs unconditionally for
  every camera, one line *above* the `runs_tracking` guard that already skips the Re-ID
  extractor. `ByteTrackTracker.__init__` eagerly imports `supervision` (matplotlib, scipy,
  PIL): **+80 MB RSS and +0.71 s startup** to build 12 ByteTrack objects that are never
  called. Move the call inside the guard.
  *Risk:* `runs_tracking` (engine) and `_occ_from_detections`
  (`camera_pipeline.py:158-165`) compute the same condition independently. If they ever
  diverge, `tracker=None` meets the tracked path and crashes. Unify them.

- **`/engine/entry-exit` foot-gun** — `app/api/routers/engine.py:40-43` passes explicit
  `camera_ids=[27]`, and `app/engine/engine.py:327-328` returns explicit ids **without**
  applying the `entry_exit_enabled` guard. One HTTP call resurrects the door pipeline that
  config says is disabled, and resumes writing ~1,440 empty heatmap grids/day.

- **Retention** — `RetentionConfig` (`app/config/schema.py:155-161`) declares six keys and
  **nothing in the codebase reads it**. `README.md:187` and `docs/TRACEABILITY.md:17`
  (NFR-04) both claim a retention policy is enforced. It is not. Either implement the
  reaper or correct the docs.

## Do NOT remove

Entry/exit is **paused, not cancelled**. `app/localisation/lines.py`,
`app/analytics/entry_exit.py`, the `lines` and `crossing_events` tables,
`RtspCaptureThread`, and the tracker all stay.

These look deletable and are load-bearing:

| Unit | Why it stays |
| --- | --- |
| `app/localisation/state_machine.py` (`ZonePresenceTracker`) | Occupancy's **tracked** path (`camera_pipeline.py:448` → `:490`) |
| `OccupancyCalculator.process()` | The tracked fallback if `count_from_detections` is turned off |
| `app/services/state_service.py`, `stats_service.py` | Shared by occupancy's `/state` and `/stats` |
| `app/api/validation_capture.py` | An occupancy feature despite sitting next to the snapshots router |
| `app/engine/round_timer.py` | Live — `PIPELINE_ROUND_TIMING=1` is set in the deployed `.env` |
| `_install_openvino_thread_cap` (`detector.py:46-84`) | The 12-camera CPU oversubscription fix |
| `_CaptureHealthMixin`, `CaptureThread`, `FrameSourceCaptureThread` (`capture.py`) | Share a file with the removable `RtspCaptureThread`; the snapshot path uses them |
| `BoundedFrameQueue.get_latest()` | Selected by the snapshot path |
| `httpx` | The ISAPI snapshot client |
| `ultralytics` (and therefore `torch`) | The occupancy detector itself |
| `app/domain/interfaces.py` + `app/inference/fakes.py` + `FakeClock` | CI installs `.[dev]` only — no torch, no OpenVINO. The fakes are the **only** reason the suite runs |

## Budget

Roughly **1,900 LOC** across Phases 1–3 — not the ~6,770 an early pass suggested, which
assumed entry/exit could go too and that `count_from_detections` was structural.
