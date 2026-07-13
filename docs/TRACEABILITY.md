# Requirements Traceability — BRD/HLD → Component → Endpoint → Test

Every functional requirement traces to the code that implements it, the API it is
served over, and the automated test(s) that verify it (vnv-guard). All tests run
on the lightweight core (no cameras/GPU) — `pytest` → **155 passed**.

| Requirement | Implementing components | Endpoint(s) | Verifying test(s) |
|---|---|---|---|
| **FR-OCC-01/02** occupancy per zone + DT display | `inference/{detector,tracker}`, `localisation/{zones,state_machine}`, `analytics/occupancy` | `GET /occupancy`, `/state`, `/stream`, `/history/occupancy` | `test_occupancy`, `test_state_machine`, `test_geometry`, `test_pipeline_recorded`, `test_api_metrics` |
| **FR-EE-01/02/03** line IN/OUT/net per area | `localisation/lines`, `analytics/entry_exit` | `GET /entry-exit`, `/state`, `/stream`, `/history/entry-exit` | `test_lines`, `test_entry_exit`, `test_pipeline_recorded`, `test_api_metrics` |
| **NFR-01** timeliness (few seconds) | `events/event_bus`, `publish/sse`, `services/projectors` | `/stream` | `test_event_bus`, `test_sse_stream`, `test_projectors` |
| **NFR-02** RTSP inputs only | `ingestion/{capture,stream_url,recorded}` | `POST /cameras/{id}/test` | `test_camera_service`, `test_frame_queue` |
| **NFR-03** accuracy (UAT target) | `inference` tuning via `config` (`detector.confidence`, tracker, Re-ID) | `GET/PUT /config` | tuned at UAT; geometry/tracker logic in `test_geometry`, `test_reid` |
| **NFR-04** retention | **NOT IMPLEMENTED** — `config.retention` is declared but read by no code; no pruning job exists | `/history/*` | none |
| **NFR-05** capacity / scale-out | worker-per-camera (`engine`), tiered fps, INT8/FP16 detector, **Redis-backed bus + state** (`redis_bus`, `redis_state_store`) for multi-process | `GET /metrics`, `/cameras/{id}/health` | `test_state_store`, `test_redis_bus`, `test_redis_state_store`, `test_serialization`, `test_engine` |
| **Cross-camera identity** (HLD 5.6) | shared OSNet `ReIDManager`; pipeline promotes `global_id` → identity. **Disabled** (`tracker.reid_enabled: false`) and unused on the snapshot-occupancy path | feeds entry/exit when the gate runs | `test_reid` (unit: cross-camera, TTL, mutual-exclusion). **No pipeline-level identity test** — it observed identity via `DwellClosed`, which was removed with waiting |
| **Security** (HLD 14) | AES-256-GCM credential cipher (`services/credentials`), write-only password, JWT auth (`utils/security`, `api/deps`) | all routes (auth); `/cameras` (password never returned) | `test_credentials`, `test_security_jwt`, `test_api_cameras` (401 + no-leak) |

## Definition of Done — status

- [x] Every FR-* traces to a component + endpoint + passing test (table above).
- [x] API contract — 39 endpoints, auth on all routes (safety/waiting/heat-map/snapshot endpoints removed with their metrics).
- [x] Camera password write-only — never returned by any GET (verified).
- [x] Pipeline survives camera disconnects (decoupled capture + reconnect/backoff + watchdog).
- [x] E2E critical path proven: synthetic frames → occupancy/crossing → read model (`test_pipeline_recorded`), and the **live snapshot-pull occupancy path** (`test_snapshot_occupancy`).
- [x] Tests green without cameras/GPU/models (FakeDetector/FakeTracker, lazy heavy imports).
- [x] Alembic baseline migration creates the full HLD §9 schema.
- [x] Re-ID is **functional** (global_id consumed as identity), not gold-plating — closes the validation-board CAIO no-go.
- [x] Real `CameraPipeline` integration-tested (alert+snapshot persist, health, Re-ID identity, snapshot-pull occupancy).
- [x] Scale-out (NFR-05) implemented: Redis bus/state, config-selected, in-process path unchanged.
- [x] All eight guard skills run (design, clean-code, test, docs, testing, e2e, vnv, validation-board).

**Deferred (documented, not gold-plated):** a cross-camera *area-occupancy* dedup
count is a metric beyond the BRD's per-zone FR-OCC; the identity foundation
(consistent `global_id`) is in place as its future consumer. HLD R5 offers
"zones one-camera-each" as the alternative.
