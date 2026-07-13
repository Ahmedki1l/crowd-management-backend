# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Backend for the **Digital Twin** Camera-Based Analytics module. It ingests live RTSP
video, detects/tracks people, and derives **occupancy** and **entry/exit** — exposed over
REST + SSE. One pipeline powers both; only the final calculator differs. It was originally
built for five metrics; safety alerts, waiting times and heat maps were removed when the
scope narrowed (see `docs/OCCUPANCY_ONLY_REMOVAL_PLAN.md`). Entry/exit is retained but
currently paused via `processing.entry_exit_enabled`. See `README.md` for the product framing and the BRD/HLD
mapping; this file captures the architecture and conventions that span multiple files.

## Commands

```bash
# Setup (lightweight core + test deps — no GPU/models/cameras needed)
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Generate the REQUIRED camera-credential key into .env (CAMERA_CREDENTIALS_KEY):
python -c "from app.services.credentials import CredentialCipher; print(CredentialCipher.generate_key())"

# Lint (must pass in CI) and tests
ruff check app scripts tests
ruff check --fix app scripts tests          # autofix
pytest                                       # full suite, green without cameras/GPU/models
pytest --timeout=30                          # guard threaded pipeline/bus tests against hangs
pytest tests/unit                            # just the pure-logic tests
pytest tests/integration/test_engine.py                       # one file
pytest tests/unit/test_occupancy.py::test_changed_count_emits_a_new_update   # one test
pytest -k occupancy                                           # by name pattern
mypy app                                     # type check (config in pyproject)

# Run modes (HLD 12.5) — same image, mode chosen by flag
# NOTE: --api/--workers default to --port 8008 (matches Dockerfile/compose); bare uvicorn binds 8000
python -m app.main --api                     # API + engine (all cameras), small deployments
python -m app.main --workers                 # pipeline workers only (no API)
python -m app.main --worker --camera 1       # one camera (debug / scale-out)
uvicorn app.api.app:app --reload             # API service only; docs at /docs

# Seed + DB
python scripts/seed_demo.py                  # register a demo camera + zones + line
alembic upgrade head                         # apply migrations (production; SQLite auto-creates in dev)

# Real pipeline (heavy extras) + model export
pip install -e ".[inference,reid]"
python scripts/export_model.py --weights yolo11n.pt --format openvino --int8
```

Docker: `docker compose up --build` runs a single `api` service (API + engine in one
process, SQLite). The split api/worker + Redis + SQL Server topology was removed: with
a shared Redis bus, `RuntimeWiring` subscribes a `PersistenceProjector` in *every*
process, and that projector appends — so each event was persisted once per process,
while `--api` also ran the engine for every camera. Every time-series row was written
roughly four times. If you scale workers out again, exactly one process may wire the
persistence projector.

## Architecture

### Six layers, one vocabulary
The pipeline flows `ingestion → inference → localisation → analytics → state/storage →
api/publish` (packages `app/ingestion`, `app/inference`, `app/localisation`,
`app/analytics`, `app/services`+`app/db`+`app/events`, `app/api`+`app/publish`).

**`app/domain/models.py` is the single vocabulary** every layer exchanges (`Detection`,
`TrackedDetection`, `FramePacket`, `CameraSpec`/`ZoneSpec`/`LineSpec`, enums, `BBox`). ORM
rows (`app/db/models`) and API payloads (`app/api/schemas`) are mapped *to/from* these
domain objects by repositories/services — never used directly in the pipeline. When adding a
concept, model it here once rather than re-inventing it per package.

**`app/domain/interfaces.py` is the contract spine.** `Detector`, `Tracker`,
`EmbeddingExtractor`, `FrameSource`, `Clock` are `Protocol`s. The engine/analytics depend on
these behaviours, not on heavy backends. Real backends (Ultralytics/OpenVINO/TensorRT,
ByteTrack, OSNet) import their heavy deps **lazily inside the concrete class**; fakes implement
the same Protocols with no heavy deps — `FakeDetector`/`FakeTracker`/`FakeEmbeddingExtractor` in
`app/inference/fakes.py`, the `FrameSource` fake `RecordedClipSource` in `app/ingestion/recorded.py`,
the `Clock` fake `FakeClock` in `app/utils/clock.py`. This is what lets the full test suite + API +
analytics run on the lightweight core.

### Event-driven read model (the key pattern)
Calculators in `app/analytics` are **pure**: they hold per-camera in-memory state, consume
localisation outputs, and **return events** (`app/events/events.py`). They never touch the DB
or the bus. The single internal event bus (`app/events/event_bus.py`) fans every event out to
consumers wired in **one place** — `app/services/runtime_wiring.py` (`RuntimeWiring`), used by
*both* the FastAPI lifespan (`app/api/app.py`) and the worker entry points (`app/main.py`) so
the wiring never drifts:

- **`StateProjector`** — updates the in-memory current-state cache (`StateStore`)
  **synchronously on the publishing thread**. Must stay fast (no I/O); it is the write side
  of the read model the API queries.
- **`PersistenceProjector`** — `handle()` only enqueues; a background thread drains and writes
  time-series via `session_scope()` + repositories. Keeps slow DB writes off the hot path.
  Occupancy samples are throttled per-zone (live updates can be per-frame).
**Exception:** alerts (with evidence snapshots) are persisted by the engine
**at the point they're produced** (`app/engine/camera_pipeline.py`), not by the projectors —
because they need the live frame image. The projectors deliberately ignore those event types.

### Per-camera pipeline isolation
`app/engine/engine.py` (multi-camera lifecycle) builds one `CameraPipeline` per camera; each
runs on its own worker thread: dedicated `RtspCaptureThread` → bounded **drop-oldest**
`BoundedFrameQueue` → detect → track → `_attach_identities` (Re-ID) → localise → analyse →
publish. One bad frame is caught/logged/skipped so it never kills the worker; one camera's
failure never affects another. A ~1 Hz `CameraHealth` heartbeat is emitted per camera. The
engine supervisor restarts a pipeline **only on worker-thread death** (`is_alive()` watches the
worker, not the capture thread); the `RtspCaptureThread` self-heals — it reconnects forever with
backoff and never exits on stream faults — so a dead/stalled RTSP source does **not** trigger a
restart, surfacing only via `is_healthy()`/`CameraHealth`.

**Re-ID detail:** when enabled, the engine builds **one shared `ReIDManager` gallery** across
all pipelines. `_attach_identities` promotes the appearance-stable `global_id` to the working
`track_id`, so every downstream consumer (zones, lines, presence) keys on identity that
survives leave/re-enter. **It does not de-duplicate across overlapping cameras**: zones are
per-camera rows and `by_space` sums their counts, so one `global_id` in two cameras still
counts twice. Cross-camera de-dup would need distinct-identity counting at the space level.
A pipeline built with
an `embedding_extractor` but `reid_manager=None` silently builds its **own per-camera** gallery
(single-camera Re-ID only) — cross-camera de-dup depends on `build_engine` injecting the shared
one, so wiring pipelines outside it loses cross-camera identity with no error.

### Pluggable backends behind interfaces
Setting `cache.url` / `REDIS_URL` transparently switches the event bus and state store from
in-process (`InMemoryEventBus`, in-memory `StateStore`) to Redis-backed (`RedisEventBus`,
`RedisStateStore`) behind the same interfaces — enabling separate worker/API processes. Empty
= single-process. The Redis connection is lazy (nothing needs a live server at import time).

### Config & data split
- **Tuning** lives in `config/config.example.yaml`, validated by the pydantic schema in
  `app/config/schema.py` (the single source of truth — every layer reads typed config, not raw
  YAML). `get_settings()` is an `lru_cache`d `AppConfig`. YAML supports `${ENV}` expansion;
  a few values (DATABASE_URL, model paths) are env-overridable. The primary accuracy/cost lever
  is `processing.fps_tiers` (per-role frame rate; a camera's most motion-sensitive role wins,
  see `CameraSpec.fps_role`). **Gotcha:** `config.example.yaml` overrides two schema defaults —
  it sets `tracker.reid_enabled: false` / `reid_runtime: torch`, while `schema.py` defaults are
  `true` / `onnx`; whether Re-ID runs depends on whether this YAML is loaded vs pydantic defaults.
- **Cameras, zones, lines** are **NOT** in YAML — they live in the database, managed via the API.
- **Secrets** are env-only (`load_secrets()`), never read from YAML or returned by any GET.

### Auth & security
- All `/api/v1` routes require a Bearer token (`require_auth` in `app/api/deps.py`) **except the
  ops/health router** (`app/api/routers/health.py` sets no auth): `/api/v1/ready`, `/api/v1/metrics`,
  `/api/v1/cameras/{id}/health` are open (as is the bare `/health` liveness outside the prefix).
  Scheme is `jwt` (HS256) or `api_key` (constant-time compare), set by `api.auth_scheme`.
  Verification reads `secrets.api_auth_secret` (env `API_AUTH_SECRET`), **not**
  `settings.api.auth_secret` from YAML — changing the YAML value won't change auth; reset
  `get_secrets`, not `get_settings`, in tests.
- **Camera passwords are write-only**: accepted on POST/PATCH, stored AES-256-GCM-encrypted
  (`CAMERA_CREDENTIALS_KEY`, key outside the DB, `app/services/credentials.py`), decrypted only
  in memory at stream-open, never returned by any GET.

### DB session model
Sync SQLAlchemy 2.0. **DB-bound API routes are plain `def`** so FastAPI runs them in a thread
pool; **only the SSE endpoint is async**. Request handlers use the `get_session` dependency;
worker/service code uses the `session_scope()` context manager (`app/db/session.py`). This
keeps one DB story across SQL Server (prod, `mssql` extra) and SQLite (dev/tests).

## Conventions & gotchas

- **Lightweight core is a hard rule.** Core deps stay minimal; anything heavy
  (ultralytics, opencv, openvino, torch, onnxruntime, Pillow, redis, pyodbc) goes in an
  optional extra in `pyproject.toml` and is **imported lazily** inside its backend/method.
  Don't add a top-level import that breaks `pip install -e ".[dev]"`-only environments.
- **Events are NOT `slots=True`.** Subclasses call zero-arg `super().payload()`; `slots=True`
  breaks the `__class__` cell `super()` needs on Python 3.11. Target is **Python 3.11+** — keep
  it 3.11-compatible (no 3.12-only syntax).
- **Enums are `str, Enum`** (not `StrEnum`) intentionally — members serialize via `.value` and
  `StrEnum` would change payload identity. `ruff` rule `UP042` is ignored for this reason.
- Calculators must stay pure (return events; no DB/bus). Side effects belong in the engine
  (live-frame-dependent) or the projectors (everything else).
- Singletons (`get_settings`, `get_secrets`, `get_state_store`, `get_event_bus`, the DB engine)
  have `reset_*`/`configure_engine` functions; tests call these for isolation.

## Testing

Tests run on the core + dev deps only (numpy, shapely, fastapi, sqlalchemy, fakeredis). Each
test gets a fresh in-memory SQLite DB and reset singletons via the autouse
`_isolated_environment` fixture (`tests/conftest.py`). Key fixtures: `client` (TestClient with
lifespan/projectors wired), `auth_headers` (valid JWT), `fake_clock` (deterministic `FakeClock`
at t=1000), `make_camera`/`make_zone`/`make_line` factories.

Perception is faked: `FakeDetector`/`FakeTracker` (`app/inference/fakes.py`), a recorded-clip
`FrameSource` (`RecordedClipSource`), and `FakeClock`. The **real `CameraPipeline` is driven
end-to-end** via the `capture_factory` injection seam in `tests/integration/test_camera_pipeline.py`
(real pipeline + threaded `BoundedFrameQueue`) — prefer this seam over mocking internals.
`test_pipeline_recorded.py` is a separate, lighter BRD-acceptance test that hand-wires
`source.read()`→detect→track→calculators with no `CameraPipeline`. The `requires_inference`/
`requires_gpu` markers are registered but currently applied to no test — the full suite runs on
core+dev deps.

## Branching & CI

Promotion flow is enforced: **`develop` → `stage` → `production`**. `stage` accepts PRs only
from `develop`; `production` only from `stage` (`.github/workflows/branch-flow.yml`). CI
(`ci.yml`) runs `ruff check` + `pytest` as required status checks on stage/production rulesets.
`@Ahmedki1l` owns all paths (CODEOWNERS) — required reviewer for protected branches. Work
branches off `develop`.
