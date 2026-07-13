# Camera-Based Analytics Service — Backend

Backend for the **Digital Twin** Camera-Based Analytics module. It ingests live
RTSP video from existing IP cameras, detects and tracks people, and derives five
operational metrics — **occupancy, entry/exit, safety alerts, waiting times, and
heat maps** — exposing them over a REST API and a real-time stream for the
Digital Twin to visualise.

> Implements the *Camera-Based Analytics — Backend HLD v1.0* against the
> *Camera-Based Analytics — BRD v1.0*. Backend only; the DT 3D front-end,
> cameras, and network are pre-existing.

---

## How it works

One pipeline powers all five use cases; only the final calculation differs:

```
Ingest ─▶ Detect ─▶ Track ─▶ Localise ─▶ Compute ─▶ Publish
(RTSP)   (YOLO)   (ByteTrack) (zones/   (5 calcs)  (REST + SSE
                   +OSNet)     lines)               + DT push)
```

Each camera is processed by an **isolated pipeline worker** (dedicated capture
thread → bounded drop-oldest queue → batched inference → tracker → localisation
→ analytics → publish), so cameras scale horizontally and one camera's failure
never affects another.

| Use case | Requirement | Method |
|---|---|---|
| Occupancy | FR-OCC-01/02 | Count distinct live tracks whose bottom-center is inside each zone; debounced by a state machine |
| Entry / Exit | FR-EE-01/02/03 | Line crossings set IN/OUT; net = IN − OUT per area |
| Safety alerts | FR-SAF-01/02/03 | Restricted-zone intrusion + overcrowding rules, debounced with cooldown, with snapshot evidence |
| Waiting times | FR-WAIT-01/02/03 | Dwell sessions opened/closed on confirmed enter/leave; current + rolling average |
| Heat maps | FR-HM-01/02/03 | Bottom-center points accumulated into a per-camera image grid; aggregated over a time range |

---

## Architecture (six layers)

| Layer | Package | Responsibility |
|---|---|---|
| 1 · Ingestion | `app/ingestion` | RTSP capture, decode, sample, reconnect; bounded drop-oldest queue |
| 2 · Inference | `app/inference` | Person detection (YOLO, OpenVINO/TensorRT) + tracking (ByteTrack) + optional OSNet Re-ID |
| 3 · Localisation | `app/localisation` | Image-space geometry (point-in-polygon, line crossing) + debounced presence state machine |
| 4 · Analytics | `app/analytics` | The five calculators (pure: consume tracked detections, emit events) |
| 5 · State & storage | `app/services/state_store`, `app/db`, `app/events` | Current-state read model, time-series DB, snapshots, event bus |
| 6 · API & publish | `app/api`, `app/publish` | REST + SSE, outbound Digital Twin push |

Shared contracts live in `app/domain` (value objects + interfaces), `app/events`
(event schema + bus), and `app/config` (typed settings). **Calculators never
touch the DB** — they emit events; a *state projector* updates the read model and
a *persistence projector* writes time-series off the hot path.

---

## Quick start (local, SQLite)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                      # core + test deps (no GPU/models needed)

cp .env.example .env                         # set CAMERA_CREDENTIALS_KEY:
python -c "from app.services.credentials import CredentialCipher; print(CredentialCipher.generate_key())"

pytest                                        # 214 tests, green without cameras/GPU/models
python scripts/seed_demo.py                   # register a demo camera + zones + line
uvicorn app.api.app:app --reload             # API at http://localhost:8008
```

Open `http://localhost:8008/docs` for the live OpenAPI UI.

### Run modes (HLD 12.5)

```bash
python -m app.main --api                      # API + engine (all cameras) — small deployments
python -m app.main --workers                  # pipeline workers only (no API)
python -m app.main --worker --camera 1        # one camera (debug / scale-out)
uvicorn app.api.app:app --host 0.0.0.0        # API service only (separate from workers)
```

For a real pipeline, install the inference extras and export a detector:

```bash
pip install -e ".[inference,reid]"
python scripts/export_model.py --weights yolo11n.pt --format openvino --int8
```

## Docker

```bash
docker compose up --build                          # one `api` service: API + engine, SQLite
docker compose run --rm api alembic upgrade head   # apply migrations
```

One process runs both the API and the camera engine (`--api`). That is deliberate.
`RuntimeWiring` subscribes a `PersistenceProjector` in **every** process it runs in,
and that projector *appends* rows — it is not idempotent. The previous split
`api` + `worker` topology sharing a Redis bus therefore persisted each event once per
process, and because `--api` also starts the engine for every camera, it ran the
pipeline twice as well: every occupancy sample and crossing event was written roughly
four times.

### Scaling out (separate worker + API processes)

Still possible — set `REDIS_URL` (`cache.url`) and the event bus and current-state read
model switch to their Redis-backed implementations (`RedisEventBus`, `RedisStateStore`)
behind the same interfaces. But **exactly one process may wire the persistence
projector**, or the time-series tables will be double-written. There is no uniqueness
constraint on any of them to catch it.

### Cross-camera identity (Re-ID)

With `tracker.reid_enabled: true`, the engine builds **one shared OSNet Re-ID
gallery** across all pipelines and promotes the appearance-stable `global_id` to
the working identity. A person therefore keeps one identity across overlapping
cameras and across a leave/re-enter that ByteTrack alone would renumber — so
occupancy is not double-counted and dwell survives a brief exit (HLD 5.6). The
gallery enforces per-frame mutual exclusion, so two people in one frame can never
collapse to one identity.

---

## API surface (`/api/v1`, JWT/API-key auth on all routes)

- **Config** — `/cameras`, `/cameras/{id}/test`, `/zones`, `/lines`, `/config`
- **Live metrics** — `/state`, `/occupancy`, `/entry-exit`, `/waiting`, `/alerts`, `/stats`
- **History** — `/history/{occupancy,entry-exit,waiting,alerts}`
- **Realtime & ops** — `/stream` (SSE), `/snapshots/{path}`, `/alerts/{id}/ack`, `/cameras/{id}/health`, `/health`, `/ready`, `/metrics`

> **Camera credentials are write-only.** Passwords are accepted on POST/PATCH
> over TLS, stored **encrypted** (AES-256-GCM via `CAMERA_CREDENTIALS_KEY`,
> key held outside the DB), decrypted only in memory at stream-open, and never
> returned by any GET.

---

## Configuration

Runtime behaviour is driven by `config/config.example.yaml` (per-environment),
with secrets via environment variables (see `.env.example`). **Cameras, zones,
and lines are managed through the API and stored in the database** — never in the
YAML. The primary accuracy/cost lever is the per-role **frame-rate tier**
(`processing.fps_tiers`): entrances/queues at 5–10 fps, occupancy/heat-maps at
1–2 fps.

---

## Project layout

```
app/
  config/        typed settings + loader
  domain/        shared value objects + interfaces (the contract spine)
  ingestion/     RTSP capture, bounded queue, stream URLs
  inference/     detector, tracker, Re-ID, fakes, factory
  localisation/  geometry, zones, lines, state machine
  analytics/     occupancy, entry_exit, safety, waiting
  engine/        per-camera pipeline + multi-camera lifecycle
  events/        event schema + in-process & Redis buses + serialization
  publish/       Digital Twin push + SSE fan-out
  api/           FastAPI app, routers, schemas, deps
  db/            ORM models, session, repositories
  services/      business logic (state store, projectors, per-use-case services)
  utils/         clock, logging, security, snapshots, time
config/ models/ scripts/ migrations/ tests/
```

## Testing

```bash
pytest                      # 214 unit + integration tests
pytest tests/unit           # geometry, state machine, calculators, Re-ID gallery, crypto, JWT
pytest --timeout=30         # guard threaded pipeline/bus tests against hangs
```

Tests run on the lightweight core (numpy, shapely, fastapi, sqlalchemy, fakeredis)
using a `FakeDetector`/`FakeTracker`, a recorded-clip source, and a `FakeClock` —
no cameras, GPU, model artifacts, or Redis server required. The real
`CameraPipeline` is driven end-to-end over a recorded clip via the
`capture_factory` injection seam. Two heat-map overlay tests skip unless Pillow is
installed (`requires_inference`).

## Security & privacy

People are tracked **anonymously** via track IDs — no face recognition, no
biometric identity. Cameras sit on an isolated VLAN; the API runs behind the
platform gateway. A retention policy governs history and snapshots
(`config.retention`).
