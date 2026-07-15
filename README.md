# Camera-Based Analytics Service — Backend

Backend for the **Digital Twin** Camera-Based Analytics module. It ingests live
video from existing IP cameras, detects people, and derives **occupancy** and
**entry/exit** — exposing them over a REST API and a real-time stream for the Digital
Twin to visualise.

> Originally built for five metrics. Safety alerts, waiting times and heat maps were
> removed once the delivered scope narrowed to occupancy; entry/exit is retained but
> currently paused via `processing.entry_exit_enabled`. See
> `docs/OCCUPANCY_ONLY_REMOVAL_PLAN.md`.

> Implements the *Camera-Based Analytics — Backend HLD v1.0* against the
> *Camera-Based Analytics — BRD v1.0*. Backend only; the DT 3D front-end,
> cameras, and network are pre-existing.

---

## How it works

One pipeline powers both use cases; only the final calculation differs:

```
Ingest ──────────▶ Detect ─▶ [Track] ─▶ Localise ─▶ Compute ─▶ Publish
(HTTP snapshot     (YOLO,    (ByteTrack   (zones/    (occupancy,  (REST + SSE)
 pull, or RTSP      OpenVINO)  — tracked    lines)     entry/exit)
 for the gate)                 path only)
```

The deployed occupancy path pulls HTTP stills and counts detections in-zone **directly**
(`snapshot_pull.count_from_detections`), skipping the tracker: at a ~1 s snapshot cadence
ByteTrack cannot reliably link people between frames, so tracking only drops real people
from the count. The tracked path (detector → tracker → presence state machine) still backs
entry/exit and remains the default when snapshot pull is off.

Each camera is processed by an **isolated pipeline worker** (dedicated capture
thread → bounded drop-oldest queue → batched inference → tracker → localisation
→ analytics → publish), so cameras scale horizontally and one camera's failure
never affects another.

| Use case | Requirement | Method |
|---|---|---|
| Occupancy | FR-OCC-01/02 | Count detections whose bbox bottom-centre is inside each zone. Tracked path (default) debounces via a presence state machine; the deployed snapshot path counts detections directly |
| Entry / Exit | FR-EE-01/02/03 | Line crossings set IN/OUT; net = IN − OUT per area |

---

## Architecture (six layers)

| Layer | Package | Responsibility |
|---|---|---|
| 1 · Ingestion | `app/ingestion` | RTSP capture, decode, sample, reconnect; bounded drop-oldest queue |
| 2 · Inference | `app/inference` | Person detection (YOLO, OpenVINO/TensorRT) + tracking (ByteTrack) + optional OSNet Re-ID |
| 3 · Localisation | `app/localisation` | Image-space geometry (point-in-polygon, line crossing) + debounced presence state machine |
| 4 · Analytics | `app/analytics` | The calculators (pure: consume tracked detections, emit events) |
| 5 · State & storage | `app/services/state_store`, `app/db`, `app/events` | Current-state read model, time-series DB, event bus |
| 6 · API & publish | `app/api`, `app/publish` | REST + SSE |

Shared contracts live in `app/domain` (value objects + interfaces), `app/events`
(event schema + bus), and `app/config` (typed settings). **Calculators never
touch the DB** — they emit events; a *state projector* updates the read model and
a *persistence projector* writes time-series off the hot path.

---

## Quick start (local, SQLite)

Requires **Python 3.11+** (`requires-python = ">=3.11"` in `pyproject.toml`).

### macOS

macOS ships Python 3.9 as `python3` and has **no bare `python`** — so `python -m venv` fails
on a clean machine, and `python3 -m venv` would build a venv on a version this project does
not support. Name the interpreter explicitly:

```bash
brew install python@3.11                     # if you don't have it
python3.11 -m venv .venv                     # NOT `python` or `python3` on macOS
source .venv/bin/activate                    # `python` now means 3.11, inside the venv
python --version                             # -> Python 3.11.x
```

### Linux

```bash
python3.11 -m venv .venv && source .venv/bin/activate
```

### Windows

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### Then, on any of them

```bash
pip install -e ".[dev]"                      # core + test deps (no GPU/models needed)

cp .env.example .env
# .env needs CAMERA_CREDENTIALS_KEY (required — the app will not start without it):
python -c "from app.services.credentials import CredentialCipher; print(CredentialCipher.generate_key())"

pytest                                       # 315 tests, green without cameras/GPU/models
python scripts/seed_demo.py                  # register a demo camera + zones + line
```

## Running the server

```bash
source .venv/bin/activate
uvicorn app.api.app:app --host 0.0.0.0 --port 8008
```

**Pass `--port 8008` explicitly.** Bare `uvicorn` binds **8000**, while the Dockerfile,
compose file and `python -m app.main` all use 8008 — so omitting it silently serves the API
somewhere other than where everything else expects it.

`uvicorn` runs the **API only**: it serves the read/config endpoints and starts the history
writers, but it does **not** start the camera engine, so it never touches your cameras. That
is what you want for API work. To also run the pipelines, use `--api` below.

Once it is up:

| URL | What |
| --- | --- |
| `http://localhost:8008/docs` | Interactive OpenAPI UI — every endpoint, try-it-out |
| `http://localhost:8008/api/v1/tools/roi` | **ROI editor** — draw occupancy zones and entry/exit lines on a live camera frame |
| `http://localhost:8008/health` | Liveness (no auth) |
| `http://localhost:8008/api/v1/ready` | Readiness: `db_ok`, `models_loaded` (no auth) |

Every other `/api/v1` route needs a Bearer token. Mint one from the secret in your `.env`:

```bash
python -c "
from app.config.settings import get_secrets
from app.utils.security import encode_jwt
print(encode_jwt({'sub': 'api'}, get_secrets().api_auth_secret))"
```

```bash
export TOKEN=$(python -c "
from app.config.settings import get_secrets
from app.utils.security import encode_jwt
print(encode_jwt({'sub':'api'}, get_secrets().api_auth_secret))")

curl -s -H "Authorization: Bearer $TOKEN" localhost:8008/api/v1/cameras
curl -s -H "Authorization: Bearer $TOKEN" localhost:8008/api/v1/occupancy
curl -s -H "Authorization: Bearer $TOKEN" \
  "localhost:8008/api/v1/history/occupancy?bucket=1h&floor=B1"
```

### The ROI editor (drawing zones and lines)

`http://localhost:8008/api/v1/tools/roi` — paste the token into the box at the top (it is
cached in `localStorage`), pick a camera, then click points to draw.

**`dt_space_id` is required.** Occupancy history is stored per *space*, not per zone, so a
zone without one is counted live and then forgotten — it appears in no history query at all.
The API rejects it with a 422. Use an existing space (`b1-waiting-area`, `gf-waiting-area`, …)
so the zone feeds that space's series.

Redrawing a polygon is safe: history is keyed by `dt_space_id`, not by zone id, so deleting
and re-creating a zone no longer orphans it. **Re-use the same `dt_space_id`** and the series
stays continuous.

### Run modes (HLD 12.5)

```bash
python -m app.main --api                     # API + engine (all cameras) — small deployments
python -m app.main --workers                 # pipeline workers only (no API)
python -m app.main --worker --camera 1       # one camera (debug / scale-out)
uvicorn app.api.app:app --port 8008          # API only, no engine (see above)
```

`--api` and `--workers` default to port 8008 already. They start the **camera engine**, which
loads the detector and opens streams — so they need the inference extras and reachable
cameras. For a real pipeline, install those and export a detector:

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

With `tracker.reid_enabled: true`, the engine builds **one shared OSNet Re-ID gallery**
across all pipelines and promotes the appearance-stable `global_id` to the working
identity, so a person keeps one identity across a leave/re-enter that ByteTrack alone
would renumber (HLD 5.6).

> **It does not de-duplicate a person across overlapping cameras**, despite what this
> section used to claim. Zones are per-camera rows and `OccupancyService.by_space` *sums*
> their counts, so the same `global_id` seen by two cameras still contributes `1 + 1`.
> Cross-camera de-duplication would require counting *distinct identities* at the space
> level, which nothing does.
>
> Re-ID is also **unreachable on the deployed occupancy path**: with
> `snapshot_pull.count_from_detections: true`, `CameraPipeline._process_frame` returns
> before the tracker runs, so there are no tracked detections for Re-ID to attach to.
> It is disabled in `config/config.local.yaml` and applies only to the tracked
> (entry/exit) path.

---

## API surface (`/api/v1`, JWT/API-key auth on all routes)

- **Config** — `/cameras`, `/cameras/{id}/test`, `/zones`, `/lines`, `/config`
- **Live metrics** — `/state`, `/occupancy`, `/occupancy/{spaces,floors}`, `/entry-exit`, `/stats`
- **History** — `/history/occupancy` (per space), `/history/occupancy/floors` (per floor),
  `/history/entry-exit`, `/history/entry-exit/daily`
- **Realtime & ops** — `/stream` (SSE), `/engine/*`, `/cameras/{id}/health`, `/health`, `/ready`, `/metrics`
- **Tools** — `/tools/roi` (the zone/line drawing page), `/tools/cameras/{id}/frame`, `/tools/capture`

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
(`processing.fps_tiers`): entrances at 5–10 fps, occupancy at 1–2 fps.

---

## Project layout

```
app/
  config/        typed settings + loader
  domain/        shared value objects + interfaces (the contract spine)
  ingestion/     RTSP capture, bounded queue, stream URLs
  inference/     detector, tracker, Re-ID, fakes, factory
  localisation/  geometry, zones, lines, state machine
  analytics/     occupancy, entry_exit
  engine/        per-camera pipeline + multi-camera lifecycle
  events/        event schema + in-process & Redis buses + serialization
  publish/       SSE fan-out
  api/           FastAPI app, routers, schemas, deps
  db/            ORM models, session, repositories
  services/      business logic (state store, projectors, per-use-case services)
  utils/         clock, logging, security, time
config/ models/ scripts/ migrations/ tests/
```

## Testing

```bash
pytest                      # 315 unit + integration tests
pytest tests/unit           # geometry, state machine, calculators, Re-ID gallery, crypto, JWT
pytest --timeout=30         # guard threaded pipeline/bus tests against hangs
```

Tests run on the lightweight core (numpy, shapely, fastapi, sqlalchemy, fakeredis)
using a `FakeDetector`/`FakeTracker`, a recorded-clip source, and a `FakeClock` —
no cameras, GPU, model artifacts, or Redis server required. The real
`CameraPipeline` is driven end-to-end over a recorded clip via the
`capture_factory` injection seam. The snapshot-pull occupancy path — the one
production runs — is covered by `test_snapshot_occupancy.py`. Pillow is
installed (`requires_inference`).

## Security & privacy

People are tracked **anonymously** via track IDs — no face recognition, no
biometric identity. Cameras sit on an isolated VLAN; the API runs behind the
platform gateway. Retention is enforced by `app/services/retention.py` (config in
`retention.*`; `0` = keep forever). History
(`config.retention`).
