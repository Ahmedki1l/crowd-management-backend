# Snapshot-pull capture (CPU-only occupancy)

A capture mode for **GPU-less, low-core deployments**. Instead of decoding a
continuous RTSP sub-stream, a camera pulls a **single still image over HTTP every
N seconds** and runs the normal pipeline on it. It's the cheap path for
low-rate roles (occupancy); the entry/exit gate stays on RTSP.

- **Author's context:** added for a 2-core, no-GPU box running 1 entry gate +
  ~4 occupancy zones (6–10 cameras), Re-ID disabled.
- **Status:** opt-in and fully backward-compatible — with the default config
  (`snapshot_pull.roles: []`) every camera stays on RTSP and nothing changes.

---

## Why it exists — the decode problem

`RtspCaptureThread` calls `cv2.VideoCapture.read()`, which **fully decodes every
frame** of the sub-stream and only *then* drops frames down to the role's
`target_fps` (`app/ingestion/capture.py`, `_handle_frame`). So:

> `processing.fps_tiers` reduces **YOLO inference** cost. It does **not** reduce
> **H.264 decode** cost — decode scales with `#cameras × native sub-stream fps`.

On 2 cores, decoding ~10 continuous sub-streams is the binding constraint. Pulling
one JPEG every 2 s decodes ~0.5 images/s per camera instead, so decode cost tracks
the *pull* rate, not the stream's native rate.

---

## How it works

```
SnapshotCaptureThread (every interval_s)
  -> frame_provider()                 # decode_jpeg(SnapshotClient.fetch_bytes())
       -> SnapshotClient.fetch_bytes  # HTTP GET (httpx, digest auth) -> JPEG bytes
       -> decode_jpeg                 # cv2.imdecode -> BGR ndarray (lazy cv2)
  -> FramePacket -> BoundedFrameQueue -> (existing) detect -> track -> analyse
```

The frame flows through the **same** `CameraPipeline` as RTSP — same detector,
zone evaluation, `OccupancyCalculator`, events, projectors, and API. Only the
frame *source* differs.

### Components

| Piece | File | Role |
|---|---|---|
| `SnapshotCaptureThread` | `app/ingestion/capture.py` | Poll loop; paces to `interval_s`, stamps `FramePacket`, self-heals |
| `SnapshotClient` | `app/ingestion/snapshot.py` | HTTP fetch of one JPEG (httpx digest/basic auth) → bytes or `None` |
| `decode_jpeg` | `app/ingestion/snapshot.py` | JPEG bytes → BGR ndarray (lazy `cv2`) or `None` |
| `snapshot_url` | `app/ingestion/stream_url.py` | Builds the snapshot URL from the camera's sub channel |
| `_CaptureHealthMixin` | `app/ingestion/capture.py` | Shared `fps_estimate` / `is_healthy` for RTSP + snapshot threads |
| `SnapshotPullConfig` | `app/config/schema.py` | Typed config block |
| wiring | `app/engine/camera_pipeline.py` | `_build_capture` picks snapshot vs RTSP per role |

### Selection is per role

`CameraPipeline._build_capture` chooses the producer:

```python
snap = self._cfg.processing.snapshot_pull
if self._spec.fps_role.value in snap.roles:
    return self._build_snapshot_capture(queue, snap)   # SnapshotCaptureThread
return self._build_rtsp_capture(queue)                 # RtspCaptureThread
```

`fps_role` is the camera's most motion-sensitive role (`entry_exit > waiting >
occupancy > heatmap`, see `CameraSpec.fps_role`). So listing `["occupancy"]` in
`snapshot_pull.roles` routes occupancy-only cameras to snapshot pull while the
entry gate (`roles: ["entry_exit"]`) stays on RTSP — which is required, because
line-crossing needs the temporal continuity a 2 s cadence can't give.

### Resilience

`SnapshotCaptureThread` mirrors the RTSP contract: a failed fetch, an empty body,
a decode failure, or a raising provider is **logged and skipped** — the thread
**never exits** on a transient fault (`_capture_once`). Staleness surfaces only
through `is_healthy()` / the `CameraHealth` heartbeat once no frame has arrived
within `processing.watchdog_stale_seconds`. The `on_close` hook releases the HTTP
client when the loop ends, so a supervisor restart doesn't leak connections.

### Dependencies

No new ones. `httpx` is a **core** dependency; `cv2` (JPEG decode) stays in the
optional `inference` extra and is imported **lazily** inside `decode_jpeg`, so the
lightweight core / test environment is unaffected.

---

## Enabling it

### 1. Config (`config/config.example.yaml` → `processing.snapshot_pull`)

```yaml
processing:
  snapshot_pull:
    roles: ["occupancy"]     # roles whose cameras pull stills; [] = everyone on RTSP
    interval_s: 2.0          # seconds between pulls (the effective frame rate)
    http_port: 80            # camera HTTP port
    scheme: "http"           # http | https
    path_template: "/ISAPI/Streaming/channels/{channel}/picture"  # {channel}=stream_channel_sub
    auth: "digest"           # digest | basic
    timeout_s: 5.0
    verify_tls: true         # https only
```

`{channel}` is filled with each camera's `stream_channel_sub` (e.g. `102`).
Defaults (Hikvision ISAPI, port 80, digest) suit the reference cameras.

### 2. Camera + zone data (via the API — not YAML)

- **Gate camera:** `PATCH /api/v1/cameras/{id}` → `{"roles": ["entry_exit"]}`, plus a
  `Line` at the door. Stays on RTSP.
- **Occupancy cameras:** `{"roles": ["occupancy"]}`. For each **logical zone** that
  spans several cameras, create one `Zone` row per camera (`type: "occupancy"`),
  all sharing the **same `dt_space_id`** (e.g. `"zone-openspace"`). Draw the
  polygons so a person falls inside only one camera's zone (no Re-ID → summing is
  only correct when the per-camera areas don't overlap).

### 3. Restrict which cameras run at all (IP allowlist)

By default the engine builds a pipeline for **every enabled camera**. On a
constrained box you usually want only a chosen few pulling feed. Set an IP
allowlist:

```yaml
processing:
  camera_allowlist_ips: ["10.1.13.21", "10.1.13.39"]   # only these pull feed
```

- Empty (default) = every enabled camera runs (unchanged behavior).
- Non-empty = only cameras whose `ip` is listed get a pipeline; a camera must
  also be `enabled`. The engine logs `running N of M enabled cameras`.
- `--worker --camera N` bypasses the allowlist (deliberate debug/scale-out).
- Applied in `_resolve_target_ids` (`app/engine/engine.py`); takes effect on
  engine (re)start.

### 4. Low-fps tuning (recommended)

At 0.5 fps the presence state machine's defaults make counts sluggish
(`confirm_enter_frames: 5` ⇒ ~10 s to count someone; `confirm_leave_frames: 8` ⇒
~16 s to drop them). For a snapshot-only occupancy deployment, lower
`state_machine.confirm_enter_frames`/`confirm_leave_frames` (e.g. `1`/`2`). These
are global; the gate uses line-crossing (not zone presence), so lowering them
only affects occupancy responsiveness.

### 5. Verify the camera actually serves snapshots

```
curl -s -o /tmp/cam.jpg --digest -u 'user:pass' \
  'http://<ip>:80/ISAPI/Streaming/channels/102/picture' && file /tmp/cam.jpg
```

Expect `JPEG image data`. A 401/404 means the path, port, or auth needs adjusting
in `snapshot_pull` before the pipeline will work.

---

## Reading per-zone occupancy: `GET /api/v1/occupancy/spaces`

A logical zone spanning several cameras is stored as several `Zone` rows sharing
one `dt_space_id`. This endpoint sums their live counts into one number.

```
GET /api/v1/occupancy/spaces           # Bearer token required
GET /api/v1/occupancy/spaces?area_id=Ground%20Floor
```

```jsonc
[
  { "dt_space_id": "zone-openspace", "count": 7, "zone_ids": [12, 13, 14], "ts": 1751630000.0 }
]
```

- Sums live per-zone counts grouped by `dt_space_id`; zones with no `dt_space_id`
  are excluded (they belong to no logical space).
- `count` = sum of member zones, `zone_ids` = the members, `ts` = newest member
  timestamp. `area_id` restricts to zones whose camera's `area` matches.
- Backed by `OccupancyService.by_space` (`app/services/occupancy_service.py`);
  the per-zone `GET /api/v1/occupancy` is unchanged.

---

## Testing & extension

- **Tests:** `tests/unit/test_snapshot_client.py` (fetch over `httpx.MockTransport`
  + URL builder), `tests/unit/test_snapshot_capture.py` (`_capture_once`, health,
  self-heal, `on_close`), `tests/unit/test_snapshot_wiring.py` (role → capture
  type), `tests/unit/test_occupancy_spaces.py` (aggregation + endpoint). All run on
  core + dev deps (no cv2, no network).
- **Extending:** `SnapshotClient` is injectable (`client=` param) for tests. A
  different vendor's snapshot path is just a `path_template` change; a different
  transport is a new `frame_provider` — `SnapshotCaptureThread` is agnostic to how
  frames are produced.
- **Not covered by this feature:** the entry/exit gate (keep it on RTSP) and
  cross-camera de-duplication (relies on non-overlapping polygons since Re-ID is
  off).
