# Syncing a deployment machine (pull from `develop`)

How to bring a second machine (e.g. the Windows box) up to date after changes are
pushed to `develop`. `git` carries the code, the database (`camera_analytics.db`
= cameras/zones/lines), `config/config.local.yaml`, and the OpenVINO model.

**Secrets are never in git.** `.env` — including `CAMERA_CREDENTIALS_KEY` (decrypts
camera passwords) and `API_AUTH_SECRET` (signs API tokens) — must be copied by hand,
once, over a secure channel (USB / password manager / scp). **If the key on the peer
does not match the key the DB was encrypted with, every camera password fails to
decrypt and no camera connects.** The key was rotated recently, so an existing peer
`.env` is stale and must be refreshed.

## Steps on the peer machine

1. **Stop the app** so it releases `camera_analytics.db`.

2. **Back up the peer's current DB** (safety; its local time-series is discarded below):
   ```
   copy camera_analytics.db camera_analytics.db.bak     # Windows
   # cp camera_analytics.db camera_analytics.db.bak     # macOS/Linux
   ```

3. **Take the incoming code + DB + config + model.** The running app leaves the DB
   file dirty, so a plain `git pull` will refuse. For a pure consumer that never edits
   code locally, hard-reset to the remote:
   ```
   git fetch origin develop
   git reset --hard origin/develop
   ```
   (If this machine *does* make local commits, instead `git checkout -- camera_analytics.db`
   to drop local DB writes, then `git pull origin develop`.)

4. **Refresh secrets.** Copy `.env` from the machine that made the changes (or at least
   update `CAMERA_CREDENTIALS_KEY` and `API_AUTH_SECRET` to match). Never commit it.

5. **Update dependencies** (some may have changed; `httpx` is now a core dep, and the
   engine needs the inference extra):
   ```
   pip install -e ".[dev,inference]"
   ```

6. **Start the app.** `.env` already sets `CONFIG_PATH=config/config.local.yaml` and
   `MODEL_DETECTOR_PATH=yolo11n_openvino_model`, so:
   ```
   python -m app.main --api --port 8008
   ```
   No `alembic upgrade` is needed — the committed DB is already at head schema.

## Verify

```
# gate camera connects and is healthy
curl -s http://127.0.0.1:8008/api/v1/cameras/27/health -H "Authorization: Bearer <token>"
# occupancy per floor populated
curl -s http://127.0.0.1:8008/api/v1/occupancy/floors -H "Authorization: Bearer <token>"
# entry/exit counts (climbs as people cross the gate line)
curl -s http://127.0.0.1:8008/api/v1/entry-exit -H "Authorization: Bearer <token>"
```

If cameras report unreachable with a credential/decrypt error, the `CAMERA_CREDENTIALS_KEY`
in `.env` does not match the one the committed DB was encrypted with — recopy `.env`.

## Notes

- Editing cameras/zones/lines on **either** machine changes `camera_analytics.db`. Since
  it is a binary file, two machines editing it will conflict on the next sync. Treat one
  machine as the source of truth for camera/zone/line edits and let the other consume.
- `config/config.local.yaml` (camera IP allowlist, model path, tuning) is shared via git.
  Machine-specific overrides that must diverge belong in `.env`, not this file.
