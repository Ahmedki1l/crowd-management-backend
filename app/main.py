"""Process entry point and run-mode selector (HLD 12.5).

    python -m app.main --api                     # API + engine (all cameras)
    python -m app.main --workers                 # pipeline workers only (no API)
    python -m app.main --worker --camera CAM-01  # one camera (debug/scale-out)
    uvicorn app.api.app:app --host 0.0.0.0       # API service only
"""

from __future__ import annotations

import argparse
import signal
import threading

from app.config.settings import get_settings
from app.utils.logging import get_logger, setup_logging

logger = get_logger("main")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="camera-analytics")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--api", action="store_true", help="API + engine (small deployments)")
    mode.add_argument("--workers", action="store_true", help="pipeline workers only")
    mode.add_argument("--worker", action="store_true", help="a single camera worker")
    parser.add_argument("--camera", type=int, default=None, help="camera id for --worker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8008)
    return parser.parse_args(argv)


def _run_api_with_engine(host: str, port: int) -> None:
    import uvicorn

    from app.api.app import app
    from app.engine.manager import get_engine_manager

    # Start via the shared manager so the engine-control endpoints
    # (/api/v1/engine/*) drive the same engine this process launched.
    manager = get_engine_manager()
    manager.start(mode="all")
    try:
        uvicorn.run(app, host=host, port=port, log_config=None)
    finally:
        manager.stop()


def _wire_runtime():
    """Attach the bus consumers (read-model, persistence, optional DT push).

    Worker processes have no FastAPI lifespan, so without this their published
    events would have no projector/persistence consumer. Returns the wiring so
    the caller can tear it down on shutdown.
    """
    from app.events.event_bus import get_event_bus
    from app.services.runtime_wiring import RuntimeWiring
    from app.services.state_store import get_state_store

    wiring = RuntimeWiring(get_event_bus(), get_state_store(), get_settings())
    wiring.start()
    return wiring


def _run_workers() -> None:
    from app.engine.engine import build_engine

    wiring = _wire_runtime()
    engine = build_engine()
    stop = threading.Event()

    def _handle(signum, _frame):  # noqa: ANN001
        logger.info("signal %s received, stopping engine", signum)
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    engine.start()
    stop.wait()
    engine.stop()
    wiring.close_sync()


def _run_single_camera(camera_id: int) -> None:
    from app.engine.engine import build_engine

    wiring = _wire_runtime()
    engine = build_engine(camera_ids=[camera_id])
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    engine.start()
    stop.wait()
    engine.stop()
    wiring.close_sync()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    setup_logging()
    get_settings()  # validate config early
    if args.worker:
        if args.camera is None:
            raise SystemExit("--worker requires --camera <id>")
        _run_single_camera(args.camera)
    elif args.workers:
        _run_workers()
    else:  # --api or default
        _run_api_with_engine(args.host, args.port)


if __name__ == "__main__":
    main()
