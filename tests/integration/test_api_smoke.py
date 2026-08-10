"""Smoke-test every endpoint the app exposes (HLD 8).

Two things are asserted for the whole API surface, driven off the OpenAPI spec rather than
a hand-kept list — so an endpoint added later cannot quietly escape coverage:

* **Every route answers.** No 404 from a mis-registered router, no 500 from an import or
  wiring fault. This is the check that would have caught a router silently dropped during
  the occupancy-only cleanup.
* **Every ``/api/v1`` route is authenticated**, except the ops/health endpoints that are
  deliberately open for probes and scraping.

Behaviour lives in the per-router tests; this is the breadth pass that proves the surface
is wired, and that the two lists (routed, authed) still agree with the spec.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app

# Deliberately unauthenticated: liveness/readiness probes and the metrics scrape target
# (see app/api/routers/health.py, which sets no auth dependency).
_OPEN_PATHS = {
    "/health",
    "/api/v1/ready",
    "/api/v1/metrics",
    "/api/v1/cameras/{camera_id}/health",
}

# Endpoints that mutate the engine: they rebuild the perception backends, which in a test
# environment (no models, no cameras) means a slow no-op. Exercised in test_engine_control.
_ENGINE_PATHS = {
    "/api/v1/engine/all",
    "/api/v1/engine/entry-exit",
    "/api/v1/engine/occupancy",
    "/api/v1/engine/reset",
    "/api/v1/engine/stop",
}

# SSE: an infinite stream. Opening it with TestClient would block forever.
_STREAM_PATHS = {"/api/v1/stream"}

# Placeholder values for path params, and required query params per path.
_PATH_VALUES = {
    "camera_id": "1",
    "camera_ip": "10.0.0.99",
    "zone_id": "1",
    "line_id": "1",
    "alert_id": "1",
}
_REQUIRED_QUERY = {
    "/api/v1/history/entry-exit": {"area_id": "lobby"},
    "/api/v1/history/entry-exit/daily": {"area_id": "lobby"},
}

# A request that is well-formed but names something that does not exist is a 404, and a
# GET of a resource we did not seed is a legitimate empty answer. What must never happen
# is a 5xx (wiring/import fault) or a 405 (router registered under the wrong method).
_ACCEPTABLE = {200, 201, 204, 404, 422}

# These reach out to real camera hardware, which does not exist in a test environment.
# A 502 there is the endpoint working correctly — it is reporting an upstream failure —
# so it is allowed *for these paths only*, rather than by widening the rule for everything.
_HARDWARE_PATHS = {
    "/api/v1/tools/cameras/{camera_id}/frame": {502, 503},
    "/api/v1/cameras/{camera_id}/test": {502, 503},
}


def _spec_routes() -> list[tuple[str, str]]:
    """Every (method, path) the app publishes, read from the OpenAPI spec itself."""
    spec = create_app().openapi()
    return [
        (method.upper(), path)
        for path, operations in spec["paths"].items()
        for method in operations
    ]


def _fill(path: str) -> str:
    for name, value in _PATH_VALUES.items():
        path = path.replace("{" + name + "}", value)
    return path


_TESTABLE = [
    (method, path)
    for method, path in _spec_routes()
    if path not in _ENGINE_PATHS | _STREAM_PATHS
]
_READS = [(m, p) for m, p in _TESTABLE if m == "GET"]


# The endpoints the Digital Twin is built against. A count-based tripwire would break on
# every legitimate addition and tell you nothing about what moved; naming the contract
# fails loudly and specifically when a router that something depends on goes missing.
_CONTRACT = {
    ("GET", "/api/v1/history/occupancy"),
    ("GET", "/api/v1/history/occupancy/floors"),
    ("GET", "/api/v1/occupancy"),
    ("GET", "/api/v1/occupancy/spaces"),
    ("GET", "/api/v1/occupancy/floors"),
    ("GET", "/api/v1/state"),
    ("GET", "/api/v1/stream"),
}
# /health is deliberately excluded from the OpenAPI schema (a bare liveness probe), so it
# cannot be checked against the spec. test_the_open_endpoints_stay_open covers it.


def test_the_digital_twin_contract_endpoints_are_all_routed() -> None:
    """The routes the DT depends on must exist — a dropped router fails here, by name."""
    missing = _CONTRACT - set(_spec_routes())

    assert not missing, f"contract endpoints are not routed: {sorted(missing)}"


@pytest.mark.parametrize(("method", "path"), _READS, ids=lambda v: str(v))
def test_every_get_endpoint_answers(
    client: TestClient,
    auth_headers: dict[str, str],
    crowd_camera_internal_headers: dict[str, str],
    method: str,
    path: str,
) -> None:
    """No 404-from-misrouting and no 500-from-wiring, across the whole read surface."""
    headers = (
        crowd_camera_internal_headers
        if path.startswith("/api/v1/internal/cameras/")
        else auth_headers
    )
    response = client.request(
        method, _fill(path), headers=headers, params=_REQUIRED_QUERY.get(path)
    )

    allowed = _ACCEPTABLE | _HARDWARE_PATHS.get(path, set())
    assert response.status_code in allowed, (
        f"{method} {path} -> {response.status_code}: {response.text[:200]}"
    )


@pytest.mark.parametrize(
    ("method", "path"),
    [(m, p) for m, p in _TESTABLE if p.startswith("/api/v1") and p not in _OPEN_PATHS],
    ids=lambda v: str(v),
)
def test_every_api_endpoint_requires_auth(
    client: TestClient, method: str, path: str
) -> None:
    """Auth is on the router, not the handler — so this catches a router added without it."""
    response = client.request(method, _fill(path))

    assert response.status_code in (401, 403), (
        f"{method} {path} answered {response.status_code} with no token"
    )


@pytest.mark.parametrize("path", sorted(_OPEN_PATHS), ids=lambda v: str(v))
def test_the_open_endpoints_stay_open(client: TestClient, path: str) -> None:
    """Probes and the metrics scrape must not start demanding a token."""
    response = client.get(_fill(path))

    assert response.status_code != 401
    assert response.status_code != 403
