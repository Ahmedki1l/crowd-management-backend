"""Contract test for the outbound Digital Twin push publisher (HLD 10).

:class:`~app.publish.dt_client.DigitalTwinPublisher` bridges the sync event bus
to the Digital Twin's HTTP ingest endpoint off the hot path: ``handle`` enqueues
an event and a background thread POSTs ``event.payload()`` as JSON to the
configured ``push_url``. These tests replace the shared module-level ``httpx``
client with one bound to an :class:`httpx.MockTransport`, which records every
request, then assert the observable POST and a clean ``aclose``.

The publisher's worker is a real background thread; a :class:`threading.Event`
set inside the transport handler makes the wait for delivery deterministic
(bounded timeout, no fixed sleeps). The module-level client is saved and restored
so the test leaves no global state behind.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator

import httpx
import pytest

import app.publish.dt_client as dt_client
from app.config.schema import DigitalTwinConfig
from app.events.events import OccupancyUpdate
from app.publish.dt_client import DigitalTwinPublisher

_PUSH_URL = "http://dt.local/ingest"
_DELIVERY_TIMEOUT_S = 5.0


class _RecordingTransport:
    """Captures the requests an :class:`httpx.MockTransport` sees.

    ``delivered`` is set as soon as the first request arrives so a test can wait
    deterministically for the background worker to POST, instead of sleeping.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.delivered = threading.Event()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.delivered.set()
        return httpx.Response(200, json={"ok": True})


@pytest.fixture
def recording_client() -> Iterator[_RecordingTransport]:
    """Install a request-recording ``httpx`` client as the module's shared client.

    Saves and restores ``dt_client._client`` so the publisher's ``aclose`` (which
    closes and nulls the shared client) cannot leak into other tests.
    """
    recorder = _RecordingTransport()
    saved = dt_client._client
    dt_client._client = httpx.Client(transport=httpx.MockTransport(recorder))
    try:
        yield recorder
    finally:
        current = dt_client._client
        if current is not None and not current.is_closed:
            current.close()
        dt_client._client = saved


async def test_handle_posts_event_payload_to_push_url(
    recording_client: _RecordingTransport,
) -> None:
    cfg = DigitalTwinConfig(push_url=_PUSH_URL, push_enabled=True, max_retries=0)
    publisher = DigitalTwinPublisher(cfg)
    publisher.start()
    event = OccupancyUpdate(ts=1000.0, zone_id=2, camera_id=1, count=5)

    publisher.handle(event)
    delivered = recording_client.delivered.wait(timeout=_DELIVERY_TIMEOUT_S)
    await publisher.aclose()

    assert delivered
    request = recording_client.requests[0]
    assert request.method == "POST"
    assert str(request.url) == _PUSH_URL
    assert json.loads(request.content) == event.payload()


async def test_aclose_closes_shared_client(
    recording_client: _RecordingTransport,
) -> None:
    cfg = DigitalTwinConfig(push_url=_PUSH_URL, push_enabled=True, max_retries=0)
    publisher = DigitalTwinPublisher(cfg)
    publisher.start()

    await publisher.aclose()

    assert dt_client._client is None
