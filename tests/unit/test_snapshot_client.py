"""Unit tests for the HTTP snapshot client and URL builder.

Exercises :class:`SnapshotClient.fetch_bytes` over an ``httpx.MockTransport`` so
no real camera/network is needed, plus the :func:`snapshot_url` builder. The
JPEG decode path (``decode_jpeg``) needs ``cv2`` (optional inference extra), so
only its no-op guards are asserted here.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx

from app.domain.models import CameraRole, CameraSpec
from app.ingestion.snapshot import SnapshotClient, decode_jpeg
from app.ingestion.stream_url import snapshot_url


def _spec(**over: object) -> CameraSpec:
    defaults: dict = {
        "id": 1,
        "name": "cam",
        "area": "lobby",
        "ip": "10.0.0.9",
        "port": 554,
        "username": "admin",
        "roles": (CameraRole.OCCUPANCY,),
        "stream_channel_sub": 102,
        "stream_channel_main": 101,
    }
    defaults.update(over)
    return CameraSpec(**defaults)  # type: ignore[arg-type]


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_snapshot_url_uses_sub_channel_and_embeds_no_credentials() -> None:
    url = snapshot_url(_spec(), scheme="http", port=80)
    assert url == "http://10.0.0.9:80/ISAPI/Streaming/channels/102/picture"
    assert "admin" not in url  # credentials are never in the URL


def test_snapshot_url_honours_custom_template_and_https() -> None:
    url = snapshot_url(
        _spec(stream_channel_sub=1), scheme="https", port=8443,
        path_template="/snap/{channel}.jpg",
    )
    assert url == "https://10.0.0.9:8443/snap/1.jpg"


def test_fetch_bytes_returns_jpeg_on_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/picture")
        return httpx.Response(200, content=b"\xff\xd8jpegbytes")

    client = SnapshotClient("http://h/picture", "admin", "pw", client=_client(handler))
    assert client.fetch_bytes() == b"\xff\xd8jpegbytes"


def test_fetch_bytes_returns_none_on_error_status() -> None:
    client = SnapshotClient(
        "http://h/picture", "admin", "pw",
        client=_client(lambda r: httpx.Response(404)),
    )
    assert client.fetch_bytes() is None


def test_fetch_bytes_returns_none_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    client = SnapshotClient("http://h/picture", "admin", "pw", client=_client(handler))
    assert client.fetch_bytes() is None  # logged and swallowed, never raised


def test_fetch_bytes_returns_none_on_empty_body() -> None:
    client = SnapshotClient(
        "http://h/picture", "admin", "pw",
        client=_client(lambda r: httpx.Response(200, content=b"")),
    )
    assert client.fetch_bytes() is None


def test_decode_jpeg_guards_empty_input() -> None:
    assert decode_jpeg(None) is None
    assert decode_jpeg(b"") is None
