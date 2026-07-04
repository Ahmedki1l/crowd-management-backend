"""HTTP still-image snapshot fetching for the snapshot-pull capture path (HLD 6.1).

Fetches a single JPEG from a camera's HTTP snapshot endpoint (e.g. Hikvision
ISAPI ``/ISAPI/Streaming/channels/<ch>/picture``) and decodes it to a BGR frame.
This is the CPU-cheap alternative to continuous RTSP decode for low-rate roles
(occupancy) on GPU-less hardware; see
:class:`~app.ingestion.capture.SnapshotCaptureThread`.

``httpx`` is a core dependency; ``cv2`` (JPEG decode) is imported lazily so this
module stays importable on the lightweight core dependency set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = get_logger(__name__)


def _build_auth(scheme: str, username: str, password: str) -> httpx.Auth:
    """Return the httpx auth for ``scheme`` — ``digest`` (default) or ``basic``."""
    if scheme.lower() == "basic":
        return httpx.BasicAuth(username, password)
    return httpx.DigestAuth(username, password)


class SnapshotClient:
    """Fetch a single JPEG from a camera's HTTP snapshot endpoint.

    Credentials are sent via HTTP auth (digest by default), never embedded in the
    URL. A failed request (network error, timeout, non-2xx) is logged without
    credentials and returned as ``None`` so the caller can skip the tick rather
    than crash — matching the capture layer's "one bad frame never kills the
    thread" contract.
    """

    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        *,
        auth: str = "digest",
        timeout_s: float = 5.0,
        verify_tls: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        """Configure the client (opens no connection until :meth:`fetch_bytes`).

        Args:
            url: Full snapshot URL, without credentials (see
                :func:`~app.ingestion.stream_url.snapshot_url`).
            username: HTTP auth username.
            password: HTTP auth password. Never logged.
            auth: ``digest`` (default) or ``basic``.
            timeout_s: Per-request timeout in seconds.
            verify_tls: Verify TLS certificates (https only).
            client: Optional pre-built httpx client (injected in tests). When
                supplied, ``timeout_s``/``verify_tls`` are assumed baked into it.
        """
        self._url = url
        self._auth = _build_auth(auth, username, password)
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_s, verify=verify_tls)

    def fetch_bytes(self) -> bytes | None:
        """GET the snapshot and return the raw JPEG bytes, or ``None`` on failure.

        Any transport error, timeout, or non-success status is logged (without
        credentials — the URL carries none) and returned as ``None``, never raised.
        """
        try:
            response = self._client.get(self._url, auth=self._auth)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "snapshot HTTP fetch failed: %s",
                exc,
                extra={"event": "snapshot_http_error", "url": self._url},
            )
            return None
        content = response.content
        if not content:
            logger.warning(
                "snapshot HTTP fetch returned empty body",
                extra={"event": "snapshot_http_empty", "url": self._url},
            )
            return None
        return content

    def close(self) -> None:
        """Close the underlying httpx client if this instance created it."""
        if self._owns_client:
            self._client.close()


def decode_jpeg(content: bytes | None) -> np.ndarray | None:
    """Decode JPEG bytes to a BGR image array, or ``None`` if decoding fails.

    ``cv2`` is imported lazily (optional ``inference`` extra); a corrupt or empty
    snapshot returns ``None`` so it is skipped rather than fatal.
    """
    if not content:
        return None
    import cv2  # lazy: optional inference dependency
    import numpy as np

    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        logger.warning(
            "snapshot JPEG decode failed", extra={"event": "snapshot_decode_failed"}
        )
        return None
    return image
