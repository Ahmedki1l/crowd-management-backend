"""RTSP URL construction and redaction (HLD 6.1 — capture).

The default convention is the Hikvision/ISAPI URL layout::

    rtsp://<user>:<pass>@<ip>:<port>/Streaming/Channels/<channel>

where ``<channel>`` selects the stream profile (e.g. ``101`` = main/channel-1,
``102`` = sub/channel-1). :class:`~app.domain.models.CameraSpec` carries the
main/sub channel numbers per camera, so :func:`main_stream_url` and
:func:`sub_stream_url` pick the right one without callers hard-coding it.

Passwords must never reach logs. Build URLs with the helpers here and log only
the output of :func:`redact`.
"""

from __future__ import annotations

from urllib.parse import quote

from app.domain.models import CameraSpec

# Hikvision-style path. Documented as the default; if a vendor differs the URL
# can be overridden upstream, but this is the convention the pipeline assumes.
_HIKVISION_STREAM_PATH = "Streaming/Channels"

# Placeholder substituted for the password when redacting a URL for logs.
_REDACTED = "****"


def build_rtsp_url(
    ip: str,
    port: int,
    username: str,
    password: str,
    channel: int,
    transport: str = "tcp",
) -> str:
    """Build a Hikvision-style RTSP URL for a camera channel.

    Args:
        ip: Camera host or IP address.
        port: RTSP port (typically 554).
        username: Stream account username.
        password: Stream account password. URL-encoded so special characters
            (``@``, ``:``, ``/`` ...) do not corrupt the authority section.
        channel: Stream profile/channel number (e.g. 101 main, 102 sub).
        transport: Requested RTSP transport. Accepted for API symmetry with the
            capture layer; the actual TCP/UDP negotiation is configured on the
            OpenCV/FFmpeg backend, not encoded in the URL. Defaults to ``"tcp"``.

    Returns:
        A fully-formed ``rtsp://`` URL string.
    """
    # transport is intentionally not embedded in the URL (FFmpeg controls it via
    # backend options); the parameter documents intent and keeps signatures
    # aligned with build_rtsp_url callers in the capture layer.
    _ = transport
    user = quote(username, safe="")
    secret = quote(password, safe="")
    return f"rtsp://{user}:{secret}@{ip}:{port}/{_HIKVISION_STREAM_PATH}/{channel}"


def redact(url: str) -> str:
    """Return ``url`` with the password replaced by ``****`` for safe logging.

    Only the password component of the ``user:pass@host`` authority is hidden;
    the username, host, port and path are preserved so the log remains useful
    for debugging. URLs without credentials are returned unchanged.

    Args:
        url: An RTSP (or any ``scheme://user:pass@host/...``) URL.

    Returns:
        The URL with its password masked, or the original string if no
        ``user:pass@`` credential section is present.
    """
    scheme_sep = "://"
    scheme_idx = url.find(scheme_sep)
    if scheme_idx == -1:
        return url
    scheme = url[: scheme_idx + len(scheme_sep)]
    remainder = url[scheme_idx + len(scheme_sep) :]

    at_idx = remainder.rfind("@")
    if at_idx == -1:
        return url
    credentials = remainder[:at_idx]
    host_part = remainder[at_idx:]  # includes the leading '@'

    if ":" not in credentials:
        # No password present (username only); nothing to redact.
        return url
    user = credentials.split(":", 1)[0]
    return f"{scheme}{user}:{_REDACTED}{host_part}"


def sub_stream_url(spec: CameraSpec, password: str, transport: str = "tcp") -> str:
    """Build the sub-stream (low-resolution) RTSP URL for ``spec``.

    Uses ``spec.stream_channel_sub`` — the lighter profile preferred for
    detection/tracking to save bandwidth and decode cost.

    Args:
        spec: Camera specification supplying host, port, username and channel.
        password: Resolved plaintext password (kept out of ``spec`` by design).
        transport: Requested RTSP transport. Defaults to ``"tcp"``.

    Returns:
        The RTSP URL for the camera's sub stream.
    """
    return build_rtsp_url(
        ip=spec.ip,
        port=spec.port,
        username=spec.username,
        password=password,
        channel=spec.stream_channel_sub,
        transport=transport,
    )


def main_stream_url(spec: CameraSpec, password: str, transport: str = "tcp") -> str:
    """Build the main-stream (high-resolution) RTSP URL for ``spec``.

    Uses ``spec.stream_channel_main`` — the full-resolution profile used when
    higher fidelity is required (e.g. snapshots).

    Args:
        spec: Camera specification supplying host, port, username and channel.
        password: Resolved plaintext password (kept out of ``spec`` by design).
        transport: Requested RTSP transport. Defaults to ``"tcp"``.

    Returns:
        The RTSP URL for the camera's main stream.
    """
    return build_rtsp_url(
        ip=spec.ip,
        port=spec.port,
        username=spec.username,
        password=password,
        channel=spec.stream_channel_main,
        transport=transport,
    )


def snapshot_url(
    spec: CameraSpec,
    *,
    scheme: str = "http",
    port: int = 80,
    path_template: str = "/ISAPI/Streaming/channels/{channel}/picture",
) -> str:
    """Build the HTTP still-image (snapshot) URL for a camera's sub stream.

    Used by the snapshot-pull capture path (CPU-constrained deployments). Unlike
    the RTSP URLs, credentials are **not** embedded — the snapshot client supplies
    them via HTTP auth — so this URL is safe to log as-is. ``{channel}`` in
    ``path_template`` is substituted with ``spec.stream_channel_sub`` (the sub
    stream, matching what RTSP detection uses).

    Args:
        spec: Camera specification supplying host and sub-stream channel.
        scheme: ``http`` or ``https``.
        port: Camera HTTP(S) port (typically 80).
        path_template: Vendor snapshot path with a ``{channel}`` placeholder.

    Returns:
        The snapshot URL, e.g. ``http://10.0.0.1:80/ISAPI/Streaming/channels/102/picture``.
    """
    path = path_template.format(channel=spec.stream_channel_sub)
    if not path.startswith("/"):
        path = "/" + path
    return f"{scheme}://{spec.ip}:{port}{path}"
