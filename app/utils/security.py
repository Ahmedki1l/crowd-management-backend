"""Minimal HS256 JWT (no external dependency) for API auth (HLD 8 / 14).

Intentionally tiny: encode/decode/verify HS256 with an ``exp`` claim. The
service runs behind the platform gateway; this is the second factor on routes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


class JWTError(Exception):
    pass


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def encode_jwt(payload: dict[str, Any], secret: str, expires_in: int = 3600) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    body = dict(payload)
    body.setdefault("iat", int(time.time()))
    body.setdefault("exp", int(time.time()) + expires_in)
    segments = [
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode()),
        _b64url_encode(json.dumps(body, separators=(",", ":")).encode()),
    ]
    signing_input = ".".join(segments).encode("ascii")
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    segments.append(_b64url_encode(signature))
    return ".".join(segments)


def decode_jwt(token: str, secret: str) -> dict[str, Any]:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError as exc:
        raise JWTError("malformed token") from exc
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64url_decode(sig_b64)):
        raise JWTError("bad signature")
    payload = json.loads(_b64url_decode(payload_b64))
    if "exp" in payload and int(payload["exp"]) < int(time.time()):
        raise JWTError("token expired")
    return payload
