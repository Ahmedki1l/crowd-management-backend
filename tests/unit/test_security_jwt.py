"""Unit tests for the minimal HS256 JWT helpers (app.utils.security)."""

from __future__ import annotations

import pytest

from app.utils.security import JWTError, decode_jwt, encode_jwt

_SECRET = "unit-secret"


def test_encode_decode_roundtrip_preserves_claims() -> None:
    token = encode_jwt({"sub": "alice"}, _SECRET, expires_in=3600)

    payload = decode_jwt(token, _SECRET)

    assert payload["sub"] == "alice"


def test_tampered_token_raises_jwt_error() -> None:
    token = encode_jwt({"sub": "alice"}, _SECRET, expires_in=3600)
    header_b64, payload_b64, sig_b64 = token.split(".")
    # Flip the signature segment so HMAC verification fails.
    tampered = f"{header_b64}.{payload_b64}.{sig_b64[:-2]}AA"

    with pytest.raises(JWTError):
        decode_jwt(tampered, _SECRET)


def test_expired_token_raises_jwt_error() -> None:
    # Negative expiry places exp in the past, so decode must reject it.
    token = encode_jwt({"sub": "alice"}, _SECRET, expires_in=-10)

    with pytest.raises(JWTError, match="expired"):
        decode_jwt(token, _SECRET)
