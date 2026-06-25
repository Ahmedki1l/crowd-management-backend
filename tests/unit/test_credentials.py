"""Unit tests for camera-credential encryption (app.services.credentials)."""

from __future__ import annotations

import pytest

from app.config.settings import reset_settings_cache
from app.services.credentials import CredentialCipher


def test_generate_key_roundtrip_encrypt_decrypt() -> None:
    cipher = CredentialCipher(CredentialCipher.generate_key())

    token = cipher.encrypt("rtsp-secret")

    assert cipher.decrypt(token) == "rtsp-secret"


def test_decrypt_with_different_key_raises() -> None:
    encrypting = CredentialCipher(CredentialCipher.generate_key())
    other = CredentialCipher(CredentialCipher.generate_key())
    token = encrypting.encrypt("rtsp-secret")

    with pytest.raises(ValueError):
        other.decrypt(token)


def test_from_env_raises_clear_error_when_key_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # conftest sets CAMERA_CREDENTIALS_KEY; remove it and rebuild the secrets
    # cache so from_env() observes an unset key.
    monkeypatch.delenv("CAMERA_CREDENTIALS_KEY", raising=False)
    reset_settings_cache()

    with pytest.raises(RuntimeError, match="CAMERA_CREDENTIALS_KEY"):
        CredentialCipher.from_env()
