"""Camera credential encryption at rest (HLD 5.8, 14).

Camera RTSP passwords must never be stored in plaintext. This module encrypts
them with **AES-256-GCM** (authenticated symmetric encryption: 256-bit key,
96-bit random nonce, built-in integrity tag), so the persistence layer stores
opaque, tamper-evident ciphertext while plaintext exists only transiently.

The stored token is ``nonce (12 bytes) || ciphertext+tag``. The encryption key
is an env-only secret (``CAMERA_CREDENTIALS_KEY``) sourced via
:func:`app.config.settings.get_secrets`; it never lives in the database, so a
database dump alone cannot reveal credentials. Plaintext is never logged.
"""

from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.config.settings import get_secrets
from app.utils.logging import get_logger

logger = get_logger(__name__)

_ENV_VAR_NAME = "CAMERA_CREDENTIALS_KEY"
_KEY_BYTES = 32  # AES-256
_NONCE_BYTES = 12  # 96-bit GCM nonce (NIST SP 800-38D recommended size)


def _decode_key(key: str) -> bytes:
    """Decode a url-safe base64 key string to its 32 raw bytes."""
    try:
        raw = base64.urlsafe_b64decode(key.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{_ENV_VAR_NAME} key is invalid; expected url-safe base64 of "
            f"{_KEY_BYTES} bytes"
        ) from exc
    if len(raw) != _KEY_BYTES:
        raise ValueError(
            f"{_ENV_VAR_NAME} key must decode to {_KEY_BYTES} bytes (AES-256), "
            f"got {len(raw)}"
        )
    return raw


class CredentialCipher:
    """Encrypt/decrypt camera credentials with AES-256-GCM.

    Construct it directly from a key string, from the environment via
    :meth:`from_env`, or in tests from a freshly generated key via
    :meth:`generate_key`. The key can be rotated and credentials re-encrypted
    without schema changes (HLD 14).
    """

    def __init__(self, key: str) -> None:
        """Build a cipher from a url-safe base64 32-byte ``key``.

        Args:
            key: A url-safe base64-encoded 32-byte key (see :meth:`generate_key`).

        Raises:
            ValueError: If ``key`` is empty or not a valid 32-byte key.
        """
        if not key:
            raise ValueError(
                f"{_ENV_VAR_NAME} key is missing; generate one with "
                "CredentialCipher.generate_key()"
            )
        self._aesgcm = AESGCM(_decode_key(key))

    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt a credential and return ``nonce || ciphertext+tag``.

        A fresh random nonce is used per call, so encrypting the same credential
        twice yields different tokens.

        Args:
            plaintext: The secret to protect. Held only transiently; never logged.

        Returns:
            The token (12-byte nonce followed by the GCM ciphertext+tag) as bytes.
        """
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return nonce + ciphertext

    def decrypt(self, token: bytes) -> str:
        """Decrypt a token produced by :meth:`encrypt` back to the credential.

        Args:
            token: A ``nonce || ciphertext+tag`` token from :meth:`encrypt`.

        Returns:
            The recovered plaintext credential.

        Raises:
            ValueError: If the token is malformed, tampered with, or was produced
                with a different key.
        """
        if len(token) <= _NONCE_BYTES:
            raise ValueError("invalid credential token: too short")
        nonce, ciphertext = token[:_NONCE_BYTES], token[_NONCE_BYTES:]
        try:
            return self._aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
        except InvalidTag as exc:
            # Never log the token or any derived plaintext.
            logger.error("Failed to decrypt camera credential: invalid or tampered token")
            raise ValueError("invalid or tampered credential token") from exc

    @classmethod
    def from_env(cls) -> CredentialCipher:
        """Build a cipher from the ``CAMERA_CREDENTIALS_KEY`` environment secret.

        Returns:
            A configured :class:`CredentialCipher`.

        Raises:
            RuntimeError: If ``CAMERA_CREDENTIALS_KEY`` is unset.
            ValueError: If the configured key is invalid.
        """
        key = get_secrets().camera_credentials_key
        if not key:
            raise RuntimeError(
                f"{_ENV_VAR_NAME} is not set; camera credential encryption is "
                "unavailable. Set it to a key from CredentialCipher.generate_key()."
            )
        return cls(key)

    @staticmethod
    def generate_key() -> str:
        """Generate a new url-safe base64 32-byte (AES-256) key for ``from_env``."""
        return base64.urlsafe_b64encode(os.urandom(_KEY_BYTES)).decode("utf-8")
