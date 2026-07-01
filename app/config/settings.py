"""Configuration loading: YAML (+ ``${ENV}`` expansion) merged with environment.

``get_settings()`` returns a cached :class:`~app.config.schema.AppConfig`.
``get_secrets()`` returns env-only secrets (never read from YAML) such as the
camera-credential encryption key.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

from app.config.schema import AppConfig

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` with the environment value (or "" if unset)."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _prune_empty_strings(value: Any) -> Any:
    """Drop empty strings so pydantic defaults apply for unset ${ENV} values."""
    if isinstance(value, dict):
        return {k: _prune_empty_strings(v) for k, v in value.items() if v != ""}
    if isinstance(value, list):
        return [_prune_empty_strings(v) for v in value]
    return value


def load_config(config_path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load and validate the YAML config, expanding environment variables."""
    path = Path(config_path or os.environ.get("CONFIG_PATH", "config/config.example.yaml"))
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    merged = _prune_empty_strings(_expand_env(raw))
    config = AppConfig.model_validate(merged)
    # Env overrides for a few values that may be set without touching YAML.
    if db_url := os.environ.get("DATABASE_URL"):
        config.database.url = db_url
    if detector_path := os.environ.get("MODEL_DETECTOR_PATH"):
        config.detector.model_path = detector_path
    if reid_path := os.environ.get("MODEL_REID_PATH"):
        config.tracker.reid_model_path = reid_path
    return config


class Secrets(BaseModel):
    """Env-only secrets. Never sourced from YAML or returned by any API route."""

    camera_credentials_key: str | None = None
    api_auth_secret: str = "change-me"
    dt_auth_header: str | None = None


def load_secrets() -> Secrets:
    return Secrets(
        camera_credentials_key=os.environ.get("CAMERA_CREDENTIALS_KEY") or None,
        api_auth_secret=os.environ.get("API_AUTH_SECRET", "change-me"),
        dt_auth_header=os.environ.get("DT_AUTH_HEADER") or None,
    )


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    return load_config()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return load_secrets()


def reset_settings_cache() -> None:
    """Clear cached settings/secrets (used by config reload and tests)."""
    get_settings.cache_clear()
    get_secrets.cache_clear()
