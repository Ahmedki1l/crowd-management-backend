"""Runtime config API schema (GET/PUT /config, HLD 8.1)."""

from __future__ import annotations

from pydantic import BaseModel

from app.config.schema import (
    DetectorConfig,
    ProcessingConfig,
    StateMachineConfig,
    TrackerConfig,
)


class RuntimeConfigOut(BaseModel):
    processing: ProcessingConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    state_machine: StateMachineConfig


class RuntimeConfigUpdate(BaseModel):
    processing: ProcessingConfig | None = None
    detector: DetectorConfig | None = None
    tracker: TrackerConfig | None = None
    state_machine: StateMachineConfig | None = None
