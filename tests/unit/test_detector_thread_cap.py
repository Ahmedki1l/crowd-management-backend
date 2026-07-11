"""OpenVINO per-model thread cap: config plumbing + the compile_model wrap.

The cap stops N per-camera detectors from each grabbing all cores (which makes
them serialise). These tests use a FAKE openvino module so they run on the core
dependency set — the real backend is never imported.
"""

from __future__ import annotations

import sys
import types

import pytest

import app.inference.detector as detector_mod
from app.config.schema import DetectorConfig


@pytest.fixture
def _reset_cap_state():
    """Isolate the module-global 'installed' flag around each test."""
    saved = detector_mod._ov_thread_cap_installed
    detector_mod._ov_thread_cap_installed = False
    saved_ov = sys.modules.get("openvino")
    try:
        yield
    finally:
        detector_mod._ov_thread_cap_installed = saved
        if saved_ov is not None:
            sys.modules["openvino"] = saved_ov
        else:
            sys.modules.pop("openvino", None)


def _fake_openvino():
    """A stand-in openvino module whose Core.compile_model records its config."""
    calls: list[dict] = []

    class Core:
        def compile_model(self, model, device_name=None, config=None, **kwargs):
            calls.append({"device_name": device_name, "config": dict(config or {})})
            return object()

    mod = types.ModuleType("openvino")
    mod.Core = Core
    mod._calls = calls
    return mod


def test_config_default_is_zero() -> None:
    assert DetectorConfig().ov_inference_num_threads == 0


def test_zero_is_a_noop_and_never_imports_openvino(_reset_cap_state) -> None:
    # No fake injected: if it tried to import openvino it would use the real one;
    # the early return means nothing is installed.
    detector_mod._install_openvino_thread_cap(0)
    assert detector_mod._ov_thread_cap_installed is False


def test_cap_injects_inference_num_threads(_reset_cap_state) -> None:
    fake = _fake_openvino()
    sys.modules["openvino"] = fake

    detector_mod._install_openvino_thread_cap(3)
    core = fake.Core()
    core.compile_model("model", device_name="CPU", config={"PERFORMANCE_HINT": "LATENCY"})

    assert fake._calls[-1]["config"]["INFERENCE_NUM_THREADS"] == "3"
    # existing config is preserved, not clobbered
    assert fake._calls[-1]["config"]["PERFORMANCE_HINT"] == "LATENCY"


def test_cap_respects_an_explicit_caller_value(_reset_cap_state) -> None:
    fake = _fake_openvino()
    sys.modules["openvino"] = fake

    detector_mod._install_openvino_thread_cap(3)
    fake.Core().compile_model("m", config={"INFERENCE_NUM_THREADS": "8"})

    assert fake._calls[-1]["config"]["INFERENCE_NUM_THREADS"] == "8"  # not overridden


def test_install_is_idempotent(_reset_cap_state) -> None:
    fake = _fake_openvino()
    sys.modules["openvino"] = fake
    first = fake.Core.compile_model

    detector_mod._install_openvino_thread_cap(3)
    wrapped_once = fake.Core.compile_model
    detector_mod._install_openvino_thread_cap(3)
    wrapped_twice = fake.Core.compile_model

    assert wrapped_once is wrapped_twice  # second install is a no-op
    assert wrapped_once is not first
