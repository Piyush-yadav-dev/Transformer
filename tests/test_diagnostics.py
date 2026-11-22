"""Verify factual reporting and failure propagation without device substitution."""

import platform

import pytest
import torch

from transformer.runtime import device as diagnostics
from transformer.runtime.device import select_device


def test_diagnostic_report_without_execution():
    report = diagnostics.collect_diagnostics(select_device("cpu")).to_dict()
    assert report["python_version"] == platform.python_version()
    assert report["pytorch_version"] == torch.__version__
    assert report["operating_system"] == platform.platform()
    assert report["machine_architecture"] == platform.machine()
    assert report["selected_device"] == "cpu"
    assert report["mps_build_support"] == torch.backends.mps.is_built()
    assert report["mps_runtime_available"] == torch.backends.mps.is_available()
    assert report["tensor_check"] == "not_run"


def test_failed_tensor_check_is_not_reported_as_success(monkeypatch):
    checked_devices = []

    def failing_check(device):
        checked_devices.append(device.type)
        raise RuntimeError("simulated backend error")

    monkeypatch.setattr(diagnostics, "check_tensor_operations", failing_check)
    with pytest.raises(RuntimeError, match="Tensor check failed on cpu: simulated backend error"):
        diagnostics.collect_diagnostics(select_device("cpu"), check=True)
    assert checked_devices == ["cpu"]
