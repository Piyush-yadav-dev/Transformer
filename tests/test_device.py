"""Simulated policy coverage is separate from actual device execution."""

import pytest
import torch

from transformer.runtime.device import DeviceUnavailableError, select_device
from transformer.runtime.device import check_tensor_operations, collect_diagnostics


def set_mps_status(monkeypatch, *, built, available):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: built)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)


@pytest.mark.parametrize("requested", ["auto", "mps"])
def test_available_mps_is_selected(monkeypatch, requested):
    set_mps_status(monkeypatch, built=True, available=True)
    selection = select_device(requested)
    assert selection.device.type == "mps"
    assert selection.requested == requested


@pytest.mark.parametrize("available", [True, False])
def test_explicit_cpu_stays_on_cpu(monkeypatch, available):
    set_mps_status(monkeypatch, built=True, available=available)
    assert select_device("cpu").device.type == "cpu"


@pytest.mark.parametrize("built", [True, False])
def test_auto_explains_cpu_fallback(monkeypatch, built):
    set_mps_status(monkeypatch, built=built, available=False)
    selection = select_device("auto")
    assert selection.device.type == "cpu"
    assert "Automatic selection uses CPU" in selection.reason
    assert ("runtime is unavailable" if built else "not built with MPS") in selection.reason


@pytest.mark.parametrize("built", [True, False])
def test_explicit_unavailable_mps_raises_without_fallback(monkeypatch, built):
    set_mps_status(monkeypatch, built=built, available=False)
    with pytest.raises(DeviceUnavailableError, match="MPS was explicitly requested.*--device cpu"):
        select_device("mps")


def test_invalid_device_is_rejected_before_backend_probe(monkeypatch):
    def unexpected_probe():
        pytest.fail("Invalid input should be rejected before probing hardware.")

    monkeypatch.setattr(torch.backends.mps, "is_built", unexpected_probe)
    with pytest.raises(ValueError, match="runtime.device"):
        select_device("invalid")


def test_real_cpu_tensor_operations():
    check_tensor_operations(torch.device("cpu"))


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS unavailable on this host; real target MacBook verification remains pending.",
)
def test_real_mps_tensor_operations():
    report = collect_diagnostics(select_device("mps"), check=True)
    assert report.selected_device == "mps"
    assert report.tensor_check == "passed"
