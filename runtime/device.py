# Device
"""Select CPU or MPS without silently changing an explicit device request."""

from dataclasses import dataclass

import torch

from transformer.configuration.settings import DevicePreference, RuntimeConfig


class DeviceUnavailableError(RuntimeError):
    """The requested accelerator is unavailable in this environment."""


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    requested: DevicePreference
    device: torch.device
    reason: str
    mps_built: bool
    mps_available: bool


def select_device(requested: DevicePreference = "auto") -> DeviceSelection:
    RuntimeConfig(device=requested)
    built = torch.backends.mps.is_built()
    available = torch.backends.mps.is_available()
    if requested == "cpu":
        return DeviceSelection(
            requested, torch.device("cpu"), "CPU explicitly requested.", built, available
        )
    if built and available:
        reason = (
            "MPS is available; automatic selection uses MPS."
            if requested == "auto"
            else "MPS explicitly requested and available."
        )
        return DeviceSelection(requested, torch.device("mps"), reason, built, available)

    unavailable_reason = (
        "This PyTorch installation was not built with MPS support."
        if not built
        else "PyTorch includes MPS support, but the MPS runtime is unavailable on this host."
    )
    if requested == "mps":
        raise DeviceUnavailableError(
            f"MPS was explicitly requested. {unavailable_reason} "
            "Check that you are using a supported macOS/Metal device and an MPS-enabled PyTorch "
            "installation. To intentionally use CPU, pass --device cpu."
        )
    return DeviceSelection(
        requested,
        torch.device("cpu"),
        f"{unavailable_reason} Automatic selection uses CPU.",
        built,
        available,
    )


# Diagnostics
"""Collect environment facts and optionally run a tiny tensor/autograd check."""

import platform
import sys
from dataclasses import asdict, dataclass

import torch



@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    python_version: str
    pytorch_version: str
    operating_system: str
    machine_architecture: str
    python_executable: str
    isolated_environment: bool
    mps_build_support: bool
    mps_runtime_available: bool
    requested_device: str
    selected_device: str
    selection_reason: str
    tensor_check: str

    def to_dict(self) -> dict:
        return asdict(self)


def check_tensor_operations(device: torch.device) -> None:
    """Check a 2x2 matrix product and its gradient; this is not a training benchmark."""
    with torch.enable_grad():
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
        result = (values @ values.T).sum()
        result.backward()
    if device.type == "mps":
        # MPS asynchronous hai; success bolne se pehle computation complete hone do.
        torch.mps.synchronize()
    torch.testing.assert_close(result.detach().cpu(), torch.tensor(52.0))
    torch.testing.assert_close(values.grad.cpu(), torch.tensor([[8.0, 12.0], [8.0, 12.0]]))


def collect_diagnostics(selection: DeviceSelection, *, check: bool = False) -> DiagnosticReport:
    if check:
        try:
            check_tensor_operations(selection.device)
        except (RuntimeError, AssertionError) as exc:
            raise RuntimeError(f"Tensor check failed on {selection.device}: {exc}") from exc
    return DiagnosticReport(
        python_version=platform.python_version(),
        pytorch_version=str(torch.__version__),
        operating_system=platform.platform(),
        machine_architecture=platform.machine(),
        python_executable=sys.executable,
        isolated_environment=sys.prefix != sys.base_prefix,
        mps_build_support=selection.mps_built,
        mps_runtime_available=selection.mps_available,
        requested_device=selection.requested,
        selected_device=str(selection.device),
        selection_reason=selection.reason,
        tensor_check="passed" if check else "not_run",
    )


# Model checks
"""One synthetic forward/backward check; no optimizer, parameter updates, or training loop."""

import torch

from transformer.configuration.settings import DevicePreference
from transformer.configuration.settings import TransformerConfig
from transformer.model.transformer import TransformerLanguageModel


def check_model(
    config: TransformerConfig, device: DevicePreference = "cpu", seed: int = 42
) -> dict:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    selection = select_device(device)
    torch.manual_seed(seed)
    model = TransformerLanguageModel(config).to(selection.device)
    model.train()
    sequence = min(8, config.context_length)
    stream = torch.randint(config.vocab_size, (2, sequence + 1), dtype=torch.int64)
    inputs = stream[:, :-1].to(selection.device)
    targets = stream[:, 1:].to(selection.device)
    output = model(inputs, targets)
    expected_shape = (2, sequence, config.vocab_size)
    if tuple(output.logits.shape) != expected_shape or not torch.isfinite(output.logits).all():
        raise RuntimeError(
            "Model check failed: logits have an incorrect shape or non-finite values."
        )
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("Model check failed: loss is missing or non-finite.")
    output.loss.backward()
    parameters = list(model.named_parameters())
    for name, parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Model check failed: missing or non-finite gradient for {name}.")
    return {
        "model": "Transformer",
        "configuration": config.to_dict(),
        "seed": seed,
        "selected_device": str(selection.device),
        "selection_reason": selection.reason,
        "torch_version": torch.__version__,
        "data": "synthetic random token IDs; not a quality evaluation",
        "input_shape": list(inputs.shape),
        "logits_shape": list(output.logits.shape),
        "loss": output.loss.detach().item(),
        "trainable_parameters": model.trainable_parameter_count(),
        "unique_parameter_tensors": len(parameters),
        "weights_tied": model.embeddings.token.weight is model.vocabulary_projection.weight,
        "forward_backward_check": "passed",
        "optimizer_steps": 0,
    }
