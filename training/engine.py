# Random state
"""Explicit RNG capture and runtime settings, including supported MPS generator state."""

import random
from contextlib import contextmanager

import numpy as np
import torch


def capture_rng(device: torch.device) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_mps": torch.mps.get_rng_state().cpu().clone() if device.type == "mps" else None,
    }


def restore_rng(state: dict, device: torch.device) -> None:
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch_cpu", "torch_mps"}:
        raise ValueError("Unsupported or incomplete RNG state; resume refused.")
    if (state["torch_mps"] is not None) != (device.type == "mps"):
        raise ValueError("RNG backend differs; accelerator state will not be discarded.")
    try:
        random.setstate(state["python"])
        array = state["numpy"]
        np.random.set_state((array[0], np.asarray(array[1], dtype=np.uint32), *array[2:]))
        torch.set_rng_state(state["torch_cpu"])
        if device.type == "mps":
            torch.mps.set_rng_state(state["torch_mps"])
    except (TypeError, ValueError, RuntimeError, IndexError) as exc:
        raise ValueError(f"Invalid saved RNG state: {exc}.") from exc


@contextmanager
def execution_settings(config, device):
    previous_rng = capture_rng(device)
    threads = torch.get_num_threads()
    dtype = torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(config.cpu_threads)
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(config.deterministic_algorithms)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        yield
    finally:
        restore_rng(previous_rng, device)
        torch.set_num_threads(threads)
        torch.set_default_dtype(dtype)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


# Optimization
"""AdamW, an update-indexed warmup/cosine schedule, and token-weighted accumulation."""

import math
import time

import torch


def make_optimizer(model, config):
    matrices, others = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (matrices if parameter.ndim >= 2 else others).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": matrices, "weight_decay": config.weight_decay},
            {"params": others, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )


def validate_optimizer_state(optimizer, state, completed_updates, expected_lr):
    """Reject missing moments, unsupported options, and inconsistent Adam update counters."""
    expected = optimizer.state_dict()
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("Unsupported optimizer checkpoint state.")
    if not isinstance(state["state"], dict):
        raise ValueError("Invalid optimizer parameter history.")
    groups = state["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(expected["param_groups"]):
        raise ValueError("Incompatible optimizer parameter groups.")
    expected_ids = set()
    for saved, original, live in zip(
        groups, expected["param_groups"], optimizer.param_groups, strict=True
    ):
        if not isinstance(saved, dict) or set(saved) != set(original):
            raise ValueError("Unsupported optimizer group options.")
        if any(saved[key] != value for key, value in original.items() if key != "lr"):
            raise ValueError("Incompatible optimizer options or parameter ordering.")
        if saved["lr"] != expected_lr:
            raise ValueError("Saved optimizer learning rate disagrees with the schedule.")
        for number, parameter in zip(saved["params"], live["params"], strict=True):
            if not completed_updates:
                continue
            expected_ids.add(number)
            item = state["state"].get(number)
            if not isinstance(item, dict) or set(item) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Missing or unsupported optimizer moment history.")
            for key in ("exp_avg", "exp_avg_sq"):
                value = item[key]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid optimizer {key} tensor.")
            step = item["step"]
            if (
                not isinstance(step, torch.Tensor)
                or step.ndim != 0
                or step.dtype != torch.float32
                or step.item() != completed_updates
            ):
                raise ValueError("Optimizer step count disagrees with training position.")
    if not isinstance(state["state"], dict) or set(state["state"]) != expected_ids:
        raise ValueError("Missing or unexpected optimizer parameter state.")


class WarmupCosineSchedule:
    def __init__(self, config) -> None:
        self.settings = {
            "total_updates": config.total_updates,
            "warmup_updates": config.warmup_updates,
            "peak_lr": config.learning_rate,
            "minimum_lr_ratio": config.minimum_lr_ratio,
        }
        self.completed_updates = 0

    def learning_rate(self, update: int) -> float:
        total, warmup = self.settings["total_updates"], self.settings["warmup_updates"]
        if not 1 <= update <= total:
            raise ValueError("Schedule update is outside its original training budget.")
        peak = self.settings["peak_lr"]
        if update <= warmup:
            return peak * update / warmup
        progress = (
            (update - warmup) / (total - warmup) if warmup else (update - 1) / max(1, total - 1)
        )
        ratio = self.settings["minimum_lr_ratio"]
        return peak * (ratio + (1 - ratio) * (1 + math.cos(math.pi * progress)) / 2)

    def state_dict(self) -> dict:
        return {"settings": self.settings.copy(), "completed_updates": self.completed_updates}

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != {"settings", "completed_updates"}:
            raise ValueError("Unsupported scheduler state.")
        if state["settings"] != self.settings:
            raise ValueError("Resume would change the learning-rate schedule.")
        count = state["completed_updates"]
        if type(count) is not int or not 0 <= count <= self.settings["total_updates"]:
            raise ValueError("Invalid scheduler update position.")
        self.completed_updates = count


def batch_from_indices(dataset, indices, device):
    pairs = [dataset[index] for index in indices]
    inputs, targets = zip(*pairs, strict=True)
    return torch.stack(inputs).to(device), torch.stack(targets).to(device)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def optimizer_update(model, optimizer, schedule, dataset, sampler, config, windows, device):
    def batches():
        remaining = windows
        while remaining:
            count = min(config.microbatch_size, remaining)
            yield batch_from_indices(dataset, sampler.take(count), device)
            remaining -= count

    return optimizer_update_batches(model, optimizer, schedule, batches(), config, windows, device)


def optimizer_update_batches(model, optimizer, schedule, batches, config, windows, device):
    """Shared update math; a resident batch iterable also permits compute-only measurements."""
    if type(windows) is not int or windows < 1:
        raise ValueError("An optimizer cycle requires positive window count.")
    model.train()
    # Purane gradients clear karo; har cycle mein sirf is update ka contribution chahiye.
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    started = time.perf_counter()
    loss_sum, remaining = 0.0, windows
    for inputs, targets in batches:
        count = inputs.shape[0]
        if count < 1 or count > remaining or count > config.microbatch_size:
            raise ValueError("Batch count disagrees with the configured accumulation cycle.")
        loss = model(inputs, targets).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite or missing training loss; this partial update was not checkpointed."
            )
        # Partial cycle ko actual token count se weight karo, fixed accumulation count se nahi.
        (loss * (count / windows)).backward()
        loss_sum += loss.detach().item() * count
        remaining -= count
    if remaining:
        raise ValueError("Incomplete batch iterable; optimizer update refused.")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError("Missing or non-finite accumulated gradients; optimizer update refused.")
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, config.gradient_clip_norm, error_if_nonfinite=True, foreach=False
    )
    learning_rate = schedule.learning_rate(schedule.completed_updates + 1)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        raise RuntimeError(
            "Non-finite parameter after optimizer step; previous checkpoint preserved."
        )
    for state in optimizer.state.values():
        if any(
            isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
            for value in state.values()
        ):
            raise RuntimeError("Non-finite optimizer state; previous checkpoint preserved.")
    schedule.completed_updates += 1
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    return {
        "training_loss": loss_sum / windows,
        "learning_rate": learning_rate,
        "gradient_norm": norm.item(),
        "update_seconds": time.perf_counter() - started,
        "cycle_tokens": windows * config.model.context_length,
    }


# Checkpoint manager
"""Atomic checkpoint publication, exclusive run ownership, and recoverable JSONL logs."""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch

from transformer.data.preparation import file_sha256


@contextmanager
def run_lock(directory: Path):
    directory.parent.mkdir(parents=True, exist_ok=True)
    with directory.with_name(f".{directory.name}.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer owns run {directory.name}.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(destination: Path, writer) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            writer(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(destination: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_write(destination, lambda output: output.write(payload))


class MetricsJournal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: dict) -> None:
        payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        with self.path.open("ab") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())

    def state_dict(self) -> dict:
        return {"bytes": self.path.stat().st_size, "sha256": file_sha256(self.path)}

    def recover(self, state: dict) -> str | None:
        if (
            not isinstance(state, dict)
            or set(state) != {"bytes", "sha256"}
            or type(state["bytes"]) is not int
            or state["bytes"] < 0
        ):
            raise ValueError("Unsupported metrics journal state.")
        if not self.path.exists() or self.path.stat().st_size < state["bytes"]:
            raise ValueError("Metrics journal is missing committed bytes; resume refused.")
        digest = hashlib.sha256()
        with self.path.open("rb") as source:
            remaining = state["bytes"]
            while remaining:
                chunk = source.read(min(remaining, 1 << 20))
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != state["sha256"]:
                raise ValueError("Committed metrics checksum differs; resume refused.")
            if self.path.stat().st_size == state["bytes"]:
                return None
            archive = self.path.with_name(f"metrics.uncommitted-{uuid.uuid4().hex}.jsonl")
            with archive.open("xb") as output:
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
        # Crash ke baad checkpoint se aage ka log archive karo, phir committed boundary par lautao.
        with self.path.open("r+b") as output:
            output.truncate(state["bytes"])
            output.flush()
            os.fsync(output.fileno())
        return archive.name


class CheckpointManager:
    def __init__(self, run_directory: Path) -> None:
        self.directory = run_directory / "checkpoints"
        self.index_path = run_directory / "checkpoints.json"

    def index(self) -> dict:
        try:
            value = json.loads(self.index_path.read_text())
            if (
                set(value) != {"format_version", "latest", "best"}
                or type(value["format_version"]) is not int
                or value["format_version"] != 1
            ):
                raise ValueError("Unsupported checkpoint index.")
            for label in ("latest", "best"):
                entry = value[label]
                if entry is None and label == "best":
                    continue
                if (
                    not isinstance(entry, dict)
                    or set(entry) != {"file", "sha256"}
                    or not isinstance(entry["file"], str)
                    or Path(entry["file"]).name != entry["file"]
                    or not entry["file"].endswith(".pt")
                ):
                    raise ValueError(f"Invalid {label} checkpoint reference.")
            return value
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot read checkpoint index: {exc}.") from exc

    def load(self, which: str = "latest") -> dict:
        if which not in ("latest", "best"):
            raise ValueError("Checkpoint choice must be latest or best.")
        index = self.index()
        for reference in (index["latest"], index["best"]):
            if reference is not None:
                referenced_path = self.directory / reference["file"]
                if (
                    referenced_path.is_symlink()
                    or file_sha256(referenced_path) != reference["sha256"]
                ):
                    raise ValueError(
                        "Checkpoint checksum mismatch; latest/best references preserved."
                    )
        entry = index[which]
        if entry is None:
            raise ValueError("No best checkpoint exists until validation has run.")
        path = self.directory / entry["file"]
        if path.is_symlink() or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch for {which}.")
        return torch.load(path, map_location="cpu", weights_only=True)

    def commit(self, payload: dict, *, improved: bool) -> None:
        self.directory.mkdir(exist_ok=True)
        previous = self.index() if self.index_path.exists() else {"latest": None, "best": None}
        name = f"update-{payload['position']['updates']:09d}-{uuid.uuid4().hex}.pt"
        destination = self.directory / name
        atomic_write(destination, lambda output: torch.save(payload, output))
        entry = {"file": name, "sha256": file_sha256(destination)}
        index = {
            "format_version": 1,
            "latest": entry,
            "best": entry if improved else previous["best"],
        }
        # Ek atomic index latest aur best dono ko saath publish karta hai.
        atomic_json(self.index_path, index)
        retained = {item["file"] for item in (index["latest"], index["best"]) if item}
        for old in (previous["latest"], previous["best"]):
            if old and old["file"] not in retained:
                (self.directory / old["file"]).unlink(missing_ok=True)


# Validation
"""Sequential validation-only evaluation with token-weighted loss and restored mode."""

import torch



def evaluate(model, dataset, microbatch_size: int, max_batches: int, device) -> dict:
    previous_mode = model.training
    loss_sum, tokens = 0.0, 0
    model.eval()
    try:
        with torch.inference_mode():
            for batch, start in enumerate(range(0, len(dataset), microbatch_size)):
                if batch == max_batches:
                    break
                indices = range(start, min(start + microbatch_size, len(dataset)))
                inputs, targets = batch_from_indices(dataset, indices, device)
                loss = model(inputs, targets).loss
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("Validation loss is missing or non-finite.")
                loss_sum += loss.item() * targets.numel()
                tokens += targets.numel()
    finally:
        model.train(previous_mode)
    if not tokens:
        raise ValueError("Validation requires at least one complete token window.")
    return {"validation_loss": loss_sum / tokens, "validation_tokens": tokens}


# Training loop
"""Configurable training orchestration; checkpoints represent only complete optimizer updates."""

import json
import math
import os
import platform
import signal
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch

from transformer.configuration.settings import PathConfig
from transformer.configuration.settings import TrainingConfig
from transformer.data.preparation import file_sha256, fingerprint, write_json
from transformer.data.loading import TokenWindowDataset
from transformer.model import TransformerLanguageModel
from transformer.runtime.device import select_device
from transformer.tokenization.character_tokenizer import CharacterTokenizer
from transformer.tokenization.token_preparation import read_token_manifest
from transformer.data.loading import WindowSampler


@contextmanager
def stop_on_interrupt():
    state = {"requested": False}
    active = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGINT) if active else None

    def request_stop(signum, frame):
        state["requested"] = True

    if active:
        signal.signal(signal.SIGINT, request_stop)
    try:
        yield state
    finally:
        if active:
            signal.signal(signal.SIGINT, previous)


def _implementation_identity() -> str:
    package = Path(__file__).resolve().parents[1]
    files = list((package / "training").glob("*.py")) + list((package / "model").glob("*.py"))
    files += [
        package / "configuration/settings.py",
        package / "configuration/settings.py",
        package / "data/loading.py",
    ]
    return fingerprint(
        {str(path.relative_to(package)): file_sha256(path) for path in sorted(files)}
    )


def _identity(config, device, manifest) -> dict:
    return {
        "configuration": config.to_dict(),
        "implementation": _implementation_identity(),
        "data": {
            "tokenizer_fingerprint": config.tokenizer_fingerprint,
            "dataset_fingerprint": config.dataset_fingerprint,
            "token_data_fingerprint": config.token_data_fingerprint,
            "train_stream": config.train_stream,
            "validation_stream": "validation",
            "streams": {
                name: manifest["streams"][name] for name in (config.train_stream, "validation")
            },
            "prepared_sources": manifest["prepared_sources"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "system": platform.platform(),
            "machine": platform.machine(),
            "device": str(device),
            "cpu_threads": config.cpu_threads,
            "deterministic_algorithms": config.deterministic_algorithms,
            "model_dtype": "float32",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1",
        },
    }


def _validate_position(position, config, sampler, schedule):
    required = {
        "updates",
        "tokens",
        "best_validation",
        "last_validation",
        "elapsed_seconds",
        "training_seconds",
    }
    if not isinstance(position, dict) or set(position) != required:
        raise ValueError("Unsupported or incomplete training position.")
    update, tokens = position["updates"], position["tokens"]
    if (
        type(update) is not int
        or not 0 <= update <= config.total_updates
        or type(tokens) is not int
    ):
        raise ValueError("Invalid checkpoint update/token count.")
    if (
        tokens != min(update * config.tokens_per_update, config.total_tokens)
        or schedule.completed_updates != update
    ):
        raise ValueError("Checkpoint scheduler, update and token positions disagree.")
    if (sampler.epoch * sampler.size + sampler.cursor) * config.model.context_length != tokens:
        raise ValueError("Sampler position disagrees with processed tokens.")
    for name in ("elapsed_seconds", "training_seconds"):
        if (
            type(position[name]) not in (float, int)
            or not math.isfinite(position[name])
            or position[name] < 0
        ):
            raise ValueError(f"Invalid checkpoint {name}.")
    for name in ("best_validation", "last_validation"):
        value = position[name]
        if value is not None and (
            not isinstance(value, dict)
            or set(value) != {"loss", "update", "tokens"}
            or not math.isfinite(value["loss"])
            or value["loss"] < 0
            or type(value["update"]) is not int
            or not 1 <= value["update"] <= update
            or type(value["tokens"]) is not int
            or value["tokens"] < 1
        ):
            raise ValueError(f"Invalid checkpoint {name}.")


def _restore(payload, identity, model, optimizer, schedule, sampler, config, device):
    required = {
        "format_version",
        "identity",
        "model",
        "optimizer",
        "scheduler",
        "sampler",
        "rng",
        "position",
        "metrics",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or type(payload["format_version"]) is not int
        or payload["format_version"] != 1
    ):
        raise ValueError("Unsupported or incomplete training checkpoint; no state was discarded.")
    if payload["identity"] != identity:
        raise ValueError(
            "Incompatible resume configuration, data, implementation, or environment; "
            "original schedule preserved."
        )
    if not isinstance(payload["model"], dict) or any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
        for value in payload["model"].values()
    ):
        raise ValueError(
            "Invalid model weights: training recovery requires finite float32 tensors."
        )
    model.load_state_dict(payload["model"], strict=True)
    schedule.load_state_dict(payload["scheduler"])
    sampler.load_state_dict(payload["sampler"])
    _validate_position(payload["position"], config, sampler, schedule)
    completed = schedule.completed_updates
    expected_lr = schedule.learning_rate(completed) if completed else config.learning_rate
    validate_optimizer_state(optimizer, payload["optimizer"], completed, expected_lr)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"], device)
    return payload["position"]


def _execute(
    config,
    identity,
    directory,
    train_data,
    validation_data,
    device,
    resume,
    stop_after_updates,
    stop,
    observer=None,
):
    model = TransformerLanguageModel(config.model).to(device)
    optimizer = make_optimizer(model, config)
    schedule = WarmupCosineSchedule(config)
    sampler = WindowSampler(len(train_data), config.seed, config.shuffle)
    manager, journal = CheckpointManager(directory), MetricsJournal(directory / "metrics.jsonl")
    position = {
        "updates": 0,
        "tokens": 0,
        "best_validation": None,
        "last_validation": None,
        "elapsed_seconds": 0.0,
        "training_seconds": 0.0,
    }
    if resume:
        payload = manager.load()
        position = _restore(payload, identity, model, optimizer, schedule, sampler, config, device)
        if stop_after_updates is not None and stop_after_updates <= position["updates"]:
            raise ValueError("stop_after_updates must be later than the saved update.")
        archived = journal.recover(payload["metrics"])
        if position["updates"] == config.total_updates:
            return {"status": "completed", "run_directory": str(directory), **position}
        journal.append(
            {
                "event": "resume",
                "update": position["updates"],
                "archived_uncommitted_metrics": archived,
            }
        )
    else:
        journal.append(
            {
                "event": "start",
                "update": 0,
                "total_updates": config.total_updates,
                "total_tokens": config.total_tokens,
            }
        )

    started, elapsed_before = time.perf_counter(), position["elapsed_seconds"]

    def commit(improved=False):
        checkpoint_started = time.perf_counter()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Checkpoint refused outside a cleared optimizer-update boundary.")
        position["elapsed_seconds"] = elapsed_before + time.perf_counter() - started
        manager.commit(
            {
                "format_version": 1,
                "identity": identity,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": schedule.state_dict(),
                "sampler": sampler.state_dict(),
                "rng": capture_rng(device),
                "position": position.copy(),
                "metrics": journal.state_dict(),
            },
            improved=improved,
        )
        return time.perf_counter() - checkpoint_started

    if not resume:
        commit()
    if observer is not None:
        observer(
            {
                "event": "ready",
                "optimizer_update": position["updates"],
                "tokens_processed": position["tokens"],
            }
        )
    improved = False
    last_observation = None
    last_notified_update = None
    limit = stop_after_updates if stop_after_updates is not None else config.total_updates
    try:
        while position["updates"] < limit and not stop["requested"]:
            remaining_windows = (
                config.total_tokens - position["tokens"]
            ) // config.model.context_length
            windows = min(
                config.microbatch_size * config.gradient_accumulation_steps, remaining_windows
            )
            result = optimizer_update(
                model, optimizer, schedule, train_data, sampler, config, windows, device
            )
            position["updates"] = schedule.completed_updates
            position["tokens"] += result["cycle_tokens"]
            position["training_seconds"] += result["update_seconds"]
            update = position["updates"]
            validation = {"validation_loss": None, "validation_tokens": 0}
            validation_seconds = 0.0
            if update % config.evaluation_interval == 0 or update == config.total_updates:
                validation_started = time.perf_counter()
                validation = evaluate(
                    model,
                    validation_data,
                    config.microbatch_size,
                    config.max_validation_batches,
                    device,
                )
                validation_seconds = time.perf_counter() - validation_started
                current = {
                    "loss": validation["validation_loss"],
                    "tokens": validation["validation_tokens"],
                    "update": update,
                }
                position["last_validation"] = current
                if (
                    position["best_validation"] is None
                    or current["loss"] < position["best_validation"]["loss"]
                ):
                    position["best_validation"], improved = current, True
            journal.append(
                {
                    "event": "update",
                    "optimizer_update": update,
                    "tokens_processed": position["tokens"],
                    **result,
                    **validation,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                    "tokens_per_second": result["cycle_tokens"] / result["update_seconds"],
                    "cumulative_training_tokens_per_second": position["tokens"]
                    / position["training_seconds"],
                }
            )
            if update % config.console_interval == 0 or validation["validation_loss"] is not None:
                print(
                    f"update {update}/{config.total_updates} | tokens {position['tokens']} "
                    f"| loss {result['training_loss']:.4f} | lr {result['learning_rate']:.3g} "
                    f"| validation {validation['validation_loss']}",
                    file=sys.stderr,
                )
            ending = update == limit or stop["requested"]
            checkpoint_seconds = 0.0
            if not ending and (
                update % config.checkpoint_interval == 0
                or validation["validation_loss"] is not None
            ):
                checkpoint_seconds = commit(improved)
                improved = False
            last_observation = {
                "event": "update",
                "optimizer_update": update,
                "tokens_processed": position["tokens"],
                **result,
                **validation,
                "validation_seconds": validation_seconds,
                "checkpoint_seconds": checkpoint_seconds,
            }
            if observer is not None and not ending:
                observer(last_observation)
                last_notified_update = update
        status = (
            "completed"
            if position["updates"] == config.total_updates
            else ("interrupted" if stop["requested"] else "stopped")
        )
        journal.append(
            {"event": status, "update": position["updates"], "tokens_processed": position["tokens"]}
        )
        checkpoint_seconds = commit(improved)
        if (
            observer is not None
            and last_observation is not None
            and last_observation["optimizer_update"] != last_notified_update
        ):
            last_observation["checkpoint_seconds"] += checkpoint_seconds
            observer(last_observation)
    except KeyboardInterrupt:
        # Unexpected exception mein optimizer aadha badal sakta hai; disk checkpoint hi safe hai.
        position = manager.load()["position"]
        status = "interrupted"
        print(
            "Interrupted outside the deferred SIGINT path; "
            "only the last committed checkpoint is recoverable.",
            file=sys.stderr,
        )
    return {"status": status, "run_directory": str(directory), **position}


def run_training(
    config: TrainingConfig,
    project_root: Path,
    paths: PathConfig | None = None,
    *,
    resume: bool = False,
    stop_after_updates: int | None = None,
    dry_run: bool = False,
    observer=None,
) -> dict:
    root = Path(project_root).resolve()
    directory = (paths or PathConfig()).resolve(root)["runs_dir"] / config.run_id
    if stop_after_updates is not None and (
        type(stop_after_updates) is not int or not 1 <= stop_after_updates <= config.total_updates
    ):
        raise ValueError("stop_after_updates must be an update within the original total budget.")
    if directory.exists() and not resume:
        raise FileExistsError(
            f"Run already exists: {directory}. "
            "Use --resume with identical settings or choose a new run_id."
        )
    if resume and not directory.is_dir():
        raise ValueError("Cannot resume: run directory does not exist.")
    selection = select_device(config.device)
    tokenizer = CharacterTokenizer.load(
        root / config.tokenizer_path, expected_fingerprint=config.tokenizer_fingerprint
    )
    if tokenizer.vocab_size != config.model.vocab_size:
        raise ValueError("Model vocabulary size differs from the frozen tokenizer.")
    token_directory = root / config.token_data_dir
    manifest = read_token_manifest(
        token_directory,
        config.tokenizer_fingerprint,
        expected_dataset_fingerprint=config.dataset_fingerprint,
    )
    if manifest["token_data_fingerprint"] != config.token_data_fingerprint:
        raise ValueError("Incompatible token-data fingerprint.")
    identity = _identity(config, selection.device, manifest)
    if resume:
        saved = json.loads((directory / "run.json").read_text())
        if saved != identity:
            raise ValueError(
                "Incompatible resume configuration, data, implementation, or environment."
            )
    with ExitStack() as stack:

        def dataset(stream):
            return stack.enter_context(
                TokenWindowDataset(
                    token_directory,
                    stream,
                    config.tokenizer_fingerprint,
                    context_length=config.model.context_length,
                    dataset_fingerprint=config.dataset_fingerprint,
                )
            )

        train_data, validation_data = dataset(config.train_stream), dataset("validation")
        if dry_run:
            with torch.device("meta"):
                count = TransformerLanguageModel(config.model).trainable_parameter_count()
            return {
                "status": "dry_run",
                "run_directory": str(directory),
                "device": str(selection.device),
                "total_updates": config.total_updates,
                "total_training_tokens": config.total_tokens,
                "tokens_per_full_update": config.tokens_per_update,
                "training_windows": len(train_data),
                "validation_windows": len(validation_data),
                "trainable_parameters": count,
                "checkpoint_bytes_rough_estimate": 12 * count + 8 * len(train_data),
                "checkpoint_peak_copies": 3,
                "estimate_excludes": "serialization/log/filesystem overhead and crash orphans",
            }
        stop = stack.enter_context(stop_on_interrupt())
        stack.enter_context(run_lock(directory))
        if not resume:
            directory.mkdir(exist_ok=False)
            config.save(directory / "configuration.json")
            write_json(directory / "run.json", identity)
        stack.enter_context(execution_settings(config, selection.device))
        return _execute(
            config,
            identity,
            directory,
            train_data,
            validation_data,
            selection.device,
            resume,
            stop_after_updates,
            stop,
            observer,
        )
