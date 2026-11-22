# Memory
"""Precisely named process and MPS counters; unified-memory counters must not be summed."""

import json
import os
import platform
import re
import resource
import subprocess

import torch


def _command(arguments):
    try:
        return subprocess.check_output(
            arguments, text=True, timeout=10, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_identity():
    result = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "machine_name": None,
        "machine_model": None,
        "chip_type": None,
        "physical_memory_bytes": None,
        "power_source": None,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if platform.system() == "Darwin":
        details = _command(["system_profiler", "SPHardwareDataType", "-json"])
        if details:
            info = json.loads(details)["SPHardwareDataType"][0]
            result.update(
                {key: info.get(key) for key in ("machine_name", "machine_model", "chip_type")}
            )
        size = _command(["sysctl", "-n", "hw.memsize"])
        result["physical_memory_bytes"] = int(size) if size else None
        power = _command(["pmset", "-g", "batt"])
        result["power_source"] = power.splitlines()[0] if power else None
    else:
        try:
            result["physical_memory_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf(
                "SC_PAGE_SIZE"
            )
        except (ValueError, OSError):
            pass
    result["matches_target_m2_air_16gb"] = (
        result["machine_name"] == "MacBook Air"
        and result["chip_type"] == "Apple M2"
        and result["physical_memory_bytes"] == 16 * 1024**3
    )
    return result


class MemoryMonitor:
    def __init__(self, device):
        self.device = device
        self.max_mps_live = self.max_mps_driver = None

    def capture_accelerator(self):
        if self.device.type != "mps":
            return {
                "mps_live_tensor_bytes": None,
                "mps_driver_allocated_bytes": None,
                "mps_recommended_working_set_bytes": None,
            }
        live = torch.mps.current_allocated_memory()
        driver = torch.mps.driver_allocated_memory()
        self.max_mps_live = max(self.max_mps_live or 0, live)
        self.max_mps_driver = max(self.max_mps_driver or 0, driver)
        return {
            "mps_live_tensor_bytes": live,
            "mps_driver_allocated_bytes": driver,
            "mps_recommended_working_set_bytes": torch.mps.recommended_max_memory(),
        }

    def snapshot(self, *, system=False):
        rss = _command(["ps", "-o", "rss=", "-p", str(os.getpid())])
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        pressure = (
            _command(["memory_pressure", "-Q"])
            if system and platform.system() == "Darwin"
            else None
        )
        match = re.search(r"System-wide memory free percentage:\s*([\d.]+)%", pressure or "")
        return {
            **self.capture_accelerator(),
            "process_current_rss_bytes": int(rss) * 1024 if rss else None,
            "process_lifetime_peak_rss_bytes": int(
                peak * (1 if platform.system() == "Darwin" else 1024)
            ),
            "system_memory_pressure_free_percent": float(match.group(1)) if match else None,
            "maximum_sampled_mps_live_tensor_bytes": self.max_mps_live,
            "maximum_sampled_mps_driver_bytes": self.max_mps_driver,
        }


def memory_limit_reason(snapshot, config):
    if snapshot["process_lifetime_peak_rss_bytes"] > config.max_process_rss_bytes:
        return "process RSS budget exceeded"
    driver = snapshot["maximum_sampled_mps_driver_bytes"]
    if driver is not None and driver > config.max_mps_driver_bytes:
        return "sampled MPS driver allocation budget exceeded"
    free = snapshot["system_memory_pressure_free_percent"]
    if free is not None and free < config.minimum_system_free_percent:
        return "system memory-pressure free percentage below reserved headroom threshold"
    return None


# Diagnostics
"""Bounded execution and learning diagnostics, explicitly separate from model-quality evaluation."""

import hashlib
import time
from pathlib import Path

import torch

from transformer.configuration.settings import TrainingConfig
from transformer.data.preparation import write_json
from transformer.model import TransformerLanguageModel
from transformer.runtime.device import select_device
from transformer.tokenization.character_tokenizer import CharacterTokenizer
from transformer.training.engine import CheckpointManager, MetricsJournal
from transformer.training.engine import WarmupCosineSchedule, make_optimizer, optimizer_update
from transformer.training.engine import execution_settings
from transformer.data.loading import WindowSampler
from transformer.training.engine import run_training
from transformer.training.engine import evaluate


def diagnostic_continuation(model, tokenizer, prompt: str, max_new_tokens: int = 48):
    """Internal greedy decoding only; not the public generation interface."""
    if not prompt or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("Diagnostic sampling requires a nonempty prompt and 1–128 new tokens.")
    previous_mode = model.training
    token_ids = tokenizer.encode(prompt, unknown="error")
    generated = []
    device = next(model.parameters()).device
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                # Context capacity ke andar aakhri tokens rakho; yeh sirf diagnostic sampling hai.
                inputs = torch.tensor(
                    [token_ids[-model.config.context_length :]], dtype=torch.int64, device=device
                )
                token = model(inputs).logits[0, -1].argmax().item()
                generated.append(token)
                token_ids.append(token)
                if token == tokenizer.eos_id:
                    break
    finally:
        model.train(previous_mode)
    return {
        "label": "DIAGNOSTIC ONLY: bounded smoke checkpoint, not evidence of language quality",
        "strategy": "greedy",
        "prompt": prompt,
        "generated_ids": generated,
        "continuation": tokenizer.decode(generated),
        "maximum_new_tokens": max_new_tokens,
        "stopped_at_eos": generated[-1] == tokenizer.eos_id,
    }


def run_smoke(config: TrainingConfig, root: Path, output: Path):
    if config.max_updates is None or not 2 <= config.max_updates <= 12:
        raise ValueError("Smoke test requires an explicit budget of 2–12 updates.")
    if (
        config.model.num_layers > 2
        or config.model.embedding_dim > 64
        or config.model.context_length > 64
    ):
        raise ValueError("Smoke tests require a small model (at most 2 layers, width/context 64).")
    output.mkdir(parents=True, exist_ok=False)
    config.save(output / "configuration.json")
    first = run_training(config, root, stop_after_updates=config.max_updates // 2)
    manager = CheckpointManager(Path(first["run_directory"]))
    loaded = manager.load()
    if loaded["position"]["updates"] != config.max_updates // 2:
        raise RuntimeError("Smoke checkpoint did not preserve the expected training position.")
    resumed = run_training(config, root, resume=True)
    latest = manager.load()
    model = TransformerLanguageModel(config.model)
    model.load_state_dict(latest["model"], strict=True)
    model.to(select_device(config.device).device)
    tokenizer = CharacterTokenizer.load(
        root / config.tokenizer_path, expected_fingerprint=config.tokenizer_fingerprint
    )
    continuation = diagnostic_continuation(model, tokenizer, "Once upon a time", 48)
    report = {
        "kind": "pipeline_smoke_test",
        "status": "passed",
        "not_model_quality_evaluation": True,
        "first_session": first,
        "resumed_session": resumed,
        "reloaded_update": loaded["position"]["updates"],
        "continuation": continuation,
    }
    if resumed["status"] != "completed" or resumed["last_validation"] is None:
        raise RuntimeError("Smoke test did not complete training and validation.")
    write_json(output / "summary.json", report)
    (output / "diagnostic_continuation.txt").write_text(
        continuation["label"]
        + "\nPrompt: "
        + continuation["prompt"]
        + "\nContinuation: "
        + continuation["continuation"]
        + "\n",
        encoding="utf-8",
    )
    return report


def run_overfit(
    config: TrainingConfig, root: Path, output: Path, *, max_seconds=30.0, success_loss=0.1
):
    if config.max_updates is None or not 1 <= config.max_updates <= 200:
        raise ValueError("Overfit check requires 1–200 maximum updates.")
    if (
        config.microbatch_size != 1
        or config.gradient_accumulation_steps != 1
        or config.model.dropout != 0
    ):
        raise ValueError("Overfit check requires microbatch/accumulation 1 and dropout 0.")
    if (
        config.model.num_layers > 2
        or config.model.embedding_dim > 64
        or config.model.context_length > 64
    ):
        raise ValueError("Overfit check requires a small model.")
    if not 0 < max_seconds <= 60 or not 0 < success_loss <= 0.5:
        raise ValueError("Invalid overfit time or loss threshold.")
    output.mkdir(parents=True, exist_ok=False)
    config.save(output / "optimizer_and_model_settings.json")
    tokenizer = CharacterTokenizer.load(
        root / config.tokenizer_path, expected_fingerprint=config.tokenizer_fingerprint
    )
    if tokenizer.vocab_size != config.model.vocab_size:
        raise ValueError("Overfit vocabulary mismatch.")
    story = ("red blue.\n" * (config.model.context_length // 10 + 1))[: config.model.context_length]
    ids = torch.tensor(tokenizer.encode(story, add_eos=True, unknown="error"), dtype=torch.int64)
    dataset = [(ids[:-1], ids[1:])]
    device = select_device(config.device).device
    journal = MetricsJournal(output / "loss_history.jsonl")
    with execution_settings(config, device):
        model = TransformerLanguageModel(config.model).to(device)
        optimizer = make_optimizer(model, config)
        schedule, sampler = WarmupCosineSchedule(config), WindowSampler(1, config.seed, False)
        initial = evaluate(model, dataset, 1, 1, device)["validation_loss"]
        journal.append({"update": 0, "fixed_training_example_loss": initial})
        final, started = initial, time.perf_counter()
        for update in range(1, config.max_updates + 1):
            result = optimizer_update(
                model, optimizer, schedule, dataset, sampler, config, 1, device
            )
            final = evaluate(model, dataset, 1, 1, device)["validation_loss"]
            journal.append({"update": update, "fixed_training_example_loss": final, **result})
            if (
                final <= min(success_loss, 0.1 * initial)
                or time.perf_counter() - started >= max_seconds
            ):
                break
        elapsed = time.perf_counter() - started
    passed = final <= min(success_loss, 0.1 * initial)
    report = {
        "kind": "tiny_data_overfit",
        "status": "passed" if passed else "failed_to_overfit",
        "data_used": "one fixed synthetic training window, not corpus or held-out text",
        "synthetic_story": story,
        "synthetic_story_sha256": hashlib.sha256(story.encode()).hexdigest(),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "initial_loss": initial,
        "final_loss": final,
        "loss_ratio": final / initial,
        "success_absolute_loss": success_loss,
        "required_loss_ratio": 0.1,
        "updates": update,
        "maximum_updates": config.max_updates,
        "maximum_seconds": max_seconds,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "not_model_quality_evaluation": True,
    }
    write_json(output / "summary.json", report)
    return report


# Worker
"""One isolated, bounded hardware case with raw iteration timing and memory records."""

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from transformer.configuration.settings import BenchmarkConfig
from transformer.configuration.settings import PathConfig
from transformer.data.preparation import write_json
from transformer.data.loading import TokenWindowDataset
from transformer.model import TransformerLanguageModel
from transformer.runtime.device import select_device
from transformer.tokenization.token_preparation import read_token_manifest
from transformer.training.engine import MetricsJournal
from transformer.training.engine import (
    WarmupCosineSchedule,
    batch_from_indices,
    make_optimizer,
    optimizer_update_batches,
)
from transformer.training.engine import execution_settings
from transformer.training.engine import run_training


class MeasurementLimit(RuntimeError):
    pass


def summarize_rows(rows):
    measured = [row for row in rows if row["phase"] == "measured"]
    if not measured:
        return {"measured_iterations": 0}
    tokens = sum(row["cycle_tokens"] for row in measured)
    seconds = sum(row["boundary_seconds"] for row in measured)
    early_count = min(5, len(measured) // 3)

    def rate(group):
        return sum(row["cycle_tokens"] for row in group) / sum(
            row["boundary_seconds"] for row in group
        )

    early = rate(measured[:early_count]) if early_count else None
    late = rate(measured[-early_count:]) if early_count else None
    return {
        "measured_iterations": len(measured),
        "measured_tokens": tokens,
        "measured_seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "median_update_seconds": statistics.median(row["boundary_seconds"] for row in measured),
        "mean_core_update_seconds": statistics.mean(row["update_seconds"] for row in measured),
        "mean_boundary_update_seconds": seconds / len(measured),
        "early_iterations": early_count,
        "early_tokens_per_second": early,
        "late_tokens_per_second": late,
        "late_over_early_rate": late / early if early else None,
        "validation_event_seconds": [
            row["validation_seconds"] for row in measured if row["validation_seconds"] > 0
        ],
        "checkpoint_event_seconds": [
            row["checkpoint_seconds"] for row in measured if row["checkpoint_seconds"] > 0
        ],
    }


def run_case(benchmark: BenchmarkConfig, root: Path, batch: int, mode: str, sustained=False):
    if batch not in benchmark.microbatch_sizes or mode not in ("compute", "pipeline"):
        raise ValueError("Unknown benchmark case.")
    if sustained and mode != "pipeline":
        raise ValueError("Sustained measurement uses the full pipeline.")
    manifest = read_token_manifest(
        root / benchmark.training.token_data_dir,
        benchmark.training.tokenizer_fingerprint,
        expected_dataset_fingerprint=benchmark.training.dataset_fingerprint,
    )
    if manifest["token_data_fingerprint"] != benchmark.training.token_data_fingerprint:
        raise ValueError("Incompatible benchmark token-data fingerprint.")
    name = f"{'sustained' if sustained else mode}-mb{batch}"
    output = root / benchmark.output_dir / name
    output.mkdir(parents=True, exist_ok=False)
    measured = benchmark.sustained_updates if sustained else benchmark.measured_updates
    seconds_limit = benchmark.sustained_max_seconds if sustained else benchmark.case_max_seconds
    device = select_device(benchmark.training.device).device
    config = replace(
        benchmark.training,
        run_id=name,
        device=device.type,
        microbatch_size=batch,
        gradient_accumulation_steps=benchmark.effective_batch_size // batch,
        max_updates=benchmark.warmup_updates + measured,
        max_training_tokens=None,
        warmup_updates=0,
        minimum_lr_ratio=1.0,
        evaluation_interval=benchmark.evaluation_interval,
        checkpoint_interval=benchmark.checkpoint_interval,
        max_validation_batches=benchmark.validation_windows // batch,
        console_interval=100,
    )
    config.save(output / "training_configuration.json")
    hardware = hardware_identity()
    write_json(output / "hardware.json", hardware)
    monitor = MemoryMonitor(device)
    baseline = monitor.snapshot(system=True)
    write_json(output / "memory_before.json", baseline)
    reason = memory_limit_reason(baseline, benchmark)
    if reason:
        report = {
            "case": name,
            "status": "memory_limit",
            "reason": reason,
            "measured_iterations": 0,
        }
        write_json(output / "summary.json", report)
        return report
    journal = MetricsJournal(output / "iterations.jsonl")
    rows = []
    boundary_started = None
    ready_seconds = None
    started = time.perf_counter()

    def observer(event):
        nonlocal boundary_started, ready_seconds
        now = time.perf_counter()
        if event["event"] == "ready":
            ready_seconds = now - started
            boundary_started = time.perf_counter()
            return
        boundary_seconds = event["update_seconds"] if mode == "compute" else now - boundary_started
        update = event["optimizer_update"]
        memory = monitor.snapshot(system=update % 10 == 0 or update == 1)
        row = {
            **event,
            "phase": "warmup" if update <= benchmark.warmup_updates else "measured",
            "boundary_seconds": boundary_seconds,
            "memory": memory,
        }
        journal.append(row)
        rows.append(row)
        reason = memory_limit_reason(memory, benchmark)
        if reason:
            raise MeasurementLimit("memory_limit: " + reason)
        if time.perf_counter() - started >= seconds_limit:
            raise MeasurementLimit("time_limit: bounded case wall time reached")
        boundary_started = time.perf_counter()

    def memory_hook(module, args, result):
        if isinstance(module, TransformerLanguageModel):
            monitor.capture_accelerator()

    hook = torch.nn.modules.module.register_module_forward_hook(memory_hook)
    status, reason = "completed", None
    try:
        if mode == "pipeline":
            run_training(
                config,
                root,
                PathConfig(runs_dir=f"{benchmark.output_dir}/{name}/runs"),
                observer=observer,
            )
        else:
            with (
                execution_settings(config, device),
                TokenWindowDataset(
                    root / config.token_data_dir,
                    config.train_stream,
                    config.tokenizer_fingerprint,
                    context_length=config.model.context_length,
                    dataset_fingerprint=config.dataset_fingerprint,
                ) as dataset,
            ):
                model = TransformerLanguageModel(config.model).to(device)
                optimizer, schedule = make_optimizer(model, config), WarmupCosineSchedule(config)
                cached = [
                    batch_from_indices(
                        dataset, [i % len(dataset) for i in range(start, start + batch)], device
                    )
                    for start in range(0, benchmark.effective_batch_size, batch)
                ]
                observer({"event": "ready"})
                for update in range(1, config.max_updates + 1):
                    result = optimizer_update_batches(
                        model,
                        optimizer,
                        schedule,
                        cached,
                        config,
                        benchmark.effective_batch_size,
                        device,
                    )
                    observer(
                        {
                            "event": "update",
                            "optimizer_update": update,
                            **result,
                            "validation_seconds": 0.0,
                            "checkpoint_seconds": 0.0,
                        }
                    )
    except MeasurementLimit as exc:
        status, reason = str(exc).split(": ", 1)
    except RuntimeError as exc:
        status = "out_of_memory" if "out of memory" in str(exc).lower() else "failed"
        reason = str(exc)
    finally:
        hook.remove()
    memory_after = monitor.snapshot(system=True)
    peak_rss = max(
        [
            baseline["process_lifetime_peak_rss_bytes"],
            memory_after["process_lifetime_peak_rss_bytes"],
        ]
        + [row["memory"]["process_lifetime_peak_rss_bytes"] for row in rows]
    )
    report = {
        "case": name,
        "status": status,
        "reason": reason,
        "mode": mode,
        "microbatch_size": batch,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "device": str(device),
        "warmup_updates": benchmark.warmup_updates,
        "maximum_updates": config.max_updates,
        "maximum_wall_seconds": seconds_limit,
        "completed_updates": len(rows),
        "case_wall_seconds": time.perf_counter() - started,
        "setup_to_ready_seconds": ready_seconds,
        "process_lifetime_peak_rss_bytes": peak_rss,
        "memory_after": memory_after,
        "hardware": hardware,
        **summarize_rows(rows),
        "not_model_quality_evaluation": True,
    }
    write_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Isolated internal benchmark worker.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--mode", choices=("compute", "pipeline"), required=True)
    parser.add_argument("--sustained", action="store_true")
    args = parser.parse_args()
    result = run_case(
        BenchmarkConfig.load(args.config), args.project_root, args.batch, args.mode, args.sustained
    )
    print(json.dumps(result))
    return 0 if result["status"] in ("completed", "time_limit", "memory_limit") else 2


if __name__ == "__main__":
    sys.exit(main())
