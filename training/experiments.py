# Baseline experiment
"""Paired experiment orchestration with preserved milestones and a baseline-only entry point."""

import hashlib
import json
import shutil
import signal
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from transformer.benchmarking.worker import diagnostic_continuation
from transformer.benchmarking.worker import MemoryMonitor, hardware_identity
from transformer.configuration.settings import ComparisonProtocol
from transformer.configuration.settings import TrainingConfig
from transformer.data.preparation import file_sha256, fingerprint, write_json
from transformer.data.loading import TokenWindowDataset
from transformer.model import TransformerLanguageModel
from transformer.runtime.device import select_device
from transformer.tokenization.character_tokenizer import CharacterTokenizer
from transformer.training.engine import (
    CheckpointManager,
    MetricsJournal,
    atomic_json,
    atomic_write,
    run_lock,
)
from transformer.training.engine import execution_settings
from transformer.training.engine import run_training


def weights_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_samples(protocol, root, directory, which="latest"):
    """Archive the exact sampled checkpoint; inference cannot change the saved training RNG."""
    with run_lock(directory):
        return _save_samples(protocol, root, directory, which)


def _save_samples(protocol, root, directory, which):
    manager = CheckpointManager(directory)
    payload = manager.load(which)
    config = TrainingConfig.from_dict(payload["identity"]["configuration"])
    member = "model_a" if config.train_stream == "train_tinystories" else "model_b"
    if config != getattr(protocol, member):
        raise ValueError("Sample checkpoint does not match the frozen comparison member.")
    update = payload["position"]["updates"]
    reference = manager.index()[which]
    diagnostics = directory / "diagnostics"
    destination = diagnostics / f"samples-update-{update:06d}.json"
    archived = diagnostics / f"checkpoint-update-{update:06d}.pt"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if (
            saved["checkpoint_sha256"] != reference["sha256"]
            or saved["checkpoint_file"] != str(archived.relative_to(directory))
            or file_sha256(archived) != reference["sha256"]
        ):
            raise ValueError("Existing samples refer to a different checkpoint.")
        return saved
    source = manager.directory / reference["file"]
    if archived.exists():
        if file_sha256(archived) != reference["sha256"]:
            raise ValueError("Archived milestone checksum mismatch.")
    else:
        with source.open("rb") as stream:
            atomic_write(archived, lambda output: shutil.copyfileobj(stream, output))
    device = select_device(config.device).device
    tokenizer = CharacterTokenizer.load(
        root / config.tokenizer_path, expected_fingerprint=config.tokenizer_fingerprint
    )
    started = time.perf_counter()
    with execution_settings(config, device):
        model = TransformerLanguageModel(config.model).to(device)
        model.load_state_dict(payload["model"], strict=True)
        samples = []
        for prompt in protocol.prompts:
            result = diagnostic_continuation(model, tokenizer, prompt, protocol.max_new_tokens)
            result.update(
                label=f"{member.replace('_', ' ').upper()} DIAGNOSTIC: not factually reliable",
                comparison_member=member,
                checkpoint_update=update,
                seed=None,
                seed_note="Greedy argmax has no seed; cross-device equality is not promised.",
                temperature=0.0,
                top_k=None,
                top_p=None,
                initialization_seed=config.seed,
            )
            samples.append(result)
    result = {
        "update": update,
        "checkpoint_sha256": reference["sha256"],
        "checkpoint_file": str(archived.relative_to(directory)),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "sampling_seconds": time.perf_counter() - started,
        "samples": samples,
    }
    atomic_json(destination, result)
    return result


def run_baseline(protocol: ComparisonProtocol, root: Path, *, resume=False):
    return run_experiment(protocol, root, member="model_a", resume=resume)


def run_experiment(
    protocol,
    root,
    *,
    member,
    resume=False,
    observer_extension=None,
    supplement=None,
    expected_initial_weights=None,
):
    if member not in ("model_a", "model_b"):
        raise ValueError("Unknown paired experiment member.")
    try:
        return _run_experiment(
            protocol,
            root,
            member=member,
            resume=resume,
            observer_extension=observer_extension,
            supplement=supplement,
            expected_initial_weights=expected_initial_weights,
        )
    except KeyboardInterrupt:
        directory = Path(root).resolve() / "runs" / getattr(protocol, member).run_id
        manager = CheckpointManager(directory)
        if not manager.index_path.exists():
            return {
                "status": "interrupted",
                "resume_required": False,
                "reason": "No checkpoint yet.",
            }
        position = manager.load()["position"]
        record = {
            "status": "interrupted",
            "resume_required": True,
            "run_directory": str(directory),
            **position,
        }
        MetricsJournal(directory / "experiment_events.jsonl").append(
            {
                "event": "orchestration_interrupted",
                "update": position["updates"],
                "utc": datetime.now(UTC).isoformat(),
            }
        )
        atomic_json(directory / "experiment_status.json", record)
        return record


def _run_experiment(
    protocol,
    root,
    *,
    member,
    resume=False,
    observer_extension=None,
    supplement=None,
    expected_initial_weights=None,
):
    root = Path(root).resolve()
    config = getattr(protocol, member)
    directory = root / "runs" / config.run_id
    document = (root / protocol.document_path).read_text()
    frozen = {
        "protocol": protocol.to_dict(),
        "document_sha256": hashlib.sha256(document.encode()).hexdigest(),
        "protocol_fingerprint": fingerprint(protocol.to_dict()),
    }
    frozen = json.loads(json.dumps(frozen))
    if supplement is not None:
        frozen["supplement"] = supplement
    if directory.exists() and not resume:
        raise FileExistsError("Baseline run already exists; use --resume with its frozen protocol.")
    if resume and not directory.is_dir():
        raise ValueError("Cannot resume a baseline that does not exist.")
    if resume and json.loads((directory / "comparison_protocol.json").read_text()) != frozen:
        raise ValueError("Resume would change the frozen comparison protocol or document.")
    if resume and expected_initial_weights is not None:
        if (
            json.loads((directory / "initialization.json").read_text())["weights_sha256"]
            != expected_initial_weights
        ):
            raise ValueError("Resume initialization does not match its paired experiment.")
    tokenizer = CharacterTokenizer.load(
        root / config.tokenizer_path, expected_fingerprint=config.tokenizer_fingerprint
    )
    for prompt in protocol.prompts:
        tokenizer.encode(prompt, unknown="error")
    # Check the full validation budget before creating any run or taking an optimizer step.
    with TokenWindowDataset(
        root / config.token_data_dir,
        "validation",
        config.tokenizer_fingerprint,
        context_length=config.model.context_length,
        dataset_fingerprint=config.dataset_fingerprint,
    ) as validation:
        if config.max_validation_batches * config.microbatch_size < len(validation):
            raise ValueError("Baseline selection must cover every prepared validation window.")
    run_training(config, root, resume=resume, dry_run=True)
    device = select_device(config.device).device
    monitor = MemoryMonitor(device)
    before = monitor.snapshot(system=True)
    if (
        before["system_memory_pressure_free_percent"] is not None
        and before["system_memory_pressure_free_percent"] < 20
    ):
        raise RuntimeError("Insufficient system memory headroom; baseline was not started.")
    manager = CheckpointManager(directory)
    events = MetricsJournal(directory / "experiment_events.jsonl")
    experiment_started = time.perf_counter()

    def observe(event):
        if event["event"] == "ready" and not (directory / "comparison_protocol.json").exists():
            # Pehla optimizer update se pehle protocol aur random initialization ka proof save karo.
            (directory / "diagnostics").mkdir()
            write_json(directory / "comparison_protocol.json", frozen)
            (directory / "experiment_protocol.md").write_text(document)
            payload = manager.load()
            initial_hash = weights_fingerprint(payload["model"])
            if expected_initial_weights is not None and initial_hash != expected_initial_weights:
                raise ValueError("Fresh random weights do not match the paired initialization.")
            write_json(
                directory / "initialization.json",
                {
                    "updates": payload["position"]["updates"],
                    "seed": config.seed,
                    "weights_sha256": initial_hash,
                    "optimizer_state_entries": len(payload["optimizer"]["state"]),
                    "origin": "fresh random initialization; no prior-run weights loaded",
                },
            )
            write_json(directory / "hardware_start.json", hardware_identity())
            write_json(directory / "memory_start.json", before)
        if (
            event["event"] == "update"
            and event["optimizer_update"] % config.evaluation_interval == 0
        ):
            memory = monitor.snapshot(system=True)
            events.append(
                {
                    "event": "validation_boundary",
                    "update": event["optimizer_update"],
                    "validation_loss": event["validation_loss"],
                    "validation_tokens": event["validation_tokens"],
                    "validation_seconds": event["validation_seconds"],
                    "checkpoint_seconds": event["checkpoint_seconds"],
                    "memory": memory,
                }
            )
            free = memory["system_memory_pressure_free_percent"]
            driver = memory["maximum_sampled_mps_driver_bytes"]
            if event["optimizer_update"] < config.total_updates and (
                memory["process_lifetime_peak_rss_bytes"] > 4 * 1024**3
                or (driver is not None and driver > 4 * 1024**3)
                or (free is not None and free < 20)
            ):
                events.append(
                    {"event": "resource_pause_requested", "update": event["optimizer_update"]}
                )
                signal.raise_signal(signal.SIGINT)
        if observer_extension is not None:
            observer_extension(event, directory)

    def observer(event):
        if threading.current_thread() is not threading.main_thread():
            observe(event)
            return
        previous = signal.getsignal(signal.SIGINT)
        interrupted = False

        def capture(signum, frame):
            nonlocal interrupted
            interrupted = True
            if callable(previous):
                previous(signum, frame)

        signal.signal(signal.SIGINT, capture)
        try:
            observe(event)
        finally:
            signal.signal(signal.SIGINT, previous)
        # Final callback mein SIGINT aaye to agla planned session shuru mat karo.
        if interrupted:
            raise KeyboardInterrupt

    current = manager.load()["position"]["updates"] if resume else 0
    # Sampling ke beech bhi experiment lock rahega; training ka apna lock alag hai.
    with run_lock(directory.with_name(directory.name + "-experiment")):
        if resume and current in protocol.sample_updates:
            save_samples(protocol, root, directory)
        result = None
        for target in protocol.sample_updates:
            if target <= current:
                continue
            started = time.perf_counter()
            result = run_training(
                config,
                root,
                resume=resume or current > 0,
                stop_after_updates=target,
                observer=observer,
            )
            events.append(
                {
                    "event": "training_session",
                    "utc": datetime.now(UTC).isoformat(),
                    "from_update": current,
                    "requested_stop_update": target,
                    "status": result["status"],
                    "final_update": result["updates"],
                    "wall_seconds": time.perf_counter() - started,
                    "planned_pause": target != config.total_updates,
                    "resume": resume or current > 0,
                }
            )
            current = result["updates"]
            if result["status"] == "interrupted":
                atomic_json(
                    directory / "experiment_status.json",
                    {**result, "status": "interrupted", "resume_required": True},
                )
                return result
            save_samples(protocol, root, directory)
        if result is None:
            result = run_training(config, root, resume=True)
        best = manager.load("best")
        save_samples(protocol, root, directory, "best")
        events.append(
            {
                "event": "experiment_invocation_completed",
                "wall_seconds": time.perf_counter() - experiment_started,
                "utc": datetime.now(UTC).isoformat(),
            }
        )
        metrics = [
            json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        updates = [row for row in metrics if row["event"] == "update"]
        validations = [
            {
                key: row[key]
                for key in (
                    "optimizer_update",
                    "tokens_processed",
                    "validation_loss",
                    "validation_tokens",
                )
            }
            for row in updates
            if row["validation_loss"] is not None
        ]
        expected_best = min(validations, key=lambda row: row["validation_loss"])
        if best["position"]["updates"] != expected_best["optimizer_update"]:
            raise RuntimeError(
                "Selected checkpoint disagrees with the predeclared validation rule."
            )
        atomic_json(
            directory / "validation_history.json",
            {
                "selection_rule": "lowest full-validation token-weighted NLL; earliest exact tie",
                "history": validations,
            },
        )
        history = [
            json.loads(line)
            for line in (directory / "experiment_events.jsonl").read_text().splitlines()
        ]
        sessions = [row for row in history if row["event"] == "training_session"]
        wall = sum(row["wall_seconds"] for row in sessions)
        core = sum(row["update_seconds"] for row in updates)
        report = {
            "status": result["status"],
            "run_directory": str(directory),
            "train_stream": config.train_stream,
            "optimizer_updates": result["updates"],
            "training_tokens": result["tokens"],
            "token_budget": config.total_tokens,
            "best_validation": expected_best,
            "last_validation": validations[-1],
            "initial_training_update_loss": updates[0]["training_loss"],
            "last_training_update_loss": updates[-1]["training_loss"],
            "last_100_update_mean_training_loss": sum(
                row["training_loss"] for row in updates[-100:]
            )
            / len(updates[-100:]),
            "training_core_seconds": core,
            "training_session_wall_seconds": wall,
            "core_training_tokens_per_second": result["tokens"] / core,
            "session_training_tokens_per_second": result["tokens"] / wall,
            "timing_definition": (
                "Core includes batching/transfers/optimizer work. Session wall also includes "
                "setup, validation, logging, checkpoint I/O, and observation; "
                "diagnostic sampling between sessions is excluded."
            ),
            "sessions": sessions,
            "checkpoint_index": manager.index(),
            "memory_end": monitor.snapshot(system=True),
            "sample_files": [
                str(path.relative_to(directory))
                for path in sorted((directory / "diagnostics").glob("samples-*.json"))
            ],
            "test_evaluation_performed": False,
            "model_b_trained": member == "model_b",
            "comparison_member": member,
        }
        atomic_json(directory / "training_summary.json", report)
        atomic_json(
            directory / "experiment_status.json",
            {"status": result["status"], "resume_required": False},
        )
    return report


# Combined experiment
"""Prepare and execute Model B without changing the paired training or selection controls."""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from transformer.configuration.settings import TrainingConfig
from transformer.data.preparation import (
    file_sha256,
    fingerprint,
    staged_directory,
    verify_artifact,
    write_json,
)
from transformer.data.loading import audit_partitions, source_exposure
from transformer.evaluation.metrics import SourceValidationDataset, evaluate_sources
from transformer.training.engine import CheckpointManager, atomic_json


def _baseline_records(protocol, root):
    directory = root / "runs" / protocol.model_a.run_id
    config = TrainingConfig.load(directory / "configuration.json")
    if config != protocol.model_a:
        raise ValueError("Actual baseline settings differ from the frozen paired protocol.")
    frozen = json.loads((directory / "comparison_protocol.json").read_text())
    if frozen["protocol"] != json.loads(json.dumps(protocol.to_dict())) or frozen[
        "document_sha256"
    ] != file_sha256(root / protocol.document_path):
        raise ValueError("The original baseline protocol/document must remain unchanged.")
    summary = json.loads((directory / "training_summary.json").read_text())
    if summary["status"] != "completed" or summary["training_tokens"] != config.total_tokens:
        raise ValueError("Model A has not completed the matched token budget.")
    payload = CheckpointManager(directory).load()
    if payload["position"]["tokens"] != config.total_tokens:
        raise ValueError("Actual baseline checkpoint disagrees with its recorded token budget.")
    hashes = json.loads((directory / "prerequisite_verification.json").read_text())["source_hashes"]
    package = Path(__file__).resolve().parents[1]
    core = [name for name in hashes if name.startswith("model/")]
    core += ["training/engine.py", "data/loading.py"]
    core += [
        "configuration/settings.py",
        "configuration/settings.py",
        "data/loading.py",
        "tokenization/character_tokenizer.py",
        "tokenization/token_preparation.py",
        "data/preparation.py",
    ]
    changed = [name for name in core if not (package / name).is_file() or file_sha256(package / name) != hashes.get(name)]
    if changed:
        raise ValueError(
            f"Paired compute/data code changed; review comparability before training: {changed}"
        )
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "system": platform.platform(),
        "machine": platform.machine(),
    }
    if any(
        payload["identity"]["environment"].get(key) != value for key, value in environment.items()
    ):
        raise ValueError("Baseline environment changed; review comparability before training.")
    initial = json.loads((directory / "initialization.json").read_text())
    if (
        initial["updates"] != 0
        or initial["optimizer_state_entries"] != 0
        or initial["seed"] != config.seed
    ):
        raise ValueError("Baseline lacks verified fresh-initialization evidence.")
    return directory, summary, initial, core


def prepare_combined(protocol, root, output, plan_path):
    root, output = Path(root).resolve(), Path(output).resolve()
    baseline, summary, initial, core = _baseline_records(protocol, root)
    views, audit = audit_partitions(root, protocol.model_b)
    package = Path(__file__).resolve().parents[1]
    identity = {
        "format_version": 1,
        "protocol": protocol.to_dict(),
        "original_protocol_document_sha256": file_sha256(root / protocol.document_path),
        "phase9_plan_sha256": file_sha256(plan_path),
        "baseline_configuration_sha256": file_sha256(baseline / "configuration.json"),
        "baseline_metrics_sha256": file_sha256(baseline / "metrics.jsonl"),
        "baseline_index_sha256": file_sha256(baseline / "checkpoints.json"),
        "baseline_initialization_sha256": file_sha256(baseline / "initialization.json"),
        "source_hashes": {
            str(p.relative_to(package)): file_sha256(p) for p in sorted(package.rglob("*.py"))
        },
        "audit": audit,
    }
    identity = json.loads(json.dumps(identity))
    if output.exists():
        return views, verify_artifact(output, identity)
    description = views["train"].description()
    planned = source_exposure(views["train"], protocol.model_b, protocol.model_b.total_tokens)
    validation = views["validation"].description()
    isolated = {}
    for source in ("tinystories", "simplestories"):
        with SourceValidationDataset(views["validation"], source) as data:
            isolated[source] = data.view_identity
    output.parent.mkdir(parents=True, exist_ok=True)
    with staged_directory(output) as stage:
        write_json(stage / "data_audit.json", audit)
        write_json(
            stage / "mixture_plan.json",
            {
                "policy": "uniform shuffled packed windows from the frozen combined stream",
                "ratio_unit": "predicted target tokens including EOS, not document counts",
                "combined_stream": description,
                "seeded_planned_exposure": planned,
                "expected_targets_over_uniform_first_epoch_draws": {
                    name: protocol.model_b.total_tokens * ratio
                    for name, ratio in description["source_eligible_target_fraction"].items()
                },
                "repetition": "Shuffle without replacement; a new permutation follows exhaustion.",
                "aggregate_validation_selector": validation,
                "source_only_validation_views": isolated,
                "source_only_scores_used_for_selection": False,
            },
        )
        write_json(
            stage / "compatibility.json",
            {
                "status": "matched_training_controls_and_selection",
                "unchanged_compute_files": core,
                "baseline_initial_weights_sha256": initial["weights_sha256"],
                "training_budget_each": protocol.model_b.total_tokens,
                "aggregate_rule": "lowest full packed-validation token-weighted NLL; earliest tie",
                "rerun_required": False,
                "reason": (
                    "Observation/orchestration is extended. Optimizer, model, sampler, loader, "
                    "schedule and aggregate selector match. Source-only views are secondary."
                ),
                "limitations": (
                    "A's source-only diagnostics are retrospective at retained milestones/best; "
                    "its checkpoint selection is unchanged. One seed cannot establish significance."
                ),
            },
        )
        (stage / "phase9_experiment_protocol.md").write_bytes(Path(plan_path).read_bytes())
        files = list((baseline / "diagnostics").glob("samples-*.json"))
        evaluated = []
        for path in sorted(files):
            sample = json.loads(path.read_text())
            checkpoint = baseline / sample["checkpoint_file"]
            if file_sha256(checkpoint) != sample["checkpoint_sha256"]:
                raise ValueError("Baseline archived milestone checksum mismatch.")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["position"]["updates"] not in (
                *protocol.sample_updates,
                summary["best_validation"]["optimizer_update"],
            ):
                continue
            evaluated.append(
                {
                    **evaluate_sources(payload, views["validation"]),
                    "checkpoint_sha256": file_sha256(checkpoint),
                }
            )
        if not set(protocol.sample_updates) <= {row["update"] for row in evaluated}:
            raise ValueError(
                "Baseline milestone checkpoints required for paired diagnostics are missing."
            )
        write_json(
            stage / "model_a_source_validation.json",
            {"retrospective": True, "history": evaluated, "checkpoint_selection_changed": False},
        )
        write_json(
            stage / "model_a_exposure.json",
            source_exposure(
                views["train_tinystories"],
                protocol.model_a,
                protocol.model_a.total_tokens,
                CheckpointManager(baseline).load()["sampler"],
            ),
        )
        manifest = {
            "identity": identity,
            "fingerprint": fingerprint(identity),
            "files": {
                p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                for p in sorted(stage.iterdir())
            },
        }
        write_json(stage / "manifest.json", manifest)
    return views, manifest


def run_combined(protocol, root, output, plan_path, *, resume=False, prepare_only=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    views, prepared = prepare_combined(protocol, root, output, plan_path)
    if prepare_only:
        return {
            "status": "prepared",
            "directory": str(output),
            "fingerprint": prepared["fingerprint"],
            "model_b_training_started": False,
        }
    config = protocol.model_b
    baseline = root / "runs" / protocol.model_a.run_id
    initial = json.loads((baseline / "initialization.json").read_text())
    supplement = {
        "preparation_fingerprint": prepared["fingerprint"],
        "plan_sha256": file_sha256(plan_path),
    }

    def assess(directory, which="latest", *, in_session=False):
        manager = CheckpointManager(directory)
        payload = manager.load(which)
        reference = manager.index()[which]
        update = payload["position"]["updates"]
        destination = directory / "source_diagnostics" / f"update-{update:06d}.json"
        if destination.exists():
            record = json.loads(destination.read_text())
            if record["checkpoint_sha256"] != reference["sha256"]:
                raise ValueError("Source audit refers to a different saved checkpoint.")
            return record
        started = time.perf_counter()
        exposure = source_exposure(
            views["train"], config, payload["position"]["tokens"], payload["sampler"]
        )
        result = {
            "update": update,
            "checkpoint_sha256": reference["sha256"],
            "source_exposure": exposure,
            "in_training_session": in_session,
        }
        if update in protocol.sample_updates or which == "best":
            result["source_only_validation"] = evaluate_sources(payload, views["validation"])
        result["additional_audit_seconds"] = time.perf_counter() - started
        atomic_json(destination, result)
        return result

    def observer(event, directory):
        diagnostics = directory / "source_diagnostics"
        if event["event"] == "ready":
            diagnostics.mkdir(exist_ok=True)
            if not (directory / "mixture_plan.json").exists():
                for name in (
                    "mixture_plan.json",
                    "data_audit.json",
                    "compatibility.json",
                    "phase9_experiment_protocol.md",
                ):
                    (directory / name).write_bytes((output / name).read_bytes())
                write_json(directory / "preparation_manifest.json", prepared)
            current = CheckpointManager(directory).load()["position"]["updates"]
            if current and current % config.evaluation_interval == 0:
                assess(directory, in_session=True)
        elif event["validation_loss"] is not None:
            assess(directory, in_session=True)

    result = run_experiment(
        protocol,
        root,
        member="model_b",
        resume=resume,
        observer_extension=observer,
        supplement=supplement,
        expected_initial_weights=initial["weights_sha256"],
    )
    if result["status"] != "completed":
        return result
    directory = root / "runs" / config.run_id
    best = assess(directory, "best")
    extra_source_seconds = 0.0
    # A best checkpoint may already have exposure-only auditing from a non-milestone update.
    if "source_only_validation" not in best:
        started = time.perf_counter()
        best = {
            **best,
            "source_only_validation": evaluate_sources(
                CheckpointManager(directory).load("best"), views["validation"]
            ),
        }
        atomic_json(directory / "source_diagnostics" / f"update-{best['update']:06d}.json", best)
        extra_source_seconds = time.perf_counter() - started
    records = [
        json.loads(p.read_text())
        for p in sorted((directory / "source_diagnostics").glob("update-*.json"))
    ]
    actual = source_exposure(
        views["train"], config, config.total_tokens, CheckpointManager(directory).load()["sampler"]
    )
    planned = json.loads((output / "mixture_plan.json").read_text())["seeded_planned_exposure"]
    if (
        actual["source_target_tokens"] != planned["source_target_tokens"]
        or actual["sampled_window_order_sha256"] != planned["sampled_window_order_sha256"]
    ):
        raise RuntimeError("Actual source exposure deviates from the frozen seeded plan.")
    atomic_json(directory / "source_exposure.json", actual)
    atomic_json(
        directory / "source_validation_history.json",
        {
            "selection_rule_changed": False,
            "history": [
                r["source_only_validation"] for r in records if "source_only_validation" in r
            ],
            "selected_best_update": best["update"],
        },
    )
    result.update(
        source_exposure=actual,
        source_only_best_validation=best["source_only_validation"],
        source_audit_seconds_in_sessions=sum(
            r["additional_audit_seconds"] for r in records if r["in_training_session"]
        ),
        post_training_source_validation_seconds=extra_source_seconds,
        comparison_limitations=(
            "Single seed and bounded prefixes; source-only diagnostics do not "
            "select checkpoints; formal test comparison deferred to Phase 10."
        ),
    )
    atomic_json(directory / "training_summary.json", result)
    return result


# Prepare english run
"""Materialize a separate English-story training configuration from verified manifests."""

import argparse
import json
from pathlib import Path

from transformer.configuration.settings import TokenPreparationConfig
from transformer.tokenization.token_preparation import prepare_tokens


def command_prepare():
    parser = argparse.ArgumentParser(description=__doc__)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--expanded", action="store_true")
    variant.add_argument("--larger", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prepared = "english-stories-expanded-v1" if args.expanded else "english-stories-v1"
    token_name = "english-characters-expanded-v1" if args.expanded else "english-characters-v1"
    if args.larger:
        prepared = "english-stories-larger-v1"
        token_name = "english-characters-larger-v1"
    dataset = json.loads((root / "data/prepared" / prepared / "manifest.json").read_text())
    token_config = TokenPreparationConfig(
        name=token_name, prepared_name=prepared, dataset_fingerprint=dataset["corpus_fingerprint"]
    )
    prepare_tokens(token_config, project_root=root)
    token_dir = Path("data/tokens") / token_name
    manifest = json.loads((root / token_dir / "manifest.json").read_text())
    cfg = json.loads((root / "configs/training/phase8-model-a.json").read_text())
    cfg.update(
        run_id="english-stories-a-v1",
        token_data_dir=token_dir.as_posix(),
        tokenizer_path=(token_dir / "tokenizer.json").as_posix(),
        dataset_fingerprint=manifest["dataset_fingerprint"],
        token_data_fingerprint=manifest["token_data_fingerprint"],
        tokenizer_fingerprint=manifest["tokenizer_fingerprint"],
        max_training_tokens=33_554_432,
        microbatch_size=8,
        gradient_accumulation_steps=2,
        checkpoint_interval=500,
        evaluation_interval=500,
        console_interval=100,
        max_validation_batches=128,
    )
    if args.expanded or args.larger:
        cfg.update(
            run_id=prepared,
            train_stream="train",
            max_training_tokens=100_663_296 if args.larger else 67_108_864,
            checkpoint_interval=1000,
            evaluation_interval=1000,
        )
    cfg["model"]["vocab_size"] = manifest["vocabulary_size"]
    config_name = "english-stories-expanded.json" if args.expanded else "english-stories-a.json"
    if args.larger:
        config_name = "english-stories-larger.json"
    output = root / "configs/training" / config_name
    contents = json.dumps(cfg, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if output.read_text() != contents:
            raise ValueError(f"Existing configuration differs: {output}")
    else:
        output.write_text(contents)
    print(output)


# Check english run
"""Review a trusted local English training checkpoint on fixed diagnostic prompts."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from transformer.configuration.settings import GenerationConfig
from transformer.configuration.settings import TransformerConfig
from transformer.inference.service import select_next_token
from transformer.model import TransformerLanguageModel
from transformer.tokenization.character_tokenizer import CharacterTokenizer

PROMPTS = (
    "Tom opened a small box and heard a quiet tapping inside.",
    "The little bird found a blue ribbon beside the old oak tree.",
    "Mia heard her friend crying behind the garden gate.",
    "Once upon a time, a fox found a little red hat.",
)


def held_out_openings(path, per_source=3):
    counts = {"tinystories": 0, "simplestories": 0}
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            name = row["source"]
            if name not in counts or counts[name] >= per_source:
                continue
            pieces = re.split(r"(?<=[.!?])\s+", row["text"].strip(), maxsplit=1)
            if len(pieces) != 2 or not 20 <= len(pieces[0]) <= 180 or len(pieces[1]) < 40:
                continue
            counts[name] += 1
            yield {
                "prompt": pieces[0],
                "source": name,
                "story_id": row["id"],
                "reference": pieces[1][:120],
            }
            if all(count == per_source for count in counts.values()):
                return
    raise ValueError("Could not find enough held-out openings for each source.")


def command_check():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--run",
        default="english-stories-a-v1",
        choices=(
            "english-stories-a-v1",
            "english-stories-combined-v1",
            "english-stories-expanded-v1",
            "english-stories-larger-v1",
        ),
    )
    parser.add_argument("--held-out", type=Path, help="Prepared test.jsonl for opening checks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / "runs" / args.run
    ref = json.loads((run / "checkpoints.json").read_text())["best"]
    checkpoint = run / "checkpoints" / ref["file"]
    with checkpoint.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != ref["sha256"]:
            raise ValueError("Checkpoint digest does not match the saved index.")
        source.seek(0)
        data = torch.load(source, map_location="cpu", weights_only=True)
    config = data["identity"]["configuration"]
    tokenizer = CharacterTokenizer.load(
        root / config["tokenizer_path"], expected_fingerprint=config["tokenizer_fingerprint"]
    )
    model = TransformerLanguageModel(TransformerConfig(**config["model"]))
    model.load_state_dict(data["model"], strict=True)
    parameter_count = model.trainable_parameter_count()
    model.eval().requires_grad_(False).to(args.device)
    torch.set_num_threads(2)
    settings = GenerationConfig(
        mode="sample",
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        max_new_tokens=120,
    )
    openings = list(held_out_openings(args.held_out)) if args.held_out else []
    cases = [{"prompt": prompt} for prompt in PROMPTS] + openings
    samples = []
    with torch.inference_mode():
        for case in cases:
            prompt = case["prompt"]
            generator = torch.Generator().manual_seed(settings.seed)
            context = tokenizer.encode(prompt, unknown="replace")
            ids = []
            for _ in range(settings.max_new_tokens):
                x = torch.tensor([context[-model.config.context_length :]], device=args.device)
                token = select_next_token(model(x).logits[0, -1], settings, generator)
                ids.append(token)
                if token == tokenizer.eos_id:
                    break
                context.append(token)
            samples.append({**case, "continuation": tokenizer.decode(ids, skip_eos=True)})
    print(
        json.dumps(
            {
                "checkpoint": ref,
                "position": data["position"],
                "parameter_count": parameter_count,
                "sampling": {
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "seed": args.seed,
                    "max_new_tokens": 120,
                },
                "samples": samples,
            },
            indent=2,
        )
    )


# Register english run
"""Register the specific local checkpoint whose diagnostic samples were reviewed."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from transformer.inference.service import load_registered_model
from transformer.inference.service import (
    REGISTRY_PATH,
    ArtifactReference,
    ModelRegistry,
    ModelVersion,
)
from transformer.training.engine import atomic_json


def command_register():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--combined", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_id = "english-stories-combined-v1" if args.combined else "english-stories-a-v1"
    member = "extended_combined" if args.combined else "english"
    run = Path("runs") / run_id
    config_path = run / "configuration.json"
    config = json.loads((root / config_path).read_text())
    if args.combined:
        with (root / run / "metrics.jsonl").open() as journal:
            final_event = None
            for line in journal:
                final_event = line
        if not final_event or json.loads(final_event).get("event") != "completed":
            raise ValueError("Complete combined training before changing the application model.")
    diagnostics = json.loads(args.diagnostics.read_text())
    ref = diagnostics["checkpoint"]
    position = diagnostics["position"]
    if ref != json.loads((root / run / "checkpoints.json").read_text())["best"]:
        raise ValueError("Best checkpoint changed since the supplied diagnostic review.")
    if position["best_validation"]["update"] != position["updates"]:
        raise ValueError("Diagnostic checkpoint is not its validation-selected best.")

    def reference(path):
        with (root / path).open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        return ArtifactReference(Path(path).as_posix(), digest)

    checkpoint = reference(run / "checkpoints" / ref["file"])
    if checkpoint.sha256 != ref["sha256"]:
        raise ValueError("Diagnostic checkpoint checksum differs.")
    metrics_path = Path("artifacts") / run_id / "registration.json"
    metrics = {
        "scope": "validation_and_diagnostic_samples",
        "identity": {"tokenizer_fingerprint": config["tokenizer_fingerprint"]},
        "runs": {
            member: {
                "checkpoint": ref,
                "configuration": config,
                "selected_update": position["updates"],
                "selected_training_tokens": position["tokens"],
                "parameter_count": diagnostics["parameter_count"],
                "selection_rule": "lowest recorded token-weighted validation loss",
                "selected_validation": position["best_validation"],
            }
        },
        "diagnostics": diagnostics["samples"],
        "sampling": diagnostics["sampling"],
    }
    (root / metrics_path).parent.mkdir(parents=True, exist_ok=True)
    if (root / metrics_path).exists():
        if json.loads((root / metrics_path).read_text()) != metrics:
            raise ValueError("Use a new model version for a different checkpoint.")
    else:
        atomic_json(root / metrics_path, metrics)
    registry = ModelRegistry(root)
    version = ModelVersion(
        model_id="transformer-combined-extended-v1"
        if args.combined
        else "transformer-tinystories-extended-v1",
        kind="combined" if args.combined else "english",
        comparison_member=member,
        run_id=config["run_id"],
        checkpoint=checkpoint,
        configuration=reference(config_path),
        tokenizer=reference(config["tokenizer_path"]),
        metrics=reference(metrics_path),
        tokenizer_fingerprint=config["tokenizer_fingerprint"],
        checkpoint_update=position["updates"],
    )
    load_registered_model(registry, version, "cpu")
    catalog_path = root / REGISTRY_PATH
    catalog = json.loads(catalog_path.read_text())
    entries = catalog["models"]
    if args.combined:
        archive = catalog_path.with_name("historical_model_registry.json")
        if not archive.exists():
            shutil.copyfile(catalog_path, archive)
        catalog["models"] = [version.to_dict()]
        atomic_json(catalog_path, catalog)
    elif any(row["model_id"] == version.model_id for row in entries):
        if version.to_dict() not in entries:
            raise ValueError("This registered model ID already describes a different checkpoint.")
    else:
        entries.append(version.to_dict())
        atomic_json(catalog_path, catalog)
    print(version.model_id)


def main():
    """Run prepare, check or register; remaining arguments go to that command."""
    import sys
    commands = {"prepare": command_prepare, "check": command_check, "register": command_register}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit("Usage: experiments.py {prepare|check|register} [options]")
    command = commands[sys.argv.pop(1)]
    command()


if __name__ == "__main__":
    main()
