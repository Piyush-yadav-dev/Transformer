"""Read-only paired checkpoint evaluation with atomic, provenance-rich outputs."""

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from transformer.configuration.settings import ComparisonProtocol
from transformer.configuration.settings import TrainingConfig
from transformer.data.preparation import (
    artifact_lock,
    file_sha256,
    fingerprint,
    iter_jsonl,
    staged_directory,
    write_json,
)
from transformer.data.loading import audit_partitions, source_exposure
from transformer.evaluation.metrics import benchmark_generation, greedy_continuation
from transformer.evaluation.metrics import score_held_out
from transformer.model import TransformerLanguageModel
from transformer.runtime.device import select_device
from transformer.tokenization.character_tokenizer import CharacterTokenizer
from transformer.training.engine import CheckpointManager
from transformer.training.engine import execution_settings


def inventory(directory):
    return {
        str(p.relative_to(directory)): file_sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def preflight(protocol, root):
    """Verify selection from validation records before loading any test model inputs."""
    views, audit = audit_partitions(root, protocol.model_a)
    payloads, records, initial_hashes = {}, {}, []
    package = Path(__file__).resolve().parents[1]
    for member in ("model_a", "model_b"):
        config = getattr(protocol, member)
        directory = root / "runs" / config.run_id
        frozen = json.loads((directory / "comparison_protocol.json").read_text())
        if frozen["protocol"] != json.loads(json.dumps(protocol.to_dict())) or frozen[
            "document_sha256"
        ] != file_sha256(root / protocol.document_path):
            raise ValueError("Original training protocol changed; comparison refused.")
        manager = CheckpointManager(directory)
        payload, latest = manager.load("best"), manager.load("latest")
        if (
            TrainingConfig.load(directory / "configuration.json") != config
            or payload["identity"]["configuration"] != config.to_dict()
            or latest["identity"]["configuration"] != config.to_dict()
        ):
            raise ValueError("Actual checkpoint configuration differs from the paired protocol.")
        summary = json.loads((directory / "training_summary.json").read_text())
        if (
            summary["status"] != "completed"
            or summary["training_tokens"] != config.total_tokens
            or latest["position"]["tokens"] != config.total_tokens
            or latest["position"]["updates"] != config.total_updates
        ):
            raise ValueError("Actual training budget is incomplete or inconsistent.")
        if summary.get("test_evaluation_performed") is not False:
            raise ValueError("Training records do not attest that test scoring was deferred.")
        history = json.loads((directory / "validation_history.json").read_text())
        expected_rule = "lowest full-validation token-weighted NLL; earliest exact tie"
        if history.get("selection_rule") != expected_rule or not history.get("history"):
            raise ValueError("Missing or incompatible validation selection rule.")
        metrics = list(iter_jsonl(directory / "metrics.jsonl"))
        updates = [r for r in metrics if r["event"] == "update"]
        if [r["optimizer_update"] for r in updates] != list(range(1, config.total_updates + 1)):
            raise ValueError("Training update history is incomplete.")
        derived = [
            {
                k: r[k]
                for k in (
                    "optimizer_update",
                    "tokens_processed",
                    "validation_loss",
                    "validation_tokens",
                )
            }
            for r in updates
            if r.get("validation_loss") is not None
        ]
        if derived != history["history"]:
            raise ValueError("Validation history differs from actual logs.")
        selected = min(derived, key=lambda r: (r["validation_loss"], r["optimizer_update"]))
        if (
            payload["position"]["updates"] != selected["optimizer_update"]
            or payload["position"]["tokens"] != selected["tokens_processed"]
            or payload["position"]["best_validation"]["loss"] != selected["validation_loss"]
            or summary["best_validation"] != selected
        ):
            raise ValueError("Best checkpoint does not match the recorded validation selector.")
        initial = json.loads((directory / "initialization.json").read_text())
        if initial["updates"] != 0 or initial["optimizer_state_entries"] != 0:
            raise ValueError("Missing fresh random initialization evidence.")
        initial_hashes.append(initial["weights_sha256"])
        exposure = source_exposure(
            views[config.train_stream], config, latest["position"]["tokens"], latest["sampler"]
        )
        records[member] = {
            "run_id": config.run_id,
            "checkpoint": manager.index()["best"],
            "selected_update": selected["optimizer_update"],
            "selected_training_tokens": selected["tokens_processed"],
            "configuration": config.to_dict(),
            "selection_rule": history["selection_rule"],
            "selected_validation": selected,
            "training_exposure": exposure,
            "training_core_seconds": summary["training_core_seconds"],
            "training_session_wall_seconds": summary["training_session_wall_seconds"],
            "training_timing_definition": summary["timing_definition"],
            "training_environment": payload["identity"]["environment"],
            "prepared_sources": payload["identity"]["data"]["prepared_sources"],
            "run_files_before": inventory(directory),
            "updates": updates,
            "validation_history": derived,
        }
        payloads[member] = payload
    if len(set(initial_hashes)) != 1:
        raise ValueError("Paired initial random weights differ.")
    hashes = json.loads(
        (root / "runs" / protocol.model_a.run_id / "prerequisite_verification.json").read_text()
    )["source_hashes"]
    core = [n for n in hashes if n.startswith("model/")]
    core += [
        "tokenization/character_tokenizer.py",
        "training/engine.py",
        "data/loading.py",
    ]
    if any(not (package / name).is_file() or file_sha256(package / name) != hashes.get(name) for name in core):
        raise ValueError("Model/token/selection code changed since training; review required.")
    return (
        views,
        payloads,
        records,
        {
            "checked_at_utc": datetime.now(UTC).isoformat(),
            "status": "passed",
            "test_model_scoring_started": False,
            "partition_audit": audit,
            "selection_used_validation_only_in_recorded_runs": True,
            "initial_weights_sha256": initial_hashes[0],
            "unchanged_core_files": core,
            "evidence_limit": "Local run records cannot rule out unrecorded external activity.",
        },
    )


def run_comparison(config, root):
    root = Path(root).resolve()
    output = root / config.output_directory
    if output.exists():
        raise FileExistsError(f"Evaluation output exists; preserved: {output}")
    protocol = ComparisonProtocol.load(root / config.comparison_protocol)
    if config.document_stride > protocol.model_a.model.context_length:
        raise ValueError("document_stride exceeds the paired model context length.")
    # Optional plotting dependency is checked before spending time on model scoring.
    try:
        from transformer.evaluation.metrics import create_charts
    except ImportError as exc:
        raise ValueError(
            "Install the evaluation extra: uv sync --locked --extra evaluation."
        ) from exc

    with artifact_lock(output), staged_directory(output) as stage:
        views, payloads, records, checks = preflight(protocol, root)
        tokenizer = CharacterTokenizer.load(
            root / protocol.model_a.tokenizer_path,
            expected_fingerprint=protocol.model_a.tokenizer_fingerprint,
        )
        device = select_device(config.device).device
        identity = {
            "configuration": config.to_dict(),
            "evaluation_protocol_sha256": file_sha256(root / config.document_path),
            "comparison_protocol_sha256": file_sha256(root / config.comparison_protocol),
            "checkpoints": {m: r["checkpoint"] for m, r in records.items()},
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "dataset_fingerprint": protocol.model_a.dataset_fingerprint,
            "token_data_fingerprint": protocol.model_a.token_data_fingerprint,
            "source_hashes": {
                str(p.relative_to(Path(__file__).resolve().parents[1])): file_sha256(p)
                for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
            },
        }
        environment = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "system": platform.platform(),
            "machine": platform.machine(),
            "device": str(device),
            "dtype": "float32",
            "cpu_threads": protocol.model_a.cpu_threads,
            "deterministic_algorithms": protocol.model_a.deterministic_algorithms,
        }
        write_json(
            stage / "pre_evaluation.json",
            {**checks, "identity": identity, "environment": environment, "runs": records},
        )
        write_json(stage / "configuration.json", config.to_dict())
        (stage / "evaluation_protocol.md").write_text((root / config.document_path).read_text())
        scores, samples = {}, {}
        with execution_settings(protocol.model_a, device):
            models = {}
            for member, payload in payloads.items():
                model = TransformerLanguageModel(getattr(protocol, member).model).to(device)
                model.load_state_dict(payload["model"], strict=True)
                model.eval()
                models[member] = model
                records[member]["parameter_count"] = model.trainable_parameter_count()
                scores[member] = {}
                for mode in ("packed", "document"):
                    scores[member][mode] = score_held_out(
                        model,
                        views["test"],
                        mode=mode,
                        stride=config.document_stride,
                        batch_size=config.batch_size,
                        tokenizer=tokenizer,
                    )
                samples[member] = [
                    {
                        **greedy_continuation(model, tokenizer, p, protocol.max_new_tokens),
                        "run_id": records[member]["run_id"],
                        "checkpoint": records[member]["checkpoint"],
                        "checkpoint_update": records[member]["selected_update"],
                    }
                    for p in protocol.prompts
                ]
            benchmark = benchmark_generation(models, tokenizer, protocol.prompts, config)
        for mode in ("packed", "document"):
            for a, b in zip(
                scores["model_a"][mode]["documents"],
                scores["model_b"][mode]["documents"],
                strict=True,
            ):
                if any(
                    a[k] != b[k]
                    for k in (
                        "story_id",
                        "target_tokens",
                        "unknown_targets",
                        "eos_targets",
                        "source",
                    )
                ):
                    raise RuntimeError("Paired held-out target inventories differ.")
        for record in records.values():
            if inventory(root / "runs" / record["run_id"]) != record["run_files_before"]:
                raise RuntimeError("A training run changed during read-only evaluation.")
        result = {
            "status": "completed",
            "identity": identity,
            "fingerprint": fingerprint(identity),
            "environment": environment,
            "scores": scores,
            "runs": records,
            "generation_benchmark": benchmark,
            "original_runs_unchanged": True,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "test_results_used_for_selection": False,
            "single_training_seed": 42,
        }
        write_json(stage / "results.json", result)
        write_json(stage / "generations.json", samples)
        write_json(stage / "generation_benchmark.json", benchmark)
        create_charts(result, stage)
        lines = [
            "# Transformer: all fixed-prompt test-phase continuations",
            "",
            "Greedy, temperature 0, no top-k/top-p, 128-token cap, EOS stopping.",
            "No human ratings; text is unedited and may be repetitive or nonsensical.",
            "",
        ]
        for member, rows in samples.items():
            for number, sample in enumerate(rows, 1):
                lines += [
                    f"## {member} — prompt {number}",
                    "",
                    f"Run: `{sample['run_id']}`; "
                    f"update {sample['checkpoint_update']}; stop: {sample['stop_reason']}.",
                    f"Checkpoint SHA-256: `{sample['checkpoint']['sha256']}`.",
                    "",
                    f"Prompt: {sample['prompt']}",
                    "",
                    "```text",
                    sample["continuation"],
                    "```",
                    "",
                ]
        (stage / "samples.md").write_text("\n".join(lines))
        write_json(
            stage / "manifest.json",
            {
                "identity": identity,
                "files": {
                    str(p.relative_to(stage)): {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                    for p in sorted(stage.rglob("*"))
                    if p.is_file()
                },
            },
        )
    return {
        "status": "completed",
        "directory": str(output),
        "fingerprint": result["fingerprint"],
        "primary": {m: s["packed"]["combined"] for m, s in scores.items()},
        "original_runs_unchanged": True,
    }
