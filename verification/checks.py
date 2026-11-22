# Failure cases
"""Exercise rejection paths only against disposable copies of synthetic artifacts."""

import json
import re
import shutil
from dataclasses import replace

import torch

from transformer.configuration.settings import TrainingConfig
from transformer.data.preparation import file_sha256
from transformer.inference import GenerationConfig
from transformer.runtime.device import select_device
from transformer.training.engine import CheckpointManager, atomic_json
from transformer.training.engine import run_training


def check_failure_cases(fixture_root, output):
    output.mkdir(exist_ok=False)
    config = TrainingConfig.load(fixture_root / "runs/fixture-baseline/configuration.json")
    unused = replace(config, run_id="unused-negative-check")
    results = []

    def rejects(name, operation, exception, message):
        try:
            operation()
        except exception as exc:
            if not re.search(message, str(exc), re.IGNORECASE):
                raise AssertionError(f"{name}: unhelpful/unexpected error: {exc}") from exc
            results.append(
                {
                    "name": name,
                    "status": "pass",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "device": "cpu",
                }
            )
        else:
            raise AssertionError(f"{name}: invalid input was accepted")

    rejects(
        "missing_data",
        lambda: run_training(
            replace(unused, token_data_dir="data/tokens/missing"), fixture_root, dry_run=True
        ),
        (ValueError, FileNotFoundError),
        "missing|manifest|No such file",
    )
    rejects(
        "invalid_configuration",
        lambda: replace(unused, microbatch_size=0),
        ValueError,
        "microbatch_size",
    )
    rejects(
        "tokenizer_mismatch",
        lambda: run_training(
            replace(unused, tokenizer_fingerprint="0" * 64), fixture_root, dry_run=True
        ),
        ValueError,
        "fingerprint",
    )
    rejects("unsupported_device", lambda: select_device("cuda"), ValueError, "device")
    rejects(
        "existing_run_directory",
        lambda: run_training(config, fixture_root),
        FileExistsError,
        "already exists",
    )
    rejects(
        "invalid_generation_parameters",
        lambda: GenerationConfig(mode="sample", temperature=0),
        ValueError,
        "temperature",
    )
    run = fixture_root / "runs" / config.run_id
    corrupt = output / "corrupt"
    shutil.copytree(run, corrupt)
    manager = CheckpointManager(corrupt)
    (manager.directory / manager.index()["latest"]["file"]).write_bytes(b"not a checkpoint")
    rejects("corrupt_checkpoint", manager.load, ValueError, "checksum")

    incomplete_root = output / "incomplete_project"
    shutil.copytree(fixture_root / "data", incomplete_root / "data")
    copied_run = incomplete_root / "runs" / config.run_id
    shutil.copytree(run, copied_run)
    manager = CheckpointManager(copied_run)
    payload = manager.load()
    del payload["optimizer"]
    index = manager.index()
    path = manager.directory / index["latest"]["file"]
    torch.save(payload, path)
    for ref in (index["latest"], index["best"]):
        if ref is not None and ref["file"] == path.name:
            ref["sha256"] = file_sha256(path)
    atomic_json(manager.index_path, index)
    rejects(
        "incomplete_checkpoint",
        lambda: run_training(config, incomplete_root, resume=True),
        ValueError,
        "incomplete training checkpoint",
    )
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


# Fixture pipeline
"""A self-contained synthetic project using the real backend APIs at every stage."""

import json
from dataclasses import replace
from pathlib import Path

import torch

import transformer
from transformer.configuration.settings import ComparisonProtocol
from transformer.configuration.settings import DataPreparationConfig, DatasetSourceConfig
from transformer.configuration.settings import EvaluationConfig
from transformer.configuration.settings import TokenPreparationConfig
from transformer.configuration.settings import TrainingConfig
from transformer.configuration.settings import TransformerConfig
from transformer.data.preparation import file_sha256, iter_jsonl, write_json
from transformer.data.preparation import prepare_data
from transformer.data.loading import audit_partitions
from transformer.evaluation.comparison import run_comparison
from transformer.inference import GenerationConfig, GenerationService
from transformer.tokenization.character_tokenizer import CharacterTokenizer
from transformer.tokenization.token_preparation import prepare_tokens
from transformer.training.experiments import run_experiment
from transformer.training.engine import CheckpointManager
from transformer.training.engine import run_training


class SyntheticStoryProvider:
    """Replaces only acquisition transport; all cleaning/token/training code is production code."""

    def __init__(self, directory):
        self.directory = directory

    def verify(self, source):
        return {
            "synthetic_fixture": True,
            "source_id": source.source_id,
            "resolved_revision": source.revision,
            "network_access": False,
        }

    def records(self, source, split):
        yield from iter_jsonl(self.directory / f"{source.name}-{split}.jsonl")


def _write_inputs(root):
    inputs = root / "synthetic_inputs"
    inputs.mkdir()
    sources = []
    for i, name in enumerate(("tinystories", "simplestories")):
        subject = "cat" if i == 0 else "bird"
        train = [
            {"text": f"A {subject} played near a tree on day {number}."} for number in range(6)
        ]
        # Ek duplicate held-out mein bhi hai: cleaning ko uski training copy hataani hogi.
        records = {
            "train": train + [train[1], {"text": " \r\n\t"}, {"text": 7}],
            "validation": [train[0], {"text": f"The {subject} found a green leaf.\r\nIt was wet."}],
            "test": [
                {"text": f"The {subject} saw a moon. Ω"},
                {"text": f"The {subject} rested beside a pond."},
            ],
        }
        for split, rows in records.items():
            with (inputs / f"{name}-{split}.jsonl").open("x") as out:
                for row in rows:
                    out.write(json.dumps(row) + "\n")
        sources.append(
            DatasetSourceConfig(
                name,
                f"synthetic-fixtures/{name}",
                ("a" if i == 0 else "b") * 40,
                "default",
                "text",
                "synthetic-fixture-only",
                {k: len(v) for k, v in records.items()},
            )
        )
    (inputs / "README.md").write_text(
        "# Synthetic backend gate fixtures\n\nThese invented tiny records are test fixtures. "
        "They are not acquired TinyStories/SimpleStories examples or research results.\n"
    )
    return inputs, DataPreparationConfig("release-gate-fixture", tuple(sources))


def run_fixture_pipeline(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs, data_config = _write_inputs(root)
    provider = SyntheticStoryProvider(inputs)
    prepared = prepare_data(data_config, root, provider=provider)
    assert prepare_data(data_config, root, provider=provider, offline=True).reused
    token_config = TokenPreparationConfig(
        "release-gate-fixture", prepared.directory.name, prepared.manifest["corpus_fingerprint"]
    )
    tokens = prepare_tokens(token_config, root)
    assert prepare_tokens(token_config, root).reused
    tokenizer = CharacterTokenizer.load(tokens.directory / "tokenizer.json")
    assert "Ω" not in tokenizer.char_to_id
    architecture = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        num_layers=1,
        num_heads=2,
        embedding_dim=16,
        context_length=16,
        dropout=0.1,
    )
    config = TrainingConfig(
        run_id="fixture-baseline",
        token_data_dir=str(tokens.directory.relative_to(root)),
        tokenizer_path=str((tokens.directory / "tokenizer.json").relative_to(root)),
        tokenizer_fingerprint=tokenizer.fingerprint,
        dataset_fingerprint=tokens.manifest["dataset_fingerprint"],
        token_data_fingerprint=tokens.manifest["token_data_fingerprint"],
        model=architecture,
        device="cpu",
        seed=42,
        cpu_threads=1,
        max_training_tokens=256,
        microbatch_size=2,
        gradient_accumulation_steps=2,
        evaluation_interval=2,
        checkpoint_interval=2,
        max_validation_batches=100,
        console_interval=100,
    )
    protocol = ComparisonProtocol(
        config,
        replace(config, run_id="fixture-combined", train_stream="train"),
        "fixture_protocol.md",
        (2, 4),
        ("A cat", "A bird"),
        max_new_tokens=8,
    )
    (root / "fixture_protocol.md").write_text(
        "# Synthetic integration protocol\nCPU only, two tiny four-update fixture runs; "
        "stop/sample/resume at two; select by full validation only. Not real model availability.\n"
    )
    (root / "evaluation_protocol.md").write_text(
        "# Synthetic evaluation protocol\nSelected fixture checkpoints only; packed and "
        "isolated document metrics. Two-token timing diagnostics, no model-quality conclusions.\n"
    )
    write_json(root / "fixture_protocol.json", protocol.to_dict())
    _, audit = audit_partitions(root, config)
    summaries = {}
    package = Path(transformer.__file__).parent
    for member in ("model_a", "model_b"):
        summaries[member] = run_experiment(protocol, root, member=member)
        run = root / "runs" / getattr(protocol, member).run_id
        write_json(
            run / "prerequisite_verification.json",
            {
                "synthetic_fixture": True,
                "source_hashes": {
                    str(p.relative_to(package)): file_sha256(p) for p in package.rglob("*.py")
                },
            },
        )
        events = list(iter_jsonl(run / "metrics.jsonl"))
        assert [e["update"] for e in events if e["event"] == "resume"] == [2]
        assert [e["optimizer_update"] for e in events if e["event"] == "update"] == [1, 2, 3, 4]
    # Resumed CPU run ko independent uninterrupted execution se exact compare karo.
    reference = replace(config, run_id="fixture-uninterrupted-reference")
    run_training(reference, root)
    resumed = CheckpointManager(root / "runs" / config.run_id).load()
    uninterrupted = CheckpointManager(root / "runs" / reference.run_id).load()
    for name, value in resumed["model"].items():
        assert torch.equal(value, uninterrupted["model"][name]), name
    assert torch.equal(resumed["rng"]["torch_cpu"], uninterrupted["rng"]["torch_cpu"])
    assert resumed["scheduler"] == uninterrupted["scheduler"]
    assert resumed["position"]["tokens"] == 256
    evaluation_config = EvaluationConfig(
        "fixture_protocol.json",
        "evaluation_protocol.md",
        "artifacts/fixture-evaluation",
        device="cpu",
        document_stride=8,
        batch_size=2,
        benchmark_repetitions=1,
        benchmark_warmups=1,
        benchmark_new_tokens=2,
    )
    evaluation = run_comparison(evaluation_config, root)
    metrics_path = root / evaluation_config.output_directory / "results.json"
    metrics = json.loads(metrics_path.read_text())
    entries = []

    def ref(path):
        return {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}

    for member, kind in (("model_a", "baseline"), ("model_b", "combined")):
        record = metrics["runs"][member]
        run = root / "runs" / record["run_id"]
        entries.append(
            {
                "model_id": f"fixture-{kind}-v1",
                "kind": kind,
                "comparison_member": member,
                "run_id": record["run_id"],
                "checkpoint": ref(run / "checkpoints" / record["checkpoint"]["file"]),
                "configuration": ref(run / "configuration.json"),
                "tokenizer": ref(root / config.tokenizer_path),
                "metrics": ref(metrics_path),
                "tokenizer_fingerprint": tokenizer.fingerprint,
                "checkpoint_update": record["selected_update"],
            }
        )
    registry = root / "configs/inference/model_registry.json"
    registry.parent.mkdir(parents=True)
    write_json(registry, {"schema_version": 1, "models": entries})
    generations = []
    with GenerationService(root, device="cpu", cpu_threads=1) as service:
        for entry in entries:
            generations.append(
                service.generate(entry["model_id"], "A cat", GenerationConfig(max_new_tokens=8))
            )
    result = {
        "status": "passed",
        "synthetic_fixture_only": True,
        "device": "cpu",
        "network_transport": "local SyntheticStoryProvider; no network or downloaded data",
        "data_counts": prepared.manifest["counts"],
        "split_integrity": audit,
        "data_and_token_reruns_reused": True,
        "training_only_vocabulary_verified": True,
        "vocabulary_size": tokenizer.vocab_size,
        "architecture": architecture.to_dict(),
        "updates_each": 4,
        "training_tokens_each": 256,
        "resume_update_each": 2,
        "exact_cpu_resume_matches_uninterrupted": True,
        "evaluation": evaluation,
        "generations": generations,
    }
    write_json(root / "fixture_verification.json", result)
    return result


# Backend gate
"""A fail-closed release gate: synthetic success cannot replace real model evidence."""

import json
import math
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import transformer
from transformer.benchmarking.worker import hardware_identity
from transformer.configuration.settings import ComparisonProtocol
from transformer.data.preparation import file_sha256
from transformer.evaluation.comparison import inventory, preflight
from transformer.evaluation.metrics import score_held_out
from transformer.inference import GenerationConfig, GenerationService
from transformer.training.engine import atomic_json
from transformer.training.engine import execution_settings


def verify_test_report(path):
    """Attach the operator's full pytest JUnit evidence; never count skips as passes."""
    document = ET.parse(path).getroot()
    cases = document.findall(".//testcase")
    totals = {
        "tests": len(cases),
        "failures": len(document.findall(".//failure")),
        "errors": len(document.findall(".//error")),
        "skipped": len(document.findall(".//skipped")),
    }
    required = (
        "test_data_preparation",
        "test_character_tokenizer",
        "test_recovery",
        "test_attention",
        "test_held_out_evaluation",
        "test_generation",
        "test_backend_gate",
    )
    for name in required:
        if not any(name in c.get("classname", "") for c in cases):
            raise ValueError(f"Full-suite evidence is missing {name}.")
    if not cases or totals["failures"] or totals["errors"]:
        raise ValueError(f"Full test suite is not passing: {totals}")
    return {
        **totals,
        "passed": len(cases) - totals["skipped"],
        "path": str(path),
        "sha256": file_sha256(path),
        "device": "per-test; see skipped case details",
        "skips": [
            {"test": c.get("name"), "reason": c.find("skipped").get("message")}
            for c in cases
            if c.find("skipped") is not None
        ],
    }


def verify_real_artifacts(root, device):
    protocol = ComparisonProtocol.load(root / "configs/experiments/phase8-comparison.json")
    original_dirs = [root / "runs" / getattr(protocol, m).run_id for m in ("model_a", "model_b")]
    evaluation_dir = root / "artifacts/phase10-comparison-v1"
    original_dirs.append(evaluation_dir)
    before = {str(p.relative_to(root)): inventory(p) for p in original_dirs}
    manifest = json.loads((evaluation_dir / "manifest.json").read_text())
    for name, reference in manifest["files"].items():
        path = evaluation_dir / name
        if not path.resolve().is_relative_to(evaluation_dir.resolve()):
            raise ValueError("Evaluation manifest path escapes its artifact directory.")
        if path.stat().st_size != reference["bytes"] or file_sha256(path) != reference["sha256"]:
            raise ValueError(f"Saved evaluation artifact changed: {name}")
    views, payloads, records, checks = preflight(protocol, root)
    del payloads  # Optimizer states CPU memory mein bhi unnecessarily hold nahi karna.
    result = {
        "preflight": checks,
        "models": {},
        "saved_evaluation_files_verified": len(manifest["files"]),
    }
    with (
        GenerationService(root, device=device) as service,
        execution_settings(protocol.model_a, service.selection.device),
    ):
        result["device"] = str(service.selection.device)
        for member, model_id in (
            ("model_a", "transformer-tinystories-v1"),
            ("model_b", "transformer-combined-v1"),
        ):
            record = records[member]
            version = service.registry.resolve(model_id)
            if (
                version.run_id != record["run_id"]
                or version.checkpoint.sha256 != record["checkpoint"]["sha256"]
                or version.metrics.path != "artifacts/phase10-comparison-v1/results.json"
            ):
                raise ValueError(
                    "Real registry does not identify the actual paired selected checkpoint."
                )
            metadata = service.load(model_id)
            saved = service.registry.read_json(version.metrics)
            if saved["runs"][member]["checkpoint"] != record["checkpoint"]:
                raise ValueError("Reported metrics do not identify the selected checkpoint.")
            reproduced = {}
            for mode in ("packed", "document"):
                scores = score_held_out(
                    service._loaded.model,
                    views["test"],
                    mode=mode,
                    stride=128,
                    batch_size=4,
                    tokenizer=service._loaded.tokenizer,
                )
                prior = saved["scores"][member][mode]
                for label in ("combined", "tinystories", "simplestories"):
                    current = (
                        scores["combined"] if label == "combined" else scores["sources"][label]
                    )
                    old = prior["combined"] if label == "combined" else prior["sources"][label]
                    if current["target_tokens"] != old["target_tokens"] or not math.isclose(
                        current["cross_entropy_nats"],
                        old["cross_entropy_nats"],
                        abs_tol=1e-6,
                        rel_tol=0,
                    ):
                        raise ValueError(f"Saved {member}/{mode}/{label} metrics do not reproduce.")
                reproduced[mode] = {
                    k: scores[k] for k in ("combined", "sources", "equal_source_secondary")
                }
            run = root / "runs" / record["run_id"]
            archived = json.loads(
                (
                    run / f"diagnostics/samples-update-{record['selected_update']:06d}.json"
                ).read_text()
            )
            generations = []
            for sample in archived["samples"]:
                row = service.generate(
                    model_id,
                    sample["prompt"],
                    GenerationConfig(max_new_tokens=protocol.max_new_tokens),
                )
                if row["generated_token_ids"] != sample["generated_ids"]:
                    raise ValueError(
                        "Fixed greedy sample differs from the archived selected-checkpoint sample."
                    )
                generations.append(row)
            if service._loaded.model.training or any(
                p.grad is not None or p.requires_grad for p in service._loaded.model.parameters()
            ):
                raise ValueError("Inference model is not frozen in evaluation mode.")
            result["models"][member] = {
                "metadata": metadata,
                "real_trained_checkpoint": True,
                "run_id": record["run_id"],
                "checkpoint": record["checkpoint"],
                "training_tokens": getattr(protocol, member).total_tokens,
                "selected_update": record["selected_update"],
                "selection_rule": record["selection_rule"],
                "source_exposure": record["training_exposure"],
                "replayed_metrics": reproduced,
                "metric_absolute_tolerance": 1e-6,
                "fixed_samples": generations,
                "fixed_sample_ids_match_archive": True,
                "no_inference_gradients": True,
            }
    after = {str(p.relative_to(root)): inventory(p) for p in original_dirs}
    if before != after:
        raise ValueError("Original run/evaluation files changed during the gate.")
    result.update(
        original_files_unchanged=True,
        original_files_verified=sum(map(len, before.values())),
        original_file_hashes=before,
    )
    return result


def run_backend_gate(root, output, *, device="auto", test_report=None):
    if not __debug__:
        raise RuntimeError("Verification requires assertions; run without Python -O.")
    root, output = Path(root).resolve(), Path(output).absolute()
    if (
        not output.resolve().is_relative_to(root / "artifacts")
        or output.resolve() == root / "artifacts"
    ):
        raise ValueError("Gate output must be a new directory inside project_root/artifacts.")
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Gate output must not use symbolic links.")
    output.mkdir(parents=True, exist_ok=False)
    package = Path(transformer.__file__).parent
    report = {
        "schema_version": 1,
        "status": "incomplete",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "hardware": hardware_identity(),
        "checks": [],
        "source_hashes": inventory_python(package),
        "scope": "Local backend release gate; no frontend, no real model training or downloads.",
    }
    destination = output / "verification.json"

    def stage(name, operation):
        started = time.perf_counter()
        try:
            detail = operation()
            row = {"name": name, "status": "pass", "details": detail}
        except Exception as exc:
            row = {
                "name": name,
                "status": "fail",
                "exception": type(exc).__name__,
                "message": str(exc),
            }
        row.update(required=True, seconds=time.perf_counter() - started)
        report["checks"].append(row)
        atomic_json(destination, report)
        return row["status"] == "pass"

    atomic_json(destination, report)
    stage("full_test_suite", lambda: verify_test_report(test_report))
    if stage("synthetic_integration", lambda: run_fixture_pipeline(output / "fixture_project")):
        stage(
            "failure_cases",
            lambda: check_failure_cases(output / "fixture_project", output / "failure_cases"),
        )
    else:
        report["checks"].append(
            {
                "name": "failure_cases",
                "status": "skip",
                "required": True,
                "reason": "Synthetic integration failed; artifacts are incomplete.",
            }
        )
    stage("real_artifacts", lambda: verify_real_artifacts(root, device))
    report["status"] = (
        "passed" if all(c["status"] == "pass" for c in report["checks"]) else "incomplete"
    )
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    atomic_json(destination, report)
    return report


def inventory_python(package):
    return {str(p.relative_to(package)): file_sha256(p) for p in sorted(package.rglob("*.py"))}
